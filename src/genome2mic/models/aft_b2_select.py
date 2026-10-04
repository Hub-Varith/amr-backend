"""Per-pair choice of the AFT model, B2 or their average (model id ``aft_b2_select``, v0.6).

Training (:func:`genome2mic.models.train.train_pair`) scores three candidate point
predictions per species x drug on out-of-fold CV predictions:

* ``aft`` -- the main AFT model (``aft_known``, or ``aft_known_unitig`` with unitigs);
* ``b2``  -- :class:`~genome2mic.models.b2_xgb_steps.B2XgbSteps` (argmax log2 step);
* ``avg`` -- the mean of the two log2 predictions (each first clipped to the panel caps).

The choice for CV fold ``f`` is made from the other folds only (nested), and the bundle's
choice from every CV fold. This class is what the bundle ships: it holds the fitted
component(s) the chosen candidate needs, plus the panel caps, and its
:meth:`AftB2Select.predict_log2` reproduces the training-time candidate exactly, so the
prediction pipeline applies it identically (raw value -> caps -> round up -> band).

Bundle layout (``models/<SPECIES>/<drug>/``)::

    params.json   {model_class: "AftB2Select", name, choice, base_name, caps, feature_names, components}
    aft/          XgbAft bundle (present when choice is "aft" or "avg")
    b2/           B2XgbSteps bundle (present when choice is "b2" or "avg")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genome2mic.models.b2_xgb_steps import B2XgbSteps
from genome2mic.models.base import ModelBundleMixin, read_json, to_csr, write_json
from genome2mic.models.xgb_aft import XgbAft

logger = logging.getLogger(__name__)

MODEL_ID = "aft_b2_select"
CHOICE_AFT = "aft"
CHOICE_B2 = "b2"
CHOICE_AVG = "avg"
CHOICES: tuple[str, ...] = (CHOICE_AFT, CHOICE_AVG, CHOICE_B2)
"""Candidates in tie-break preference order (the AFT model is the incumbent)."""
AFT_DIR = "aft"
B2_DIR = "b2"


def combine_log2(aft: Any, b2: Any, choice: str) -> np.ndarray:
    """Candidate point prediction (log2 mg/L) from the AFT and B2 predictions of the same rows.

    Both inputs must already be clipped to the panel caps. ``aft`` / ``b2`` return their
    own array (NaN stays NaN); ``avg`` is the mean where both exist and the available one
    where only one does (a fold whose B2 fit failed keeps the AFT value).
    """
    if choice not in CHOICES:
        raise ValueError(f"unknown candidate {choice!r}; expected one of {CHOICES}")
    a = np.asarray(aft, dtype=np.float64)
    b = np.asarray(b2, dtype=np.float64)
    if choice == CHOICE_AFT:
        return a.copy()
    if choice == CHOICE_B2:
        return b.copy()
    if a.shape != b.shape:
        raise ValueError(f"aft and b2 predictions differ in shape: {a.shape} vs {b.shape}")
    out = (a + b) / 2.0
    only_a = np.isnan(b) & ~np.isnan(a)
    only_b = np.isnan(a) & ~np.isnan(b)
    out[only_a] = a[only_a]
    out[only_b] = b[only_b]
    return out


def _clip(pred: np.ndarray, caps: tuple[float, float] | None) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    return pred if caps is None else np.clip(pred, caps[0], caps[1])


class AftB2Select(ModelBundleMixin):
    """The bundle model of ``aft_b2_select``: the fitted component(s) of the chosen candidate.

    Not fitted directly: training fits the AFT and B2 models on the train rows and
    assembles this object (:meth:`from_parts`). ``feature_names_`` are the AFT model's
    (``features.json``); B2 reads its own columns (a subset: B2 never uses unitigs) by name.
    """

    def __init__(
        self,
        *,
        name: str = MODEL_ID,
        choice: str = CHOICE_AFT,
        aft: XgbAft | None = None,
        b2: B2XgbSteps | None = None,
        caps: tuple[float, float] | None = None,
        base_name: str = "aft_known",
        feature_names: list[str] | None = None,
    ) -> None:
        if choice not in CHOICES:
            raise ValueError(f"unknown candidate {choice!r}; expected one of {CHOICES}")
        self.name = name
        self.choice = choice
        self.aft = aft
        self.b2 = b2
        self.caps = None if caps is None else (float(caps[0]), float(caps[1]))
        self.base_name = base_name
        self.feature_names_: list[str] | None = None if feature_names is None else [str(n) for n in feature_names]
        self._aft_index: np.ndarray | None = None
        self._b2_index: np.ndarray | None = None
        if self.feature_names_ is not None:
            self._index_components()

    # ------------------------------------------------------------ assembly
    @classmethod
    def from_parts(
        cls,
        choice: str,
        aft: XgbAft | None,
        b2: B2XgbSteps | None,
        caps: tuple[float, float] | None,
        feature_names: list[str],
        *,
        base_name: str = "aft_known",
    ) -> "AftB2Select":
        """Keep only the component(s) ``choice`` needs."""
        need_aft = choice in (CHOICE_AFT, CHOICE_AVG)
        need_b2 = choice in (CHOICE_B2, CHOICE_AVG)
        return cls(choice=choice, aft=aft if need_aft else None, b2=b2 if need_b2 else None, caps=caps,
                   base_name=base_name, feature_names=feature_names)

    def _index_components(self) -> None:
        names = self.feature_names_ or []
        position = {n: i for i, n in enumerate(names)}
        if self.choice in (CHOICE_AFT, CHOICE_AVG) and self.aft is None:
            raise ValueError(f"choice {self.choice!r} needs the AFT component")
        if self.choice in (CHOICE_B2, CHOICE_AVG) and self.b2 is None:
            raise ValueError(f"choice {self.choice!r} needs the B2 component")
        for label, part in (("aft", self.aft), ("b2", self.b2)):
            if part is None:
                continue
            part_names = list(part.feature_names_ or [])
            missing = [n for n in part_names if n not in position]
            if missing:
                raise ValueError(f"{label} component reads {len(missing)} feature(s) the bundle does not list, e.g. {missing[:3]}")
            index = np.array([position[n] for n in part_names], dtype=np.int64)
            if label == "aft":
                self._aft_index = index
            else:
                self._b2_index = index

    def fit(self, *args: Any, **kwargs: Any) -> "AftB2Select":
        raise TypeError("AftB2Select is assembled by training from fitted AFT/B2 models (from_parts); it has no fit()")

    # -------------------------------------------------------------- predict
    def component_log2(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(aft, b2)`` log2 predictions, each clipped to the panel caps (NaN for a missing component)."""
        matrix = to_csr(self._check_predict_input(X))
        n = matrix.shape[0]
        aft = np.full(n, np.nan)
        b2 = np.full(n, np.nan)
        if self.aft is not None and n:
            aft = _clip(self.aft.predict_log2(matrix[:, self._aft_index]), self.caps)
        if self.b2 is not None and n:
            b2 = _clip(self.b2.predict_log2(matrix[:, self._b2_index]), self.caps)
        return aft, b2

    def predict_log2(self, X: Any) -> np.ndarray:
        """The chosen candidate's log2 MIC (components clipped to the panel caps first, as in training)."""
        aft, b2 = self.component_log2(X)
        return combine_log2(aft, b2, self.choice)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """The AFT component's importances (B2's when the choice is ``b2``); unused features = 0."""
        names = list(self.feature_names_ or [])
        out = pd.Series(0.0, index=names, dtype=float)
        if self.aft is not None and hasattr(self.aft, "feature_importance"):
            series = self.aft.feature_importance(importance_type)
            out.loc[[k for k in series.index if k in out.index]] = series[[k for k in series.index if k in out.index]]
        elif self.b2 is not None and self.b2.booster_ is not None:
            scores = self.b2.booster_.get_score(importance_type=importance_type)
            for k, v in scores.items():
                if k in out.index:
                    out.loc[k] = float(v)
        return out.sort_values(ascending=False)

    # ------------------------------------------------------------ persistence
    def save(self, dir: Path) -> None:
        """Write ``params.json`` and the component bundle(s) into ``aft/`` / ``b2/``."""
        self._require_fitted()
        target = Path(dir)
        target.mkdir(parents=True, exist_ok=True)
        components: dict[str, str] = {}
        if self.aft is not None:
            self.aft.save(target / AFT_DIR)
            components["aft"] = AFT_DIR
        if self.b2 is not None:
            self.b2.save(target / B2_DIR)
            components["b2"] = B2_DIR
        write_json(
            target / self.PARAMS_FILE,
            {
                "model_class": type(self).__name__,
                "name": self.name,
                "choice": self.choice,
                "base_name": self.base_name,
                "caps": None if self.caps is None else [float(self.caps[0]), float(self.caps[1])],
                "feature_names": self.feature_names_,
                "components": components,
            },
        )
        logger.info("[%s] saved choice %r (%s) to %s", self.name, self.choice, ", ".join(components), target)

    @classmethod
    def load(cls, dir: Path) -> "AftB2Select":
        source = Path(dir)
        meta = read_json(source / cls.PARAMS_FILE)
        if meta.get("model_class") != cls.__name__:
            raise ValueError(f"{source} holds a {meta.get('model_class')}, not {cls.__name__}")
        components = meta.get("components") or {}
        aft = XgbAft.load(source / components["aft"]) if "aft" in components else None
        b2 = B2XgbSteps.load(source / components["b2"]) if "b2" in components else None
        caps = meta.get("caps")
        return cls(
            name=meta.get("name", MODEL_ID),
            choice=meta["choice"],
            aft=aft,
            b2=b2,
            caps=None if caps is None else (float(caps[0]), float(caps[1])),
            base_name=meta.get("base_name", "aft_known"),
            feature_names=[str(n) for n in meta["feature_names"]],
        )


__all__ = ["AFT_DIR", "AftB2Select", "B2_DIR", "CHOICES", "CHOICE_AFT", "CHOICE_AVG", "CHOICE_B2", "MODEL_ID", "combine_log2"]
