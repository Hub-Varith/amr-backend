"""XGBoost accelerated-failure-time model on MIC intervals (the main model).

``survival:aft`` consumes the label interval ``(mic_lower, mic_upper]`` directly:
``label_lower_bound = lo`` and ``label_upper_bound = hi`` in mg/L, with ``lo == 0``
(left-censored) and ``hi == inf`` (right-censored) allowed. XGBoost takes logs
internally, so nothing is guessed at the panel edges (CLAUDE.md, "MIC intervals").

Training protocol (deterministic given ``seed`` and ``nthread``):

1. A seeded 20% holdout is carved out of the rows passed to :meth:`XgbAft.fit`
   (optionally by whole ``groups``). The caller never passes validation data; the
   holdout lives *inside* the training fold, so validation folds and the test set
   stay untouched (CLAUDE.md rule 8).
2. For each ``aft_loss_distribution_scale`` in ``{0.5, 1.0, 1.5}`` a booster is
   trained on the other 80% with early stopping on the holdout's ``aft-nloglik``.
3. The scale with the lowest holdout ``aft-nloglik`` wins, and ``n_rounds`` is its
   best iteration + 1.
4. The final booster is refitted on all rows with the chosen scale and ``n_rounds``.

``predict_log2`` returns ``log2(booster.predict(X))``; rounding up to the grid and
conformal bands are applied by the caller.

Input matrices are converted to CSR before reaching XGBoost (see
:func:`genome2mic.models.base.to_csr`) so dense and sparse callers get the same
model and the same predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from genome2mic.models.base import (
    FeatureMatrix,
    ModelBundleMixin,
    check_feature_names,
    holdout_split,
    read_json,
    take_rows,
    to_csr,
    validate_intervals,
    write_json,
)

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.0,
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bynode": 0.5,
}
"""Main-model parameters from CLAUDE.md. ``aft_loss_distribution_scale`` is tuned."""

SCALE_GRID: tuple[float, ...] = (0.5, 1.0, 1.5)
"""Candidate ``aft_loss_distribution_scale`` values tried on the in-fold holdout."""

HOLDOUT_NAME = "holdout"
# Smallest positive float32 that survives log2 without becoming -inf.
_TINY = np.finfo(np.float32).tiny


class XgbAft(ModelBundleMixin):
    """``survival:aft`` booster with in-fold early stopping and scale tuning.

    Args:
        params: Overrides merged on top of :data:`DEFAULT_PARAMS`. ``seed`` and
            ``nthread`` are always taken from the keyword arguments below.
        name: Model id for the predictions table (``aft_known`` or
            ``aft_known_unitig`` -- same class, different feature set).
        scales: Candidate ``aft_loss_distribution_scale`` values.
        max_rounds: Upper bound on boosting rounds during early stopping.
        early_stopping_rounds: Patience on the holdout ``aft-nloglik``.
        holdout_fraction: Share of the fit rows held out for early stopping.
        fallback_rounds: Rounds used when the data is too small for a holdout.
        min_rounds: Floor on ``n_rounds`` after early stopping.
        seed: Random seed for the holdout split and xgboost's own sampling.
        nthread: Threads xgboost may use (1 = fully deterministic).
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        name: str = "aft_known",
        scales: tuple[float, ...] = SCALE_GRID,
        max_rounds: int = 1000,
        early_stopping_rounds: int = 30,
        holdout_fraction: float = 0.2,
        fallback_rounds: int = 200,
        min_rounds: int = 10,
        seed: int = 0,
        nthread: int = 1,
    ) -> None:
        if not scales:
            raise ValueError("scales must contain at least one value")
        if any(s <= 0 for s in scales):
            raise ValueError("every aft_loss_distribution_scale must be > 0")
        self.name = name
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.params["seed"] = int(seed)
        self.params["nthread"] = int(nthread)
        self.scales = tuple(float(s) for s in scales)
        self.max_rounds = int(max_rounds)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.holdout_fraction = float(holdout_fraction)
        self.fallback_rounds = int(fallback_rounds)
        self.min_rounds = int(min_rounds)
        self.seed = int(seed)
        self.nthread = int(nthread)

        # fitted state
        self.booster_: xgb.Booster | None = None
        self.feature_names_: list[str] | None = None
        self.n_rounds_: int | None = None
        self.scale_: float | None = None
        self.tuning_: list[dict[str, float | int]] = []
        self.n_train_: int | None = None
        self.n_holdout_: int | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: FeatureMatrix,
        lo: np.ndarray,
        hi: np.ndarray,
        feature_names: list[str],
        groups: np.ndarray | None = None,
    ) -> "XgbAft":
        """Fit on intervals ``(lo, hi]`` in mg/L.

        Args:
            X: Feature matrix (CSR or dense), rows aligned with ``lo``/``hi``.
            lo: Exclusive lower bounds; ``0`` = left-censored.
            hi: Inclusive upper bounds; ``inf`` = right-censored.
            feature_names: One name per column; validated against the leakage rules.
            groups: Optional per-row group labels (e.g. lineage cluster) used only
                to keep whole groups on one side of the early-stopping holdout.
                Never used as a feature.
        """
        matrix = to_csr(X)
        low, high = validate_intervals(lo, hi, matrix.shape[0])
        names = check_feature_names(feature_names, matrix.shape[1])
        n = matrix.shape[0]
        if n < 2:
            raise ValueError("need at least 2 rows to fit XgbAft")

        fit_idx, hold_idx = holdout_split(n, self.holdout_fraction, self.seed, groups)
        self.tuning_ = []
        if hold_idx.size == 0:
            logger.warning(
                "[%s] only %d rows: skipping early stopping, using %d rounds at scale %.2f",
                self.name, n, self.fallback_rounds, self.params["aft_loss_distribution_scale"],
            )
            scale = float(self.params["aft_loss_distribution_scale"])
            n_rounds = self.fallback_rounds
        else:
            d_fit = self._dmatrix(take_rows(matrix, fit_idx), names, low[fit_idx], high[fit_idx])
            d_hold = self._dmatrix(take_rows(matrix, hold_idx), names, low[hold_idx], high[hold_idx])
            for candidate in self.scales:
                params = {**self.params, "aft_loss_distribution_scale": candidate}
                booster = xgb.train(
                    params,
                    d_fit,
                    num_boost_round=self.max_rounds,
                    evals=[(d_hold, HOLDOUT_NAME)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False,
                )
                record = {
                    "scale": float(candidate),
                    "best_iteration": int(booster.best_iteration),
                    "best_score": float(booster.best_score),
                }
                self.tuning_.append(record)
                logger.info(
                    "[%s] scale=%.2f holdout aft-nloglik=%.4f at iteration %d",
                    self.name, candidate, record["best_score"], record["best_iteration"],
                )
            best = min(self.tuning_, key=lambda r: (r["best_score"], abs(r["scale"] - 1.0)))
            scale = float(best["scale"])
            n_rounds = max(self.min_rounds, int(best["best_iteration"]) + 1)

        final_params = {**self.params, "aft_loss_distribution_scale": scale}
        d_all = self._dmatrix(matrix, names, low, high)
        self.booster_ = xgb.train(final_params, d_all, num_boost_round=n_rounds)
        self.params = final_params
        self.feature_names_ = names
        self.n_rounds_ = int(n_rounds)
        self.scale_ = scale
        self.n_train_ = int(n)
        self.n_holdout_ = int(hold_idx.size)
        logger.info(
            "[%s] fitted on %d rows x %d features: scale=%.2f, n_rounds=%d (holdout %d rows)",
            self.name, n, len(names), scale, n_rounds, hold_idx.size,
        )
        return self

    # -------------------------------------------------------------- predict
    def predict_log2(self, X: FeatureMatrix) -> np.ndarray:
        """``log2`` of the booster's MIC prediction (mg/L), one float per row."""
        matrix = to_csr(self._check_predict_input(X))
        assert self.booster_ is not None and self.feature_names_ is not None
        if matrix.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        dmat = self._dmatrix(matrix, self.feature_names_)
        pred = self.booster_.predict(dmat).astype(np.float64)
        return np.log2(np.maximum(pred, _TINY))

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Per-feature importance mapped to ``feature_names`` (unused features = 0).

        Returns a float ``Series`` indexed by feature name, sorted descending.
        ``importance_type`` is any value xgboost accepts (``gain``, ``weight``,
        ``cover``, ``total_gain``, ``total_cover``).
        """
        self._require_fitted()
        assert self.booster_ is not None and self.feature_names_ is not None
        scores = self.booster_.get_score(importance_type=importance_type)
        values = [float(scores.get(name, 0.0)) for name in self.feature_names_]
        series = pd.Series(values, index=pd.Index(self.feature_names_, name="feature"), name=importance_type)
        return series.sort_values(ascending=False, kind="stable")

    # ------------------------------------------------------------ persistence
    def save(self, dir: Path) -> None:
        """Write ``model.ubj`` (booster) and ``params.json`` (everything else) into ``dir``."""
        self._require_fitted()
        assert self.booster_ is not None
        target = Path(dir)
        target.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(target / self.MODEL_FILE))
        write_json(
            target / self.PARAMS_FILE,
            {
                "model_class": type(self).__name__,
                "name": self.name,
                "params": self.params,
                "n_rounds": self.n_rounds_,
                "scale": self.scale_,
                "scales": list(self.scales),
                "max_rounds": self.max_rounds,
                "early_stopping_rounds": self.early_stopping_rounds,
                "holdout_fraction": self.holdout_fraction,
                "fallback_rounds": self.fallback_rounds,
                "min_rounds": self.min_rounds,
                "seed": self.seed,
                "nthread": self.nthread,
                "feature_names": self.feature_names_,
                "n_train": self.n_train_,
                "n_holdout": self.n_holdout_,
                "tuning": self.tuning_,
                "xgboost_version": xgb.__version__,
            },
        )
        logger.info("[%s] saved model to %s", self.name, target)

    @classmethod
    def load(cls, dir: Path) -> "XgbAft":
        """Read a bundle written by :meth:`save`."""
        source = Path(dir)
        meta = read_json(source / cls.PARAMS_FILE)
        if meta.get("model_class") not in (None, cls.__name__):
            raise ValueError(f"{source} holds a {meta.get('model_class')}, not {cls.__name__}")
        params = dict(meta["params"])
        model = cls(
            params,
            name=meta.get("name", "aft_known"),
            scales=tuple(meta.get("scales", SCALE_GRID)),
            max_rounds=meta.get("max_rounds", 1000),
            early_stopping_rounds=meta.get("early_stopping_rounds", 30),
            holdout_fraction=meta.get("holdout_fraction", 0.2),
            fallback_rounds=meta.get("fallback_rounds", 200),
            min_rounds=meta.get("min_rounds", 10),
            seed=meta.get("seed", params.get("seed", 0)),
            nthread=meta.get("nthread", params.get("nthread", 1)),
        )
        booster = xgb.Booster()
        booster.load_model(str(source / cls.MODEL_FILE))
        model.booster_ = booster
        model.feature_names_ = [str(n) for n in meta["feature_names"]]
        model.n_rounds_ = meta.get("n_rounds")
        model.scale_ = meta.get("scale")
        model.tuning_ = list(meta.get("tuning", []))
        model.n_train_ = meta.get("n_train")
        model.n_holdout_ = meta.get("n_holdout")
        return model

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _dmatrix(
        X: FeatureMatrix,
        feature_names: list[str],
        lo: np.ndarray | None = None,
        hi: np.ndarray | None = None,
    ) -> xgb.DMatrix:
        dmat = xgb.DMatrix(X, feature_names=feature_names)
        if lo is not None and hi is not None:
            dmat.set_float_info("label_lower_bound", np.asarray(lo, dtype=np.float32))
            dmat.set_float_info("label_upper_bound", np.asarray(hi, dtype=np.float32))
        return dmat
