"""Baseline B1: known-AMR profile lookup.

Key = the tuple of known-AMR feature values of a genome (its resistance "profile");
value = the median ``label_point_log2`` of the training genomes sharing that exact
profile. Unseen profiles fall back to the global median. Censored rows take part
through :func:`genome2mic.mic.label_point_log2` (left/interval -> ``log2(hi)``,
right -> ``log2(lo) + 1``), so no label is imputed or dropped.

B1 is meant for the small, dense known-AMR matrix. It refuses matrices wider than
:attr:`B1Lookup.MAX_COLUMNS` so nobody accidentally densifies a unitig matrix.
Saved as a single JSON file (``lookup.json``).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from genome2mic.models.base import (
    FeatureMatrix,
    ModelBundleMixin,
    as_matrix,
    check_feature_names,
    label_point_log2_array,
    read_json,
    write_json,
)

logger = logging.getLogger(__name__)

ProfileKey = tuple[int | float, ...]


def _normalize_value(value: Any) -> int | float:
    """Integral values become ``int`` (so int8 ``1`` and float ``1.0`` share a key)."""
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _profile_keys(dense: np.ndarray) -> list[ProfileKey]:
    return [tuple(_normalize_value(v) for v in row) for row in dense]


class B1Lookup(ModelBundleMixin):
    """Median ``label_point_log2`` per known-AMR profile, with a global fallback.

    Args:
        name: Model id for the predictions table.
    """

    MAX_COLUMNS = 10_000
    LOOKUP_FILE = "lookup.json"

    def __init__(self, *, name: str = "b1_lookup") -> None:
        self.name = name
        self.feature_names_: list[str] | None = None
        self.table_: dict[ProfileKey, float] = {}
        self.counts_: dict[ProfileKey, int] = {}
        self.global_median_: float | None = None
        self.n_train_: int | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: FeatureMatrix,
        lo: np.ndarray,
        hi: np.ndarray,
        feature_names: list[str],
    ) -> "B1Lookup":
        """Build the profile -> median table from all rows (censored rows included)."""
        matrix = as_matrix(X)
        names = check_feature_names(feature_names, matrix.shape[1])
        dense = self._densify(matrix)
        y = label_point_log2_array(lo, hi)
        if len(y) != dense.shape[0]:
            raise ValueError(f"X has {dense.shape[0]} rows but labels have {len(y)}")
        if len(y) == 0:
            raise ValueError("B1Lookup needs at least one row")

        buckets: dict[ProfileKey, list[float]] = defaultdict(list)
        for key, value in zip(_profile_keys(dense), y.tolist(), strict=True):
            buckets[key].append(value)
        self.table_ = {key: float(np.median(values)) for key, values in buckets.items()}
        self.counts_ = {key: len(values) for key, values in buckets.items()}
        self.global_median_ = float(np.median(y))
        self.feature_names_ = names
        self.n_train_ = int(len(y))
        logger.info(
            "[%s] fitted on %d rows: %d distinct profiles over %d features, global median log2=%.2f",
            self.name, len(y), len(self.table_), len(names), self.global_median_,
        )
        return self

    # -------------------------------------------------------------- predict
    def predict_log2(self, X: FeatureMatrix) -> np.ndarray:
        """Median log2 MIC of the matching profile, else the global median."""
        matrix = self._check_predict_input(X)
        assert self.global_median_ is not None
        dense = self._densify(matrix)
        keys = _profile_keys(dense)
        out = np.fromiter(
            (self.table_.get(key, self.global_median_) for key in keys),
            dtype=np.float64,
            count=len(keys),
        )
        n_fallback = sum(1 for key in keys if key not in self.table_)
        level = logging.INFO if n_fallback else logging.DEBUG
        logger.log(level, "[%s] predicted %d rows; %d unseen profile(s) used the global median",
                   self.name, len(keys), n_fallback)
        return out

    # ------------------------------------------------------------ persistence
    def save(self, dir: Path) -> None:
        """Write ``lookup.json`` into ``dir``."""
        self._require_fitted()
        target = Path(dir)
        target.mkdir(parents=True, exist_ok=True)
        entries = [
            {"key": list(key), "median_log2": value, "n": self.counts_.get(key, 0)}
            for key, value in sorted(self.table_.items(), key=lambda kv: kv[0])
        ]
        write_json(
            target / self.LOOKUP_FILE,
            {
                "model_class": type(self).__name__,
                "name": self.name,
                "feature_names": self.feature_names_,
                "global_median_log2": self.global_median_,
                "n_train": self.n_train_,
                "n_profiles": len(entries),
                "table": entries,
            },
        )
        logger.info("[%s] saved %d profiles to %s", self.name, len(entries), target)

    @classmethod
    def load(cls, dir: Path) -> "B1Lookup":
        """Read ``lookup.json`` written by :meth:`save`."""
        source = Path(dir)
        meta = read_json(source / cls.LOOKUP_FILE)
        if meta.get("model_class") not in (None, cls.__name__):
            raise ValueError(f"{source} holds a {meta.get('model_class')}, not {cls.__name__}")
        model = cls(name=meta.get("name", "b1_lookup"))
        model.feature_names_ = [str(n) for n in meta["feature_names"]]
        model.global_median_ = float(meta["global_median_log2"])
        model.n_train_ = meta.get("n_train")
        model.table_ = {}
        model.counts_ = {}
        for entry in meta["table"]:
            key = tuple(_normalize_value(v) for v in entry["key"])
            model.table_[key] = float(entry["median_log2"])
            model.counts_[key] = int(entry.get("n", 0))
        return model

    # --------------------------------------------------------------- helpers
    @classmethod
    def _densify(cls, matrix: FeatureMatrix) -> np.ndarray:
        if matrix.shape[1] > cls.MAX_COLUMNS:
            raise ValueError(
                f"B1Lookup is for the known-AMR matrix; refusing to densify {matrix.shape[1]} columns "
                f"(limit {cls.MAX_COLUMNS})"
            )
        if sp.issparse(matrix):
            return np.asarray(matrix.toarray())
        return np.asarray(matrix)
