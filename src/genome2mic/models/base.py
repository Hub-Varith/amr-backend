"""Shared protocol and helpers for the per-drug MIC models.

Every model in :mod:`genome2mic.models` follows the same minimal protocol
(:class:`MicModel`) so the training orchestrator and the prediction pipeline can
drive them interchangeably::

    model = XgbAft().fit(X, lo, hi, feature_names)
    pred_log2 = model.predict_log2(X_test)        # raw log2 MIC, not rounded
    model.save(bundle_dir)
    model = XgbAft.load(bundle_dir)

``X`` is a feature matrix (``scipy.sparse.csr_matrix`` or a dense ``ndarray``) whose
columns are the known-AMR (``gene_*``, ``point_*``, ``n_class_*``) and unitig (``u_*``)
features. ``lo``/``hi`` are the label interval bounds from ``labels.parquet`` in
mg/L: ``lo == 0`` is left-censored and ``hi == inf`` is right-censored.

This module also holds the input validation shared by every model, including the
guard that keeps evaluation-only columns (``lineage_cluster``, ``st``, ``country``,
``year``, ``source``, ``isolation_source``, ``biosample``, ``split``, ``fold``) out of
any feature matrix (CLAUDE.md non-negotiable rule 2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp

from genome2mic.mic import exact_interval_mask

logger = logging.getLogger(__name__)

FeatureMatrix = sp.csr_matrix | np.ndarray

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "lineage_cluster",
        "st",
        "country",
        "year",
        "source",
        "isolation_source",
        "biosample",
        "split",
        "fold",
        "external_set",
        "lolo_lineage",
        "genome_id",
        "species",
        "drug",
        "mic_lower",
        "mic_upper",
        "censor",
        "sir",
    }
)
"""Column names that must never appear in a feature matrix.

Metadata used for splitting/evaluation (contract rule 2) plus the label and key
columns, which would be trivial leakage if they slipped into ``X``.
"""

ALLOWED_FEATURE_PREFIXES: tuple[str, ...] = ("gene_", "point_", "n_class_", "u_")
"""Feature column prefixes from CLAUDE.md conventions."""

# xgboost rejects these characters in feature names.
_BAD_NAME_CHARS = ("[", "]", "<")


@runtime_checkable
class MicModel(Protocol):
    """Protocol every per-drug MIC model implements.

    Attributes:
        name: Model id used in the ``model`` column of the predictions table
            (``b1_lookup``, ``b2_xgb_steps``, ``aft_known``, ``aft_known_unitig``).
    """

    name: str

    def fit(
        self,
        X: FeatureMatrix,
        lo: np.ndarray,
        hi: np.ndarray,
        feature_names: list[str],
    ) -> "MicModel":
        """Fit on interval labels ``(lo, hi]`` in mg/L and return ``self``."""
        ...

    def predict_log2(self, X: FeatureMatrix) -> np.ndarray:
        """Raw log2 MIC prediction per row (float64, not rounded to the grid)."""
        ...

    def save(self, dir: Path) -> None:
        """Write the fitted model into ``dir`` (created if missing)."""
        ...

    @classmethod
    def load(cls, dir: Path) -> "MicModel":
        """Read a model written by :meth:`save`."""
        ...


# ------------------------------------------------------------------ validation
def check_feature_names(feature_names: list[str], n_columns: int) -> list[str]:
    """Validate a feature-name list against the matrix width and the leakage rules.

    Raises ``ValueError`` when the length does not match ``n_columns``, a name is
    duplicated, a name is in :data:`FORBIDDEN_FEATURES`, or a name contains a
    character xgboost rejects (``[``, ``]``, ``<``). Returns the names as a list of
    ``str``. Prefixes outside :data:`ALLOWED_FEATURE_PREFIXES` are allowed but
    logged at WARNING so an unexpected column is visible.
    """
    names = [str(n) for n in feature_names]
    if len(names) != n_columns:
        raise ValueError(f"feature_names has {len(names)} entries but X has {n_columns} columns")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate feature names: {dupes[:10]}")
    forbidden = sorted(n for n in names if n.lower() in FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(
            f"forbidden feature columns in X (evaluation/label metadata, contract rule 2): {forbidden}"
        )
    bad = [n for n in names if any(c in n for c in _BAD_NAME_CHARS)]
    if bad:
        raise ValueError(f"feature names may not contain [, ] or <: {bad[:10]}")
    unexpected = [n for n in names if not n.startswith(ALLOWED_FEATURE_PREFIXES)]
    if unexpected:
        logger.warning(
            "%d feature name(s) do not use a known prefix %s, e.g. %s",
            len(unexpected),
            ALLOWED_FEATURE_PREFIXES,
            unexpected[:5],
        )
    return names


def as_matrix(X: Any) -> FeatureMatrix:
    """Coerce ``X`` to a 2-D CSR matrix or a 2-D dense ``ndarray`` without densifying.

    Sparse input of any format becomes CSR (int8 or float32 preserved as given);
    dense input becomes a 2-D ``ndarray``. Pandas frames are accepted via
    ``to_numpy()``. Raises ``ValueError`` for anything that is not 2-D.
    """
    if sp.issparse(X):
        matrix = X.tocsr()
        if matrix.ndim != 2:
            raise ValueError("X must be 2-D")
        return matrix
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {arr.shape}")
    return arr


def validate_intervals(lo: Any, hi: Any, n_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate label intervals and return them as float64 arrays.

    Rules (contract, stage 2): no nulls, ``lo >= 0``, ``lo < hi``, ``lo`` finite,
    ``hi`` may be ``inf``; ``(0, inf)`` is rejected because it carries no
    information. If ``n_rows`` is given the lengths must match it.
    """
    low = np.asarray(lo, dtype=np.float64).ravel()
    high = np.asarray(hi, dtype=np.float64).ravel()
    if low.shape != high.shape:
        raise ValueError(f"lo and hi must have the same length, got {low.shape} and {high.shape}")
    if n_rows is not None and len(low) != n_rows:
        raise ValueError(f"X has {n_rows} rows but labels have {len(low)}")
    if np.isnan(low).any() or np.isnan(high).any():
        raise ValueError("label bounds contain nulls; a missing lab result must be a missing row")
    if (low < 0).any():
        raise ValueError("mic_lower must be >= 0")
    if np.isinf(low).any():
        raise ValueError("mic_lower must be finite")
    if not (low < high).all():
        raise ValueError("every row needs mic_lower < mic_upper")
    if ((low == 0) & np.isinf(high)).any():
        raise ValueError("interval (0, inf) carries no information and is not allowed")
    return low, high


def exact_mask(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Boolean mask of exact-MIC rows: one doubling step, see :func:`genome2mic.mic.exact_interval_mask`.

    Multi-step ``interval`` rows (an ``I``-only label such as ``(2, 8]``) are not exact.
    """
    return exact_interval_mask(lo, hi)


def label_point_log2_array(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Vectorized :func:`genome2mic.mic.label_point_log2`.

    interval / left-censored -> ``log2(hi)``; right-censored -> ``log2(lo) + 1``.
    Inputs are validated with :func:`validate_intervals`.
    """
    low, high = validate_intervals(lo, hi)
    right = np.isinf(high)
    out = np.empty(len(low), dtype=np.float64)
    out[~right] = np.log2(high[~right])
    out[right] = np.log2(low[right]) + 1.0
    return out


# -------------------------------------------------------------------- splitting
def holdout_split(
    n_rows: int,
    fraction: float,
    seed: int,
    groups: np.ndarray | None = None,
    min_rows: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded random split of ``range(n_rows)`` into (fit_idx, holdout_idx).

    When ``groups`` is given (one label per row, e.g. a lineage cluster) whole
    groups are moved into the holdout until it holds at least ``fraction`` of
    the rows, so near-identical genomes never straddle the two sides. Groups are
    a *splitting* aid here and never become a feature. Without ``groups`` every
    row is its own group.

    Returns two sorted index arrays. If either side would have fewer than
    ``min_rows`` rows, both are returned empty-holdout (``holdout_idx`` has size 0)
    so the caller can fall back to a fixed number of rounds.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_rows)
    if groups is None:
        order = rng.permutation(n_rows)
        n_hold = int(round(fraction * n_rows))
        hold = np.sort(order[:n_hold])
    else:
        g = np.asarray(groups)
        if len(g) != n_rows:
            raise ValueError(f"groups has {len(g)} entries but X has {n_rows} rows")
        uniq = np.unique(g)
        shuffled = uniq[rng.permutation(len(uniq))]
        target = fraction * n_rows
        chosen: list[Any] = []
        count = 0
        for label in shuffled:
            if count >= target:
                break
            chosen.append(label)
            count += int((g == label).sum())
        hold = np.sort(all_idx[np.isin(g, chosen)])
    fit = np.setdiff1d(all_idx, hold, assume_unique=True)
    if len(hold) < min_rows or len(fit) < min_rows:
        return all_idx, np.empty(0, dtype=np.int64)
    return fit, hold


def take_rows(X: FeatureMatrix, idx: np.ndarray) -> FeatureMatrix:
    """Row subset that works for CSR and dense input alike."""
    return X[idx]


def to_csr(X: FeatureMatrix) -> sp.csr_matrix:
    """Canonical CSR form of a feature matrix for XGBoost.

    XGBoost treats the structural zeros of a CSR matrix as *missing* but the zeros
    of a dense array as the value ``0``. A model fitted on one representation and
    queried with the other would route rows differently through the trees. Every
    xgboost-backed model therefore converts its input to CSR first, so training
    and prediction agree no matter which form the caller passes. Presence/absence
    and count features lose nothing: "absent" and "missing" are the same statement.
    """
    matrix = as_matrix(X)
    if sp.issparse(matrix):
        return matrix
    return sp.csr_matrix(matrix)


# -------------------------------------------------------------------- json io
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as indented JSON (parents created; numpy scalars coerced).

    ``inf``/``nan`` are written as the JSON-incompatible tokens Python emits by
    default; :func:`read_json` reads them back. Only model metadata goes through
    here, never data tables.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object written by :func:`write_json`."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return payload


class ModelBundleMixin:
    """Small helpers shared by concrete models (file names and fitted-state checks)."""

    PARAMS_FILE: ClassVar[str] = "params.json"
    MODEL_FILE: ClassVar[str] = "model.ubj"

    name: str
    feature_names_: list[str] | None

    def _require_fitted(self) -> None:
        if getattr(self, "feature_names_", None) is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit() or load() first")

    def _check_predict_input(self, X: Any) -> FeatureMatrix:
        self._require_fitted()
        matrix = as_matrix(X)
        expected = len(self.feature_names_ or [])
        if matrix.shape[1] != expected:
            raise ValueError(
                f"X has {matrix.shape[1]} columns but the model was fitted on {expected} features"
            )
        return matrix
