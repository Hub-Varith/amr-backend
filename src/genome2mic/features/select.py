"""Fold-side feature selection (``DATA_CONTRACT.md`` stage 8 note, CLAUDE.md rule 4).

Everything here runs **inside a training fold**: the frequency filters and the
correlation ranking see only the rows the caller marks as training rows
(``train_idx``). Running them on the full dataset would leak test information
into feature choice.

* :func:`select_known` -- known-AMR columns present in at least ``min_count``
  training genomes (the contract's "rare feature filter", applied fold-side).
* :func:`select_unitigs` -- unitig/k-mer pattern columns: training-frequency
  window, then the ``top_k`` columns by ``|Pearson correlation|`` with the log2
  label point. All algebra is sparse (``U.T @ y``); the matrix is never densified.
  This is the stand-in for pyseer's mixed-model selection; ``select_unitigs_pyseer``
  is left for when ``pyseer`` is on PATH (not implemented here).
* :data:`FORBIDDEN_FEATURES` / :func:`assert_no_forbidden` -- the hard guard that
  ``lineage_cluster``, ``st``, ``country``, ``year``, ``source``,
  ``isolation_source``, ``biosample``, ``split`` and ``fold`` (and label / key
  columns) never enter a feature matrix. Raises ``ContractViolation``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation

logger = logging.getLogger(__name__)

__all__ = [
    "FORBIDDEN_FEATURES",
    "NON_FEATURE_COLUMNS",
    "KNOWN_PREFIXES",
    "UNITIG_PREFIX",
    "FEATURE_PREFIXES",
    "STAGE",
    "assert_no_forbidden",
    "known_feature_columns",
    "select_known",
    "column_correlations",
    "select_unitigs",
]

STAGE: str = "select"

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"lineage_cluster", "st", "country", "year", "source", "isolation_source", "biosample", "split", "fold"}
)
"""Metadata that exists for splitting / evaluation only. Never a feature (contract rule 2)."""

NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        # identifiers
        "genome_id", "species", "drug",
        # labels and label-derived columns (stage 2)
        "mic_lower", "mic_upper", "censor", "sir", "raw_result", "method", "standard",
        "standard_year", "lab_sir", "lab_lower", "lab_upper", "y", "y_point",
        # split-table extras (stage 7)
        "external_set", "lolo_lineage", "cluster_method",
    }
)
"""Key, label and split-table columns: not forbidden metadata, but never a feature either."""

KNOWN_PREFIXES: tuple[str, ...] = ("gene_", "point_", "n_class_")
UNITIG_PREFIX: str = "u_"
FEATURE_PREFIXES: tuple[str, ...] = (*KNOWN_PREFIXES, UNITIG_PREFIX)


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
def assert_no_forbidden(feature_names: Iterable[str], *, require_prefix: bool = False) -> None:
    """Raise ``ContractViolation`` if any name is forbidden metadata or a label/key column.

    Matching is case-insensitive on the stripped name. With ``require_prefix=True``
    every name must also start with one of :data:`FEATURE_PREFIXES`
    (``gene_``, ``point_``, ``n_class_``, ``u_``).

    Args:
        feature_names: Column names about to enter a feature matrix.
        require_prefix: Also enforce the feature-prefix convention.
    """
    names = [str(n) for n in feature_names]
    lowered = [n.strip().lower() for n in names]
    forbidden = sorted({n for n, low in zip(names, lowered) if low in FORBIDDEN_FEATURES})
    if forbidden:
        raise ContractViolation(
            f"forbidden metadata in feature matrix: {forbidden} "
            "(lineage/geography/time/source/split columns are for evaluation only)",
            STAGE,
        )
    non_feature = sorted({n for n, low in zip(names, lowered) if low in NON_FEATURE_COLUMNS})
    if non_feature:
        raise ContractViolation(
            f"label or key columns in feature matrix: {non_feature}", STAGE
        )
    if require_prefix:
        bad = sorted({n for n, low in zip(names, lowered) if not low.startswith(FEATURE_PREFIXES)})
        if bad:
            raise ContractViolation(
                f"feature names without a gene_/point_/n_class_/u_ prefix: {bad[:10]}"
                + (" ..." if len(bad) > 10 else ""),
                STAGE,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_positions(train_idx: Sequence[int] | np.ndarray | pd.Series, n_rows: int) -> np.ndarray:
    """Normalise ``train_idx`` (boolean mask or integer positions) to ``int64`` positions."""
    if isinstance(train_idx, pd.Series):
        train_idx = train_idx.to_numpy()
    arr = np.asarray(train_idx)
    if arr.dtype == bool:
        if arr.shape != (n_rows,):
            raise ValueError(f"boolean train_idx has shape {arr.shape}; expected ({n_rows},)")
        pos = np.flatnonzero(arr).astype(np.int64)
    else:
        if arr.ndim != 1:
            raise ValueError("train_idx must be a 1-D array of positions or a boolean mask")
        if arr.size and not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"train_idx must be integer positions or a boolean mask; got {arr.dtype}")
        pos = arr.astype(np.int64)
        if pos.size and (pos.min() < 0 or pos.max() >= n_rows):
            raise ValueError(f"train_idx positions out of range for {n_rows} rows")
        if np.unique(pos).size != pos.size:
            raise ValueError("train_idx contains duplicate positions")
    if pos.size == 0:
        raise ValueError("train_idx selects no training rows")
    return pos


def known_feature_columns(X_known: pd.DataFrame) -> list[str]:
    """Known-AMR feature columns of a stage-5 table (``gene_``, ``point_``, ``n_class_``).

    Raises ``ContractViolation`` if the table carries a forbidden column at all;
    other unprefixed columns (``genome_id``, ``species``) are ignored, and any
    unexpected unprefixed column is reported at WARNING.
    """
    columns = [str(c) for c in X_known.columns]
    assert_no_forbidden([c for c in columns if c.strip().lower() not in ("genome_id", "species")])
    features = [c for c in columns if c.startswith(KNOWN_PREFIXES)]
    stray = [c for c in columns if c not in features and c.strip().lower() not in ("genome_id", "species")]
    if stray:
        logger.warning("select_known: ignoring %d unprefixed non-feature columns: %s", len(stray), stray[:10])
    return features


# ---------------------------------------------------------------------------
# Known-AMR selection
# ---------------------------------------------------------------------------
def select_known(
    X_known: pd.DataFrame,
    train_idx: Sequence[int] | np.ndarray | pd.Series,
    min_count: int = 5,
    droplog: DropLog | None = None,
) -> list[str]:
    """Known-AMR columns present (non-zero) in at least ``min_count`` training genomes.

    Args:
        X_known: Stage-5 wide table (``genome_id, species, gene_*, point_*, n_class_*``),
            rows aligned with the model's row order.
        train_idx: Positions (or boolean mask) of the fold's training rows. Only
            these rows are counted.
        min_count: Minimum number of training genomes carrying the feature.
        droplog: Receives the ``known_rare_feature`` count; a local log is used when ``None``.

    Returns:
        Kept column names in their original order.

    Raises:
        ContractViolation: forbidden/label columns in ``X_known`` or nulls in feature columns.
    """
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    log = droplog if droplog is not None else DropLog(STAGE)
    features = known_feature_columns(X_known)
    pos = _as_positions(train_idx, len(X_known))
    if not features:
        log.drop("known_rare_feature", 0, detail="no known-AMR feature columns")
        return []
    Xt = X_known.iloc[pos][features]
    if Xt.isna().to_numpy().any():
        bad = [c for c in features if Xt[c].isna().any()]
        raise ContractViolation(f"nulls in known-AMR feature columns {bad[:10]} (contract: zero-filled)", STAGE)
    counts = (Xt.to_numpy() != 0).sum(axis=0)
    keep = counts >= min_count
    kept = [c for c, k in zip(features, keep) if k]
    log.drop(
        "known_rare_feature", int((~keep).sum()),
        detail=f"present in < {min_count} of {pos.size} training genomes",
    )
    logger.info("select_known: kept %d of %d known-AMR columns (min_count=%d, n_train=%d)",
                len(kept), len(features), min_count, pos.size)
    return kept


# ---------------------------------------------------------------------------
# Unitig selection
# ---------------------------------------------------------------------------
SUM_CHUNK_NNZ: int = 1 << 22
"""Stored entries processed per step by the column sums below (bounds temporaries to ~50 MB)."""


def _binary_training_block(U: sp.spmatrix, pos: np.ndarray) -> sp.csr_matrix:
    """Training rows of ``U`` as a canonical CSR ``int8`` 0/1 matrix (one copy, ~5 B per stored entry).

    Explicit zeros are dropped in the input dtype first, so e.g. a float 0.5 counts
    as presence exactly as before; duplicates are merged and every stored entry
    is 1. No float copy of the matrix is made (the sums below work on the int8
    block directly).
    """
    Ut = sp.csr_matrix(U)[pos]
    Ut.eliminate_zeros()
    if Ut.dtype != np.int8:
        Ut = sp.csr_matrix((np.ones(Ut.nnz, dtype=np.int8), Ut.indices, Ut.indptr), shape=Ut.shape)
    else:
        Ut.data[:] = 1
    if not Ut.has_canonical_format:  # rare (stage-8 matrices are canonical): sum in int32, no int8 overflow
        Ut = sp.csr_matrix(Ut, dtype=np.int32)
        Ut.sum_duplicates()
        Ut = sp.csr_matrix((np.ones(Ut.nnz, dtype=np.int8), Ut.indices, Ut.indptr), shape=Ut.shape)
    return Ut


def _row_chunks(indptr: np.ndarray, n_rows: int, chunk_nnz: int) -> Iterable[tuple[int, int]]:
    """Consecutive row ranges ``[r0, r1)`` holding about ``chunk_nnz`` stored entries each."""
    r0 = 0
    while r0 < n_rows:
        r1 = int(np.searchsorted(indptr, indptr[r0] + chunk_nnz, side="right")) - 1
        r1 = min(max(r1, r0 + 1), n_rows)
        yield r0, r1
        r0 = r1


def _column_counts(Ut: sp.csr_matrix, rows: np.ndarray | None = None) -> np.ndarray:
    """Stored entries per column of a binary CSR block (``int64``), over all rows or a boolean row subset."""
    n_rows, n_cols = Ut.shape
    counts = np.zeros(n_cols, dtype=np.int64)
    for r0, r1 in _row_chunks(Ut.indptr, n_rows, SUM_CHUNK_NNZ):
        idx = Ut.indices[Ut.indptr[r0] : Ut.indptr[r1]]
        if rows is not None:
            idx = idx[np.repeat(rows[r0:r1], np.diff(Ut.indptr[r0 : r1 + 1]))]
        counts += np.bincount(idx, minlength=n_cols)
    return counts


def _column_dot(Ut: sp.csr_matrix, w: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """``Ut.T @ w`` for a binary CSR block (``float64``), optionally over a boolean row subset.

    Accumulates in row-major order with :func:`numpy.add.at` -- the same order as
    scipy's ``csc_matvec`` -- so the result is bit-identical to ``Ut.T @ w`` on a
    float copy, without making that copy.
    """
    n_rows, n_cols = Ut.shape
    out = np.zeros(n_cols, dtype=np.float64)
    for r0, r1 in _row_chunks(Ut.indptr, n_rows, SUM_CHUNK_NNZ):
        lengths = np.diff(Ut.indptr[r0 : r1 + 1])
        idx = Ut.indices[Ut.indptr[r0] : Ut.indptr[r1]]
        weights = np.repeat(w[r0:r1], lengths)
        if rows is not None:
            keep = np.repeat(rows[r0:r1], lengths)
            idx, weights = idx[keep], weights[keep]
        np.add.at(out, idx, weights)
    return out


def _correlations_from_block(Ut: sp.csr_matrix, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of a binary CSR block with ``y`` (rows aligned)."""
    n_cols = Ut.shape[1]
    finite = np.isfinite(y)
    n_bad = int((~finite).sum())
    rows: np.ndarray | None = None
    if n_bad:
        logger.warning("column_correlations: %d training rows with non-finite y_point ignored", n_bad)
        rows = finite
    yf = y[finite]
    n = yf.size
    corr = np.full(n_cols, np.nan, dtype=np.float64)
    if n < 2:
        return corr
    mean = yf.mean()
    yc_f = yf - mean
    sy = float(np.sqrt(np.mean(yc_f * yc_f)))
    if sy == 0.0:
        logger.warning("column_correlations: y_point is constant on the training rows; correlations undefined")
        return corr
    yc = np.zeros(y.size, dtype=np.float64)
    yc[finite] = yc_f  # non-finite rows are skipped via ``rows``; their value is never read
    p = _column_counts(Ut, rows).astype(np.float64) / n
    cov = _column_dot(Ut, yc, rows) / n
    var_x = p * (1.0 - p)
    ok = var_x > 0
    corr[ok] = cov[ok] / (np.sqrt(var_x[ok]) * sy)
    return corr


def _align_y(y_point: np.ndarray, pos: np.ndarray, n_rows: int) -> np.ndarray:
    """``y`` for the training rows, accepting full-matrix or train-aligned vectors.

    ``len(y) == n_rows`` -> ``y[pos]`` (aligned with the rows of ``U``);
    otherwise ``len(y)`` must equal ``len(pos)`` and is taken as already aligned
    with ``train_idx``. When both lengths coincide the full-matrix reading wins.
    """
    y = np.asarray(y_point, dtype=np.float64).ravel()
    if y.size == n_rows:
        return y[pos]
    if y.size == pos.size:
        return y
    raise ValueError(
        f"y_point has length {y.size}; expected {n_rows} (one per matrix row) or {pos.size} (one per training row)"
    )


def column_correlations(
    U: sp.spmatrix,
    train_idx: Sequence[int] | np.ndarray | pd.Series,
    y_point: np.ndarray,
) -> np.ndarray:
    """Pearson correlation of every (binary) column with ``y_point`` on the training rows.

    Computed with sparse algebra only: ``cov = U.T @ (y - mean(y)) / n`` and
    ``var(x) = p (1 - p)`` for a 0/1 column with frequency ``p``. Columns with
    zero variance, or a constant ``y``, get ``NaN``. Rows with a non-finite ``y``
    are excluded from the correlation (not from the frequency).

    The training rows are binarised once into an ``int8`` block; the sums run on
    it in bounded chunks (no float copy of the matrix).

    Returns:
        ``float64`` array of length ``U.shape[1]``.
    """
    n_rows, _n_cols = U.shape
    pos = _as_positions(train_idx, n_rows)
    y = _align_y(y_point, pos, n_rows)
    return _correlations_from_block(_binary_training_block(U, pos), y)


def select_unitigs(
    U: sp.spmatrix,
    train_idx: Sequence[int] | np.ndarray | pd.Series,
    y_point: np.ndarray,
    min_freq: float = 0.01,
    max_freq: float = 0.99,
    top_k: int = 2000,
    droplog: DropLog | None = None,
) -> np.ndarray:
    """Pattern columns to use in this fold: frequency window, then top-``k`` by ``|corr|``.

    Args:
        U: Sparse genomes x patterns matrix (any scipy format; CSR preferred), int8 0/1.
        train_idx: Positions or boolean mask of the fold's **training** rows.
        y_point: Log2 label point (``mic.label_point_log2``), either one value per
            matrix row or one per training row (see :func:`column_correlations`).
        min_freq, max_freq: Inclusive training-frequency window (contract: drop
            ``< 1 %`` or ``> 99 %``).
        top_k: Maximum number of columns to keep.
        droplog: Receives ``unitig_below_min_freq``, ``unitig_above_max_freq`` and
            ``unitig_beyond_top_k`` counts; a local log is used when ``None``.

    Returns:
        Column indices (``int64``) sorted ascending. Ranking: ``|corr|`` descending,
        ties broken by column index; columns with undefined correlation (constant
        ``y``) fall back to column order with a warning.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not 0.0 <= min_freq <= max_freq <= 1.0:
        raise ValueError("need 0 <= min_freq <= max_freq <= 1")
    log = droplog if droplog is not None else DropLog(STAGE)
    n_rows, n_cols = U.shape
    pos = _as_positions(train_idx, n_rows)
    y = _align_y(y_point, pos, n_rows)
    if n_cols == 0:
        for reason in ("unitig_below_min_freq", "unitig_above_max_freq", "unitig_beyond_top_k"):
            log.drop(reason, 0, detail="no unitig columns")
        return np.empty(0, dtype=np.int64)

    Ut = _binary_training_block(U, pos)  # binarised once; reused for the correlations below
    n_train = pos.size
    freq = _column_counts(Ut).astype(np.float64) / n_train
    below = freq < min_freq
    above = freq > max_freq
    log.drop("unitig_below_min_freq", int(below.sum()),
             detail=f"present in < {min_freq:.0%} of {n_train} training rows")
    log.drop("unitig_above_max_freq", int(above.sum()),
             detail=f"present in > {max_freq:.0%} of {n_train} training rows")
    candidates = np.flatnonzero(~(below | above)).astype(np.int64)
    if candidates.size == 0:
        log.drop("unitig_beyond_top_k", 0, detail="no columns passed the frequency filter")
        logger.warning("select_unitigs: no unitig column passed the frequency window")
        return candidates

    corr = _correlations_from_block(Ut, y)  # Ut is already the binary training block
    del Ut
    score = np.abs(corr[candidates])
    if np.isnan(score).all():
        logger.warning("select_unitigs: correlations undefined; keeping the first %d columns by index", top_k)
        chosen = candidates[:top_k]
    else:
        score = np.where(np.isnan(score), -np.inf, score)
        order = np.lexsort((candidates, -score))  # primary: |corr| desc; secondary: index asc
        chosen = candidates[order][:top_k]
    log.drop("unitig_beyond_top_k", int(candidates.size - chosen.size),
             detail=f"ranked below top_k={top_k} by |corr| with y_point")
    chosen = np.sort(chosen)
    logger.info("select_unitigs: %d of %d columns kept (freq window %.0f%%-%.0f%%, top_k=%d, n_train=%d)",
                chosen.size, n_cols, min_freq * 100, max_freq * 100, top_k, n_train)
    return chosen
