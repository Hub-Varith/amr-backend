"""Tests for ``genome2mic.features.select`` (fold-side selection and the forbidden-feature guard)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse import _compressed

from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.features import select as fs

N_TRAIN = 30
N_TEST = 10
N_ROWS = N_TRAIN + N_TEST


@pytest.fixture
def unitig_world() -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """``(U, train_idx, y)``: 40 genomes x 7 pattern columns with known behaviour.

    col0 tracks ``y`` (present when y > median); col1 is its complement (same
    |corr|); col2 is noise; col3 is in every genome (dropped > 99 %); col4 in none
    (dropped < 1 %); col5 appears only in test rows (train frequency 0 -> dropped);
    col6 is col0 with five flips (moderate correlation).
    """
    rng = np.random.default_rng(42)
    y = rng.normal(loc=1.0, scale=2.0, size=N_ROWS)
    y = np.round(y)  # log2 steps
    col0 = (y > np.median(y)).astype(np.int8)
    col1 = 1 - col0
    col2 = rng.integers(0, 2, size=N_ROWS).astype(np.int8)
    col3 = np.ones(N_ROWS, dtype=np.int8)
    col4 = np.zeros(N_ROWS, dtype=np.int8)
    col5 = np.zeros(N_ROWS, dtype=np.int8)
    col5[N_TRAIN:] = 1
    col6 = col0.copy()
    flips = rng.choice(N_TRAIN, size=5, replace=False)
    col6[flips] = 1 - col6[flips]
    dense = np.stack([col0, col1, col2, col3, col4, col5, col6], axis=1)
    U = sp.csr_matrix(dense, dtype=np.int8)
    train_idx = np.arange(N_TRAIN)
    return U, train_idx, y


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
def test_assert_no_forbidden_catches_st_and_friends() -> None:
    with pytest.raises(ContractViolation, match="forbidden metadata"):
        fs.assert_no_forbidden(["gene_blakpc_2", "st"])
    with pytest.raises(ContractViolation):
        fs.assert_no_forbidden(["u_000001", " ST "])  # case/whitespace-insensitive
    for name in sorted(fs.FORBIDDEN_FEATURES):
        with pytest.raises(ContractViolation):
            fs.assert_no_forbidden(["gene_x", name])
    fs.assert_no_forbidden(["gene_blakpc_2", "point_gyra_s83l", "n_class_beta_lactam", "u_000017"])
    fs.assert_no_forbidden([])


def test_assert_no_forbidden_catches_label_and_key_columns_and_prefixes() -> None:
    with pytest.raises(ContractViolation, match="label or key"):
        fs.assert_no_forbidden(["gene_x", "mic_upper"])
    with pytest.raises(ContractViolation):
        fs.assert_no_forbidden(["genome_id"])
    fs.assert_no_forbidden(["gene_x", "something_else"])  # unprefixed is allowed by default
    with pytest.raises(ContractViolation, match="prefix"):
        fs.assert_no_forbidden(["gene_x", "something_else"], require_prefix=True)
    fs.assert_no_forbidden(["gene_x", "u_000001"], require_prefix=True)
    assert fs.FORBIDDEN_FEATURES == {
        "lineage_cluster", "st", "country", "year", "source", "isolation_source", "biosample", "split", "fold"
    }


# ---------------------------------------------------------------------------
# select_known
# ---------------------------------------------------------------------------
def _known_frame() -> pd.DataFrame:
    n = 12
    frame = pd.DataFrame(
        {
            "genome_id": pd.array([f"g{i:02d}" for i in range(n)], dtype="str"),
            "species": pd.array(["KPNEU"] * n, dtype="str"),
            "gene_a": np.zeros(n, dtype=np.int8),
            "gene_b": np.zeros(n, dtype=np.int8),
            "point_c": np.zeros(n, dtype=np.int8),
            "n_class_x": np.zeros(n, dtype=np.int8),
        }
    )
    frame.loc[0:5, "gene_a"] = 1          # 6 training genomes
    frame.loc[[0, 1, 8, 9, 10, 11], "gene_b"] = 1  # 2 training + 4 test genomes
    frame.loc[0:4, "point_c"] = 1         # exactly 5 training genomes
    frame.loc[0:6, "n_class_x"] = 2       # counts, non-zero in 7 training genomes
    return frame


def test_select_known_counts_training_rows_only() -> None:
    X = _known_frame()
    train_idx = np.arange(8)
    log = DropLog("select")
    kept = fs.select_known(X, train_idx, min_count=5, droplog=log)
    assert kept == ["gene_a", "point_c", "n_class_x"]  # gene_b has 6 overall but only 2 in training
    assert log.records[0].reason == "known_rare_feature" and log.records[0].n_dropped == 1
    # Boolean mask form and a stricter threshold.
    mask = np.zeros(len(X), dtype=bool)
    mask[:8] = True
    assert fs.select_known(X, mask, min_count=6) == ["gene_a", "n_class_x"]
    assert fs.select_known(X, train_idx, min_count=1) == ["gene_a", "gene_b", "point_c", "n_class_x"]
    assert fs.select_known(X[["genome_id", "species"]], train_idx) == []


def test_select_known_rejects_forbidden_label_and_null_columns() -> None:
    X = _known_frame()
    with pytest.raises(ContractViolation):
        fs.select_known(X.assign(st=pd.array(["ST258"] * len(X), dtype="str")), np.arange(8))
    with pytest.raises(ContractViolation):
        fs.select_known(X.assign(fold=0), np.arange(8))
    with pytest.raises(ContractViolation):
        fs.select_known(X.assign(mic_upper=8.0), np.arange(8))
    with_null = X.astype({"gene_a": "Int8"})
    with_null.loc[2, "gene_a"] = pd.NA
    with pytest.raises(ContractViolation, match="nulls"):
        fs.select_known(with_null, np.arange(8))
    with pytest.raises(ValueError):
        fs.select_known(X, np.array([0, 0, 1]))  # duplicate positions
    with pytest.raises(ValueError):
        fs.select_known(X, np.array([], dtype=int))
    with pytest.raises(ValueError):
        fs.select_known(X, np.arange(8), min_count=0)


# ---------------------------------------------------------------------------
# select_unitigs
# ---------------------------------------------------------------------------
def test_select_unitigs_picks_the_column_correlated_with_y(unitig_world) -> None:
    U, train_idx, y = unitig_world
    log = DropLog("select")
    top2 = fs.select_unitigs(U, train_idx, y, top_k=2, droplog=log)
    assert top2.tolist() == [0, 1]  # col0 and its complement tie on |corr|; both beat the rest
    assert top2.dtype == np.int64
    assert fs.select_unitigs(U, train_idx, y, top_k=1).tolist() == [0]  # tie -> lowest index
    top3 = fs.select_unitigs(U, train_idx, y, top_k=3)
    assert top3.tolist() == [0, 1, 6]  # the 5-flip copy ranks next
    everything = fs.select_unitigs(U, train_idx, y, top_k=100)
    assert everything.tolist() == [0, 1, 2, 6]  # 3 (all), 4 (none), 5 (test-only) never pass the frequency window
    reasons = {r.reason: r.n_dropped for r in log.records}
    assert reasons == {"unitig_below_min_freq": 2, "unitig_above_max_freq": 1, "unitig_beyond_top_k": 2}


def test_select_unitigs_frequency_uses_training_rows_only(unitig_world) -> None:
    U, train_idx, y = unitig_world
    chosen = fs.select_unitigs(U, train_idx, y, top_k=100)
    assert 5 not in chosen  # present in 10/40 genomes overall, 0/30 training genomes
    # Make the training rows the test rows instead: col5 becomes constant 1 there and is dropped as too frequent.
    test_idx = np.arange(N_TRAIN, N_ROWS)
    log = DropLog("select")
    fs.select_unitigs(U, test_idx, y, top_k=100, droplog=log)
    reasons = {r.reason: r.n_dropped for r in log.records}
    assert reasons["unitig_above_max_freq"] == 2  # col3 and col5


def test_select_unitigs_accepts_y_aligned_with_matrix_or_with_train_idx(unitig_world) -> None:
    U, train_idx, y = unitig_world
    full = fs.select_unitigs(U, train_idx, y, top_k=3)
    train_only = fs.select_unitigs(U, train_idx, y[train_idx], top_k=3)
    mask = np.zeros(N_ROWS, dtype=bool)
    mask[train_idx] = True
    masked = fs.select_unitigs(U, mask, y, top_k=3)
    assert full.tolist() == train_only.tolist() == masked.tolist()
    # A permuted train_idx with a train-aligned y must still agree with the matrix-aligned call.
    perm = np.random.default_rng(0).permutation(train_idx)
    assert fs.select_unitigs(U, perm, y[perm], top_k=3).tolist() == full.tolist()
    with pytest.raises(ValueError, match="y_point has length"):
        fs.select_unitigs(U, train_idx, y[:7], top_k=3)


def test_column_correlations_match_dense_pearson(unitig_world) -> None:
    U, train_idx, y = unitig_world
    corr = fs.column_correlations(U, train_idx, y)
    dense = U[train_idx].toarray().astype(float)  # dense only to compute the expected values in the test
    for j in (0, 1, 2, 6):
        expected = np.corrcoef(dense[:, j], y[train_idx])[0, 1]
        assert corr[j] == pytest.approx(expected, abs=1e-12)
    assert np.isnan(corr[[3, 4, 5]]).all()  # zero variance on training rows
    assert corr[0] == pytest.approx(-corr[1])
    # Constant y -> undefined everywhere, and select_unitigs falls back to column order.
    assert np.isnan(fs.column_correlations(U, train_idx, np.full(N_ROWS, 3.0))).all()
    assert fs.select_unitigs(U, train_idx, np.full(N_ROWS, 3.0), top_k=2).tolist() == [0, 1]


def test_select_unitigs_ignores_non_finite_y_rows(unitig_world) -> None:
    U, train_idx, y = unitig_world
    y_bad = y.copy()
    y_bad[[1, 2]] = np.nan
    corr = fs.column_correlations(U, train_idx, y_bad)
    keep = np.setdiff1d(train_idx, [1, 2])
    expected = np.corrcoef(U[keep].toarray()[:, 0].astype(float), y[keep])[0, 1]
    assert corr[0] == pytest.approx(expected, abs=1e-12)


def test_select_unitigs_never_densifies(unitig_world, monkeypatch) -> None:
    U, train_idx, y = unitig_world

    def boom(*_a, **_k):
        raise AssertionError("sparse matrix was densified")

    monkeypatch.setattr(_compressed._cs_matrix, "toarray", boom)
    monkeypatch.setattr(_compressed._cs_matrix, "todense", boom)
    chosen = fs.select_unitigs(U, train_idx, y, top_k=3)
    assert chosen.tolist() == [0, 1, 6]
    big = sp.csr_matrix((np.ones(3, dtype=np.int8), ([0, 1, 2], [5, 1_000_000, 1_999_999])),
                        shape=(N_ROWS, 2_000_000), dtype=np.int8)
    assert fs.select_unitigs(big, train_idx, y, top_k=5).tolist() == [5, 1_000_000, 1_999_999]


def test_select_unitigs_edge_cases(unitig_world) -> None:
    U, train_idx, y = unitig_world
    empty = sp.csr_matrix((N_ROWS, 0), dtype=np.int8)
    log = DropLog("select")
    assert fs.select_unitigs(empty, train_idx, y, droplog=log).size == 0
    assert len(log) == 3
    # Nothing passes the window -> empty result, not an error.
    only_constant = sp.csr_matrix(U[:, [3, 4]])
    assert fs.select_unitigs(only_constant, train_idx, y).size == 0
    # Accepts CSC / COO input and int8 values > 1 (presence is non-zero).
    coo = sp.coo_matrix(U)
    assert fs.select_unitigs(coo, train_idx, y, top_k=2).tolist() == [0, 1]
    scaled = sp.csr_matrix(U, dtype=np.int8)
    scaled.data[:] = 3
    assert fs.select_unitigs(scaled, train_idx, y, top_k=2).tolist() == [0, 1]
    with pytest.raises(ValueError):
        fs.select_unitigs(U, train_idx, y, top_k=0)
    with pytest.raises(ValueError):
        fs.select_unitigs(U, train_idx, y, min_freq=0.9, max_freq=0.1)
    with pytest.raises(ValueError):
        fs.select_unitigs(U, np.array([0, 99]), y)
    with pytest.raises(ValueError):
        fs.select_unitigs(U, np.array([0.5, 1.5]), y)


# ---------------------------------------------------------------------------
# #29: binarise once, no float copy -- results identical to the previous implementation
# ---------------------------------------------------------------------------
def _old_binary_block(U: sp.spmatrix, pos: np.ndarray) -> sp.csr_matrix:
    """Verbatim pre-#29 ``_binary_training_block`` (float64 copy)."""
    Ut = sp.csr_matrix(U)[pos]
    Ut = sp.csr_matrix(Ut, dtype=np.float64)
    Ut.eliminate_zeros()
    Ut.data[:] = 1.0
    Ut.sum_duplicates()
    Ut.data[:] = 1.0
    return Ut


def _old_correlations(U: sp.spmatrix, pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Verbatim pre-#29 ``column_correlations`` body (y already aligned with ``pos``)."""
    Ut = _old_binary_block(U, pos)
    finite = np.isfinite(y)
    if (~finite).any():
        Ut = sp.csr_matrix(Ut[np.flatnonzero(finite)])
        y = y[finite]
    n = y.size
    corr = np.full(U.shape[1], np.nan)
    if n < 2:
        return corr
    yc = y - y.mean()
    sy = float(np.sqrt(np.mean(yc * yc)))
    if sy == 0.0:
        return corr
    p = np.asarray(Ut.sum(axis=0)).ravel() / n
    cov = np.asarray(Ut.T @ yc).ravel() / n
    var_x = p * (1.0 - p)
    ok = var_x > 0
    corr[ok] = cov[ok] / (np.sqrt(var_x[ok]) * sy)
    return corr


def _messy_matrix(rng: np.random.Generator, n_rows: int, n_cols: int, dtype) -> sp.csr_matrix:
    """Random sparse 0/1-ish matrix with explicit zeros, duplicates and non-unit values."""
    nnz = int(n_rows * n_cols * 0.3)
    r = rng.integers(0, n_rows, size=nnz)
    c = rng.integers(0, n_cols, size=nnz)
    v = rng.choice(np.array([0, 1, 1, 1, 2], dtype=np.float64), size=nnz)
    if np.dtype(dtype).kind == "f":
        v = v * 0.5  # 0.5 and 1.0 are presence, 0.0 an explicit zero
    m = sp.csr_matrix((v.astype(dtype), (r, c)), shape=(n_rows, n_cols))  # COO -> CSR sums duplicates
    m.data[: min(3, m.nnz)] = 0  # explicit stored zeros
    # Re-introduce duplicate entries (non-canonical CSR): row 0's entries stored twice.
    k = int(m.indptr[1])
    indices = np.concatenate([m.indices[:k], m.indices[:k], m.indices[k:]])
    data = np.concatenate([m.data[:k], m.data[:k], m.data[k:]])
    indptr = m.indptr.copy()
    indptr[1:] += k
    dup = sp.csr_matrix((data, indices, indptr), shape=m.shape)
    assert not dup.has_canonical_format or k == 0
    return dup


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("dtype", [np.int8, np.float64])
def test_column_correlations_and_selection_identical_to_previous_implementation(seed: int, dtype, monkeypatch) -> None:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = 120, 60
    U = _messy_matrix(rng, n_rows, n_cols, dtype)
    y = np.round(rng.normal(1.0, 2.0, size=n_rows))
    if seed % 2:
        y[rng.choice(n_rows, size=7, replace=False)] = np.nan  # non-finite rows are skipped
    pos = np.sort(rng.choice(n_rows, size=90, replace=False))
    monkeypatch.setattr(fs, "SUM_CHUNK_NNZ", 17)  # many chunks: accumulation order must still match
    new = fs.column_correlations(U, pos, y)
    old = _old_correlations(U, pos, y[pos])
    assert np.array_equal(np.isnan(new), np.isnan(old))
    assert np.array_equal(new[~np.isnan(new)], old[~np.isnan(old)])  # bit-identical, not approx

    # select_unitigs: same frequency window and the same chosen columns as the old pipeline
    Ut_old = _old_binary_block(U, pos)
    freq_old = np.asarray(Ut_old.sum(axis=0)).ravel() / pos.size
    cand = np.flatnonzero((freq_old >= 0.05) & (freq_old <= 0.95))
    score = np.where(np.isnan(old[cand]), -np.inf, np.abs(old[cand]))
    expected = np.sort(cand[np.lexsort((cand, -score))][:10])
    got = fs.select_unitigs(U, pos, y, min_freq=0.05, max_freq=0.95, top_k=10)
    assert np.array_equal(got, expected)


def test_binary_training_block_is_one_int8_copy() -> None:
    U = sp.csr_matrix(np.array([[0, 2, 0], [1, 0, 1], [0, 0, 1]], dtype=np.int8))
    U.data[0] = 0  # stored zero
    Ut = fs._binary_training_block(U, np.array([0, 1]))
    assert Ut.dtype == np.int8 and Ut.has_canonical_format
    assert Ut.toarray().tolist() == [[0, 0, 0], [1, 0, 1]]
    assert (U.data == 0).sum() == 1  # the caller's matrix is untouched
    F = sp.csr_matrix(np.array([[0.5, 0.0], [0.0, 3.0]]))
    assert fs._binary_training_block(F, np.array([0, 1])).toarray().tolist() == [[1, 0], [0, 1]]
    # 256 duplicate int8 ones would wrap to 0 if summed in int8; they are presence.
    dup = sp.csr_matrix((np.ones(256, dtype=np.int8), np.zeros(256, dtype=np.int32), np.array([0, 256, 256])), shape=(2, 3))
    assert fs._binary_training_block(dup, np.array([0, 1])).toarray().tolist() == [[1, 0, 0], [0, 0, 0]]


def test_select_unitigs_peak_memory_is_below_one_float_copy(monkeypatch) -> None:
    import tracemalloc

    rng = np.random.default_rng(3)
    n_rows, n_cols = 4000, 2000
    U = sp.random(n_rows, n_cols, density=0.25, format="csr", random_state=4, dtype=np.float64)
    U = sp.csr_matrix(U, dtype=np.int8)
    U.data[:] = 1
    y = rng.normal(size=n_rows)
    pos = np.arange(n_rows)
    monkeypatch.setattr(fs, "SUM_CHUNK_NNZ", 1 << 18)  # chunk temporaries small relative to this matrix
    tracemalloc.start()
    try:
        fs.select_unitigs(U, pos, y, top_k=50)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The old code held two float64 copies (12 B per stored entry each) plus the int8 slice at
    # once (~29 B per entry); now: one int8 block (5 B per entry) plus bounded chunk temporaries.
    assert peak < 9 * U.nnz, (peak, U.nnz)
