"""Tests for the sparse, block-wise sketch distances (``sketch.iter_distance_edges``) and the
order-preserving process pool they run on (``genome2mic.parallel``).

Scale finding #25/#33: lineage clustering must never build an ``n x n`` matrix, and the
per-genome work must use every core while giving byte-identical output.
"""

from __future__ import annotations

import operator

import numpy as np
import pytest

from genome2mic import parallel
from genome2mic import sketch as sk


def random_sketches(rng: np.random.Generator, n: int, s: int = 150, n_families: int = 4) -> np.ndarray:
    families = [np.unique(rng.integers(1, 2**63, size=3 * s, dtype=np.uint64)) for _ in range(n_families)]
    rows = []
    for _ in range(n):
        base = families[int(rng.integers(0, n_families))]
        n_swap = int(rng.integers(0, s // 3))
        kept = np.delete(base, rng.choice(base.size, size=n_swap, replace=False))
        rows.append(np.unique(np.concatenate([kept, rng.integers(1, 2**63, size=n_swap, dtype=np.uint64)]))[:s])
    return sk.stack_sketches(rows)


def dense_edges(D: np.ndarray, cut: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(np.triu(D <= cut, k=1))
    return rows, cols, D[rows, cols]


@pytest.mark.parametrize("cut", [0.0, 0.002, 0.01, 0.05, 1.0])
def test_distance_edges_equal_the_dense_matrix_within_the_cut(cut: float) -> None:
    S = random_sketches(np.random.default_rng(1), 45)
    D = sk.pairwise_distances(S)
    rows, cols, d = dense_edges(D, cut)
    for block_rows, rows_per_task in ((512, 64), (3, 2), (1, 1)):
        blocks = list(sk.iter_distance_edges(S, max_distance=cut, block_rows=block_rows, rows_per_task=rows_per_task, threads=1))
        got = sk.DistanceEdges.concat(blocks)
        assert np.array_equal(got.rows, rows) and np.array_equal(got.cols, cols)
        assert np.array_equal(got.distances, d)  # same kernel -> bit-identical values
        assert got.rows.dtype == np.int64 and got.distances.dtype == np.float64
    assert sk.distance_edges(S, max_distance=cut, threads=1).size == rows.size


def test_distance_edges_parallel_equals_serial(monkeypatch) -> None:
    S = random_sketches(np.random.default_rng(2), 60)
    serial = sk.distance_edges(S, max_distance=0.03, threads=1)
    monkeypatch.setattr(sk, "PARALLEL_MIN_WORK", 0.0)
    pooled = sk.DistanceEdges.concat(list(sk.iter_distance_edges(S, max_distance=0.03, rows_per_task=4, threads=3)))
    assert np.array_equal(serial.rows, pooled.rows) and np.array_equal(serial.cols, pooled.cols)
    assert np.array_equal(serial.distances, pooled.distances)
    assert sk._EDGE_STATE == {}  # nothing left behind in the parent


def test_distance_edges_edge_cases() -> None:
    S = random_sketches(np.random.default_rng(3), 3)
    assert list(sk.iter_distance_edges(S[:1], max_distance=1.0)) == []
    assert sk.distance_edges(S[:1]).size == 0
    with pytest.raises(ValueError):
        list(sk.iter_distance_edges(S, max_distance=-0.1))
    with pytest.raises(ValueError):
        list(sk.iter_distance_edges(S, block_rows=0))
    with pytest.raises(ValueError):
        list(sk.iter_distance_edges(np.array([[3, 1]], dtype=np.uint64)))  # unsorted row


def test_pairwise_distances_unchanged_by_kernel_refactor() -> None:
    S = random_sketches(np.random.default_rng(4), 12)
    D = sk.pairwise_distances(S)
    for i in range(12):
        for j in range(12):
            assert D[i, j] == pytest.approx(sk.mash_distance(S[i], S[j]), abs=1e-12)


# ---------------------------------------------------------------------------
# genome2mic.parallel
# ---------------------------------------------------------------------------
def test_worker_count_rules(monkeypatch) -> None:
    assert parallel.resolve_threads(3) == 3
    assert parallel.resolve_threads(None) >= 1
    with pytest.raises(ValueError):
        parallel.resolve_threads(0)
    monkeypatch.setattr(parallel, "MIN_PARALLEL_ITEMS", 10)
    assert parallel.worker_count(4, 9) == 1  # small input stays in-process
    assert parallel.worker_count(4, 10) == 4
    assert parallel.worker_count(4, 3, min_items=1) == 3  # never more workers than items
    assert parallel.worker_count(1, 1000) == 1


@pytest.mark.parametrize("workers", [1, 2])
def test_ordered_map_preserves_order_and_propagates_errors(workers: int) -> None:
    items = list(range(-20, 20))
    out = list(parallel.ordered_map(operator.neg, items, workers=workers, chunksize=3, max_pending=2))
    assert out == [-x for x in items]
    with pytest.raises(ValueError):
        list(parallel.ordered_map(int, ["1", "2", "x", "4"], workers=workers))
    with pytest.raises(ValueError):
        list(parallel.ordered_map(operator.neg, [1], workers=workers, chunksize=0))


def test_share_arrays_memory_maps_for_workers_and_passes_through_in_process() -> None:
    arrays = {"a": np.arange(10, dtype=np.uint64)}
    with parallel.share_arrays(arrays, enabled=True) as spec:
        assert isinstance(spec["a"], str)
        loaded = parallel.load_shared(spec)
        assert isinstance(loaded["a"], np.memmap) and np.array_equal(loaded["a"], arrays["a"])
        assert not loaded["a"].flags.writeable
    with parallel.share_arrays(arrays, enabled=False) as spec:
        assert parallel.load_shared(spec)["a"] is arrays["a"]


def _double_or_die_in_worker(task: tuple[int, int]) -> int:
    """Doubles x; a *worker* process (pid != parent) handling x == 5 dies abruptly."""
    import os

    x, parent_pid = task
    if x == 5 and os.getpid() != parent_pid:
        os._exit(1)
    return 2 * x


def test_ordered_map_falls_back_in_process_when_a_worker_dies(caplog: pytest.LogCaptureFixture) -> None:
    import os

    caplog.set_level("WARNING", logger="genome2mic")
    tasks = [(x, os.getpid()) for x in range(12)]
    out = list(parallel.ordered_map(_double_or_die_in_worker, tasks, workers=2, max_pending=1))
    assert out == [2 * x for x in range(12)]  # same output, in order, no duplicates
    assert "computing the remaining" in caplog.text
