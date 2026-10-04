"""Pure-numpy Mash-like bottom-*s* MinHash sketches of genome assemblies.

Used by lineage clustering, QC species identification and the
nearest-training-genome distance in prediction. No external tool is needed.

How a sketch is built
---------------------
1. Every window of ``k`` bases (default ``K = 21``) is 2-bit encoded
   (A=0, C=1, G=2, T=3) into a ``uint64``. Windows that contain any
   non-ACGT base (``N``, IUPAC codes, gaps, ...) are skipped; the count of
   skipped windows is logged and, when a ``DropLog`` is supplied, recorded.
2. The window and its reverse complement are both encoded and the smaller
   code is kept (the *canonical* k-mer), so a genome and its reverse
   complement give identical sketches.
3. Each canonical code is mixed with splitmix64 (a 64-bit bijection, so
   distinct k-mers never collide for ``k <= 32``).
4. The ``s`` smallest distinct hashes form the sketch (sorted ascending).

Distance
--------
``mash_distance`` follows Mash (Ondov et al. 2016): merge two bottom
sketches, keep the ``s`` smallest values of the union, and estimate the
Jaccard index as ``j = shared / s``. Then ``d = -ln(2j / (1 + j)) / k``,
with ``j == 0 -> 1.0`` and identical sketches ``-> 0.0``.

All public functions are pure. Sketches are 1-D ``uint64`` arrays, sorted
ascending with no duplicates; a sketch collection is a 2-D ``uint64`` array
of shape ``(n_genomes, s)``.

Scale
-----
``pairwise_distances`` returns a dense ``n x n`` matrix and is meant for small
collections only (``16 n^2`` bytes: 20 GB at n = 50,000). Lineage clustering
uses :func:`iter_distance_edges` instead: the same all-pairs computation, run
block-wise (optionally in worker processes) and keeping only the pairs within a
distance cut, so memory is ``O(n s + edges)``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np

from genome2mic import parallel

logger = logging.getLogger(__name__)

__all__ = [
    "K",
    "SKETCH_SIZE",
    "MAX_K",
    "PARALLEL_MIN_WORK",
    "DropLogLike",
    "DistanceEdges",
    "splitmix64",
    "kmer_hashes",
    "sketch",
    "jaccard_counts",
    "distance_from_jaccard",
    "mash_distance",
    "distances_to",
    "pairwise_distances",
    "iter_distance_edges",
    "distance_edges",
    "stack_sketches",
    "save_sketches",
    "load_sketches",
]

K: int = 21
"""Default k-mer length (odd, so a k-mer is never its own reverse complement)."""

SKETCH_SIZE: int = 1000
"""Default number of hashes kept per genome (Mash's ``-s`` default)."""

MAX_K: int = 32
"""Largest k whose 2-bit encoding fits in 64 bits."""

PARALLEL_MIN_WORK: float = 1e9
"""``n_pairs x s`` below which :func:`iter_distance_edges` stays in-process (~1-2 s of work)."""

ROWS_PER_TASK: int = 64
"""Query rows per worker task in :func:`iter_distance_edges` (load balancing only)."""

# ---------------------------------------------------------------------------
# splitmix64 constants (Steele, Lea & Flood 2014). Arithmetic wraps mod 2**64.
# ---------------------------------------------------------------------------
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_SH30 = np.uint64(30)
_SH27 = np.uint64(27)
_SH31 = np.uint64(31)
_TWO = np.uint64(2)

# Byte -> 2-bit code lookup. 4 marks anything that is not A/C/G/T (either case).
_INVALID_CODE = np.uint8(4)
_ENCODE = np.full(256, _INVALID_CODE, dtype=np.uint8)
for _base, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _ENCODE[ord(_base)] = _code
    _ENCODE[ord(_base.lower())] = _code

_NON_ACGT_REASON = "non_acgt_window"


class DropLogLike(Protocol):
    """Structural type for ``genome2mic.droplog.DropLog`` (duck-typed here)."""

    def drop(self, reason: str, n: int, detail: str | None = None) -> None: ...


# ---------------------------------------------------------------------------
# Hashing and encoding
# ---------------------------------------------------------------------------
def splitmix64(x: np.ndarray | int) -> np.ndarray:
    """Mix 64-bit integers with the splitmix64 finaliser.

    Implemented with numpy ``uint64`` arithmetic; every add and multiply wraps
    modulo 2**64 (``np.errstate(over="ignore")`` makes that explicit). The
    function is a bijection on 64-bit values, so distinct inputs never collide.

    Args:
        x: Integer array (any shape) or scalar. Converted to ``uint64``.

    Returns:
        ``uint64`` array of the same shape as ``x`` (at least 1-D).
    """
    z = np.array(x, dtype=np.uint64, copy=True, ndmin=1)
    tmp = np.empty_like(z)
    with np.errstate(over="ignore"):
        np.add(z, _GOLDEN, out=z)
        np.right_shift(z, _SH30, out=tmp)
        np.bitwise_xor(z, tmp, out=z)
        np.multiply(z, _MIX1, out=z)
        np.right_shift(z, _SH27, out=tmp)
        np.bitwise_xor(z, tmp, out=z)
        np.multiply(z, _MIX2, out=z)
        np.right_shift(z, _SH31, out=tmp)
        np.bitwise_xor(z, tmp, out=z)
    return z


def _check_k(k: int) -> None:
    if not isinstance(k, (int, np.integer)) or k < 1 or k > MAX_K:
        raise ValueError(f"k must be an integer in [1, {MAX_K}]; got {k!r}")


def _encode(seq: str | bytes) -> np.ndarray:
    """Map a sequence to 2-bit codes (``uint8``), 4 for any non-ACGT byte."""
    if isinstance(seq, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(bytes(seq), dtype=np.uint8)
    else:
        # errors="replace" keeps one byte per character, so window positions
        # stay aligned; the replacement byte maps to the invalid code.
        raw = np.frombuffer(str(seq).encode("ascii", errors="replace"), dtype=np.uint8)
    return _ENCODE[raw]


def _canonical_kmer_codes(codes: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Canonical 2-bit codes for every valid k-window of ``codes``.

    Vectorised rolling encoding: ``k`` passes over O(L) ``uint64`` arrays.

    Returns:
        ``(canonical_codes, n_skipped)`` where ``n_skipped`` is the number of
        windows dropped because they contain a non-ACGT base.
    """
    n_win = int(codes.size) - k + 1
    if n_win <= 0:
        return np.empty(0, dtype=np.uint64), 0

    invalid = codes > 3
    if invalid.any():
        # Windows [i, i+k) with at least one invalid base, via a prefix sum.
        prefix = np.concatenate(([0], np.cumsum(invalid, dtype=np.int64)))
        bad = (prefix[k : k + n_win] - prefix[:n_win]) > 0
        clean = np.where(invalid, np.uint8(0), codes)
    else:
        bad = None
        clean = codes

    fwd_codes = clean.astype(np.uint64)
    rev_codes = (3 - clean).astype(np.uint64)  # complement; clean <= 3 so no underflow

    fwd = np.zeros(n_win, dtype=np.uint64)
    rev = np.zeros(n_win, dtype=np.uint64)
    for t in range(k):
        # forward: base t of the window lands at bit position 2*(k-1-t)
        np.left_shift(fwd, _TWO, out=fwd)
        np.bitwise_or(fwd, fwd_codes[t : t + n_win], out=fwd)
        # reverse complement: complement of base t lands at bit position 2*t
        np.bitwise_or(rev, rev_codes[t : t + n_win] << np.uint64(2 * t), out=rev)

    canonical = np.minimum(fwd, rev)
    if bad is None:
        return canonical, 0
    n_skipped = int(np.count_nonzero(bad))
    return canonical[~bad], n_skipped


def _hashes_with_skip_count(seq: str | bytes, k: int) -> tuple[np.ndarray, int]:
    """``kmer_hashes`` plus the number of non-ACGT windows that were skipped."""
    _check_k(k)
    canonical, n_skipped = _canonical_kmer_codes(_encode(seq), k)
    if n_skipped:
        logger.debug(
            "kmer_hashes: skipped %d of %d windows containing non-ACGT bases (k=%d)",
            n_skipped,
            canonical.size + n_skipped,
            k,
        )
    return splitmix64(canonical), n_skipped


def kmer_hashes(seq: str | bytes, k: int = K) -> np.ndarray:
    """Hash every canonical k-mer of one sequence.

    Args:
        seq: DNA sequence. Case-insensitive. Any byte other than A/C/G/T
            (``N``, IUPAC codes, ``-``, whitespace, ...) invalidates every
            window that contains it.
        k: k-mer length, ``1 <= k <= 32``.

    Returns:
        ``uint64`` array with one hash per valid window, in sequence order
        (duplicates are kept; ``sketch`` de-duplicates). Empty when the
        sequence is shorter than ``k`` or has no valid window.
    """
    hashes, _ = _hashes_with_skip_count(seq, k)
    return hashes


# ---------------------------------------------------------------------------
# Sketching
# ---------------------------------------------------------------------------
def _bottom_s(values: np.ndarray, s: int) -> np.ndarray:
    """The ``s`` smallest *distinct* values of ``values``, sorted ascending.

    Uses an O(L) partition first and falls back to a full sort only when the
    partitioned candidates hold fewer than ``s`` distinct values.
    """
    if values.size == 0:
        return np.empty(0, dtype=np.uint64)
    kth = 2 * s
    if values.size > kth + 1:
        candidates = np.partition(values, kth)[: kth + 1]
        uniq = np.unique(candidates)
        if uniq.size >= s:
            return uniq[:s]
    return np.unique(values)[:s]


def sketch(
    seqs: Iterable[str | bytes] | str | bytes,
    k: int = K,
    s: int = SKETCH_SIZE,
    droplog: DropLogLike | None = None,
) -> np.ndarray:
    """Bottom-``s`` MinHash sketch of a genome given as one or more sequences.

    Args:
        seqs: Contig sequences (e.g. ``[seq for _, seq in read_fasta(p)]``).
            A single ``str``/``bytes`` is treated as one sequence.
        k: k-mer length.
        s: Sketch size (number of hashes kept).
        droplog: Optional ``DropLog``; when given, the number of windows
            skipped for containing non-ACGT bases is recorded under reason
            ``"non_acgt_window"``.

    Returns:
        Sorted ``uint64`` array of the ``s`` smallest distinct k-mer hashes.
        Shorter than ``s`` only when the genome has fewer than ``s`` distinct
        canonical k-mers (a warning is logged).
    """
    _check_k(k)
    if not isinstance(s, (int, np.integer)) or s < 1:
        raise ValueError(f"s must be a positive integer; got {s!r}")
    if isinstance(seqs, (str, bytes, bytearray)):
        seqs = [seqs]

    current = np.empty(0, dtype=np.uint64)
    n_seqs = n_valid_windows = n_skipped = 0
    for seq in seqs:
        n_seqs += 1
        hashes, skipped = _hashes_with_skip_count(seq, k)
        n_skipped += skipped
        n_valid_windows += int(hashes.size)
        if hashes.size == 0:
            continue
        if current.size >= s:
            # Only hashes below the current s-th smallest can enter the sketch.
            hashes = hashes[hashes < current[-1]]
            if hashes.size == 0:
                continue
        pool = np.concatenate((current, hashes)) if current.size else hashes
        current = _bottom_s(pool, s)

    logger.debug(
        "sketch: %d sequence(s), %d valid windows, %d non-ACGT windows skipped, "
        "%d hashes kept (k=%d, s=%d)",
        n_seqs,
        n_valid_windows,
        n_skipped,
        current.size,
        k,
        s,
    )
    if droplog is not None and n_skipped:
        droplog.drop(
            _NON_ACGT_REASON,
            n_skipped,
            detail=f"k={k}; windows containing a non-ACGT base are not hashed",
        )
    if current.size < s:
        logger.warning(
            "sketch has only %d distinct k-mers, fewer than s=%d; "
            "distances involving it will be noisy",
            current.size,
            s,
        )
    return current


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def _as_sketch(a: np.ndarray, name: str = "sketch") -> np.ndarray:
    arr = np.asarray(a, dtype=np.uint64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty (no valid k-mers); cannot compute a distance")
    return arr


def _as_sorted_sketch(a: np.ndarray, name: str = "sketch") -> np.ndarray:
    arr = _as_sketch(a, name)
    if arr.size > 1 and not np.all(arr[1:] > arr[:-1]):
        raise ValueError(f"{name} must be sorted ascending with no duplicates")
    return arr


def _as_sketch_matrix(sketches: np.ndarray, name: str = "sketches") -> np.ndarray:
    S = np.asarray(sketches, dtype=np.uint64)
    if S.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array of shape (n, s); got shape {S.shape}")
    if S.shape[1] == 0:
        raise ValueError(f"{name} has sketch size 0")
    if S.shape[1] > 1 and not np.all(S[:, 1:] > S[:, :-1]):
        raise ValueError(f"every row of {name} must be sorted ascending with no duplicates")
    return S


def jaccard_counts(a: np.ndarray, b: np.ndarray, s: int | None = None) -> tuple[int, int]:
    """Mash's merged-sketch Jaccard estimate as ``(shared, denominator)``.

    The two bottom sketches are merged; the ``s`` smallest distinct values of
    the union are kept (``denominator = min(s, |a ∪ b|)``) and ``shared`` is
    how many of those occur in both sketches.

    Args:
        a, b: Sketches (``uint64``). Need not be sorted.
        s: Sketch size. Defaults to ``max(len(a), len(b))``, which equals the
            sketch size whenever at least one genome filled its sketch. Pass
            ``min(s_a, s_b)`` explicitly when comparing sketches built with
            different sizes (Mash's convention).

    Raises:
        ValueError: if either sketch is empty.
    """
    a = _as_sketch(a, "a")
    b = _as_sketch(b, "b")
    size = max(a.size, b.size) if s is None else int(s)
    if size < 1:
        raise ValueError(f"s must be positive; got {s!r}")
    union = np.union1d(a, b)
    denom = min(size, int(union.size))
    threshold = union[denom - 1]
    shared = np.intersect1d(a, b)
    common = int(np.count_nonzero(shared <= threshold))
    return common, denom


def distance_from_jaccard(j: float, k: int = K) -> float:
    """Mash distance from a Jaccard estimate: ``-ln(2j / (1 + j)) / k``.

    ``j <= 0`` gives ``1.0`` (nothing shared); ``j >= 1`` gives ``0.0``.
    """
    _check_k(k)
    if j <= 0.0:
        return 1.0
    if j >= 1.0:
        return 0.0
    return -math.log(2.0 * j / (1.0 + j)) / k


def _distances_from_common(common: np.ndarray, denom: int, k: int) -> np.ndarray:
    """Vectorised ``distance_from_jaccard`` for shared counts over ``denom``."""
    common = np.asarray(common, dtype=np.float64)
    j = common / float(denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -np.log(2.0 * j / (1.0 + j)) / k
    d = np.where(common <= 0, 1.0, d)
    d = np.where(common >= denom, 0.0, d)
    return d


def mash_distance(a: np.ndarray, b: np.ndarray, k: int = K, s: int | None = None) -> float:
    """Mash distance between two sketches.

    ``d = -ln(2j / (1 + j)) / k`` with ``j`` from ``jaccard_counts``.
    Identical sketches give ``0.0``; sketches sharing nothing give ``1.0``.

    Args:
        a, b: Sketches built with the same ``k``.
        k: k-mer length used to build the sketches.
        s: Sketch size override, see ``jaccard_counts``.
    """
    _check_k(k)
    common, denom = jaccard_counts(a, b, s)
    if common >= denom:
        return 0.0
    if common == 0:
        return 1.0
    return distance_from_jaccard(common / denom, k)


def _common_counts(a: np.ndarray, S: np.ndarray, s_param: int) -> np.ndarray:
    """Shared-hash counts between one sketch and many, Mash semantics, vectorised.

    For each row ``b`` of ``S`` counts the elements that lie in ``a ∩ b`` and
    among the ``s_param`` smallest values of ``a ∪ b``.

    The rank of ``b[c]`` in the sorted union is
    ``(c + 1) + |{x in a : x <= b[c]}| - |{x in a ∩ b : x <= b[c]}|``; every
    shared element is an element of ``b``, so counting over ``b`` is complete.

    Args:
        a: 1-D sorted unique ``uint64`` array.
        S: 2-D ``(m, s)`` array with sorted unique rows.
        s_param: Sketch size (denominator).
    """
    m, s = S.shape
    left = np.searchsorted(a, S, side="left")  # (m, s): # of a strictly below value
    in_a = a[np.minimum(left, a.size - 1)] == S  # value present in a
    n_a_le = left + in_a  # # of a <= value (a is unique)
    cum_shared = np.cumsum(in_a, axis=1)  # # of shared <= value
    union_rank = np.arange(1, s + 1, dtype=np.int64)[None, :] + n_a_le - cum_shared
    return np.count_nonzero(in_a & (union_rank <= s_param), axis=1)


def distances_to(query: np.ndarray, sketches: np.ndarray, k: int = K) -> np.ndarray:
    """Mash distance from one sketch to every row of a sketch matrix.

    Vectorised equivalent of ``[mash_distance(query, row, k) for row in sketches]``
    (used for nearest-training-genome lookups).

    Args:
        query: 1-D sorted unique sketch.
        sketches: 2-D ``(n, s)`` array with sorted unique rows.
        k: k-mer length.

    Returns:
        ``float64`` array of shape ``(n,)``.
    """
    _check_k(k)
    a = _as_sorted_sketch(query, "query")
    S = _as_sketch_matrix(sketches)
    s_param = max(a.size, S.shape[1])
    return _distances_from_common(_common_counts(a, S, s_param), s_param, k)


def _dense_ranks(S: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace every hash by its dense, order-preserving global rank (``int32``).

    Sorted rows stay sorted, so "is x in a" and "how many of a are <= x" become
    table lookups rather than binary searches. Returns ``(ranks, n_unique)``.
    """
    uniq, inverse = np.unique(S, return_inverse=True)
    if uniq.size >= np.iinfo(np.int32).max:  # pragma: no cover - 2^31 distinct hashes
        raise ValueError("too many distinct hashes for int32 ranks")
    return np.ascontiguousarray(inverse.reshape(S.shape), dtype=np.int32), int(uniq.size)


def _row_distance_blocks(
    ranks: np.ndarray, n_unique: int, rows: Iterable[int], k: int, block_rows: int
) -> Iterator[tuple[int, int, int, np.ndarray]]:
    """Distances from each row ``i`` in ``rows`` to every later row, block by block.

    Yields ``(i, start, stop, d)`` with ``d[j - start]`` = distance(i, j) for
    ``start <= j < stop``. Same merged-sketch estimate as ``mash_distance`` (see
    ``_common_counts`` for the rank identity), vectorised over pairs; ``ranks``
    comes from :func:`_dense_ranks` (may be a read-only memory map).
    """
    n, s = ranks.shape
    present = np.zeros(n_unique, dtype=bool)  # membership table for the current row
    cols = np.arange(1, s + 1, dtype=np.int32)  # (c + 1) term of the union rank
    for i in rows:
        a = np.asarray(ranks[i])
        present[a] = True
        n_a_le = np.cumsum(present, dtype=np.int32)  # n_a_le[r] = |{x in a : rank(x) <= r}|
        for start in range(i + 1, n, block_rows):
            stop = min(start + block_rows, n)
            rest = np.asarray(ranks[start:stop])  # (m, s)
            in_a = present[rest]
            union_rank = n_a_le[rest]
            np.add(union_rank, cols, out=union_rank)
            np.subtract(union_rank, np.cumsum(in_a, axis=1, dtype=np.int32), out=union_rank)
            hit = union_rank <= s
            np.logical_and(hit, in_a, out=hit)
            yield i, start, stop, _distances_from_common(np.count_nonzero(hit, axis=1), s, k)
        present[a] = False


def pairwise_distances(sketches: np.ndarray, k: int = K, block_rows: int = 512) -> np.ndarray:
    """Symmetric matrix of Mash distances between all pairs of sketches.

    Dense ``(n, n)`` output: use it for small collections only (tests,
    nearest-neighbour lookups on a few thousand genomes). For lineage
    clustering at scale use :func:`iter_distance_edges`, which computes the
    same values block-wise and keeps only pairs within a cut.

    Same merged-sketch estimate as ``mash_distance`` (see ``_common_counts``
    for the rank identity), vectorised over pairs on dense global hash ranks.
    Each row is compared with all later rows in blocks of ``block_rows`` (a
    memory knob) and the result is mirrored. Measured: n = 2000, s = 1000
    heavily overlapping sketches in about 10 s on a laptop.

    Args:
        sketches: 2-D ``(n, s)`` ``uint64`` array with sorted unique rows
            (as returned by ``sketch`` / ``stack_sketches`` / ``load_sketches``).
        k: k-mer length used to build the sketches.
        block_rows: Rows compared per inner step (memory only; no effect on values).

    Returns:
        ``float64`` array of shape ``(n, n)`` with zeros on the diagonal.
    """
    _check_k(k)
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    S = _as_sketch_matrix(sketches)
    n, s = S.shape
    D = np.zeros((n, n), dtype=np.float64)
    if n < 2:
        return D
    ranks, n_unique = _dense_ranks(S)
    for i, start, stop, d in _row_distance_blocks(ranks, n_unique, range(n - 1), k, block_rows):
        D[i, start:stop] = d
        D[start:stop, i] = d
    logger.debug("pairwise_distances: %d sketches, s=%d, k=%d", n, s, k)
    return D


class DistanceEdges(NamedTuple):
    """Sparse list of sketch pairs ``i < j`` (row positions) with their Mash distance."""

    rows: np.ndarray
    """``int64`` row index ``i``."""
    cols: np.ndarray
    """``int64`` row index ``j`` (``j > i``)."""
    distances: np.ndarray
    """``float64`` Mash distance of each pair."""

    @property
    def size(self) -> int:
        return int(self.rows.size)

    @classmethod
    def empty(cls) -> "DistanceEdges":
        return cls(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))

    @classmethod
    def concat(cls, parts: Sequence["DistanceEdges"]) -> "DistanceEdges":
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.rows for p in parts]).astype(np.int64, copy=False),
            np.concatenate([p.cols for p in parts]).astype(np.int64, copy=False),
            np.concatenate([p.distances for p in parts]).astype(np.float64, copy=False),
        )


# Per-process state for distance-edge workers (set by _init_edge_worker).
_EDGE_STATE: dict[str, Any] = {}


def _init_edge_worker(spec: Mapping[str, Any], n_unique: int, k: int, block_rows: int, max_distance: float) -> None:
    _EDGE_STATE.clear()
    _EDGE_STATE.update(parallel.load_shared(spec))
    _EDGE_STATE.update(n_unique=int(n_unique), k=int(k), block_rows=int(block_rows), max_distance=float(max_distance))


def _edge_task(row_range: tuple[int, int]) -> DistanceEdges:
    """Worker task: every pair ``(i, j > i)`` with ``i`` in ``row_range`` and ``d <= max_distance``."""
    state = _EDGE_STATE
    ranks = state["ranks"]
    cut = state["max_distance"]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    dists: list[np.ndarray] = []
    blocks = _row_distance_blocks(ranks, state["n_unique"], range(*row_range), state["k"], state["block_rows"])
    for i, start, _stop, d in blocks:
        near = np.flatnonzero(d <= cut)
        if near.size:
            rows.append(np.full(near.size, i, dtype=np.int64))
            cols.append(near.astype(np.int64) + start)
            dists.append(d[near])
    if not rows:
        return DistanceEdges.empty()
    return DistanceEdges(np.concatenate(rows), np.concatenate(cols), np.concatenate(dists))


def iter_distance_edges(
    sketches: np.ndarray,
    k: int = K,
    max_distance: float = 1.0,
    *,
    block_rows: int = 512,
    rows_per_task: int = ROWS_PER_TASK,
    threads: int | None = None,
) -> Iterator[DistanceEdges]:
    """All pairs ``i < j`` with Mash distance ``<= max_distance``, in blocks of query rows.

    Computes exactly the values of :func:`pairwise_distances` (same kernel) but
    never holds more than one row block: memory is the ``(n, s)`` rank table
    plus the edges kept. Blocks of ``rows_per_task`` query rows run in
    ``threads`` worker processes (``None`` = all cores; small inputs stay
    in-process, see :data:`PARALLEL_MIN_WORK`). Blocks are yielded in row
    order, edges within a block ordered by ``(i, j)``, so the output does not
    depend on ``threads``.

    Args:
        sketches: ``(n, s)`` sorted-row sketch matrix.
        k: k-mer length of the sketches.
        max_distance: Inclusive cut; pairs above it are discarded.
        block_rows: Inner comparison block (memory knob, no effect on values).
        rows_per_task: Query rows per worker task (load balancing).
        threads: Worker processes.
    """
    _check_k(k)
    if block_rows < 1 or rows_per_task < 1:
        raise ValueError("block_rows and rows_per_task must be positive")
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError(f"max_distance must be a finite non-negative number; got {max_distance!r}")
    S = _as_sketch_matrix(sketches)
    n, s = S.shape
    if n < 2:
        return
    ranks, n_unique = _dense_ranks(S)
    tasks = [(start, min(start + rows_per_task, n - 1)) for start in range(0, n - 1, rows_per_task)]
    work = n * (n - 1) / 2.0 * s
    workers = parallel.worker_count(threads, len(tasks), min_items=2) if work >= PARALLEL_MIN_WORK else 1
    logger.info(
        "distance edges: %d sketches (s=%d, k=%d), cut d <= %g, %d task(s) on %d process(es)",
        n, s, k, max_distance, len(tasks), workers,
    )
    try:
        with parallel.share_arrays({"ranks": ranks}, enabled=workers > 1) as spec:
            yield from parallel.ordered_map(
                _edge_task,
                tasks,
                workers=workers,
                initializer=_init_edge_worker,
                initargs=(spec, n_unique, k, block_rows, max_distance),
            )
    finally:
        _EDGE_STATE.clear()


def distance_edges(
    sketches: np.ndarray,
    k: int = K,
    max_distance: float = 1.0,
    *,
    block_rows: int = 512,
    threads: int | None = None,
) -> DistanceEdges:
    """All pairs within ``max_distance`` as one :class:`DistanceEdges` (see :func:`iter_distance_edges`)."""
    return DistanceEdges.concat(
        list(iter_distance_edges(sketches, k, max_distance, block_rows=block_rows, threads=threads))
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def stack_sketches(sketches: Sequence[np.ndarray]) -> np.ndarray:
    """Stack equal-length 1-D sketches into a ``(n, s)`` ``uint64`` matrix.

    Raises:
        ValueError: if the list is empty or the sketches differ in length
            (a genome with fewer than ``s`` distinct k-mers cannot be stacked;
            such genomes should already have failed QC).
    """
    rows = [np.asarray(x, dtype=np.uint64).ravel() for x in sketches]
    if not rows:
        raise ValueError("no sketches to stack")
    sizes = sorted({int(r.size) for r in rows})
    if len(sizes) != 1:
        raise ValueError(f"all sketches must have the same length to be stacked; got sizes {sizes}")
    return np.vstack(rows)


def save_sketches(
    path: str | Path,
    genome_ids: Sequence[str],
    sketches: np.ndarray | Sequence[np.ndarray],
    k: int = K,
) -> None:
    """Write sketches to an ``.npz`` with arrays ``ids``, ``sketches``, ``k``, ``s``.

    Args:
        path: Output file (parent directories are created). Written through an
            open file handle so numpy never appends a ``.npz`` suffix.
        genome_ids: One id per row; must be unique (``genome_id`` is a primary key).
        sketches: ``(n, s)`` array or a list of equal-length 1-D sketches.
        k: k-mer length the sketches were built with.
    """
    _check_k(k)
    ids = [str(g) for g in genome_ids]
    if isinstance(sketches, np.ndarray) and sketches.ndim == 2:
        S = _as_sketch_matrix(sketches)
    else:
        S = _as_sketch_matrix(stack_sketches(list(sketches)))
    if len(ids) != S.shape[0]:
        raise ValueError(f"{len(ids)} genome_ids for {S.shape[0]} sketches")
    if len(set(ids)) != len(ids):
        raise ValueError("genome_ids must be unique")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        np.savez(
            fh,
            ids=np.asarray(ids, dtype=np.str_),
            sketches=S,
            k=np.int64(k),
            s=np.int64(S.shape[1]),
        )
    logger.info("wrote %d sketches (k=%d, s=%d) to %s", S.shape[0], k, S.shape[1], out)


def load_sketches(path: str | Path) -> tuple[list[str], np.ndarray, int]:
    """Read an ``.npz`` written by ``save_sketches``.

    Returns:
        ``(genome_ids, sketches, k)`` where ``sketches`` is ``(n, s)`` ``uint64``.
        The sketch size is ``sketches.shape[1]``.

    Raises:
        ValueError: if the file is malformed (shape/id mismatch, bad ``s``).
    """
    src = Path(path)
    with np.load(src, allow_pickle=False) as z:
        missing = {"ids", "sketches", "k", "s"} - set(z.files)
        if missing:
            raise ValueError(f"{src} is not a sketch file; missing arrays {sorted(missing)}")
        ids = [str(x) for x in z["ids"]]
        S = np.ascontiguousarray(z["sketches"], dtype=np.uint64)
        k = int(z["k"])
        s = int(z["s"])
    if S.ndim != 2:
        raise ValueError(f"{src}: sketches must be 2-D; got shape {S.shape}")
    if S.shape[0] != len(ids):
        raise ValueError(f"{src}: {len(ids)} ids but {S.shape[0]} sketch rows")
    if S.shape[1] != s:
        raise ValueError(f"{src}: stored s={s} but sketches have {S.shape[1]} columns")
    _check_k(k)
    logger.debug("loaded %d sketches (k=%d, s=%d) from %s", len(ids), k, s, src)
    return ids, S, k
