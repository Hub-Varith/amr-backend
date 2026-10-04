"""Stage 6 -- lineage clusters (``data/processed/lineages.parquet``).

Why lineages matter
-------------------
Thousands of k-mers act as a fingerprint of a bacterial lineage. If genomes from
one lineage sit in both train and test, a model can score well by recognising the
lineage rather than the resistance mechanism (the "lineage trap" in ``CLAUDE.md``).
Splitting by lineage cluster is the defence; this module produces the clusters.

Method (default backend ``mash_single_linkage``)
------------------------------------------------
1. Every QC-passing genome is sketched with :mod:`genome2mic.sketch` (bottom-*s*
   MinHash with a Mash-compatible distance), in worker processes
   (``threads``; results in genome order, so output does not depend on it).
   Sketches are saved per species to ``processed/sketches_<SPECIES>.npz`` so
   later stages (nearest-training-genome distance, QC species ID) can reuse them
   instead of re-reading FASTAs.
2. All-pairs Mash distances per species, computed block-wise
   (:func:`genome2mic.sketch.iter_distance_edges`) and keeping only the pairs with
   ``d <= max(threshold, outbreak_distance)``. No ``n x n`` matrix is ever built:
   memory is the ``(n, s)`` sketch table plus the kept edges, so tens of
   thousands of genomes per species fit on one VM.
3. Single-linkage clusters cut at ``distance <= threshold`` (default
   :data:`DEFAULT_THRESHOLD` = 0.005) = connected components of the graph of
   pairs within the threshold (:func:`scipy.sparse.csgraph.connected_components`).
   Under single linkage two genomes share a cluster whenever a *chain* of genomes
   connects them with every link at or below the threshold -- exactly the
   behaviour wanted for outbreak chains and re-submissions of the same strain.
   The edge stream is folded into a spanning forest as it arrives, so even a
   large clonal group (all pairs within the threshold) never holds all its
   edges at once. :func:`single_linkage_clusters` (scipy hierarchical linkage on a
   dense matrix) is kept as the small-``n`` reference implementation; both give
   identical cluster numbers.

Why 0.005
---------
Mash distance approximates ``1 - ANI`` (average nucleotide identity), so
``d <= 0.005`` means roughly ``>= 99.5 %`` ANI. Isolates of one MLST sequence
type (ECOLI ST131, KPNEU ST258, ...) typically sit well inside that radius,
while different sequence types of the same species are usually ``d ~ 0.01-0.03``
apart. The threshold is therefore tight enough to keep a sequence type together
(the contract's requirement that known lineages stay intact) and comfortably
above the 1e-4 outbreak radius enforced by the spot-check. It is exposed as a
parameter so it can be re-tuned on real data; lowering it below
:data:`OUTBREAK_DISTANCE` will trip the spot-check by design.

Contract (``DATA_CONTRACT.md`` stage 6)
---------------------------------------
Columns ``genome_id, species, lineage_cluster, st, cluster_method``. Cluster ids
are species-prefixed and zero-padded to three digits (``KPNEU_ML_001``, largest
cluster first). ``st`` is the MLST sequence type read from
``data/interim/<gid>/mlst.tsv`` or the literal string ``"NA"`` when unknown --
the contract names this sentinel explicitly for this one column. All pairs of
genomes within Mash distance :data:`OUTBREAK_DISTANCE` must share a cluster; a
violation raises :class:`~genome2mic.errors.ContractViolation`.

PopPUNK is the contract's preferred method but is not installed here; the
``poppunk`` backend is a stub that raises :class:`~genome2mic.errors.ToolNotAvailable`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

from genome2mic import parallel
from genome2mic import sketch as sk
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation, ToolNotAvailable
from genome2mic.io import read_fasta, read_parquet, write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE",
    "DEFAULT_THRESHOLD",
    "OUTBREAK_DISTANCE",
    "CLUSTER_METHOD_MASH",
    "CLUSTER_METHOD_POPPUNK",
    "BACKENDS",
    "ST_MISSING",
    "MLST_FILE",
    "COLUMNS",
    "read_st",
    "single_linkage_clusters",
    "threshold_clusters",
    "cluster_label",
    "check_outbreak_pairs",
    "check_outbreak_edges",
    "sketch_genomes",
    "cluster_species",
    "build_lineage_frame",
    "cluster_poppunk",
    "qc_passing_genomes",
    "run",
]

STAGE = "lineages"
"""Stage name used for the drop log and ``ContractViolation`` messages."""

DEFAULT_THRESHOLD: float = 0.005
"""Single-linkage cut: Mash distance ``<= 0.005`` (about 99.5 % ANI). See module docs."""

OUTBREAK_DISTANCE: float = 1e-4
"""Contract spot-check radius: genomes this close must share a cluster."""

CLUSTER_METHOD_MASH = "mash_single_linkage"
CLUSTER_METHOD_POPPUNK = "poppunk"
BACKENDS: tuple[str, ...] = (CLUSTER_METHOD_MASH, CLUSTER_METHOD_POPPUNK)

ST_MISSING = "NA"
"""Contract value for an unknown MLST sequence type (the one agreed sentinel)."""

MLST_FILE = "mlst.tsv"
COLUMNS: tuple[str, ...] = ("genome_id", "species", "lineage_cluster", "st", "cluster_method")

_ST_NULL_TOKENS: frozenset[str] = frozenset({"", "-", "na", "n/a", "none", "null", "nan", "?"})
_CLUSTER_TAG = "ML"
_ID_WIDTH = 3


# --------------------------------------------------------------------------- #
# MLST
# --------------------------------------------------------------------------- #
def read_st(path: Path) -> str:
    """Return the MLST sequence type from an ``mlst.tsv`` file, or ``"NA"``.

    Accepts both layouts produced by the ``mlst`` tool and the synthetic
    generator:

    * a header line containing a column named ``ST`` (case-insensitive) followed
      by one data line -- the ``ST`` column of that data line is used;
    * the tool's headerless default ``<file>\\t<scheme>\\t<ST>\\t<alleles...>`` --
      the third tab-separated field is used.

    A missing or empty file, a row with fewer than three fields, or an ST token
    the tool uses for "unknown" (``-``, ``NA``, empty, ``?``) all give
    :data:`ST_MISSING`. Anything else is returned stripped but otherwise as
    reported (``"258"``, ``"258~"``).
    """
    target = Path(path)
    if not target.is_file():
        return ST_MISSING
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - unusual filesystem failure
        logger.warning("could not read %s (%s); st set to %s", target, exc, ST_MISSING)
        return ST_MISSING
    rows = [line.split("\t") for line in text.splitlines() if line.strip()]
    if not rows:
        return ST_MISSING
    header_upper = [cell.strip().upper() for cell in rows[0]]
    if "ST" in header_upper:
        column = header_upper.index("ST")
        if len(rows) < 2:
            return ST_MISSING
        fields = rows[1]
    else:
        column = 2
        fields = rows[0]
    if len(fields) <= column:
        return ST_MISSING
    return _normalize_st(fields[column])


def _normalize_st(raw: str) -> str:
    value = raw.strip()
    if value.lower() in _ST_NULL_TOKENS:
        return ST_MISSING
    return value


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def single_linkage_clusters(distances: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Single-linkage clusters of a square distance matrix, cut at ``<= threshold``.

    Two genomes share a cluster when a chain of genomes connects them with every
    link ``<= threshold`` (equivalently: connected components of the graph of
    pairs within the threshold). Implemented with
    :func:`scipy.cluster.hierarchy.linkage` (``method="single"``) and
    :func:`scipy.cluster.hierarchy.fcluster` (``criterion="distance"``).

    Dense reference implementation for small ``n`` (``O(n^2)`` memory). The stage
    itself uses :func:`threshold_clusters` on sparse edges, which returns the
    identical numbering.

    Args:
        distances: ``(n, n)`` symmetric matrix, zeros on the diagonal, no NaN.
        threshold: Inclusive distance cut, ``>= 0``.

    Returns:
        ``int64`` array of cluster numbers ``1..C`` per row. Numbering is
        deterministic: clusters are ordered by size (largest first) and, for
        ties, by the smallest member index.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a finite non-negative number; got {threshold!r}")
    D = np.asarray(distances, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distances must be a square matrix; got shape {D.shape}")
    n = D.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.ones(1, dtype=np.int64)
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("distances must be finite and non-negative")
    if not np.allclose(D, D.T, rtol=0.0, atol=1e-12):
        raise ValueError("distances must be symmetric")
    if np.any(np.diag(D) != 0.0):
        raise ValueError("distances must have zeros on the diagonal")

    condensed = squareform(D, checks=False)
    tree = linkage(condensed, method="single")
    raw = fcluster(tree, t=threshold, criterion="distance")
    return _relabel_by_size(np.asarray(raw))


def _relabel_by_size(raw: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster labels to ``1..C`` ordered by size desc, then first index."""
    labels = np.asarray(raw)
    uniq, first_index, inverse, counts = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
    order = np.lexsort((first_index, -counts))  # primary: size desc; ties: first member index
    rank = np.empty(uniq.size, dtype=np.int64)
    rank[order] = np.arange(1, uniq.size + 1, dtype=np.int64)
    return rank[np.asarray(inverse).ravel()]


def _components(n: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Connected-component label per node of the undirected graph ``rows <-> cols``."""
    if n == 0:
        return np.empty(0, dtype=np.int64)
    graph = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n, n),
    ).tocsr()
    _n_comp, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64)


def _spanning_forest(n: int, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """At most ``n - 1`` edges with the same connected components as ``rows <-> cols``.

    Every node is linked to the first (smallest-index) node of its component (a
    star per component), so connectivity is preserved exactly.
    """
    labels = _components(n, rows, cols)
    _uniq, first = np.unique(labels, return_index=True)
    root = first[np.searchsorted(_uniq, labels)]
    nodes = np.arange(n, dtype=np.int64)
    keep = nodes != root
    return nodes[keep], root[keep].astype(np.int64)


def threshold_clusters(
    n: int,
    edge_blocks: Iterable[sk.DistanceEdges],
    threshold: float = DEFAULT_THRESHOLD,
    *,
    outbreak_distance: float | None = None,
    compact_every: int = 20_000_000,
) -> tuple[np.ndarray, sk.DistanceEdges]:
    """Single-linkage clusters (cut ``<= threshold``) from a stream of sparse distance edges.

    Single linkage cut at ``t`` = connected components of the graph whose edges are
    the pairs with ``d <= t``; this is exactly what :func:`single_linkage_clusters`
    computes from a dense matrix, with the same ``1..C`` numbering (size
    descending, ties by smallest member index). Pairs absent from the stream are
    taken to be farther apart than the threshold, so the stream must contain
    every pair with ``d <= threshold`` (as :func:`genome2mic.sketch.iter_distance_edges`
    with ``max_distance >= threshold`` does).

    Edges are folded into a spanning forest whenever more than ``compact_every``
    accumulate, bounding memory by ``O(n + compact_every)`` regardless of how many
    pairs fall within the threshold.

    Args:
        n: Number of genomes (nodes).
        edge_blocks: :class:`~genome2mic.sketch.DistanceEdges` blocks.
        threshold: Inclusive single-linkage cut.
        outbreak_distance: When given, the pairs with ``d <= outbreak_distance`` are
            collected and returned for :func:`check_outbreak_edges`.
        compact_every: Edge count that triggers a forest compaction (memory knob).

    Returns:
        ``(cluster_numbers, close_pairs)``: ``int64`` numbers ``1..C`` per node and
        the outbreak-radius pairs (empty when ``outbreak_distance`` is ``None``).
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a finite non-negative number; got {threshold!r}")
    if n < 0:
        raise ValueError("n must be non-negative")
    forest_r = np.empty(0, dtype=np.int64)
    forest_c = np.empty(0, dtype=np.int64)
    pending_r: list[np.ndarray] = []
    pending_c: list[np.ndarray] = []
    n_pending = n_links = 0
    close: list[sk.DistanceEdges] = []
    for block in edge_blocks:
        if block.size == 0:
            continue
        if block.rows.max() >= n or block.cols.max() >= n or min(block.rows.min(), block.cols.min()) < 0:
            raise ValueError("edge index out of range for n nodes")
        link = block.distances <= threshold
        n_link = int(np.count_nonzero(link))
        if n_link:
            pending_r.append(block.rows[link])
            pending_c.append(block.cols[link])
            n_pending += n_link
            n_links += n_link
        if outbreak_distance is not None:
            near = block.distances <= outbreak_distance
            if near.any():
                close.append(sk.DistanceEdges(block.rows[near], block.cols[near], block.distances[near]))
        if n_pending > compact_every:
            forest_r, forest_c = _spanning_forest(
                n, np.concatenate([forest_r, *pending_r]), np.concatenate([forest_c, *pending_c])
            )
            pending_r, pending_c, n_pending = [], [], 0
    if n == 0:
        return np.empty(0, dtype=np.int64), sk.DistanceEdges.concat(close)
    labels = _components(n, np.concatenate([forest_r, *pending_r]), np.concatenate([forest_c, *pending_c]))
    logger.debug("threshold_clusters: %d node(s), %d link(s) <= %g", n, n_links, threshold)
    return _relabel_by_size(labels), sk.DistanceEdges.concat(close)


def cluster_label(species: str, number: int, method_tag: str = _CLUSTER_TAG) -> str:
    """Species-prefixed cluster id: ``cluster_label("KPNEU", 7) -> "KPNEU_ML_007"``.

    Numbers above 999 simply use more digits (``KPNEU_ML_1000``).
    """
    key = str(species).strip().upper()
    if not key:
        raise ValueError("species must be a non-empty string")
    if int(number) < 1:
        raise ValueError(f"cluster numbers start at 1; got {number!r}")
    return f"{key}_{method_tag}_{int(number):0{_ID_WIDTH}d}"


def check_outbreak_pairs(
    distances: np.ndarray,
    cluster_numbers: np.ndarray,
    genome_ids: Sequence[str],
    max_distance: float = OUTBREAK_DISTANCE,
    species: str | None = None,
) -> int:
    """Contract spot-check on a dense matrix: every pair with ``d <= max_distance`` shares a cluster.

    Small-``n`` convenience wrapper around :func:`check_outbreak_edges` (the stage
    itself never builds a dense matrix).

    Args:
        distances: ``(n, n)`` distance matrix.
        cluster_numbers: Cluster per row, as from :func:`single_linkage_clusters`.
        genome_ids: Row labels for the error message.
        max_distance: Spot-check radius (contract: 1e-4).
        species: For log and error messages only.

    Returns:
        Number of pairs that were within ``max_distance`` (all of them passed).

    Raises:
        ContractViolation: if any near-identical pair is split across clusters.
    """
    D = np.asarray(distances, dtype=np.float64)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[1] != n:
        raise ValueError(f"distances must be a square matrix; got shape {D.shape}")
    rows, cols = np.nonzero(np.triu(D <= max_distance, k=1))
    edges = sk.DistanceEdges(rows.astype(np.int64), cols.astype(np.int64), D[rows, cols])
    return check_outbreak_edges(edges, cluster_numbers, genome_ids, max_distance=max_distance, species=species)


def check_outbreak_edges(
    edges: sk.DistanceEdges,
    cluster_numbers: np.ndarray,
    genome_ids: Sequence[str],
    max_distance: float = OUTBREAK_DISTANCE,
    species: str | None = None,
) -> int:
    """Contract spot-check on sparse pairs: every pair with ``d <= max_distance`` shares a cluster.

    Args:
        edges: Pairs ``(i, j, d)``; must include every pair with ``d <= max_distance``
            (pairs farther apart are ignored).
        cluster_numbers: Cluster per genome.
        genome_ids: Genome labels for the error message.
        max_distance: Spot-check radius (contract: 1e-4).
        species: For log and error messages only.

    Returns:
        Number of pairs within ``max_distance`` (all of them passed).

    Raises:
        ContractViolation: if any near-identical pair is split across clusters.
    """
    labels = np.asarray(cluster_numbers)
    ids = list(genome_ids)
    n = labels.shape[0]
    if len(ids) != n:
        raise ValueError("cluster_numbers and genome_ids must have the same length")
    tag = f"{species}: " if species else ""
    if n < 2:
        logger.info("[%s] %soutbreak spot-check: fewer than two genomes, nothing to check", STAGE, tag)
        return 0
    rows = np.asarray(edges.rows, dtype=np.int64)
    cols = np.asarray(edges.cols, dtype=np.int64)
    dist = np.asarray(edges.distances, dtype=np.float64)
    if rows.size and (max(rows.max(), cols.max()) >= n or min(rows.min(), cols.min()) < 0):
        raise ValueError("edge index out of range for cluster_numbers")
    close = dist <= max_distance
    n_pairs = int(np.count_nonzero(close))
    split = close & (labels[rows] != labels[cols])
    if np.any(split):
        bad = [(ids[i], ids[j], float(d)) for i, j, d in zip(rows[split], cols[split], dist[split])]
        examples = "; ".join(f"{a} vs {b} (d={d:.2e})" for a, b, d in bad[:5])
        raise ContractViolation(
            f"{tag}{len(bad)} near-identical pair(s) within Mash distance {max_distance:g} "
            f"fall in different lineage clusters: {examples}",
            stage=STAGE,
        )
    logger.info(
        "[%s] %soutbreak spot-check passed: %d pair(s) within d <= %g all share a cluster",
        STAGE,
        tag,
        n_pairs,
        max_distance,
    )
    return n_pairs


# --------------------------------------------------------------------------- #
# Sketching
# --------------------------------------------------------------------------- #
class _WindowCounter:
    """Collects the non-ACGT window counts ``sketch`` reports so we log one total."""

    def __init__(self) -> None:
        self.n = 0

    def drop(self, reason: str, n: int, detail: str | None = None) -> None:  # noqa: D401 - protocol
        self.n += int(n)


def _sketch_fasta(task: tuple[str, int, int]) -> tuple[np.ndarray | None, int]:
    """Worker task: ``(sketch, non_acgt_windows)`` of one FASTA, or ``(None, 0)`` if it is missing."""
    path, k, sketch_size = task
    fasta = Path(path)
    if not fasta.is_file():
        return None, 0
    windows = _WindowCounter()
    seqs = [seq for _, seq in read_fasta(fasta)]
    return sk.sketch(seqs, k=k, s=sketch_size, droplog=windows), windows.n


def sketch_genomes(
    paths: Paths,
    genome_ids: Sequence[str],
    *,
    k: int = sk.K,
    sketch_size: int = sk.SKETCH_SIZE,
    droplog: DropLog | None = None,
    species: str | None = None,
    threads: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """Sketch each genome's FASTA (``paths.genome_fasta``) with :func:`genome2mic.sketch.sketch`.

    Genomes whose FASTA is missing, or which have fewer than ``sketch_size``
    distinct k-mers (their sketch cannot be stacked and their distances would
    be noise), are dropped and counted in ``droplog`` under ``fasta_missing`` /
    ``sketch_too_small``. The number of k-mer windows skipped for non-ACGT bases
    is recorded once under ``non_acgt_window`` (a window count, not a row count).

    Genomes are sketched in ``threads`` worker processes (``None`` = all cores;
    small inputs stay in-process). Results come back in ``genome_ids`` order, so
    the output is identical for any ``threads``.

    Returns:
        ``(kept_ids, sketches)`` with ``sketches`` of shape ``(len(kept_ids), sketch_size)``.
    """
    kept_ids: list[str] = []
    kept: list[np.ndarray] = []
    missing: list[str] = []
    too_small: list[str] = []
    windows = _WindowCounter()
    ids = [str(g) for g in genome_ids]
    tasks = [(str(paths.genome_fasta(gid)), int(k), int(sketch_size)) for gid in ids]
    workers = parallel.worker_count(threads, len(tasks))
    results = parallel.ordered_map(_sketch_fasta, tasks, workers=workers, chunksize=8 if workers > 1 else 1)
    for i, (gid, (arr, n_skipped)) in enumerate(zip(ids, results, strict=True), start=1):
        if arr is None:
            missing.append(gid)
            continue
        windows.drop("non_acgt_window", n_skipped)
        if arr.size < sketch_size:
            too_small.append(gid)
        else:
            kept_ids.append(gid)
            kept.append(arr)
        if i % 5000 == 0:
            logger.info("[%s] %ssketched %d/%d genome(s)", STAGE, f"{species}: " if species else "", i, len(ids))

    tag = f"{species}: " if species else ""
    if droplog is not None:
        droplog.drop("fasta_missing", len(missing), detail=f"{tag}{_examples(missing)}" if missing else tag or None)
        droplog.drop(
            "sketch_too_small",
            len(too_small),
            detail=f"{tag}fewer than s={sketch_size} distinct {k}-mers; {_examples(too_small)}"
            if too_small
            else f"{tag}fewer than s={sketch_size} distinct {k}-mers",
        )
        droplog.drop(
            "non_acgt_window",
            windows.n,
            detail=f"{tag}k-mer windows skipped for non-ACGT bases while sketching (window count, not genomes)",
        )
    logger.info(
        "[%s] %ssketched %d genome(s) (k=%d, s=%d); %d FASTA missing, %d too small",
        STAGE,
        tag,
        len(kept_ids),
        k,
        sketch_size,
        len(missing),
        len(too_small),
    )
    matrix = sk.stack_sketches(kept) if kept else np.empty((0, sketch_size), dtype=np.uint64)
    return kept_ids, matrix


def _examples(items: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(str(x) for x in list(items)[:limit])
    if len(items) > limit:
        shown += f", ... (+{len(items) - limit})"
    return shown


# --------------------------------------------------------------------------- #
# Per-species clustering
# --------------------------------------------------------------------------- #
def cluster_species(
    species: str,
    genome_ids: Sequence[str],
    sketches: np.ndarray,
    *,
    k: int = sk.K,
    threshold: float = DEFAULT_THRESHOLD,
    outbreak_distance: float = OUTBREAK_DISTANCE,
    threads: int | None = None,
) -> tuple[np.ndarray, sk.DistanceEdges]:
    """Sparse all-pairs Mash distances, single-linkage clusters and the outbreak spot-check.

    Distances are computed block-wise (:func:`genome2mic.sketch.iter_distance_edges`,
    ``threads`` worker processes) keeping only pairs with
    ``d <= max(threshold, outbreak_distance)``; clusters are the connected
    components of the ``d <= threshold`` graph (:func:`threshold_clusters`),
    identical to :func:`single_linkage_clusters` on the dense matrix. No ``n x n``
    array is allocated. Logs one line with the number of clusters and the
    largest cluster.

    Returns:
        ``(cluster_numbers, close_pairs)`` -- ``int64`` numbers ``1..C`` aligned
        with ``genome_ids`` and the pairs within ``outbreak_distance`` (the
        spot-checked pairs; positions index ``genome_ids``).
    """
    ids = list(genome_ids)
    S = np.asarray(sketches, dtype=np.uint64)
    if S.ndim != 2 or S.shape[0] != len(ids):
        raise ValueError(f"sketches must be (n, s) with n == len(genome_ids); got {S.shape} for {len(ids)} ids")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a finite non-negative number; got {threshold!r}")
    if not ids:
        return np.empty(0, dtype=np.int64), sk.DistanceEdges.empty()

    cut = max(float(threshold), float(outbreak_distance))
    blocks = sk.iter_distance_edges(S, k=k, max_distance=cut, threads=threads)
    numbers, close = threshold_clusters(len(ids), blocks, threshold, outbreak_distance=outbreak_distance)
    check_outbreak_edges(close, numbers, ids, max_distance=outbreak_distance, species=species)

    sizes = np.bincount(numbers)[1:]
    largest_number = int(np.argmax(sizes)) + 1
    logger.info(
        "[%s] %s: %d genome(s) -> %d lineage cluster(s) (%s, threshold d <= %g); "
        "largest cluster %s has %d genome(s); %d singleton(s)",
        STAGE,
        species,
        len(ids),
        int(sizes.size),
        CLUSTER_METHOD_MASH,
        threshold,
        cluster_label(species, largest_number),
        int(sizes.max()),
        int(np.count_nonzero(sizes == 1)),
    )
    return numbers, close


def build_lineage_frame(
    species: str,
    genome_ids: Sequence[str],
    cluster_numbers: np.ndarray,
    sts: Sequence[str],
    cluster_method: str = CLUSTER_METHOD_MASH,
) -> pd.DataFrame:
    """Assemble the contract table for one species (all columns ``str`` dtype)."""
    ids = [str(g) for g in genome_ids]
    numbers = np.asarray(cluster_numbers)
    if len(ids) != numbers.shape[0] or len(ids) != len(sts):
        raise ValueError("genome_ids, cluster_numbers and sts must have the same length")
    key = str(species).strip().upper()
    return pd.DataFrame(
        {
            "genome_id": pd.array(ids, dtype="str"),
            "species": pd.array([key] * len(ids), dtype="str"),
            "lineage_cluster": pd.array([cluster_label(key, int(n)) for n in numbers], dtype="str"),
            "st": pd.array([str(s) if s is not None else ST_MISSING for s in sts], dtype="str"),
            "cluster_method": pd.array([cluster_method] * len(ids), dtype="str"),
        },
        columns=list(COLUMNS),
    )


def cluster_poppunk(paths: Paths, config: Any) -> pd.DataFrame:
    """PopPUNK backend -- not implemented; always raises :class:`ToolNotAvailable`.

    The contract prefers PopPUNK and lists "PopPUNK or Mash?" as an open
    question. This stub keeps the backend name reserved so a future
    implementation can slot in without changing callers; until then the
    default Mash single-linkage backend is the only one that produces output.
    """
    raise ToolNotAvailable(
        "poppunk",
        hint=(
            "The poppunk lineage backend is a stub and is not implemented yet; "
            f"use backend={CLUSTER_METHOD_MASH!r} (the default)."
        ),
    )


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def qc_passing_genomes(paths: Paths, config: Any, droplog: DropLog) -> pd.DataFrame:
    """QC-passing genomes with their species: columns ``genome_id, species``.

    Species comes from ``labels.parquet`` (the contract table); a QC-passing
    genome without a label row falls back to ``qc.parquet``'s ``mash_species``.
    Every filter is counted in ``droplog``:

    * ``qc_fail`` -- genomes with ``qc_pass != True``;
    * ``labelled_genome_without_qc_record`` -- label genomes absent from ``qc.parquet``
      (they cannot be sketched because QC never saw them);
    * ``species_unknown`` -- no species from labels or Mash;
    * ``species_not_in_config`` -- species key not in ``config.species`` (skipped
      when ``config`` is ``None``).

    Raises:
        FileNotFoundError: if ``qc.parquet`` or ``labels.parquet`` is missing.
        ContractViolation: duplicate ``genome_id`` in ``qc.parquet`` or a genome
            with more than one species in ``labels.parquet``.
    """
    if not paths.qc.is_file():
        raise FileNotFoundError(
            f"{paths.qc} not found; run the qc stage first (lineages are built from QC-passing genomes only)"
        )
    if not paths.labels.is_file():
        raise FileNotFoundError(f"{paths.labels} not found; run the ingest stage first")

    qc = read_parquet(paths.qc)
    for col in ("genome_id", "qc_pass"):
        if col not in qc.columns:
            raise ContractViolation(f"qc.parquet lacks required column {col!r}", stage=STAGE)
    if qc["genome_id"].duplicated().any():
        dups = qc.loc[qc["genome_id"].duplicated(), "genome_id"].tolist()
        raise ContractViolation(f"qc.parquet has duplicate genome_id values: {_examples(dups)}", stage=STAGE)

    labels = read_parquet(paths.labels, columns=["genome_id", "species"])
    label_species = labels.dropna(subset=["species"]).drop_duplicates(["genome_id", "species"])
    conflicting = label_species.loc[label_species["genome_id"].duplicated(), "genome_id"].tolist()
    if conflicting:
        raise ContractViolation(
            f"labels.parquet maps a genome to more than one species: {_examples(conflicting)}",
            stage=STAGE,
        )

    all_qc_ids = set(qc["genome_id"].astype(str))
    labelled_missing = sorted(set(label_species["genome_id"].astype(str)) - all_qc_ids)
    droplog.drop(
        "labelled_genome_without_qc_record",
        len(labelled_missing),
        detail=_examples(labelled_missing) if labelled_missing else None,
    )

    passing = qc["qc_pass"].astype("boolean").fillna(False).astype(bool).to_numpy()
    qc = droplog.keep_where(qc, passing, "qc_fail", detail="qc_pass is not True")

    merged = qc[["genome_id"]].merge(label_species, on="genome_id", how="left")
    species = merged["species"].astype("str")
    if "mash_species" in qc.columns:
        fallback = qc.set_index("genome_id")["mash_species"].reindex(merged["genome_id"]).to_numpy()
        n_fallback = int(np.count_nonzero(species.isna().to_numpy() & ~pd.isna(fallback)))
        species = species.where(species.notna(), pd.array(fallback, dtype="str"))
        if n_fallback:
            logger.info(
                "[%s] %d QC-passing genome(s) have no label row; species taken from qc.mash_species",
                STAGE,
                n_fallback,
            )
    merged = merged.assign(species=species.astype("str"))
    merged = droplog.keep_where(merged, merged["species"].notna().to_numpy(), "species_unknown")

    if config is not None:
        known = {str(k).upper() for k in getattr(config, "species", {})}
        in_config = merged["species"].isin(known).to_numpy()
        unknown = sorted(set(merged.loc[~in_config, "species"]))
        merged = droplog.keep_where(
            merged,
            in_config,
            "species_not_in_config",
            detail=_examples(unknown) if unknown else None,
        )

    out = merged[["genome_id", "species"]].reset_index(drop=True)
    return out.assign(genome_id=out["genome_id"].astype("str"), species=out["species"].astype("str"))


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
def run(
    paths: Paths,
    config: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    backend: str = CLUSTER_METHOD_MASH,
    k: int = sk.K,
    sketch_size: int = sk.SKETCH_SIZE,
    threads: int | None = None,
) -> pd.DataFrame:
    """Build ``lineages.parquet`` and ``sketches_<SPECIES>.npz`` for every species.

    Args:
        paths: Project paths (reads ``qc.parquet``, ``labels.parquet``, genome
            FASTAs and ``interim/<gid>/mlst.tsv``; writes ``lineages.parquet``,
            ``sketches_<SPECIES>.npz`` and ``drop_log_lineages.csv``).
        config: Loaded :class:`genome2mic.config.Config` (only ``.species`` keys are
            used, to drop genomes of species outside the project scope). ``None``
            skips that filter.
        threshold: Single-linkage cut on Mash distance (see module docs).
        backend: ``"mash_single_linkage"`` (default) or ``"poppunk"`` (stub;
            raises :class:`ToolNotAvailable`).
        k: k-mer length for sketching.
        sketch_size: Hashes per sketch.
        threads: Worker processes for sketching and distances (``None`` = all
            cores). Results do not depend on it.

    Returns:
        The lineages table, one row per QC-passing genome that could be sketched.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown lineage backend {backend!r}; expected one of {BACKENDS}")
    if backend == CLUSTER_METHOD_POPPUNK:
        return cluster_poppunk(paths, config)

    droplog = DropLog(STAGE)
    genomes = qc_passing_genomes(paths, config, droplog)
    logger.info("[%s] %d QC-passing genome(s) across %d species", STAGE, len(genomes), genomes["species"].nunique())

    frames: list[pd.DataFrame] = []
    for species, group in genomes.groupby("species", sort=True):
        species_key = str(species)
        gids = sorted(group["genome_id"].astype(str).tolist())
        ids, sketches = sketch_genomes(
            paths, gids, k=k, sketch_size=sketch_size, droplog=droplog, species=species_key, threads=threads
        )
        if not ids:
            logger.warning("[%s] %s: no genome could be sketched; species skipped", STAGE, species_key)
            continue
        sk.save_sketches(paths.sketches(species_key), ids, sketches, k=k)
        numbers, _ = cluster_species(species_key, ids, sketches, k=k, threshold=threshold, threads=threads)
        sts = [read_st(paths.interim_dir(gid) / MLST_FILE) for gid in ids]
        n_with_st = sum(1 for s in sts if s != ST_MISSING)
        logger.info("[%s] %s: MLST sequence type known for %d of %d genome(s)", STAGE, species_key, n_with_st, len(ids))
        frames.append(build_lineage_frame(species_key, ids, numbers, sts))

    if frames:
        lineages = pd.concat(frames, ignore_index=True)
    else:
        lineages = build_lineage_frame("NONE", [], np.empty(0, dtype=np.int64), [])
        logger.warning("[%s] no genomes to cluster; writing an empty lineages table", STAGE)
    _validate_frame(lineages)

    write_parquet(lineages, paths.lineages)
    droplog.write(paths.drop_log(STAGE))
    logger.info(
        "[%s] wrote %d row(s), %d cluster(s) to %s",
        STAGE,
        len(lineages),
        int(lineages["lineage_cluster"].nunique()),
        paths.lineages,
    )
    return lineages


def _validate_frame(frame: pd.DataFrame) -> None:
    """Contract checks on the assembled table (raise instead of writing bad data)."""
    if list(frame.columns) != list(COLUMNS):
        raise ContractViolation(f"lineages columns {list(frame.columns)} != {list(COLUMNS)}", stage=STAGE)
    if frame["genome_id"].duplicated().any():
        raise ContractViolation("duplicate genome_id in lineages", stage=STAGE)
    for col in COLUMNS:
        if frame[col].isna().any():
            raise ContractViolation(f"null values in lineages column {col!r}", stage=STAGE)
    prefixed = np.array(
        [str(c).startswith(f"{s}_") for c, s in zip(frame["lineage_cluster"], frame["species"])], dtype=bool
    )
    bad_prefix = frame.loc[~prefixed, "genome_id"].tolist() if len(frame) else []
    if bad_prefix:
        raise ContractViolation(
            f"lineage_cluster not species-prefixed for {_examples(bad_prefix)}", stage=STAGE
        )
