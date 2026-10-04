"""Unitig / k-mer presence-absence features (``DATA_CONTRACT.md`` stage 8).

What this stage produces, per species
-------------------------------------
``unitigs_<SPECIES>.npz``
    ``scipy.sparse`` CSR ``int8`` matrix, genomes x patterns, rows in
    ``unitigs_<SPECIES>_rows.parquet`` order. Never densified.
``unitigs_<SPECIES>_rows.parquet``
    ``row_index, genome_id, split, role``. ``role`` is ``built`` for the training
    genomes the k-mer set was built from and ``queried`` for every other
    QC-passing genome of the species (test rows). The leakage check uses it to
    verify that only ``split == 'train'`` rows were ``built``.
``unitigs_<SPECIES>_index.parquet``
    ``col_index, pattern_id, n_unitigs, unitig_sequences (list[str]),
    train_frequency``. For the k-mer backend ``unitig_sequences`` are the member
    canonical 31-mers decoded to strings.
``unitigs_<SPECIES>_kmers.npz``
    The frozen k-mer set (sorted ``uint64``) plus ``pattern_col`` per k-mer.
    :func:`query_genome` needs only this file, not the matrix, so the prediction
    pipeline can turn any new FASTA into a pattern vector.

Build procedure (order matters, contract stage 8)
-------------------------------------------------
1. Read ``splits.parquet`` and ``qc.parquet`` **only** (never ``labels.parquet``).
   Training genomes = ``split == 'train'`` and ``qc_pass`` for the species.
2. :class:`KmerBackend` (default): every canonical 31-mer of every training genome,
   2-bit encoded into ``uint64`` (A=0, C=1, G=2, T=3; k=31 fits in 62 bits), one
   genome at a time with ``np.unique`` per genome, merged into a global
   ``(kmer, count)`` table by an incremental sorted merge (memory stays bounded
   by the distinct k-mer count, not by genomes x genome length).
3. Frequency filter: keep k-mers present in ``>= min_freq`` and ``<= max_freq`` of
   the training genomes (defaults 1 % / 99 %; the contract's "drop < 1 % or
   > 99 %" is the inclusive form, which is what is implemented).
4. **Collapse identical presence/absence columns into patterns**: every kept
   k-mer column is keyed by its row-index set (two independent 64-bit splitmix64
   sums plus the column count, so a false merge needs a 128-bit collision).
   The keys are sums, so the k-mer backend accumulates them genome by genome
   (O(kept k-mers) memory) and never materialises the ``genomes x kept k-mers``
   matrix; a third pass then reads each training genome's pattern row from the
   pattern's first k-mer. Patterns are numbered in order of their first
   (smallest) member k-mer: ``u_000000, u_000001, ...``.
5. Every other QC-passing genome is **queried** against the frozen set and its
   row appended. The set is never rebuilt to include them.

Scale
-----
Per-genome work (k-mer extraction, set membership, queries) runs in ``threads``
worker processes (:mod:`genome2mic.parallel`; results in genome order, so the
output does not depend on ``threads``). The pure-Python k-mer backend still
re-encodes each training FASTA up to three times and keeps a global k-mer count
table, so it refuses more than ``max_genomes`` training genomes (default
:data:`DEFAULT_MAX_KMER_GENOMES`); use ``--unitig-backend unitig-caller`` (or
``auto`` with unitig-caller installed) on real data.

Pattern presence in a queried genome
------------------------------------
A pattern is called present when **at least 50 %** of its member k-mers are
present in the genome (``presence_fraction``). A single SNP destroys up to 31
consecutive k-mers of a unitig; the majority rule keeps the pattern call stable
for near-identical sequence while a genuinely absent block (0 % of k-mers)
stays absent. Training (``built``) rows are exact by construction: all member
k-mers of a pattern co-occur in every training genome.

Backends
--------
:class:`KmerBackend` is pure numpy and the default. :class:`UnitigCallerBackend`
shells out to ``unitig-caller`` (``--call`` to build, ``--query`` to query) when
it is on ``PATH`` and parses its ``rtab`` / ``pyseer`` output; otherwise it raises
:class:`~genome2mic.errors.ToolNotAvailable`. Both backends end in the same
:class:`KmerSet`, so :func:`query_genome` works without the external tool.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from genome2mic import parallel
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation, ToolNotAvailable
from genome2mic.io import iter_fasta, read_parquet, write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

__all__ = [
    "K",
    "MIN_FREQ",
    "MAX_FREQ",
    "PRESENCE_FRACTION",
    "DEFAULT_MAX_KMER_GENOMES",
    "INDEX_COLUMNS",
    "LOAD_INDEX_COLUMNS",
    "ROLE_BUILT",
    "ROLE_QUERIED",
    "STAGE",
    "TooManyGenomesForKmerBackend",
    "KmerSet",
    "BuildResult",
    "UnitigBuild",
    "UnitigBackend",
    "KmerBackend",
    "UnitigCallerBackend",
    "encode_kmers",
    "decode_kmer",
    "decode_kmers",
    "genome_kmers",
    "collapse_patterns",
    "pattern_id",
    "save_kmer_set",
    "load_kmer_set",
    "read_kmer_set_sha1",
    "query_kmer_set",
    "query_genome",
    "make_backend",
    "training_and_query_genomes",
    "build",
    "run",
    "load_unitigs",
    "parse_rtab",
    "parse_pyseer",
]

K: int = 31
"""Default k-mer length. 2 bits x 31 = 62 bits, so a canonical k-mer fits a ``uint64``."""

MAX_K: int = 32
"""Largest k whose 2-bit encoding fits in 64 bits."""

MIN_FREQ: float = 0.01
"""Keep k-mers present in at least this share of training genomes (contract: drop < 1 %)."""

MAX_FREQ: float = 0.99
"""Keep k-mers present in at most this share of training genomes (contract: drop > 99 %)."""

PRESENCE_FRACTION: float = 0.5
"""Share of a pattern's member k-mers that must be present to call the pattern present."""

DEFAULT_MAX_KMER_GENOMES: int = 1000
"""Training genomes above which :class:`KmerBackend` refuses to run (CLI ``--unitig-max-kmer-genomes``)."""

INDEX_COLUMNS: tuple[str, ...] = ("col_index", "pattern_id", "n_unitigs", "unitig_sequences", "train_frequency")
"""Columns of ``unitigs_<SPECIES>_index.parquet`` (contract stage 8)."""

LOAD_INDEX_COLUMNS: tuple[str, ...] = ("col_index", "pattern_id", "n_unitigs", "train_frequency")
"""What :func:`load_unitigs` reads by default: everything except the (large) member sequences."""

INDEX_BATCH_KMERS: int = 10_000_000
"""Member k-mers decoded per index row group (bounds writer memory, ~0.3 GB of letters)."""

ROLE_BUILT: str = "built"
ROLE_QUERIED: str = "queried"
STAGE: str = "unitigs"
TRAIN_SPLIT: str = "train"

CHUNK_WINDOWS: int = 1_000_000
"""Sequence windows encoded per numpy pass; bounds memory for multi-megabase contigs."""

# ---------------------------------------------------------------------------
# 2-bit encoding
# ---------------------------------------------------------------------------
_INVALID = np.uint8(4)
_ENCODE = np.full(256, _INVALID, dtype=np.uint8)
for _b, _c in zip(b"ACGT", range(4)):
    _ENCODE[_b] = _c
    _ENCODE[_b + 32] = _c  # lower case
_DECODE = np.frombuffer(b"ACGT", dtype=np.uint8)
_TWO = np.uint64(2)
_THREE = np.uint64(3)

# splitmix64 constants (Steele, Lea & Flood 2014); arithmetic wraps mod 2**64.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_SALT2 = np.uint64(0xA5A5A5A5A5A5A5A5)


def _check_k(k: int) -> None:
    if not isinstance(k, (int, np.integer)) or k < 1 or k > MAX_K:
        raise ValueError(f"k must be an integer in [1, {MAX_K}]; got {k!r}")


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on a ``uint64`` array (bijection; wraps mod 2**64)."""
    z = np.array(x, dtype=np.uint64, copy=True, ndmin=1)
    tmp = np.empty_like(z)
    with np.errstate(over="ignore"):
        np.add(z, _GOLDEN, out=z)
        np.right_shift(z, np.uint64(30), out=tmp)
        np.bitwise_xor(z, tmp, out=z)
        np.multiply(z, _MIX1, out=z)
        np.right_shift(z, np.uint64(27), out=tmp)
        np.bitwise_xor(z, tmp, out=z)
        np.multiply(z, _MIX2, out=z)
        np.right_shift(z, np.uint64(31), out=tmp)
        np.bitwise_xor(z, tmp, out=z)
    return z


def _to_codes(seq: str | bytes) -> np.ndarray:
    """Sequence -> 2-bit codes (``uint8``), 4 for any non-ACGT byte."""
    if isinstance(seq, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(bytes(seq), dtype=np.uint8)
    else:
        # errors="replace" keeps one byte per character so window positions
        # stay aligned; the replacement byte maps to the invalid code.
        raw = np.frombuffer(str(seq).encode("ascii", errors="replace"), dtype=np.uint8)
    return _ENCODE[raw]


def _encode_with_skips(seq: str | bytes, k: int) -> tuple[np.ndarray, int]:
    """Canonical codes of every valid k-window plus the number of skipped windows."""
    codes = _to_codes(seq)
    n_win = int(codes.size) - k + 1
    if n_win <= 0:
        return np.empty(0, dtype=np.uint64), 0

    invalid = codes > 3
    if invalid.any():
        prefix = np.concatenate(([0], np.cumsum(invalid, dtype=np.int64)))
        bad = (prefix[k : k + n_win] - prefix[:n_win]) > 0
        clean = np.where(invalid, np.uint8(0), codes)
    else:
        bad = None
        clean = codes

    fwd_codes = clean.astype(np.uint64)
    rev_codes = (3 - clean).astype(np.uint64)  # complement (clean <= 3)
    fwd = np.zeros(n_win, dtype=np.uint64)
    rev = np.zeros(n_win, dtype=np.uint64)
    for t in range(k):
        # forward: base t lands at bit 2*(k-1-t); reverse complement: at bit 2*t
        np.left_shift(fwd, _TWO, out=fwd)
        np.bitwise_or(fwd, fwd_codes[t : t + n_win], out=fwd)
        np.bitwise_or(rev, rev_codes[t : t + n_win] << np.uint64(2 * t), out=rev)
    canonical = np.minimum(fwd, rev)
    if bad is None:
        return canonical, 0
    return canonical[~bad], int(np.count_nonzero(bad))


def encode_kmers(seq: str | bytes, k: int = K) -> np.ndarray:
    """Canonical 2-bit codes of every k-window of ``seq``, in sequence order.

    The canonical form is ``min(code(window), code(reverse_complement(window)))``,
    so a sequence and its reverse complement give the same k-mers. Windows that
    contain any non-ACGT byte (``N``, IUPAC codes, gaps) are skipped. Duplicates
    are kept; :func:`genome_kmers` de-duplicates.

    Args:
        seq: DNA, case-insensitive.
        k: k-mer length, ``1 <= k <= 32``.

    Returns:
        ``uint64`` array with one code per valid window.
    """
    _check_k(k)
    codes, _ = _encode_with_skips(seq, k)
    return codes


def _decode_letters(codes: np.ndarray, k: int) -> np.ndarray:
    """``(n, k)`` ``uint8`` ASCII letters of 2-bit codes (no Python strings)."""
    arr = np.asarray(codes, dtype=np.uint64).ravel()
    letters = np.empty((arr.size, k), dtype=np.uint8)
    for t in range(k):
        shift = np.uint64(2 * (k - 1 - t))
        letters[:, t] = _DECODE[((arr >> shift) & _THREE).astype(np.intp)]
    return letters


def decode_kmers(codes: np.ndarray, k: int = K) -> list[str]:
    """Decode 2-bit codes back to upper-case strings of length ``k`` (vectorised)."""
    _check_k(k)
    arr = np.asarray(codes, dtype=np.uint64).ravel()
    if arr.size == 0:
        return []
    letters = _decode_letters(arr, k)
    return [s.decode("ascii") for s in letters.view(f"S{k}").ravel()]


def decode_kmer(code: int, k: int = K) -> str:
    """Decode one 2-bit code to its ``k``-letter string."""
    return decode_kmers(np.array([code], dtype=np.uint64), k)[0]


def genome_kmers(
    fasta_path: Path | str,
    k: int = K,
    chunk_windows: int = CHUNK_WINDOWS,
) -> np.ndarray:
    """Sorted, unique canonical k-mer codes of one genome assembly.

    Contigs are processed in chunks of ``chunk_windows`` windows (overlapping by
    ``k - 1`` bases) so peak memory is governed by the chunk size and the number
    of distinct k-mers, not by contig length. Non-ACGT windows are skipped and
    their count is logged at DEBUG.

    Returns:
        ``uint64`` array, sorted ascending, no duplicates (empty for a genome
        with no valid window).
    """
    _check_k(k)
    if chunk_windows < 1:
        raise ValueError("chunk_windows must be positive")
    parts: list[np.ndarray] = []
    n_contigs = n_skipped = 0
    for _header, seq in iter_fasta(Path(fasta_path)):
        n_contigs += 1
        n_win = len(seq) - k + 1
        for start in range(0, max(n_win, 0), chunk_windows):
            sub = seq[start : start + chunk_windows + k - 1]
            codes, skipped = _encode_with_skips(sub, k)
            n_skipped += skipped
            if codes.size:
                parts.append(np.unique(codes))
    if not parts:
        logger.warning("%s: no valid %d-mer window in %d contig(s)", fasta_path, k, n_contigs)
        return np.empty(0, dtype=np.uint64)
    kmers = parts[0] if len(parts) == 1 else np.unique(np.concatenate(parts))
    logger.debug(
        "%s: %d contigs, %d distinct canonical %d-mers, %d non-ACGT windows skipped",
        fasta_path, n_contigs, kmers.size, k, n_skipped,
    )
    return kmers


# ---------------------------------------------------------------------------
# Frozen k-mer set
# ---------------------------------------------------------------------------
def pattern_id(col_index: int) -> str:
    """Pattern id for a matrix column: ``u_000017`` (zero-padded to 6 digits)."""
    return f"u_{int(col_index):06d}"


@dataclass(frozen=True)
class KmerSet:
    """The frozen k-mer set a species' models were built on.

    Attributes:
        kmers: Sorted unique canonical codes (``uint64``), length ``n_kmers``.
        pattern_col: Matrix column (pattern) of each k-mer (``int32``), same length.
        n_patterns: Number of pattern columns.
        k: k-mer length.
        presence_fraction: Majority rule for :func:`query_kmer_set`.
        species: Species key the set was built for (informational).
        backend: ``kmer`` or ``unitig-caller``.
    """

    kmers: np.ndarray
    pattern_col: np.ndarray
    n_patterns: int
    k: int = K
    presence_fraction: float = PRESENCE_FRACTION
    species: str | None = None
    backend: str = "kmer"

    def __post_init__(self) -> None:
        kmers = np.ascontiguousarray(self.kmers, dtype=np.uint64).ravel()
        cols = np.ascontiguousarray(self.pattern_col, dtype=np.int32).ravel()
        if kmers.size != cols.size:
            raise ValueError(f"{kmers.size} k-mers but {cols.size} pattern columns")
        if kmers.size > 1 and not np.all(kmers[1:] > kmers[:-1]):
            raise ValueError("kmers must be sorted ascending with no duplicates")
        if cols.size and (cols.min() < 0 or cols.max() >= self.n_patterns):
            raise ValueError("pattern_col out of range for n_patterns")
        if not 0.0 < self.presence_fraction <= 1.0:
            raise ValueError("presence_fraction must be in (0, 1]")
        _check_k(self.k)
        object.__setattr__(self, "kmers", kmers)
        object.__setattr__(self, "pattern_col", cols)
        object.__setattr__(self, "n_patterns", int(self.n_patterns))
        object.__setattr__(self, "_sizes", None)  # pattern_sizes() cache (not a dataclass field)

    @property
    def n_kmers(self) -> int:
        return int(self.kmers.size)

    def pattern_sizes(self) -> np.ndarray:
        """Member k-mer count per pattern (``int64``, length ``n_patterns``).

        Computed once and cached (the set is immutable); the returned array is
        read-only so the cache cannot be corrupted by a caller.
        """
        sizes = self._sizes  # type: ignore[attr-defined]
        if sizes is None:
            sizes = np.bincount(self.pattern_col, minlength=self.n_patterns).astype(np.int64)
            sizes.flags.writeable = False
            object.__setattr__(self, "_sizes", sizes)
        return sizes

    def pattern_ids(self) -> list[str]:
        return [pattern_id(i) for i in range(self.n_patterns)]

    def sha1(self) -> str:
        """Content hash of the set (k-mers + pattern assignment) for audit logs."""
        h = hashlib.sha1()
        h.update(np.int64(self.k).tobytes())
        h.update(self.kmers.tobytes())
        h.update(self.pattern_col.tobytes())
        return h.hexdigest()

    def member_order(self) -> tuple[np.ndarray, np.ndarray]:
        """``(order, bounds)``: ``kmers[order[bounds[p]:bounds[p + 1]]]`` are pattern ``p``'s members."""
        order = np.argsort(self.pattern_col, kind="stable")
        bounds = np.zeros(self.n_patterns + 1, dtype=np.int64)
        np.cumsum(self.pattern_sizes(), out=bounds[1:])
        return order, bounds

    def member_sequences(self) -> list[list[str]]:
        """Decoded member k-mers per pattern, in pattern order.

        Builds one Python string per k-mer: fine for small sets and tests; the
        stage writes the index with :func:`_write_index`, which never does.
        """
        order = np.argsort(self.pattern_col, kind="stable")
        bounds = np.searchsorted(self.pattern_col[order], np.arange(self.n_patterns + 1))
        decoded = decode_kmers(self.kmers[order], self.k)
        return [decoded[bounds[i] : bounds[i + 1]] for i in range(self.n_patterns)]

    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[str],
        pattern_col: np.ndarray,
        n_patterns: int,
        k: int = K,
        presence_fraction: float = PRESENCE_FRACTION,
        species: str | None = None,
        backend: str = "unitig-caller",
    ) -> "KmerSet":
        """Decompose unitig sequences into canonical k-mers carrying their pattern column.

        Used by the unitig-caller backend so querying works with the pure-Python
        matcher. A k-mer occurring in several unitigs keeps the first assignment.
        """
        cols = np.asarray(pattern_col, dtype=np.int32).ravel()
        if len(sequences) != cols.size:
            raise ValueError("one pattern column per sequence is required")
        parts: list[np.ndarray] = []
        part_cols: list[np.ndarray] = []
        for seq, col in zip(sequences, cols):
            codes = np.unique(encode_kmers(seq, k))
            if codes.size == 0:
                logger.warning("unitig shorter than k=%d skipped: %r", k, seq[:40])
                continue
            parts.append(codes)
            part_cols.append(np.full(codes.size, col, dtype=np.int32))
        if not parts:
            return cls(np.empty(0, np.uint64), np.empty(0, np.int32), n_patterns, k,
                       presence_fraction, species, backend)
        kmers = np.concatenate(parts)
        colv = np.concatenate(part_cols)
        uniq, first = np.unique(kmers, return_index=True)
        return cls(uniq, colv[first], n_patterns, k, presence_fraction, species, backend)


def save_kmer_set(path: Path | str, kmer_set: KmerSet) -> Path:
    """Write a :class:`KmerSet` as ``.npz`` (arrays ``kmers, pattern_col, pattern_size``
    plus scalars ``k, n_patterns, presence_fraction, species, backend, sha1``)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        np.savez_compressed(
            fh,
            kmers=kmer_set.kmers,
            pattern_col=kmer_set.pattern_col,
            pattern_size=kmer_set.pattern_sizes(),
            k=np.int64(kmer_set.k),
            n_patterns=np.int64(kmer_set.n_patterns),
            presence_fraction=np.float64(kmer_set.presence_fraction),
            species=np.str_(kmer_set.species or ""),
            backend=np.str_(kmer_set.backend),
            sha1=np.str_(kmer_set.sha1()),
        )
    logger.info(
        "wrote k-mer set: %d k-mers in %d patterns (k=%d, sha1=%s) to %s",
        kmer_set.n_kmers, kmer_set.n_patterns, kmer_set.k, kmer_set.sha1()[:12], out,
    )
    return out


def load_kmer_set(path: Path | str) -> KmerSet:
    """Read a :class:`KmerSet` written by :func:`save_kmer_set`."""
    src = Path(path)
    with np.load(src, allow_pickle=False) as z:
        missing = {"kmers", "pattern_col", "k", "n_patterns", "presence_fraction"} - set(z.files)
        if missing:
            raise ValueError(f"{src} is not a k-mer set file; missing arrays {sorted(missing)}")
        species = str(z["species"]) if "species" in z.files else ""
        backend = str(z["backend"]) if "backend" in z.files else "kmer"
        ks = KmerSet(
            kmers=z["kmers"],
            pattern_col=z["pattern_col"],
            n_patterns=int(z["n_patterns"]),
            k=int(z["k"]),
            presence_fraction=float(z["presence_fraction"]),
            species=species or None,
            backend=backend,
        )
    logger.debug("loaded k-mer set: %d k-mers, %d patterns from %s", ks.n_kmers, ks.n_patterns, src)
    return ks


def read_kmer_set_sha1(path: Path | str) -> str:
    """:meth:`KmerSet.sha1` of a set written by :func:`save_kmer_set`, without loading the arrays.

    Reads the ``sha1`` scalar :func:`save_kmer_set` stores (one small npz member);
    a file without it (written before the scalar existed) is loaded and hashed.
    Training records this value in every unitig model's ``features.json``
    (``unitig_kmer_set_sha1``) so a bundle can prove which set its columns index.
    """
    src = Path(path)
    with np.load(src, allow_pickle=False) as z:
        if "sha1" in z.files:
            value = str(z["sha1"])
            if value:
                return value
    return load_kmer_set(src).sha1()


def _present_positions(sorted_set: np.ndarray, sorted_query: np.ndarray) -> np.ndarray:
    """Positions in ``sorted_set`` of the elements that also occur in ``sorted_query``.

    Binary-searches the (small, one genome) query into the (large, frozen) set:
    ``O(|query| log |set|)`` and no ``|set|``-sized temporary. Both inputs must be
    sorted and unique; the output (``int64``) is then strictly increasing.
    """
    if sorted_query.size == 0 or sorted_set.size == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(sorted_set, sorted_query)
    inside = pos < sorted_set.size
    pos = pos[inside]
    hit = sorted_set[pos] == sorted_query[inside]
    return pos[hit].astype(np.int64, copy=False)


def _present_mask(sorted_set: np.ndarray, sorted_query: np.ndarray) -> np.ndarray:
    """Boolean mask over ``sorted_set``: which elements occur in ``sorted_query``."""
    mask = np.zeros(sorted_set.size, dtype=bool)
    mask[_present_positions(sorted_set, sorted_query)] = True
    return mask


def _called_patterns(
    kmers: np.ndarray,
    pattern_col: np.ndarray,
    sizes: np.ndarray,
    presence_fraction: float,
    genome: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Boolean pattern calls for one genome's sorted unique k-mers, plus the k-mer hit count."""
    pos = _present_positions(kmers, genome)
    hits = np.bincount(np.asarray(pattern_col)[pos], minlength=sizes.size)
    return (sizes > 0) & (hits >= presence_fraction * sizes), int(pos.size)


def query_kmer_set(kmer_set: KmerSet, fasta_path: Path | str) -> np.ndarray:
    """Pattern presence vector (``int8``, length ``n_patterns``) for one genome.

    A pattern is 1 when at least ``presence_fraction`` (default 50 %) of its
    member k-mers occur in the genome; see the module docstring for why.
    """
    km = genome_kmers(fasta_path, kmer_set.k)
    called, n_present = _called_patterns(
        kmer_set.kmers, kmer_set.pattern_col, kmer_set.pattern_sizes(), kmer_set.presence_fraction, km
    )
    logger.debug(
        "query %s: %d/%d k-mers present, %d/%d patterns called",
        fasta_path, n_present, kmer_set.n_kmers, int(called.sum()), kmer_set.n_patterns,
    )
    return called.astype(np.int8)


def query_genome(
    kmers_npz_path: Path | str | Paths,
    fasta_path: Path | str,
    fasta: Path | str | None = None,
) -> np.ndarray:
    """Pattern vector (``int8``, ``n_patterns``) for a new genome from the frozen k-mer set.

    Two call forms are accepted:

    * ``query_genome(kmers_npz_path, fasta_path)`` -- the prediction-pipeline
      form; needs only ``unitigs_<SPECIES>_kmers.npz``, never the matrix.
    * ``query_genome(paths, species, fasta_path)`` -- the ``DESIGN.md`` form;
      resolves ``paths.unitig_kmers(species)`` first.

    Callers querying many genomes should :func:`load_kmer_set` once and call
    :func:`query_kmer_set` directly.
    """
    if isinstance(kmers_npz_path, Paths):
        if fasta is None:
            raise TypeError("query_genome(paths, species, fasta_path) requires the FASTA path")
        npz = kmers_npz_path.unitig_kmers(str(fasta_path))
        target = Path(fasta)
    else:
        if fasta is not None:
            raise TypeError("query_genome(kmers_npz_path, fasta_path) takes two arguments")
        npz = Path(kmers_npz_path)
        target = Path(fasta_path)
    return query_kmer_set(load_kmer_set(npz), target)


# ---------------------------------------------------------------------------
# Pattern collapse
# ---------------------------------------------------------------------------
def _segment_sums(values: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    """Per-segment ``uint64`` sums (mod 2**64) of ``values`` cut by ``indptr``.

    A cumulative sum with a leading zero handles empty segments, unlike
    ``np.add.reduceat``.
    """
    cs = np.zeros(values.size + 1, dtype=np.uint64)
    np.cumsum(values, dtype=np.uint64, out=cs[1:])
    idx = np.asarray(indptr, dtype=np.int64)
    return cs[idx[1:]] - cs[idx[:-1]]


def collapse_patterns(matrix: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Group identical presence/absence columns into patterns.

    Each column is keyed by ``(sum splitmix64(row), sum splitmix64(row ^ salt),
    nnz)`` over its row indices -- two independent 64-bit set hashes plus the
    count. Patterns are numbered by the first column that carries them, so
    column order (sorted k-mer code) fixes the pattern order deterministically.

    Args:
        matrix: Binary sparse matrix, rows = genomes, columns = k-mers/unitigs.

    Returns:
        ``(pattern_col, representative_cols)``: ``pattern_col[j]`` is the pattern
        of column ``j`` (``int32``); ``representative_cols[p]`` is the first
        column of pattern ``p`` (``int64``), so ``matrix[:, representative_cols]``
        is the pattern matrix.
    """
    csc = sp.csc_matrix(matrix)
    csc.sum_duplicates()
    csc.eliminate_zeros()
    csc.sort_indices()
    n_cols = csc.shape[1]
    if n_cols == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    rows = csc.indices.astype(np.uint64)
    nnz = np.diff(csc.indptr).astype(np.uint64)
    h1 = _segment_sums(_splitmix64(rows), csc.indptr)
    h2 = _segment_sums(_splitmix64(rows ^ _SALT2), csc.indptr)
    return _patterns_from_keys(h1, h2, nnz)


def _row_salts(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row contributions to the two column set-hashes of :func:`collapse_patterns`."""
    rows = np.arange(n_rows, dtype=np.uint64)
    return _splitmix64(rows), _splitmix64(rows ^ _SALT2)


def _patterns_from_keys(h1: np.ndarray, h2: np.ndarray, nnz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pattern ids from per-column ``(set-hash 1, set-hash 2, count)`` keys.

    Columns with equal keys share a pattern; patterns are numbered by their first
    column. Shared by :func:`collapse_patterns` (keys from a matrix) and the
    streaming k-mer build (keys accumulated genome by genome), so both give
    identical patterns for the same presence/absence data.
    """
    n_cols = int(np.asarray(h1).size)
    if n_cols == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    keys = np.stack(
        [np.asarray(h1, dtype=np.uint64), np.asarray(h2, dtype=np.uint64), np.asarray(nnz).astype(np.uint64)], axis=1
    )
    _uniq, first_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    del keys, _uniq
    inverse = np.asarray(inverse).ravel()
    order = np.argsort(first_idx, kind="stable")
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    pattern_col = rank[inverse].astype(np.int32)
    representative = first_idx[order].astype(np.int64)
    logger.info("collapsed %d columns into %d distinct patterns", n_cols, representative.size)
    return pattern_col, representative


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    """Output of a backend build on the training genomes.

    Attributes:
        kmer_set: The frozen set with per-k-mer pattern columns.
        matrix: CSR ``int8`` training genomes x patterns, rows in ``genome_ids`` order.
        genome_ids: Training genome ids in row order.
        train_frequency: Share of training genomes carrying each pattern.
        n_kmers_total: Distinct k-mers (or unitigs) seen before the frequency filter.
        n_kmers_kept: Count after the filter.
    """

    kmer_set: KmerSet
    matrix: sp.csr_matrix
    genome_ids: list[str]
    train_frequency: np.ndarray
    n_kmers_total: int
    n_kmers_kept: int


class UnitigBackend(Protocol):
    """Structural type for the two build backends."""

    name: str

    def build(self, fastas: Mapping[str, Path], droplog: DropLog) -> BuildResult: ...


class _KmerCounter:
    """Incremental merge of per-genome unique k-mer arrays into ``(kmers, counts)``.

    Per-genome arrays are buffered until ``buffer_elements`` codes accumulate,
    then reduced with one ``np.unique(return_counts=True)`` and merged into the
    sorted global table with ``searchsorted`` + ``np.insert`` (O(n) per flush).
    Peak memory is the buffer plus the distinct-k-mer table.
    """

    def __init__(self, buffer_elements: int = 50_000_000) -> None:
        if buffer_elements < 1:
            raise ValueError("buffer_elements must be positive")
        self._limit = int(buffer_elements)
        self._kmers = np.empty(0, dtype=np.uint64)
        self._counts = np.empty(0, dtype=np.int32)
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self.n_genomes = 0

    def add(self, unique_kmers: np.ndarray) -> None:
        self.n_genomes += 1
        if unique_kmers.size == 0:
            return
        self._buffer.append(unique_kmers)
        self._buffered += int(unique_kmers.size)
        if self._buffered >= self._limit:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        cat = self._buffer[0] if len(self._buffer) == 1 else np.concatenate(self._buffer)
        self._buffer = []
        self._buffered = 0
        u, c = np.unique(cat, return_counts=True)
        c = c.astype(np.int32)
        if self._kmers.size == 0:
            self._kmers, self._counts = u, c
            return
        pos = np.searchsorted(self._kmers, u)
        in_range = pos < self._kmers.size
        hit = np.zeros(u.size, dtype=bool)
        hit[in_range] = self._kmers[pos[in_range]] == u[in_range]
        self._counts[pos[hit]] += c[hit]
        new = ~hit
        if new.any():
            self._kmers = np.insert(self._kmers, pos[new], u[new])
            self._counts = np.insert(self._counts, pos[new], c[new])

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        """``(kmers, counts)``: sorted unique codes and the number of genomes carrying each."""
        self._flush()
        return self._kmers, self._counts


class TooManyGenomesForKmerBackend(ValueError):
    """The pure-Python k-mer backend was asked to build from more genomes than it allows."""


# --------------------------------------------------------------------------- #
# Worker-process tasks (genome2mic.parallel). State is per process.
# --------------------------------------------------------------------------- #
_WORKER_STATE: dict[str, Any] = {}


def _init_worker(spec: Mapping[str, Any], scalars: Mapping[str, Any] | None = None) -> None:
    """Load shared arrays (memory-mapped in worker processes) and scalars into this process."""
    _WORKER_STATE.clear()
    _WORKER_STATE.update(parallel.load_shared(spec))
    if scalars:
        _WORKER_STATE.update(scalars)


def _genome_kmers_task(task: tuple[str, int]) -> np.ndarray:
    """Sorted unique canonical k-mers of one FASTA."""
    path, k = task
    return genome_kmers(path, k)


def _present_task(task: tuple[str, int]) -> np.ndarray:
    """Positions of one genome's k-mers in the shared sorted set ``_WORKER_STATE['set']``."""
    path, k = task
    return _present_positions(_WORKER_STATE["set"], genome_kmers(path, k))


def _query_task(task: tuple[str, int]) -> np.ndarray:
    """Called pattern columns (``int32``) of one genome against the shared frozen k-mer set."""
    path, k = task
    state = _WORKER_STATE
    called, _ = _called_patterns(
        state["kmers"], state["pattern_col"], state["sizes"], state["presence_fraction"], genome_kmers(path, k)
    )
    return np.flatnonzero(called).astype(np.int32)


def _iter_present_positions(
    sorted_set: np.ndarray,
    fastas: Sequence[str],
    k: int,
    cache: Sequence[np.ndarray] | None,
    threads: int | None,
) -> Iterator[np.ndarray]:
    """Per genome (in order): positions in ``sorted_set`` of its k-mers.

    Uses the in-memory per-genome k-mer arrays when ``cache`` is given, else
    re-reads each FASTA in worker processes sharing a memory-mapped ``sorted_set``.
    """
    if cache is not None:
        for km in cache:
            yield _present_positions(sorted_set, km)
        return
    workers = parallel.worker_count(threads, len(fastas))
    try:
        with parallel.share_arrays({"set": sorted_set}, enabled=workers > 1) as spec:
            yield from parallel.ordered_map(
                _present_task, [(p, k) for p in fastas], workers=workers, initializer=_init_worker, initargs=(spec,)
            )
    finally:
        _WORKER_STATE.clear()


class KmerBackend:
    """Pure-numpy canonical k-mer backend (default).

    Three passes over the training genomes, none of which materialises the
    ``genomes x kept k-mers`` matrix:

    1. per-genome unique k-mers -> global counts -> frequency filter;
    2. per kept k-mer, the order-independent set-hash of the genomes carrying it
       (``O(kept k-mers)`` memory) -> patterns (identical to
       :func:`collapse_patterns` on the explicit matrix);
    3. per genome, presence of each pattern's first k-mer -> the pattern row.

    Args:
        k: k-mer length (31).
        min_freq, max_freq: Inclusive training-frequency window for keeping a k-mer.
        presence_fraction: Majority rule stored in the :class:`KmerSet` for querying.
        max_cached_elements: Per-genome k-mer arrays are kept in memory for
            passes 2-3 while their total stays below this; beyond it they are
            recomputed from FASTA (in worker processes), bounding memory.
        buffer_elements: Flush threshold of the count merger.
        max_genomes: Refuse to build from more training genomes than this
            (:class:`TooManyGenomesForKmerBackend`); ``None`` disables the guard.
        threads: Worker processes for the per-genome passes (``None`` = all cores).
    """

    name = "kmer"

    def __init__(
        self,
        k: int = K,
        min_freq: float = MIN_FREQ,
        max_freq: float = MAX_FREQ,
        presence_fraction: float = PRESENCE_FRACTION,
        max_cached_elements: int = 100_000_000,
        buffer_elements: int = 50_000_000,
        max_genomes: int | None = DEFAULT_MAX_KMER_GENOMES,
        threads: int | None = None,
    ) -> None:
        _check_k(k)
        if not 0.0 <= min_freq <= max_freq <= 1.0:
            raise ValueError("need 0 <= min_freq <= max_freq <= 1")
        if max_genomes is not None and int(max_genomes) < 1:
            raise ValueError("max_genomes must be >= 1 (or None to disable the guard)")
        self.k = int(k)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.presence_fraction = float(presence_fraction)
        self.max_cached_elements = int(max_cached_elements)
        self.buffer_elements = int(buffer_elements)
        self.max_genomes = None if max_genomes is None else int(max_genomes)
        self.threads = threads

    def check_size(self, n_train: int) -> None:
        """Raise :class:`TooManyGenomesForKmerBackend` when ``n_train`` exceeds ``max_genomes``."""
        if self.max_genomes is not None and n_train > self.max_genomes:
            raise TooManyGenomesForKmerBackend(
                f"{n_train} training genomes exceed the pure-Python k-mer backend limit of {self.max_genomes} "
                "(it re-encodes every FASTA up to three times and holds a global k-mer count table). "
                "Use `--unitig-backend unitig-caller` (or `auto` with unitig-caller on PATH), or raise "
                "`--unitig-max-kmer-genomes` if this machine has the RAM and time for it (docs/VM_RUNBOOK.md)."
            )

    def build(self, fastas: Mapping[str, Path], droplog: DropLog) -> BuildResult:
        """Build the k-mer set, frequency filter, patterns and pattern matrix from training FASTAs.

        Args:
            fastas: Ordered ``genome_id -> FASTA path`` of the **training** genomes
                only. The caller is responsible for that selection.
            droplog: Receives the frequency-filter counts.

        Raises:
            TooManyGenomesForKmerBackend: more than ``max_genomes`` training genomes.
        """
        genome_ids = list(fastas)
        n_train = len(genome_ids)
        if n_train == 0:
            raise ValueError("cannot build a k-mer set from zero training genomes")
        self.check_size(n_train)
        paths_in_order = [str(fastas[g]) for g in genome_ids]
        workers = parallel.worker_count(self.threads, n_train)
        logger.info("k-mer build: %d training genomes, k=%d, %d process(es)", n_train, self.k, workers)

        # Pass 1: per-genome unique k-mers -> global counts (merged in genome order).
        counter = _KmerCounter(self.buffer_elements)
        cache: list[np.ndarray] | None = []
        cached = 0
        pass1 = parallel.ordered_map(
            _genome_kmers_task,
            [(p, self.k) for p in paths_in_order],
            workers=workers,
            max_pending=workers + 2,  # each result is one genome's k-mer array; keep few in flight
        )
        for i, km in enumerate(pass1, start=1):
            counter.add(km)
            if cache is not None:
                cached += int(km.size)
                if cached > self.max_cached_elements:
                    logger.info("k-mer cache limit reached; passes 2-3 will re-read FASTAs")
                    cache = None
                else:
                    cache.append(km)
            if i % 100 == 0 or i == n_train:
                logger.info("k-mer pass 1: %d/%d genomes", i, n_train)
        all_kmers, counts = counter.finish()
        n_total = int(all_kmers.size)

        # Frequency filter (inclusive window; contract: drop < 1 % or > 99 %).
        freq = counts.astype(np.float64) / n_train
        below = freq < self.min_freq
        above = freq > self.max_freq
        keep = ~(below | above)
        droplog.drop(
            "kmer_below_min_freq", int(below.sum()),
            detail=f"present in < {self.min_freq:.0%} of {n_train} training genomes",
        )
        droplog.drop(
            "kmer_above_max_freq", int(above.sum()),
            detail=f"present in > {self.max_freq:.0%} of {n_train} training genomes",
        )
        kept = all_kmers[keep]
        n_kept = int(kept.size)
        logger.info("frequency filter: %d of %d distinct k-mers kept", n_kept, n_total)
        del all_kmers, counts, freq, below, above, keep

        # Pass 2: per kept k-mer, accumulate the set-hash keys of collapse_patterns
        # (sums over carrying genomes, so order-free) -- O(n_kept), no k-mer matrix.
        salt1, salt2 = _row_salts(n_train)
        h1 = np.zeros(n_kept, dtype=np.uint64)
        h2 = np.zeros(n_kept, dtype=np.uint64)
        nnz = np.zeros(n_kept, dtype=np.int64)
        with np.errstate(over="ignore"):
            for r, cols in enumerate(_iter_present_positions(kept, paths_in_order, self.k, cache, self.threads)):
                h1[cols] += salt1[r]  # cols are unique per genome, so fancy += is exact
                h2[cols] += salt2[r]
                nnz[cols] += 1
                if (r + 1) % 100 == 0 or r + 1 == n_train:
                    logger.info("k-mer pass 2 (pattern keys): %d/%d genomes", r + 1, n_train)
        pattern_col, rep_cols = _patterns_from_keys(h1, h2, nnz)
        del h1, h2, nnz

        # Pass 3: pattern rows. All members of a pattern co-occur in every training
        # genome, so a genome carries pattern p iff it carries p's first k-mer.
        # rep_cols is increasing, so the representatives stay sorted.
        representatives = kept[rep_cols]
        indptr = np.zeros(n_train + 1, dtype=np.int64)
        index_parts: list[np.ndarray] = []
        for r, cols in enumerate(_iter_present_positions(representatives, paths_in_order, self.k, cache, self.threads)):
            index_parts.append(cols.astype(np.int32))
            indptr[r + 1] = indptr[r] + cols.size
        del cache
        indices = np.concatenate(index_parts) if index_parts else np.empty(0, dtype=np.int32)
        pattern_matrix = sp.csr_matrix(
            (np.ones(indices.size, dtype=np.int8), indices, indptr), shape=(n_train, int(rep_cols.size)), dtype=np.int8
        )
        pattern_matrix.has_sorted_indices = True
        counts_per_pattern = np.bincount(indices, minlength=int(rep_cols.size))
        train_frequency = counts_per_pattern.astype(np.float64) / n_train
        kmer_set = KmerSet(
            kmers=kept,
            pattern_col=pattern_col,
            n_patterns=int(rep_cols.size),
            k=self.k,
            presence_fraction=self.presence_fraction,
            backend=self.name,
        )
        return BuildResult(kmer_set, pattern_matrix, genome_ids, train_frequency, n_total, n_kept)


class UnitigCallerBackend:
    """``unitig-caller`` wrapper. Raises :class:`ToolNotAvailable` when the CLI is missing.

    Build: ``unitig-caller --call --refs refs.txt --out <prefix> --rtab`` on the
    training genomes, parse the ``.rtab``, frequency-filter and collapse unitigs
    into patterns exactly like the k-mer backend, then decompose every surviving
    unitig into canonical k-mers so :func:`query_genome` works without the tool.
    Query: :meth:`query_cli` runs ``--query`` for callers that want the tool's
    own presence calls.

    Args:
        executable: Program name or path.
        threads: ``--threads`` value (``None`` = all usable cores).
        k, min_freq, max_freq, presence_fraction: As for :class:`KmerBackend`.
    """

    name = "unitig-caller"

    def __init__(
        self,
        executable: str = "unitig-caller",
        threads: int | None = None,
        k: int = K,
        min_freq: float = MIN_FREQ,
        max_freq: float = MAX_FREQ,
        presence_fraction: float = PRESENCE_FRACTION,
    ) -> None:
        self.executable = executable
        self.threads = parallel.resolve_threads(threads)
        self.k = int(k)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.presence_fraction = float(presence_fraction)

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def _require(self) -> str:
        exe = shutil.which(self.executable)
        if exe is None:
            raise ToolNotAvailable(
                self.executable,
                hint="Use the default KmerBackend (pure Python) or install unitig-caller "
                     "(conda install -c bioconda unitig-caller).",
            )
        return exe

    @staticmethod
    def _write_refs(fastas: Mapping[str, Path], path: Path) -> dict[str, str]:
        """Write ``refs.txt`` (one FASTA path per line); return sample-name -> genome_id."""
        names: dict[str, str] = {}
        with path.open("w", encoding="utf-8") as fh:
            for gid, fasta in fastas.items():
                p = Path(fasta).resolve()
                fh.write(f"{p}\n")
                stem = p.name
                for suffix in (".gz", ".fasta", ".fa", ".fna", ".fas"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                names[str(p)] = gid
                names[p.name] = gid
                names[stem] = gid
                names[gid] = gid
        return names

    def _run(self, args: list[str], cwd: Path) -> None:
        exe = self._require()
        cmd = [exe, *args]
        logger.info("running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.executable} failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}"
            )

    def build(self, fastas: Mapping[str, Path], droplog: DropLog) -> BuildResult:
        self._require()
        genome_ids = list(fastas)
        n_train = len(genome_ids)
        if n_train == 0:
            raise ValueError("cannot build unitigs from zero training genomes")
        with tempfile.TemporaryDirectory(prefix="unitig_caller_") as tmp:
            work = Path(tmp)
            names = self._write_refs(fastas, work / "refs.txt")
            self._run(
                ["--call", "--refs", "refs.txt", "--out", "unitigs", "--rtab",
                 "--threads", str(self.threads)],
                cwd=work,
            )
            rtab = work / "unitigs.rtab"
            if not rtab.exists():
                candidates = sorted(work.glob("unitigs*.rtab"))
                if not candidates:
                    raise RuntimeError(f"{self.executable} produced no .rtab file in {work}")
                rtab = candidates[0]
            sequences, samples, presence = parse_rtab(rtab)
        return self._finish(sequences, samples, presence, names, genome_ids, droplog)

    def _finish(
        self,
        sequences: list[str],
        samples: list[str],
        presence: sp.csr_matrix,
        names: Mapping[str, str],
        genome_ids: list[str],
        droplog: DropLog,
    ) -> BuildResult:
        """Frequency filter + pattern collapse of a parsed unitig table."""
        n_train = len(genome_ids)
        row_of = {gid: i for i, gid in enumerate(genome_ids)}
        sample_rows = []
        for s in samples:
            gid = names.get(s)
            if gid is None or gid not in row_of:
                raise ContractViolation(f"unitig-caller sample {s!r} is not a training genome", STAGE)
            sample_rows.append(row_of[gid])
        if len(set(sample_rows)) != n_train:
            raise ContractViolation("unitig-caller output does not cover every training genome", STAGE)
        # presence: samples x unitigs -> reorder rows to genome_ids order.
        order = np.argsort(np.asarray(sample_rows))
        matrix = sp.csr_matrix(presence[order], dtype=np.int8)
        n_total = matrix.shape[1]
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        freq = counts / n_train
        below = freq < self.min_freq
        above = freq > self.max_freq
        keep = ~(below | above)
        droplog.drop("unitig_below_min_freq", int(below.sum()),
                     detail=f"present in < {self.min_freq:.0%} of {n_train} training genomes")
        droplog.drop("unitig_above_max_freq", int(above.sum()),
                     detail=f"present in > {self.max_freq:.0%} of {n_train} training genomes")
        kept_cols = np.flatnonzero(keep)
        kept_seqs = [sequences[j] for j in kept_cols]
        kept_matrix = sp.csr_matrix(matrix[:, kept_cols], dtype=np.int8)
        pattern_col, rep_cols = collapse_patterns(kept_matrix)
        pattern_matrix = sp.csr_matrix(kept_matrix[:, rep_cols], dtype=np.int8)
        pattern_matrix.sort_indices()
        train_frequency = np.asarray(pattern_matrix.sum(axis=0)).ravel() / n_train
        kmer_set = KmerSet.from_sequences(
            kept_seqs, pattern_col, int(rep_cols.size), self.k, self.presence_fraction,
            backend=self.name,
        )
        return BuildResult(kmer_set, pattern_matrix, genome_ids, train_frequency,
                           int(n_total), int(kept_cols.size))

    def query_cli(
        self, fastas: Mapping[str, Path], unitig_sequences: Sequence[str]
    ) -> tuple[list[str], sp.csr_matrix]:
        """Run ``unitig-caller --query`` for ``fastas`` against fixed unitigs.

        Returns:
            ``(genome_ids, presence)`` with ``presence`` CSR ``int8`` genomes x unitigs
            in the order of ``unitig_sequences``.
        """
        self._require()
        with tempfile.TemporaryDirectory(prefix="unitig_caller_q_") as tmp:
            work = Path(tmp)
            names = self._write_refs(fastas, work / "refs.txt")
            with (work / "unitigs.fasta").open("w", encoding="utf-8") as fh:
                for i, seq in enumerate(unitig_sequences):
                    fh.write(f">u{i}\n{seq}\n")
            self._run(
                ["--query", "--refs", "refs.txt", "--unitigs", "unitigs.fasta",
                 "--out", "query", "--rtab", "--threads", str(self.threads)],
                cwd=work,
            )
            sequences, samples, presence = parse_rtab(work / "query.rtab")
        seq_pos = {s: i for i, s in enumerate(unitig_sequences)}
        col_map = np.array([seq_pos.get(s, -1) for s in sequences])
        if (col_map < 0).any():
            raise RuntimeError("unitig-caller --query returned unitigs that were not requested")
        gids = [names.get(s, s) for s in samples]
        out = sp.lil_matrix((len(gids), len(unitig_sequences)), dtype=np.int8)
        coo = presence.tocoo()
        out[coo.row, col_map[coo.col]] = 1
        return gids, sp.csr_matrix(out, dtype=np.int8)


def parse_rtab(path: Path | str) -> tuple[list[str], list[str], sp.csr_matrix]:
    """Parse a unitig-caller ``.rtab`` table.

    Layout: a header line whose first field names the sequence column and whose
    remaining fields are sample names; one line per unitig with the sequence then
    0/1 per sample.

    Returns:
        ``(sequences, samples, presence)`` with ``presence`` CSR ``int8`` of shape
        samples x unitigs.
    """
    sequences: list[str] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    samples: list[str] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
        if not header:
            raise ValueError(f"{path}: empty rtab")
        samples = header.split("\t")[1:]
        for j, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != len(samples) + 1:
                raise ValueError(f"{path}: line {j + 2} has {len(fields)} fields, expected {len(samples) + 1}")
            sequences.append(fields[0].upper())
            vals = np.array(fields[1:], dtype=np.int8)
            hit = np.flatnonzero(vals)
            rows.append(hit)
            cols.append(np.full(hit.size, j, dtype=np.int64))
    r = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    presence = sp.csr_matrix(
        (np.ones(r.size, dtype=np.int8), (r, c)), shape=(len(samples), len(sequences)), dtype=np.int8
    )
    return sequences, samples, presence


def parse_pyseer(
    path: Path | str, samples: Sequence[str] | None = None
) -> tuple[list[str], list[str], sp.csr_matrix]:
    """Parse unitig-caller ``--pyseer`` output (``SEQ | sample1:1 sample2:1 ...``).

    Args:
        path: The ``.pyseer`` file.
        samples: Optional full sample list fixing the row order; otherwise samples
            appear in first-seen order (absent genomes would then be missing).

    Returns:
        ``(sequences, samples, presence)`` as for :func:`parse_rtab`.
    """
    sequences: list[str] = []
    sample_index: dict[str, int] = {s: i for i, s in enumerate(samples)} if samples else {}
    rows: list[int] = []
    cols: list[int] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seq, _, rest = line.partition("|")
            j = len(sequences)
            sequences.append(seq.strip().upper())
            for token in rest.split():
                name, _, val = token.rpartition(":")
                if not name:
                    name, val = token, "1"
                if val.strip() in ("0", "0.0"):
                    continue
                if name not in sample_index:
                    if samples is not None:
                        raise ValueError(f"{path}: sample {name!r} not in the supplied sample list")
                    sample_index[name] = len(sample_index)
                rows.append(sample_index[name])
                cols.append(j)
    ordered = [s for s, _ in sorted(sample_index.items(), key=lambda kv: kv[1])]
    presence = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(len(ordered), len(sequences)),
        dtype=np.int8,
    )
    return sequences, ordered, presence


def make_backend(
    name: str = "kmer",
    *,
    threads: int | None = None,
    max_kmer_genomes: int | None = DEFAULT_MAX_KMER_GENOMES,
    **kwargs: object,
) -> KmerBackend | UnitigCallerBackend:
    """Backend factory: ``kmer`` (default), ``unitig-caller``, or ``auto``.

    ``auto`` picks unitig-caller when it is on ``PATH`` and the k-mer backend
    otherwise. ``threads`` goes to both backends (worker processes /
    ``--threads``); ``max_kmer_genomes`` is the k-mer backend's size guard;
    other keyword arguments (``k``, ``min_freq``, ...) go to both.
    """
    key = name.strip().lower().replace("_", "-")
    if key == "kmer":
        return KmerBackend(max_genomes=max_kmer_genomes, threads=threads, **kwargs)  # type: ignore[arg-type]
    if key in ("unitig-caller", "unitigcaller"):
        return UnitigCallerBackend(threads=threads, **kwargs)  # type: ignore[arg-type]
    if key == "auto":
        cli = UnitigCallerBackend(threads=threads, **kwargs)  # type: ignore[arg-type]
        if cli.available():
            logger.info("unitig-caller found on PATH; using it")
            return cli
        logger.info("unitig-caller not on PATH; using the pure-Python k-mer backend")
        return KmerBackend(max_genomes=max_kmer_genomes, threads=threads, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown unitig backend {name!r}; expected kmer, unitig-caller or auto")


# ---------------------------------------------------------------------------
# Stage: genome selection, build, run
# ---------------------------------------------------------------------------
@dataclass
class UnitigBuild:
    """Summary of one species build (also what :func:`run` tabulates)."""

    species: str
    backend: str
    n_train: int
    n_queried: int
    n_kmers_total: int
    n_kmers_kept: int
    n_patterns: int
    kmer_set_sha1: str
    matrix_path: Path
    rows_path: Path
    index_path: Path
    kmers_path: Path
    rows: pd.DataFrame = field(repr=False)
    index: pd.DataFrame = field(repr=False)

    def as_record(self) -> dict[str, object]:
        return {
            "species": self.species,
            "backend": self.backend,
            "n_train": self.n_train,
            "n_queried": self.n_queried,
            "n_kmers_total": self.n_kmers_total,
            "n_kmers_kept": self.n_kmers_kept,
            "n_patterns": self.n_patterns,
            "kmer_set_sha1": self.kmer_set_sha1,
        }


def training_and_query_genomes(
    paths: Paths, species: str, droplog: DropLog
) -> tuple[dict[str, Path], dict[str, Path], pd.DataFrame]:
    """QC-passing genomes of ``species`` split into build (train) and query sets.

    Reads **only** ``splits.parquet`` (species, split) and ``qc.parquet``
    (qc_pass). Labels are never opened here. Every exclusion is counted in
    ``droplog``: no QC row, QC fail, missing FASTA.

    Returns:
        ``(train_fastas, query_fastas, genomes)`` -- two ordered ``genome_id ->
        FASTA`` maps (sorted by genome_id) and the surviving genome table with
        columns ``genome_id, split``.
    """
    key = species.strip().upper()
    splits = read_parquet(paths.splits, columns=["genome_id", "species", "split"])
    qc = read_parquet(paths.qc, columns=["genome_id", "qc_pass"])
    sp_rows = splits.loc[splits["species"] == key, ["genome_id", "split"]].copy()
    logger.info("%s: %d genomes in splits.parquet", key, len(sp_rows))
    if sp_rows["genome_id"].duplicated().any():
        raise ContractViolation(f"{key}: duplicate genome_id in splits.parquet", STAGE)

    qc_dedup = qc.drop_duplicates("genome_id")
    merged = sp_rows.merge(qc_dedup, on="genome_id", how="left")
    has_qc = merged["qc_pass"].notna()
    merged = droplog.keep_where(merged, has_qc, "no_qc_row", detail=f"species={key}")
    qc_pass = merged["qc_pass"].astype(bool)
    merged = droplog.keep_where(merged, qc_pass, "qc_fail", detail=f"species={key}")

    fasta_paths = [paths.genome_fasta(g) for g in merged["genome_id"]]
    exists = np.array([p.exists() for p in fasta_paths], dtype=bool)
    merged = droplog.keep_where(merged, exists, "fasta_missing", detail=f"species={key}")
    merged = merged.sort_values("genome_id", kind="stable").reset_index(drop=True)

    is_train = (merged["split"] == TRAIN_SPLIT).to_numpy(dtype=bool)
    train = {g: paths.genome_fasta(g) for g in merged.loc[is_train, "genome_id"]}
    query = {g: paths.genome_fasta(g) for g in merged.loc[~is_train, "genome_id"]}
    logger.info("%s: %d training genomes to build from, %d genomes to query", key, len(train), len(query))
    return train, query, merged[["genome_id", "split"]]


def _rows_frame(train_ids: Sequence[str], query_ids: Sequence[str], split_of: Mapping[str, str]) -> pd.DataFrame:
    ids = list(train_ids) + list(query_ids)
    roles = [ROLE_BUILT] * len(train_ids) + [ROLE_QUERIED] * len(query_ids)
    return pd.DataFrame(
        {
            "row_index": np.arange(len(ids), dtype=np.int64),
            "genome_id": pd.array(ids, dtype="str"),
            "split": pd.array([split_of.get(g) for g in ids], dtype="str"),
            "role": pd.array(roles, dtype="str"),
        }
    )


def _index_schema() -> pa.Schema:
    """Arrow schema of ``unitigs_<SPECIES>_index.parquet`` (:data:`INDEX_COLUMNS`)."""
    return pa.schema(
        [
            pa.field("col_index", pa.int64()),
            pa.field("pattern_id", pa.string()),
            pa.field("n_unitigs", pa.int64()),
            pa.field("unitig_sequences", pa.list_(pa.string())),
            pa.field("train_frequency", pa.float64()),
        ]
    )


def _iter_index_batches(
    kmer_set: KmerSet, train_frequency: np.ndarray, batch_kmers: int = INDEX_BATCH_KMERS
) -> Iterator[pa.RecordBatch]:
    """Index rows in pattern order, in record batches of about ``batch_kmers`` member k-mers.

    Member sequences are decoded straight into an Arrow string buffer (``k``
    ASCII bytes per k-mer) -- no Python ``str`` per k-mer -- so memory per batch
    is about ``batch_kmers x (k + 16)`` bytes whatever the size of the set.
    """
    n = kmer_set.n_patterns
    freq = np.asarray(train_frequency, dtype=np.float64).ravel()
    if freq.size != n:
        raise ValueError(f"train_frequency has {freq.size} values for {n} patterns")
    if batch_kmers < 1:
        raise ValueError("batch_kmers must be positive")
    schema = _index_schema()
    sizes = kmer_set.pattern_sizes()
    order, bounds = kmer_set.member_order()
    k = kmer_set.k
    p0 = 0
    while p0 < n:
        p1 = int(np.searchsorted(bounds, bounds[p0] + batch_kmers, side="right")) - 1
        p1 = min(max(p1, p0 + 1), n)  # at least one pattern per batch
        lo, hi = int(bounds[p0]), int(bounds[p1])
        m = hi - lo
        if m * k >= 2**31:  # pragma: no cover - one pattern with > 69M k-mers
            raise ValueError(f"pattern batch of {m} k-mers is too large for one Arrow string array")
        letters = _decode_letters(kmer_set.kmers[order[lo:hi]], k).reshape(-1)
        offsets = np.arange(m + 1, dtype=np.int32) * np.int32(k)
        strings = pa.Array.from_buffers(pa.string(), m, [None, pa.py_buffer(offsets), pa.py_buffer(letters)])
        member_offsets = (bounds[p0 : p1 + 1] - lo).astype(np.int32)
        sequences = pa.ListArray.from_arrays(pa.array(member_offsets, pa.int32()), strings)
        yield pa.record_batch(
            [
                pa.array(np.arange(p0, p1, dtype=np.int64), pa.int64()),
                pa.array([pattern_id(i) for i in range(p0, p1)], pa.string()),
                pa.array(np.asarray(sizes[p0:p1], dtype=np.int64), pa.int64()),
                sequences,
                pa.array(freq[p0:p1], pa.float64()),
            ],
            schema=schema,
        )
        p0 = p1


def _write_index(path: Path, kmer_set: KmerSet, train_frequency: np.ndarray, batch_kmers: int = INDEX_BATCH_KMERS) -> None:
    """Write the pattern index parquet batch by batch (see :func:`_iter_index_batches`)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = _index_schema()
    with pq.ParquetWriter(target, schema) as writer:
        n_batches = 0
        for batch in _iter_index_batches(kmer_set, train_frequency, batch_kmers):
            writer.write_batch(batch)
            n_batches += 1
        if n_batches == 0:
            writer.write_table(schema.empty_table())


def _light_index(kmer_set: KmerSet, train_frequency: np.ndarray) -> pd.DataFrame:
    """The pattern index without member sequences (:data:`LOAD_INDEX_COLUMNS`), built in memory."""
    n = kmer_set.n_patterns
    return pd.DataFrame(
        {
            "col_index": np.arange(n, dtype=np.int64),
            "pattern_id": pd.array(kmer_set.pattern_ids(), dtype="str"),
            "n_unitigs": np.array(kmer_set.pattern_sizes(), dtype=np.int64),
            "train_frequency": np.asarray(train_frequency, dtype=np.float64),
        },
        columns=list(LOAD_INDEX_COLUMNS),
    )


def _query_rows(
    kmer_set: KmerSet, fastas: Sequence[str], threads: int | None, species: str
) -> sp.csr_matrix:
    """CSR ``int8`` pattern rows of ``fastas`` queried against the frozen set (in input order).

    Same calls as :func:`query_kmer_set`, computed in worker processes that share
    one memory-mapped copy of the set. Only non-zero columns are kept, so the
    block is assembled directly as CSR (never a dense ``n x patterns`` array).
    """
    n_patterns = kmer_set.n_patterns
    indptr = np.zeros(len(fastas) + 1, dtype=np.int64)
    parts: list[np.ndarray] = []
    workers = parallel.worker_count(threads, len(fastas))
    shared = {
        "kmers": kmer_set.kmers,
        "pattern_col": kmer_set.pattern_col,
        "sizes": np.asarray(kmer_set.pattern_sizes()),
    }
    try:
        with parallel.share_arrays(shared, enabled=workers > 1) as spec:
            results = parallel.ordered_map(
                _query_task,
                [(p, kmer_set.k) for p in fastas],
                workers=workers,
                initializer=_init_worker,
                initargs=(spec, {"presence_fraction": kmer_set.presence_fraction}),
            )
            for i, cols in enumerate(results, start=1):
                parts.append(cols)
                indptr[i] = indptr[i - 1] + cols.size
                if i % 100 == 0 or i == len(fastas):
                    logger.info("%s: queried %d/%d genomes", species, i, len(fastas))
    finally:
        _WORKER_STATE.clear()
    indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
    return sp.csr_matrix(
        (np.ones(indices.size, dtype=np.int8), indices, indptr), shape=(len(fastas), n_patterns), dtype=np.int8
    )


def build(
    paths: Paths,
    config: object | None,
    species: str,
    backend: UnitigBackend | None = None,
    droplog: DropLog | None = None,
    *,
    threads: int | None = None,
) -> UnitigBuild:
    """Build the unitig/k-mer feature files for one species.

    Steps: select genomes (:func:`training_and_query_genomes`), build the frozen
    set from the training genomes with ``backend`` (default :class:`KmerBackend`),
    query every other QC-passing genome, write the four stage-8 files. Never
    reads ``labels.parquet``. An existing k-mer set with a different content hash
    is overwritten with a WARNING (it means the training genome set changed).

    Args:
        paths: Project paths.
        config: Loaded ``Config`` (optional; used to validate the species key).
        species: 5-letter species key.
        backend: Build backend; ``None`` -> :class:`KmerBackend` (with ``threads``).
        droplog: Stage drop log; a local one is created when ``None``.
        threads: Worker processes for querying the non-training genomes
            (``None`` = all cores). Results do not depend on it.

    Returns:
        The build summary; ``index`` holds the pattern index *without* member
        sequences (read ``index_path`` for those).
    """
    key = species.strip().upper()
    if config is not None and hasattr(config, "species") and key not in config.species:
        raise ValueError(f"unknown species {species!r}; configured: {sorted(config.species)}")
    be = backend or KmerBackend(threads=threads)
    log = droplog if droplog is not None else DropLog(STAGE)

    train_fastas, query_fastas, genomes = training_and_query_genomes(paths, key, log)
    if not train_fastas:
        raise ValueError(f"{key}: no QC-passing training genomes with a FASTA; nothing to build")
    split_of = dict(zip(genomes["genome_id"].tolist(), genomes["split"].tolist()))

    result = be.build(train_fastas, log)
    kmer_set = KmerSet(
        kmers=result.kmer_set.kmers,
        pattern_col=result.kmer_set.pattern_col,
        n_patterns=result.kmer_set.n_patterns,
        k=result.kmer_set.k,
        presence_fraction=result.kmer_set.presence_fraction,
        species=key,
        backend=result.kmer_set.backend,
    )
    sha1 = kmer_set.sha1()
    kmers_path = paths.unitig_kmers(key)
    if kmers_path.exists():
        try:
            old = load_kmer_set(kmers_path).sha1()
        except (ValueError, OSError):  # pragma: no cover - corrupt file
            old = None
        if old is not None and old != sha1:
            logger.warning(
                "%s: existing k-mer set %s differs from the rebuilt set %s; the training "
                "genome set changed (splits are frozen -- check before training)",
                key, old[:12], sha1[:12],
            )

    # Query every other QC-passing genome against the frozen set.
    query_ids = list(query_fastas)
    n_patterns = kmer_set.n_patterns
    if query_ids:
        queried = _query_rows(kmer_set, [str(query_fastas[g]) for g in query_ids], threads, key)
        matrix = sp.vstack([result.matrix, queried], format="csr", dtype=np.int8)
        del queried
    else:
        matrix = sp.csr_matrix(result.matrix, dtype=np.int8)
    matrix.sort_indices()
    if matrix.shape != (len(result.genome_ids) + len(query_ids), n_patterns):
        raise ContractViolation(f"{key}: unitig matrix shape {matrix.shape} does not match rows", STAGE)

    rows = _rows_frame(result.genome_ids, query_ids, split_of)
    built_not_train = rows[(rows["role"] == ROLE_BUILT) & (rows["split"] != TRAIN_SPLIT)]
    if len(built_not_train):
        raise ContractViolation(
            f"{key}: {len(built_not_train)} non-train genomes were used to build the k-mer set", STAGE
        )
    matrix_path = paths.unitigs(key)
    rows_path = paths.unitig_rows(key)
    index_path = paths.unitig_index(key)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("wb") as fh:
        sp.save_npz(fh, matrix, compressed=True)
    write_parquet(rows, rows_path)
    _write_index(index_path, kmer_set, result.train_frequency)
    save_kmer_set(kmers_path, kmer_set)
    logger.info(
        "%s: wrote unitig matrix %s (%d built + %d queried rows x %d patterns; %d/%d k-mers kept; sha1 %s)",
        key, matrix_path, len(result.genome_ids), len(query_ids), n_patterns,
        result.n_kmers_kept, result.n_kmers_total, sha1[:12],
    )
    return UnitigBuild(
        species=key,
        backend=be.name,
        n_train=len(result.genome_ids),
        n_queried=len(query_ids),
        n_kmers_total=result.n_kmers_total,
        n_kmers_kept=result.n_kmers_kept,
        n_patterns=n_patterns,
        kmer_set_sha1=sha1,
        matrix_path=matrix_path,
        rows_path=rows_path,
        index_path=index_path,
        kmers_path=kmers_path,
        rows=rows,
        index=_light_index(kmer_set, result.train_frequency),
    )


def run(
    paths: Paths,
    config: object | None,
    species: Iterable[str] | None = None,
    backend: str | UnitigBackend = "kmer",
    *,
    max_kmer_genomes: int = DEFAULT_MAX_KMER_GENOMES,
    threads: int | None = None,
) -> pd.DataFrame:
    """Stage entry point: build unitig features for every species with training genomes.

    Args:
        paths: Project paths.
        config: Loaded ``Config`` (restricts species to configured keys when given).
        species: Species keys to build; default = every species in ``splits.parquet``.
        backend: Backend name (``kmer``, ``unitig-caller``, ``auto``) or instance.
        max_kmer_genomes: Size guard of the pure-Python k-mer backend: a species
            with more training genomes raises :class:`TooManyGenomesForKmerBackend`
            (CLI ``--unitig-max-kmer-genomes``). Ignored for a backend instance.
        threads: Worker processes for per-genome work and ``unitig-caller
            --threads`` (``None`` = all cores; CLI ``--unitig-threads``).

    Returns:
        One summary row per species (``species, backend, n_train, n_queried,
        n_kmers_total, n_kmers_kept, n_patterns, kmer_set_sha1``). Species without
        training genomes are skipped with a warning. Writes
        ``drop_log_unitigs.csv``.
    """
    if isinstance(backend, str):
        be = make_backend(backend, threads=threads, max_kmer_genomes=max_kmer_genomes)
    else:
        be = backend
    log = DropLog(STAGE)
    if species is None:
        found = read_parquet(paths.splits, columns=["species"])["species"].dropna().unique().tolist()
        keys = sorted(str(s).upper() for s in found)
        if config is not None and hasattr(config, "species"):
            unknown = [k for k in keys if k not in config.species]
            if unknown:
                logger.warning("skipping species not in config: %s", unknown)
            keys = [k for k in keys if k in config.species]
    else:
        keys = [s.strip().upper() for s in species]

    records: list[dict[str, object]] = []
    for key in keys:
        probe = DropLog(STAGE)
        train, _query, _ = training_and_query_genomes(paths, key, probe)
        if not train:
            logger.warning("%s: no QC-passing training genomes; skipping unitig build", key)
            log.extend(probe)
            records.append({"species": key, "backend": be.name, "n_train": 0, "n_queried": 0,
                            "n_kmers_total": 0, "n_kmers_kept": 0, "n_patterns": 0, "kmer_set_sha1": None})
            continue
        if isinstance(be, KmerBackend):
            be.check_size(len(train))  # fail before any FASTA is read
        summary = build(paths, config, key, backend=be, droplog=log, threads=threads)
        records.append(summary.as_record())
    log.write(paths.drop_log(STAGE))
    frame = pd.DataFrame.from_records(
        records,
        columns=["species", "backend", "n_train", "n_queried", "n_kmers_total",
                 "n_kmers_kept", "n_patterns", "kmer_set_sha1"],
    )
    logger.info("unitigs stage summary:\n%s", frame.to_string(index=False))
    return frame


def load_unitigs(
    paths: Paths,
    species: str,
    *,
    index_columns: Sequence[str] | None = LOAD_INDEX_COLUMNS,
) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load ``(matrix, rows, index)`` for a species (matrix stays sparse CSR int8).

    Args:
        paths: Project paths.
        species: Species key.
        index_columns: Index columns to read. The default skips
            ``unitig_sequences`` (one decoded string per member k-mer -- by far
            the largest column, and not needed for training); ``None`` reads all.
    """
    key = species.strip().upper()
    matrix = sp.csr_matrix(sp.load_npz(paths.unitigs(key)), dtype=np.int8)
    rows = read_parquet(paths.unitig_rows(key))
    index = read_parquet(paths.unitig_index(key), columns=None if index_columns is None else list(index_columns))
    if matrix.shape[0] != len(rows):
        raise ContractViolation(f"{key}: matrix has {matrix.shape[0]} rows but rows.parquet has {len(rows)}", STAGE)
    if matrix.shape[1] != len(index):
        raise ContractViolation(f"{key}: matrix has {matrix.shape[1]} columns but index.parquet has {len(index)}", STAGE)
    return matrix, rows, index
