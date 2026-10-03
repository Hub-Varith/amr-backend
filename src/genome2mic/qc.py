"""Stage 3 -- assembly QC and species identification (``data/processed/qc.parquet``).

For every genome in ``labels.parquet`` or ``data/raw/genomes/`` this module computes
assembly statistics from the FASTA, identifies the species by Mash distance to the
species references, applies the four fail rules from ``DATA_CONTRACT.md`` stage 3
and writes one row per genome.

Species identification backends, tried in this order per genome:

1. ``data/interim/<genome_id>/mash.tsv`` -- ``mash dist`` output with columns
   ``reference, query, distance, p_value, shared_hashes`` (a header line is
   optional). The reference file name ``<SPECIES>.fasta`` (``.fasta``, ``.fa``,
   ``.fna``, ``.fas``, ``.msh``, optionally ``.gz``) is mapped to the species key;
   the nearest recognised reference wins. Unrecognised references are ignored.
2. Pure-Python fallback: sketch the genome with :mod:`genome2mic.sketch` and compare
   it with reference sketches built from ``data/raw/references/<SPECIES>.fasta`` (or
   loaded from the ``reference_sketch`` path in ``species.yaml`` when it is an
   ``.npz`` written by :func:`genome2mic.sketch.save_sketches`).
3. Neither available -> ``mash_species`` and ``mash_distance`` are null.

Fail rules. ``qc_fail_reason`` lists every failing rule, ``;``-joined, in this order:

====================  ===========================================================
``too_fragmented``    ``n_contigs > qc.max_contigs``
``wrong_size``        ``total_length`` outside ``expected_genome_size`` +/-
                      ``size_tolerance`` for the labelled species (or the Mash
                      species when the genome has no label)
``species_mismatch``  ``mash_species != species`` from the label table. A labelled
                      genome with no Mash result cannot be confirmed and fails this
                      rule as well (conservative; the drop log says so).
``too_distant``       ``mash_distance > qc.max_mash_distance`` from every reference.
                      A missing distance also fails (conservative).
====================  ===========================================================

A genome listed in ``labels.parquet`` that has no FASTA gets ``qc_pass = False`` and
``qc_fail_reason = 'missing_fasta'``; its statistics are null.

``gc_percent`` is ``100 * (G + C) / (A + C + G + T)``; ambiguous bases are excluded
from both numerator and denominator. ``n50`` is the length of the shortest contig in
the smallest set of longest contigs that together cover at least half of
``total_length``.

The ``species`` column is the species the rules were evaluated against: the label
species when the genome is in ``labels.parquet``, otherwise the Mash species. It is
one column beyond the contract's stage 3 list; it makes the expected-size check
auditable and lets stage 5 take species from ``qc.parquet`` directly.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from genome2mic import sketch as sk
from genome2mic.config import Config
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.io import iter_fasta, read_parquet, write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "qc"

RULE_TOO_FRAGMENTED = "too_fragmented"
RULE_WRONG_SIZE = "wrong_size"
RULE_SPECIES_MISMATCH = "species_mismatch"
RULE_TOO_DISTANT = "too_distant"
RULE_MISSING_FASTA = "missing_fasta"

FAIL_RULES: tuple[str, ...] = (
    RULE_TOO_FRAGMENTED,
    RULE_WRONG_SIZE,
    RULE_SPECIES_MISMATCH,
    RULE_TOO_DISTANT,
)
"""The four contract rules, in ``qc_fail_reason`` order."""

ALL_REASONS: frozenset[str] = frozenset(FAIL_RULES) | {RULE_MISSING_FASTA}
REASON_SEPARATOR = ";"

QC_COLUMNS: tuple[str, ...] = (
    "genome_id",
    "species",
    "n_contigs",
    "total_length",
    "n50",
    "gc_percent",
    "mash_species",
    "mash_distance",
    "qc_pass",
    "qc_fail_reason",
)
"""Output column order. Contract columns plus ``species`` (see module docstring)."""

MASH_TSV_NAME = "mash.tsv"
MASH_TSV_COLUMNS: tuple[str, ...] = ("reference", "query", "distance", "p_value", "shared_hashes")

BACKEND_MASH_TSV = "mash_tsv"
BACKEND_SKETCH = "sketch"
BACKEND_NONE = "none"

_REFERENCE_SUFFIXES: tuple[str, ...] = (".gz", ".fasta", ".fa", ".fna", ".fas", ".msh")
_GENOME_GLOB = "*.fasta"
_ID_PREVIEW = 10


# --------------------------------------------------------------------------- #
# Assembly statistics
# --------------------------------------------------------------------------- #


def n50(lengths: Iterable[int]) -> int | None:
    """N50 of a set of contig lengths.

    Sort lengths descending and accumulate; N50 is the length at which the running
    total first reaches half of the total length. ``None`` (null) when there are no
    contigs or the total length is zero, because N50 is undefined there.

    >>> n50([10, 20, 30, 40])
    30
    """
    arr = np.asarray(list(lengths), dtype=np.int64)
    if arr.size == 0:
        return None
    total = int(arr.sum())
    if total <= 0:
        return None
    desc = np.sort(arr)[::-1]
    cumulative = np.cumsum(desc)
    index = int(np.argmax(cumulative >= total / 2.0))
    return int(desc[index])


def assembly_stats(fasta: Path) -> dict[str, int | float | None]:
    """Contig count, total length, N50 and GC% of one FASTA assembly.

    Streams the file; sequences are upper-cased by :func:`genome2mic.io.iter_fasta`.
    ``gc_percent`` is null when the assembly has no unambiguous base and ``n50`` is
    null when it has no sequence at all.
    """
    lengths: list[int] = []
    gc = 0
    acgt = 0
    for _header, seq in iter_fasta(Path(fasta)):
        lengths.append(len(seq))
        g = seq.count("G")
        c = seq.count("C")
        gc += g + c
        acgt += g + c + seq.count("A") + seq.count("T")
    total = int(sum(lengths))
    stats: dict[str, int | float | None] = {
        "n_contigs": len(lengths),
        "total_length": total,
        "n50": n50(lengths),
        "gc_percent": (100.0 * gc / acgt) if acgt else None,
    }
    logger.debug("assembly_stats %s: %s", fasta, stats)
    return stats


# --------------------------------------------------------------------------- #
# Species identification
# --------------------------------------------------------------------------- #


class SpeciesCall(NamedTuple):
    """Result of species identification for one genome."""

    mash_species: str | None
    mash_distance: float | None
    backend: str
    """One of ``mash_tsv``, ``sketch``, ``none``."""


def reference_to_species(reference: str, species_keys: Iterable[str]) -> str | None:
    """Map a Mash reference file name to a species key, or ``None`` if unknown.

    ``data/raw/references/KPNEU.fasta`` -> ``KPNEU``; ``kpneu.fna.gz`` -> ``KPNEU``;
    ``KPNEU.msh`` -> ``KPNEU``; ``GCF_000240185.fasta`` -> ``None``.
    """
    keys = {str(k) for k in species_keys}
    name = Path(str(reference).strip()).name.lower()
    stripped = True
    while stripped:
        stripped = False
        for suffix in _REFERENCE_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name = name[: -len(suffix)]
                stripped = True
    key = name.upper()
    return key if key in keys else None


def read_mash_tsv(path: Path) -> list[tuple[str, float]]:
    """``(reference, distance)`` pairs from a ``mash dist`` table.

    Accepts the headerless ``mash dist`` output and the same table with a header
    line (``reference`` in the first cell). Rows with fewer than three fields or a
    non-numeric / negative distance are skipped with a warning.
    """
    out: list[tuple[str, float]] = []
    first = True
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if first:
                first = False
                if row[0].strip().lower() == MASH_TSV_COLUMNS[0]:
                    continue
            if len(row) < 3:
                logger.warning("%s line %d: expected >= 3 tab-separated fields, got %d", path, line_no, len(row))
                continue
            try:
                distance = float(row[2])
            except ValueError:
                logger.warning("%s line %d: distance %r is not a number", path, line_no, row[2])
                continue
            if math.isnan(distance) or distance < 0:
                logger.warning("%s line %d: invalid distance %r", path, line_no, row[2])
                continue
            out.append((row[0].strip(), distance))
    return out


def species_from_mash_tsv(path: Path, species_keys: Iterable[str]) -> tuple[str | None, float | None]:
    """Nearest species and its distance from ``mash.tsv``; ``(None, None)`` if no known reference."""
    keys = tuple(species_keys)
    best: tuple[str, float] | None = None
    for reference, distance in read_mash_tsv(path):
        key = reference_to_species(reference, keys)
        if key is None:
            logger.debug("%s: reference %r does not name a known species; ignored", path, reference)
            continue
        if best is None or distance < best[1]:
            best = (key, distance)
    if best is None:
        logger.warning("%s: no row names a known species reference", path)
        return None, None
    return best


@dataclass(frozen=True)
class ReferenceSketches:
    """MinHash sketches of the species reference genomes, one row per species."""

    species: tuple[str, ...]
    sketches: np.ndarray
    """``(n_species, s)`` sorted-unique ``uint64`` rows."""
    k: int

    @property
    def sketch_size(self) -> int:
        """Number of hashes per sketch (``s``)."""
        return int(self.sketches.shape[1])


def _load_reference_sketch(path: Path, k: int) -> np.ndarray | None:
    try:
        _ids, matrix, loaded_k = sk.load_sketches(path)
    except (OSError, ValueError) as error:
        logger.warning("reference_sketch %s could not be read (%s); falling back to the FASTA", path, error)
        return None
    if loaded_k != k:
        raise ValueError(f"reference sketch {path} was built with k={loaded_k}, expected k={k}")
    if matrix.shape[0] != 1:
        logger.warning("reference_sketch %s holds %d sketches; using the first", path, matrix.shape[0])
    return matrix[0]


def build_reference_sketches(
    paths: Paths,
    config: Config,
    *,
    k: int = sk.K,
    sketch_size: int = sk.SKETCH_SIZE,
) -> ReferenceSketches | None:
    """Sketch every species reference available on disk.

    Per species: ``reference_sketch`` from ``species.yaml`` when it is a readable
    ``.npz`` sketch file (relative paths resolve against ``paths.root``), else
    ``data/raw/references/<SPECIES>.fasta``. Species with neither are skipped with a
    warning. Returns ``None`` when no reference at all is available.

    All sketches are truncated to the smallest available size (a bottom-``s`` sketch
    cut to its first ``s'`` hashes is exactly the bottom-``s'`` sketch), so the
    stacked matrix is valid for :func:`genome2mic.sketch.distances_to`.
    """
    keys: list[str] = []
    rows: list[np.ndarray] = []
    for key, species in config.species.items():
        arr: np.ndarray | None = None
        if species.reference_sketch:
            sketch_path = Path(species.reference_sketch)
            if not sketch_path.is_absolute():
                sketch_path = paths.root / sketch_path
            if sketch_path.is_file():
                arr = _load_reference_sketch(sketch_path, k)
        if arr is None:
            fasta = paths.reference_fasta(key)
            if fasta.is_file():
                arr = sk.sketch((seq for _h, seq in iter_fasta(fasta)), k=k, s=sketch_size)
        if arr is None or arr.size == 0:
            logger.warning("no usable reference for species %s (looked for %s)", key, paths.reference_fasta(key))
            continue
        keys.append(key)
        rows.append(np.asarray(arr, dtype=np.uint64))
    if not rows:
        logger.warning("no species references found under %s; sketch-based species ID unavailable", paths.references_dir)
        return None
    size = min(int(r.size) for r in rows)
    matrix = np.vstack([r[:size] for r in rows])
    logger.info("built %d reference sketches (k=%d, s=%d): %s", len(keys), k, size, ", ".join(keys))
    return ReferenceSketches(species=tuple(keys), sketches=matrix, k=k)


def species_from_sketch(fasta: Path, references: ReferenceSketches) -> tuple[str | None, float | None]:
    """Nearest reference species by MinHash distance; ``(None, None)`` if the genome has no valid k-mer."""
    query = sk.sketch((seq for _h, seq in iter_fasta(Path(fasta))), k=references.k, s=references.sketch_size)
    if query.size == 0:
        logger.warning("%s: no valid k-mers; cannot identify species", fasta)
        return None, None
    distances = sk.distances_to(query, references.sketches, k=references.k)
    index = int(np.argmin(distances))
    return references.species[index], float(distances[index])


def species_from_mash(
    mash_tsv: Path | None,
    fasta: Path | None,
    species_keys: Iterable[str],
    references: ReferenceSketches | Callable[[], ReferenceSketches | None] | None,
) -> SpeciesCall:
    """Identify the species: ``mash.tsv`` first, then the sketch fallback, else nulls.

    ``references`` may be a :class:`ReferenceSketches`, a zero-argument callable
    returning one (built lazily on first use), or ``None``.
    """
    keys = tuple(species_keys)
    if mash_tsv is not None and Path(mash_tsv).is_file():
        species, distance = species_from_mash_tsv(Path(mash_tsv), keys)
        if species is not None:
            return SpeciesCall(species, distance, BACKEND_MASH_TSV)
    if fasta is not None and Path(fasta).is_file():
        refs = references() if callable(references) else references
        if refs is not None:
            species, distance = species_from_sketch(Path(fasta), refs)
            if species is not None:
                return SpeciesCall(species, distance, BACKEND_SKETCH)
    return SpeciesCall(None, None, BACKEND_NONE)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def expected_size_range(config: Config, species: str) -> tuple[float, float]:
    """``(low, high)`` genome length accepted for ``species`` (inclusive bounds)."""
    spec = config.species.get(species)
    if spec is None:
        raise ContractViolation(f"species {species!r} is not defined in species.yaml", stage=STAGE)
    size = float(spec.expected_genome_size)
    return size * (1.0 - spec.size_tolerance), size * (1.0 + spec.size_tolerance)


def evaluate_rules(
    *,
    n_contigs: int,
    total_length: int,
    species: str | None,
    mash_species: str | None,
    mash_distance: float | None,
    config: Config,
) -> list[str]:
    """Names of the contract rules a genome fails, in :data:`FAIL_RULES` order.

    Args:
        n_contigs, total_length: From :func:`assembly_stats`.
        species: Label species, or ``None`` when the genome is not in the label table.
        mash_species, mash_distance: From :func:`species_from_mash`; may be ``None``.
        config: Thresholds (``qc_max_contigs``, ``qc_max_mash_distance``) and sizes.

    Size is checked against the label species, or the Mash species for unlabelled
    genomes; with neither it cannot be checked. ``species_mismatch`` applies only to
    labelled genomes and also fires when there is no Mash result. ``too_distant``
    fires on a missing distance.
    """
    reasons: list[str] = []
    if n_contigs > config.qc_max_contigs:
        reasons.append(RULE_TOO_FRAGMENTED)
    size_species = species if species is not None else mash_species
    if size_species is not None:
        low, high = expected_size_range(config, size_species)
        if not (low <= total_length <= high):
            reasons.append(RULE_WRONG_SIZE)
    if species is not None and (mash_species is None or mash_species != species):
        reasons.append(RULE_SPECIES_MISMATCH)
    if mash_distance is None or math.isnan(mash_distance) or mash_distance > config.qc_max_mash_distance:
        reasons.append(RULE_TOO_DISTANT)
    return reasons


def join_reasons(reasons: Sequence[str]) -> str | None:
    """``;``-join rule names, or ``None`` when there are none (= pass)."""
    return REASON_SEPARATOR.join(reasons) if reasons else None


# --------------------------------------------------------------------------- #
# Per-genome and stage orchestration
# --------------------------------------------------------------------------- #


def qc_genome(
    genome_id: str,
    paths: Paths,
    config: Config,
    *,
    label_species: str | None,
    references: ReferenceSketches | Callable[[], ReferenceSketches | None] | None,
) -> dict[str, object]:
    """QC one genome and return its ``qc.parquet`` row (plus ``backend`` for logging)."""
    fasta = paths.genome_fasta(genome_id)
    row: dict[str, object] = {column: None for column in QC_COLUMNS}
    row["genome_id"] = genome_id
    row["species"] = label_species
    if not fasta.is_file():
        row["qc_pass"] = False
        row["qc_fail_reason"] = RULE_MISSING_FASTA
        row["backend"] = BACKEND_NONE
        return row

    stats = assembly_stats(fasta)
    call = species_from_mash(
        paths.interim_dir(genome_id) / MASH_TSV_NAME,
        fasta,
        config.species.keys(),
        references,
    )
    reasons = evaluate_rules(
        n_contigs=int(stats["n_contigs"]),
        total_length=int(stats["total_length"]),
        species=label_species,
        mash_species=call.mash_species,
        mash_distance=call.mash_distance,
        config=config,
    )
    row.update(stats)
    row["species"] = label_species if label_species is not None else call.mash_species
    row["mash_species"] = call.mash_species
    row["mash_distance"] = call.mash_distance
    row["qc_pass"] = not reasons
    row["qc_fail_reason"] = join_reasons(reasons)
    row["backend"] = call.backend
    return row


def read_label_species(paths: Paths) -> dict[str, str]:
    """``genome_id -> species`` from ``labels.parquet``; empty when the file is absent.

    Raises ``ContractViolation`` if a genome has a null species or more than one.
    """
    if not paths.labels.is_file():
        logger.warning("labels file %s not found; QC will cover the genomes directory only", paths.labels)
        return {}
    labels = read_parquet(paths.labels, columns=["genome_id", "species"]).drop_duplicates()
    if labels["species"].isna().any() or labels["genome_id"].isna().any():
        raise ContractViolation("labels.parquet has null genome_id or species", stage=STAGE)
    per_genome = labels.groupby("genome_id", sort=True)["species"].nunique()
    conflicts = per_genome[per_genome > 1]
    if len(conflicts):
        raise ContractViolation(
            f"{len(conflicts)} genome(s) carry more than one species in labels.parquet: "
            f"{list(conflicts.index[:_ID_PREVIEW])}",
            stage=STAGE,
        )
    return {str(g): str(s) for g, s in zip(labels["genome_id"], labels["species"], strict=True)}


def list_genome_fastas(paths: Paths) -> list[str]:
    """Genome ids with a FASTA under ``data/raw/genomes`` (``<genome_id>.fasta``)."""
    if not paths.genomes_dir.is_dir():
        return []
    return sorted(p.stem for p in paths.genomes_dir.glob(_GENOME_GLOB) if p.is_file())


def _to_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    def column(name: str) -> list[object]:
        return [row[name] for row in rows]

    return pd.DataFrame(
        {
            "genome_id": pd.array(column("genome_id"), dtype="str"),
            "species": pd.array(column("species"), dtype="str"),
            "n_contigs": pd.array(column("n_contigs"), dtype="Int64"),
            "total_length": pd.array(column("total_length"), dtype="Int64"),
            "n50": pd.array(column("n50"), dtype="Int64"),
            "gc_percent": pd.array(column("gc_percent"), dtype="Float64"),
            "mash_species": pd.array(column("mash_species"), dtype="str"),
            "mash_distance": pd.array(column("mash_distance"), dtype="Float64"),
            "qc_pass": np.array(column("qc_pass"), dtype=bool),
            "qc_fail_reason": pd.array(column("qc_fail_reason"), dtype="str"),
        },
        columns=list(QC_COLUMNS),
    )


def validate_qc(frame: pd.DataFrame) -> None:
    """Acceptance checks for ``qc.parquet``; raises ``ContractViolation``.

    * every column of :data:`QC_COLUMNS` present
    * ``genome_id`` non-null and unique
    * ``qc_pass`` boolean, non-null
    * ``qc_fail_reason`` null exactly when ``qc_pass`` is true, and otherwise a
      ``;``-joined list of known rule names
    * passing rows have non-null ``species`` and statistics
    """
    missing = [c for c in QC_COLUMNS if c not in frame.columns]
    if missing:
        raise ContractViolation(f"qc.parquet is missing columns {missing}", stage=STAGE)
    if frame["genome_id"].isna().any():
        raise ContractViolation("qc.parquet has null genome_id", stage=STAGE)
    duplicated = frame["genome_id"][frame["genome_id"].duplicated()]
    if len(duplicated):
        raise ContractViolation(f"qc.parquet has duplicate genome_id: {list(duplicated[:_ID_PREVIEW])}", stage=STAGE)
    if frame["qc_pass"].isna().any() or frame["qc_pass"].dtype != bool:
        raise ContractViolation("qc_pass must be a non-null boolean column", stage=STAGE)
    passing = frame["qc_pass"].to_numpy(dtype=bool)
    reason_null = frame["qc_fail_reason"].isna().to_numpy()
    if np.any(passing & ~reason_null):
        raise ContractViolation("qc_pass rows must have a null qc_fail_reason", stage=STAGE)
    if np.any(~passing & reason_null):
        raise ContractViolation("failing rows must carry a qc_fail_reason", stage=STAGE)
    for reason in frame.loc[~frame["qc_fail_reason"].isna(), "qc_fail_reason"]:
        parts = str(reason).split(REASON_SEPARATOR)
        unknown = [p for p in parts if p not in ALL_REASONS]
        if unknown or not parts:
            raise ContractViolation(f"unknown qc_fail_reason parts {unknown} in {reason!r}", stage=STAGE)
    for column in ("species", "n_contigs", "total_length", "n50", "mash_species", "mash_distance"):
        if frame.loc[passing, column].isna().any():
            raise ContractViolation(f"QC-passing rows must have non-null {column}", stage=STAGE)


def run(
    paths: Paths,
    config: Config,
    *,
    k: int = sk.K,
    sketch_size: int = sk.SKETCH_SIZE,
) -> pd.DataFrame:
    """Build and write ``qc.parquet`` plus ``drop_log_qc.csv``; return the table.

    Covers every genome in ``labels.parquet`` or ``data/raw/genomes``. Reference
    sketches for the fallback backend are built lazily, only if some genome lacks a
    usable ``mash.tsv``.
    """
    log = DropLog(STAGE)
    label_species = read_label_species(paths)
    unknown = sorted(set(label_species.values()) - set(config.species))
    if unknown:
        raise ContractViolation(f"labels.parquet species not in species.yaml: {unknown}", stage=STAGE)
    fasta_ids = list_genome_fastas(paths)
    genome_ids = sorted(set(label_species) | set(fasta_ids))
    if not genome_ids:
        raise FileNotFoundError(
            f"no genomes to QC: neither {paths.labels} nor FASTA files under {paths.genomes_dir} were found"
        )
    logger.info(
        "QC: %d genomes (%d in labels, %d FASTA files, %d in labels without FASTA)",
        len(genome_ids),
        len(label_species),
        len(fasta_ids),
        len(set(label_species) - set(fasta_ids)),
    )

    cache: dict[str, ReferenceSketches | None] = {}

    def references() -> ReferenceSketches | None:
        if "refs" not in cache:
            cache["refs"] = build_reference_sketches(paths, config, k=k, sketch_size=sketch_size)
        return cache["refs"]

    rows = [
        qc_genome(gid, paths, config, label_species=label_species.get(gid), references=references)
        for gid in genome_ids
    ]
    backends = Counter(str(row.pop("backend")) for row in rows)
    frame = _to_frame(rows)

    # Counts. Rules overlap, so per-rule counts can sum to more than the total.
    failing = frame.loc[~frame["qc_pass"], ["genome_id", "qc_fail_reason"]]
    reason_counts: Counter[str] = Counter()
    for reason in failing["qc_fail_reason"]:
        reason_counts.update(str(reason).split(REASON_SEPARATOR))
    log.drop(RULE_MISSING_FASTA, reason_counts[RULE_MISSING_FASTA], detail="in labels.parquet but no FASTA under data/raw/genomes")
    log.drop(RULE_TOO_FRAGMENTED, reason_counts[RULE_TOO_FRAGMENTED], detail=f"n_contigs > {config.qc_max_contigs}")
    log.drop(RULE_WRONG_SIZE, reason_counts[RULE_WRONG_SIZE], detail="total_length outside expected_genome_size +/- size_tolerance")
    log.drop(RULE_SPECIES_MISMATCH, reason_counts[RULE_SPECIES_MISMATCH], detail="mash_species != label species, or no Mash result for a labelled genome")
    log.drop(RULE_TOO_DISTANT, reason_counts[RULE_TOO_DISTANT], detail=f"mash_distance > {config.qc_max_mash_distance} from every reference, or no Mash result")
    log.drop(
        "qc_fail_any",
        int(len(failing)),
        detail="distinct genomes failing >= 1 rule (per-rule counts above overlap); excluded from training and the unitig build",
    )
    logger.info(
        "QC: n_total=%d n_pass=%d n_fail=%d; species backends: %s",
        len(frame),
        int(frame["qc_pass"].sum()),
        len(failing),
        dict(backends),
    )
    if backends.get(BACKEND_NONE, 0) and label_species:
        logger.warning(
            "%d genome(s) had no mash.tsv and no reference sketches; they fail species_mismatch/too_distant",
            backends[BACKEND_NONE],
        )

    validate_qc(frame)
    write_parquet(frame, paths.qc)
    log.write(paths.drop_log(STAGE))
    logger.info("wrote %s (%d rows)", paths.qc, len(frame))
    return frame


__all__ = [
    "ALL_REASONS",
    "FAIL_RULES",
    "MASH_TSV_COLUMNS",
    "QC_COLUMNS",
    "REASON_SEPARATOR",
    "ReferenceSketches",
    "SpeciesCall",
    "assembly_stats",
    "build_reference_sketches",
    "evaluate_rules",
    "expected_size_range",
    "join_reasons",
    "list_genome_fastas",
    "n50",
    "qc_genome",
    "read_label_species",
    "read_mash_tsv",
    "reference_to_species",
    "run",
    "species_from_mash",
    "species_from_mash_tsv",
    "species_from_sketch",
    "validate_qc",
]
