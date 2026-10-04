"""Assembly stats and QC rules for one uploaded genome (DATA_CONTRACT.md stage 3)."""

import gzip
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from genome2mic.qc.constants import GENOME_SIZE_TOLERANCE, MAX_CONTIGS, MAX_MASH_DISTANCE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyStats:
    """The qc.parquet stats columns, computed in Python (training used seqkit; same definitions)."""

    n_contigs: int
    total_length: int
    n50: int
    gc_percent: float

    @classmethod
    def from_fasta(cls, path: Path) -> "AssemblyStats":
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            return cls.from_sequences(cls._contigs(handle))

    @staticmethod
    def _contigs(lines: Iterable[str]) -> Iterable[str]:
        parts: list[str] = []
        for line in lines:
            if line.startswith(">"):
                if parts:
                    yield "".join(parts)
                parts = []
            else:
                parts.append(line.strip())
        if parts:
            yield "".join(parts)

    @classmethod
    def from_sequences(cls, contigs: Iterable[str]) -> "AssemblyStats":
        lengths, gc, called = [], 0, 0
        for contig in contigs:
            upper = contig.upper()
            lengths.append(len(upper))
            gc += upper.count("G") + upper.count("C")
            called += gc_and_at(upper)
        total = sum(lengths)
        return cls(len(lengths), total, n50(lengths), 100.0 * gc / called if called else 0.0)


def gc_and_at(sequence: str) -> int:
    """Bases that count toward GC% (N and other ambiguity codes do not)."""
    return sum(sequence.count(base) for base in "ACGT")


def n50(lengths: list[int]) -> int:
    """Length of the contig at which the sorted cumulative length first reaches half the total."""
    half, running = sum(lengths) / 2, 0
    for length in sorted(lengths, reverse=True):
        running += length
        if running >= half:
            return length
    return 0


@dataclass(frozen=True)
class QcResult:
    stats: AssemblyStats
    qc_pass: bool
    qc_fail_reason: str | None


class GenomeQc:
    """Stage 3 rules. A QC failure is reported on the report, not fatal (INTEGRATION.md 3.1)."""

    def __init__(self, expected_sizes: dict[str, int]) -> None:
        self.expected_sizes = expected_sizes

    def evaluate(self, stats: AssemblyStats, species: str | None, mash_distance: float | None) -> QcResult:
        if stats.n_contigs == 0:
            return QcResult(stats, False, "empty_assembly")
        reasons = []
        if stats.n_contigs > MAX_CONTIGS:
            reasons.append("too_fragmented")
        if species is None:
            reasons.append("species_not_covered")
        elif abs(stats.total_length / self.expected_sizes[species] - 1) > GENOME_SIZE_TOLERANCE:
            reasons.append("wrong_genome_size")
        if species is not None and mash_distance is not None and mash_distance > MAX_MASH_DISTANCE:
            reasons.append("too_distant")
        result = QcResult(stats, not reasons, ";".join(reasons) or None)
        logger.info("Genome QC", extra={"qc_pass": result.qc_pass, "reason": result.qc_fail_reason, "n_contigs": stats.n_contigs})
        return result
