"""Thin file helpers: Parquet and CSV tables, FASTA records.

Conventions (CLAUDE.md): Parquet via pyarrow for every processed table; CSV only
for raw downloads and small summaries; parent directories are created on write.
FASTA sequences are returned upper-cased with all whitespace removed, so callers
can index and k-merize them directly.
"""

from __future__ import annotations

import gzip
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO

import pandas as pd

logger = logging.getLogger(__name__)

FastaRecord = tuple[str, str]
"""``(header, sequence)`` -- header without the leading ``>``, sequence upper-case."""


# ------------------------------------------------------------------- directories
def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


# ------------------------------------------------------------------------ tables
def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet file with pyarrow, optionally selecting ``columns``."""
    frame = pd.read_parquet(Path(path), engine="pyarrow", columns=columns)
    logger.debug("read %d rows x %d cols from %s", len(frame), frame.shape[1], path)
    return frame


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``frame`` as Parquet (pyarrow, no index), creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, engine="pyarrow", index=False)
    logger.debug("wrote %d rows x %d cols to %s", len(frame), frame.shape[1], target)
    return target


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    """``pd.read_csv`` wrapper; keyword arguments pass through."""
    return pd.read_csv(Path(path), **kwargs)  # type: ignore[arg-type]


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``frame`` as CSV without the index, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    logger.debug("wrote %d rows to %s", len(frame), target)
    return target


# ------------------------------------------------------------------------- FASTA
def _open_text(path: Path, mode: str) -> IO[str]:
    """Open plain or gzip-compressed text depending on the ``.gz`` suffix."""
    target = Path(path)
    if target.suffix == ".gz":
        return gzip.open(target, mode + "t", encoding="utf-8")  # type: ignore[return-value]
    return open(target, mode, encoding="utf-8")


def iter_fasta(path: Path) -> Iterator[FastaRecord]:
    """Yield ``(header, sequence)`` records from a FASTA file (``.gz`` accepted).

    The header is the line after ``>`` with surrounding whitespace stripped (the
    full description, not just the first token). Sequence lines are upper-cased
    and every whitespace character (spaces, tabs, ``\\r``) is removed; blank lines
    are ignored. A sequence line before the first header raises ``ValueError``.
    """
    header: str | None = None
    chunks: list[str] = []
    with _open_text(path, "r") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
                continue
            if header is None:
                raise ValueError(f"{path}: sequence data before the first '>' header (line {line_no})")
            chunks.append("".join(line.split()).upper())
    if header is not None:
        yield header, "".join(chunks)


def read_fasta(path: Path) -> list[FastaRecord]:
    """All records of a FASTA file as a list of ``(header, upper-case sequence)``."""
    records = list(iter_fasta(path))
    logger.debug("read %d FASTA records from %s", len(records), path)
    return records


def write_fasta(records: Iterable[FastaRecord], path: Path, line_width: int = 80) -> Path:
    """Write ``(header, sequence)`` records as FASTA, wrapping at ``line_width``.

    ``line_width <= 0`` writes each sequence on one line. Parent directories are
    created. ``.gz`` paths are gzip-compressed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open_text(target, "w") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            if line_width > 0:
                for start in range(0, len(seq), line_width):
                    handle.write(seq[start : start + line_width] + "\n")
            else:
                handle.write(seq + "\n")
            n += 1
    logger.debug("wrote %d FASTA records to %s", n, target)
    return target
