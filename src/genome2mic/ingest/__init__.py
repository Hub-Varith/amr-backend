"""Stage 1 + 2: ingest raw AST downloads and harmonize them into ``labels.parquet``.

::

    data/raw/ast_bvbrc.csv  --bvbrc.read_raw-->  common raw schema  \\
                                                                      +--> harmonize --> labels.parquet
    data/raw/ast_ncbi.csv   --ncbi_ast.read_raw-> common raw schema  /                   label_counts.csv
    data/raw/genome_metadata.csv (optional)  fills biosample / species / metadata        pairs_kept.csv
                                                                                          drop_log_ingest.csv

:func:`run` is the stage entry point used by ``python -m genome2mic ingest``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from genome2mic.config import Config
from genome2mic.droplog import DropLog
from genome2mic.ingest import bvbrc, harmonize, ncbi_ast

# ``harmonize`` above is the *module* (DESIGN.md: ``harmonize.harmonize(...)``); the
# function itself is deliberately not re-exported under the same name.
from genome2mic.ingest.harmonize import (
    COUNT_COLUMNS,
    LABEL_COLUMNS,
    LABEL_TEXT_COLUMNS,
    PAIRS_KEPT_COLUMNS,
    RAW_COLUMNS,
    Reason,
    check_labels,
    count_table,
    finalize_raw,
    pairs_kept,
    read_genome_metadata,
)
from genome2mic.io import write_csv, write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "ingest"

_READERS: tuple[tuple[str, Callable[..., pd.DataFrame]], ...] = (
    ("bvbrc", bvbrc.read_raw),
    ("ncbi", ncbi_ast.read_raw),
)


def load_raw(paths: Paths, config: Config) -> pd.DataFrame:
    """Read every raw AST file that exists and return one common-raw-schema frame.

    A missing raw file is skipped with a warning; if none exists this raises
    ``FileNotFoundError``. ``genome_metadata.csv`` is used when present.
    """
    species_names = {key: spec.name for key, spec in config.species.items()}
    metadata = None
    if paths.genome_metadata.is_file():
        metadata = read_genome_metadata(paths.genome_metadata)
    else:
        logger.info("no genome metadata file at %s; biosamples come from the raw files only", paths.genome_metadata)

    frames: list[pd.DataFrame] = []
    for source, reader in _READERS:
        path = paths.raw_ast(source)
        if not path.is_file():
            logger.warning("raw AST file not found, skipping source %s: %s", source.upper(), path)
            continue
        frame = reader(path, metadata=metadata, species_names=species_names)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no raw AST files found; expected at least one of "
            f"{[str(paths.raw_ast(source)) for source, _ in _READERS]}"
        )
    raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    raw = finalize_raw(raw)
    logger.info("loaded %d raw AST rows from %d source file(s)", len(raw), len(frames))
    return raw


def labels_for_parquet(labels: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``labels`` with text columns as pandas ``string`` so Parquet gets a
    typed string column even when a text column is entirely null."""
    return labels.astype({column: "string" for column in LABEL_TEXT_COLUMNS if column in labels.columns})


def run(
    paths: Paths,
    config: Config,
    *,
    min_ns: int = 50,
    min_s: int = 50,
    min_levels: int = 4,
) -> pd.DataFrame:
    """Run the ingest stage and write its outputs under ``paths.processed_dir``.

    Writes ``labels.parquet`` (stage-2 contract), ``label_counts.csv``,
    ``pairs_kept.csv`` and ``drop_log_ingest.csv``; returns the labels frame.
    The pair-inclusion thresholds default to the contract's 50 / 50 / 4.
    """
    droplog = DropLog(STAGE)
    raw = load_raw(paths, config)
    labels = harmonize.harmonize(raw, config, droplog)
    counts = count_table(labels)
    kept = pairs_kept(counts, min_ns=min_ns, min_s=min_s, min_levels=min_levels)

    write_parquet(labels_for_parquet(labels), paths.labels)
    write_csv(counts, paths.label_counts)
    write_csv(kept, paths.pairs_kept)
    droplog.write(paths.drop_log(STAGE))

    logger.info("wrote %d labels to %s", len(labels), paths.labels)
    logger.info("label counts (species x drug):\n%s", counts.to_string(index=False) if not counts.empty else "(none)")
    logger.info(
        "pairs kept (%d): %s",
        len(kept),
        ", ".join(f"{s}/{d}" for s, d in zip(kept["species"], kept["drug"])) or "(none)",
    )
    return labels


__all__ = [
    "COUNT_COLUMNS",
    "LABEL_COLUMNS",
    "PAIRS_KEPT_COLUMNS",
    "RAW_COLUMNS",
    "STAGE",
    "Reason",
    "bvbrc",
    "check_labels",
    "count_table",
    "harmonize",  # the submodule
    "labels_for_parquet",
    "load_raw",
    "ncbi_ast",
    "pairs_kept",
    "read_genome_metadata",
    "run",
]
