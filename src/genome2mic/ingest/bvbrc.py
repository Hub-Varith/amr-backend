"""Reader for the BV-BRC AST export (``data/raw/ast_bvbrc.csv``).

Returns the common raw schema (:data:`genome2mic.ingest.harmonize.RAW_COLUMNS`).
No filtering happens here -- ``evidence`` passes through untouched so that the
``Laboratory Method`` filter is one visible, counted step in :func:`harmonize`.
Species is mapped from ``genome_name`` (``Klebsiella pneumoniae subsp. ...`` ->
``KPNEU``); organisms outside scope are left null and logged so the drop shows up
in the harmonize drop log.

BV-BRC exports vary slightly in column names; :data:`COLUMN_ALIASES` lists the
accepted spellings (after header normalization) in preference order.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from genome2mic.ingest.harmonize import (
    SOURCE_BVBRC,
    apply_metadata,
    country_from_text,
    finalize_raw,
    read_csv_text,
    select_columns,
    species_key_from_organism,
    unknown_organism_summary,
    year_from_text,
)

logger = logging.getLogger(__name__)

COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "genome_id": ("genome_id",),
    "genome_name": ("genome_name", "organism", "organism_name", "species_name", "scientific_name"),
    "antibiotic": ("antibiotic", "antibiotic_name"),
    "resistant_phenotype": ("resistant_phenotype", "resistance_phenotype", "phenotype"),
    "measurement_sign": ("measurement_sign", "sign"),
    "measurement_value": ("measurement_value", "measurement", "mic_value", "mic"),
    "measurement_unit": ("measurement_unit", "measurement_units", "unit", "units"),
    "laboratory_typing_method": ("laboratory_typing_method", "lab_typing_method", "typing_method", "method"),
    "testing_standard": ("testing_standard", "standard"),
    "testing_standard_year": ("testing_standard_year", "standard_year"),
    "evidence": ("evidence",),
    "biosample": ("biosample", "biosample_accession", "biosample_id"),
    "isolation_source": ("isolation_source",),
    "country": ("isolation_country", "country"),
    "year": ("collection_year", "collection_date", "year"),
}
"""Canonical BV-BRC field -> accepted normalized header names."""

REQUIRED: tuple[str, ...] = ("genome_id", "antibiotic")


def read_raw(
    path: Path,
    metadata: pd.DataFrame | None = None,
    species_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read a BV-BRC AST export into the common raw schema.

    Args:
        path: ``ast_bvbrc.csv``.
        metadata: Optional ``genome_metadata.csv`` table (see
            :func:`genome2mic.ingest.harmonize.read_genome_metadata`); fills
            ``biosample``, ``species``, ``isolation_source``, ``country`` and ``year``
            where the export lacks them, joined on ``genome_id``.
        species_names: Species key -> binomial name used for the organism mapping.
            Defaults to the contract's five species.

    Raises:
        ValueError: if ``genome_id`` or ``antibiotic`` cannot be found.
    """
    path = Path(path)
    table = read_csv_text(path)
    columns = select_columns(table, COLUMN_ALIASES, REQUIRED, what=str(path))

    species = columns["genome_name"].map(lambda name: species_key_from_organism(name, species_names))
    raw = pd.DataFrame(
        {
            "genome_id": columns["genome_id"],
            "biosample": columns["biosample"],
            "species": species,
            "antibiotic_raw": columns["antibiotic"],
            "sir_raw": columns["resistant_phenotype"],
            "sign": columns["measurement_sign"],
            "value": columns["measurement_value"],
            "unit": columns["measurement_unit"],
            "method_raw": columns["laboratory_typing_method"],
            "standard": columns["testing_standard"],
            "standard_year": columns["testing_standard_year"],
            "evidence": columns["evidence"],
            "source": pd.Series([SOURCE_BVBRC] * len(table), index=table.index, dtype=object),
            "isolation_source": columns["isolation_source"],
            "country": columns["country"].map(country_from_text),
            "year": columns["year"].map(year_from_text),
        }
    )
    raw = apply_metadata(raw, metadata, on="genome_id")
    raw = finalize_raw(raw)

    unknown = unknown_organism_summary(columns["genome_name"], raw["species"])
    if unknown is not None:
        logger.info(
            "%s: %d rows with an organism outside scope (species left null): %s",
            path.name,
            int(raw["species"].isna().sum()),
            unknown,
        )
    logger.info(
        "%s: read %d rows for %d genomes (%d with a biosample)",
        path.name,
        len(raw),
        raw["genome_id"].nunique(),
        int(raw["biosample"].notna().sum()),
    )
    return raw


__all__ = ["COLUMN_ALIASES", "REQUIRED", "read_raw"]
