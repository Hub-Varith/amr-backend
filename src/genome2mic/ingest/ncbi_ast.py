"""Reader for the NCBI AST browser export (``data/raw/ast_ncbi.csv``).

Returns the common raw schema (:data:`genome2mic.ingest.harmonize.RAW_COLUMNS`).
NCBI rows are lab measurements by definition, so ``evidence`` defaults to
``Laboratory Method`` when the export has no such column. ``genome_id`` is
``NCBI_<biosample>`` unless ``genome_metadata.csv`` maps the biosample to a BV-BRC
genome id, which the contract prefers. ``country`` is the part of ``geo_loc_name``
before the first ``:``; ``year`` is the four-digit year in ``collection_date``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from genome2mic.ingest.harmonize import (
    LAB_EVIDENCE,
    SOURCE_NCBI,
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

GENOME_ID_PREFIX = "NCBI_"

COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "biosample": ("biosample", "biosample_accession", "sample"),
    "organism": ("organism", "scientific_name", "organism_name", "species"),
    "antibiotic": ("antibiotic", "antibiotic_name"),
    "resistance_phenotype": ("resistance_phenotype", "resistant_phenotype", "phenotype"),
    "measurement_sign": ("measurement_sign", "sign"),
    "measurement": ("measurement", "measurement_value", "mic_value", "mic"),
    "measurement_units": ("measurement_units", "measurement_unit", "units", "unit"),
    "laboratory_typing_method": ("laboratory_typing_method", "lab_typing_method", "typing_method", "method"),
    "testing_standard": ("testing_standard", "standard"),
    "testing_standard_year": ("testing_standard_year", "standard_year"),
    "evidence": ("evidence",),
    "isolation_source": ("isolation_source",),
    "geo_loc_name": ("geo_loc_name", "location", "geographic_location", "country"),
    "collection_date": ("collection_date", "collection_year", "year"),
}
"""Canonical NCBI field -> accepted normalized header names (``#BioSample`` -> ``biosample``)."""

REQUIRED: tuple[str, ...] = ("biosample", "antibiotic")


def read_raw(
    path: Path,
    metadata: pd.DataFrame | None = None,
    species_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read an NCBI AST export into the common raw schema.

    Args:
        path: ``ast_ncbi.csv``.
        metadata: Optional ``genome_metadata.csv`` table; joined on ``biosample``.
            A mapped ``genome_id`` replaces the ``NCBI_<biosample>`` fallback; other
            columns only fill nulls.
        species_names: Species key -> binomial name for the organism mapping.
            Defaults to the contract's five species.

    Raises:
        ValueError: if ``biosample`` or ``antibiotic`` cannot be found.
    """
    path = Path(path)
    table = read_csv_text(path)
    columns = select_columns(table, COLUMN_ALIASES, REQUIRED, what=str(path))

    biosample = columns["biosample"]
    genome_id = biosample.map(lambda sample: None if sample is None else f"{GENOME_ID_PREFIX}{sample}")
    species = columns["organism"].map(lambda name: species_key_from_organism(name, species_names))
    evidence = columns["evidence"].map(lambda value: LAB_EVIDENCE if value is None else value)
    raw = pd.DataFrame(
        {
            "genome_id": genome_id,
            "biosample": biosample,
            "species": species,
            "antibiotic_raw": columns["antibiotic"],
            "sir_raw": columns["resistance_phenotype"],
            "sign": columns["measurement_sign"],
            "value": columns["measurement"],
            "unit": columns["measurement_units"],
            "method_raw": columns["laboratory_typing_method"],
            "standard": columns["testing_standard"],
            "standard_year": columns["testing_standard_year"],
            "evidence": evidence,
            "source": pd.Series([SOURCE_NCBI] * len(table), index=table.index, dtype=object),
            "isolation_source": columns["isolation_source"],
            "country": columns["geo_loc_name"].map(country_from_text),
            "year": columns["collection_date"].map(year_from_text),
        }
    )
    raw = apply_metadata(raw, metadata, on="biosample")
    raw = finalize_raw(raw)

    n_no_biosample = int(raw["biosample"].isna().sum())
    if n_no_biosample:
        logger.warning("%s: %d rows have no biosample and therefore no genome_id", path.name, n_no_biosample)
    unknown = unknown_organism_summary(columns["organism"], raw["species"])
    if unknown is not None:
        logger.info(
            "%s: %d rows with an organism outside scope (species left null): %s",
            path.name,
            int(raw["species"].isna().sum()),
            unknown,
        )
    logger.info(
        "%s: read %d rows for %d biosamples (%d re-keyed to a metadata genome_id)",
        path.name,
        len(raw),
        raw["biosample"].nunique(),
        int((~raw["genome_id"].fillna("").str.startswith(GENOME_ID_PREFIX) & raw["genome_id"].notna()).sum()),
    )
    return raw


__all__ = ["COLUMN_ALIASES", "GENOME_ID_PREFIX", "REQUIRED", "read_raw"]
