"""Acceptance checks for labels.parquet. See DATA_CONTRACT.md stage 2."""

import logging

import numpy as np
import pandas as pd

from genome2mic.ingest.constants import LABEL_COLUMNS

logger = logging.getLogger(__name__)

SPECIES_KEYS = {"ECOLI", "KPNEU", "SAUR", "PAER", "ABAU"}
NON_NULL_COLUMNS = ("genome_id", "species", "drug", "mic_lower", "mic_upper", "censor", "raw_result", "method", "source")
TEXT_COLUMNS = ("genome_id", "biosample", "species", "drug", "censor", "sir", "raw_result", "method", "standard",
                "source", "isolation_source", "country")
ALLOWED_VALUES = {
    "censor": {"interval", "left", "right"},
    "sir": {"S", "I", "R"},
    "method": {"dilution", "gradient", "disk"},
    "standard": {"CLSI", "EUCAST"},
    "source": {"BVBRC", "NCBI"},
    "species": SPECIES_KEYS,
}
PLACEHOLDERS = {"", "NA", "nan", "None", "-1"}


class LabelsValidator:
    """Runs every stage 2 acceptance check and reports all failures at once."""

    @staticmethod
    def failures(labels: pd.DataFrame) -> list[str]:
        """Return a list of failed checks. Empty means the table meets the contract."""
        problems = []
        if list(labels.columns) != list(LABEL_COLUMNS):
            problems.append(f"columns differ from contract: {list(labels.columns)}")
            return problems
        if labels.duplicated(subset=["genome_id", "drug"]).any():
            problems.append("(genome_id, drug) is not unique")
        for column in NON_NULL_COLUMNS:
            if labels[column].isna().any():
                problems.append(f"nulls in non-null column {column}")
        if not (labels["mic_lower"] < labels["mic_upper"]).all():
            problems.append("mic_lower < mic_upper fails on some rows")
        is_left = labels["censor"] == "left"
        if not (is_left == (labels["mic_lower"] == 0)).all():
            problems.append("censor == 'left' does not match mic_lower == 0")
        is_right = labels["censor"] == "right"
        if not (is_right == np.isinf(labels["mic_upper"])).all():
            problems.append("censor == 'right' does not match mic_upper == inf")
        for column, allowed in ALLOWED_VALUES.items():
            unexpected = set(labels[column].dropna()) - allowed
            if unexpected:
                problems.append(f"unexpected values in {column}: {sorted(unexpected)[:10]}")
        for column in TEXT_COLUMNS:
            placeholders = set(labels[column].dropna().astype(str).str.strip()) & PLACEHOLDERS
            if placeholders:
                problems.append(f"placeholder strings used for missing values in {column}: {sorted(placeholders)}")
        if str(labels["standard_year"].dtype) != "Int64" or str(labels["year"].dtype) != "Int64":
            problems.append("standard_year and year must be nullable integers")
        return problems

    @staticmethod
    def check(labels: pd.DataFrame) -> None:
        """Raise ValueError listing every failed check."""
        problems = LabelsValidator.failures(labels)
        if problems:
            raise ValueError("labels.parquet failed acceptance checks:\n- " + "\n- ".join(problems))
        logger.info("labels.parquet passed all acceptance checks: rows=%s", len(labels))
