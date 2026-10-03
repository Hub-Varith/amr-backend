"""Acceptance checks for qc.parquet. See DATA_CONTRACT.md stage 3."""

import logging

import pandas as pd

from genome2mic.qc.constants import QC_COLUMNS

logger = logging.getLogger(__name__)


class QcValidator:
    """Runs every stage 3 check and reports all failures at once."""

    @staticmethod
    def failures(qc: pd.DataFrame) -> list[str]:
        """Return a list of failed checks. Empty means the table meets the contract."""
        if list(qc.columns) != QC_COLUMNS:
            return [f"columns differ from contract: {list(qc.columns)}"]
        problems = []
        if qc["genome_id"].duplicated().any():
            problems.append("genome_id is not unique")
        if qc["qc_pass"].isna().any() or qc["qc_pass"].dtype != bool:
            problems.append("qc_pass must be a non-null bool")
        if not (qc["qc_pass"] == qc["qc_fail_reason"].isna()).all():
            problems.append("qc_fail_reason must be null exactly when qc_pass is true")
        if (qc["n_contigs"] < 0).any():
            problems.append("negative n_contigs")
        return problems

    @staticmethod
    def check(qc: pd.DataFrame) -> None:
        """Raise ValueError listing every failed check."""
        problems = QcValidator.failures(qc)
        if problems:
            raise ValueError("qc.parquet failed acceptance checks:\n- " + "\n- ".join(problems))
        logger.info("qc.parquet passed all acceptance checks: rows=%s", len(qc))
