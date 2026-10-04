"""Applies the genome QC rules to assembly stats and Mash distances. See DATA_CONTRACT.md stage 3."""

import logging

import pandas as pd

from genome2mic.qc.constants import GENOME_SIZE_TOLERANCE, MAX_CONTIGS, MAX_MASH_DISTANCE, QC_COLUMNS

logger = logging.getLogger(__name__)


class AssemblyQc:
    """Turns per-genome stats and Mash distances into qc.parquet rows with every failure reason."""

    def __init__(self, expected_sizes: dict[str, int]) -> None:
        self.expected_sizes = expected_sizes

    def evaluate(self, manifest: pd.DataFrame, stats: pd.DataFrame, mash: pd.DataFrame) -> pd.DataFrame:
        """Return one row per manifest genome.

        `stats` has genome_id, n_contigs, total_length, n50, gc_percent.
        `mash` has genome_id, reference_species, distance (one row per reference).
        """
        logger.info("QC start: genomes=%s", len(manifest))
        nearest = mash.sort_values("distance").drop_duplicates(subset="genome_id")
        nearest = nearest.rename(columns={"reference_species": "mash_species", "distance": "mash_distance"})
        qc = manifest[["genome_id", "species"]].merge(stats, on="genome_id", how="left")
        qc = qc.merge(nearest[["genome_id", "mash_species", "mash_distance"]], on="genome_id", how="left")

        qc["n_contigs"] = qc["n_contigs"].fillna(0).astype(int)
        is_empty = qc["n_contigs"] == 0
        qc.loc[is_empty, ["total_length", "n50", "gc_percent", "mash_species", "mash_distance"]] = None

        expected = qc["species"].map(self.expected_sizes)
        size_ratio = qc["total_length"] / expected
        failed_checks = pd.DataFrame(
            {
                "too_fragmented": qc["n_contigs"] > MAX_CONTIGS,
                "wrong_genome_size": (size_ratio - 1).abs() > GENOME_SIZE_TOLERANCE,
                "no_mash_result": qc["mash_species"].isna(),
                "species_mismatch": qc["mash_species"].notna() & (qc["mash_species"] != qc["species"]),
                "too_distant": qc["mash_distance"] > MAX_MASH_DISTANCE,
            }
        )
        qc["qc_fail_reason"] = failed_checks.apply(lambda row: ";".join(row.index[row]) or None, axis=1)
        qc.loc[is_empty, "qc_fail_reason"] = "download_failed"
        qc["qc_pass"] = qc["qc_fail_reason"].isna()
        for column in ("total_length", "n50"):
            qc[column] = qc[column].astype("Int64")
        qc["gc_percent"] = qc["gc_percent"].astype(float)
        qc["mash_distance"] = qc["mash_distance"].astype(float)
        qc["mash_species"] = qc["mash_species"].astype(object).where(qc["mash_species"].notna(), None)

        reason_counts = qc["qc_fail_reason"].str.split(";").explode().value_counts()
        logger.info("QC done: pass=%s fail=%s reasons=%s", int(qc["qc_pass"].sum()), int((~qc["qc_pass"]).sum()),
                    reason_counts.to_dict())
        return qc[QC_COLUMNS].reset_index(drop=True)
