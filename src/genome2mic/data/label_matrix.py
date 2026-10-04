"""Wide label tensor built from the long labels.parquet (DATA_CONTRACT.md stage 2)."""

import logging

import numpy as np
import pandas as pd

from genome2mic.data.constants import LABEL_COLUMNS, SPECIES_KEYS
from genome2mic.data.mic_steps import MicSteps

logger = logging.getLogger(__name__)


class LabelMatrix:
    """Genomes x drugs bounds in log2 steps, plus a mask that says where a lab result exists.

    A missing lab result is simply mask == False. It is never filled in (contract rule 1).
    """

    def __init__(
        self,
        genome_ids: np.ndarray,
        species: np.ndarray,
        drugs: list[str],
        lower_log2: np.ndarray,
        upper_log2: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self.genome_ids = genome_ids
        self.species = species
        self.drugs = drugs
        self.lower_log2 = lower_log2
        self.upper_log2 = upper_log2
        self.mask = mask
        self.species_index = np.array([SPECIES_KEYS.index(key) for key in species], dtype=np.int64)
        self.genome_position = {genome_id: position for position, genome_id in enumerate(genome_ids)}

    @classmethod
    def from_labels(cls, labels: pd.DataFrame, drugs: list[str] | None = None) -> "LabelMatrix":
        """Pivot the long table. Drugs default to every drug present, sorted."""
        missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
        if missing:
            raise ValueError(f"labels is missing columns {missing}")
        duplicated = labels.duplicated(subset=["genome_id", "drug"]).sum()
        if duplicated:
            raise ValueError(f"labels has {duplicated} duplicate (genome_id, drug) rows")
        if not (labels["mic_lower"] < labels["mic_upper"]).all():
            raise ValueError("labels has rows where mic_lower >= mic_upper")
        unknown_species = set(labels["species"]) - set(SPECIES_KEYS)
        if unknown_species:
            raise ValueError(f"labels has unknown species {sorted(unknown_species)}")

        if drugs is None:
            drugs = sorted(labels["drug"].unique())
        kept = labels[labels["drug"].isin(drugs)]
        logger.info(
            "LabelMatrix build",
            extra={"n_rows_in": len(labels), "n_rows_kept": len(kept), "n_drugs": len(drugs)},
        )

        genome_table = kept[["genome_id", "species"]].drop_duplicates("genome_id").sort_values("genome_id")
        if genome_table["genome_id"].duplicated().any():
            raise ValueError("a genome_id maps to more than one species")
        genome_ids = genome_table["genome_id"].to_numpy()
        species = genome_table["species"].to_numpy()

        n_genomes, n_drugs = len(genome_ids), len(drugs)
        lower_log2 = np.full((n_genomes, n_drugs), np.nan)
        upper_log2 = np.full((n_genomes, n_drugs), np.nan)
        mask = np.zeros((n_genomes, n_drugs), dtype=bool)

        row_positions = pd.Index(genome_ids).get_indexer(kept["genome_id"])
        col_positions = pd.Index(drugs).get_indexer(kept["drug"])
        lower_log2[row_positions, col_positions] = MicSteps.to_log2(kept["mic_lower"].to_numpy())
        upper_log2[row_positions, col_positions] = MicSteps.to_log2(kept["mic_upper"].to_numpy())
        mask[row_positions, col_positions] = True
        return cls(genome_ids, species, drugs, lower_log2, upper_log2, mask)

    def subset(self, positions: np.ndarray) -> "LabelMatrix":
        """Rows at the given positions, same drug order."""
        return LabelMatrix(
            self.genome_ids[positions],
            self.species[positions],
            self.drugs,
            self.lower_log2[positions],
            self.upper_log2[positions],
            self.mask[positions],
        )

    def positions_of(self, genome_ids: np.ndarray) -> np.ndarray:
        """Row positions for genome_ids. Raises if any is unknown."""
        unknown = [genome_id for genome_id in genome_ids if genome_id not in self.genome_position]
        if unknown:
            raise KeyError(f"{len(unknown)} genome_ids not in labels, e.g. {unknown[:3]}")
        return np.array([self.genome_position[genome_id] for genome_id in genome_ids], dtype=np.int64)

    def label_counts(self) -> pd.DataFrame:
        """Species x drug count of present labels. Emit this before training anything."""
        rows = []
        for species_key in SPECIES_KEYS:
            species_rows = self.species == species_key
            if not species_rows.any():
                continue
            for drug_position, drug in enumerate(self.drugs):
                present = self.mask[species_rows, drug_position]
                rows.append({"species": species_key, "drug": drug, "n_labels": int(present.sum())})
        return pd.DataFrame(rows)
