"""Known-AMR features from known_amr.parquet (DATA_CONTRACT.md stage 5)."""

import logging

import numpy as np
import pandas as pd

from genome2mic.data.constants import FEATURE_PREFIXES, FORBIDDEN_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class KnownAmrTable:
    """Dense genome x feature matrix for every known-AMR column, indexed by genome_id."""

    def __init__(self, table: pd.DataFrame) -> None:
        forbidden = FORBIDDEN_FEATURE_COLUMNS & set(table.columns)
        if forbidden:
            raise ValueError(f"known_amr contains forbidden columns {sorted(forbidden)} (contract rule 2)")
        if table["genome_id"].duplicated().any():
            raise ValueError("known_amr has duplicate genome_id rows")
        self.columns = [column for column in table.columns if column.startswith(FEATURE_PREFIXES)]
        if not self.columns:
            raise ValueError("known_amr has no gene_/point_/n_class_ columns")
        feature_block = table[self.columns]
        if feature_block.isna().any().any():
            raise ValueError("known_amr feature columns contain nulls")
        self.genome_ids = table["genome_id"].to_numpy()
        self.species = table["species"].to_numpy() if "species" in table.columns else None
        self.values = feature_block.to_numpy(dtype=np.float32)
        self.position = {genome_id: index for index, genome_id in enumerate(self.genome_ids)}
        logger.info("KnownAmrTable loaded", extra={"n_genomes": len(self.genome_ids), "n_columns": len(self.columns)})

    @classmethod
    def from_parquet(cls, path: str) -> "KnownAmrTable":
        return cls(pd.read_parquet(path))

    def rows(self, genome_ids: np.ndarray) -> np.ndarray:
        """Return measured feature rows; an absent row is not evidence of no AMR genes."""
        missing = [genome_id for genome_id in genome_ids if genome_id not in self.position]
        if missing:
            raise ValueError(f"Missing known_amr rows for {len(missing)} genomes, e.g. {missing[:3]}")
        return self.values[[self.position[genome_id] for genome_id in genome_ids]].copy()

    def frequent_columns(self, genome_ids: np.ndarray, species: np.ndarray, min_count: int) -> np.ndarray:
        """Column positions present in at least min_count of the given genomes within some species.

        Call this with training-fold genomes only (contract rule 4).
        """
        rows = self.rows(genome_ids)
        keep = np.zeros(len(self.columns), dtype=bool)
        for species_key in np.unique(species):
            species_rows = rows[species == species_key]
            keep |= (species_rows > 0).sum(axis=0) >= min_count
        logger.info(
            "Known-AMR rare filter",
            extra={"min_count": min_count, "n_columns_in": len(self.columns), "n_columns_kept": int(keep.sum())},
        )
        return np.flatnonzero(keep)
