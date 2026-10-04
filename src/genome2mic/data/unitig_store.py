"""Per-species unitig pattern matrices (DATA_CONTRACT.md stage 8)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse

logger = logging.getLogger(__name__)


class UnitigStore:
    """Holds one sparse CSR matrix per species and looks rows up by genome_id.

    The matrix is never densified. Column selection happens per training fold
    through select_columns, never on the full dataset.
    """

    def __init__(self) -> None:
        self.matrices: dict[str, sparse.csr_matrix] = {}
        self.pattern_ids: dict[str, np.ndarray] = {}
        self.positions: dict[str, dict[str, int]] = {}

    @classmethod
    def from_processed_dir(cls, processed_dir: Path, species_keys: list[str]) -> "UnitigStore":
        """Load unitigs_<SPECIES>.npz and its row/index parquet files for each species that has them."""
        store = cls()
        for species_key in species_keys:
            matrix_path = processed_dir / f"unitigs_{species_key}.npz"
            if not matrix_path.exists():
                logger.warning("No unitig matrix for species", extra={"species": species_key})
                continue
            matrix = sparse.load_npz(matrix_path).tocsr()
            rows = pd.read_parquet(processed_dir / f"unitigs_{species_key}_rows.parquet")
            index = pd.read_parquet(processed_dir / f"unitigs_{species_key}_index.parquet")
            store.add_species(species_key, matrix, rows["genome_id"].to_numpy(), index["pattern_id"].to_numpy())
        return store

    def add_species(
        self, species_key: str, matrix: sparse.csr_matrix, genome_ids: np.ndarray, pattern_ids: np.ndarray
    ) -> None:
        if matrix.shape[0] != len(genome_ids) or matrix.shape[1] != len(pattern_ids):
            raise ValueError(f"unitig matrix shape {matrix.shape} does not match its row/column index")
        self.matrices[species_key] = matrix.astype(np.float32)
        self.pattern_ids[species_key] = pattern_ids
        self.positions[species_key] = {genome_id: index for index, genome_id in enumerate(genome_ids)}
        logger.info(
            "Unitig matrix loaded",
            extra={"species": species_key, "n_genomes": matrix.shape[0], "n_patterns": matrix.shape[1]},
        )

    def has_species(self, species_key: str) -> bool:
        return species_key in self.matrices

    def rows(self, species_key: str, genome_ids: np.ndarray, columns: np.ndarray) -> sparse.csr_matrix:
        """Sparse block of the selected columns for genome_ids. Unknown genomes become empty rows."""
        matrix = self.matrices[species_key]
        row_positions = []
        present = []
        for genome_id in genome_ids:
            index = self.positions[species_key].get(genome_id)
            present.append(index is not None)
            row_positions.append(0 if index is None else index)
        block = matrix[row_positions][:, columns]
        n_missing = len(present) - sum(present)
        if n_missing:
            logger.warning("Genomes without unitig rows set to zero", extra={"species": species_key, "n_missing": n_missing})
            block = sparse.diags(np.asarray(present, dtype=np.float32)) @ block
        return sparse.csr_matrix(block, dtype=np.float32)

    def select_columns(
        self,
        species_key: str,
        genome_ids: np.ndarray,
        targets: np.ndarray,
        min_frac: float,
        max_frac: float,
        max_columns: int,
    ) -> np.ndarray:
        """Pick columns inside a training fold: frequency window, then top correlation with any drug.

        targets is (n_genomes, n_drugs) of finite log2 MIC proxies, NaN where no label. It is a
        cheap stand-in for pyseer selection and must only ever see training-fold genomes.
        See MODEL_DESIGN.md.
        """
        all_columns = np.arange(self.matrices[species_key].shape[1])
        block = self.rows(species_key, genome_ids, all_columns)
        frequency = np.asarray(block.mean(axis=0)).ravel()
        candidates = np.flatnonzero((frequency >= min_frac) & (frequency <= max_frac))
        logger.info(
            "Unitig frequency filter",
            extra={"species": species_key, "n_in": len(all_columns), "n_in_window": len(candidates)},
        )
        if len(candidates) <= max_columns:
            return candidates

        best_correlation = np.zeros(len(candidates))
        candidate_block = block[:, candidates]
        for drug_position in range(targets.shape[1]):
            target = targets[:, drug_position]
            known = ~np.isnan(target)
            if known.sum() < 3:
                continue
            centered = target[known] - target[known].mean()
            target_norm = np.sqrt((centered**2).sum()) + 1e-9
            known_block = candidate_block[known]
            column_sums = np.asarray(known_block.sum(axis=0)).ravel()
            column_norm = np.sqrt(np.maximum(column_sums - column_sums**2 / known.sum(), 0.0)) + 1e-9
            cross = np.asarray(known_block.T @ centered).ravel()
            best_correlation = np.maximum(best_correlation, np.abs(cross) / (column_norm * target_norm))
        top = np.argsort(-best_correlation)[:max_columns]
        selected = np.sort(candidates[top])
        logger.info("Unitig correlation filter", extra={"species": species_key, "n_selected": len(selected)})
        return selected
