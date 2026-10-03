"""Fold-specific feature view: which known-AMR columns and which unitig columns to use."""

import logging

import numpy as np
import scipy.sparse as sparse

from genome2mic.data.constants import SPECIES_KEYS
from genome2mic.data.known_amr_table import KnownAmrTable
from genome2mic.data.label_matrix import LabelMatrix
from genome2mic.data.unitig_store import UnitigStore
from genome2mic.models.train_config import TrainConfig

logger = logging.getLogger(__name__)


class FeatureBundle:
    """Column choices fitted on one training fold, then applied to any genome.

    Fitting touches training-fold genomes only (contract rule 4). The chosen columns are
    saved with the model so prediction uses exactly the same ones.
    """

    def __init__(
        self,
        known: KnownAmrTable,
        unitigs: UnitigStore,
        known_columns: np.ndarray,
        unitig_columns: dict[str, np.ndarray],
    ) -> None:
        self.known = known
        self.unitigs = unitigs
        self.known_columns = known_columns
        self.unitig_columns = unitig_columns

    @classmethod
    def fit(
        cls,
        known: KnownAmrTable,
        unitigs: UnitigStore,
        labels: LabelMatrix,
        train_positions: np.ndarray,
        config: TrainConfig,
    ) -> "FeatureBundle":
        """Choose columns using the training-fold genomes in labels."""
        train_ids = labels.genome_ids[train_positions]
        train_species = labels.species[train_positions]
        known_columns = known.frequent_columns(train_ids, train_species, config.known_min_count)

        unitig_columns: dict[str, np.ndarray] = {}
        if config.use_unitigs:
            # Finite proxy for the lab value: the reported bound. Used only to rank columns.
            proxy = np.where(np.isfinite(labels.upper_log2), labels.upper_log2, labels.lower_log2)
            proxy = np.where(labels.mask, proxy, np.nan)
            for species_key in SPECIES_KEYS:
                if not unitigs.has_species(species_key):
                    continue
                species_rows = train_positions[train_species == species_key]
                if len(species_rows) == 0:
                    continue
                unitig_columns[species_key] = unitigs.select_columns(
                    species_key,
                    labels.genome_ids[species_rows],
                    proxy[species_rows],
                    config.unitig_min_frac,
                    config.unitig_max_frac,
                    config.unitig_max_columns,
                )
        logger.info(
            "FeatureBundle fitted",
            extra={
                "n_known_columns": len(known_columns),
                "n_unitig_columns": {key: len(value) for key, value in unitig_columns.items()},
            },
        )
        return cls(known, unitigs, known_columns, unitig_columns)

    @property
    def known_column_names(self) -> list[str]:
        return [self.known.columns[index] for index in self.known_columns]

    def unitig_width(self, species_key: str) -> int:
        return len(self.unitig_columns.get(species_key, np.array([], dtype=np.int64)))

    def known_rows(self, genome_ids: np.ndarray) -> np.ndarray:
        """Dense (n, n_known_selected) float32 block. Count columns are log-compressed."""
        rows = self.known.rows(genome_ids)[:, self.known_columns]
        return np.log1p(rows).astype(np.float32)

    def unitig_rows(self, species_key: str, genome_ids: np.ndarray) -> sparse.csr_matrix | None:
        """Sparse selected-column block for one species, or None when that species has no unitigs."""
        if species_key not in self.unitig_columns:
            return None
        return self.unitigs.rows(species_key, genome_ids, self.unitig_columns[species_key])
