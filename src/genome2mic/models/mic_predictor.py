"""Turns a loaded ModelArtifact into MIC predictions for new genomes."""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch

from genome2mic.data.constants import SPECIES_KEYS
from genome2mic.data.mic_steps import MicSteps
from genome2mic.models.genome_batcher import GenomeBatcher
from genome2mic.models.model_artifact import ModelArtifact

logger = logging.getLogger(__name__)


class MicPredictor:
    """Takes known-AMR values and a unitig presence vector for one species, returns every drug's MIC and band."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def known_vector(self, known_row: dict[str, int]) -> np.ndarray:
        """Order a {column_name: value} dict into the model's known-AMR columns. Unknown columns are ignored and logged."""
        unseen = set(known_row) - set(self.artifact.known_columns)
        if unseen:
            logger.info("Known-AMR columns not in the model ignored", extra={"n_ignored": len(unseen)})
        values = np.array([known_row.get(column, 0) for column in self.artifact.known_columns], dtype=np.float32)
        return np.log1p(values)

    def unitig_vector(self, species_key: str, present_pattern_ids: set[str]) -> sparse.csr_matrix | None:
        """Presence of the model's selected patterns for this species, from a unitig-caller --query result."""
        pattern_ids = self.artifact.unitig_pattern_ids.get(species_key)
        if not pattern_ids:
            return None
        values = np.array([pattern_id in present_pattern_ids for pattern_id in pattern_ids], dtype=np.float32)
        return sparse.csr_matrix(values.reshape(1, -1))

    def predict(self, species_key: str, known: np.ndarray, unitigs: sparse.csr_matrix | None) -> pd.DataFrame:
        """known: (n, n_known) already ordered. unitigs: (n, n_selected) or None. One species per call."""
        if species_key not in SPECIES_KEYS:
            raise ValueError(f"unknown species {species_key}")
        species_index = torch.full((known.shape[0],), SPECIES_KEYS.index(species_key), dtype=torch.int64)
        unitig_tensor = None if unitigs is None else GenomeBatcher.to_sparse_tensor(unitigs)
        with torch.no_grad():
            mu, _ = self.artifact.model(species_index, torch.from_numpy(known.astype(np.float32)), unitig_tensor)
        mu = mu.numpy()

        reportable = set(self.artifact.drugs_by_species.get(species_key, []))
        rows = []
        for row in range(known.shape[0]):
            for drug_position, drug in enumerate(self.artifact.drugs):
                if drug not in reportable:
                    continue
                point = float(mu[row, drug_position])
                q = self.artifact.conformal.half_width(species_key, drug)
                rows.append(
                    {
                        "row": row,
                        "species": species_key,
                        "drug": drug,
                        "mu_log2": point,
                        "pred_mic": float(MicSteps.round_up_to_step(np.array([point]))[0]),
                        "band_low": float(MicSteps.round_up_to_step(np.array([point - q]))[0]),
                        "band_high": float(MicSteps.round_up_to_step(np.array([point + q]))[0]),
                    }
                )
        logger.info("Predicted", extra={"species": species_key, "n_genomes": known.shape[0], "n_rows": len(rows)})
        return pd.DataFrame(rows)
