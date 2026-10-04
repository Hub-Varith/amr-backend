"""Builds species-homogeneous mini-batches of tensors from a LabelMatrix and a FeatureBundle."""

import numpy as np
import scipy.sparse as sparse
import torch

from genome2mic.data.constants import SPECIES_KEYS
from genome2mic.data.label_matrix import LabelMatrix
from genome2mic.models.feature_bundle import FeatureBundle


class GenomeBatcher:
    """Each batch holds genomes of one species, so the unitig adapter for that species applies to all rows."""

    def __init__(self, labels: LabelMatrix, bundle: FeatureBundle, batch_size: int, seed: int) -> None:
        self.labels = labels
        self.bundle = bundle
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def batches(self, positions: np.ndarray, shuffle: bool) -> list[np.ndarray]:
        """Row-position chunks, one species per chunk, optionally shuffled across species."""
        chunks = []
        for species_key in SPECIES_KEYS:
            species_positions = positions[self.labels.species[positions] == species_key]
            if shuffle:
                species_positions = self.rng.permutation(species_positions)
            for start in range(0, len(species_positions), self.batch_size):
                chunks.append(species_positions[start : start + self.batch_size])
        if shuffle:
            order = self.rng.permutation(len(chunks))
            chunks = [chunks[index] for index in order]
        return chunks

    def tensors(self, positions: np.ndarray) -> dict[str, torch.Tensor | None]:
        """Model inputs and loss targets for one species-homogeneous chunk."""
        species_key = str(self.labels.species[positions[0]])
        genome_ids = self.labels.genome_ids[positions]
        unitig_block = self.bundle.unitig_rows(species_key, genome_ids)
        return {
            "species": torch.from_numpy(self.labels.species_index[positions]),
            "known": torch.from_numpy(self.bundle.known_rows(genome_ids)),
            "unitigs": None if unitig_block is None else self.to_sparse_tensor(unitig_block),
            "lower": torch.from_numpy(self.labels.lower_log2[positions].astype(np.float32)),
            "upper": torch.from_numpy(self.labels.upper_log2[positions].astype(np.float32)),
            "mask": torch.from_numpy(self.labels.mask[positions].astype(np.float32)),
        }

    @staticmethod
    def to_sparse_tensor(block: sparse.csr_matrix) -> torch.Tensor:
        coo = block.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        values = torch.from_numpy(coo.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, coo.shape, check_invariants=False).coalesce()
