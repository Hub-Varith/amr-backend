"""Shared-body network: one forward pass gives a log2 MIC for every drug."""

import torch
from torch import nn


class MultitaskMicNet(nn.Module):
    """Species embedding + known-AMR features + per-species sparse unitig adapter -> shared trunk -> per-drug heads.

    Outputs mu (batch, n_drugs) in log2 mg/L and a learned log_sigma per drug. Drugs a genome
    was never tested for are masked in the loss, so one model serves every species x drug pair.
    See MODEL_DESIGN.md for the reasoning.
    """

    def __init__(
        self,
        n_species: int,
        n_known: int,
        n_unitig_by_species: dict[int, int],
        n_drugs: int,
        species_dim: int = 16,
        known_dim: int = 256,
        unitig_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_drugs = n_drugs
        self.unitig_dim = unitig_dim
        self.species_embedding = nn.Embedding(n_species, species_dim)
        self.known_encoder = nn.Sequential(
            nn.Linear(n_known, known_dim), nn.GELU(), nn.Dropout(dropout)
        )
        # One sparse linear layer per species, since each species has its own unitig columns.
        self.unitig_adapters = nn.ModuleDict(
            {str(species): nn.Linear(n_columns, unitig_dim) for species, n_columns in n_unitig_by_species.items()}
        )
        trunk_in = species_dim + known_dim + unitig_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.drug_heads = nn.Linear(hidden_dim, n_drugs)
        # Per-species shift per drug: the same gene set can mean a different MIC in another species.
        self.species_drug_bias = nn.Embedding(n_species, n_drugs)
        nn.init.zeros_(self.species_drug_bias.weight)
        self.log_sigma = nn.Parameter(torch.zeros(n_drugs))

    def forward(
        self, species: torch.Tensor, known: torch.Tensor, unitigs: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """species: (batch,) ints, one species per batch. known: (batch, n_known). unitigs: sparse (batch, n_cols) or None."""
        species_vector = self.species_embedding(species)
        known_vector = self.known_encoder(known)
        species_id = str(int(species[0]))
        if unitigs is None or species_id not in self.unitig_adapters:
            unitig_vector = known.new_zeros((known.shape[0], self.unitig_dim))
        else:
            adapter = self.unitig_adapters[species_id]
            unitig_vector = torch.sparse.mm(unitigs, adapter.weight.t()) + adapter.bias
        hidden = self.trunk(torch.cat([species_vector, known_vector, unitig_vector], dim=1))
        mu = self.drug_heads(hidden) + self.species_drug_bias(species)
        return mu, self.log_sigma
