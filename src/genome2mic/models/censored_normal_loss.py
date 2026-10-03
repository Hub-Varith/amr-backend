"""Interval-censored normal negative log-likelihood on the log2 MIC scale."""

import torch
from torch import nn


class CensoredNormalLoss(nn.Module):
    """-log P(lower < Y <= upper) for Y ~ Normal(mu, sigma), summed only where mask is True.

    lower may be -inf (left-censored) and upper may be +inf (right-censored). This is the
    neural-network twin of XGBoost's survival:aft with a normal distribution.
    """

    def __init__(self, drug_balanced: bool = False) -> None:
        super().__init__()
        self.drug_balanced = drug_balanced

    def forward(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """All tensors are (batch, n_drugs) except log_sigma, which broadcasts (n_drugs,)."""
        # Masked cells carry NaN bounds; swap in a harmless interval so NaN never reaches the sum.
        present = mask > 0
        lower = torch.where(present, lower, torch.full_like(lower, -1.0))
        upper = torch.where(present, upper, torch.full_like(upper, 1.0))
        # Cap infinite bounds before any arithmetic: inf / sigma has a NaN gradient for sigma.
        lower = torch.clamp(lower, min=-1e3)
        upper = torch.clamp(upper, max=1e3)
        sigma = torch.exp(log_sigma)
        z_upper = (upper - mu) / sigma
        z_lower = (lower - mu) / sigma

        # log(Phi(a) - Phi(b)) is stable when a <= 0; mirror the interval when both are positive.
        flip = (z_upper + z_lower) > 0
        high = torch.where(flip, -z_lower, z_upper)
        low = torch.where(flip, -z_upper, z_lower)
        log_high = torch.special.log_ndtr(high)
        log_low = torch.special.log_ndtr(low)
        log_prob = log_high + torch.log1p(-torch.exp(torch.clamp(log_low - log_high, max=-1e-7)))
        nll = torch.where(present, -log_prob, torch.zeros_like(log_prob))

        if not self.drug_balanced:
            return nll.sum() / mask.sum().clamp(min=1)
        per_drug_count = mask.sum(dim=0)
        per_drug_mean = nll.sum(dim=0) / per_drug_count.clamp(min=1)
        return per_drug_mean[per_drug_count > 0].mean()
