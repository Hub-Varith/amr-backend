"""Tier of the calibrated probability that a drug works."""

from enum import StrEnum

from genome2mic.models.calibration import tier_of


class ProbTier(StrEnum):
    """Plain-language band of ``prob_works`` (>= 0.90, 0.70-0.90, 0.30-0.70, 0.10-0.30, <= 0.10)."""

    VERY_LIKELY_WORKS = "very_likely_works"
    PROBABLY_WORKS = "probably_works"
    UNCERTAIN = "uncertain"
    PROBABLY_FAILS = "probably_fails"
    VERY_LIKELY_FAILS = "very_likely_fails"

    @classmethod
    def from_probability(cls, p: float) -> "ProbTier":
        tier = tier_of(p)
        if tier is None:
            raise ValueError("probability is null")
        return cls(tier)
