"""Tier of the calibrated probability that a drug works."""

from enum import StrEnum

from genome2mic.models.calibration import tier_for_call, tier_of


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

    @classmethod
    def for_call(cls, p: float, call: str | None) -> "ProbTier":
        """The displayed tier: :meth:`from_probability` capped to agree with the call (v0.7)."""
        tier = tier_for_call(p, None if call is None else str(call))
        if tier is None:
            raise ValueError("probability is null")
        return cls(tier)
