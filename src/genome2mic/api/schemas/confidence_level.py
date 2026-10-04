"""Confidence level of one drug prediction."""

from enum import StrEnum


class ConfidenceLevel(StrEnum):
    """Range of the calibrated P(active) (DATA_CONTRACT.md stage 12)."""

    VERY_LIKELY_INACTIVE = "very_likely_inactive"
    PROBABLY_INACTIVE = "probably_inactive"
    LEANS_INACTIVE = "leans_inactive"
    UNCLEAR = "unclear"
    LEANS_ACTIVE = "leans_active"
    PROBABLY_ACTIVE = "probably_active"
    VERY_LIKELY_ACTIVE = "very_likely_active"
