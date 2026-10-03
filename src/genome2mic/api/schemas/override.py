"""Rule overrides applied after the model."""

from enum import StrEnum


class Override(StrEnum):
    """Rule that forced a drug to likely_inactive regardless of the model."""

    NATURAL_RESISTANCE = "natural_resistance"
    STRONG_MARKER = "strong_marker"
