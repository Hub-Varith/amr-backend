"""Per-drug call."""

from enum import StrEnum


class Call(StrEnum):
    """Call made by comparing the MIC band to the breakpoints."""

    LIKELY_ACTIVE = "likely_active"
    UNCERTAIN = "uncertain"
    LIKELY_INACTIVE = "likely_inactive"
