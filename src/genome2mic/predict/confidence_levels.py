"""Names for ranges of the calibrated P(active)."""

import bisect

from genome2mic.predict.constants import CONFIDENCE_LEVEL_EDGES, CONFIDENCE_LEVEL_NAMES


class ConfidenceLevels:
    """Seven levels from very_likely_inactive to very_likely_active. A lower edge belongs to the higher level."""

    @staticmethod
    def level_for(p_active: float) -> str:
        if not 0.0 <= p_active <= 1.0:
            raise ValueError(f"p_active must be within 0 and 1, got {p_active}")
        return CONFIDENCE_LEVEL_NAMES[bisect.bisect_right(CONFIDENCE_LEVEL_EDGES, p_active)]
