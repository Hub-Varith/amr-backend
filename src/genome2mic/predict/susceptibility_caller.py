"""Turn a predicted MIC into a call against breakpoints (DATA_CONTRACT.md stage 12 call logic)."""

import math

import numpy as np
from scipy.special import ndtr

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.predict.call_thresholds import CallThresholds
from genome2mic.predict.confidence_levels import ConfidenceLevels
from genome2mic.predict.constants import CALL_STANDARD, CALL_YEAR, LIKELY_ACTIVE, LIKELY_INACTIVE, UNCERTAIN
from genome2mic.predict.drug_call import DrugCall


class SusceptibilityCaller:
    """Calls from p_active where the pair has fitted thresholds, else from the band.

    Band rule: likely active when the whole band is at or below S; likely inactive when the
    whole band is above R. Both rules only call a drug active when the evidence is strong.
    """

    def __init__(
        self,
        breakpoints: BreakpointTable,
        standard: str = CALL_STANDARD,
        year: int = CALL_YEAR,
        thresholds: CallThresholds | None = None,
    ) -> None:
        self.breakpoints = breakpoints
        self.standard = standard
        self.year = year
        self.thresholds = thresholds

    @staticmethod
    def probability_active(
        mu_log2: float | np.ndarray, sigma_log2: float | np.ndarray, s_breakpoint: float | np.ndarray
    ) -> float | np.ndarray:
        """P(MIC <= S breakpoint) under the model's normal on log2 MIC."""
        return ndtr((np.log2(s_breakpoint) - mu_log2) / sigma_log2)

    def call(
        self, species: str, drug: str, band_low: float, band_high: float, p_active: float | None = None
    ) -> DrugCall:
        """Band edges are in mg/L, already rounded up to doubling steps. p_active is the calibrated probability."""
        found = self.breakpoints.lookup(species, drug, self.standard, self.year)
        if found is None:
            return DrugCall(UNCERTAIN, None, None, None, reason=f"no {self.standard} breakpoint for this drug")
        s_breakpoint, r_breakpoint = found
        # A band top above S has no margin, which can happen under the probability rule.
        margin_steps = max(0, round(math.log2(s_breakpoint) - math.log2(band_high)))
        level = None if p_active is None else ConfidenceLevels.level_for(p_active)

        pair_thresholds = None
        if self.thresholds is not None and p_active is not None:
            pair_thresholds = self.thresholds.lookup(species, drug)
        if pair_thresholds is not None:
            active_min, inactive_max = pair_thresholds
            is_active = p_active >= active_min
            is_inactive = p_active <= inactive_max
        else:
            is_active = band_high <= s_breakpoint
            is_inactive = band_low > r_breakpoint

        if is_active:
            return DrugCall(LIKELY_ACTIVE, s_breakpoint, r_breakpoint, margin_steps, p_active=p_active,
                            confidence_level=level)
        if is_inactive:
            return DrugCall(LIKELY_INACTIVE, s_breakpoint, r_breakpoint, None, p_active=p_active,
                            confidence_level=level)
        return DrugCall(UNCERTAIN, s_breakpoint, r_breakpoint, None, p_active=p_active, confidence_level=level)
