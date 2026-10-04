"""Compare a predicted MIC band with breakpoints to make a call (CLAUDE.md call logic)."""

import math

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.predict.constants import CALL_STANDARD, CALL_YEAR, LIKELY_ACTIVE, LIKELY_INACTIVE, UNCERTAIN
from genome2mic.predict.drug_call import DrugCall


class SusceptibilityCaller:
    """Likely active when the whole band is at or below S; likely inactive when the whole band is above R.

    Using the band, not the point prediction, is the safer choice: a drug is called active only
    when even the high end of the band clears the S breakpoint.
    """

    def __init__(self, breakpoints: BreakpointTable, standard: str = CALL_STANDARD, year: int = CALL_YEAR) -> None:
        self.breakpoints = breakpoints
        self.standard = standard
        self.year = year

    def call(self, species: str, drug: str, band_low: float, band_high: float) -> DrugCall:
        """Band edges are in mg/L, already rounded up to doubling steps."""
        found = self.breakpoints.lookup(species, drug, self.standard, self.year)
        if found is None:
            return DrugCall(UNCERTAIN, None, None, None, reason=f"no {self.standard} breakpoint for this drug")
        s_breakpoint, r_breakpoint = found

        if band_high <= s_breakpoint:
            margin_steps = round(math.log2(s_breakpoint) - math.log2(band_high))
            return DrugCall(LIKELY_ACTIVE, s_breakpoint, r_breakpoint, margin_steps)
        if band_low > r_breakpoint:
            return DrugCall(LIKELY_INACTIVE, s_breakpoint, r_breakpoint, None)
        return DrugCall(UNCERTAIN, s_breakpoint, r_breakpoint, None)
