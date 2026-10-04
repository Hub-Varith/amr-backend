"""The call for one drug."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DrugCall:
    """Call, breakpoints used (mg/L), margin in doubling steps below S, and the calibrated P(active) with its level when known."""

    call: str
    s_breakpoint: float | None
    r_breakpoint: float | None
    margin_steps: int | None
    reason: str | None = None
    p_active: float | None = None
    confidence_level: str | None = None
