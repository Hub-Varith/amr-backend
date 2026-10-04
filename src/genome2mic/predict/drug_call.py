"""The call for one drug."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DrugCall:
    """Call, breakpoints used (mg/L), margin in doubling steps below S, and P(MIC <= S) when known."""

    call: str
    s_breakpoint: float | None
    r_breakpoint: float | None
    margin_steps: int | None
    reason: str | None = None
    p_active: float | None = None
