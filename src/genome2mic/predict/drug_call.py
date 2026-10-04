"""The call for one drug."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DrugCall:
    """Call, breakpoints used (mg/L), and margin in doubling steps below the S breakpoint."""

    call: str
    s_breakpoint: float | None
    r_breakpoint: float | None
    margin_steps: int | None
    reason: str | None = None
