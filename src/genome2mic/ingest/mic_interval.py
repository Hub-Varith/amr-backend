"""Turns one lab result into an MIC interval (mic_lower, mic_upper]. See DATA_CONTRACT.md stage 2."""

import math
import re

from genome2mic.ingest.constants import SIGN_ALIASES, SNAP_TOLERANCE

FIRST_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)")


class MicIntervalConverter:
    """Pure conversions for the interval rule. Every method is static."""

    @staticmethod
    def normalize_sign(raw_sign: str) -> str | None:
        """Return one of `=`, `<=`, `<`, `>`, `>=`, or None when the sign is not recognized."""
        return SIGN_ALIASES.get(raw_sign.strip())

    @staticmethod
    def parse_value(text: str) -> float | None:
        """Return the first number in the text. Combination drugs report `8/4`; the first part is the MIC."""
        match = FIRST_NUMBER.match(text)
        if match is None:
            return None
        return float(match.group(1))

    @staticmethod
    def snap_to_doubling_step(value: float) -> float | None:
        """Return the doubling step (2^k) within 5% of the value, or None when the value sits between steps."""
        if value <= 0:
            return None
        step = 2.0 ** round(math.log2(value))
        if abs(value - step) / step > SNAP_TOLERANCE:
            return None
        return step

    @staticmethod
    def from_mic(sign: str, value: float) -> tuple[float, float, str]:
        """Interval for a measured MIC. `<` is read as `<=`: a wider interval that is never wrong."""
        if sign == "=":
            return value / 2, value, "interval"
        if sign in ("<=", "<"):
            return 0.0, value, "left"
        if sign == ">":
            return value, math.inf, "right"
        if sign == ">=":
            return value / 2, math.inf, "right"
        raise ValueError(f"Unknown sign: {sign!r}")

    @staticmethod
    def from_sir(sir: str, s_breakpoint: float, r_breakpoint: float) -> tuple[float, float, str] | None:
        """Interval for an S/I/R-only result. Breakpoints use the form S if MIC <= s, R if MIC > r."""
        if sir == "S":
            return 0.0, s_breakpoint, "left"
        if sir == "R":
            return r_breakpoint, math.inf, "right"
        if sir == "I" and s_breakpoint < r_breakpoint:
            return s_breakpoint, r_breakpoint, "interval"
        return None
