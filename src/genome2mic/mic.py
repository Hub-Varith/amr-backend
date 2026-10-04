"""MIC arithmetic on the doubling grid and the interval rule from ``DATA_CONTRACT.md``.

Everything here is a pure function of its arguments. MICs are in mg/L and are
positive floats; they are not required to sit on the doubling grid (gradient
strips report values such as 0.19 or 6), but every *bound* this module returns
is ``0.0``, ``inf`` or an exact power of two, so downstream comparisons are stable
and :func:`label_point_log2` is always integer-valued.

The interval rule (contract, stage 2) turns one lab result into ``(lo, hi]``:

=============  ===========  ===========  ============
raw result     ``mic_lower``  ``mic_upper``  ``censor``
=============  ===========  ===========  ============
``= 8``        4            8            ``interval``
``<= 0.25``    0            0.25         ``left``
``< 0.5``      0            0.25         ``left``
``> 32``       32           inf          ``right``
``>= 16``      8            inf          ``right``
S only, S<=1   0            1            ``left``
R only, R>2    2            inf          ``right``
I only         1            2            ``interval``
=============  ===========  ===========  ============

Off-grid values are placed in the grid cell that contains them: ``= 6`` lies in
``(4, 8]``; ``= 0.19`` in ``(0.125, 0.25]``. Censored bounds are widened, never
narrowed, so an interval is always a true statement about the lab result.
``< X`` is read as "at most the step below X" (``< 0.5`` -> ``(0, 0.25]``), which is
how dilution panels and gradient strips use the sign.

Decimal renderings of powers of two (``0.016`` for ``2**-6``, ``0.008`` for ``2**-7``,
``0.12`` for ``2**-3``) are not off-grid values: the ingest stage passes every reported
number through :func:`snap_reported_mic` first, so ``= 0.016`` is ``(2**-7, 2**-6]``,
not the cell above.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

GRID_MIN_EXPONENT = -10
GRID_MAX_EXPONENT = 12

DOUBLING_GRID: tuple[float, ...] = tuple(
    math.ldexp(1.0, k) for k in range(GRID_MIN_EXPONENT, GRID_MAX_EXPONENT + 1)
)
"""Reference MIC panel: ``2**k`` for ``k`` in -10..12 (0.00098 ... 4096 mg/L).

The rounding functions are not clipped to this range; it documents the span real
panels use and is handy for axis ticks and confusion matrices.
"""

INF = math.inf

CENSOR_INTERVAL = "interval"
CENSOR_LEFT = "left"
CENSOR_RIGHT = "right"

_SIGN_ALIASES: dict[str, str] = {
    "": "=",
    "=": "=",
    "==": "=",
    "<=": "<=",
    "=<": "<=",
    "≤": "<=",  # ≤
    "<": "<",
    ">": ">",
    ">=": ">=",
    "=>": ">=",
    "≥": ">=",  # ≥
}

_SIR_ALIASES: dict[str, str] = {
    "S": "S",
    "SUSCEPTIBLE": "S",
    "I": "I",
    "INTERMEDIATE": "I",
    "R": "R",
    "RESISTANT": "R",
}

# Absolute tolerance on log2(MIC) below which a value is treated as *on* the grid.
# 1e-9 in log2 space is ~7e-10 relative, far above IEEE drift (~1e-16) and far
# below any real panel spacing, so 8.000000000000002 -> 8 but 8.001 -> 16.
_GRID_TOL = 1e-9

SNAP_TOL_LOG2 = 0.1
"""Ingest-only tolerance (log2 units) of :func:`snap_reported_mic`.

Panels and gradient strips print small powers of two as rounded decimals that sit on
either side of the grid: ``0.016, 0.032, 0.064, 0.008, 0.004, 0.002, 0.001`` are
``|log2| ~ 0.034`` above ``2**-6 .. 2**-10``; ``0.015, 0.03, 0.06, 0.12`` are ~0.059 below
``2**-6 .. 2**-3``. Gradient half steps (0.023, 0.047, 0.094, 0.19, 0.38, 0.75, 1.5,
3, 6, 12, 24, 48, ...) are at least 0.39 from the grid, so 0.1 separates the two
families with a wide margin. This is *not* ``_GRID_TOL``: the rounding functions stay
exact, only reported lab values are snapped."""


class BreakpointLike(Protocol):
    """Anything with S and R breakpoints in mg/L (``config.Breakpoint`` fits).

    Convention (contract): S if MIC <= ``s_breakpoint``; R if MIC > ``r_breakpoint``;
    I otherwise. ``s_breakpoint <= r_breakpoint`` always.
    """

    s_breakpoint: float
    r_breakpoint: float


# --------------------------------------------------------------------------- helpers
def _is_missing(value: Any) -> bool:
    """True for ``None``, float NaN and ``pandas.NA`` (duck-typed; no pandas import)."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(value))
    return type(value).__name__ == "NAType"


def _as_positive_float(value: Any, name: str = "mic") -> float:
    """Coerce to float and require ``> 0`` (``inf`` allowed, NaN/None rejected)."""
    if _is_missing(value):
        raise ValueError(f"{name} is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if math.isnan(number) or number <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def _nearest_int_if_close(x: float) -> int | None:
    """Return ``round(x)`` when ``x`` is within ``_GRID_TOL`` of an integer, else ``None``."""
    k = round(x)
    return k if math.isclose(x, k, abs_tol=_GRID_TOL, rel_tol=0.0) else None


def _exponent(mic: float, mode: str) -> int:
    """Integer ``k`` such that ``2**k`` is the requested rounding of ``mic``.

    ``mode`` is ``up`` (ceil), ``down`` (floor) or ``nearest`` (ties go up).
    Values already on the grid (within float drift) are returned unchanged.
    """
    l2 = math.log2(mic)
    if mode == "nearest":
        shifted = l2 + 0.5
        k = _nearest_int_if_close(shifted)
        return k if k is not None else math.floor(shifted)
    k = _nearest_int_if_close(l2)
    if k is not None:
        return k
    return math.ceil(l2) if mode == "up" else math.floor(l2)


# ------------------------------------------------------------------- scalar API
def log2_step(mic: float) -> float:
    """``log2(mic)``: position of an MIC on the doubling scale (``8 -> 3``, ``0.25 -> -2``).

    ``inf`` maps to ``inf``. Raises ``ValueError`` for non-positive or missing values.
    """
    return math.log2(_as_positive_float(mic))


def round_up_to_step(mic: float) -> float:
    """Smallest ``2**k >= mic``; exact powers of two are returned unchanged.

    ``0.19 -> 0.25``, ``6 -> 8``, ``8 -> 8``, ``inf -> inf``. Rounding up is the
    safer error for a predicted MIC (CLAUDE.md rule 9).
    """
    value = _as_positive_float(mic)
    if math.isinf(value):
        return INF
    return math.ldexp(1.0, _exponent(value, "up"))


def round_down_to_step(mic: float) -> float:
    """Largest ``2**k <= mic``; exact powers of two are returned unchanged.

    ``6 -> 4``, ``0.19 -> 0.125``, ``8 -> 8``, ``inf -> inf``. Used for lower bounds
    (band lows, ``>`` results) so an interval is widened rather than narrowed.
    """
    value = _as_positive_float(mic)
    if math.isinf(value):
        return INF
    return math.ldexp(1.0, _exponent(value, "down"))


def round_to_nearest_step(mic: float) -> float:
    """Closest ``2**k`` in log space; geometric midpoints (``2**2.5``) round up.

    ``3 -> 4``, ``2.5 -> 2``, ``6 -> 8``, ``0.17 -> 0.125``, ``inf -> inf``.
    """
    value = _as_positive_float(mic)
    if math.isinf(value):
        return INF
    return math.ldexp(1.0, _exponent(value, "nearest"))


def steps_between(a: float, b: float) -> float:
    """Doubling steps from ``a`` to ``b``: ``log2(b) - log2(a)`` (``1, 8 -> 3``; ``8, 1 -> -3``).

    Either argument may be ``inf``; the result is then ``+-inf`` (or NaN for
    ``inf, inf``). Both must be positive.
    """
    return log2_step(b) - log2_step(a)


def normalize_sign(sign: str | None) -> str:
    """Canonical measurement sign: one of ``=``, ``<=``, ``<``, ``>``, ``>=``.

    ``None``, NaN, ``pd.NA`` and blank mean ``=``. Accepts ``==``, ``=<``, ``=>`` and
    the Unicode ``≤``/``≥``. Anything else raises ``ValueError``.
    """
    if _is_missing(sign):
        return "="
    key = str(sign).strip()
    try:
        return _SIGN_ALIASES[key]
    except KeyError:
        raise ValueError(f"unknown measurement sign {sign!r}") from None


def normalize_sir(sir: str) -> str:
    """Canonical S/I/R code from ``S``/``I``/``R`` or the full words, any case.

    Raises ``ValueError`` for anything else (``SDD``, ``NS``, blanks): the caller
    decides what to do with those, this module does not guess.
    """
    if _is_missing(sir):
        raise ValueError("sir is missing")
    key = str(sir).strip().upper()
    try:
        return _SIR_ALIASES[key]
    except KeyError:
        raise ValueError(f"unknown S/I/R value {sir!r}") from None


def snap_reported_mic(value: float) -> float:
    """Snap a decimal rendering of a power of two to that power; leave anything else as is.

    Labs print ``2**-6`` as ``0.016`` or ``0.015``, ``2**-7`` as ``0.008`` and ``2**-3``
    as ``0.12``. Fed straight into :func:`interval_from_result`, ``0.016`` would land in
    the cell above (``(2**-6, 2**-5]``) and ``> 0.03`` would widen to ``(2**-6, inf)``.
    Returns ``2**k`` when ``|log2(value) - k| <= SNAP_TOL_LOG2`` for an integer ``k``,
    else ``value`` unchanged (gradient half steps such as ``0.19`` or ``6`` keep their
    own grid cell). Ingest calls this before the interval rule; nothing else should.

    Raises ``ValueError`` for a missing, non-positive or non-finite value.
    """
    number = _as_positive_float(value, "MIC value")
    if math.isinf(number):
        raise ValueError("MIC value must be finite")
    l2 = math.log2(number)
    k = round(l2)
    if abs(l2 - k) <= SNAP_TOL_LOG2:
        return math.ldexp(1.0, k)
    return number


def interval_from_result(sign: str | None, value: float | None) -> tuple[float, float, str]:
    """Interval ``(lo, hi]`` and censor type for a numeric MIC result.

    Let ``g = round_up_to_step(value)`` (the top of the grid cell containing ``value``):

    * ``=`` or no sign -> ``(g/2, g, "interval")``       ``=8 -> (4, 8]``; ``=6 -> (4, 8]``
    * ``<=``           -> ``(0, g, "left")``             ``<=0.25 -> (0, 0.25]``
    * ``<``            -> ``(0, g/2, "left")``           ``<0.5 -> (0, 0.25]``  (at most the step below)
    * ``>``            -> ``(round_down(value), inf, "right")``  ``>32 -> (32, inf)``; ``>6 -> (4, inf)``
    * ``>=``           -> ``(g/2, inf, "right")``        ``>=16 -> (8, inf)``; ``>=6 -> (4, inf)``

    Censored bounds for off-grid values are widened to the grid so the interval
    stays a true statement. Raises ``ValueError`` when ``value`` is missing,
    non-positive or infinite, or the sign is unknown; S/I/R-only rows go through
    :func:`interval_from_sir` instead.
    """
    op = normalize_sign(sign)
    mic = _as_positive_float(value, "MIC value")
    if math.isinf(mic):
        raise ValueError("MIC value must be finite")

    top = round_up_to_step(mic)  # top of the grid cell containing the value
    below = top / 2.0  # the step below that cell

    if op == "=":
        return (below, top, CENSOR_INTERVAL)
    if op == "<=":
        return (0.0, top, CENSOR_LEFT)
    if op == "<":
        return (0.0, below, CENSOR_LEFT)
    if op == ">":
        return (round_down_to_step(mic), INF, CENSOR_RIGHT)
    if op == ">=":
        return (below, INF, CENSOR_RIGHT)
    raise ValueError(f"unhandled sign {op!r}")  # pragma: no cover - normalize_sign guards this


def interval_from_sir(sir: str, bp: BreakpointLike) -> tuple[float, float, str]:
    """Interval for an S/I/R-only result under breakpoint ``bp`` (duck-typed).

    * ``S`` -> ``(0, s, "left")``
    * ``R`` -> ``(r, inf, "right")``
    * ``I`` -> ``(s, r, "interval")``; raises ``ValueError`` when ``s == r`` (no I category)

    Breakpoints are normally powers of two and pass through unchanged. An off-grid
    breakpoint (EUCAST's ``0.001`` placeholder) is widened to the grid (lower bounds
    round down, upper bounds round up) so bounds stay grid-aligned without ever
    narrowing the interval. Raises ``ValueError`` for an unknown ``sir`` or an
    inconsistent breakpoint (``r < s``, non-positive, non-finite).
    """
    code = normalize_sir(sir)
    s = _as_positive_float(bp.s_breakpoint, "s_breakpoint")
    r = _as_positive_float(bp.r_breakpoint, "r_breakpoint")
    if math.isinf(s) or math.isinf(r):
        raise ValueError("breakpoints must be finite")
    if r < s and not math.isclose(r, s):
        raise ValueError(f"r_breakpoint ({r}) must be >= s_breakpoint ({s})")

    if code == "S":
        return (0.0, round_up_to_step(s), CENSOR_LEFT)
    if code == "R":
        return (round_down_to_step(r), INF, CENSOR_RIGHT)
    # I
    if math.isclose(s, r):
        raise ValueError(f"'I' is invalid when s_breakpoint == r_breakpoint ({s})")
    return (round_down_to_step(s), round_up_to_step(r), CENSOR_INTERVAL)


def label_point_log2(lo: float, hi: float) -> float:
    """The reported step of an interval as log2 MIC (used by B1/B2 and residuals).

    * interval / left-censored -> ``log2(hi)``   (``(4, 8] -> 3``; ``(0, 0.25] -> -2``)
    * right-censored           -> ``log2(lo) + 1`` (``(32, inf) -> 6``: the step above the edge)

    Raises ``ValueError`` for ``lo >= hi``, negative ``lo`` or the uninformative
    ``(0, inf)``.
    """
    low = _as_bound(lo, "lo")
    high = _as_bound(hi, "hi")
    if low < 0:
        raise ValueError(f"lo must be >= 0, got {lo!r}")
    if not high > low:
        raise ValueError(f"need lo < hi, got ({lo!r}, {hi!r})")
    if math.isinf(high):
        if low <= 0:
            raise ValueError("(0, inf) carries no information; refuse to assign a step")
        return math.log2(low) + 1.0
    return math.log2(high)


def censor_of(lo: float, hi: float) -> str:
    """Censor label implied by the bounds: ``lo == 0`` -> left; ``hi == inf`` -> right; else interval.

    Raises ``ValueError`` for ``lo >= hi``, negative ``lo`` or ``(0, inf)``.
    """
    low = _as_bound(lo, "lo")
    high = _as_bound(hi, "hi")
    if low < 0:
        raise ValueError(f"lo must be >= 0, got {lo!r}")
    if not high > low:
        raise ValueError(f"need lo < hi, got ({lo!r}, {hi!r})")
    if low == 0 and math.isinf(high):
        raise ValueError("(0, inf) is neither left- nor right-censored")
    if low == 0:
        return CENSOR_LEFT
    if math.isinf(high):
        return CENSOR_RIGHT
    return CENSOR_INTERVAL


def _as_bound(value: Any, name: str) -> float:
    if _is_missing(value):
        raise ValueError(f"{name} is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if math.isnan(number):
        raise ValueError(f"{name} is NaN")
    return number


# -------------------------------------------------------------------- array API
def exact_interval_mask(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Rows that pin the MIC to one doubling step: ``lo > 0``, ``hi < inf`` and ``hi == 2 * lo``.

    This is the single definition of an "exact MIC" used by B2, conformal residuals,
    EA / exact agreement / band coverage and the count table. An ``I``-only result
    such as ``(2, 8]`` has ``censor == 'interval'`` but spans two steps, so it is
    *not* exact. NaN bounds are never exact.
    """
    low = np.asarray(lo, dtype=np.float64)
    high = np.asarray(hi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (low > 0) & np.isfinite(high) & (high > low)
        width = np.where(ok, np.log2(np.where(ok, high, 1.0)) - np.log2(np.where(ok, low, 0.5)), np.nan)
    return ok & np.isclose(width, 1.0, atol=1e-6, rtol=0.0)


METHOD_DISK = "disk"
"""``labels.method`` value of disk-diffusion results (zone diameter -> S/I/R, never an MIC)."""


def lab_exact_mask(lo: np.ndarray, hi: np.ndarray, method: Any = None) -> np.ndarray:
    """Rows whose lab result is an **exact measured MIC**: one doubling step and not disk diffusion.

    :func:`exact_interval_mask` judges the interval alone. A disk-diffusion result
    reaches stage 2 through the S/I/R path, so its interval is a breakpoint range,
    not a reading; when the ``I`` range happens to be one doubling step (CLSI
    meropenem ``I`` = ``(1, 2]``) the interval looks exact but no MIC was measured
    (contract stage 2, method filter: disk -> "MIC not usable"). Training (B2,
    conformal residuals), the ``lab_exact`` preds column and the count table use
    this mask.

    Args:
        lo, hi: Lab interval bounds (mg/L).
        method: ``labels.method`` per row (``dilution`` / ``gradient`` / ``disk``;
            compared case-insensitively, surrounding blanks ignored). Null entries
            are not disk. ``None`` (no method information) applies the interval
            rule alone.
    """
    exact = exact_interval_mask(lo, hi)
    if method is None:
        return exact
    values = np.asarray(method, dtype=object).ravel()
    if values.size != exact.size:
        raise ValueError(f"method must have the same length as the bounds ({values.size} != {exact.size})")
    disk = np.fromiter(
        (isinstance(v, str) and v.strip().lower() == METHOD_DISK for v in values), dtype=bool, count=values.size
    )
    return exact & ~disk


def round_up_to_step_array(mic: np.ndarray) -> np.ndarray:
    """Vectorized :func:`round_up_to_step`. NaN stays NaN (missing), ``inf`` stays ``inf``.

    Raises ``ValueError`` if any non-missing value is ``<= 0``.
    """
    return _round_array(mic, "up")


def round_down_to_step_array(mic: np.ndarray) -> np.ndarray:
    """Vectorized :func:`round_down_to_step`. NaN stays NaN, ``inf`` stays ``inf``."""
    return _round_array(mic, "down")


def _round_array(mic: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(mic, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    missing = np.isnan(x)
    if np.any(x[~missing] <= 0):
        raise ValueError("MIC values must be > 0")
    infinite = np.isinf(x) & ~missing
    out[infinite] = np.inf
    finite = ~missing & ~infinite
    if not np.any(finite):
        return out
    l2 = np.log2(x[finite])
    nearest = np.round(l2)
    on_grid = np.isclose(l2, nearest, atol=_GRID_TOL, rtol=0.0)
    rounded = np.ceil(l2) if mode == "up" else np.floor(l2)
    k = np.where(on_grid, nearest, rounded).astype(np.int32)
    out[finite] = np.ldexp(1.0, k)
    return out


def panel_caps_log2(lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """Panel-edge caps ``(cap_low, cap_high)`` in log2 mg/L from the *fitting* rows' lab intervals.

    "Finite bounds" are every ``mic_lower > 0`` and every finite ``mic_upper`` of the
    rows given. ``cap_low = log2(min finite bound) - 1`` and ``cap_high =
    log2(max finite bound) + 1``: one doubling step beyond the lowest and highest
    concentrations the training panels actually tested, so a prediction never wanders
    further past a panel edge than the data can support. Bounds are snapped to the
    doubling grid first (low down, high up), so the caps are whole steps. With no
    finite bound at all the reference grid edges are returned.

    Callers pass the rows a model is *fitted* on (a CV fold's training folds, all
    train rows for the final bundle), never the rows it predicts.
    """
    low = np.asarray(lo, dtype=np.float64).ravel()
    high = np.asarray(hi, dtype=np.float64).ravel()
    finite = np.concatenate([low[np.isfinite(low) & (low > 0)], high[np.isfinite(high) & (high > 0)]])
    if finite.size == 0:
        return float(GRID_MIN_EXPONENT), float(GRID_MAX_EXPONENT)
    bottom = float(np.log2(round_down_to_step_array(np.array([finite.min()]))[0]))
    top = float(np.log2(round_up_to_step_array(np.array([finite.max()]))[0]))
    cap_low = max(float(GRID_MIN_EXPONENT), bottom - 1.0)
    cap_high = min(float(GRID_MAX_EXPONENT), top + 1.0)
    return cap_low, cap_high
