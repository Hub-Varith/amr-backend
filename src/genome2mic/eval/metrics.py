"""Pure metric functions for MIC predictions and the stage-11 summary table.

Inputs follow the stage-10 predictions schema in ``DATA_CONTRACT.md``: a point
prediction ``pred_mic`` (mg/L, rounded **up** to the doubling grid), a 90% band
``band_low``/``band_high``, the lab interval ``(lab_lower, lab_upper]`` copied from
stage 2, the S/I/R calls ``pred_sir``/``lab_sir`` and, optionally,
``lab_sir_rederived`` (see *Re-derived lab S/I/R* below).

Exact and censored lab rows
---------------------------
A lab result is never a point; it is an interval ``(lo, hi]`` on the doubling grid.
``(4, 8]`` is an exact reading of 8, ``(0, 0.25]`` is ``<= 0.25`` (left-censored),
``(32, inf)`` is ``> 32`` (right-censored), and an ``I``-only label can be a
multi-step interval such as ``(2, 8]``.

An **exact** row is one whose lab result is a measured MIC of one doubling step:
the lab interval pins the MIC to one step (:func:`genome2mic.mic.exact_interval_mask`:
``lo > 0``, ``hi < inf``, ``hi == 2 * lo``) **and**, when the preds carry the
boolean ``lab_exact`` column written by training, ``lab_exact`` is true. Training
sets ``lab_exact`` false for disk-diffusion results (:func:`genome2mic.mic.lab_exact_mask`):
a disk ``I``-only row whose ``I`` range is one step (CLSI meropenem ``(1, 2]``) has an
exact-looking interval but no MIC was measured (contract stage 2, method filter).
Older preds without the column fall back to the interval rule; a null ``lab_exact``
counts as not exact. See :func:`exact_lab_mask`.

:func:`summarize` evaluates **essential agreement, exact agreement and band
coverage on exact rows only**, so ``n_exact`` is the denominator of EA and exact
agreement (contract section 7) and ``n_band`` the denominator of band coverage.
Censored rows would make EA degenerate (``(X, inf)`` agrees with any ``pred >= X``)
and band coverage would no longer be the quantity the conformal ``q`` calibrates
(``q`` comes from the same exact rows' residuals).

The per-row functions below still accept any informative interval, with these
rules (they reduce to the DESIGN.md wording on the three common cases):

* **Essential agreement** -- the prediction is within one doubling step of the
  interval: ``lo <= pred <= 2 * hi``.
  Exact ``(4, 8]``: ``|log2(pred) - 3| <= 1``.
  Left ``(0, X]``: ``pred <= 2X`` (at most one step above the censoring edge).
  Right ``(X, inf)``: ``pred >= X`` (at most one step below the first step inside).
* **Exact agreement** -- the prediction lies inside the interval: ``lo < pred <= hi``.
  Exact rows: same step. Left: ``pred <= X``. Right: ``pred > X``.
* **Band coverage** -- the band overlaps the interval: ``band_low <= hi`` and
  ``band_high > lo``. For an exact row this is ``band_low <= hi <= band_high``
  (the reported step is inside the band); for a censored row it is overlap.

A ``(0, inf)`` interval carries no information and is never evaluated.

Re-derived lab S/I/R
--------------------
``lab_sir`` is the lab's own call under the lab's standard and year, while
``pred_sir`` is the prediction classified under the call breakpoint. The training
stage also writes ``lab_sir_rederived``: the lab interval classified under the
same call breakpoint (null where the interval straddles a breakpoint or there is
no call breakpoint). :func:`summarize` reports the categorical metrics against both:
the as-reported block (``vme_rate`` ...) and the ``*_rederived`` block, computed the
same way. Without the column the re-derived rates are NaN and their counts 0.

Predictions are snapped **up** to the grid and bands are widened to it (low down,
high up) before comparing; this is a no-op for contract-compliant inputs and only
guards against float drift and off-grid values. All comparisons run in ``log2``
space with a tolerance of 1e-9 steps.

Null handling
-------------
The per-row functions (:func:`essential_agreement`, :func:`exact_agreement`,
:func:`band_covers`) and their means require complete inputs and raise
``ValueError`` on nulls -- a boolean array cannot carry a null, and silently
counting a missing row as a miss would bias the metric. :func:`categorical` and
:func:`auroc` skip null rows and report how many. :func:`summarize` does the
masking for a whole predictions table and records every exclusion in a
:class:`~genome2mic.droplog.DropLog`; rows with a null ``pred_mic`` (e.g.
``b0_resfinder``) contribute only to the categorical metrics.

Column order of :func:`summarize` puts ``vme_rate`` first after the keys: it is the
error that harms a patient (CLAUDE.md rule 10). The re-derived block follows the
counts and starts with ``vme_rate_rederived``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from genome2mic.droplog import DropLog
from genome2mic.mic import exact_interval_mask, round_down_to_step_array, round_up_to_step_array
from genome2mic.models.calibration import FAILS_TIERS, PROB_TIERS, WORKS_TIERS, prob_tier
from genome2mic.models.conformal import DEFAULT_FOLD_P_THRESHOLD, DEFAULT_VME_TARGET, binomial_excess_p

logger = logging.getLogger(__name__)

ArrayLike = Sequence[Any] | np.ndarray | pd.Series

SIR_VALUES: tuple[str, ...] = ("S", "I", "R")

KEY_COLUMNS: tuple[str, ...] = ("species", "drug", "model", "split")
CALL_METRIC_COLUMNS: tuple[str, ...] = ("call_vme_rate", "call_me_rate", "active_call_rate_s", "uncertain_rate")
"""Call-level safety metrics over the preds ``call`` column (the call a clinician would see):
``call_vme_rate`` = lab R called ``likely_active`` / lab R; ``call_me_rate`` = lab S called
``likely_inactive`` / lab S; ``active_call_rate_s`` = lab S called ``likely_active`` / lab S
(usefulness); ``uncertain_rate`` = ``uncertain`` / rows with a call and a lab category."""
CALL_COUNT_COLUMNS: tuple[str, ...] = ("n_call", "n_call_lab_r", "n_call_lab_s")
CALL_VALUES: tuple[str, ...] = ("likely_active", "uncertain", "likely_inactive")
CALL_COLUMN = "call"
"""Optional preds column written by training (:func:`genome2mic.predict.rank.call_array`)."""

METRIC_COLUMNS: tuple[str, ...] = (
    "vme_rate",
    *CALL_METRIC_COLUMNS,
    "me_rate",
    "mine_rate",
    "categorical_agreement",
    "essential_agreement",
    "exact_agreement",
    "auroc",
    "band_coverage",
    "band_width_steps",
)
COUNT_COLUMNS: tuple[str, ...] = ("n", "n_exact", "n_band", "n_cat", "n_lab_r", "n_lab_s", *CALL_COUNT_COLUMNS)
"""``n`` rows in the group; ``n_exact`` = EA / exact-agreement denominator; ``n_band`` =
band-coverage denominator; ``n_cat`` / ``n_lab_r`` / ``n_lab_s`` = CA / VME / ME denominators."""

REDERIVED_RATE_COLUMNS: tuple[str, ...] = (
    "vme_rate_rederived",
    *(f"{c}_rederived" for c in CALL_METRIC_COLUMNS),
    "me_rate_rederived",
    "mine_rate_rederived",
    "categorical_agreement_rederived",
)
REDERIVED_COUNT_COLUMNS: tuple[str, ...] = (
    "n_cat_rederived", "n_lab_r_rederived", "n_lab_s_rederived",
    *(f"{c}_rederived" for c in CALL_COUNT_COLUMNS),
)
REDERIVED_COLUMNS: tuple[str, ...] = REDERIVED_RATE_COLUMNS + REDERIVED_COUNT_COLUMNS
"""Categorical metrics against ``lab_sir_rederived``; ``vme_rate_rederived`` first."""

SUMMARY_COLUMNS: tuple[str, ...] = KEY_COLUMNS + METRIC_COLUMNS + COUNT_COLUMNS + REDERIVED_COLUMNS
"""Column order of :func:`summarize`. ``vme_rate`` is the first metric."""

BIN_COLUMNS: tuple[str, ...] = ("distance_bin", "bin_low", "bin_high")
DISTANCE_COLUMNS: tuple[str, ...] = KEY_COLUMNS + BIN_COLUMNS + METRIC_COLUMNS + COUNT_COLUMNS + REDERIVED_COLUMNS
"""Column order of :func:`by_distance_bin`."""

REQUIRED_PRED_COLUMNS: tuple[str, ...] = KEY_COLUMNS + (
    "pred_mic",
    "lab_lower",
    "lab_upper",
    "pred_sir",
    "lab_sir",
)
BAND_COLUMNS: tuple[str, ...] = ("band_low", "band_high")
REDERIVED_SIR_COLUMN = "lab_sir_rederived"
"""Optional preds column: the lab interval classified under the call breakpoint."""
LAB_EXACT_COLUMN = "lab_exact"
"""Optional boolean preds column: the lab result is an exact measured MIC (one step, not disk)."""

INTERVAL_NOT_EXACT_REASON = (
    "lab interval censored or wider than one doubling step: excluded from EA, exact agreement and band coverage"
)
NOT_MEASURED_REASON = (
    "lab_exact false (one-step interval but not a measured MIC, e.g. disk diffusion): "
    "excluded from EA, exact agreement and band coverage"
)
"""Drop reason for one-step lab intervals that ``lab_exact`` marks as not an MIC."""
LAB_EXACT_NULL_REASON = "lab_exact null: treated as not an exact MIC"

_FLOAT_COLUMNS: frozenset[str] = frozenset(METRIC_COLUMNS + REDERIVED_RATE_COLUMNS + ("bin_low", "bin_high"))
_INT_COLUMNS: frozenset[str] = frozenset(COUNT_COLUMNS + REDERIVED_COUNT_COLUMNS)

DEFAULT_DISTANCE_BINS: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, math.inf)
"""Mash-distance edges for :func:`by_distance_bin`: 0.001 ~ same clone, 0.05 ~ species edge."""

# Tolerance in log2 space (doubling steps) below which two grid positions are equal.
_TOL = 1e-9

_BIN_CODE = "_distance_bin_code"


# --------------------------------------------------------------------------- coercion
def _as_float_array(values: ArrayLike, name: str) -> np.ndarray:
    """1-D float64 array; ``None`` becomes NaN. Raises ``ValueError`` on non-numeric input."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _as_sir_array(values: ArrayLike, name: str) -> np.ndarray:
    """Object array of ``'S'``/``'I'``/``'R'`` or ``None`` (any null representation -> ``None``).

    Raises ``ValueError`` for any other value (``'s'``, ``'Susceptible'``, ``''`` ...);
    this module does not normalise labels, stage 2 does.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values), dtype=object)
    raw = series.to_numpy(dtype=object)
    out = np.empty(len(raw), dtype=object)
    bad: set[str] = set()
    for i, v in enumerate(raw):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            out[i] = None
        elif v in SIR_VALUES:
            out[i] = v
        else:
            bad.add(repr(v))
    if bad:
        raise ValueError(f"{name} contains values outside S/I/R: {sorted(bad)}")
    return out


def _check_same_length(**arrays: np.ndarray) -> int:
    lengths = {name: len(arr) for name, arr in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"inputs must have the same length, got {lengths}")
    return next(iter(lengths.values()))


def _log2(x: np.ndarray) -> np.ndarray:
    """``log2`` with ``0 -> -inf`` and ``inf -> inf`` and no warnings."""
    with np.errstate(divide="ignore"):
        return np.log2(x)


# ------------------------------------------------------------------------- validation
def _validate_pred(pred: np.ndarray) -> None:
    if np.isnan(pred).any():
        raise ValueError(
            "pred_mic contains nulls; filter rows without a prediction first (summarize does this)"
        )
    if (pred <= 0).any():
        raise ValueError("pred_mic must be > 0 mg/L on every row")


def _validate_lab_interval(lo: np.ndarray, hi: np.ndarray) -> None:
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError(
            "lab_lower/lab_upper contain nulls; filter rows without a lab interval first (summarize does this)"
        )
    if (lo < 0).any():
        raise ValueError("lab_lower must be >= 0 on every row")
    if not (lo < hi).all():
        raise ValueError("lab interval must satisfy lab_lower < lab_upper on every row")
    if ((lo == 0) & np.isinf(hi)).any():
        raise ValueError("lab interval (0, inf) carries no information and cannot be evaluated")


def _validate_band(low: np.ndarray, high: np.ndarray) -> None:
    if np.isnan(low).any() or np.isnan(high).any():
        raise ValueError(
            "band_low/band_high contain nulls; filter rows without a band first (summarize does this)"
        )
    if (low <= 0).any():
        raise ValueError("band_low must be > 0 mg/L on every row")
    if np.isinf(low).any():
        raise ValueError("band_low must be finite")
    if (low > high).any():
        raise ValueError("band must satisfy band_low <= band_high on every row")


def _pred_steps(pred: np.ndarray) -> np.ndarray:
    """log2 of the prediction snapped up to the doubling grid (contract: round up)."""
    return _log2(round_up_to_step_array(pred))


def _band_steps(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log2 of the band widened to the grid: low rounds down, high rounds up."""
    return _log2(round_down_to_step_array(low)), _log2(round_up_to_step_array(high))


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _lab_exact_values(preds: pd.DataFrame) -> tuple[np.ndarray | None, int]:
    """``(lab_exact as bool with nulls -> False, number of nulls)``; ``(None, 0)`` without the column."""
    if LAB_EXACT_COLUMN not in preds.columns:
        return None, 0
    try:
        values = pd.array(preds[LAB_EXACT_COLUMN].to_numpy(dtype=object), dtype="boolean")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{LAB_EXACT_COLUMN} must be boolean (true / false / null)") from exc
    nulls = values.isna()
    return np.asarray(values.fillna(False), dtype=bool), int(nulls.sum())


def exact_lab_mask(preds: pd.DataFrame) -> np.ndarray:
    """Rows of a preds table with an exact lab MIC: the rows EA, exact agreement and band coverage use.

    One-step lab interval (:func:`genome2mic.mic.exact_interval_mask`) and, when the
    ``lab_exact`` column is present, ``lab_exact`` true (null -> not exact). The
    column can only narrow the interval rule. Shared with the MIC confusion figure.
    """
    lo = _as_float_array(preds["lab_lower"], "lab_lower")
    hi = _as_float_array(preds["lab_upper"], "lab_upper")
    mask = exact_interval_mask(lo, hi)
    lab_exact, _ = _lab_exact_values(preds)
    return mask if lab_exact is None else mask & lab_exact


# ------------------------------------------------------------------- per-row metrics
def is_exact_interval(lab_lower: ArrayLike, lab_upper: ArrayLike) -> np.ndarray:
    """True where the lab interval pins the MIC to one doubling step (``hi == 2 * lo``).

    Delegates to :func:`genome2mic.mic.exact_interval_mask`, the project's single
    definition of an exact MIC. Censored intervals, multi-step ``I``-only intervals
    such as ``(2, 8]`` and null bounds give False. Only these rows enter EA, exact
    agreement and band coverage in :func:`summarize`.
    """
    lo = _as_float_array(lab_lower, "lab_lower")
    hi = _as_float_array(lab_upper, "lab_upper")
    _check_same_length(lab_lower=lo, lab_upper=hi)
    return exact_interval_mask(lo, hi)


def essential_agreement(pred_mic: ArrayLike, lab_lower: ArrayLike, lab_upper: ArrayLike) -> np.ndarray:
    """Per-row essential agreement: prediction within one doubling step of the lab interval.

    Rule: ``lab_lower <= pred <= 2 * lab_upper`` after snapping ``pred`` up to the grid.

    * exact ``(4, 8]``   -> ``|log2(pred) - log2(8)| <= 1``  (pred in {4, 8, 16})
    * left  ``(0, X]``   -> ``pred <= 2X``   (one step above the censoring edge still agrees)
    * right ``(X, inf)`` -> ``pred >= X``    (one step below the first in-interval step agrees)
    * multi-step ``(2, 8]`` -> ``2 <= pred <= 16``

    Returns a bool array. Raises ``ValueError`` on nulls, non-positive predictions,
    ``lab_lower >= lab_upper`` or a ``(0, inf)`` interval; see the module docstring.
    """
    pred = _as_float_array(pred_mic, "pred_mic")
    lo = _as_float_array(lab_lower, "lab_lower")
    hi = _as_float_array(lab_upper, "lab_upper")
    n = _check_same_length(pred_mic=pred, lab_lower=lo, lab_upper=hi)
    if n == 0:
        return np.zeros(0, dtype=bool)
    _validate_pred(pred)
    _validate_lab_interval(lo, hi)
    p = _pred_steps(pred)
    return (p >= _log2(lo) - _TOL) & (p <= _log2(hi) + 1.0 + _TOL)


def exact_agreement(pred_mic: ArrayLike, lab_lower: ArrayLike, lab_upper: ArrayLike) -> np.ndarray:
    """Per-row exact agreement: prediction inside the lab interval ``(lab_lower, lab_upper]``.

    * exact ``(4, 8]``   -> same doubling step (pred == 8 after snapping up)
    * left  ``(0, X]``   -> ``pred <= X``
    * right ``(X, inf)`` -> ``pred > X``
    * multi-step ``(2, 8]`` -> pred in {4, 8}

    Returns a bool array. Same input requirements as :func:`essential_agreement`.
    """
    pred = _as_float_array(pred_mic, "pred_mic")
    lo = _as_float_array(lab_lower, "lab_lower")
    hi = _as_float_array(lab_upper, "lab_upper")
    n = _check_same_length(pred_mic=pred, lab_lower=lo, lab_upper=hi)
    if n == 0:
        return np.zeros(0, dtype=bool)
    _validate_pred(pred)
    _validate_lab_interval(lo, hi)
    p = _pred_steps(pred)
    return (p > _log2(lo) + _TOL) & (p <= _log2(hi) + _TOL)


def band_covers(
    band_low: ArrayLike, band_high: ArrayLike, lab_lower: ArrayLike, lab_upper: ArrayLike
) -> np.ndarray:
    """Per-row band coverage: the band ``[band_low, band_high]`` overlaps ``(lab_lower, lab_upper]``.

    Rule: ``band_low <= lab_upper`` and ``band_high > lab_lower`` after widening the
    band to the grid (low down, high up).

    * exact ``(4, 8]``   -> ``band_low <= 8 <= band_high`` (the reported step is in the band)
    * left  ``(0, X]``   -> ``band_low <= X``
    * right ``(X, inf)`` -> ``band_high > X`` (a band topping out exactly at X misses)

    Returns a bool array. Raises ``ValueError`` on nulls, ``band_low <= 0``,
    ``band_low > band_high`` or an invalid lab interval.
    """
    low = _as_float_array(band_low, "band_low")
    high = _as_float_array(band_high, "band_high")
    lo = _as_float_array(lab_lower, "lab_lower")
    hi = _as_float_array(lab_upper, "lab_upper")
    n = _check_same_length(band_low=low, band_high=high, lab_lower=lo, lab_upper=hi)
    if n == 0:
        return np.zeros(0, dtype=bool)
    _validate_band(low, high)
    _validate_lab_interval(lo, hi)
    bl, bh = _band_steps(low, high)
    return (bl <= _log2(hi) + _TOL) & (bh > _log2(lo) + _TOL)


# ------------------------------------------------------------------ aggregate metrics
def categorical(pred_sir: ArrayLike, lab_sir: ArrayLike) -> dict[str, float | int | None]:
    """Categorical agreement and the CLSI / ISO 20776-2 error rates.

    Rows where either side is null are excluded and counted in ``n_excluded``.
    Any other value outside ``S``/``I``/``R`` raises ``ValueError``.

    Definitions (over the ``n_cat`` complete rows):

    * ``vme_rate`` = (pred S and lab R) / ``n_lab_r``  -- very major error, reported first
    * ``me_rate``  = (pred R and lab S) / ``n_lab_s``  -- major error
    * ``mine_rate`` = (exactly one side is I) / ``n_cat``  -- minor error (I vs I agrees)
    * ``ca`` = (pred == lab) / ``n_cat``

    A rate is ``None`` when its denominator is zero. Returned keys: ``ca``, ``vme_rate``,
    ``me_rate``, ``mine_rate``, ``n_cat``, ``n_lab_r``, ``n_lab_s``, ``n_vme``, ``n_me``,
    ``n_mine``, ``n_agree``, ``n_excluded``.
    """
    pred = _as_sir_array(pred_sir, "pred_sir")
    lab = _as_sir_array(lab_sir, "lab_sir")
    _check_same_length(pred_sir=pred, lab_sir=lab)

    complete = np.array([p is not None and l is not None for p, l in zip(pred, lab)], dtype=bool)
    n_excluded = int((~complete).sum())
    p = pred[complete]
    l = lab[complete]
    n_cat = int(len(p))

    lab_r = l == "R"
    lab_s = l == "S"
    n_lab_r = int(lab_r.sum())
    n_lab_s = int(lab_s.sum())
    n_vme = int(((p == "S") & lab_r).sum())
    n_me = int(((p == "R") & lab_s).sum())
    n_mine = int(((p == "I") ^ (l == "I")).sum())
    n_agree = int((p == l).sum())

    return {
        "ca": _ratio(n_agree, n_cat),
        "vme_rate": _ratio(n_vme, n_lab_r),
        "me_rate": _ratio(n_me, n_lab_s),
        "mine_rate": _ratio(n_mine, n_cat),
        "n_cat": n_cat,
        "n_lab_r": n_lab_r,
        "n_lab_s": n_lab_s,
        "n_vme": n_vme,
        "n_me": n_me,
        "n_mine": n_mine,
        "n_agree": n_agree,
        "n_excluded": n_excluded,
    }


def _as_call_array(values: ArrayLike, name: str = CALL_COLUMN) -> np.ndarray:
    """Object array of call strings with ``None`` for nulls; any other value raises."""
    raw = values.to_numpy(dtype=object, na_value=None) if isinstance(values, pd.Series) else np.asarray(values, dtype=object)
    out = np.empty(len(raw), dtype=object)
    for i, value in enumerate(raw):
        if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
            out[i] = None
        elif isinstance(value, str) and value in CALL_VALUES:
            out[i] = value
        else:
            raise ValueError(f"{name} must be one of {CALL_VALUES} or null, got {value!r}")
    return out


def call_metrics(call: ArrayLike, lab_sir: ArrayLike) -> dict[str, float | int | None]:
    """Call-level safety metrics: what a clinician would see (the call), against the lab category.

    Rows with a null call or lab category are excluded (``n_call_excluded``). Over the
    ``n_call`` complete rows:

    * ``call_vme_rate`` = (call ``likely_active`` and lab R) / ``n_call_lab_r`` -- the
      call-level very major error, the one that harms a patient;
    * ``call_me_rate`` = (call ``likely_inactive`` and lab S) / ``n_call_lab_s``;
    * ``active_call_rate_s`` = (call ``likely_active`` and lab S) / ``n_call_lab_s`` --
      usefulness: how many truly susceptible isolates get an actionable answer;
    * ``uncertain_rate`` = ``uncertain`` / ``n_call`` ("wait for the lab").

    ``uncertain`` is never an error. A rate is ``None`` when its denominator is zero.
    """
    calls = _as_call_array(call)
    lab = _as_sir_array(lab_sir, "lab_sir")
    _check_same_length(call=calls, lab_sir=lab)
    complete = np.array([c is not None and l is not None for c, l in zip(calls, lab)], dtype=bool)
    c = calls[complete]
    l = lab[complete]
    lab_r = l == "R"
    lab_s = l == "S"
    active = c == "likely_active"
    n_call = int(len(c))
    n_lab_r = int(lab_r.sum())
    n_lab_s = int(lab_s.sum())
    return {
        "call_vme_rate": _ratio(int((active & lab_r).sum()), n_lab_r),
        "call_me_rate": _ratio(int(((c == "likely_inactive") & lab_s).sum()), n_lab_s),
        "active_call_rate_s": _ratio(int((active & lab_s).sum()), n_lab_s),
        "uncertain_rate": _ratio(int((c == "uncertain").sum()), n_call),
        "n_call": n_call,
        "n_call_lab_r": n_lab_r,
        "n_call_lab_s": n_lab_s,
        "n_call_excluded": int((~complete).sum()),
    }


def auroc(pred_log2: ArrayLike, lab_sir: ArrayLike) -> float | None:
    """AUROC of the predicted log2 MIC as a score for lab R (positive) versus lab S.

    ``I`` rows and rows with a null score or label are excluded (the contract defines
    this metric as R vs S). Scores are rank-transformed before
    ``sklearn.metrics.roc_auc_score`` so that ``inf`` (a prediction beyond the panel
    top) is handled; AUROC depends on ranks only, so nothing changes otherwise.

    Returns ``None`` when fewer than two rows remain or one class is missing.
    """
    scores = _as_float_array(pred_log2, "pred_log2")
    labels = _as_sir_array(lab_sir, "lab_sir")
    _check_same_length(pred_log2=scores, lab_sir=labels)

    keep = ~np.isnan(scores) & ((labels == "R") | (labels == "S"))
    s = scores[keep]
    y = (labels[keep] == "R").astype(int)
    if len(s) < 2 or y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, rankdata(s)))


def band_coverage(
    band_low: ArrayLike, band_high: ArrayLike, lab_lower: ArrayLike, lab_upper: ArrayLike
) -> float | None:
    """Share of rows whose band covers the lab interval (mean of :func:`band_covers`).

    Should be about 0.90 for a 90% conformal band when given exact rows (the rows
    the conformal ``q`` is calibrated on); :func:`summarize` passes exact rows only.
    ``None`` on empty input.
    """
    return _mean_or_none(band_covers(band_low, band_high, lab_lower, lab_upper))


def band_width_steps(band_low: ArrayLike, band_high: ArrayLike) -> float | None:
    """Mean band width in doubling steps: ``log2(band_high) - log2(band_low)``.

    The band is widened to the grid first (low down, high up). An infinite
    ``band_high`` gives an infinite width, which propagates to the mean. ``None`` on
    empty input. Raises ``ValueError`` on nulls or an invalid band.
    """
    low = _as_float_array(band_low, "band_low")
    high = _as_float_array(band_high, "band_high")
    n = _check_same_length(band_low=low, band_high=high)
    if n == 0:
        return None
    _validate_band(low, high)
    bl, bh = _band_steps(low, high)
    return float(np.mean(bh - bl))


# ------------------------------------------------------------------------ summarize
def _require_columns(preds: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in preds.columns]
    if missing:
        raise ValueError(f"preds is missing required columns: {missing}")


def _models_detail(models: np.ndarray, mask: np.ndarray) -> str | None:
    if not mask.any():
        return None
    names = sorted({str(m) for m in models[mask]})
    return "models: " + ", ".join(names)


def _summarize_groups(
    preds: pd.DataFrame, group_columns: Sequence[str], log: DropLog
) -> list[dict[str, Any]]:
    """Compute one metrics row per distinct value of ``group_columns``.

    Validates the whole table once, records every exclusion in ``log`` once (not per
    group), then evaluates each group on positional index arrays.
    """
    _require_columns(preds, REQUIRED_PRED_COLUMNS)
    for key in group_columns:
        if preds[key].isna().any():
            raise ValueError(f"key column {key!r} has null values")

    n_rows = len(preds)
    models = preds["model"].to_numpy(dtype=object)
    pred = _as_float_array(preds["pred_mic"], "pred_mic")
    lo = _as_float_array(preds["lab_lower"], "lab_lower")
    hi = _as_float_array(preds["lab_upper"], "lab_upper")
    pred_sir = _as_sir_array(preds["pred_sir"], "pred_sir")
    lab_sir = _as_sir_array(preds["lab_sir"], "lab_sir")
    has_rederived_column = REDERIVED_SIR_COLUMN in preds.columns
    if has_rederived_column:
        lab_sir_rederived = _as_sir_array(preds[REDERIVED_SIR_COLUMN], REDERIVED_SIR_COLUMN)
    else:
        logger.info("preds has no %s column; re-derived categorical metrics will be null", REDERIVED_SIR_COLUMN)
        lab_sir_rederived = np.full(n_rows, None, dtype=object)
    has_call_column = CALL_COLUMN in preds.columns
    if has_call_column:
        calls = _as_call_array(preds[CALL_COLUMN])
    else:
        logger.info("preds has no %s column; call-level metrics will be null", CALL_COLUMN)
        calls = np.full(n_rows, None, dtype=object)
    if all(c in preds.columns for c in BAND_COLUMNS):
        band_lo = _as_float_array(preds["band_low"], "band_low")
        band_hi = _as_float_array(preds["band_high"], "band_high")
    else:
        logger.info("preds has no band_low/band_high columns; band metrics will be null")
        band_lo = np.full(n_rows, np.nan)
        band_hi = np.full(n_rows, np.nan)

    # Whole-table validation of the non-null values (contract acceptance checks).
    has_pred = ~np.isnan(pred)
    if (pred[has_pred] <= 0).any():
        raise ValueError("pred_mic must be > 0 mg/L where present")
    has_lab = ~np.isnan(lo) & ~np.isnan(hi)
    if (lo[has_lab] < 0).any() or not (lo[has_lab] < hi[has_lab]).all():
        raise ValueError("lab interval must satisfy 0 <= lab_lower < lab_upper where present")
    if (np.isnan(lo) != np.isnan(hi)).any():
        raise ValueError("lab_lower and lab_upper must be null together")
    has_band = ~np.isnan(band_lo) & ~np.isnan(band_hi)
    if has_band.any():
        _validate_band(band_lo[has_band], band_hi[has_band])

    with np.errstate(invalid="ignore"):
        informative = has_lab & ~((lo == 0) & np.isinf(hi))
    interval_exact = has_lab & exact_interval_mask(lo, hi)
    lab_exact, n_lab_exact_null = _lab_exact_values(preds)
    exact_lab = interval_exact if lab_exact is None else interval_exact & lab_exact
    # EA, exact agreement and band coverage: exact (one-step, measured) lab MICs only.
    mic_ok = has_pred & exact_lab
    band_ok = has_band & exact_lab
    not_exact = has_pred & informative & ~interval_exact
    not_measured = has_pred & interval_exact & ~exact_lab
    has_sir = np.array([p is not None and l is not None for p, l in zip(pred_sir, lab_sir)], dtype=bool)
    has_sir_rederived = np.array([p is not None and l is not None for p, l in zip(pred_sir, lab_sir_rederived)], dtype=bool)
    lab_is_i = lab_sir == "I"
    auroc_ok = has_pred & ((lab_sir == "R") | (lab_sir == "S"))

    # Every filter logs a count, including zero counts, so the log proves it ran.
    log.drop("pred_mic null: categorical metrics only", int((~has_pred).sum()), _models_detail(models, ~has_pred))
    log.drop("lab_lower/lab_upper null: excluded from MIC metrics", int((~has_lab).sum()), _models_detail(models, ~has_lab))
    log.drop("lab interval (0, inf): excluded from MIC metrics", int((has_lab & ~informative).sum()))
    log.drop(INTERVAL_NOT_EXACT_REASON, int(not_exact.sum()), _models_detail(models, not_exact))
    log.drop(
        NOT_MEASURED_REASON,
        int(not_measured.sum()),
        _models_detail(models, not_measured) if lab_exact is not None else f"column {LAB_EXACT_COLUMN} absent: interval rule only",
    )
    if n_lab_exact_null:
        log.drop(LAB_EXACT_NULL_REASON, n_lab_exact_null)
    log.drop("pred_sir or lab_sir null: excluded from categorical metrics", int((~has_sir).sum()), _models_detail(models, ~has_sir))
    log.drop(
        f"pred_sir or {REDERIVED_SIR_COLUMN} null: excluded from re-derived categorical metrics",
        int((~has_sir_rederived).sum()),
        None if has_rederived_column else f"column {REDERIVED_SIR_COLUMN} absent",
    )
    log.drop("lab_sir I: excluded from AUROC", int((has_pred & lab_is_i).sum()))
    has_call = np.array([c is not None and l is not None for c, l in zip(calls, lab_sir)], dtype=bool)
    has_call_rederived = np.array([c is not None and l is not None for c, l in zip(calls, lab_sir_rederived)], dtype=bool)
    absent = None if has_call_column else f"column {CALL_COLUMN} absent"
    log.drop("call or lab_sir null: excluded from call-level metrics", int((~has_call).sum()), absent)
    log.drop(
        f"call or {REDERIVED_SIR_COLUMN} null: excluded from re-derived call-level metrics",
        int((~has_call_rederived).sum()),
        absent,
    )
    log.drop("band_low/band_high null: excluded from band metrics", int((~has_band).sum()), _models_detail(models, ~has_band))

    rows: list[dict[str, Any]] = []
    if n_rows == 0:
        return rows

    grouped = preds.groupby(list(group_columns), sort=True, observed=True, dropna=False).indices
    for key, idx in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        idx = np.asarray(idx, dtype=np.intp)

        cat = categorical(pred_sir[idx], lab_sir[idx])
        cat_rederived = categorical(pred_sir[idx], lab_sir_rederived[idx])
        call_cat = call_metrics(calls[idx], lab_sir[idx])
        call_rederived = call_metrics(calls[idx], lab_sir_rederived[idx])

        mic_idx = idx[mic_ok[idx]]
        ea = _mean_or_none(essential_agreement(pred[mic_idx], lo[mic_idx], hi[mic_idx]))
        exact = _mean_or_none(exact_agreement(pred[mic_idx], lo[mic_idx], hi[mic_idx]))

        au_idx = idx[auroc_ok[idx]]
        au = auroc(_log2(pred[au_idx]), lab_sir[au_idx])

        cov_idx = idx[band_ok[idx]]
        coverage = band_coverage(band_lo[cov_idx], band_hi[cov_idx], lo[cov_idx], hi[cov_idx])
        width_idx = idx[has_band[idx]]
        width = band_width_steps(band_lo[width_idx], band_hi[width_idx])

        row: dict[str, Any] = dict(zip(group_columns, key_tuple))
        row.update(
            {
                "vme_rate": cat["vme_rate"],
                "me_rate": cat["me_rate"],
                "mine_rate": cat["mine_rate"],
                "categorical_agreement": cat["ca"],
                "essential_agreement": ea,
                "exact_agreement": exact,
                "auroc": au,
                "band_coverage": coverage,
                "band_width_steps": width,
                "n": int(len(idx)),
                "n_exact": int(mic_idx.size),
                "n_band": int(cov_idx.size),
                "n_cat": cat["n_cat"],
                "n_lab_r": cat["n_lab_r"],
                "n_lab_s": cat["n_lab_s"],
                **{c: call_cat[c] for c in (*CALL_METRIC_COLUMNS, *CALL_COUNT_COLUMNS)},
                **{f"{c}_rederived": call_rederived[c] for c in (*CALL_METRIC_COLUMNS, *CALL_COUNT_COLUMNS)},
                "vme_rate_rederived": cat_rederived["vme_rate"],
                "me_rate_rederived": cat_rederived["me_rate"],
                "mine_rate_rederived": cat_rederived["mine_rate"],
                "categorical_agreement_rederived": cat_rederived["ca"],
                "n_cat_rederived": cat_rederived["n_cat"],
                "n_lab_r_rederived": cat_rederived["n_lab_r"],
                "n_lab_s_rederived": cat_rederived["n_lab_s"],
            }
        )
        rows.append(row)
    return rows


def _rows_to_frame(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """Build the output table with stable dtypes: keys ``str``, metrics ``float64``, counts ``int64``."""
    data: dict[str, Any] = {}
    for col in columns:
        values = [row.get(col) for row in rows]
        if col in _FLOAT_COLUMNS:
            data[col] = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
        elif col in _INT_COLUMNS:
            data[col] = np.array(values, dtype=np.int64)
        else:
            data[col] = pd.array(values, dtype="str")
    return pd.DataFrame(data, columns=list(columns))


def summarize(preds: pd.DataFrame, drop_log: DropLog | None = None) -> pd.DataFrame:
    """One metrics row per ``species x drug x model x split`` from a stage-10 predictions table.

    Columns, in this order (VME first after the keys, as the contract demands)::

        species, drug, model, split,
        vme_rate, me_rate, mine_rate, categorical_agreement,
        essential_agreement, exact_agreement, auroc, band_coverage, band_width_steps,
        n, n_exact, n_band, n_cat, n_lab_r, n_lab_s,
        vme_rate_rederived, me_rate_rederived, mine_rate_rederived,
        categorical_agreement_rederived, n_cat_rederived, n_lab_r_rederived, n_lab_s_rederived

    * ``n`` counts every row of the group. ``n_cat``, ``n_lab_r``, ``n_lab_s`` are the
      categorical denominators (rows with both S/I/R calls; lab R; lab S).
    * EA and exact agreement use rows with a prediction and an **exact** lab MIC (one
      doubling step and, when the column exists, ``lab_exact`` true -- see
      :func:`exact_lab_mask`); ``n_exact`` counts exactly those rows, so it is their
      denominator (0 for ``b0_resfinder``, which has no MIC).
    * ``band_coverage`` uses rows with a band and an exact lab MIC (``n_band``), the
      rows the conformal ``q`` is calibrated on; ``band_width_steps`` uses every row
      with a band.
    * ``auroc`` scores ``log2(pred_mic)`` for lab R versus lab S.
    * The ``*_rederived`` block repeats VME / ME / minor error / CA against the optional
      ``lab_sir_rederived`` column (NaN rates and 0 counts when it is absent).
    * Rows with a null ``pred_mic`` (e.g. ``b0_resfinder``) contribute only to the
      categorical metrics; EA / exact / AUROC / band metrics are null for such groups.
      ``band_low``/``band_high`` may be absent entirely.

    Every exclusion is recorded in ``drop_log`` (a ``DropLog("eval")`` is created when
    none is given, which still logs at INFO). Rows are sorted by the key columns.
    Raises ``ValueError`` for missing required columns, null keys, S/I/R values
    outside ``S``/``I``/``R``, or lab intervals / bands that break the contract.
    """
    log = drop_log if drop_log is not None else DropLog("eval")
    rows = _summarize_groups(preds, KEY_COLUMNS, log)
    out = _rows_to_frame(rows, SUMMARY_COLUMNS)
    logger.info("summarized %d prediction rows into %d species x drug x model x split groups", len(preds), len(out))
    return out


# ------------------------------------------------------------------ distance bins
def _format_edge(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x:g}"


def _bin_label(edges: np.ndarray, code: int) -> str:
    left, right = _format_edge(float(edges[code])), _format_edge(float(edges[code + 1]))
    open_bracket = "[" if code == 0 else "("
    return f"{open_bracket}{left}, {right}]"


def _align_distances(preds: pd.DataFrame, nearest_distance: Any) -> np.ndarray:
    """Float array of distances aligned to ``preds`` rows (NaN where unknown).

    * ``pd.Series`` or ``Mapping``: keyed by ``genome_id`` (requires that column).
    * anything else array-like: positional, must have ``len(preds)`` entries.
    """
    if isinstance(nearest_distance, pd.DataFrame):
        raise TypeError("nearest_distance must be a Series/Mapping keyed by genome_id or a positional array")
    if isinstance(nearest_distance, (pd.Series, Mapping)):
        if "genome_id" not in preds.columns:
            raise ValueError("preds needs a genome_id column to align a keyed nearest_distance")
        series = nearest_distance if isinstance(nearest_distance, pd.Series) else pd.Series(dict(nearest_distance), dtype=np.float64)
        if not series.index.is_unique:
            raise ValueError("nearest_distance has duplicate genome_id keys")
        mapped = preds["genome_id"].map(series.to_dict())
        return _as_float_array(mapped, "nearest_distance")
    arr = _as_float_array(nearest_distance, "nearest_distance")
    if len(arr) != len(preds):
        raise ValueError(f"nearest_distance has {len(arr)} entries for {len(preds)} prediction rows")
    return arr


def by_distance_bin(
    preds: pd.DataFrame,
    nearest_distance: pd.Series | Mapping[str, float] | ArrayLike,
    bins: Sequence[float] = DEFAULT_DISTANCE_BINS,
    drop_log: DropLog | None = None,
) -> pd.DataFrame:
    """:func:`summarize` stratified by the genome's distance to the nearest training genome.

    Args:
        preds: Stage-10 predictions table (see :func:`summarize`).
        nearest_distance: Mash distance per genome. A ``pd.Series``/mapping is keyed by
            ``genome_id``; a plain array is positional and must match ``len(preds)``.
            Genomes with no distance are excluded and counted.
        bins: Strictly increasing edges. The first bin is closed on both sides
            (``[e0, e1]``), later bins are ``(e_i, e_i+1]``. Distances outside the
            edges are excluded and counted. ``inf`` is a valid last edge.
        drop_log: Receives the exclusion counts; a ``DropLog("eval")`` is created otherwise.

    Returns one row per ``species x drug x model x split x distance_bin`` with columns::

        species, drug, model, split, distance_bin, bin_low, bin_high,
        <metrics as in summarize, VME first>, n, n_exact, n_band, n_cat, n_lab_r, n_lab_s,
        <re-derived block as in summarize, vme_rate_rederived first>

    sorted by the keys and then by bin position (``distance_bin`` is a string label).
    """
    log = drop_log if drop_log is not None else DropLog("eval")
    edges = _as_float_array(list(bins), "bins")
    if len(edges) < 2:
        raise ValueError("bins needs at least two edges")
    if np.isnan(edges).any() or not (np.diff(edges) > 0).all():
        raise ValueError("bins must be strictly increasing and non-null")

    dist = _align_distances(preds, nearest_distance)
    codes = np.asarray(pd.cut(dist, edges, right=True, include_lowest=True, labels=False), dtype=np.float64)
    missing = np.isnan(dist)
    outside = ~missing & np.isnan(codes)
    log.drop("nearest_distance null: excluded from distance bins", int(missing.sum()))
    log.drop("nearest_distance outside bins: excluded from distance bins", int(outside.sum()))

    keep = ~np.isnan(codes)
    binned = preds.loc[keep].reset_index(drop=True)
    binned[_BIN_CODE] = codes[keep].astype(np.int64)

    rows = _summarize_groups(binned, list(KEY_COLUMNS) + [_BIN_CODE], log)
    for row in rows:
        code = int(row.pop(_BIN_CODE))
        row["distance_bin"] = _bin_label(edges, code)
        row["bin_low"] = float(edges[code])
        row["bin_high"] = float(edges[code + 1])
    out = _rows_to_frame(rows, DISTANCE_COLUMNS)
    logger.info("summarized %d prediction rows into %d distance-binned groups", int(keep.sum()), len(out))
    return out


# ------------------------------------------------------- probability that the drug works
# ``prob_works`` (preds, v0.6) is the calibrated P(lab S under the call breakpoint) from
# :mod:`genome2mic.models.calibration`; ``prob_tier`` its tier. "Works" = the lab MIC
# re-derived under the call breakpoint is S; I and R do not work. Every danger
# (VME-like) column comes first.


PROB_COLUMN = "prob_works"
"""Optional preds column: calibrated probability that the drug works (v0.6)."""
TIER_COLUMN = "prob_tier"

PROB_METRIC_COLUMNS: tuple[str, ...] = (
    "call_danger_rate",
    "tier_danger_rate",
    "forced_danger_rate",
    "confident_rate",
    "confident_right_rate",
    "very_likely_works_right_rate",
    "call_answer_rate",
    "call_right_rate",
    "forced_accuracy",
    "brier",
)
"""Per-group probability metrics, danger first:

* ``call_danger_rate`` = lab R called ``likely_active`` / lab R with a call (call VME);
* ``tier_danger_rate`` = lab R given a 'works' tier (P >= 0.70) / lab R with a probability;
* ``forced_danger_rate`` = lab R with P >= 0.5 / lab R with a probability (forced answer);
* ``confident_rate`` = a 'works' or 'fails' tier (P >= 0.70 or P <= 0.30) / rows with a probability;
* ``confident_right_rate`` = right among those ('works' tier and lab S, or 'fails' tier and lab I/R);
* ``very_likely_works_right_rate`` = lab S among 'very likely works' rows;
* ``call_answer_rate`` = ``likely_active`` or ``likely_inactive`` / rows with a call;
* ``call_right_rate`` = right among answered calls (active and lab S, inactive and lab I/R);
* ``forced_accuracy`` = right when every row is forced to works (P >= 0.5) or fails;
* ``brier`` = mean squared error of P against works (0/1)."""
PROB_COUNT_COLUMNS: tuple[str, ...] = (
    "n_prob", "n_prob_lab_r", "n_confident", "n_very_likely_works", "n_call", "n_call_lab_r", "n_call_answer",
)
PROB_SUMMARY_COLUMNS: tuple[str, ...] = ("species", "model", "split", "n_drugs", *PROB_METRIC_COLUMNS, *PROB_COUNT_COLUMNS)
"""Column order of :func:`probability_summary` (default grouping)."""

DEFAULT_PROB_BINS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
"""Probability bin edges of :func:`calibration_table` (first bin closed, later bins ``(a, b]``)."""
CALIBRATION_VALUE_COLUMNS: tuple[str, ...] = ("prob_bin", "bin_low", "bin_high", "n", "mean_prob", "observed_works")
"""Columns of :func:`calibration_table` after the group columns."""

FOLD_VME_COLUMNS: tuple[str, ...] = (
    "species", "drug", "model", "fold", "calling", "n_lab_r", "n_vme", "call_vme_rate", "p_value", "significant",
)
"""Columns of :func:`call_vme_by_fold`, VME first after the keys."""


def _as_prob_array(values: ArrayLike, name: str = PROB_COLUMN) -> np.ndarray:
    arr = _as_float_array(values, name)
    finite = ~np.isnan(arr)
    if ((arr[finite] < -_TOL) | (arr[finite] > 1 + _TOL)).any():
        raise ValueError(f"{name} must be in [0, 1] where present")
    return arr


def probability_metrics(prob: ArrayLike, call: ArrayLike, lab_sir: ArrayLike) -> dict[str, float | int | None]:
    """The user's table for one group of rows (see :data:`PROB_METRIC_COLUMNS`).

    Probability metrics use rows with a probability and a lab category; call metrics rows
    with a call and a lab category. A rate is ``None`` when its denominator is zero.
    """
    p = _as_prob_array(prob)
    calls = _as_call_array(call)
    lab = _as_sir_array(lab_sir, "lab_sir")
    _check_same_length(prob=p, call=calls, lab_sir=lab)
    has_lab = np.array([v is not None for v in lab], dtype=bool)
    is_s = lab == "S"
    is_r = lab == "R"

    with_p = has_lab & ~np.isnan(p)
    tiers = prob_tier(np.where(with_p, p, np.nan))
    works_tier = np.array([t in WORKS_TIERS for t in tiers], dtype=bool) & with_p
    fails_tier = np.array([t in FAILS_TIERS for t in tiers], dtype=bool) & with_p
    vlw = np.array([t == PROB_TIERS[0] for t in tiers], dtype=bool) & with_p
    forced_works = with_p & (np.nan_to_num(p, nan=-1.0) >= 0.5 - _TOL)
    confident = works_tier | fails_tier
    n_prob = int(with_p.sum())
    n_prob_r = int((with_p & is_r).sum())

    with_call = has_lab & np.array([c is not None for c in calls], dtype=bool)
    active = with_call & (calls == "likely_active")
    inactive = with_call & (calls == "likely_inactive")
    answered = active | inactive
    n_call_r = int((with_call & is_r).sum())

    brier = None
    if n_prob:
        brier = float(np.mean((p[with_p] - is_s[with_p].astype(float)) ** 2))
    return {
        "call_danger_rate": _ratio(int((active & is_r).sum()), n_call_r),
        "tier_danger_rate": _ratio(int((works_tier & is_r).sum()), n_prob_r),
        "forced_danger_rate": _ratio(int((forced_works & is_r).sum()), n_prob_r),
        "confident_rate": _ratio(int(confident.sum()), n_prob),
        "confident_right_rate": _ratio(int(((works_tier & is_s) | (fails_tier & ~is_s)).sum()), int(confident.sum())),
        "very_likely_works_right_rate": _ratio(int((vlw & is_s).sum()), int(vlw.sum())),
        "call_answer_rate": _ratio(int(answered.sum()), int(with_call.sum())),
        "call_right_rate": _ratio(int(((active & is_s) | (inactive & ~is_s)).sum()), int(answered.sum())),
        "forced_accuracy": _ratio(int(((forced_works & is_s) | (with_p & ~forced_works & ~is_s)).sum()), n_prob),
        "brier": brier,
        "n_prob": n_prob,
        "n_prob_lab_r": n_prob_r,
        "n_confident": int(confident.sum()),
        "n_very_likely_works": int(vlw.sum()),
        "n_call": int(with_call.sum()),
        "n_call_lab_r": n_call_r,
        "n_call_answer": int(answered.sum()),
    }


def probability_summary(
    preds: pd.DataFrame,
    group_columns: Sequence[str] = ("species", "model", "split"),
    lab_column: str = REDERIVED_SIR_COLUMN,
) -> pd.DataFrame:
    """:func:`probability_metrics` per group (default: species x model x split), danger first.

    ``n_drugs`` counts the drugs of the group with at least one probability. Empty (with
    the columns) when the preds carry no ``prob_works`` or no ``lab_column``.
    """
    columns = (*group_columns, "n_drugs", *PROB_METRIC_COLUMNS, *PROB_COUNT_COLUMNS)
    if PROB_COLUMN not in preds.columns or lab_column not in preds.columns or preds.empty:
        logger.info("preds have no %s / %s column (or no rows); probability summary is empty", PROB_COLUMN, lab_column)
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    calls = preds[CALL_COLUMN] if CALL_COLUMN in preds.columns else pd.Series([None] * len(preds), index=preds.index)
    rows = []
    for key, block in preds.groupby(list(group_columns), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        stats = probability_metrics(block[PROB_COLUMN], calls.loc[block.index], block[lab_column])
        has_p = pd.to_numeric(block[PROB_COLUMN], errors="coerce").notna()
        row = dict(zip(group_columns, key_tuple))
        row["n_drugs"] = int(block.loc[has_p, "drug"].nunique()) if "drug" in block.columns else 0
        row.update(stats)
        rows.append(row)
    out = pd.DataFrame(rows, columns=list(columns))
    for c in PROB_METRIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in (*PROB_COUNT_COLUMNS, "n_drugs"):
        out[c] = out[c].astype("int64")
    return out


def calibration_table(
    preds: pd.DataFrame,
    group_columns: Sequence[str] = ("species", "model", "split"),
    edges: Sequence[float] = DEFAULT_PROB_BINS,
    lab_column: str = REDERIVED_SIR_COLUMN,
) -> pd.DataFrame:
    """Calibration check: per group and probability bin, ``n``, mean P and the observed share that worked.

    Rows need a probability and a lab category. Only non-empty bins are returned. Bins
    are ``[e0, e1]``, then ``(e_i, e_i+1]``.
    """
    columns = [*group_columns, *CALIBRATION_VALUE_COLUMNS]
    if PROB_COLUMN not in preds.columns or lab_column not in preds.columns or preds.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    p = _as_prob_array(preds[PROB_COLUMN])
    lab = _as_sir_array(preds[lab_column], lab_column)
    keep = ~np.isnan(p) & np.array([v is not None for v in lab], dtype=bool)
    edge_arr = np.asarray(list(edges), dtype=np.float64)
    codes = np.asarray(pd.cut(p, edge_arr, right=True, include_lowest=True, labels=False), dtype=np.float64)
    keep &= ~np.isnan(codes)
    work = preds.loc[keep, list(group_columns)].copy()
    work["_code"] = codes[keep].astype(np.int64)
    work["_p"] = p[keep]
    work["_works"] = (lab[keep] == "S").astype(float)
    rows = []
    for key, block in work.groupby([*group_columns, "_code"], sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        code = int(key_tuple[-1])
        row = dict(zip(group_columns, key_tuple[:-1]))
        row.update({
            "prob_bin": _bin_label(edge_arr, code),
            "bin_low": float(edge_arr[code]),
            "bin_high": float(edge_arr[code + 1]),
            "n": int(len(block)),
            "mean_prob": float(block["_p"].mean()),
            "observed_works": float(block["_works"].mean()),
        })
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out["n"] = out["n"].astype("int64")
    return out


def call_vme_by_fold(
    preds: pd.DataFrame,
    folds: pd.Series | Mapping[str, float],
    target: float = DEFAULT_VME_TARGET,
    p_threshold: float = DEFAULT_FOLD_P_THRESHOLD,
    lab_column: str = REDERIVED_SIR_COLUMN,
) -> pd.DataFrame:
    """Out-of-fold call VME per species x drug x model x CV fold, with a one-sided exact binomial p.

    Only ``split == 'cv'`` rows are used; ``folds`` maps ``genome_id`` to its fold.
    ``calling`` = the fold issued at least one ``likely_active`` call; ``p_value`` =
    ``P(X >= n_vme)`` for ``X ~ Binomial(n_lab_r, target)``; ``significant`` = calling and
    ``p_value < p_threshold`` (the bundle gate closes on such a fold).
    """
    need = {"genome_id", "species", "drug", "model", "split", CALL_COLUMN, lab_column}
    if not need.issubset(preds.columns):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in FOLD_VME_COLUMNS})
    cv = preds.loc[(preds["split"] == "cv").fillna(False).to_numpy(dtype=bool)]
    fold_map = folds.to_dict() if isinstance(folds, pd.Series) else dict(folds)
    fold = pd.to_numeric(cv["genome_id"].map(fold_map), errors="coerce").to_numpy(dtype=np.float64)
    calls = _as_call_array(cv[CALL_COLUMN])
    lab = _as_sir_array(cv[lab_column], lab_column)
    active = calls == "likely_active"
    is_r = lab == "R"
    keys = cv[["species", "drug", "model"]].astype(str).to_numpy()
    frame = pd.DataFrame({"species": keys[:, 0], "drug": keys[:, 1], "model": keys[:, 2], "fold": fold,
                          "_active": active, "_r": is_r, "_vme": active & is_r})
    frame = frame.loc[~np.isnan(fold)]
    rows = []
    for (species, drug, model, f), block in frame.groupby(["species", "drug", "model", "fold"], sort=True):
        n_r = int(block["_r"].sum())
        k = int(block["_vme"].sum())
        calling = bool(block["_active"].any())
        p = binomial_excess_p(k, n_r, target)
        rows.append({
            "species": species, "drug": drug, "model": model, "fold": int(f), "calling": calling,
            "n_lab_r": n_r, "n_vme": k, "call_vme_rate": (k / n_r) if n_r else np.nan, "p_value": p,
            "significant": bool(calling and p < p_threshold),
        })
    out = pd.DataFrame(rows, columns=list(FOLD_VME_COLUMNS))
    if not out.empty:
        out = out.astype({"fold": "int64", "calling": bool, "n_lab_r": "int64", "n_vme": "int64",
                          "call_vme_rate": "float64", "p_value": "float64", "significant": bool})
    return out.reset_index(drop=True)


# ------------------------------------------------------- calibration error (v0.7)


CALIBRATION_ERROR_COLUMNS: tuple[str, ...] = ("n", "ece", "max_gap", "brier")
"""Columns of :func:`calibration_error_summary` after the group columns."""
TOLD_WORKS_THRESHOLD = 0.70
"""'Told works' = P(works) >= 0.70 (a 'works' tier)."""


def calibration_error(
    prob: ArrayLike, lab_sir: ArrayLike, edges: Sequence[float] = DEFAULT_PROB_BINS
) -> dict[str, float | int | None]:
    """Expected calibration error over :data:`DEFAULT_PROB_BINS`, the largest bin gap and the Brier score.

    ``ece = sum_b n_b / N * |mean P_b - share lab S_b|``; ``max_gap`` = the largest bin
    ``|mean P - share lab S|``; rows need a probability and a lab category (I and R = did
    not work). Rates are ``None`` without a usable row.
    """
    p = _as_prob_array(prob)
    lab = _as_sir_array(lab_sir, "lab_sir")
    _check_same_length(prob=p, lab_sir=lab)
    keep = ~np.isnan(p) & np.array([v is not None for v in lab], dtype=bool)
    n = int(keep.sum())
    if n == 0:
        return {"n": 0, "ece": None, "max_gap": None, "brier": None}
    pk = p[keep]
    works = (lab[keep] == "S").astype(float)
    edge_arr = np.asarray(list(edges), dtype=np.float64)
    codes = np.asarray(pd.cut(pk, edge_arr, right=True, include_lowest=True, labels=False), dtype=np.float64)
    ece, gap = 0.0, 0.0
    for code in np.unique(codes[~np.isnan(codes)]):
        in_bin = codes == code
        diff = abs(float(pk[in_bin].mean()) - float(works[in_bin].mean()))
        ece += int(in_bin.sum()) / n * diff
        gap = max(gap, diff)
    return {"n": n, "ece": float(ece), "max_gap": float(gap), "brier": float(np.mean((pk - works) ** 2))}


def calibration_error_summary(
    preds: pd.DataFrame,
    group_columns: Sequence[str] = ("species", "model", "split"),
    lab_column: str = REDERIVED_SIR_COLUMN,
) -> pd.DataFrame:
    """:func:`calibration_error` per group (default species x model x split)."""
    columns = [*group_columns, *CALIBRATION_ERROR_COLUMNS]
    if PROB_COLUMN not in preds.columns or lab_column not in preds.columns or preds.empty:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    rows = []
    for key, block in preds.groupby(list(group_columns), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_tuple))
        row.update(calibration_error(block[PROB_COLUMN], block[lab_column]))
        rows.append(row)
    out = pd.DataFrame(rows, columns=columns)
    out["n"] = out["n"].astype("int64")
    for c in ("ece", "max_gap", "brier"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def told_works_by_pair(
    preds: pd.DataFrame,
    threshold: float = TOLD_WORKS_THRESHOLD,
    lab_column: str = REDERIVED_SIR_COLUMN,
) -> pd.DataFrame:
    """Per species x drug: lab R told the drug works (``P >= threshold``) over lab R with a probability.

    Danger first, sorted worst first (rate, then count). Columns: ``species``, ``drug``,
    ``told_works_rate``, ``n_told_works``, ``n_lab_r``.
    """
    columns = ["species", "drug", "told_works_rate", "n_told_works", "n_lab_r"]
    need = {"species", "drug", PROB_COLUMN, lab_column}
    if preds.empty or not need.issubset(preds.columns):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    p = _as_prob_array(preds[PROB_COLUMN])
    lab = _as_sir_array(preds[lab_column], lab_column)
    is_r = (lab == "R") & ~np.isnan(p)
    told = is_r & (np.nan_to_num(p, nan=-1.0) >= threshold - _TOL)
    frame = pd.DataFrame({"species": preds["species"].astype(str).to_numpy(), "drug": preds["drug"].astype(str).to_numpy(),
                          "_r": is_r, "_told": told})
    out = frame.groupby(["species", "drug"], sort=True).agg(n_lab_r=("_r", "sum"), n_told_works=("_told", "sum")).reset_index()
    out = out.loc[out["n_lab_r"] > 0].copy()
    out["told_works_rate"] = out["n_told_works"] / out["n_lab_r"]
    out = out.sort_values(["told_works_rate", "n_told_works", "species", "drug"], ascending=[False, False, True, True])
    out = out[columns].astype({"n_told_works": "int64", "n_lab_r": "int64", "told_works_rate": "float64"})
    return out.reset_index(drop=True)
