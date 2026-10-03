"""Pure metric functions for MIC predictions and the stage-11 summary table.

Inputs follow the stage-10 predictions schema in ``DATA_CONTRACT.md``: a point
prediction ``pred_mic`` (mg/L, rounded **up** to the doubling grid), a 90% band
``band_low``/``band_high``, the lab interval ``(lab_lower, lab_upper]`` copied from
stage 2, and the S/I/R calls ``pred_sir``/``lab_sir``.

Censored lab rows
-----------------
A lab result is never a point; it is an interval ``(lo, hi]`` on the doubling grid.
``(4, 8]`` is an exact reading of 8, ``(0, 0.25]`` is ``<= 0.25`` (left-censored),
``(32, inf)`` is ``> 32`` (right-censored), and an ``I``-only label can be a
multi-step interval such as ``(2, 8]``. The rules below are stated for a general
interval and reduce to the DESIGN.md wording on the three common cases:

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
* **n_exact** -- rows whose lab interval is finite on both sides (``censor ==
  'interval'``), i.e. the rows a classical EA would use.

A ``(0, inf)`` interval carries no information and is never evaluated.

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
error that harms a patient (CLAUDE.md rule 10).
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
from genome2mic.mic import round_down_to_step_array, round_up_to_step_array

logger = logging.getLogger(__name__)

ArrayLike = Sequence[Any] | np.ndarray | pd.Series

SIR_VALUES: tuple[str, ...] = ("S", "I", "R")

KEY_COLUMNS: tuple[str, ...] = ("species", "drug", "model", "split")
METRIC_COLUMNS: tuple[str, ...] = (
    "vme_rate",
    "me_rate",
    "mine_rate",
    "categorical_agreement",
    "essential_agreement",
    "exact_agreement",
    "auroc",
    "band_coverage",
    "band_width_steps",
)
COUNT_COLUMNS: tuple[str, ...] = ("n", "n_exact", "n_cat", "n_lab_r", "n_lab_s")

SUMMARY_COLUMNS: tuple[str, ...] = KEY_COLUMNS + METRIC_COLUMNS + COUNT_COLUMNS
"""Column order of :func:`summarize`. ``vme_rate`` is the first metric."""

BIN_COLUMNS: tuple[str, ...] = ("distance_bin", "bin_low", "bin_high")
DISTANCE_COLUMNS: tuple[str, ...] = KEY_COLUMNS + BIN_COLUMNS + METRIC_COLUMNS + COUNT_COLUMNS
"""Column order of :func:`by_distance_bin`."""

REQUIRED_PRED_COLUMNS: tuple[str, ...] = KEY_COLUMNS + (
    "pred_mic",
    "lab_lower",
    "lab_upper",
    "pred_sir",
    "lab_sir",
)
BAND_COLUMNS: tuple[str, ...] = ("band_low", "band_high")

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


# ------------------------------------------------------------------- per-row metrics
def is_exact_interval(lab_lower: ArrayLike, lab_upper: ArrayLike) -> np.ndarray:
    """True where the lab interval is finite on both sides (``censor == 'interval'``).

    Null bounds give False. These are the rows a classical essential-agreement
    computation would use and the rows counted in ``n_exact``.
    """
    lo = _as_float_array(lab_lower, "lab_lower")
    hi = _as_float_array(lab_upper, "lab_upper")
    _check_same_length(lab_lower=lo, lab_upper=hi)
    with np.errstate(invalid="ignore"):
        return (lo > 0) & np.isfinite(hi) & ~np.isnan(lo)


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

    Should be about 0.90 for a 90% conformal band. ``None`` on empty input.
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
        exact_lab = has_lab & (lo > 0) & np.isfinite(hi)
    mic_ok = has_pred & informative
    band_ok = has_band & informative
    has_sir = np.array([p is not None and l is not None for p, l in zip(pred_sir, lab_sir)], dtype=bool)
    lab_is_i = lab_sir == "I"
    auroc_ok = has_pred & ((lab_sir == "R") | (lab_sir == "S"))

    # Every filter logs a count, including zero counts, so the log proves it ran.
    log.drop("pred_mic null: categorical metrics only", int((~has_pred).sum()), _models_detail(models, ~has_pred))
    log.drop("lab_lower/lab_upper null: excluded from MIC metrics", int((~has_lab).sum()), _models_detail(models, ~has_lab))
    log.drop("lab interval (0, inf): excluded from MIC metrics", int((has_lab & ~informative).sum()))
    log.drop("pred_sir or lab_sir null: excluded from categorical metrics", int((~has_sir).sum()), _models_detail(models, ~has_sir))
    log.drop("lab_sir I: excluded from AUROC", int((has_pred & lab_is_i).sum()))
    log.drop("band_low/band_high null: excluded from band metrics", int((~has_band).sum()), _models_detail(models, ~has_band))

    rows: list[dict[str, Any]] = []
    if n_rows == 0:
        return rows

    grouped = preds.groupby(list(group_columns), sort=True, observed=True, dropna=False).indices
    for key, idx in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        idx = np.asarray(idx, dtype=np.intp)

        cat = categorical(pred_sir[idx], lab_sir[idx])

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
                "n_exact": int(exact_lab[idx].sum()),
                "n_cat": cat["n_cat"],
                "n_lab_r": cat["n_lab_r"],
                "n_lab_s": cat["n_lab_s"],
            }
        )
        rows.append(row)
    return rows


def _rows_to_frame(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """Build the output table with stable dtypes: keys ``str``, metrics ``float64``, counts ``int64``."""
    data: dict[str, Any] = {}
    for col in columns:
        values = [row.get(col) for row in rows]
        if col in METRIC_COLUMNS or col in ("bin_low", "bin_high"):
            data[col] = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
        elif col in COUNT_COLUMNS:
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
        n, n_exact, n_cat, n_lab_r, n_lab_s

    * ``n`` counts every row of the group. ``n_cat``, ``n_lab_r``, ``n_lab_s`` are the
      categorical denominators (rows with both S/I/R calls; lab R; lab S).
    * ``n_exact`` counts rows whose lab interval is finite on both sides.
    * EA / exact agreement use every row with a prediction and an informative lab
      interval, censored rows included (see the module docstring for the rule).
    * ``auroc`` scores ``log2(pred_mic)`` for lab R versus lab S.
    * Band metrics use rows with a band; coverage additionally needs a lab interval.
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
        <metrics as in summarize, VME first>, n, n_exact, n_cat, n_lab_r, n_lab_s

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
