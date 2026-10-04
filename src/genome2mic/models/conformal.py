"""Split-conformal uncertainty bands on log2 MIC.

CLAUDE.md: take ``|pred - true|`` in doubling steps on validation rows with exact
MICs (one doubling step and not disk diffusion: the orchestrator passes its
``lab_exact`` mask); the finite-sample-corrected 90th percentile ``q`` gives a
``+-q``-step band.
The *upper* end of the band is what gets compared with the S breakpoint.

Residuals come from out-of-fold predictions (never the test set, CLAUDE.md rule 8);
the orchestrator feeds ``log2(pred_mic)`` of the rounded-up prediction here.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genome2mic.droplog import DropLog
from genome2mic.mic import round_down_to_step_array, round_up_to_step_array
from genome2mic.models.base import exact_mask, read_json, validate_intervals, write_json

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.10
"""Miscoverage level: ``1 - alpha`` = 90% nominal band coverage."""


NOT_MEASURED_REASON = "one-step lab interval but not a measured MIC (e.g. disk diffusion): conformal residuals use exact MICs only"
"""Drop reason for rows the caller's ``exact_rows`` mask removes from a one-step interval."""


def residual_steps(
    pred_log2_rounded_up: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    droplog: DropLog | None = None,
    *,
    exact_rows: np.ndarray | None = None,
) -> np.ndarray:
    """Absolute residuals in doubling steps on exact rows only.

    Args:
        pred_log2_rounded_up: ``log2(pred_mic)`` where ``pred_mic`` was rounded up
            to the grid, one value per row (NaN = no prediction; such rows are
            excluded and counted).
        lo, hi: Lab interval bounds in mg/L, aligned with the predictions.
        droplog: Where to record the censored/missing rows that were skipped. A
            local ``DropLog("conformal")`` is used when omitted.
        exact_rows: Optional boolean mask of rows whose lab result is a measured
            MIC (training passes ``lab_exact``,
            :func:`genome2mic.mic.lab_exact_mask`, which excludes disk diffusion). It
            only restricts: a row must also pass the one-step interval rule. The
            one-step rows it removes are counted separately.

    Returns:
        ``|pred_step - log2(hi)|`` for exact rows (one doubling step, and in
        ``exact_rows`` when given) with a non-missing prediction, in input order.
    """
    pred = np.asarray(pred_log2_rounded_up, dtype=np.float64).ravel()
    low, high = validate_intervals(lo, hi, len(pred))
    log = droplog if droplog is not None else DropLog("conformal")
    interval_exact = exact_mask(low, high)
    exact = interval_exact
    if exact_rows is not None:
        measured = np.asarray(exact_rows, dtype=bool).ravel()
        if measured.size != len(pred):
            raise ValueError(f"exact_rows has {measured.size} entries for {len(pred)} rows")
        exact = interval_exact & measured
    log.drop("censored row (conformal residuals use exact MICs only)", int((~interval_exact).sum()))
    missing = np.isnan(pred)
    log.drop("missing prediction (conformal residuals)", int((missing & exact).sum()))
    if exact_rows is not None:
        log.drop(NOT_MEASURED_REASON, int((interval_exact & ~exact).sum()))
    keep = exact & ~missing
    return np.abs(pred[keep] - np.log2(high[keep]))


def conformal_q(residuals: np.ndarray, alpha: float = DEFAULT_ALPHA) -> float:
    """Finite-sample-corrected ``(1 - alpha)`` quantile of the residuals, in steps.

    With ``n`` residuals sorted ascending, returns the ``k``-th smallest where
    ``k = ceil((n + 1) * (1 - alpha))`` (split conformal; Vovk et al.). When
    ``k > n`` the sample is too small to certify ``1 - alpha`` coverage and the
    answer is ``inf`` (an uninformative band), logged at WARNING. The result is
    never negative. Raises ``ValueError`` for an empty sample, NaN residuals or
    ``alpha`` outside ``(0, 1)``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    r = np.asarray(residuals, dtype=np.float64).ravel()
    n = len(r)
    if n == 0:
        raise ValueError("conformal_q needs at least one residual")
    if np.isnan(r).any():
        raise ValueError("residuals contain NaN")
    if (r < 0).any():
        raise ValueError("residuals must be non-negative (absolute errors)")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        logger.warning(
            "conformal_q: %d residuals cannot certify %.0f%% coverage (need >= %d); q = inf",
            n, 100 * (1 - alpha), math.ceil((1 - alpha) / alpha),
        )
        return math.inf
    q = float(np.sort(r)[k - 1])
    return max(0.0, q)


def band(pred_mic: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """``(pred / 2**q, pred * 2**q)`` snapped outward to the doubling grid, mg/L.

    The low end is rounded *down* and the high end *up*, so the band only ever
    widens. NaN predictions give NaN bounds (missing stays missing). An infinite
    ``q`` yields the uninformative band ``(0, inf)`` for every finite prediction.
    """
    if q is None or (isinstance(q, float) and math.isnan(q)) or q < 0:
        raise ValueError(f"q must be a non-negative number, got {q!r}")
    pred = np.asarray(pred_mic, dtype=np.float64).ravel()
    finite = ~np.isnan(pred)
    if (pred[finite] <= 0).any():
        raise ValueError("pred_mic must be > 0")
    if math.isinf(q):
        low = np.where(finite, 0.0, np.nan)
        high = np.where(finite, np.inf, np.nan)
        return low, high
    factor = 2.0**q
    low = round_down_to_step_array(pred / factor)
    high = round_up_to_step_array(pred * factor)
    return low, high


def write_conformal(path: Path, q: float, alpha: float, n_residuals: int, extra: dict[str, Any] | None = None) -> None:
    """Write ``conformal.json`` (``q`` in steps, ``alpha``, ``n_residuals``, ...)."""
    payload: dict[str, Any] = {"q": q, "alpha": alpha, "n_residuals": int(n_residuals)}
    if extra:
        payload.update(extra)
    write_json(Path(path), payload)


def read_conformal(path: Path) -> dict[str, Any]:
    """Read ``conformal.json`` written by :func:`write_conformal`."""
    payload = read_json(Path(path))
    if "q" not in payload:
        raise ValueError(f"{path} has no 'q'")
    return payload


# --------------------------------------------------------------------------- #
# Asymmetric bands with an upper level tuned for call-level VME (lever L5)
# --------------------------------------------------------------------------- #

DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.08, 0.05, 0.025, 0.01, 0.005)
"""Candidate upper miscoverage levels, widest-first order of preference (largest alpha = narrowest band)."""

DEFAULT_ALPHA_LOW = 0.05
"""Fixed lower miscoverage level of the asymmetric band."""

DEFAULT_VME_TARGET = 0.015
"""Call-level VME target the upper level must certify inside the training folds."""

FALLBACK_STEPS = 2.0
"""Finite half-width used when too few residuals exist to certify a level."""


def signed_residual_steps(
    pred_log2_rounded_up: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    exact_rows: np.ndarray | None = None,
) -> np.ndarray:
    """``log2(hi) - pred`` (doubling steps; positive = lab MIC above the prediction) on exact rows.

    Same row rule as :func:`residual_steps` (one-step interval, in ``exact_rows`` when
    given, non-missing prediction) but keeps the sign, so the upper and lower band
    ends can be calibrated separately.
    """
    pred = np.asarray(pred_log2_rounded_up, dtype=np.float64).ravel()
    low, high = validate_intervals(lo, hi, len(pred))
    keep = exact_mask(low, high) & ~np.isnan(pred)
    if exact_rows is not None:
        measured = np.asarray(exact_rows, dtype=bool).ravel()
        if measured.size != len(pred):
            raise ValueError(f"exact_rows has {measured.size} entries for {len(pred)} rows")
        keep &= measured
    return np.log2(high[keep]) - pred[keep]


def level_quantile(values: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile: the ``ceil((n + 1) * level)``-th smallest value; ``inf`` if that exceeds ``n``.

    Values may be negative (signed residuals). Raises ``ValueError`` for an empty
    sample or a level outside ``(0, 1)``.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    v = np.asarray(values, dtype=np.float64).ravel()
    n = len(v)
    if n == 0:
        raise ValueError("level_quantile needs at least one value")
    k = math.ceil((n + 1) * level - 1e-12)
    if k > n:
        return math.inf
    k = max(k, 1)
    return float(np.partition(v, k - 1)[k - 1])


def asym_quantiles(signed: np.ndarray, alpha_up: float, alpha_low: float) -> tuple[float, float, bool]:
    """``(q_up, q_low, upper_certified)`` half-widths in steps from signed residuals.

    ``q_up`` is the ``1 - alpha_up`` quantile of ``log2(lab) - pred`` and ``q_low`` the
    ``1 - alpha_low`` quantile of ``pred - log2(lab)``; both are floored at 0 so the band
    always contains the prediction. When the sample is too small to certify a level the
    half-width falls back to a finite, conservative value
    (``max(FALLBACK_STEPS, largest residual on that side)``); for the upper end this is
    reported as ``upper_certified = False`` and the caller must then withhold
    ``likely_active`` calls (an uncertified upper end can never support one). An empty
    sample gives ``(FALLBACK_STEPS, FALLBACK_STEPS, False)``.
    """
    r = np.asarray(signed, dtype=np.float64).ravel()
    if r.size == 0:
        return FALLBACK_STEPS, FALLBACK_STEPS, False
    q_up = level_quantile(r, 1 - alpha_up)
    q_low = level_quantile(-r, 1 - alpha_low)
    certified = math.isfinite(q_up)
    if not certified:
        q_up = max(FALLBACK_STEPS, float(r.max()))
    if not math.isfinite(q_low):
        q_low = max(FALLBACK_STEPS, float((-r).max()))
    return max(0.0, float(q_up)), max(0.0, float(q_low)), certified


def asym_band(pred_mic: np.ndarray, q_up: float, q_low: float) -> tuple[np.ndarray, np.ndarray]:
    """``(pred / 2**q_low, pred * 2**q_up)`` snapped outward to the doubling grid (mg/L).

    :func:`band` is the special case ``q_up == q_low``. Both half-widths must be finite
    and non-negative.
    """
    for name, q in (("q_up", q_up), ("q_low", q_low)):
        if q is None or not math.isfinite(float(q)) or q < 0:
            raise ValueError(f"{name} must be a finite non-negative number, got {q!r}")
    pred = np.asarray(pred_mic, dtype=np.float64).ravel()
    finite = ~np.isnan(pred)
    if (pred[finite] <= 0).any():
        raise ValueError("pred_mic must be > 0")
    low = round_down_to_step_array(pred / 2.0**q_low)
    high = round_up_to_step_array(pred * 2.0**q_up)
    return low, high


def vme_certified(n_vme: int, n_lab_r: int, target: float = DEFAULT_VME_TARGET) -> bool:
    """Upper-confidence rule ``(n_vme + 1) / (n_lab_r + 1) <= target``.

    The ``+1`` makes the rule fail when there are too few lab-R isolates to show the
    target at all (at 1.5 % it needs at least 66 lab-R rows with zero VME), so a drug
    whose resistant isolates are rare never gets ``likely_active`` calls on faith.
    """
    return (int(n_vme) + 1) / (int(n_lab_r) + 1) <= target + 1e-12


@dataclass(frozen=True)
class BandParams:
    """Band and gate for one model of one pair (``conformal.json`` of the bundle).

    ``active_gate_open`` False means no candidate upper level certified the call-level
    VME target inside the training folds (or the chosen level could not be certified on
    the calibration residuals): ``likely_active`` calls are then replaced by
    ``uncertain``.
    """

    q_up: float
    q_low: float
    alpha_up: float
    alpha_low: float
    active_gate_open: bool
    n_residuals: int
    inner_vme_ucb: float | None = None
    kind: str = "asymmetric_tuned"
    q_low_widened: bool = False
    """True when :func:`robust_q_low` widened the lower end (leave-one-fold-out values disagreed)."""
    passing_alphas: tuple[float, ...] = ()
    """Every grid level that passed the nested call-VME rule on the calibration folds (narrowest first)."""
    allowed_alphas: tuple[float, ...] | None = None
    """Bundle only: the levels that passed in every fold that issues calls (None = unrestricted)."""
    oof_call_vme_ucb: float | None = None
    """Bundle only: ``(k + 1) / (n_lab_R + 1)`` of the out-of-fold CV calls over the calling folds (None = not checked)."""
    q_up_widened: bool = False
    """True when :func:`robust_q_up` widened the upper end (a calibration fold missed it significantly often)."""
    fold_call_vme: tuple[dict[str, Any], ...] | None = None
    """Bundle only: per-CV-fold call VME of the out-of-fold calls with its one-sided exact binomial p-value
    (:func:`fold_call_vme_table`); None = not checked."""
    fold_gate_closed: bool = False
    """Bundle only: the gate was closed because a calling fold's call VME was significantly above the target."""

    def band(self, pred_mic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return asym_band(pred_mic, self.q_up, self.q_low)

    def as_json(self) -> dict[str, Any]:
        return {
            "band_kind": self.kind,
            "q_up": float(self.q_up),
            "q_low": float(self.q_low),
            "alpha_up": float(self.alpha_up),
            "alpha_low": float(self.alpha_low),
            "active_gate_open": bool(self.active_gate_open),
            "inner_call_vme_ucb": None if self.inner_vme_ucb is None else float(self.inner_vme_ucb),
            "q_low_widened": bool(self.q_low_widened),
            "passing_alpha_up": [float(a) for a in self.passing_alphas],
            "allowed_alpha_up": None if self.allowed_alphas is None else [float(a) for a in self.allowed_alphas],
            "oof_call_vme_ucb": None if self.oof_call_vme_ucb is None else float(self.oof_call_vme_ucb),
            "q_up_widened": bool(self.q_up_widened),
            "fold_call_vme": None if self.fold_call_vme is None else [dict(r) for r in self.fold_call_vme],
            "fold_gate_closed": bool(self.fold_gate_closed),
        }


def symmetric_params(q: float, alpha: float, n_residuals: int) -> BandParams:
    """The legacy ``+-q`` band as :class:`BandParams` (gate always open)."""
    return BandParams(q_up=float(q), q_low=float(q), alpha_up=alpha, alpha_low=alpha, active_gate_open=True,
                      n_residuals=int(n_residuals), kind="symmetric")


def gate_calls(calls: np.ndarray, gate_open: bool) -> np.ndarray:
    """``likely_active`` -> ``uncertain`` when the gate is closed (other calls unchanged)."""
    out = np.asarray(calls, dtype=object).copy()
    if not gate_open:
        out[out == "likely_active"] = "uncertain"
    return out


def passing_levels(
    pred_log2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    exact_rows: np.ndarray,
    lab_sir: np.ndarray,
    folds: np.ndarray,
    cal_folds: Sequence[int],
    calls_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    alpha_low: float = DEFAULT_ALPHA_LOW,
    target: float = DEFAULT_VME_TARGET,
    strict_folds: bool = False,
) -> list[tuple[float, float]]:
    """Every grid level that passes the nested cross-conformal call-VME rule on ``cal_folds``, in grid order.

    Every fold ``g`` of ``cal_folds`` gets an asymmetric band calibrated on the exact
    residuals of the *other* folds of ``cal_folds``; ``calls_fn(idx, band_low, band_high)``
    turns it into calls, compared with ``lab_sir``. Only folds that actually issue
    ``likely_active`` calls at a level count: their VMEs and lab-R rows are pooled and the
    level passes when :func:`vme_certified` holds on that pool (a fold with no active call
    cannot dilute the folds that do; a level at which no fold calls never passes). With
    ``strict_folds`` every calling fold must also have ``n_vme_g <= target * n_lab_R_g``.
    Rows of folds not in ``cal_folds`` are never read.

    Returns:
        ``[(alpha_up, vme_ucb_over_calling_folds), ...]`` for the passing levels.
    """
    p = np.asarray(pred_log2, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.float64)
    lab = np.asarray(lab_sir, dtype=object)
    cal_folds = [int(f) for f in cal_folds]
    in_cal = np.isin(folds, cal_folds) & ~np.isnan(p)
    is_r = np.array([v == "R" for v in lab], dtype=bool)
    exact = np.asarray(exact_rows, dtype=bool)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    pieces = []
    for g in cal_folds:
        ev = np.where((folds == g) & in_cal)[0]
        cal = in_cal & (folds != g)
        if ev.size == 0:
            continue
        pieces.append((ev, signed_residual_steps(p[cal], lo[cal], hi[cal], exact_rows=exact[cal])))
    out: list[tuple[float, float]] = []
    for a in grid:
        n_vme = n_r = 0
        every_fold_ok = True
        for ev, r in pieces:
            q_up, q_low, certified = asym_quantiles(r, a, alpha_low)
            low, high = asym_band(np.exp2(p[ev]), q_up, q_low)
            active = gate_calls(calls_fn(ev, low, high), certified) == "likely_active"
            if not active.any():
                continue  # this fold issues no active call: it neither passes nor dilutes
            k = int((active & is_r[ev]).sum())
            n_r_g = int(is_r[ev].sum())
            n_vme += k
            n_r += n_r_g
            if strict_folds and k > target * n_r_g + 1e-12:
                every_fold_ok = False
        if every_fold_ok and vme_certified(n_vme, n_r, target):
            out.append((float(a), (n_vme + 1) / (n_r + 1)))
    return out


def tune_band(
    pred_log2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    exact_rows: np.ndarray,
    lab_sir: np.ndarray,
    folds: np.ndarray,
    cal_folds: Sequence[int],
    calls_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    alpha_low: float = DEFAULT_ALPHA_LOW,
    target: float = DEFAULT_VME_TARGET,
    callable_pair: bool = True,
    strict_folds: bool = False,
) -> tuple[float, bool, float | None]:
    """Choose the upper level by nested cross-conformal calls on ``cal_folds`` only.

    The first (narrowest) level of :func:`passing_levels` is returned with
    ``gate_open = True``; if none passes the smallest level of ``grid`` is returned with
    ``gate_open = False`` (an empty grid closes the gate). The VME rule counts only the
    folds that issue ``likely_active`` calls (see :func:`passing_levels`).

    Args:
        pred_log2: ``log2`` of the rounded-up (capped) out-of-fold predictions; NaN = none.
        lab_sir: Lab S/I/R per row (re-derived under the call breakpoint); ``None`` = unknown.
        folds: Fold number per row (NaN outside the train folds).
        grid: Candidate upper miscoverage levels, narrowest band (largest alpha) first.
        callable_pair: False (no call breakpoint, or natural resistance) skips tuning:
            ``(0.05, True, None)``; calls are fixed by the override or absent.

    Returns:
        ``(alpha_up, gate_open, inner_vme_ucb)``; the UCB is over the calling folds.
    """
    if not callable_pair:
        return 0.05, True, None
    grid = [float(a) for a in grid]
    if not grid:
        return float(DEFAULT_ALPHA_GRID[-1]), False, None
    passed = passing_levels(pred_log2, lo, hi, exact_rows, lab_sir, folds, cal_folds, calls_fn, grid=grid,
                            alpha_low=alpha_low, target=target, strict_folds=strict_folds)
    if passed:
        return passed[0][0], True, passed[0][1]
    return float(grid[-1]), False, None


DEFAULT_Q_LOW_DISAGREEMENT = 1.0
"""Steps by which leave-one-fold-out lower half-widths may differ before the lower end is widened."""


def robust_q_low(
    signed_by_fold: Sequence[np.ndarray],
    q_low: float,
    alpha_low: float,
    alpha_sym: float,
    *,
    max_disagreement: float = DEFAULT_Q_LOW_DISAGREEMENT,
) -> tuple[float, bool]:
    """Lower half-width that does not collapse when one calibration fold behaves differently.

    ``signed_by_fold`` holds the signed exact residuals (``log2(lab) - pred``) of each
    calibration fold. For every fold ``g`` the lower half-width is recomputed without
    ``g`` (leave-one-fold-out). When those values differ by more than
    ``max_disagreement`` steps the lower quantile is unstable across folds, and the
    returned half-width is the largest of: the pooled ``q_low``, every leave-one-fold-out
    value, and the symmetric half-width (the ``1 - alpha_sym`` quantile of ``|residual|``,
    i.e. the lower end of the legacy symmetric band). Otherwise ``q_low`` is returned
    unchanged. Fewer than two folds with residuals: unchanged.

    Returns:
        ``(q_low, widened)``.
    """
    parts = [np.asarray(r, dtype=np.float64).ravel() for r in signed_by_fold]
    parts = [r for r in parts if r.size]
    if len(parts) < 2:
        return float(q_low), False
    lofo = []
    for i in range(len(parts)):
        rest = np.concatenate([r for j, r in enumerate(parts) if j != i])
        lofo.append(asym_quantiles(rest, 0.5, alpha_low)[1])
    if max(lofo) - min(lofo) <= max_disagreement + 1e-12:
        return float(q_low), False
    pooled = np.abs(np.concatenate(parts))
    q_sym = level_quantile(pooled, 1 - alpha_sym)
    if not math.isfinite(q_sym):
        q_sym = max(FALLBACK_STEPS, float(pooled.max()))
    return float(max(q_low, max(lofo), q_sym)), True


def calling_fold_vme(calls: np.ndarray, lab_sir: np.ndarray, folds: np.ndarray) -> tuple[int, int, int]:
    """``(n_vme, n_lab_R, n_calling_folds)`` over the folds that issue at least one ``likely_active`` call.

    Used on the out-of-fold CV calls to check the shipped bundle's gate: folds without an
    active call are left out so they cannot dilute the VME of the folds that have one.
    """
    calls = np.asarray(calls, dtype=object)
    active = np.array([c == "likely_active" for c in calls], dtype=bool)
    is_r = np.array([v == "R" for v in np.asarray(lab_sir, dtype=object)], dtype=bool)
    folds = np.asarray(folds, dtype=np.float64)
    calling = np.unique(folds[active & ~np.isnan(folds)])
    in_calling = np.isin(folds, calling)
    return int((active & is_r).sum()), int((is_r & in_calling).sum()), int(calling.size)


# --------------------------------------------------------------------------- #
# Per-fold safety: exact binomial test of call VME and the upper-end guard
# --------------------------------------------------------------------------- #

DEFAULT_FOLD_P_THRESHOLD = 0.01
"""One-sided p-value below which a fold's rate counts as significantly above its target."""


def binomial_excess_p(k: int, n: int, rate: float) -> float:
    """One-sided exact binomial p-value ``P(X >= k)`` for ``X ~ Binomial(n, rate)``.

    ``1.0`` when ``k <= 0`` or ``n == 0`` (nothing observed cannot be "too many").
    """
    from scipy.stats import binom  # noqa: PLC0415

    k, n = int(k), int(n)
    if k <= 0 or n <= 0:
        return 1.0
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    return float(binom.sf(k - 1, n, rate))


def fold_call_vme_table(
    calls: np.ndarray,
    lab_sir: np.ndarray,
    folds: np.ndarray,
    target: float = DEFAULT_VME_TARGET,
    p_threshold: float = DEFAULT_FOLD_P_THRESHOLD,
) -> list[dict[str, Any]]:
    """Per fold: lab R, VMEs (lab R called ``likely_active``), the rate and its one-sided p-value.

    ``calling`` = the fold issued at least one ``likely_active`` call; ``significant`` =
    calling and ``p < p_threshold`` under ``Binomial(n_lab_R, target)``. Rows with a NaN
    fold are ignored.
    """
    calls = np.asarray(calls, dtype=object)
    folds = np.asarray(folds, dtype=np.float64)
    active = np.array([c == "likely_active" for c in calls], dtype=bool)
    is_r = np.array([v == "R" for v in np.asarray(lab_sir, dtype=object)], dtype=bool)
    out: list[dict[str, Any]] = []
    for fold in sorted({int(f) for f in folds[~np.isnan(folds)]}):
        in_f = folds == fold
        n_r = int((is_r & in_f).sum())
        k = int((is_r & active & in_f).sum())
        calling = bool((active & in_f).any())
        p = binomial_excess_p(k, n_r, target)
        out.append({
            "fold": fold,
            "calling": calling,
            "n_lab_r": n_r,
            "n_vme": k,
            "call_vme": (k / n_r) if n_r else None,
            "p_value": p,
            "significant": bool(calling and p < p_threshold),
        })
    return out


def robust_q_up(
    signed_by_fold: Sequence[np.ndarray],
    q_up: float,
    alpha_up: float,
    *,
    p_threshold: float = DEFAULT_FOLD_P_THRESHOLD,
) -> tuple[float, bool]:
    """Upper half-width that no calibration fold misses significantly more often than ``alpha_up``.

    ``signed_by_fold`` holds the signed exact residuals (``log2(lab) - pred``) of each
    calibration fold. A fold *misses* the upper end when a residual exceeds ``q_up``. When
    some fold's miss count is significantly above ``alpha_up`` (one-sided exact binomial
    p < ``p_threshold``; e.g. a fold whose lab MICs sit far above the predictions), the
    upper half-width is raised to the smallest residual value at which no fold is
    significant any more (at worst the largest residual: no misses). Otherwise ``q_up``
    is returned unchanged. The upper-end analogue of :func:`robust_q_low`.

    Returns:
        ``(q_up, widened)``.
    """
    parts = [np.asarray(r, dtype=np.float64).ravel() for r in signed_by_fold]
    parts = [r for r in parts if r.size]
    if not parts or not 0 < alpha_up < 1:
        return float(q_up), False

    def any_significant(q: float) -> bool:
        return any(binomial_excess_p(int((r > q + 1e-9).sum()), r.size, alpha_up) < p_threshold for r in parts)

    if not any_significant(float(q_up)):
        return float(q_up), False
    candidates = np.unique(np.concatenate(parts))
    for q in candidates[candidates > q_up + 1e-9]:
        if not any_significant(float(q)):
            return float(q), True
    return float(max(q_up, candidates.max())), True
