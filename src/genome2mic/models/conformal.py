"""Split-conformal uncertainty bands on log2 MIC.

CLAUDE.md: take ``|pred - true|`` in doubling steps on validation rows with exact
MICs; the finite-sample-corrected 90th percentile ``q`` gives a ``+-q``-step band.
The *upper* end of the band is what gets compared with the S breakpoint.

Residuals come from out-of-fold predictions (never the test set, CLAUDE.md rule 8);
the orchestrator feeds ``log2(pred_mic)`` of the rounded-up prediction here.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from genome2mic.droplog import DropLog
from genome2mic.mic import round_down_to_step_array, round_up_to_step_array
from genome2mic.models.base import exact_mask, read_json, validate_intervals, write_json

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.10
"""Miscoverage level: ``1 - alpha`` = 90% nominal band coverage."""


def residual_steps(
    pred_log2_rounded_up: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    droplog: DropLog | None = None,
) -> np.ndarray:
    """Absolute residuals in doubling steps on exact rows only.

    Args:
        pred_log2_rounded_up: ``log2(pred_mic)`` where ``pred_mic`` was rounded up
            to the grid, one value per row (NaN = no prediction; such rows are
            excluded and counted).
        lo, hi: Lab interval bounds in mg/L, aligned with the predictions.
        droplog: Where to record the censored/missing rows that were skipped. A
            local ``DropLog("conformal")`` is used when omitted.

    Returns:
        ``|pred_step - log2(hi)|`` for rows with ``lo > 0`` and ``hi < inf`` and a
        non-missing prediction, in input order.
    """
    pred = np.asarray(pred_log2_rounded_up, dtype=np.float64).ravel()
    low, high = validate_intervals(lo, hi, len(pred))
    log = droplog if droplog is not None else DropLog("conformal")
    exact = exact_mask(low, high)
    log.drop("censored row (conformal residuals use exact MICs only)", int((~exact).sum()))
    missing = np.isnan(pred)
    log.drop("missing prediction (conformal residuals)", int((missing & exact).sum()))
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
