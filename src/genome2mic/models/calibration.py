"""Calibrated probability that a drug works: ``P(lab S under the call breakpoint)``.

The point model predicts an MIC; a clinician asks "will this drug work?". This module
turns the predicted MIC into that probability, calibrated on out-of-fold data only:

* ``d = log2(pred_mic) - log2(S breakpoint)`` -- how many doubling steps the (capped,
  rounded-up) prediction sits above the S breakpoint (negative = below).
* ``works`` = the lab MIC re-derived under the call breakpoint is ``S`` (1); ``I`` or
  ``R`` = 0; a lab interval that straddles a breakpoint is unknown and never used
  (no label is imputed).
* ``P(works)`` is an isotonic, **non-increasing** map ``d -> share of lab S``
  (:class:`ProbMap`, scikit-learn ``IsotonicRegression(increasing=False)``, linear
  between knots, flat beyond the ends).

Overrides keep the probability honest instead of forcing a number:

* natural resistance (``configs/natural_resistance.csv``) -> ``0``;
* strong-marker override present (the same rule as the call: ``strong_markers``
  columns and, when the subclass map is available, acquired genes with a
  ``strong_subclasses`` Subclass) -> a separate isotonic map fit on the override rows
  only when at least :data:`MIN_OVERRIDE_ROWS` of them exist in the fitting rows,
  else the constant :data:`OVERRIDE_FALLBACK_PROB` (0.02). The main map is fit on the
  rows **without** the override, the population it is applied to.

Cross-fitting (:func:`cross_fit_probabilities`): the probability written for a CV row
of fold ``f`` comes from a map fit on the CV rows of the *other* folds only. The bundle
stores the map fit on every out-of-fold row (``calibration.json``); test, LOLO and new
genomes use that one. The test split never fits anything.

Tiers (:func:`prob_tier`), symmetric around 0.5::

    very_likely_works   p >= 0.90
    probably_works      0.70 <= p < 0.90
    uncertain           0.30 <  p < 0.70
    probably_fails      0.10 <  p <= 0.30
    very_likely_fails   p <= 0.10

The probability is reported **next to** the call; it never replaces it. The call stays
the safety-gated decision (band vs breakpoint, overrides, active-call gate).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genome2mic.models.base import read_json, write_json

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATION_FILE",
    "FAILS_TIERS",
    "MIN_OVERRIDE_ROWS",
    "NATURAL_RESISTANCE_PROB",
    "OVERRIDE_FALLBACK_PROB",
    "PROB_TIERS",
    "TIER_THRESHOLDS",
    "WORKS_TIERS",
    "PairCalibration",
    "ProbMap",
    "cross_fit_probabilities",
    "fit_pair_calibration",
    "fit_prob_map",
    "prob_tier",
    "step_distance",
    "summarize_fits",
    "tier_of",
    "works_from_sir",
]

CALIBRATION_FILE = "calibration.json"
KIND = "isotonic_step_distance_to_s_breakpoint"

TIER_VERY_LIKELY_WORKS = "very_likely_works"
TIER_PROBABLY_WORKS = "probably_works"
TIER_UNCERTAIN = "uncertain"
TIER_PROBABLY_FAILS = "probably_fails"
TIER_VERY_LIKELY_FAILS = "very_likely_fails"
PROB_TIERS: tuple[str, ...] = (
    TIER_VERY_LIKELY_WORKS, TIER_PROBABLY_WORKS, TIER_UNCERTAIN, TIER_PROBABLY_FAILS, TIER_VERY_LIKELY_FAILS,
)
"""Tier names, from 'works' to 'fails'."""
WORKS_TIERS: frozenset[str] = frozenset({TIER_VERY_LIKELY_WORKS, TIER_PROBABLY_WORKS})
FAILS_TIERS: frozenset[str] = frozenset({TIER_PROBABLY_FAILS, TIER_VERY_LIKELY_FAILS})
TIER_THRESHOLDS: dict[str, float] = {
    "very_likely_works_min": 0.90,
    "probably_works_min": 0.70,
    "probably_fails_max": 0.30,
    "very_likely_fails_max": 0.10,
}
"""Tier edges: ``>= 0.90``, ``>= 0.70``, ``(0.30, 0.70)``, ``<= 0.30``, ``<= 0.10``."""

MIN_OVERRIDE_ROWS = 30
"""Strong-marker rows needed in the fitting rows before they get their own isotonic map."""
OVERRIDE_FALLBACK_PROB = 0.02
"""P(works) for strong-marker rows when fewer than :data:`MIN_OVERRIDE_ROWS` were seen."""
NATURAL_RESISTANCE_PROB = 0.0
"""P(works) under natural resistance (override 1)."""

_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #


def tier_of(p: float | None) -> str | None:
    """Tier of one probability (``None`` / NaN -> ``None``). Raises outside ``[0, 1]``."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return None
    value = float(p)
    if math.isnan(value):
        return None
    if value < -_TOL or value > 1 + _TOL:
        raise ValueError(f"probability must be in [0, 1], got {value!r}")
    if value >= TIER_THRESHOLDS["very_likely_works_min"] - _TOL:
        return TIER_VERY_LIKELY_WORKS
    if value >= TIER_THRESHOLDS["probably_works_min"] - _TOL:
        return TIER_PROBABLY_WORKS
    if value <= TIER_THRESHOLDS["very_likely_fails_max"] + _TOL:
        return TIER_VERY_LIKELY_FAILS
    if value <= TIER_THRESHOLDS["probably_fails_max"] + _TOL:
        return TIER_PROBABLY_FAILS
    return TIER_UNCERTAIN


def prob_tier(p: Any) -> np.ndarray:
    """Vectorised :func:`tier_of`: object array of tier names, ``None`` where ``p`` is null."""
    values = np.asarray(p, dtype=np.float64).ravel()
    return np.array([tier_of(v) for v in values], dtype=object)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def step_distance(pred_mic: Any, s_breakpoint: float | None) -> np.ndarray:
    """``log2(pred_mic) - log2(s_breakpoint)`` in doubling steps; NaN where ``pred_mic`` is null.

    All NaN when there is no S breakpoint.
    """
    pred = np.asarray(pred_mic, dtype=np.float64).ravel()
    if s_breakpoint is None or not math.isfinite(float(s_breakpoint)) or float(s_breakpoint) <= 0:
        return np.full(pred.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log2(pred) - math.log2(float(s_breakpoint))
    out[~np.isfinite(pred) | (pred <= 0)] = np.nan
    return out


def works_from_sir(lab_sir: Any) -> np.ndarray:
    """``1.0`` for lab ``S``, ``0.0`` for ``I`` / ``R``, NaN when the lab category is unknown."""
    out = []
    for value in np.asarray(lab_sir, dtype=object).ravel():
        if not isinstance(value, str):
            out.append(np.nan)  # None, NaN, pd.NA: unknown, never imputed
        elif value == "S":
            out.append(1.0)
        elif value in ("I", "R"):
            out.append(0.0)
        else:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Isotonic map
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProbMap:
    """Non-increasing piecewise-linear map ``d -> P(works)`` given by its knots.

    ``predict`` interpolates linearly between knots and is flat beyond the first and
    last knot (scikit-learn's ``out_of_bounds="clip"``). ``n`` / ``n_works`` are the
    fitting rows and how many of them were lab S.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    n: int
    n_works: int

    def __post_init__(self) -> None:
        if len(self.x) == 0 or len(self.x) != len(self.y):
            raise ValueError("ProbMap needs the same, non-zero number of x and y knots")
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("ProbMap knots must be finite")
        if (np.diff(x) < -_TOL).any():
            raise ValueError("ProbMap x knots must be sorted ascending")
        if (np.diff(y) > _TOL).any():
            raise ValueError("ProbMap must be non-increasing (a higher predicted MIC never raises P(works))")
        if (y < -_TOL).any() or (y > 1 + _TOL).any():
            raise ValueError("ProbMap probabilities must be in [0, 1]")

    def predict(self, d: Any) -> np.ndarray:
        values = np.asarray(d, dtype=np.float64).ravel()
        out = np.interp(values, np.asarray(self.x), np.asarray(self.y))
        out[np.isnan(values)] = np.nan
        return np.clip(out, 0.0, 1.0)

    def as_json(self) -> dict[str, Any]:
        return {"x": [float(v) for v in self.x], "y": [float(v) for v in self.y], "n": int(self.n),
                "n_works": int(self.n_works)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ProbMap:
        return cls(x=tuple(float(v) for v in data["x"]), y=tuple(float(v) for v in data["y"]),
                   n=int(data.get("n", 0)), n_works=int(data.get("n_works", 0)))


def fit_prob_map(d: Any, works: Any) -> ProbMap | None:
    """Isotonic non-increasing fit of ``works`` (0/1) on ``d``; ``None`` without a usable row.

    Rows with a NaN ``d`` or ``works`` are skipped (never imputed).
    """
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    x = np.asarray(d, dtype=np.float64).ravel()
    y = np.asarray(works, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"d has {x.size} entries, works {y.size}")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size == 0:
        return None
    if np.unique(x).size == 1:
        share = float(y.mean())
        return ProbMap(x=(float(x[0]),), y=(share,), n=int(x.size), n_works=int(y.sum()))
    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)
    return ProbMap(
        x=tuple(float(v) for v in iso.X_thresholds_),
        y=tuple(float(v) for v in iso.y_thresholds_),
        n=int(x.size),
        n_works=int(y.sum()),
    )


# --------------------------------------------------------------------------- #
# One pair: main map + override handling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairCalibration:
    """Everything needed to turn a predicted MIC into ``P(works)`` for one species x drug.

    ``main`` applies to rows without a strong-marker override, ``override`` (or
    ``override_constant`` when too few override rows were seen) to rows with one.
    ``natural_resistance`` gives 0 for every row. Without an S breakpoint every
    probability is null.
    """

    s_breakpoint: float | None
    main: ProbMap | None
    override: ProbMap | None
    override_constant: float | None
    natural_resistance: bool = False
    n_override: int = 0
    min_override_rows: int = MIN_OVERRIDE_ROWS

    def predict(self, pred_mic: Any, strong_marker: Any = None) -> np.ndarray:
        """``P(works)`` per row (NaN where it cannot be given: no breakpoint, no prediction, no map)."""
        pred = np.asarray(pred_mic, dtype=np.float64).ravel()
        n = pred.size
        if self.natural_resistance:
            return np.full(n, NATURAL_RESISTANCE_PROB)
        if self.s_breakpoint is None:
            return np.full(n, np.nan)
        marker = np.zeros(n, dtype=bool) if strong_marker is None else np.asarray(strong_marker, dtype=bool).ravel()
        if marker.size != n:
            raise ValueError(f"strong_marker has {marker.size} entries for {n} rows")
        d = step_distance(pred, self.s_breakpoint)
        out = np.full(n, np.nan)
        plain = ~marker
        if self.main is not None and plain.any():
            out[plain] = self.main.predict(d[plain])
        if marker.any():
            if self.override is not None:
                out[marker] = self.override.predict(d[marker])
            elif self.override_constant is not None:
                out[marker] = float(self.override_constant)
        return out

    def predict_one(self, pred_mic: float | None, strong_marker: bool = False) -> float | None:
        """Scalar :meth:`predict` (``None`` instead of NaN)."""
        value = self.predict(np.array([np.nan if pred_mic is None else pred_mic]), np.array([bool(strong_marker)]))[0]
        return None if math.isnan(value) else float(value)

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": KIND,
            "target": "lab S re-derived under the call breakpoint (I and R = does not work)",
            "s_breakpoint": None if self.s_breakpoint is None else float(self.s_breakpoint),
            "natural_resistance": bool(self.natural_resistance),
            "main": None if self.main is None else self.main.as_json(),
            "override": None if self.override is None else self.override.as_json(),
            "override_constant": None if self.override_constant is None else float(self.override_constant),
            "n_override": int(self.n_override),
            "min_override_rows": int(self.min_override_rows),
            "tiers": dict(TIER_THRESHOLDS),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PairCalibration:
        if data.get("kind") != KIND:
            raise ValueError(f"calibration kind must be {KIND!r}, got {data.get('kind')!r}")
        s = data.get("s_breakpoint")
        const = data.get("override_constant")
        return cls(
            s_breakpoint=None if s is None else float(s),
            main=None if data.get("main") is None else ProbMap.from_json(data["main"]),
            override=None if data.get("override") is None else ProbMap.from_json(data["override"]),
            override_constant=None if const is None else float(const),
            natural_resistance=bool(data.get("natural_resistance", False)),
            n_override=int(data.get("n_override", 0)),
            min_override_rows=int(data.get("min_override_rows", MIN_OVERRIDE_ROWS)),
        )

    def save(self, path: Path, extra: Mapping[str, Any] | None = None) -> None:
        payload = self.as_json()
        if extra:
            payload.update(extra)
        write_json(Path(path), payload)

    @classmethod
    def load(cls, path: Path) -> PairCalibration:
        return cls.from_json(read_json(Path(path)))


def fit_pair_calibration(
    pred_mic: Any,
    works: Any,
    strong_marker: Any,
    s_breakpoint: float | None,
    *,
    natural_resistance: bool = False,
    min_override_rows: int = MIN_OVERRIDE_ROWS,
    override_fallback: float = OVERRIDE_FALLBACK_PROB,
) -> PairCalibration:
    """Fit the main map on the rows without an override and the override map on the rows with one.

    The override rows get their own isotonic map only when at least ``min_override_rows``
    of them have a prediction and a known lab category; otherwise the constant
    ``override_fallback``.
    """
    pred = np.asarray(pred_mic, dtype=np.float64).ravel()
    y = np.asarray(works, dtype=np.float64).ravel()
    marker = np.zeros(pred.size, dtype=bool) if strong_marker is None else np.asarray(strong_marker, dtype=bool).ravel()
    if not (pred.size == y.size == marker.size):
        raise ValueError("pred_mic, works and strong_marker must have the same length")
    if natural_resistance or s_breakpoint is None:
        return PairCalibration(s_breakpoint=s_breakpoint, main=None, override=None, override_constant=None,
                               natural_resistance=bool(natural_resistance), n_override=0,
                               min_override_rows=min_override_rows)
    d = step_distance(pred, s_breakpoint)
    usable = np.isfinite(d) & np.isfinite(y)
    main = fit_prob_map(d[~marker], y[~marker])
    n_override = int((usable & marker).sum())
    override = fit_prob_map(d[marker], y[marker]) if n_override >= min_override_rows else None
    return PairCalibration(
        s_breakpoint=float(s_breakpoint),
        main=main,
        override=override,
        override_constant=None if override is not None else float(override_fallback),
        natural_resistance=False,
        n_override=n_override,
        min_override_rows=int(min_override_rows),
    )


def cross_fit_probabilities(
    pred_mic: Any,
    works: Any,
    strong_marker: Any,
    folds: Any,
    s_breakpoint: float | None,
    *,
    natural_resistance: bool = False,
    min_override_rows: int = MIN_OVERRIDE_ROWS,
    override_fallback: float = OVERRIDE_FALLBACK_PROB,
) -> tuple[np.ndarray, dict[int, PairCalibration]]:
    """``P(works)`` for every row from a calibration fit on the *other* folds only.

    Rows with a NaN fold get NaN. Returns the probabilities and the per-fold fits
    (keyed by fold), so the bundle can record what each fold's rows were scored with.
    """
    pred = np.asarray(pred_mic, dtype=np.float64).ravel()
    y = np.asarray(works, dtype=np.float64).ravel()
    marker = np.zeros(pred.size, dtype=bool) if strong_marker is None else np.asarray(strong_marker, dtype=bool).ravel()
    f = np.asarray(folds, dtype=np.float64).ravel()
    if not (pred.size == y.size == marker.size == f.size):
        raise ValueError("pred_mic, works, strong_marker and folds must have the same length")
    out = np.full(pred.size, np.nan)
    fits: dict[int, PairCalibration] = {}
    for fold in sorted({int(v) for v in f[~np.isnan(f)]}):
        held = f == fold
        fit_rows = ~np.isnan(f) & ~held
        cal = fit_pair_calibration(
            pred[fit_rows], y[fit_rows], marker[fit_rows], s_breakpoint, natural_resistance=natural_resistance,
            min_override_rows=min_override_rows, override_fallback=override_fallback,
        )
        fits[fold] = cal
        out[held] = cal.predict(pred[held], marker[held])
    return out, fits


def summarize_fits(fits: Mapping[int, PairCalibration]) -> dict[str, Any]:
    """Compact per-fold record for ``calibration.json`` (``n`` of each map, override handling)."""
    return {
        str(k): {
            "n_main": None if v.main is None else int(v.main.n),
            "n_override": int(v.n_override),
            "override": "isotonic" if v.override is not None else ("constant" if v.override_constant is not None else None),
        }
        for k, v in sorted(fits.items())
    }
