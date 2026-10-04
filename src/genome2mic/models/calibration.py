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
  ``strong_subclasses`` Subclass) -> **one smoothed rate per pair**,
  ``(works + 1) / (n + 2)`` over the override rows of the fitting rows (v0.7; never a
  d-dependent map, which reached 1.0 next to a ``likely_inactive`` call). The main map
  is fit on the rows **without** the override, the population it is applied to.

Main map source (v0.7, :func:`species_fit` / :func:`species_cross_fit`): per pair, the
main map is the pair's own isotonic map (``pair``), the species-pooled isotonic map on
the same ``d`` (``species``: every non-override row of every drug of the species), or
the shrinkage blend ``w * pair + (1 - w) * species`` with ``w = n_pair / (n_pair + 100)``
(``blend``). The source is chosen by cross-fitted Brier score **inside the fitting
folds** (inner folds), ties to ``pair``; the chosen map is then fit on every fitting fold.

Displayed tier (:func:`tier_for_call`): a 'works' tier is never shown next to a
``likely_inactive`` call, nor a 'fails' tier next to ``likely_active``; both become
``uncertain``. The probability itself is not changed.

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
    "MAIN_SOURCES",
    "MIN_OVERRIDE_ROWS",
    "NATURAL_RESISTANCE_PROB",
    "OVERRIDE_FALLBACK_PROB",
    "PROB_TIERS",
    "SHRINK_N0",
    "TIER_THRESHOLDS",
    "WORKS_TIERS",
    "CalRows",
    "PairCalibration",
    "ProbMap",
    "blend_maps",
    "cross_fit_probabilities",
    "fit_pair_calibration",
    "fit_prob_map",
    "prob_tier",
    "prob_tier_for_calls",
    "smoothed_rate",
    "species_cross_fit",
    "species_fit",
    "step_distance",
    "summarize_fits",
    "tier_for_call",
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
"""Legacy (v0.6): override rows needed for an override isotonic map. v0.7 always uses
:func:`smoothed_rate`; kept so v0.6 ``calibration.json`` files still load."""
OVERRIDE_FALLBACK_PROB = 0.02
"""Legacy (v0.6) constant for override rows; v0.7 uses :func:`smoothed_rate`."""
SHRINK_N0 = 100.0
"""Blend weight ``w = n_pair / (n_pair + SHRINK_N0)`` of the ``blend`` main-map source (fixed, not tuned)."""
SOURCE_PAIR = "pair"
SOURCE_BLEND = "blend"
SOURCE_SPECIES = "species"
MAIN_SOURCES: tuple[str, ...] = (SOURCE_PAIR, SOURCE_BLEND, SOURCE_SPECIES)
"""Main-map sources in tie-break preference order (the pair's own map first)."""
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


def tier_for_call(p: float | None, call: str | None) -> str | None:
    """Displayed tier, capped to agree with the call (the probability is not changed).

    A 'works' tier next to ``likely_inactive`` (e.g. a strong-marker override on a pair
    where marker carriers often test S) and a 'fails' tier next to ``likely_active``
    become ``uncertain``. Any other call (``uncertain``, ``None``) keeps :func:`tier_of`.
    """
    tier = tier_of(p)
    if tier is None:
        return None
    if call == "likely_inactive" and tier in WORKS_TIERS:
        return TIER_UNCERTAIN
    if call == "likely_active" and tier in FAILS_TIERS:
        return TIER_UNCERTAIN
    return tier


def prob_tier_for_calls(p: Any, calls: Any) -> np.ndarray:
    """Vectorised :func:`tier_for_call`."""
    values = np.asarray(p, dtype=np.float64).ravel()
    call_arr = np.asarray(calls, dtype=object).ravel()
    if call_arr.size != values.size:
        raise ValueError(f"calls has {call_arr.size} entries for {values.size} probabilities")
    return np.array([tier_for_call(v, c if isinstance(c, str) else None) for v, c in zip(values, call_arr)], dtype=object)


def smoothed_rate(works: Any) -> tuple[float, int]:
    """``((works + 1) / (n + 2), n)`` over the rows with a known lab category (Laplace smoothing)."""
    y = np.asarray(works, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    return float((y.sum() + 1.0) / (y.size + 2.0)), int(y.size)


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


def blend_maps(pair: ProbMap | None, species: ProbMap | None, w: float) -> ProbMap | None:
    """``w * pair + (1 - w) * species`` as one :class:`ProbMap` (exact: both are piecewise linear).

    The union of both knot sets carries the blend exactly (each map is linear between its
    own knots and flat beyond them), so the result is again non-increasing. ``None`` maps
    fall back to the other one.
    """
    if pair is None:
        return species
    if species is None:
        return pair
    w = float(min(max(w, 0.0), 1.0))
    xs = np.union1d(np.asarray(pair.x, dtype=np.float64), np.asarray(species.x, dtype=np.float64))
    ys = w * pair.predict(xs) + (1.0 - w) * species.predict(xs)
    ys = np.minimum.accumulate(np.clip(ys, 0.0, 1.0))  # guard float noise: non-increasing
    return ProbMap(x=tuple(float(v) for v in xs), y=tuple(float(v) for v in ys), n=int(pair.n), n_works=int(pair.n_works))


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
    main_source: str | None = None
    """v0.7: ``pair`` / ``blend`` / ``species`` (:data:`MAIN_SOURCES`); ``None`` = the pair's own map (v0.6)."""
    blend_weight: float | None = None
    """v0.7: weight ``w`` of the pair map in the shipped ``blend`` (also recorded for the other sources)."""

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
            "override_rule": "smoothed rate (works + 1) / (n + 2) over the pair's strong-marker rows",
            "min_override_rows": int(self.min_override_rows),
            "main_source": self.main_source,
            "blend_weight": None if self.blend_weight is None else float(self.blend_weight),
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
            main_source=data.get("main_source"),
            blend_weight=None if data.get("blend_weight") is None else float(data["blend_weight"]),
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
    """Fit the main map on the rows without an override and the override rate on the rows with one.

    v0.7: the override rows get one smoothed rate ``(works + 1) / (n + 2)`` over those
    with a prediction and a known lab category (:func:`smoothed_rate`), whatever their
    predicted MIC. ``min_override_rows`` / ``override_fallback`` are accepted for
    compatibility and ignored.
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
    rate, n_override = smoothed_rate(y[usable & marker])
    return PairCalibration(
        s_breakpoint=float(s_breakpoint),
        main=main,
        override=None,
        override_constant=rate,
        natural_resistance=False,
        n_override=n_override,
        min_override_rows=int(min_override_rows),
        main_source=SOURCE_PAIR,
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
            "override": "isotonic" if v.override is not None else ("smoothed_rate" if v.override_constant is not None else None),
            "override_rate": v.override_constant,
            "main_source": v.main_source,
        }
        for k, v in sorted(fits.items())
    }


# --------------------------------------------------------------------------- #
# Species-pooled main map, chosen per pair inside the fitting folds (v0.7)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalRows:
    """One species x drug pair's out-of-fold rows for the species-level calibration.

    ``pred_mic`` (rounded-up predicted MIC), ``works`` (1 / 0 / NaN unknown), ``marker``
    (strong-marker override fires) and ``folds`` (CV fold; NaN = not a CV row, never
    fits anything) are row-aligned.
    """

    pred_mic: np.ndarray
    works: np.ndarray
    marker: np.ndarray
    folds: np.ndarray
    s_breakpoint: float | None
    natural_resistance: bool = False

    def __post_init__(self) -> None:
        n = np.asarray(self.pred_mic).size
        for name in ("works", "marker", "folds"):
            if np.asarray(getattr(self, name)).size != n:
                raise ValueError(f"CalRows.{name} has {np.asarray(getattr(self, name)).size} entries for {n} rows")

    @property
    def active(self) -> bool:
        """Takes part in the pooled map: has an S breakpoint and is not naturally resistant."""
        return self.s_breakpoint is not None and not self.natural_resistance

    def d(self) -> np.ndarray:
        return step_distance(self.pred_mic, self.s_breakpoint)


class _SpeciesPool:
    """Pre-computed arrays and a per-fold-set cache of the species map."""

    def __init__(self, rows_by_pair: Mapping[str, CalRows], n0: float) -> None:
        self.rows = dict(rows_by_pair)
        self.n0 = float(n0)
        self.d: dict[str, np.ndarray] = {}
        self.y: dict[str, np.ndarray] = {}
        self.f: dict[str, np.ndarray] = {}
        self.main_ok: dict[str, np.ndarray] = {}
        self.marker_ok: dict[str, np.ndarray] = {}
        pool_d, pool_y, pool_f = [], [], []
        for key, r in self.rows.items():
            d = r.d() if r.active else np.full(np.asarray(r.pred_mic).size, np.nan)
            y = np.asarray(r.works, dtype=np.float64).ravel()
            f = np.asarray(r.folds, dtype=np.float64).ravel()
            m = np.asarray(r.marker, dtype=bool).ravel()
            pred_ok = np.isfinite(np.asarray(r.pred_mic, dtype=np.float64).ravel())
            self.d[key], self.y[key], self.f[key] = d, y, f
            self.main_ok[key] = np.isfinite(d) & np.isfinite(y) & np.isfinite(f) & ~m
            self.marker_ok[key] = pred_ok & np.isfinite(y) & np.isfinite(f) & m
            if r.active:
                keep = self.main_ok[key]
                pool_d.append(d[keep])
                pool_y.append(y[keep])
                pool_f.append(f[keep])
        self.pool_d = np.concatenate(pool_d) if pool_d else np.empty(0)
        self.pool_y = np.concatenate(pool_y) if pool_y else np.empty(0)
        self.pool_f = np.concatenate(pool_f) if pool_f else np.empty(0)
        self._species_cache: dict[frozenset[int], ProbMap | None] = {}

    def species_map(self, folds: frozenset[int]) -> ProbMap | None:
        if folds not in self._species_cache:
            keep = np.isin(self.pool_f, list(folds))
            self._species_cache[folds] = fit_prob_map(self.pool_d[keep], self.pool_y[keep])
        return self._species_cache[folds]

    def maps(self, key: str, folds: frozenset[int]) -> tuple[dict[str, ProbMap | None], float]:
        keep = self.main_ok[key] & np.isin(self.f[key], list(folds))
        pair = fit_prob_map(self.d[key][keep], self.y[key][keep])
        species = self.species_map(folds)
        n = 0 if pair is None else pair.n
        w = n / (n + self.n0)
        blend = None if pair is None or species is None else blend_maps(pair, species, w)
        return {SOURCE_PAIR: pair, SOURCE_BLEND: blend, SOURCE_SPECIES: species}, float(w)

    def choose(self, key: str, folds: frozenset[int]) -> tuple[str, dict[str, float | None]]:
        """Source with the lowest Brier score over inner folds of ``folds`` (ties: :data:`MAIN_SOURCES` order)."""
        sse = {s: 0.0 for s in MAIN_SOURCES}
        count = {s: 0 for s in MAIN_SOURCES}
        missing = {s: False for s in MAIN_SOURCES}
        if len(folds) >= 2:
            for g in sorted(folds):
                inner = frozenset(folds - {g})
                rows = self.main_ok[key] & (self.f[key] == g)
                if not rows.any():
                    continue
                maps, _ = self.maps(key, inner)
                for source in MAIN_SOURCES:
                    if maps[source] is None:
                        missing[source] = True
                        continue
                    p = maps[source].predict(self.d[key][rows])
                    sse[source] += float(np.sum((p - self.y[key][rows]) ** 2))
                    count[source] += int(rows.sum())
        brier: dict[str, float | None] = {
            s: (sse[s] / count[s]) if count[s] and not missing[s] else None for s in MAIN_SOURCES
        }
        scored = [(brier[s], i, s) for i, s in enumerate(MAIN_SOURCES) if brier[s] is not None]
        if not scored:
            return SOURCE_PAIR, brier
        best = min(b for b, _, _ in scored)
        choice = next(s for b, _, s in sorted(scored, key=lambda t: t[1]) if b <= best + 1e-12)
        return choice, brier

    def fit(self, key: str, folds: frozenset[int]) -> tuple[PairCalibration, dict[str, Any]]:
        r = self.rows[key]
        if not r.active:
            return PairCalibration(s_breakpoint=r.s_breakpoint, main=None, override=None, override_constant=None,
                                   natural_resistance=bool(r.natural_resistance)), {"main_source": None}
        choice, brier = self.choose(key, folds)
        maps, w = self.maps(key, folds)
        main = maps[choice]
        if main is None:  # e.g. the pair has no fitting rows: use what exists
            choice = next((s for s in MAIN_SOURCES if maps[s] is not None), choice)
            main = maps[choice]
        marker_rows = self.marker_ok[key] & np.isin(self.f[key], list(folds))
        rate, n_override = smoothed_rate(self.y[key][marker_rows])
        fit = PairCalibration(
            s_breakpoint=float(r.s_breakpoint), main=main, override=None, override_constant=rate,
            natural_resistance=False, n_override=n_override, main_source=choice, blend_weight=w,
        )
        n_pair = int((self.main_ok[key] & np.isin(self.f[key], list(folds))).sum())
        return fit, {"main_source": choice, "brier": brier, "blend_weight": w, "n_pair": n_pair,
                     "n_species": int(np.isin(self.pool_f, list(folds)).sum()), "override_rate": rate,
                     "n_override": n_override}


def _all_folds(pool: _SpeciesPool) -> list[int]:
    folds: set[int] = set()
    for f in pool.f.values():
        folds |= {int(v) for v in f[np.isfinite(f)]}
    return sorted(folds)


def species_fit(
    rows_by_pair: Mapping[str, CalRows], *, n0: float = SHRINK_N0
) -> dict[str, tuple[PairCalibration, dict[str, Any]]]:
    """The shipped calibration per pair: source chosen by CV over every fold, then fit on every fold.

    ``rows_by_pair`` holds every pair of **one species** (out-of-fold rows only).
    Returns ``pair -> (PairCalibration, record)``; the record has the chosen
    ``main_source``, the inner Brier score of each source and the row counts.
    """
    pool = _SpeciesPool(rows_by_pair, n0)
    folds = frozenset(_all_folds(pool))
    return {key: pool.fit(key, folds) for key in pool.rows}


def species_cross_fit(
    rows_by_pair: Mapping[str, CalRows], *, n0: float = SHRINK_N0
) -> dict[str, tuple[np.ndarray, dict[int, dict[str, Any]]]]:
    """Cross-fitted ``P(works)`` per pair: fold ``f``'s rows use a source chosen and fit on the other folds.

    Neither the source choice (inner CV over the other folds) nor the pair map, the
    species map or the override rate ever sees fold ``f``'s labels. Rows with a NaN
    fold get NaN; natural-resistance pairs get 0; pairs without an S breakpoint NaN.
    """
    pool = _SpeciesPool(rows_by_pair, n0)
    every = _all_folds(pool)
    out: dict[str, tuple[np.ndarray, dict[int, dict[str, Any]]]] = {}
    for key, r in pool.rows.items():
        n = np.asarray(r.pred_mic).size
        p = np.full(n, np.nan)
        record: dict[int, dict[str, Any]] = {}
        f = pool.f[key]
        marker = np.asarray(r.marker, dtype=bool).ravel()
        pred = np.asarray(r.pred_mic, dtype=np.float64).ravel()
        for fold in every:
            held = f == fold
            fit, rec = pool.fit(key, frozenset(set(every) - {fold}))
            record[fold] = rec
            if held.any():
                p[held] = fit.predict(pred[held], marker[held])
        out[key] = (p, record)
    return out
