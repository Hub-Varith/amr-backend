"""Tests for ``genome2mic.models.calibration`` (P(drug works) from the predicted MIC)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from genome2mic.models import calibration as cal


def test_step_distance_and_works() -> None:
    d = cal.step_distance([0.5, 2.0, 8.0, np.nan], 2.0)
    assert d[:3].tolist() == [-2.0, 0.0, 2.0] and math.isnan(d[3])
    assert np.isnan(cal.step_distance([1.0], None)).all()
    assert cal.works_from_sir(["S", "I", "R", None, "x"]).tolist()[:3] == [1.0, 0.0, 0.0]
    assert np.isnan(cal.works_from_sir([None, "x"])).all()


def test_fit_prob_map_is_non_increasing_bounded_and_flat_outside() -> None:
    rng = np.random.default_rng(0)
    d = rng.integers(-4, 5, size=400).astype(float)
    works = (rng.random(400) < 1 / (1 + np.exp(1.5 * d))).astype(float)
    pm = cal.fit_prob_map(d, works)
    assert pm is not None and pm.n == 400 and pm.n_works == int(works.sum())
    grid = np.linspace(-10, 10, 101)
    p = pm.predict(grid)
    assert (np.diff(p) <= 1e-12).all() and (p >= 0).all() and (p <= 1).all()
    assert p[0] == pytest.approx(pm.y[0]) and p[-1] == pytest.approx(pm.y[-1])  # flat beyond the knots
    assert math.isnan(pm.predict([np.nan])[0])


def test_fit_prob_map_skips_unknown_rows_and_handles_degenerate_input() -> None:
    pm = cal.fit_prob_map([0.0, 1.0, 2.0, 3.0], [1.0, np.nan, 0.0, np.nan])
    assert pm is not None and pm.n == 2
    assert cal.fit_prob_map([np.nan], [1.0]) is None
    one = cal.fit_prob_map([1.0, 1.0, 1.0, 1.0], [1, 1, 0, 1])
    assert one is not None and one.predict([-5, 5]).tolist() == [0.75, 0.75]


def test_prob_map_rejects_an_increasing_map() -> None:
    with pytest.raises(ValueError, match="non-increasing"):
        cal.ProbMap(x=(0.0, 1.0), y=(0.2, 0.8), n=2, n_works=1)


def _pair_data(n_marker: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 300
    pred = 2.0 ** rng.integers(-3, 6, size=n).astype(float)  # S breakpoint 2 -> d in [-4, 4]
    d = np.log2(pred) - 1.0
    works = (rng.random(n) < 1 / (1 + np.exp(2 * d))).astype(float)
    marker = np.zeros(n, dtype=bool)
    marker[:n_marker] = True
    works[:n_marker] = 1.0  # override rows that (oddly) all worked: must not leak into the main map
    return pred, works, marker


def test_override_rows_get_their_own_map_or_the_fallback() -> None:
    pred, works, marker = _pair_data(n_marker=10)
    few = cal.fit_pair_calibration(pred, works, marker, 2.0)
    assert few.override is None and few.override_constant == cal.OVERRIDE_FALLBACK_PROB and few.n_override == 10
    p = few.predict(pred, marker)
    assert (p[marker] == 0.02).all()
    # The main map is fit on the non-override rows only.
    main_only = cal.fit_prob_map(np.log2(pred[~marker]) - 1.0, works[~marker])
    assert few.main == main_only
    pred, works, marker = _pair_data(n_marker=40)
    many = cal.fit_pair_calibration(pred, works, marker, 2.0)
    assert many.override is not None and many.override_constant is None and many.n_override == 40
    assert np.allclose(many.predict(pred, marker)[marker], 1.0)  # every override row here was lab S


def test_natural_resistance_and_missing_breakpoint() -> None:
    pred, works, marker = _pair_data(n_marker=0)
    nat = cal.fit_pair_calibration(pred, works, marker, 2.0, natural_resistance=True)
    assert (nat.predict(pred) == 0.0).all() and nat.predict_one(1.0) == 0.0
    nobp = cal.fit_pair_calibration(pred, works, marker, None)
    assert np.isnan(nobp.predict(pred)).all() and nobp.predict_one(1.0) is None


def test_cross_fit_uses_only_the_other_folds() -> None:
    pred, works, marker = _pair_data(n_marker=0, seed=3)
    folds = np.arange(pred.size) % 5
    p, fits = cal.cross_fit_probabilities(pred, works, marker, folds, 2.0)
    assert sorted(fits) == [0, 1, 2, 3, 4] and not np.isnan(p).any()
    # Flipping fold 0's labels cannot change fold 0's probabilities (they never fit its map) ...
    flipped = works.copy()
    flipped[folds == 0] = 1.0 - flipped[folds == 0]
    q, _ = cal.cross_fit_probabilities(pred, flipped, marker, folds, 2.0)
    assert np.array_equal(p[folds == 0], q[folds == 0])
    # ... but it does change the other folds' maps.
    assert not np.array_equal(p[folds != 0], q[folds != 0])
    # Rows without a fold get no probability.
    folds_nan = folds.astype(float)
    folds_nan[:3] = np.nan
    r, _ = cal.cross_fit_probabilities(pred, works, marker, folds_nan, 2.0)
    assert np.isnan(r[:3]).all()


def test_json_round_trip(tmp_path) -> None:  # noqa: ANN001
    pred, works, marker = _pair_data(n_marker=40)
    fit = cal.fit_pair_calibration(pred, works, marker, 2.0)
    path = tmp_path / cal.CALIBRATION_FILE
    fit.save(path, extra={"model": "aft_known"})
    back = cal.PairCalibration.load(path)
    assert back == fit
    assert np.allclose(back.predict(pred, marker), fit.predict(pred, marker))
    with pytest.raises(ValueError):
        cal.PairCalibration.from_json({"kind": "something else"})


def test_tier_of_matches_prob_tier() -> None:
    values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    assert [cal.tier_of(v) for v in values] == cal.prob_tier(values).tolist()
    assert cal.tier_of(None) is None and cal.tier_of(float("nan")) is None
