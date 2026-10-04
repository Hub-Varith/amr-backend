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


def test_override_rows_get_one_smoothed_rate_not_a_d_map() -> None:
    """Strong-marker rows: one (works + 1) / (n + 2) rate per pair, never a d-dependent map reaching 1.0."""
    pred, works, marker = _pair_data(n_marker=10)
    few = cal.fit_pair_calibration(pred, works, marker, 2.0)
    assert few.override is None and few.n_override == 10
    assert few.override_constant == pytest.approx(11 / 12)  # all 10 worked -> 11/12, never 1.0
    p = few.predict(pred, marker)
    assert np.allclose(p[marker], 11 / 12) and (p[marker] < 1.0).all()
    # The main map is fit on the non-override rows only.
    main_only = cal.fit_prob_map(np.log2(pred[~marker]) - 1.0, works[~marker])
    assert few.main == main_only
    pred, works, marker = _pair_data(n_marker=40)
    works[:40] = 0.0
    works[:4] = 1.0
    many = cal.fit_pair_calibration(pred, works, marker, 2.0)
    assert many.override is None and many.override_constant == pytest.approx(5 / 42) and many.n_override == 40
    assert np.unique(many.predict(pred, marker)[marker]).size == 1  # same value whatever the predicted MIC
    none = cal.fit_pair_calibration(*_pair_data(n_marker=0)[:2], np.zeros(300, dtype=bool), 2.0)
    assert none.override_constant == pytest.approx(0.5) and none.n_override == 0


def test_override_rate_is_cross_fitted_by_fold() -> None:
    pred, works, marker = _pair_data(n_marker=60, seed=5)
    folds = np.arange(pred.size) % 5
    p, fits = cal.cross_fit_probabilities(pred, works, marker, folds, 2.0)
    for f, fit in fits.items():
        rows = marker & (folds != f)
        expected = (works[rows].sum() + 1) / (rows.sum() + 2)
        assert fit.override_constant == pytest.approx(expected)
        assert np.allclose(p[marker & (folds == f)], expected)
    flipped = works.copy()
    flipped[(folds == 0) & marker] = 1.0 - flipped[(folds == 0) & marker]
    q, _ = cal.cross_fit_probabilities(pred, flipped, marker, folds, 2.0)
    assert np.array_equal(p[folds == 0], q[folds == 0])


def test_tier_is_capped_consistently_with_the_call() -> None:
    assert cal.tier_for_call(0.95, "likely_inactive") == cal.TIER_UNCERTAIN
    assert cal.tier_for_call(0.75, "likely_inactive") == cal.TIER_UNCERTAIN
    assert cal.tier_for_call(0.5, "likely_inactive") == cal.TIER_UNCERTAIN
    assert cal.tier_for_call(0.05, "likely_inactive") == cal.TIER_VERY_LIKELY_FAILS
    assert cal.tier_for_call(0.05, "likely_active") == cal.TIER_UNCERTAIN
    assert cal.tier_for_call(0.95, "likely_active") == cal.TIER_VERY_LIKELY_WORKS
    assert cal.tier_for_call(0.95, "uncertain") == cal.TIER_VERY_LIKELY_WORKS
    assert cal.tier_for_call(0.95, None) == cal.TIER_VERY_LIKELY_WORKS
    assert cal.tier_for_call(None, "likely_inactive") is None
    tiers = cal.prob_tier_for_calls([0.95, 0.95, np.nan], ["likely_inactive", None, "likely_active"])
    assert tiers.tolist() == [cal.TIER_UNCERTAIN, cal.TIER_VERY_LIKELY_WORKS, None]


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


# --------------------------------------------------------------------------- #
# Species-pooled map: per-pair choice inside the training folds
# --------------------------------------------------------------------------- #


def _rows(n: int, slope: float, shift: float, seed: int, n_marker: int = 0) -> cal.CalRows:
    rng = np.random.default_rng(seed)
    pred = 2.0 ** rng.integers(-3, 6, size=n).astype(float)
    d = np.log2(pred) - 1.0
    works = (rng.random(n) < 1 / (1 + np.exp(slope * (d - shift)))).astype(float)
    marker = np.zeros(n, dtype=bool)
    marker[:n_marker] = True
    return cal.CalRows(pred_mic=pred, works=works, marker=marker, folds=(np.arange(n) % 5).astype(float),
                       s_breakpoint=2.0, natural_resistance=False)


def test_blend_maps_is_the_weighted_mean_and_non_increasing() -> None:
    a = cal.ProbMap(x=(-2.0, 0.0, 2.0), y=(1.0, 0.6, 0.0), n=10, n_works=5)
    b = cal.ProbMap(x=(-1.0, 1.0), y=(0.9, 0.1), n=100, n_works=50)
    m = cal.blend_maps(a, b, 0.25)
    grid = np.linspace(-5, 5, 41)
    assert np.allclose(m.predict(grid), 0.25 * a.predict(grid) + 0.75 * b.predict(grid))
    assert (np.diff(m.predict(grid)) <= 1e-12).all()
    assert cal.blend_maps(None, b, 0.5) == b and cal.blend_maps(a, None, 0.5) == a


def test_small_pair_borrows_the_species_map_and_large_distinct_pair_keeps_its_own() -> None:
    pool = {
        "small": _rows(40, slope=2.0, shift=0.0, seed=11),
        "big1": _rows(3000, slope=2.0, shift=0.0, seed=12),
        "big2": _rows(3000, slope=2.0, shift=0.0, seed=13),
    }
    fits = cal.species_fit(pool)
    assert fits["small"][0].main_source in ("species", "blend")
    odd_pool = {
        "odd": _rows(4000, slope=2.0, shift=3.0, seed=14),  # its own curve, far from the species pool
        "big1": _rows(3000, slope=2.0, shift=0.0, seed=12),
        "big2": _rows(3000, slope=2.0, shift=0.0, seed=13),
    }
    assert cal.species_fit(odd_pool)["odd"][0].main_source == "pair"
    rec = fits["small"][1]
    assert set(rec["brier"]) == set(cal.MAIN_SOURCES)
    probs = cal.species_cross_fit(pool)
    for key, rows in pool.items():
        p, by_fold = probs[key]
        assert p.shape == rows.pred_mic.shape and not np.isnan(p).any()
        assert set(by_fold) == {0, 1, 2, 3, 4} and all(v["main_source"] in cal.MAIN_SOURCES for v in by_fold.values())


def test_species_cross_fit_never_uses_the_held_out_fold() -> None:
    pool = {"a": _rows(200, 2.0, 0.0, 21, n_marker=20), "b": _rows(800, 2.0, 0.5, 22, n_marker=30)}
    base = cal.species_cross_fit(pool)
    flipped = {}
    for key, rows in pool.items():
        w = rows.works.copy()
        held = rows.folds == 0
        w[held] = 1.0 - w[held]
        flipped[key] = cal.CalRows(rows.pred_mic, w, rows.marker, rows.folds, rows.s_breakpoint, rows.natural_resistance)
    again = cal.species_cross_fit(flipped)
    for key, rows in pool.items():
        held = rows.folds == 0
        assert np.array_equal(base[key][0][held], again[key][0][held]), key
        assert not np.array_equal(base[key][0][~held], again[key][0][~held]), key


def test_species_fit_natural_resistance_and_no_breakpoint() -> None:
    nat = _rows(100, 2.0, 0.0, 31)
    nat = cal.CalRows(nat.pred_mic, nat.works, nat.marker, nat.folds, 2.0, True)
    nobp = _rows(100, 2.0, 0.0, 32)
    nobp = cal.CalRows(nobp.pred_mic, nobp.works, nobp.marker, nobp.folds, None, False)
    pool = {"nat": nat, "nobp": nobp, "ok": _rows(300, 2.0, 0.0, 33)}
    fits = cal.species_fit(pool)
    assert fits["nat"][0].natural_resistance and (fits["nat"][0].predict(nat.pred_mic) == 0.0).all()
    assert np.isnan(fits["nobp"][0].predict(nobp.pred_mic)).all()
    probs = cal.species_cross_fit(pool)
    assert (probs["nat"][0] == 0.0).all() and np.isnan(probs["nobp"][0]).all()


def test_json_round_trip_keeps_the_main_source(tmp_path) -> None:  # noqa: ANN001
    pool = {"x": _rows(60, 2.0, 0.0, 41, n_marker=5), "y": _rows(2000, 2.0, 0.0, 42)}
    fit, record = cal.species_fit(pool)["x"]
    path = tmp_path / cal.CALIBRATION_FILE
    fit.save(path)
    back = cal.PairCalibration.load(path)
    assert back.main_source == fit.main_source and back.blend_weight == fit.blend_weight
    assert np.allclose(back.predict(pool["x"].pred_mic, pool["x"].marker), fit.predict(pool["x"].pred_mic, pool["x"].marker))
