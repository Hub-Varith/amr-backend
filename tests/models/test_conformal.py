"""Tests for genome2mic.models.conformal: residuals, the corrected quantile and bands."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from genome2mic.droplog import DropLog
from genome2mic.models.conformal import (
    DEFAULT_ALPHA,
    band,
    conformal_q,
    read_conformal,
    residual_steps,
    write_conformal,
)

INF = math.inf


def _is_power_of_two(x: float) -> bool:
    return x > 0 and math.isfinite(x) and math.log2(x) == int(math.log2(x))


# -------------------------------------------------------------- residual_steps
class TestResidualSteps:
    def test_exact_rows_only_and_drop_is_logged(self) -> None:
        pred = np.array([3.0, 2.0, 0.0, 5.0, -2.0])
        lo = np.array([4.0, 0.0, 0.5, 32.0, 0.125])
        hi = np.array([8.0, 0.25, 1.0, INF, 0.25])  # rows 1 (left) and 3 (right) are censored
        log = DropLog("conformal_test")
        resid = residual_steps(pred, lo, hi, droplog=log)
        np.testing.assert_array_equal(resid, [0.0, 0.0, 0.0])
        assert log.records[0].n_dropped == 2 and "censored" in log.records[0].reason
        assert log.records[1].n_dropped == 0

    def test_absolute_value_in_steps(self) -> None:
        resid = residual_steps(np.array([1.0, 5.0]), np.array([4.0, 4.0]), np.array([8.0, 8.0]))
        np.testing.assert_array_equal(resid, [2.0, 2.0])

    def test_missing_predictions_are_skipped_and_counted(self) -> None:
        log = DropLog("conformal_test")
        resid = residual_steps(np.array([np.nan, 3.0]), np.array([4.0, 4.0]), np.array([8.0, 8.0]), droplog=log)
        np.testing.assert_array_equal(resid, [0.0])
        assert log.records[1].n_dropped == 1

    def test_uses_a_local_droplog_when_none_given(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="genome2mic.droplog"):
            residual_steps(np.array([3.0, 3.0]), np.array([4.0, 0.0]), np.array([8.0, 1.0]))
        assert any("[conformal] dropped 1 rows" in r.getMessage() for r in caplog.records)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            residual_steps(np.array([3.0]), np.array([4.0, 4.0]), np.array([8.0, 8.0]))

    def test_exact_rows_mask_excludes_one_step_rows_that_are_not_mics(self) -> None:
        """A disk I-only (1, 2] row looks exact by its interval but is not a measured MIC."""
        pred = np.array([3.0, 1.0, 1.0, 4.0])
        lo = np.array([4.0, 1.0, 1.0, 4.0])
        hi = np.array([8.0, 2.0, 2.0, 8.0])
        measured = np.array([True, False, True, True])  # row 1: disk diffusion
        log = DropLog("conformal_test")
        resid = residual_steps(pred, lo, hi, droplog=log, exact_rows=measured)
        np.testing.assert_array_equal(resid, [0.0, 0.0, 1.0])
        reasons = {r.reason: r.n_dropped for r in log.records}
        assert [n for reason, n in reasons.items() if "not a measured MIC" in reason] == [1]
        np.testing.assert_array_equal(residual_steps(pred, lo, hi), [0.0, 0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="exact_rows"):
            residual_steps(pred, lo, hi, exact_rows=measured[:2])


# ----------------------------------------------------------------- conformal_q
class TestConformalQ:
    def test_matches_hand_computed_quantile_at_alpha_0_1(self) -> None:
        # n = 10: k = ceil(11 * 0.9) = ceil(9.9) = 10 -> the largest residual.
        residuals = np.array([0, 0, 1, 1, 2, 3, 0, 1, 2, 5], dtype=float)
        assert conformal_q(residuals, alpha=0.10) == 5.0
        # n = 19: k = ceil(20 * 0.9) = 18 -> the 18th smallest (index 17).
        residuals = np.array([0] * 10 + [1] * 5 + [2] * 2 + [3, 7], dtype=float)
        assert sorted(residuals)[17] == 3.0
        assert conformal_q(residuals, alpha=0.10) == 3.0
        # n = 99: k = ceil(100 * 0.9) = 90 -> index 89 of the sorted array.
        rng = np.random.default_rng(0)
        residuals = rng.integers(0, 6, size=99).astype(float)
        assert conformal_q(residuals) == float(np.sort(residuals)[89])

    def test_default_alpha_is_ten_percent(self) -> None:
        assert DEFAULT_ALPHA == 0.10
        residuals = np.arange(20, dtype=float)
        # k = ceil(21 * 0.9) = 19 -> index 18 -> 18.0
        assert conformal_q(residuals) == 18.0

    def test_other_alphas(self) -> None:
        residuals = np.arange(1, 11, dtype=float)  # 1..10
        assert conformal_q(residuals, alpha=0.5) == 6.0  # k = ceil(11 * 0.5) = 6
        assert conformal_q(residuals, alpha=0.25) == 9.0  # k = ceil(11 * 0.75) = 9

    def test_floor_at_zero_and_order_independence(self) -> None:
        assert conformal_q(np.zeros(50)) == 0.0
        residuals = np.array([3.0, 0.0, 2.0, 1.0, 5.0, 1.0, 0.0, 2.0, 1.0, 0.0, 4.0, 2.0])
        assert conformal_q(residuals) == conformal_q(np.sort(residuals)[::-1])

    def test_too_few_residuals_gives_inf_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="genome2mic.models.conformal"):
            q = conformal_q(np.array([0.0, 1.0, 1.0, 2.0, 0.0]), alpha=0.10)  # k = 6 > 5
        assert q == INF
        assert any("cannot certify" in r.getMessage() for r in caplog.records)
        # n = 9 is exactly enough: k = ceil(10 * 0.9) = 9
        assert conformal_q(np.arange(9, dtype=float)) == 8.0

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            conformal_q(np.array([]))
        with pytest.raises(ValueError):
            conformal_q(np.array([np.nan, 1.0]))
        with pytest.raises(ValueError):
            conformal_q(np.array([-1.0, 1.0]))
        with pytest.raises(ValueError):
            conformal_q(np.array([1.0]), alpha=0.0)
        with pytest.raises(ValueError):
            conformal_q(np.array([1.0]), alpha=1.0)


# ------------------------------------------------------------------------ band
class TestBand:
    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    def test_bands_sit_on_the_grid_and_contain_the_prediction(self, q: float) -> None:
        pred = np.array([0.0625, 0.25, 1.0, 2.0, 8.0, 64.0, 6.0, 0.19])
        low, high = band(pred, q)
        assert low.shape == high.shape == pred.shape
        assert all(_is_power_of_two(v) for v in low)
        assert all(_is_power_of_two(v) for v in high)
        assert np.all(low <= pred) and np.all(pred <= high)
        assert np.all(np.log2(high) - np.log2(low) >= 2 * q - 1e-9)

    def test_known_values(self) -> None:
        low, high = band(np.array([8.0]), 1.0)
        assert low.tolist() == [4.0] and high.tolist() == [16.0]
        low, high = band(np.array([8.0]), 1.5)  # 8/2.83 = 2.83 -> 2 ; 8*2.83 = 22.6 -> 32
        assert low.tolist() == [2.0] and high.tolist() == [32.0]
        low, high = band(np.array([6.0]), 1.0)  # off-grid prediction: 3 -> 2 ; 12 -> 16
        assert low.tolist() == [2.0] and high.tolist() == [16.0]
        low, high = band(np.array([8.0]), 0.0)
        assert low.tolist() == [8.0] and high.tolist() == [8.0]

    def test_low_rounds_down_and_high_rounds_up(self) -> None:
        low, high = band(np.array([1.0]), 0.5)  # 0.707 -> 0.5 ; 1.414 -> 2
        assert low.tolist() == [0.5] and high.tolist() == [2.0]

    def test_nan_prediction_stays_missing(self) -> None:
        low, high = band(np.array([np.nan, 4.0]), 1.0)
        assert math.isnan(low[0]) and math.isnan(high[0])
        assert low[1] == 2.0 and high[1] == 8.0

    def test_infinite_q_gives_uninformative_band(self) -> None:
        low, high = band(np.array([4.0, np.nan]), INF)
        assert low[0] == 0.0 and high[0] == INF
        assert math.isnan(low[1]) and math.isnan(high[1])

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            band(np.array([0.0]), 1.0)
        with pytest.raises(ValueError):
            band(np.array([4.0]), -1.0)
        with pytest.raises(ValueError):
            band(np.array([4.0]), math.nan)

    def test_outputs_are_float_arrays(self) -> None:
        low, high = band(np.array([2.0]), 1.0)
        assert low.dtype == np.float64 and high.dtype == np.float64


# ------------------------------------------------------------------ persistence
class TestConformalFile:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "conformal.json"
        write_conformal(path, q=1.0, alpha=0.1, n_residuals=123, extra={"model": "aft_known_unitig"})
        payload = read_conformal(path)
        assert payload["q"] == 1.0 and payload["alpha"] == 0.1
        assert payload["n_residuals"] == 123 and payload["model"] == "aft_known_unitig"

    def test_inf_q_survives_json(self, tmp_path: Path) -> None:
        path = tmp_path / "conformal.json"
        write_conformal(path, q=INF, alpha=0.1, n_residuals=3)
        assert read_conformal(path)["q"] == INF


# ------------------------------------------------------- end-to-end sanity check
def test_residuals_to_band_pipeline_has_nominal_coverage() -> None:
    """On the calibration rows themselves the q-step band covers >= 90% of exact labels."""
    rng = np.random.default_rng(1)
    n = 300
    truth_step = rng.integers(-4, 6, size=n).astype(float)
    pred_log2 = truth_step + rng.normal(0, 0.8, size=n)
    pred_mic = 2.0 ** np.ceil(pred_log2)  # rounded up to the grid
    lo, hi = 2.0 ** (truth_step - 1), 2.0**truth_step
    resid = residual_steps(np.log2(pred_mic), lo, hi)
    q = conformal_q(resid, alpha=0.10)
    low, high = band(pred_mic, q)
    covered = (low <= hi) & (hi <= high)
    assert covered.mean() >= 0.90


# -------------------------------------------------------------- tuned asymmetric bands (L5)
from genome2mic.models.conformal import (  # noqa: E402
    BandParams,
    asym_band,
    asym_quantiles,
    gate_calls,
    level_quantile,
    passing_levels,
    robust_q_low,
    signed_residual_steps,
    symmetric_params,
    tune_band,
    vme_certified,
)


class TestAsymmetricBands:
    def test_signed_residuals_keep_sign_and_use_exact_rows_only(self) -> None:
        pred = np.log2([4.0, 4.0, 4.0, 4.0])
        lo = np.array([8.0, 1.0, 0.0, 2.0])
        hi = np.array([16.0, 2.0, 2.0, 4.0])
        exact = np.array([True, True, True, False])
        r = signed_residual_steps(pred, lo, hi, exact_rows=exact)
        # row 2 is censored (lo == 0), row 3 not measured
        assert list(r) == [2.0, -1.0]

    def test_level_quantile_is_finite_sample_and_inf_when_too_few(self) -> None:
        v = np.arange(1, 10, dtype=float)  # n = 9
        assert level_quantile(v, 0.9) == 9.0  # ceil(10 * 0.9) = 9th smallest
        assert level_quantile(v, 0.5) == 5.0
        assert math.isinf(level_quantile(v, 0.95))  # ceil(10 * 0.95) = 10 > 9
        assert level_quantile(np.array([-3.0, -1.0, -2.0]), 0.5) == -2.0

    def test_asym_quantiles_floor_at_zero_and_flag_uncertified_upper(self) -> None:
        r = -np.ones(50)  # the model always over-predicts by one step
        q_up, q_low, certified = asym_quantiles(r, 0.05, 0.05)
        assert q_up == 0.0 and q_low == 1.0 and certified
        q_up, q_low, certified = asym_quantiles(np.array([0.0, 1.0, 3.0]), 0.005, 0.05)
        assert not certified and q_up == max(2.0, 3.0) and math.isfinite(q_low)
        assert asym_quantiles(np.array([]), 0.05, 0.05) == (2.0, 2.0, False)

    def test_asym_band_matches_symmetric_band_when_equal(self) -> None:
        pred = np.array([0.25, 1.0, 8.0, 64.0])
        for q in (0.0, 1.0, 2.0, 3.0):
            low, high = asym_band(pred, q, q)
            low2, high2 = band(pred, q)
            assert np.array_equal(low, low2) and np.array_equal(high, high2)
        low, high = asym_band(np.array([4.0]), 3.0, 1.0)
        assert (low[0], high[0]) == (2.0, 32.0)
        with pytest.raises(ValueError):
            asym_band(pred, INF, 1.0)

    def test_vme_ucb_rule_needs_enough_resistant_isolates(self) -> None:
        assert not vme_certified(0, 0)
        assert not vme_certified(0, 65)  # 1/66 > 1.5 %
        assert vme_certified(0, 66)  # 1/67 <= 1.5 %
        assert vme_certified(2, 199) and not vme_certified(3, 199)

    def test_gate_calls_only_touches_active(self) -> None:
        calls = np.array(["likely_active", "uncertain", "likely_inactive", None], dtype=object)
        assert list(gate_calls(calls, True)) == list(calls)
        assert list(gate_calls(calls, False)) == ["uncertain", "uncertain", "likely_inactive", None]

    def test_band_params_round_trip_and_symmetric(self) -> None:
        params = BandParams(q_up=3.0, q_low=1.0, alpha_up=0.025, alpha_low=0.05, active_gate_open=False, n_residuals=10)
        payload = params.as_json()
        assert payload["q_up"] == 3.0 and payload["q_low"] == 1.0 and payload["active_gate_open"] is False
        sym = symmetric_params(2.0, 0.1, 5)
        assert sym.q_up == sym.q_low == 2.0 and sym.active_gate_open and sym.kind == "symmetric"


def _tuning_case(n_per_fold: int = 120, seed: int = 0):
    """5 folds; lab R rows have MIC 16, lab S rows MIC 0.0625; the model predicts every S exactly and 10 % of R as 1.

    A band whose upper end clears the mispredicted R rows (4 steps) still calls the S rows
    likely active (0.0625 * 16 = 1 <= S breakpoint 2), so the safe level issues calls.
    """
    rng = np.random.default_rng(seed)
    n = 5 * n_per_fold
    folds = np.repeat(np.arange(5, dtype=float), n_per_fold)
    is_r = rng.random(n) < 0.5
    lab_mic = np.where(is_r, 16.0, 0.0625)
    pred = np.where(is_r, np.where(rng.random(n) < 0.9, 16.0, 1.0), 0.0625)
    lab = np.where(is_r, "R", "S").astype(object)
    return np.log2(pred), lab_mic / 2, lab_mic, np.ones(n, bool), lab, folds


def _calls(bp_s: float = 2.0, bp_r: float = 4.0):
    def fn(idx, low, high):
        out = np.full(len(idx), "uncertain", dtype=object)
        out[high <= bp_s] = "likely_active"
        out[(high > bp_s) & (low > bp_r)] = "likely_inactive"
        return out
    return fn


class TestTuneBand:
    def test_prefers_the_widest_alpha_that_certifies_vme(self) -> None:
        p, lo, hi, exact, lab, folds = _tuning_case()
        grid = (0.2, 0.08, 0.05, 0.025, 0.01, 0.005)
        a, gate, ucb = tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls(), grid=grid)
        # 10 % of R are predicted 1: only a band whose upper end clears them (>= 4 steps) is safe.
        assert gate and ucb is not None and ucb <= 0.015
        assert a < 0.2  # the narrow 80 % upper end lets the mispredicted R rows through
        for wider in grid[: grid.index(a)]:  # every narrower band (larger alpha) failed
            assert tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls(), grid=(wider,))[1] is False

    def test_gate_closes_without_enough_resistant_isolates(self) -> None:
        p, lo, hi, exact, lab, folds = _tuning_case(n_per_fold=20)
        lab = np.where(np.arange(len(lab)) % 50 == 0, "R", "S").astype(object)  # 2 R rows
        a, gate, ucb = tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls())
        assert not gate and ucb is None and a == 0.005

    def test_held_out_fold_never_influences_the_choice(self) -> None:
        p, lo, hi, exact, lab, folds = _tuning_case()
        base = tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls())
        p2, lab2, hi2 = p.copy(), lab.copy(), hi.copy()
        held = folds == 4
        p2[held] = -5.0
        lab2[held] = "R"
        hi2[held] = 512.0
        assert tune_band(p2, lo, hi2, exact, lab2, folds, [0, 1, 2, 3], _calls()) == base

    def test_uncallable_pair_skips_tuning(self) -> None:
        p, lo, hi, exact, lab, folds = _tuning_case()
        assert tune_band(p, lo, hi, exact, lab, folds, [0, 1], _calls(), callable_pair=False) == (0.05, True, None)

    def test_deterministic(self) -> None:
        args = _tuning_case(seed=3)
        assert tune_band(*args, [0, 1, 2, 3, 4], _calls()) == tune_band(*args, [0, 1, 2, 3, 4], _calls())

    def test_level_that_issues_no_active_call_cannot_open_the_gate(self) -> None:
        # Every S row sits at the breakpoint: any band with q_up > 0 issues no active call.
        p, lo, hi, exact, lab, folds = _tuning_case()
        a, gate, ucb = tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], lambda idx, low, high: np.full(len(idx), "uncertain", dtype=object))
        assert not gate and ucb is None


def _fixed_calls_case(n_r: int = 100, n_s: int = 100):
    """4 folds x (n_r R + n_s S) rows, all exact and perfectly predicted (bands do not matter here)."""
    per = n_r + n_s
    folds = np.repeat(np.arange(4, dtype=float), per)
    lab = np.tile(np.array(["R"] * n_r + ["S"] * n_s, dtype=object), 4)
    mic = np.where(lab == "R", 16.0, 0.5)
    return np.log2(mic), mic / 2, mic, np.ones(len(mic), bool), lab, folds


def _calls_by_fold(lab: np.ndarray, folds: np.ndarray, vme_by_fold: dict[int, int]):
    """Active calls for every S row of the folds in ``vme_by_fold`` plus that many R rows; others uncertain."""
    def fn(idx, low, high):
        out = np.full(len(idx), "uncertain", dtype=object)
        for f, k in vme_by_fold.items():
            in_f = folds[idx] == f
            out[in_f & (lab[idx] == "S")] = "likely_active"
            r_rows = np.where(in_f & (lab[idx] == "R"))[0][:k]
            out[r_rows] = "likely_active"
        return out
    return fn


class TestCallingFoldRule:
    def test_folds_without_active_calls_do_not_dilute_vme(self) -> None:
        p, lo, hi, exact, lab, folds = _fixed_calls_case()
        # Fold 0 alone calls, with 1 VME among its 100 lab-R: (1 + 1) / (100 + 1) = 2 % fails.
        # The old pooled rule over all four folds, (1 + 1) / (400 + 1) = 0.5 %, would have passed.
        a, gate, ucb = tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls_by_fold(lab, folds, {0: 1}))
        assert not gate and ucb is None

    def test_strict_folds_makes_every_calling_fold_pass_on_its_own(self) -> None:
        p, lo, hi, exact, lab, folds = _fixed_calls_case()
        # All four folds call; pooled (2 + 1) / (400 + 1) = 0.75 % passes, but fold 0 has 2 / 100 = 2 % VME.
        fn = _calls_by_fold(lab, folds, {0: 2, 1: 0, 2: 0, 3: 0})
        assert tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], fn)[1] is True  # default: pooled calling folds only
        assert tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], fn, strict_folds=True)[1] is False
        fn_ok = _calls_by_fold(lab, folds, {0: 1, 1: 0, 2: 0, 3: 0})  # 1 % in fold 0, pooled 2 / 401
        a, gate, ucb = tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], fn_ok, strict_folds=True)
        assert gate and ucb == pytest.approx(2 / 401)

    def test_empty_grid_closes_the_gate(self) -> None:
        p, lo, hi, exact, lab, folds = _fixed_calls_case()
        assert tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls_by_fold(lab, folds, {0: 0}), grid=())[1] is False


class TestRobustLowerEnd:
    def test_unchanged_when_folds_agree(self) -> None:
        folds = [-np.r_[np.zeros(95), np.ones(5)] for _ in range(4)]
        q_low, widened = robust_q_low(folds, 1.0, 0.05, 0.10)
        assert (q_low, widened) == (1.0, False)

    def test_widened_when_one_fold_over_predicts(self) -> None:
        calm = [np.zeros(100) for _ in range(3)]
        heavy = [-np.r_[np.zeros(60), np.full(40, 4.0)]]  # model over-predicts this fold by 4 steps
        parts = calm + heavy
        pooled = asym_quantiles(np.concatenate(parts), 0.5, 0.05)[1]
        q_low, widened = robust_q_low(parts, pooled, 0.05, 0.10)
        lofo = [asym_quantiles(np.concatenate([r for j, r in enumerate(parts) if j != i]), 0.5, 0.05)[1] for i in range(4)]
        assert max(lofo) - min(lofo) > 1 and widened
        assert q_low >= max(lofo) and q_low >= pooled and q_low == 4.0

    def test_single_fold_is_left_alone(self) -> None:
        assert robust_q_low([np.zeros(10)], 0.0, 0.05, 0.10) == (0.0, False)
        assert robust_q_low([np.zeros(10), np.array([])], 0.0, 0.05, 0.10) == (0.0, False)

    def test_passing_levels_lists_every_passing_level_in_grid_order(self) -> None:
        p, lo, hi, exact, lab, folds = _tuning_case()
        grid = (0.2, 0.08, 0.05, 0.025, 0.01, 0.005)
        levels = passing_levels(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls(), grid=grid)
        a, gate, ucb = tune_band(p, lo, hi, exact, lab, folds, [0, 1, 2, 3], _calls(), grid=grid)
        assert levels and gate and levels[0] == (a, ucb)
        assert [x for x, _ in levels] == [x for x in grid if x in {y for y, _ in levels}]


def test_calling_fold_vme_ignores_folds_without_active_calls() -> None:
    from genome2mic.models.conformal import calling_fold_vme  # noqa: PLC0415

    calls = np.array(["likely_active", "uncertain", "likely_active", "uncertain", "uncertain", None], dtype=object)
    lab = np.array(["R", "R", "S", "R", "R", "R"], dtype=object)
    folds = np.array([0, 0, 0, 1, 1, np.nan])
    assert calling_fold_vme(calls, lab, folds) == (1, 2, 1)  # fold 1 (2 lab R, no active call) is left out
    assert calling_fold_vme(np.array(["uncertain"], dtype=object), np.array(["R"], dtype=object), np.array([0.0])) == (0, 0, 0)


# --------------------------------------------------------------- per-fold safety (v0.6)

def test_fold_call_vme_table_flags_a_significant_calling_fold() -> None:
    from genome2mic.models.conformal import binomial_excess_p, fold_call_vme_table  # noqa: PLC0415

    # Fold 0: 1000 lab R, no VME. Fold 1: 100 lab R, 7 called likely_active. Fold 2: no active call.
    calls = np.array(["uncertain"] * 1000 + ["likely_active"] * 7 + ["uncertain"] * 93 + ["uncertain"] * 50 + ["likely_active"],
                     dtype=object)
    lab = np.array(["R"] * 1100 + ["R"] * 50 + ["S"], dtype=object)
    folds = np.array([0.0] * 1000 + [1.0] * 100 + [2.0] * 50 + [0.0])
    table = {r["fold"]: r for r in fold_call_vme_table(calls, lab, folds, 0.015)}
    assert table[0]["calling"] and table[0]["n_vme"] == 0 and table[0]["p_value"] == 1.0 and not table[0]["significant"]
    assert table[1]["n_vme"] == 7 and table[1]["n_lab_r"] == 100
    assert table[1]["p_value"] == pytest.approx(binomial_excess_p(7, 100, 0.015)) and table[1]["significant"]
    assert not table[2]["calling"] and table[2]["call_vme"] == 0.0 and not table[2]["significant"]


def test_robust_q_up_widens_only_for_a_significantly_missed_fold() -> None:
    from genome2mic.models.conformal import robust_q_up  # noqa: PLC0415

    rng = np.random.default_rng(5)
    stable = [rng.integers(-2, 2, size=200).astype(float) for _ in range(3)]
    q, widened = robust_q_up(stable, 1.0, 0.05)
    assert (q, widened) == (1.0, False)
    # One fold whose lab MICs sit 3 steps above the predictions: 70 % of it misses q_up = 1.
    shifted = np.concatenate([np.full(140, 3.0), np.zeros(60)])
    q, widened = robust_q_up([*stable, shifted], 1.0, 0.05)
    assert widened and q == 3.0
    # Nothing to check with no residuals.
    assert robust_q_up([np.array([])], 1.0, 0.05) == (1.0, False)


def test_band_params_json_records_the_new_keys() -> None:
    from genome2mic.models.conformal import BandParams  # noqa: PLC0415

    params = BandParams(q_up=1.0, q_low=1.0, alpha_up=0.05, alpha_low=0.05, active_gate_open=False, n_residuals=3,
                        q_up_widened=True, fold_call_vme=({"fold": 0, "n_vme": 1},), fold_gate_closed=True)
    out = params.as_json()
    assert out["q_up_widened"] is True and out["fold_gate_closed"] is True
    assert out["fold_call_vme"] == [{"fold": 0, "n_vme": 1}]
