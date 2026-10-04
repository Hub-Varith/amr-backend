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
