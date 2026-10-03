"""Tests for genome2mic.mic: the doubling grid and the contract's interval rule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from genome2mic import mic
from genome2mic.mic import (
    DOUBLING_GRID,
    censor_of,
    interval_from_result,
    interval_from_sir,
    label_point_log2,
    log2_step,
    normalize_sign,
    normalize_sir,
    round_down_to_step,
    round_down_to_step_array,
    round_to_nearest_step,
    round_up_to_step,
    round_up_to_step_array,
    steps_between,
)

INF = math.inf


@dataclass(frozen=True)
class FakeBreakpoint:
    """Duck-typed stand-in for config.Breakpoint (only the two attributes mic.py reads)."""

    s_breakpoint: float
    r_breakpoint: float


def _is_power_of_two(x: float) -> bool:
    return x > 0 and math.isfinite(x) and math.log2(x) == int(math.log2(x))


# ----------------------------------------------------------------------- grid
class TestDoublingGrid:
    def test_span_and_size(self) -> None:
        assert len(DOUBLING_GRID) == 23
        assert DOUBLING_GRID[0] == 2.0**-10 == pytest.approx(0.00098, abs=1e-5)
        assert DOUBLING_GRID[-1] == 4096.0

    def test_every_entry_is_an_exact_power_of_two_and_sorted(self) -> None:
        assert list(DOUBLING_GRID) == sorted(DOUBLING_GRID)
        for value in DOUBLING_GRID:
            assert _is_power_of_two(value)
        assert 1.0 in DOUBLING_GRID and 0.25 in DOUBLING_GRID and 8.0 in DOUBLING_GRID


# ------------------------------------------------------------------ log2_step
class TestLog2Step:
    @pytest.mark.parametrize("mic_value, expected", [(8, 3.0), (0.25, -2.0), (1, 0.0), (0.0625, -4.0)])
    def test_exact(self, mic_value: float, expected: float) -> None:
        assert log2_step(mic_value) == expected

    def test_inf_passes_through(self) -> None:
        assert log2_step(INF) == INF

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), None, "abc"])
    def test_rejects_non_positive_or_missing(self, bad: object) -> None:
        with pytest.raises(ValueError):
            log2_step(bad)  # type: ignore[arg-type]


# ------------------------------------------------------------- rounding scalar
class TestRounding:
    @pytest.mark.parametrize("value", DOUBLING_GRID)
    def test_exact_powers_unchanged_by_every_rounding(self, value: float) -> None:
        assert round_up_to_step(value) == value
        assert round_down_to_step(value) == value
        assert round_to_nearest_step(value) == value

    @pytest.mark.parametrize(
        "value, expected",
        [(0.19, 0.25), (6, 8.0), (0.1, 0.125), (3, 4.0), (1.5, 2.0), (12, 16.0), (0.03, 0.03125), (5000, 8192.0)],
    )
    def test_round_up(self, value: float, expected: float) -> None:
        assert round_up_to_step(value) == expected

    @pytest.mark.parametrize("value, expected", [(6, 4.0), (0.19, 0.125), (3, 2.0), (1.5, 1.0), (12, 8.0)])
    def test_round_down(self, value: float, expected: float) -> None:
        assert round_down_to_step(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [(3, 4.0), (2.5, 2.0), (6, 8.0), (0.19, 0.25), (0.17, 0.125), (2**2.5, 8.0), (2**-1.5, 0.5)],
    )
    def test_round_nearest_with_ties_going_up(self, value: float, expected: float) -> None:
        assert round_to_nearest_step(value) == expected

    def test_float_drift_does_not_bump_a_grid_value(self) -> None:
        drifted_up = 2.0**3.0000000000000004  # 8.000000000000002
        drifted_down = 7.999999999999999
        assert round_up_to_step(drifted_up) == 8.0
        assert round_down_to_step(drifted_down) == 8.0
        assert round_to_nearest_step(drifted_up) == 8.0
        # 2**pred_log2 round trips for integer predictions
        for k in range(-10, 13):
            assert round_up_to_step(2.0 ** float(k)) == 2.0**k

    def test_genuinely_off_grid_values_are_not_forgiven(self) -> None:
        assert round_up_to_step(8.001) == 16.0
        assert round_down_to_step(7.99) == 4.0

    def test_inf(self) -> None:
        assert round_up_to_step(INF) == INF
        assert round_down_to_step(INF) == INF
        assert round_to_nearest_step(INF) == INF

    @pytest.mark.parametrize("bad", [0, -2, float("nan"), None])
    def test_rejects_non_positive(self, bad: object) -> None:
        for fn in (round_up_to_step, round_down_to_step, round_to_nearest_step):
            with pytest.raises(ValueError):
                fn(bad)  # type: ignore[arg-type]

    def test_results_are_exact_powers_of_two(self) -> None:
        rng = np.random.default_rng(0)
        for value in rng.uniform(0.001, 5000, size=200):
            assert _is_power_of_two(round_up_to_step(value))
            assert _is_power_of_two(round_down_to_step(value))
            assert round_down_to_step(value) <= value <= round_up_to_step(value)


class TestRoundingArrays:
    def test_round_up_array_matches_scalar_and_keeps_nan_and_inf(self) -> None:
        x = np.array([0.19, 8.0, 6.0, np.nan, np.inf, 2.0**3.0000000000000004])
        out = round_up_to_step_array(x)
        assert out[0] == 0.25 and out[1] == 8.0 and out[2] == 8.0 and out[5] == 8.0
        assert np.isnan(out[3]) and out[4] == np.inf
        assert out.dtype == np.float64

    def test_round_down_array(self) -> None:
        out = round_down_to_step_array([6.0, 0.19, 0.25])
        assert out.tolist() == [4.0, 0.125, 0.25]

    def test_empty_and_all_nan(self) -> None:
        assert round_up_to_step_array(np.array([])).shape == (0,)
        assert np.isnan(round_up_to_step_array(np.array([np.nan]))).all()

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            round_up_to_step_array(np.array([1.0, 0.0]))


# -------------------------------------------------------------- steps_between
class TestStepsBetween:
    @pytest.mark.parametrize("a, b, expected", [(1, 8, 3.0), (8, 1, -3.0), (0.25, 1, 2.0), (4, 4, 0.0)])
    def test_values(self, a: float, b: float, expected: float) -> None:
        assert steps_between(a, b) == expected

    def test_inf(self) -> None:
        assert steps_between(4, INF) == INF
        assert steps_between(INF, 4) == -INF

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            steps_between(0, 4)


# ------------------------------------------------------------ sign/sir parsing
class TestNormalize:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, "="), ("", "="), ("  ", "="), ("=", "="), ("==", "="), (float("nan"), "="),
            ("<=", "<="), (" <= ", "<="), ("=<", "<="), ("≤", "<="),
            ("<", "<"), (">", ">"), (">=", ">="), ("=>", ">="), ("≥", ">="),
        ],
    )
    def test_sign_aliases(self, raw: object, expected: str) -> None:
        assert normalize_sign(raw) == expected  # type: ignore[arg-type]

    def test_pandas_na_means_no_sign(self) -> None:
        pd = pytest.importorskip("pandas")
        assert normalize_sign(pd.NA) == "="  # type: ignore[arg-type]

    @pytest.mark.parametrize("raw", ["~", "<>", "approx", "8"])
    def test_unknown_sign_raises(self, raw: str) -> None:
        with pytest.raises(ValueError):
            normalize_sign(raw)

    @pytest.mark.parametrize(
        "raw, expected",
        [("S", "S"), ("s", "S"), (" Susceptible ", "S"), ("I", "I"), ("intermediate", "I"), ("R", "R"), ("RESISTANT", "R")],
    )
    def test_sir_aliases(self, raw: str, expected: str) -> None:
        assert normalize_sir(raw) == expected

    @pytest.mark.parametrize("raw", ["SDD", "NS", "", "Non-susceptible", None])
    def test_unknown_sir_raises(self, raw: object) -> None:
        with pytest.raises(ValueError):
            normalize_sir(raw)  # type: ignore[arg-type]


# ------------------------------------------------------- interval_from_result
class TestIntervalFromResult:
    # Every numeric row of the contract's interval table, plus the DESIGN examples.
    @pytest.mark.parametrize(
        "sign, value, expected",
        [
            ("=", 8, (4.0, 8.0, "interval")),
            ("<=", 0.25, (0.0, 0.25, "left")),
            (">", 32, (32.0, INF, "right")),
            (">=", 16, (8.0, INF, "right")),
            ("<", 0.5, (0.0, 0.25, "left")),
            (None, 8, (4.0, 8.0, "interval")),
        ],
    )
    def test_contract_table(self, sign: str | None, value: float, expected: tuple) -> None:
        assert interval_from_result(sign, value) == expected

    @pytest.mark.parametrize(
        "sign, value, expected",
        [
            ("", 8, (4.0, 8.0, "interval")),  # blank sign == "="
            ("==", 2, (1.0, 2.0, "interval")),
            ("≤", 1, (0.0, 1.0, "left")),
            ("≥", 2, (1.0, INF, "right")),
            ("=", "8", (4.0, 8.0, "interval")),  # numeric string from a CSV
            ("=", 1, (0.5, 1.0, "interval")),
            ("<=", 0.0009765625, (0.0, 2.0**-10, "left")),
            (">", 4096, (4096.0, INF, "right")),
        ],
    )
    def test_sign_variants_and_edges(self, sign: str, value: object, expected: tuple) -> None:
        assert interval_from_result(sign, value) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "sign, value, expected",
        [
            # off-grid values land in the grid cell that contains them
            ("=", 6, (4.0, 8.0, "interval")),
            ("=", 0.19, (0.125, 0.25, "interval")),
            ("=", 1.5, (1.0, 2.0, "interval")),
            ("=", 12, (8.0, 16.0, "interval")),
            # censored bounds widen to the grid, never narrow
            ("<=", 0.19, (0.0, 0.25, "left")),
            ("<=", 6, (0.0, 8.0, "left")),
            ("<", 0.19, (0.0, 0.125, "left")),  # "< X" = at most the step below X
            ("<", 6, (0.0, 4.0, "left")),
            (">", 6, (4.0, INF, "right")),
            (">", 0.19, (0.125, INF, "right")),
            (">=", 6, (4.0, INF, "right")),
            (">=", 0.19, (0.125, INF, "right")),
        ],
    )
    def test_off_grid_values(self, sign: str, value: float, expected: tuple) -> None:
        assert interval_from_result(sign, value) == expected

    def test_off_grid_intervals_contain_the_reported_value(self) -> None:
        for value in (0.19, 0.38, 0.75, 1.5, 3, 6, 12, 24, 0.094, 0.047):
            lo, hi, _ = interval_from_result("=", value)
            assert lo < value <= hi
            lo, hi, _ = interval_from_result("<=", value)
            assert lo == 0.0 and value <= hi
            lo, hi, _ = interval_from_result(">", value)
            assert lo <= value and hi == INF
            lo, hi, _ = interval_from_result(">=", value)
            assert lo < value and hi == INF

    def test_bounds_are_zero_inf_or_exact_powers_of_two(self) -> None:
        for sign in ("=", "<=", "<", ">", ">="):
            for value in (0.19, 0.25, 1, 3, 8, 6, 0.016, 1000):
                lo, hi, censor = interval_from_result(sign, value)
                assert lo < hi
                assert lo == 0.0 or _is_power_of_two(lo)
                assert hi == INF or _is_power_of_two(hi)
                assert censor_of(lo, hi) == censor
                step = label_point_log2(lo, hi)
                assert step == int(step)

    def test_floats_are_python_floats(self) -> None:
        lo, hi, censor = interval_from_result("=", np.float64(8))
        assert type(lo) is float and type(hi) is float and censor == "interval"

    @pytest.mark.parametrize("value", [None, float("nan"), 0, -1, INF, "", "abc"])
    def test_missing_or_invalid_value_raises(self, value: object) -> None:
        with pytest.raises(ValueError):
            interval_from_result("=", value)  # type: ignore[arg-type]

    def test_pandas_na_value_raises(self) -> None:
        pd = pytest.importorskip("pandas")
        with pytest.raises(ValueError):
            interval_from_result("=", pd.NA)  # type: ignore[arg-type]

    def test_unknown_sign_raises(self) -> None:
        with pytest.raises(ValueError):
            interval_from_result("~", 8)


# ---------------------------------------------------------- interval_from_sir
class TestIntervalFromSir:
    BP = FakeBreakpoint(s_breakpoint=1.0, r_breakpoint=2.0)

    def test_contract_rows(self) -> None:
        assert interval_from_sir("S", self.BP) == (0.0, 1.0, "left")
        assert interval_from_sir("R", self.BP) == (2.0, INF, "right")
        assert interval_from_sir("I", self.BP) == (1.0, 2.0, "interval")

    def test_case_and_full_words(self) -> None:
        assert interval_from_sir("s", self.BP) == (0.0, 1.0, "left")
        assert interval_from_sir("Resistant", self.BP) == (2.0, INF, "right")
        assert interval_from_sir(" intermediate ", self.BP) == (1.0, 2.0, "interval")

    def test_i_invalid_when_s_equals_r(self) -> None:
        bp = FakeBreakpoint(s_breakpoint=2.0, r_breakpoint=2.0)
        with pytest.raises(ValueError):
            interval_from_sir("I", bp)
        # S and R are still fine with a collapsed I category
        assert interval_from_sir("S", bp) == (0.0, 2.0, "left")
        assert interval_from_sir("R", bp) == (2.0, INF, "right")

    def test_eucast_meropenem_like_breakpoint(self) -> None:
        bp = FakeBreakpoint(s_breakpoint=2.0, r_breakpoint=8.0)
        assert interval_from_sir("I", bp) == (2.0, 8.0, "interval")
        assert label_point_log2(*interval_from_sir("I", bp)[:2]) == 3.0

    def test_off_grid_breakpoint_is_widened_to_the_grid(self) -> None:
        bp = FakeBreakpoint(s_breakpoint=0.001, r_breakpoint=0.5)
        lo, hi, censor = interval_from_sir("S", bp)
        assert (lo, censor) == (0.0, "left") and hi == 2.0**-9 and hi >= 0.001
        lo, hi, censor = interval_from_sir("I", bp)
        assert (hi, censor) == (0.5, "interval") and lo == 2.0**-10 and lo <= 0.001
        assert interval_from_sir("R", bp) == (0.5, INF, "right")

    @pytest.mark.parametrize("raw", ["SDD", "X", "", None])
    def test_unknown_sir_raises(self, raw: object) -> None:
        with pytest.raises(ValueError):
            interval_from_sir(raw, self.BP)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bp",
        [
            FakeBreakpoint(4.0, 2.0),  # r < s
            FakeBreakpoint(0.0, 2.0),
            FakeBreakpoint(1.0, INF),
            FakeBreakpoint(float("nan"), 2.0),
        ],
    )
    def test_inconsistent_breakpoint_raises(self, bp: FakeBreakpoint) -> None:
        with pytest.raises(ValueError):
            interval_from_sir("S", bp)

    def test_duck_typing_accepts_any_object_with_the_two_attributes(self) -> None:
        class Anything:
            s_breakpoint = 0.25
            r_breakpoint = 0.5

        assert interval_from_sir("I", Anything()) == (0.25, 0.5, "interval")


# ----------------------------------------------------------- label_point_log2
class TestLabelPointLog2:
    @pytest.mark.parametrize(
        "lo, hi, expected",
        [
            (4, 8, 3.0),  # interval -> log2(hi)
            (0, 0.25, -2.0),  # left -> log2(hi)
            (0, 1, 0.0),
            (32, INF, 6.0),  # right -> log2(lo) + 1
            (8, INF, 4.0),
            (0.5, 1.0, 0.0),
        ],
    )
    def test_values(self, lo: float, hi: float, expected: float) -> None:
        assert label_point_log2(lo, hi) == expected

    def test_round_trips_interval_from_result(self) -> None:
        assert label_point_log2(*interval_from_result("=", 8)[:2]) == 3.0
        assert label_point_log2(*interval_from_result("<=", 0.25)[:2]) == -2.0
        assert label_point_log2(*interval_from_result(">", 32)[:2]) == 6.0
        assert label_point_log2(*interval_from_result(">=", 16)[:2]) == 4.0  # ">= 16" reports step 16

    @pytest.mark.parametrize("lo, hi", [(0, INF), (8, 4), (4, 4), (-1, 4), (float("nan"), 4), (None, 4)])
    def test_invalid_intervals_raise(self, lo: object, hi: object) -> None:
        with pytest.raises(ValueError):
            label_point_log2(lo, hi)  # type: ignore[arg-type]


class TestCensorOf:
    def test_values(self) -> None:
        assert censor_of(0, 1) == "left"
        assert censor_of(2, INF) == "right"
        assert censor_of(1, 2) == "interval"

    @pytest.mark.parametrize("lo, hi", [(0, INF), (2, 1), (1, 1), (-1, 2)])
    def test_invalid(self, lo: float, hi: float) -> None:
        with pytest.raises(ValueError):
            censor_of(lo, hi)


def test_module_exports_everything_design_lists() -> None:
    for name in (
        "DOUBLING_GRID", "log2_step", "round_up_to_step", "round_to_nearest_step", "steps_between",
        "interval_from_result", "interval_from_sir", "label_point_log2",
    ):
        assert hasattr(mic, name), name


def test_exact_interval_mask_requires_one_doubling_step():
    lo = np.array([4.0, 2.0, 0.0, 32.0, np.nan, 0.0625])
    hi = np.array([8.0, 8.0, 0.25, np.inf, 8.0, 0.125])
    assert mic.exact_interval_mask(lo, hi).tolist() == [True, False, False, False, False, True]
