"""Tests for ``genome2mic.eval.metrics``.

Every expected value here is computed by hand in the test body, on tiny frames, so a
regression in any metric shows up as a concrete wrong number rather than a drifted
snapshot. The lab interval convention is the contract's ``(mic_lower, mic_upper]``:
``(4, 8]`` is an exact reading of 8; ``(0, 0.25]`` is ``<= 0.25``; ``(32, inf)`` is
``> 32``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from genome2mic.droplog import DropLog
from genome2mic.eval import metrics as m
from genome2mic.eval.metrics import (
    DISTANCE_COLUMNS,
    SUMMARY_COLUMNS,
    auroc,
    band_coverage,
    band_covers,
    band_width_steps,
    by_distance_bin,
    categorical,
    essential_agreement,
    exact_agreement,
    summarize,
)

INF = math.inf


# ---------------------------------------------------------------------------
# essential_agreement
# ---------------------------------------------------------------------------
class TestEssentialAgreement:
    def test_exact_rows_within_one_doubling_step(self) -> None:
        # Lab (4, 8] is an exact MIC of 8; EA iff pred in {4, 8, 16}.
        pred = [4.0, 8.0, 16.0, 2.0, 32.0]
        lo = [4.0] * 5
        hi = [8.0] * 5
        assert essential_agreement(pred, lo, hi).tolist() == [True, True, True, False, False]

    def test_left_censored_pred_at_most_one_step_above_edge(self) -> None:
        # Lab <= 0.25 -> (0, 0.25]. EA iff pred <= 2 * 0.25 = 0.5.
        pred = [0.125, 0.25, 0.5, 1.0]
        lo = [0.0] * 4
        hi = [0.25] * 4
        assert essential_agreement(pred, lo, hi).tolist() == [True, True, True, False]

    def test_right_censored_pred_at_least_the_edge(self) -> None:
        # Lab > 32 -> (32, inf). EA iff pred >= 32.
        pred = [16.0, 32.0, 64.0, 1024.0]
        lo = [32.0] * 4
        hi = [INF] * 4
        assert essential_agreement(pred, lo, hi).tolist() == [False, True, True, True]

    def test_multi_step_interval_uses_one_step_beyond_either_edge(self) -> None:
        # An I-only label under S<=2 / R>8 is the two-step interval (2, 8].
        # Steps inside: 4, 8. EA iff 2 <= pred <= 16.
        pred = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        lo = [2.0] * 6
        hi = [8.0] * 6
        assert essential_agreement(pred, lo, hi).tolist() == [False, True, True, True, True, False]

    def test_off_grid_prediction_is_rounded_up_first(self) -> None:
        # 6 -> 8 (same step as lab 8); 2.1 -> 4 (one step below 8).
        assert essential_agreement([6.0, 2.1], [4.0, 4.0], [8.0, 8.0]).tolist() == [True, True]

    def test_float_drift_on_grid_value_is_tolerated(self) -> None:
        drifted = 8.000000000000002
        assert essential_agreement([drifted], [4.0], [8.0]).tolist() == [True]

    def test_accepts_series_and_returns_bool_ndarray(self) -> None:
        out = essential_agreement(pd.Series([8.0]), pd.Series([4.0]), pd.Series([8.0]))
        assert isinstance(out, np.ndarray) and out.dtype == bool and out.tolist() == [True]

    def test_empty_input_returns_empty_bool_array(self) -> None:
        out = essential_agreement([], [], [])
        assert out.dtype == bool and out.shape == (0,)

    @pytest.mark.parametrize(
        "pred, lo, hi",
        [
            ([np.nan], [4.0], [8.0]),  # missing prediction
            ([8.0], [np.nan], [8.0]),  # missing lab bound
            ([8.0], [8.0], [8.0]),  # lo >= hi
            ([8.0], [16.0], [8.0]),  # lo > hi
            ([8.0], [0.0], [INF]),  # uninformative lab interval
            ([0.0], [4.0], [8.0]),  # non-positive prediction
            ([8.0], [-1.0], [8.0]),  # negative lower bound
        ],
    )
    def test_invalid_inputs_raise(self, pred: list, lo: list, hi: list) -> None:
        with pytest.raises(ValueError):
            essential_agreement(pred, lo, hi)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            essential_agreement([8.0, 8.0], [4.0], [8.0])


# ---------------------------------------------------------------------------
# exact_agreement
# ---------------------------------------------------------------------------
class TestExactAgreement:
    def test_exact_rows_same_step_only(self) -> None:
        pred = [4.0, 8.0, 16.0]
        assert exact_agreement(pred, [4.0] * 3, [8.0] * 3).tolist() == [False, True, False]

    def test_left_censored_inside_interval(self) -> None:
        # (0, 0.25]: anything at or below 0.25 agrees with "<= 0.25".
        pred = [0.125, 0.25, 0.5]
        assert exact_agreement(pred, [0.0] * 3, [0.25] * 3).tolist() == [True, True, False]

    def test_right_censored_strictly_above_edge(self) -> None:
        # (32, inf): 32 itself is NOT inside "> 32".
        pred = [16.0, 32.0, 64.0]
        assert exact_agreement(pred, [32.0] * 3, [INF] * 3).tolist() == [False, False, True]

    def test_multi_step_interval_any_inside_step_agrees(self) -> None:
        pred = [2.0, 4.0, 8.0, 16.0]
        assert exact_agreement(pred, [2.0] * 4, [8.0] * 4).tolist() == [False, True, True, False]

    def test_float_drift_does_not_fall_off_the_edge(self) -> None:
        assert exact_agreement([8.000000000000002], [4.0], [8.0]).tolist() == [True]
        # A drifted value sitting on the *exclusive* lower bound stays outside.
        assert exact_agreement([4.000000000000001], [4.0], [8.0]).tolist() == [False]

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            exact_agreement([np.nan], [4.0], [8.0])
        with pytest.raises(ValueError):
            exact_agreement([8.0], [0.0], [INF])


# ---------------------------------------------------------------------------
# categorical
# ---------------------------------------------------------------------------
class TestCategorical:
    # (pred_sir, lab_sir) for 10 rows, chosen so every cell of the definition is hit.
    PRED = ["S", "S", "R", "R", "I", "R", "S", "S", "S", "I"]
    LAB_ = ["R", "R", "R", "R", "R", "S", "S", "S", "S", "I"]
    # lab R: rows 1-5 (5). lab S: rows 6-9 (4). lab I: row 10.
    # VME (pred S, lab R): rows 1, 2 -> 2 / 5 = 0.4
    # ME  (pred R, lab S): row 6     -> 1 / 4 = 0.25
    # mIE (exactly one I): row 5     -> 1 / 10 = 0.1   (row 10 is I/I: not a minor error)
    # agree: rows 3, 4, 7, 8, 9, 10  -> 6 / 10 = 0.6

    def test_hand_computed_rates(self) -> None:
        out = categorical(self.PRED, self.LAB_)
        assert out["n_cat"] == 10
        assert out["n_lab_r"] == 5
        assert out["n_lab_s"] == 4
        assert out["n_vme"] == 2 and out["vme_rate"] == pytest.approx(0.4)
        assert out["n_me"] == 1 and out["me_rate"] == pytest.approx(0.25)
        assert out["n_mine"] == 1 and out["mine_rate"] == pytest.approx(0.1)
        assert out["n_agree"] == 6 and out["ca"] == pytest.approx(0.6)
        assert out["n_excluded"] == 0

    def test_null_on_either_side_is_excluded_and_counted(self) -> None:
        pred = self.PRED + [None, "S"]
        lab = self.LAB_ + ["R", None]
        out = categorical(pd.Series(pred, dtype="str"), pd.Series(lab, dtype="str"))
        assert out["n_excluded"] == 2
        assert out["n_cat"] == 10
        assert out["vme_rate"] == pytest.approx(0.4)
        assert out["me_rate"] == pytest.approx(0.25)
        assert out["ca"] == pytest.approx(0.6)

    def test_rates_are_none_when_denominator_is_zero(self) -> None:
        out = categorical(["S", "S"], ["S", "S"])  # no lab R rows
        assert out["vme_rate"] is None
        assert out["me_rate"] == pytest.approx(0.0)
        assert out["n_lab_r"] == 0 and out["n_lab_s"] == 2

    def test_empty_input(self) -> None:
        out = categorical([], [])
        assert out["n_cat"] == 0
        assert out["ca"] is None and out["vme_rate"] is None
        assert out["me_rate"] is None and out["mine_rate"] is None

    def test_all_null_input(self) -> None:
        out = categorical([None, None], ["R", None])
        assert out["n_cat"] == 0 and out["n_excluded"] == 2

    @pytest.mark.parametrize("bad", ["X", "s", "Susceptible", "", "NA"])
    def test_values_outside_sir_raise(self, bad: str) -> None:
        with pytest.raises(ValueError):
            categorical(["S", bad], ["R", "R"])
        with pytest.raises(ValueError):
            categorical(["S", "S"], ["R", bad])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            categorical(["S"], ["R", "R"])

    def test_i_vs_i_is_agreement_not_minor_error(self) -> None:
        out = categorical(["I"], ["I"])
        assert out["ca"] == pytest.approx(1.0) and out["mine_rate"] == pytest.approx(0.0)

    def test_pred_i_lab_r_and_pred_r_lab_i_are_minor_errors_not_vme_or_me(self) -> None:
        out = categorical(["I", "R", "S"], ["R", "I", "I"])
        assert out["n_mine"] == 3 and out["mine_rate"] == pytest.approx(1.0)
        assert out["n_vme"] == 0 and out["n_me"] == 0
        assert out["vme_rate"] == pytest.approx(0.0)  # n_lab_r = 1, no VME
        assert out["me_rate"] is None  # no lab S rows


# ---------------------------------------------------------------------------
# auroc
# ---------------------------------------------------------------------------
class TestAuroc:
    def test_perfect_separation(self) -> None:
        assert auroc([0.0, 1.0, 2.0, 3.0], ["S", "S", "R", "R"]) == pytest.approx(1.0)

    def test_perfectly_inverted(self) -> None:
        assert auroc([3.0, 2.0, 1.0, 0.0], ["S", "S", "R", "R"]) == pytest.approx(0.0)

    def test_hand_computed_partial_ranking(self) -> None:
        # R scores {2, 4}; S scores {1, 3}. Pairs (R > S): 2>1 yes, 2>3 no, 4>1 yes, 4>3 yes -> 3/4.
        assert auroc([1.0, 2.0, 3.0, 4.0], ["S", "R", "S", "R"]) == pytest.approx(0.75)

    def test_ties_count_half(self) -> None:
        assert auroc([1.0, 1.0], ["S", "R"]) == pytest.approx(0.5)

    def test_intermediate_rows_are_excluded(self) -> None:
        # Without the I row this is the perfect case above.
        assert auroc([0.0, 1.0, 2.5, 2.0, 3.0], ["S", "S", "I", "R", "R"]) == pytest.approx(1.0)

    def test_null_rows_are_excluded(self) -> None:
        scores = pd.Series([0.0, 1.0, np.nan, 2.0, 3.0])
        labels = pd.Series(["S", "S", "R", "R", None], dtype="str")
        # Remaining: (0,S), (1,S), (2,R) -> perfect.
        assert auroc(scores, labels) == pytest.approx(1.0)

    def test_none_when_one_class_missing(self) -> None:
        assert auroc([0.0, 1.0], ["S", "S"]) is None
        assert auroc([0.0, 1.0], ["R", "R"]) is None
        assert auroc([0.0, 1.0, 2.0], ["R", "R", "I"]) is None

    def test_none_when_fewer_than_two_rows(self) -> None:
        assert auroc([], []) is None
        assert auroc([1.0], ["R"]) is None

    def test_infinite_scores_are_handled_by_rank(self) -> None:
        # A right-censored prediction at the top of the panel is a legitimate "largest" score.
        assert auroc([0.0, np.inf, 1.0], ["S", "R", "S"]) == pytest.approx(1.0)

    def test_invalid_label_raises(self) -> None:
        with pytest.raises(ValueError):
            auroc([0.0, 1.0], ["S", "Resistant"])


# ---------------------------------------------------------------------------
# band coverage and width
# ---------------------------------------------------------------------------
class TestBand:
    def test_exact_row_covered_iff_reported_step_inside_band(self) -> None:
        # lab (4, 8] -> reported 8.
        lows = [4.0, 1.0, 16.0, 8.0]
        highs = [16.0, 4.0, 32.0, 8.0]
        out = band_covers(lows, highs, [4.0] * 4, [8.0] * 4)
        assert out.tolist() == [True, False, False, True]

    def test_left_censored_covered_iff_band_reaches_down_to_edge(self) -> None:
        # lab (0, 0.25]: band must start at or below 0.25.
        out = band_covers([0.125, 0.5, 0.25], [0.5, 1.0, 1.0], [0.0] * 3, [0.25] * 3)
        assert out.tolist() == [True, False, True]

    def test_right_censored_covered_iff_band_top_strictly_above_edge(self) -> None:
        # lab (32, inf): band_high == 32 does not reach into "> 32".
        out = band_covers([16.0, 8.0, 8.0], [64.0, 32.0, 64.0], [32.0] * 3, [INF] * 3)
        assert out.tolist() == [True, False, True]

    def test_multi_step_interval_overlap(self) -> None:
        # lab (2, 8]: band [2, 4] contains step 4, so it is consistent with the lab.
        assert band_covers([2.0], [4.0], [2.0], [8.0]).tolist() == [True]
        assert band_covers([16.0], [32.0], [2.0], [8.0]).tolist() == [False]

    def test_band_coverage_is_the_mean(self) -> None:
        lows = [4.0, 1.0, 16.0, 8.0]
        highs = [16.0, 4.0, 32.0, 8.0]
        # Same rows as the first test: T, F, F, T -> 2 / 4.
        assert band_coverage(lows, highs, [4.0] * 4, [8.0] * 4) == pytest.approx(0.5)

    def test_band_coverage_empty_is_none(self) -> None:
        assert band_coverage([], [], [], []) is None

    def test_band_width_steps_mean(self) -> None:
        # widths: log2(8/2)=2, log2(1/1)=0, log2(4/0.5)=3 -> mean 5/3
        assert band_width_steps([2.0, 1.0, 0.5], [8.0, 1.0, 4.0]) == pytest.approx(5.0 / 3.0)

    def test_band_width_steps_empty_is_none(self) -> None:
        assert band_width_steps([], []) is None

    def test_band_width_infinite_top_is_infinite(self) -> None:
        assert band_width_steps([2.0], [INF]) == INF

    @pytest.mark.parametrize(
        "low, high",
        [([0.0], [8.0]), ([-1.0], [8.0]), ([16.0], [8.0]), ([np.nan], [8.0]), ([4.0], [np.nan])],
    )
    def test_invalid_bands_raise(self, low: list, high: list) -> None:
        with pytest.raises(ValueError):
            band_width_steps(low, high)
        with pytest.raises(ValueError):
            band_covers(low, high, [4.0], [8.0])


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def _preds_frame() -> pd.DataFrame:
    """Hand-built preds table: three (species, drug, model, split) groups.

    Group A — KPNEU / meropenem / aft_known / test, 6 rows (breakpoints S<=2, R>4):
      r1 pred 8    band [4, 16]   lab (4, 8]     R/R   EA yes  exact yes  covered yes  width 2
      r2 pred 0.25 band [.125,.5] lab (0, 0.25]  S/S   EA yes  exact yes  covered yes  width 2
      r3 pred 2    band [1, 4]    lab (32, inf)  S/R   EA no   exact no   covered no   width 2  <- VME
      r4 pred 16   band [8, 32]   lab (0.5, 1]   R/S   EA no   exact no   covered no   width 2  <- ME
      r5 pred 4    band [2, 8]    lab (1, 2]     I/S   EA yes  exact no   covered yes  width 2  <- mIE
      r6 pred 64   band [16, 128] lab (32, inf)  R/R   EA yes  exact yes  covered yes  width 3
      -> n 6, n_cat 6, n_lab_r 3, n_lab_s 3, vme 1/3, me 1/3, mine 1/6, ca 3/6,
         EA 4/6, exact 3/6, coverage 4/6, width 13/6, n_exact 3 (r1, r4, r5),
         AUROC: R scores log2 {3, 1, 6} vs S scores {-2, 4, 2}: wins 2+1+3 = 6 / 9.

    Group B — same pair and split, model b0_resfinder (no MIC, no band), 3 rows:
      b1 S/R lab (32, inf)  <- VME ; b2 R/R lab (4, 8] ; b3 S/S lab (0, 0.25]
      -> n 3, n_cat 3, n_lab_r 2, n_lab_s 1, vme 1/2, me 0, mine 0, ca 2/3, n_exact 1,
         EA / exact / auroc / band metrics all null.

    Group C — KPNEU / meropenem / aft_known / cv, 2 perfect rows:
      c1 pred 8 band [8, 8]  lab (4, 8]   R/R ; c2 pred 1 band [0.5, 2] lab (0.5, 1] S/S
      -> everything 1.0 or 0.0, width (0 + 2) / 2 = 1, n_exact 2, auroc 1.0.
    """
    a = pd.DataFrame(
        {
            "genome_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "species": "KPNEU",
            "drug": "meropenem",
            "split": "test",
            "model": "aft_known",
            "pred_mic": [8.0, 0.25, 2.0, 16.0, 4.0, 64.0],
            "band_low": [4.0, 0.125, 1.0, 8.0, 2.0, 16.0],
            "band_high": [16.0, 0.5, 4.0, 32.0, 8.0, 128.0],
            "lab_lower": [4.0, 0.0, 32.0, 0.5, 1.0, 32.0],
            "lab_upper": [8.0, 0.25, INF, 1.0, 2.0, INF],
            "pred_sir": ["R", "S", "S", "R", "I", "R"],
            "lab_sir": ["R", "S", "R", "S", "S", "R"],
            "run_id": "abc",
        }
    )
    b = pd.DataFrame(
        {
            "genome_id": ["g3", "g1", "g2"],
            "species": "KPNEU",
            "drug": "meropenem",
            "split": "test",
            "model": "b0_resfinder",
            "pred_mic": [np.nan, np.nan, np.nan],
            "band_low": [np.nan, np.nan, np.nan],
            "band_high": [np.nan, np.nan, np.nan],
            "lab_lower": [32.0, 4.0, 0.0],
            "lab_upper": [INF, 8.0, 0.25],
            "pred_sir": ["S", "R", "S"],
            "lab_sir": ["R", "R", "S"],
            "run_id": "abc",
        }
    )
    c = pd.DataFrame(
        {
            "genome_id": ["g7", "g8"],
            "species": "KPNEU",
            "drug": "meropenem",
            "split": "cv",
            "model": "aft_known",
            "pred_mic": [8.0, 1.0],
            "band_low": [8.0, 0.5],
            "band_high": [8.0, 2.0],
            "lab_lower": [4.0, 0.5],
            "lab_upper": [8.0, 1.0],
            "pred_sir": ["R", "S"],
            "lab_sir": ["R", "S"],
            "run_id": "abc",
        }
    )
    # Deliberately shuffled so the output ordering is the function's doing, not ours.
    return pd.concat([b, c, a], ignore_index=True)


class TestSummarize:
    def test_vme_is_the_first_metric_column(self) -> None:
        out = summarize(_preds_frame())
        keys = ["species", "drug", "model", "split"]
        assert list(out.columns[: len(keys)]) == keys
        assert out.columns[len(keys)] == "vme_rate", "VME must be reported first"

    def test_exact_column_order(self) -> None:
        expected = [
            "species", "drug", "model", "split",
            "vme_rate", "me_rate", "mine_rate", "categorical_agreement",
            "essential_agreement", "exact_agreement", "auroc",
            "band_coverage", "band_width_steps",
            "n", "n_exact", "n_cat", "n_lab_r", "n_lab_s",
        ]  # fmt: skip
        assert list(SUMMARY_COLUMNS) == expected
        assert list(summarize(_preds_frame()).columns) == expected

    def test_one_row_per_group_sorted_by_keys(self) -> None:
        out = summarize(_preds_frame())
        assert len(out) == 3
        assert out[["species", "drug", "model", "split"]].values.tolist() == [
            ["KPNEU", "meropenem", "aft_known", "cv"],
            ["KPNEU", "meropenem", "aft_known", "test"],
            ["KPNEU", "meropenem", "b0_resfinder", "test"],
        ]

    def test_group_a_hand_computed(self) -> None:
        out = summarize(_preds_frame()).set_index(["model", "split"])
        row = out.loc[("aft_known", "test")]
        assert row["n"] == 6
        assert row["n_cat"] == 6
        assert row["n_lab_r"] == 3
        assert row["n_lab_s"] == 3
        assert row["n_exact"] == 3
        assert row["vme_rate"] == pytest.approx(1 / 3)
        assert row["me_rate"] == pytest.approx(1 / 3)
        assert row["mine_rate"] == pytest.approx(1 / 6)
        assert row["categorical_agreement"] == pytest.approx(3 / 6)
        assert row["essential_agreement"] == pytest.approx(4 / 6)
        assert row["exact_agreement"] == pytest.approx(3 / 6)
        assert row["auroc"] == pytest.approx(6 / 9)
        assert row["band_coverage"] == pytest.approx(4 / 6)
        assert row["band_width_steps"] == pytest.approx(13 / 6)

    def test_group_b_without_mic_contributes_only_categorical(self) -> None:
        out = summarize(_preds_frame()).set_index(["model", "split"])
        row = out.loc[("b0_resfinder", "test")]
        assert row["n"] == 3
        assert row["n_cat"] == 3
        assert row["n_lab_r"] == 2
        assert row["n_lab_s"] == 1
        assert row["n_exact"] == 1
        assert row["vme_rate"] == pytest.approx(0.5)
        assert row["me_rate"] == pytest.approx(0.0)
        assert row["mine_rate"] == pytest.approx(0.0)
        assert row["categorical_agreement"] == pytest.approx(2 / 3)
        for col in ("essential_agreement", "exact_agreement", "auroc", "band_coverage", "band_width_steps"):
            assert pd.isna(row[col]), col

    def test_group_c_perfect(self) -> None:
        out = summarize(_preds_frame()).set_index(["model", "split"])
        row = out.loc[("aft_known", "cv")]
        assert row["n"] == 2 and row["n_exact"] == 2
        assert row["vme_rate"] == pytest.approx(0.0)
        assert row["me_rate"] == pytest.approx(0.0)
        assert row["mine_rate"] == pytest.approx(0.0)
        assert row["categorical_agreement"] == pytest.approx(1.0)
        assert row["essential_agreement"] == pytest.approx(1.0)
        assert row["exact_agreement"] == pytest.approx(1.0)
        assert row["auroc"] == pytest.approx(1.0)
        assert row["band_coverage"] == pytest.approx(1.0)
        assert row["band_width_steps"] == pytest.approx(1.0)

    def test_dtypes(self) -> None:
        out = summarize(_preds_frame())
        for col in ("n", "n_exact", "n_cat", "n_lab_r", "n_lab_s"):
            assert pd.api.types.is_integer_dtype(out[col]), col
        for col in SUMMARY_COLUMNS[4:13]:
            assert pd.api.types.is_float_dtype(out[col]), col

    def test_missing_band_columns_give_null_band_metrics(self) -> None:
        preds = _preds_frame().drop(columns=["band_low", "band_high"])
        out = summarize(preds)
        assert out["band_coverage"].isna().all()
        assert out["band_width_steps"].isna().all()
        # The other metrics are unaffected.
        row = out.set_index(["model", "split"]).loc[("aft_known", "test")]
        assert row["essential_agreement"] == pytest.approx(4 / 6)

    def test_null_sir_rows_excluded_from_categorical_but_counted_in_n(self) -> None:
        preds = _preds_frame()
        preds = preds[(preds["model"] == "aft_known") & (preds["split"] == "cv")].copy()
        extra = preds.iloc[[0]].copy()
        extra["genome_id"] = "g9"
        extra["pred_sir"] = None
        preds = pd.concat([preds, extra], ignore_index=True)
        out = summarize(preds).iloc[0]
        assert out["n"] == 3
        assert out["n_cat"] == 2
        assert out["essential_agreement"] == pytest.approx(1.0)  # the extra row still has a MIC

    def test_uninformative_lab_interval_excluded_from_mic_metrics(self) -> None:
        preds = _preds_frame()
        preds = preds[(preds["model"] == "aft_known") & (preds["split"] == "cv")].copy()
        extra = preds.iloc[[0]].copy()
        extra["genome_id"] = "g9"
        extra["lab_lower"] = 0.0
        extra["lab_upper"] = INF
        preds = pd.concat([preds, extra], ignore_index=True)
        out = summarize(preds).iloc[0]
        assert out["n"] == 3
        assert out["n_exact"] == 2
        assert out["essential_agreement"] == pytest.approx(1.0)
        assert out["band_coverage"] == pytest.approx(1.0)

    def test_empty_input_returns_empty_frame_with_columns(self) -> None:
        empty = _preds_frame().iloc[0:0]
        out = summarize(empty)
        assert list(out.columns) == list(SUMMARY_COLUMNS)
        assert len(out) == 0

    def test_missing_required_column_raises(self) -> None:
        with pytest.raises(ValueError, match="pred_sir"):
            summarize(_preds_frame().drop(columns=["pred_sir"]))

    def test_invalid_lab_interval_raises(self) -> None:
        preds = _preds_frame()
        preds.loc[preds.index[0], ["lab_lower", "lab_upper"]] = [8.0, 4.0]
        with pytest.raises(ValueError):
            summarize(preds)

    def test_drop_log_records_every_exclusion(self) -> None:
        log = DropLog("eval")
        summarize(_preds_frame(), drop_log=log)
        reasons = {r.reason: r.n_dropped for r in log.records}
        # 3 b0_resfinder rows have no pred_mic and no band.
        assert reasons["pred_mic null: categorical metrics only"] == 3
        assert reasons["band_low/band_high null: excluded from band metrics"] == 3
        # No null S/I/R, no null lab bounds, no (0, inf) rows in the fixture; filters still logged.
        assert reasons["pred_sir or lab_sir null: excluded from categorical metrics"] == 0
        assert reasons["lab_lower/lab_upper null: excluded from MIC metrics"] == 0
        assert reasons["lab interval (0, inf): excluded from MIC metrics"] == 0
        # r5 is pred I / lab S: no lab I row in the fixture, so nothing leaves the R-vs-S AUROC.
        assert reasons["lab_sir I: excluded from AUROC"] == 0

    def test_drop_log_counts_lab_i_rows_left_out_of_auroc(self) -> None:
        preds = _preds_frame()
        preds = preds[(preds["model"] == "aft_known") & (preds["split"] == "cv")].copy()
        extra = preds.iloc[[0]].copy()
        extra["genome_id"] = "g9"
        extra["lab_sir"] = "I"
        extra["lab_lower"], extra["lab_upper"] = 2.0, 4.0  # (2, 4]: I under S<=2, R>4
        preds = pd.concat([preds, extra], ignore_index=True)
        log = DropLog("eval")
        out = summarize(preds, drop_log=log).iloc[0]
        reasons = {r.reason: r.n_dropped for r in log.records}
        assert reasons["lab_sir I: excluded from AUROC"] == 1
        assert out["n_cat"] == 3  # the I row still counts for categorical metrics ...
        assert out["auroc"] == pytest.approx(1.0)  # ... but not for the R-vs-S AUROC

    def test_summarize_logs_without_an_explicit_drop_log(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.INFO, logger="genome2mic.droplog"):
            summarize(_preds_frame())
        assert any("pred_mic null" in r.getMessage() for r in caplog.records)

    def test_extra_columns_are_ignored(self) -> None:
        preds = _preds_frame()
        preds["external_set"] = None
        preds["lineage_cluster"] = "KPNEU_PP_1"  # evaluation-only column; must not break anything
        assert len(summarize(preds)) == 3


# ---------------------------------------------------------------------------
# by_distance_bin
# ---------------------------------------------------------------------------
class TestByDistanceBin:
    @staticmethod
    def _group_a() -> pd.DataFrame:
        preds = _preds_frame()
        return preds[(preds["model"] == "aft_known") & (preds["split"] == "test")].reset_index(drop=True)

    def test_bins_from_genome_id_series(self) -> None:
        preds = self._group_a()
        # g1, g2 -> (0, 0.01]; g3, g4 -> (0.01, 0.05]; g5 null; g6 beyond the last edge.
        dist = pd.Series({"g1": 0.001, "g2": 0.002, "g3": 0.03, "g4": 0.04, "g5": np.nan, "g6": 0.2})
        log = DropLog("eval")
        out = by_distance_bin(preds, dist, bins=[0.0, 0.01, 0.05], drop_log=log)

        assert list(out.columns) == list(DISTANCE_COLUMNS)
        assert out.columns[:5].tolist() == ["species", "drug", "model", "split", "distance_bin"]
        assert out.columns[7] == "vme_rate"  # after bin_low / bin_high, VME comes first
        assert len(out) == 2

        near = out[out["distance_bin"] == "[0, 0.01]"].iloc[0]
        far = out[out["distance_bin"] == "(0.01, 0.05]"].iloc[0]

        assert near["bin_low"] == pytest.approx(0.0) and near["bin_high"] == pytest.approx(0.01)
        assert near["n"] == 2
        assert near["vme_rate"] == pytest.approx(0.0)  # r1 lab R predicted R
        assert near["me_rate"] == pytest.approx(0.0)  # r2 lab S predicted S
        assert near["essential_agreement"] == pytest.approx(1.0)
        assert near["categorical_agreement"] == pytest.approx(1.0)

        assert far["n"] == 2
        assert far["vme_rate"] == pytest.approx(1.0)  # r3: pred S, lab R
        assert far["me_rate"] == pytest.approx(1.0)  # r4: pred R, lab S
        assert far["essential_agreement"] == pytest.approx(0.0)
        assert far["band_coverage"] == pytest.approx(0.0)

        reasons = {r.reason: r.n_dropped for r in log.records}
        assert reasons["nearest_distance null: excluded from distance bins"] == 1
        assert reasons["nearest_distance outside bins: excluded from distance bins"] == 1

    def test_positional_array_of_distances(self) -> None:
        preds = self._group_a()
        dist = np.array([0.001, 0.002, 0.03, 0.04, 0.03, 0.04])
        out = by_distance_bin(preds, dist, bins=[0.0, 0.01, 0.05])
        assert out["n"].tolist() == [2, 4]

    def test_mapping_of_distances(self) -> None:
        preds = self._group_a()
        dist = {"g1": 0.001, "g2": 0.002, "g3": 0.03, "g4": 0.04, "g5": 0.03, "g6": 0.04}
        out = by_distance_bin(preds, dist, bins=[0.0, 0.01, 0.05])
        assert out["n"].tolist() == [2, 4]

    def test_genome_missing_from_mapping_is_excluded_and_counted(self) -> None:
        preds = self._group_a()
        dist = {"g1": 0.001}
        log = DropLog("eval")
        out = by_distance_bin(preds, dist, bins=[0.0, 0.01, 0.05], drop_log=log)
        assert out["n"].tolist() == [1]
        reasons = {r.reason: r.n_dropped for r in log.records}
        assert reasons["nearest_distance null: excluded from distance bins"] == 5

    def test_wrong_length_array_raises(self) -> None:
        with pytest.raises(ValueError):
            by_distance_bin(self._group_a(), np.array([0.1, 0.2]), bins=[0.0, 1.0])

    def test_bad_bins_raise(self) -> None:
        with pytest.raises(ValueError):
            by_distance_bin(self._group_a(), {"g1": 0.1}, bins=[0.5])
        with pytest.raises(ValueError):
            by_distance_bin(self._group_a(), {"g1": 0.1}, bins=[0.5, 0.1])

    def test_default_bins_exist_and_are_increasing(self) -> None:
        edges = list(m.DEFAULT_DISTANCE_BINS)
        assert edges == sorted(edges) and edges[0] == 0.0 and len(edges) >= 3

    def test_bins_spanning_several_models_keep_group_keys(self) -> None:
        preds = _preds_frame()
        dist = {f"g{i}": 0.001 for i in range(1, 9)}
        out = by_distance_bin(preds, dist, bins=[0.0, 0.01])
        assert out[["model", "split"]].values.tolist() == [
            ["aft_known", "cv"],
            ["aft_known", "test"],
            ["b0_resfinder", "test"],
        ]
        assert out["distance_bin"].unique().tolist() == ["[0, 0.01]"]
