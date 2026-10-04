"""Tests for the probability ("does the drug work?") metrics and the per-fold call-VME table.

Written before the implementation (CLAUDE.md: tests first for ``eval/metrics.py``). Every
expected number is worked out by hand in the test body on tiny frames. "Works" means the
lab MIC re-derived under the call breakpoint is ``S``; lab ``I`` and ``R`` do not work.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from genome2mic.eval import metrics as m


# ----------------------------------------------------------------------------- tiers

@pytest.mark.parametrize(
    ("p", "tier"),
    [
        (1.0, "very_likely_works"),
        (0.90, "very_likely_works"),
        (0.8999, "probably_works"),
        (0.70, "probably_works"),
        (0.6999, "uncertain"),
        (0.5, "uncertain"),
        (0.3001, "uncertain"),
        (0.30, "probably_fails"),
        (0.1001, "probably_fails"),
        (0.10, "very_likely_fails"),
        (0.0, "very_likely_fails"),
    ],
)
def test_prob_tier_boundaries(p: float, tier: str) -> None:
    assert m.prob_tier(np.array([p]))[0] == tier


def test_prob_tier_null_stays_null_and_rejects_out_of_range() -> None:
    out = m.prob_tier(np.array([np.nan, 0.5]))
    assert out[0] is None and out[1] == "uncertain"
    with pytest.raises(ValueError):
        m.prob_tier(np.array([1.2]))
    with pytest.raises(ValueError):
        m.prob_tier(np.array([-0.01]))


def test_prob_tiers_are_ordered_works_to_fails() -> None:
    assert m.PROB_TIERS == ("very_likely_works", "probably_works", "uncertain", "probably_fails", "very_likely_fails")


# ------------------------------------------------------------------- per-group numbers

def _rows() -> pd.DataFrame:
    """Ten rows of one species, two drugs; hand-checked numbers below.

    idx prob  lab call
    0   0.95  S   likely_active      works tier, right; forced works, right; call right
    1   0.95  R   likely_active      works tier, DANGER; forced works, DANGER; call DANGER
    2   0.75  S   uncertain          works tier, right; forced works, right; call no answer
    3   0.60  R   uncertain          not confident;      forced works, DANGER
    4   0.40  S   uncertain          not confident;      forced fails, wrong
    5   0.20  R   likely_inactive    fails tier, right;  forced fails, right; call right
    6   0.05  I   likely_inactive    fails tier, right (I does not work); forced fails, right; call right
    7   0.05  S   likely_inactive    fails tier, wrong;  forced fails, wrong; call wrong
    8   0.50  None (straddling)      excluded everywhere
    9   NaN   S   None               no probability and no call: excluded
    """
    return pd.DataFrame({
        "species": ["KPNEU"] * 10,
        "drug": ["meropenem"] * 5 + ["ciprofloxacin"] * 5,
        "model": ["aft_known"] * 10,
        "split": ["cv"] * 10,
        "prob_works": [0.95, 0.95, 0.75, 0.60, 0.40, 0.20, 0.05, 0.05, 0.50, np.nan],
        "call": ["likely_active", "likely_active", "uncertain", "uncertain", "uncertain",
                 "likely_inactive", "likely_inactive", "likely_inactive", "uncertain", None],
        "lab_sir_rederived": ["S", "R", "S", "R", "S", "R", "I", "S", None, "S"],
    })


def test_probability_metrics_by_hand() -> None:
    d = _rows()
    out = m.probability_metrics(d["prob_works"], d["call"], d["lab_sir_rederived"])
    # Rows 0-7 have a probability and a lab category (8: no lab category, 9: no probability).
    assert out["n_prob"] == 8
    assert out["n_prob_lab_r"] == 3  # rows 1, 3, 5
    # Danger (VME-like) first: lab R given a works tier (p >= 0.70) -> row 1 of 3 lab R.
    assert out["tier_danger_rate"] == pytest.approx(1 / 3)
    # Forced at 0.5: lab R with p >= 0.5 -> rows 1 and 3.
    assert out["forced_danger_rate"] == pytest.approx(2 / 3)
    # Call danger: lab R called likely_active -> row 1 of the 3 lab R rows with a call.
    assert out["call_danger_rate"] == pytest.approx(1 / 3)
    assert out["n_call_lab_r"] == 3
    # Confident = works or fails tier: rows 0, 1, 2, 5, 6, 7 -> 6 of 8.
    assert out["n_confident"] == 6
    assert out["confident_rate"] == pytest.approx(6 / 8)
    # Right when confident: 0, 2, 5, 6 right; 1, 7 wrong -> 4 / 6.
    assert out["confident_right_rate"] == pytest.approx(4 / 6)
    # 'Very likely works' actually worked: rows 0 (S) and 1 (R) -> 1 / 2.
    assert out["very_likely_works_right_rate"] == pytest.approx(0.5)
    # Calls: rows 0-7 have a call and a lab category -> 8; answers = 0, 1, 5, 6, 7.
    assert out["n_call"] == 8
    assert out["call_answer_rate"] == pytest.approx(5 / 8)
    # Call right among answers: 0, 5, 6 right; 1, 7 wrong -> 3 / 5.
    assert out["call_right_rate"] == pytest.approx(3 / 5)
    # Forced accuracy at 0.5: right = 0, 2, 5, 6 -> 4 / 8.
    assert out["forced_accuracy"] == pytest.approx(4 / 8)
    # Brier score on the 8 rows with a probability and a lab category.
    works = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    p = np.array([0.95, 0.95, 0.75, 0.60, 0.40, 0.20, 0.05, 0.05])
    assert out["brier"] == pytest.approx(float(np.mean((p - works) ** 2)))


def test_probability_metrics_empty_and_none_rates() -> None:
    out = m.probability_metrics([], [], [])
    assert out["n_prob"] == 0 and out["tier_danger_rate"] is None and out["confident_rate"] is None
    # No lab R: danger rates are None, not 0.
    out = m.probability_metrics([0.9, 0.1], ["likely_active", "likely_inactive"], ["S", "S"])
    assert out["tier_danger_rate"] is None and out["forced_danger_rate"] is None and out["call_danger_rate"] is None
    assert out["forced_accuracy"] == pytest.approx(0.5)


def test_probability_metrics_rejects_bad_probability() -> None:
    with pytest.raises(ValueError):
        m.probability_metrics([1.5], ["uncertain"], ["S"])


def test_probability_summary_per_species_is_danger_first() -> None:
    d = _rows()
    other = d.assign(species="ECOLI", prob_works=0.95, call="likely_active", lab_sir_rederived="S")
    table = m.probability_summary(pd.concat([d, other], ignore_index=True))
    assert list(table.columns[:4]) == ["species", "model", "split", "n_drugs"]
    metric_cols = [c for c in table.columns if c in m.PROB_METRIC_COLUMNS]
    assert metric_cols[0] == "call_danger_rate"  # VME (danger) first
    assert metric_cols[:3] == ["call_danger_rate", "tier_danger_rate", "forced_danger_rate"]
    k = table.set_index("species").loc["KPNEU"]
    assert k["n_drugs"] == 2 and k["n_prob"] == 8
    assert k["confident_rate"] == pytest.approx(6 / 8)
    e = table.set_index("species").loc["ECOLI"]
    assert e["confident_right_rate"] == pytest.approx(1.0) and math.isnan(e["tier_danger_rate"])


def test_probability_summary_without_prob_column_is_empty() -> None:
    d = _rows().drop(columns=["prob_works"])
    table = m.probability_summary(d)
    assert table.empty and list(table.columns) == list(m.PROB_SUMMARY_COLUMNS)


def test_calibration_table_bins_and_observed_share() -> None:
    d = _rows()
    table = m.calibration_table(d, group_columns=("model", "split"))
    assert list(table.columns) == ["model", "split", *m.CALIBRATION_VALUE_COLUMNS]
    assert list(m.CALIBRATION_VALUE_COLUMNS[:2]) == ["prob_bin", "bin_low"]
    by_bin = table.set_index("prob_bin")
    # [0, 0.1]: rows 6 (I) and 7 (S) -> mean p 0.05, observed works 1 / 2.
    low = by_bin.loc["[0, 0.1]"]
    assert low["n"] == 2 and low["mean_prob"] == pytest.approx(0.05) and low["observed_works"] == pytest.approx(0.5)
    # (0.9, 1]: rows 0 (S) and 1 (R) -> 0.5 observed.
    high = by_bin.loc["(0.9, 1]"]
    assert high["n"] == 2 and high["observed_works"] == pytest.approx(0.5)
    # (0.5, 0.7]: row 3 only (0.60, R); row 8 (p 0.5, no lab) is excluded.
    mid = by_bin.loc["(0.5, 0.7]"]
    assert mid["n"] == 1 and mid["observed_works"] == 0.0
    assert int(table["n"].sum()) == 8


# ------------------------------------------------------------ per-fold call VME + p

def test_binomial_excess_p_matches_hand_computation() -> None:
    # P(X >= 1 | n=10, p=0.015) = 1 - 0.985^10
    assert m.binomial_excess_p(1, 10, 0.015) == pytest.approx(1 - 0.985**10)
    assert m.binomial_excess_p(0, 10, 0.015) == 1.0
    assert m.binomial_excess_p(0, 0, 0.015) == 1.0
    # KPNEU trimethoprim-sulfamethoxazole fold 4 of the previous run: 19 / 440 -> p ~ 5e-5.
    assert m.binomial_excess_p(19, 440, 0.015) < 1e-4


def test_call_vme_by_fold() -> None:
    gids = [f"g{i}" for i in range(12)]
    preds = pd.DataFrame({
        "genome_id": gids,
        "species": "KPNEU", "drug": "meropenem", "model": "aft_known", "split": "cv",
        # fold 0: 3 lab R, 2 called active (calling fold); fold 1: 3 lab R, no active call at all.
        "call": ["likely_active", "likely_active", "uncertain", "likely_active", "uncertain", "uncertain",
                 "uncertain", "uncertain", "uncertain", "likely_inactive", "uncertain", "uncertain"],
        "lab_sir_rederived": ["R", "R", "R", "S", "S", "S", "R", "R", "R", "S", "S", None],
    })
    folds = pd.Series([0] * 6 + [1] * 6, index=gids)
    table = m.call_vme_by_fold(preds, folds)
    assert list(table.columns) == list(m.FOLD_VME_COLUMNS)
    f0 = table.loc[table["fold"] == 0].iloc[0]
    assert f0["calling"] and f0["n_lab_r"] == 3 and f0["n_vme"] == 2
    assert f0["call_vme_rate"] == pytest.approx(2 / 3)
    assert f0["p_value"] == pytest.approx(m.binomial_excess_p(2, 3, 0.015))
    assert f0["significant"]  # p << 0.01
    f1 = table.loc[table["fold"] == 1].iloc[0]
    assert not f1["calling"] and f1["n_vme"] == 0 and f1["p_value"] == 1.0 and not f1["significant"]
    # Only cv rows count; a test row in the same table is ignored.
    more = pd.concat([preds, preds.assign(split="test")], ignore_index=True)
    assert m.call_vme_by_fold(more, folds).equals(table)
