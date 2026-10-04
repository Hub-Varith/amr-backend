"""Tests for the v0.6 ``aft_b2_select`` model: candidate combination, bundle round trip and the choice rule.

Synthetic toy data only (the ``test_models`` fixture): nothing here says anything about real genomes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from genome2mic.models import MODEL_CLASSES, AftB2Select, B2XgbSteps, XgbAft
from genome2mic.models import aft_b2_select as sel
from genome2mic.models import train
from genome2mic.models.train import choose_candidate, select_key, select_score
from tests.models.test_models import FEATURE_NAMES, make_synthetic

CAPS = (-3.0, 3.0)
"""Tight caps so the clip is visible: the toy MICs span about -4.5 .. +5.5 log2."""


@pytest.fixture(scope="module")
def toy() -> dict:
    data = make_synthetic(n=300, seed=5)
    # The bundle's feature list: the AFT model reads every column (here one extra "unitig"),
    # B2 only the known-AMR ones -- as with aft_known_unitig + b2_xgb_steps.
    unitig = (np.arange(data.X.shape[0]) % 3 == 0).astype(np.int8)
    X_all = sp.csr_matrix(sp.hstack([data.X, sp.csr_matrix(unitig.reshape(-1, 1))]))
    names_all = [*FEATURE_NAMES, "u_000007"]
    aft = XgbAft(name="aft_known_unitig", seed=0, nthread=1, max_rounds=40).fit(X_all, data.lo, data.hi, names_all)
    b2 = B2XgbSteps(seed=0, nthread=1, max_rounds=40).fit(data.X, data.lo, data.hi, FEATURE_NAMES)
    return {"data": data, "X_all": X_all, "names_all": names_all, "aft": aft, "b2": b2}


# --------------------------------------------------------------------------- combine_log2


def test_combine_log2_candidates_and_fallback() -> None:
    a = np.array([1.0, 2.0, np.nan, np.nan, -1.5])
    b = np.array([3.0, np.nan, 4.0, np.nan, -2.0])
    np.testing.assert_array_equal(sel.combine_log2(a, b, "aft"), a)
    np.testing.assert_array_equal(sel.combine_log2(a, b, "b2"), b)
    avg = sel.combine_log2(a, b, "avg")
    np.testing.assert_allclose(avg[[0, 1, 2, 4]], [2.0, 2.0, 4.0, -1.75])  # one missing -> the other one
    assert np.isnan(avg[3])
    with pytest.raises(ValueError):
        sel.combine_log2(a, b, "median")


def test_preference_order_is_aft_avg_b2() -> None:
    assert sel.CHOICES == ("aft", "avg", "b2")


# --------------------------------------------------------------------------- bundle model


@pytest.mark.parametrize("choice", ["aft", "avg", "b2"])
def test_select_model_predicts_the_capped_candidate_and_round_trips(toy: dict, choice: str, tmp_path: Path) -> None:
    X, aft, b2 = toy["X_all"], toy["aft"], toy["b2"]
    model = AftB2Select.from_parts(choice, aft, b2, CAPS, toy["names_all"], base_name="aft_known_unitig")
    a = np.clip(aft.predict_log2(X), *CAPS)
    b = np.clip(b2.predict_log2(X[:, :2]), *CAPS)  # B2 reads only its own (known-AMR) columns
    expected = sel.combine_log2(a, b, choice)
    got = model.predict_log2(X)
    np.testing.assert_allclose(got, expected)
    assert got.min() >= CAPS[0] and got.max() <= CAPS[1]
    if choice == "avg":
        assert not np.allclose(got, a) and not np.allclose(got, b)

    model.save(tmp_path)
    assert (tmp_path / "aft" / "params.json").is_file() == (choice in ("aft", "avg"))
    assert (tmp_path / "b2" / "params.json").is_file() == (choice in ("b2", "avg"))
    loaded = MODEL_CLASSES["aft_b2_select"].load(tmp_path)
    assert isinstance(loaded, AftB2Select)
    assert loaded.choice == choice and loaded.caps == CAPS and loaded.base_name == "aft_known_unitig"
    assert loaded.feature_names_ == toy["names_all"]
    np.testing.assert_allclose(loaded.predict_log2(X), got)
    # one genome as a dense row (what the prediction pipeline passes)
    np.testing.assert_allclose(loaded.predict_log2(X[:1].toarray()), got[:1])
    importance = loaded.feature_importance("gain")
    assert set(importance.index) == set(toy["names_all"]) and (importance >= 0).all()


def test_select_model_needs_its_components_and_names(toy: dict) -> None:
    with pytest.raises(ValueError, match="AFT component"):
        AftB2Select(choice="avg", aft=None, b2=toy["b2"], feature_names=toy["names_all"])
    with pytest.raises(ValueError, match="B2 component"):
        AftB2Select(choice="b2", aft=toy["aft"], b2=None, feature_names=toy["names_all"])
    with pytest.raises(ValueError, match="does not list"):
        AftB2Select.from_parts("avg", toy["aft"], toy["b2"], CAPS, ["gene_blakpc_2"])
    with pytest.raises(ValueError):
        AftB2Select(choice="best")
    with pytest.raises(TypeError):
        AftB2Select().fit(None, None, None, [])
    with pytest.raises(ValueError, match="columns"):
        AftB2Select.from_parts("aft", toy["aft"], toy["b2"], CAPS, toy["names_all"]).predict_log2(np.zeros((1, 2)))


def test_from_parts_keeps_only_what_the_choice_needs(toy: dict) -> None:
    only_aft = AftB2Select.from_parts("aft", toy["aft"], toy["b2"], None, toy["names_all"])
    assert only_aft.aft is toy["aft"] and only_aft.b2 is None
    only_b2 = AftB2Select.from_parts("b2", toy["aft"], toy["b2"], None, toy["names_all"])
    assert only_b2.aft is None and only_b2.b2 is toy["b2"]
    # b2-only bundle: importances come from the B2 booster
    assert only_b2.feature_importance().index.isin(toy["names_all"]).all()


# --------------------------------------------------------------------------- choice rule


def _score(vme_ok: bool, active_s: float, ea: float) -> dict:
    return {"vme_ok": vme_ok, "active_s": active_s, "ea": ea}


def test_choice_rule_safety_first_then_active_calls_then_ea_then_preference() -> None:
    # A candidate that fails the call-VME rule never wins on its active calls.
    assert choose_candidate({"aft": _score(False, 0.60, 0.9), "avg": _score(True, 0.10, 0.5), "b2": _score(True, 0.0, 0.9)}) == "avg"
    # Equal % active to 1 pp -> EA decides.
    assert choose_candidate({"aft": _score(True, 0.2512, 0.60), "avg": _score(True, 0.2549, 0.70), "b2": _score(True, 0.0, 0.99)}) == "avg"
    # Full tie -> AFT (incumbent) > avg > B2.
    tie = _score(True, 0.3, 0.8)
    assert choose_candidate({"b2": tie, "avg": tie, "aft": tie}) == "aft"
    assert choose_candidate({"b2": tie, "avg": tie}) == "avg"
    # No candidate passes: active calls count as 0, so EA decides.
    assert choose_candidate({"aft": _score(False, 0.5, 0.6), "avg": _score(False, 0.4, 0.7), "b2": _score(False, 0.0, 0.65)}) == "avg"
    assert select_key("aft", _score(False, 0.9, 0.5))[1] == 0.0
    with pytest.raises(ValueError):
        choose_candidate({})


def test_select_score_counts_calling_folds_only_and_ea_on_exact_rows() -> None:
    # 8 rows, folds 0 and 1. Fold 1 issues no active call, so its lab-R rows are not counted.
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    lo = np.array([1, 1, 4, 8, 1, 1, 4, 8], dtype=float)
    hi = 2 * lo
    hi[3] = np.inf  # right-censored: never an EA row
    pd_ = SimpleNamespace(folds=folds, lo=lo, hi=hi, lab_exact=np.array([True] * 8))
    lab = np.array(["S", "S", "R", "R", "S", "S", "R", "R"], dtype=object)
    calls = np.array(["likely_active", "uncertain", "likely_active", "uncertain",
                      "uncertain", "uncertain", "uncertain", "likely_inactive"], dtype=object)
    pred_mic = np.array([2, 8, 8, 16, 2, 2, 8, 16], dtype=float)
    idx = np.arange(8)
    s = select_score(calls, idx, pred_mic, pd_, lab, target=0.015)
    assert (s["n_vme"], s["n_lab_r_calling"], s["n_calling_folds"]) == (1, 2, 1)
    assert s["vme_ok"] is False  # (1 + 1) / (2 + 1) > 1.5 %
    assert s["active_s"] == pytest.approx(1 / 4)
    # exact rows: 0,1,2,4,5,6 (row 3 right-censored, row 7 hi=16 exact) -> 7 exact rows
    assert s["n_exact"] == 7
    ea_rows = [abs(np.log2(pred_mic[i]) - np.log2(hi[i])) <= 1 for i in (0, 1, 2, 4, 5, 6, 7)]
    assert s["ea"] == pytest.approx(np.mean(ea_rows))
    # No active call anywhere: the rule passes with 0 active (nothing to certify).
    quiet = select_score(np.array(["uncertain"] * 8, dtype=object), idx, pred_mic, pd_, lab, 0.015)
    assert quiet["vme_ok"] is True and quiet["active_s"] == 0.0
    # Uncallable pair (no breakpoint): lab unknown everywhere.
    none = select_score(np.array([None] * 8, dtype=object), idx, pred_mic, pd_, None, 0.015)
    assert none["vme_ok"] is True and none["n_lab_s"] == 0


def test_train_config_rejects_select_as_a_fitted_model_and_records_the_flag() -> None:
    with pytest.raises(ValueError, match="derived"):
        train.TrainConfig(models=("b2_xgb_steps", "aft_known", "aft_b2_select")).effective_models(False)
    assert train.TrainConfig().as_dict()["model_select"] is True
    assert train.TrainConfig(model_select=False).as_dict()["model_select"] is False


def test_gate_rules_pooled_ucb_and_per_fold_test() -> None:
    """``_gate_rules``: nothing to certify passes; otherwise pooled UCB over calling folds and no significant fold."""
    n = 400
    folds = np.repeat([0.0, 1.0], n // 2)
    pd_ = SimpleNamespace(folds=folds)
    lab = np.array(["R"] * n, dtype=object)
    idx = np.arange(n)
    quiet = np.array(["uncertain"] * n, dtype=object)
    ok, rec = train._gate_rules([(quiet, idx)], pd_, lab, folds, 0.015, 0.01)
    assert ok and rec["n_calling_folds"] == 0 and rec["ucb"] is None
    # 1 VME among 200 lab R in fold 0 only: UCB 2/201 < 1.5 %, fold not significant -> passes.
    calls = quiet.copy()
    calls[0] = "likely_active"
    ok, rec = train._gate_rules([(calls, idx)], pd_, lab, folds, 0.015, 0.01)
    assert ok and rec["n_vme"] == 1 and rec["n_lab_r_calling"] == 200 and rec["significant_folds"] == []
    # 12 VMEs in fold 0 (6 %): significant at p < 0.01 and the pooled UCB fails.
    calls[:12] = "likely_active"
    ok, rec = train._gate_rules([(calls, idx)], pd_, lab, folds, 0.015, 0.01)
    assert not ok and rec["significant_folds"] == [0]
    assert train._gate_rules([(calls, idx)], pd_, None, folds, 0.015, 0.01)[0] is True
