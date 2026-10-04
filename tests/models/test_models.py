"""Tests for genome2mic.models: B1Lookup, B2XgbSteps, XgbAft and the shared base helpers.

The fixture is a synthetic known-AMR-like matrix (n=400) with two binary features
that add +6 and +3 log2 steps to a base of -4 with N(0, 0.5) noise. Lab readings are
the grid cell containing the true MIC; ~10% of rows are reported as ``<= reading``
(left-censored) and ~10% as ``> reading/2`` (right-censored), all via
``genome2mic.mic.interval_from_result`` so the intervals are exactly what the ingest
stage would produce. Synthetic data -- no metric here says anything about real genomes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from genome2mic import mic
from genome2mic.droplog import DropLog
from genome2mic.models import MODEL_CLASSES, B1Lookup, B2XgbSteps, MicModel, XgbAft, make_model
from genome2mic.models import base

BASE_LOG2 = -4.0
EFFECT_F1 = 6.0
EFFECT_F2 = 3.0
NOISE_SD = 0.5
N_ROWS = 400
FEATURE_NAMES = ["gene_blakpc_2", "point_gyra_s83l"]


@dataclass(frozen=True)
class Synthetic:
    """One synthetic species x drug training set."""

    X: sp.csr_matrix
    lo: np.ndarray
    hi: np.ndarray
    reading_log2: np.ndarray  # the lab's reported step (log2 of the grid cell top)
    censor: np.ndarray  # "interval" / "left" / "right"
    f1: np.ndarray
    f2: np.ndarray

    @property
    def exact(self) -> np.ndarray:
        return self.censor == "interval"

    @property
    def dense(self) -> np.ndarray:
        return self.X.toarray()


def make_synthetic(n: int = N_ROWS, seed: int = 11) -> Synthetic:
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 2, size=n)
    f2 = rng.integers(0, 2, size=n)
    true_log2 = BASE_LOG2 + EFFECT_F1 * f1 + EFFECT_F2 * f2 + rng.normal(0.0, NOISE_SD, size=n)
    reading = np.array([mic.round_up_to_step(2.0**v) for v in true_log2])
    u = rng.random(n)
    lo = np.empty(n)
    hi = np.empty(n)
    censor = np.empty(n, dtype=object)
    for i in range(n):
        if u[i] < 0.10:  # reported as "<= reading": true statement, left-censored
            lo[i], hi[i], censor[i] = mic.interval_from_result("<=", reading[i])
        elif u[i] < 0.20:  # reported as "> reading/2": true statement, right-censored
            lo[i], hi[i], censor[i] = mic.interval_from_result(">", reading[i] / 2.0)
        else:
            lo[i], hi[i], censor[i] = mic.interval_from_result("=", reading[i])
    X = sp.csr_matrix(np.column_stack([f1, f2]).astype(np.int8))
    return Synthetic(X, lo, hi, np.log2(reading), censor.astype(str), f1, f2)


@pytest.fixture(scope="module")
def data() -> Synthetic:
    return make_synthetic()


@pytest.fixture(scope="module")
def aft_fitted(data: Synthetic) -> XgbAft:
    return XgbAft(seed=0, nthread=1).fit(data.X, data.lo, data.hi, FEATURE_NAMES)


@pytest.fixture(scope="module")
def b2_fitted(data: Synthetic) -> B2XgbSteps:
    return B2XgbSteps(seed=0, nthread=1).fit(data.X, data.lo, data.hi, FEATURE_NAMES)


@pytest.fixture(scope="module")
def b1_fitted(data: Synthetic) -> B1Lookup:
    return B1Lookup().fit(data.X, data.lo, data.hi, FEATURE_NAMES)


def essential_agreement(pred_log2: np.ndarray, truth_log2: np.ndarray) -> float:
    """Share of rows within +-1 doubling step (on log2 steps)."""
    return float(np.mean(np.abs(pred_log2 - truth_log2) <= 1.0))


def group_means(pred: np.ndarray, d: Synthetic) -> dict[tuple[int, int], float]:
    return {
        (a, b): float(pred[(d.f1 == a) & (d.f2 == b)].mean())
        for a in (0, 1)
        for b in (0, 1)
    }


# ----------------------------------------------------------------- fixture sanity
class TestSyntheticFixture:
    def test_about_twenty_percent_censored_and_bounds_follow_the_contract(self, data: Synthetic) -> None:
        share = 1.0 - data.exact.mean()
        assert 0.12 < share < 0.28
        assert (data.censor == "left").sum() > 20
        assert (data.censor == "right").sum() > 20
        assert ((data.lo == 0) == (data.censor == "left")).all()
        assert (np.isinf(data.hi) == (data.censor == "right")).all()
        assert (data.lo < data.hi).all()

    def test_label_point_matches_mic_module_elementwise(self, data: Synthetic) -> None:
        got = base.label_point_log2_array(data.lo, data.hi)
        want = np.array([mic.label_point_log2(a, b) for a, b in zip(data.lo, data.hi, strict=True)])
        np.testing.assert_array_equal(got, want)
        # with this censoring scheme the label point recovers the reported step exactly
        np.testing.assert_array_equal(got, data.reading_log2)


# -------------------------------------------------------------------- base helpers
class TestBaseHelpers:
    def test_check_feature_names_rejects_forbidden_and_malformed(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            base.check_feature_names(["gene_a", "lineage_cluster"], 2)
        with pytest.raises(ValueError, match="forbidden"):
            base.check_feature_names(["gene_a", "ST"], 2)
        with pytest.raises(ValueError, match="duplicate"):
            base.check_feature_names(["gene_a", "gene_a"], 2)
        with pytest.raises(ValueError, match="columns"):
            base.check_feature_names(["gene_a"], 2)
        with pytest.raises(ValueError, match="may not contain"):
            base.check_feature_names(["gene_a<b"], 1)
        assert base.check_feature_names(["gene_a", "u_000001"], 2) == ["gene_a", "u_000001"]

    @pytest.mark.parametrize(
        "forbidden",
        ["lineage_cluster", "st", "country", "year", "source", "isolation_source", "biosample", "split", "fold"],
    )
    def test_every_contract_metadata_column_is_forbidden(self, forbidden: str) -> None:
        assert forbidden in base.FORBIDDEN_FEATURES

    def test_validate_intervals_rules(self) -> None:
        lo, hi = base.validate_intervals([0.0, 4.0, 32.0], [0.25, 8.0, math.inf])
        assert lo.dtype == np.float64 and hi.dtype == np.float64
        with pytest.raises(ValueError, match="nulls"):
            base.validate_intervals([np.nan], [1.0])
        with pytest.raises(ValueError, match=">= 0"):
            base.validate_intervals([-1.0], [1.0])
        with pytest.raises(ValueError, match="mic_lower < mic_upper"):
            base.validate_intervals([2.0], [2.0])
        with pytest.raises(ValueError, match="no information"):
            base.validate_intervals([0.0], [math.inf])
        with pytest.raises(ValueError, match="rows"):
            base.validate_intervals([0.0], [1.0], n_rows=3)

    def test_exact_mask(self) -> None:
        mask = base.exact_mask(np.array([0.0, 4.0, 32.0]), np.array([0.25, 8.0, math.inf]))
        assert mask.tolist() == [False, True, False]

    def test_holdout_split_is_seeded_and_disjoint(self) -> None:
        fit_a, hold_a = base.holdout_split(100, 0.2, seed=3)
        fit_b, hold_b = base.holdout_split(100, 0.2, seed=3)
        np.testing.assert_array_equal(hold_a, hold_b)
        assert len(hold_a) == 20 and len(fit_a) == 80
        assert not set(fit_a) & set(hold_a)
        _, hold_c = base.holdout_split(100, 0.2, seed=4)
        assert not np.array_equal(hold_a, hold_c)

    def test_holdout_split_keeps_groups_together_and_degrades_for_tiny_n(self) -> None:
        groups = np.repeat(np.arange(10), 10)
        fit, hold = base.holdout_split(100, 0.2, seed=0, groups=groups)
        assert len(hold) >= 20
        assert not set(groups[fit]) & set(groups[hold])
        fit, hold = base.holdout_split(6, 0.2, seed=0)
        assert hold.size == 0 and len(fit) == 6

    def test_as_matrix_keeps_sparse_sparse(self) -> None:
        coo = sp.coo_matrix(np.eye(3, dtype=np.int8))
        out = base.as_matrix(coo)
        assert sp.isspmatrix_csr(out)
        dense = base.as_matrix([[1, 0], [0, 1]])
        assert isinstance(dense, np.ndarray) and dense.shape == (2, 2)
        with pytest.raises(ValueError):
            base.as_matrix(np.zeros(3))


# -------------------------------------------------------------------------- registry
class TestRegistry:
    def test_model_classes_contents(self) -> None:
        assert MODEL_CLASSES == {
            "b1_lookup": B1Lookup,
            "b2_xgb_steps": B2XgbSteps,
            "aft_known": XgbAft,
            "aft_known_unitig": XgbAft,
        }

    def test_make_model_sets_name_and_unknown_id_raises(self) -> None:
        m = make_model("aft_known_unitig", seed=1)
        assert isinstance(m, XgbAft) and m.name == "aft_known_unitig" and m.seed == 1
        with pytest.raises(KeyError):
            make_model("nope")

    def test_every_model_satisfies_the_protocol(self) -> None:
        for cls in {B1Lookup, B2XgbSteps, XgbAft}:
            instance = cls()
            assert isinstance(instance, MicModel)
            for method in ("fit", "predict_log2", "save", "load"):
                assert callable(getattr(instance, method))


# ------------------------------------------------------------------------- XgbAft
class TestXgbAft:
    def test_recovers_the_planted_effects(self, aft_fitted: XgbAft, data: Synthetic) -> None:
        pred = aft_fitted.predict_log2(data.X)
        assert pred.shape == (N_ROWS,) and pred.dtype == np.float64 and np.isfinite(pred).all()
        means = group_means(pred, data)
        assert means[(1, 0)] - means[(0, 0)] == pytest.approx(EFFECT_F1, abs=1.0)
        assert means[(0, 1)] - means[(0, 0)] == pytest.approx(EFFECT_F2, abs=1.0)
        assert means[(1, 1)] - means[(0, 0)] == pytest.approx(EFFECT_F1 + EFFECT_F2, abs=1.0)

    def test_mean_abs_residual_on_exact_rows_below_one_step(self, aft_fitted: XgbAft, data: Synthetic) -> None:
        pred = aft_fitted.predict_log2(data.X)
        exact = data.exact
        raw_resid = np.abs(pred[exact] - np.log2(data.hi[exact]))
        assert raw_resid.mean() < 1.0
        rounded_up = np.log2(mic.round_up_to_step_array(2.0 ** pred[exact]))
        assert np.abs(rounded_up - np.log2(data.hi[exact])).mean() < 1.0
        assert essential_agreement(rounded_up, np.log2(data.hi[exact])) > 0.85

    def test_scale_tuned_over_grid_and_rounds_from_early_stopping(self, aft_fitted: XgbAft) -> None:
        assert aft_fitted.scale_ in XgbAft(seed=0).scales
        assert [r["scale"] for r in aft_fitted.tuning_] == [0.5, 1.0, 1.5]
        assert all(math.isfinite(r["best_score"]) for r in aft_fitted.tuning_)
        assert aft_fitted.n_rounds_ == max(aft_fitted.min_rounds, min(aft_fitted.tuning_, key=lambda r: r["best_score"])["best_iteration"] + 1)
        assert aft_fitted.n_holdout_ == round(0.2 * N_ROWS)
        assert aft_fitted.params["objective"] == "survival:aft"
        assert aft_fitted.params["aft_loss_distribution_scale"] == aft_fitted.scale_
        assert aft_fitted.params["seed"] == 0 and aft_fitted.params["nthread"] == 1

    def test_deterministic_given_seed(self, data: Synthetic) -> None:
        a = XgbAft(seed=5, nthread=1, max_rounds=100).fit(data.X, data.lo, data.hi, FEATURE_NAMES)
        b = XgbAft(seed=5, nthread=1, max_rounds=100).fit(data.X, data.lo, data.hi, FEATURE_NAMES)
        np.testing.assert_array_equal(a.predict_log2(data.X), b.predict_log2(data.X))
        assert a.tuning_ == b.tuning_

    def test_save_load_round_trip_is_exact(self, aft_fitted: XgbAft, data: Synthetic, tmp_path: Path) -> None:
        aft_fitted.save(tmp_path / "aft")
        assert (tmp_path / "aft" / "model.ubj").exists()
        assert (tmp_path / "aft" / "params.json").exists()
        loaded = XgbAft.load(tmp_path / "aft")
        np.testing.assert_array_equal(loaded.predict_log2(data.X), aft_fitted.predict_log2(data.X))
        assert loaded.feature_names_ == FEATURE_NAMES
        assert loaded.scale_ == aft_fitted.scale_ and loaded.n_rounds_ == aft_fitted.n_rounds_
        assert loaded.name == aft_fitted.name
        meta = base.read_json(tmp_path / "aft" / "params.json")
        assert meta["params"]["objective"] == "survival:aft" and meta["feature_names"] == FEATURE_NAMES

    def test_feature_importance_maps_to_names(self, aft_fitted: XgbAft) -> None:
        imp = aft_fitted.feature_importance("gain")
        assert set(imp.index) == set(FEATURE_NAMES)
        assert (imp >= 0).all()
        assert imp.iloc[0] >= imp.iloc[-1]
        # the +6 step feature carries more gain than the +3 one
        assert imp["gene_blakpc_2"] > imp["point_gyra_s83l"]

    def test_handles_zero_lower_and_inf_upper_bounds(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        X = rng.integers(0, 2, size=(n, 2)).astype(np.int8)
        lo = np.where(X[:, 0] == 1, 32.0, 0.0)
        hi = np.where(X[:, 0] == 1, math.inf, 0.25)
        model = XgbAft(seed=0, max_rounds=60).fit(X, lo, hi, ["gene_a", "gene_b"])
        pred = model.predict_log2(X)
        assert np.isfinite(pred).all()
        # right-censored rows predict above left-censored ones
        assert pred[X[:, 0] == 1].mean() > pred[X[:, 0] == 0].mean()

    def test_dense_and_sparse_inputs_give_identical_models_and_predictions(self, data: Synthetic) -> None:
        # XGBoost treats CSR structural zeros as missing and dense zeros as 0; the models
        # canonicalize to CSR so the caller's representation cannot change the answer.
        a = XgbAft(seed=1, max_rounds=80).fit(data.X, data.lo, data.hi, FEATURE_NAMES)
        b = XgbAft(seed=1, max_rounds=80).fit(data.dense, data.lo, data.hi, FEATURE_NAMES)
        np.testing.assert_array_equal(a.predict_log2(data.X), b.predict_log2(data.dense))
        np.testing.assert_array_equal(a.predict_log2(data.dense), a.predict_log2(data.X))

    def test_tiny_data_falls_back_to_fixed_rounds(self, caplog: pytest.LogCaptureFixture) -> None:
        X = np.array([[0], [1], [0], [1], [1], [0]], dtype=np.int8)
        lo, hi = np.array([0.5, 4.0, 0.5, 4.0, 8.0, 0.25]), np.array([1.0, 8.0, 1.0, 8.0, 16.0, 0.5])
        with caplog.at_level(logging.WARNING, logger="genome2mic.models.xgb_aft"):
            m = XgbAft(seed=0, fallback_rounds=20).fit(X, lo, hi, ["gene_a"])
        assert m.n_rounds_ == 20 and m.tuning_ == [] and m.n_holdout_ == 0
        assert any("skipping early stopping" in r.getMessage() for r in caplog.records)
        assert m.predict_log2(X).shape == (6,)

    def test_rejects_forbidden_features_and_bad_shapes(self, data: Synthetic) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            XgbAft().fit(data.X, data.lo, data.hi, ["gene_a", "year"])
        with pytest.raises(ValueError, match="columns"):
            XgbAft().fit(data.X, data.lo, data.hi, ["gene_a"])
        small = XgbAft(max_rounds=20).fit(data.X, data.lo, data.hi, FEATURE_NAMES)
        with pytest.raises(ValueError, match="columns"):
            small.predict_log2(np.zeros((3, 3)))
        with pytest.raises(RuntimeError, match="not fitted"):
            XgbAft().predict_log2(data.X)

    def test_groups_keep_holdout_whole(self, data: Synthetic) -> None:
        groups = np.repeat(np.arange(20), N_ROWS // 20)
        m = XgbAft(seed=0, max_rounds=40).fit(data.X, data.lo, data.hi, FEATURE_NAMES, groups=groups)
        assert m.n_holdout_ >= 0.2 * N_ROWS and m.n_holdout_ % (N_ROWS // 20) == 0


# --------------------------------------------------------------------- B2XgbSteps
class TestB2XgbSteps:
    def test_trains_on_exact_rows_only_and_logs_the_drop(self, data: Synthetic) -> None:
        log = DropLog("b2_test")
        m = B2XgbSteps(seed=0).fit(data.X, data.lo, data.hi, FEATURE_NAMES, droplog=log)
        assert m.n_exact_ == int(data.exact.sum())
        assert log.total() == int((~data.exact).sum())
        assert "censored" in log.records[0].reason

    def test_exact_rows_mask_drops_one_step_rows_that_are_not_mics(self, data: Synthetic) -> None:
        """train passes lab_exact: a one-step disk-diffusion interval is not an exact MIC."""
        not_mic = np.zeros(N_ROWS, dtype=bool)
        not_mic[np.flatnonzero(data.exact)[:7]] = True  # e.g. disk I-only rows with a one-step I range
        log = DropLog("b2_test")
        m = B2XgbSteps(seed=0).fit(data.X, data.lo, data.hi, FEATURE_NAMES, droplog=log, exact_rows=~not_mic)
        assert m.n_exact_ == int(data.exact.sum()) - 7
        reasons = {r.reason: r.n_dropped for r in log.records}
        assert reasons["censored row (B2 trains on exact MICs only)"] == int((~data.exact).sum())
        assert [n for reason, n in reasons.items() if "not a measured MIC" in reason] == [7]
        with pytest.raises(ValueError, match="exact_rows"):
            B2XgbSteps(seed=0).fit(data.X, data.lo, data.hi, FEATURE_NAMES, exact_rows=np.ones(3, dtype=bool))

    def test_classes_are_integer_steps_from_exact_rows(self, b2_fitted: B2XgbSteps, data: Synthetic) -> None:
        want = np.unique(np.rint(np.log2(data.hi[data.exact])).astype(int))
        np.testing.assert_array_equal(b2_fitted.classes_, want)

    def test_predicts_argmax_step_with_reasonable_ea(self, b2_fitted: B2XgbSteps, data: Synthetic) -> None:
        pred = b2_fitted.predict_log2(data.X)
        assert pred.dtype == np.float64
        assert set(pred.tolist()) <= set(b2_fitted.classes_.astype(float).tolist())
        assert np.all(pred == np.rint(pred))
        exact = data.exact
        assert essential_agreement(pred[exact], np.log2(data.hi[exact])) >= 0.85
        means = group_means(pred, data)
        assert means[(1, 0)] - means[(0, 0)] == pytest.approx(EFFECT_F1, abs=1.0)
        assert means[(0, 1)] - means[(0, 0)] == pytest.approx(EFFECT_F2, abs=1.0)
        proba = b2_fitted.predict_proba(data.X)
        assert proba.shape == (N_ROWS, len(b2_fitted.classes_))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load_round_trip_is_exact(self, b2_fitted: B2XgbSteps, data: Synthetic, tmp_path: Path) -> None:
        b2_fitted.save(tmp_path / "b2")
        assert (tmp_path / "b2" / "model.ubj").exists()
        loaded = B2XgbSteps.load(tmp_path / "b2")
        np.testing.assert_array_equal(loaded.predict_log2(data.X), b2_fitted.predict_log2(data.X))
        np.testing.assert_array_equal(loaded.classes_, b2_fitted.classes_)
        assert loaded.feature_names_ == FEATURE_NAMES and loaded.params["num_class"] == len(b2_fitted.classes_)

    def test_single_step_becomes_a_constant_model(self, tmp_path: Path) -> None:
        X = np.array([[0], [1], [1], [0]], dtype=np.int8)
        lo, hi = np.array([4.0, 4.0, 4.0, 0.0]), np.array([8.0, 8.0, 8.0, 2.0])
        m = B2XgbSteps().fit(X, lo, hi, ["gene_a"])
        np.testing.assert_array_equal(m.predict_log2(X), [3.0, 3.0, 3.0, 3.0])
        m.save(tmp_path / "const")
        assert not (tmp_path / "const" / "model.ubj").exists()
        loaded = B2XgbSteps.load(tmp_path / "const")
        np.testing.assert_array_equal(loaded.predict_log2(X), [3.0, 3.0, 3.0, 3.0])

    def test_no_exact_rows_raises(self) -> None:
        X = np.array([[0], [1]], dtype=np.int8)
        with pytest.raises(ValueError, match="exact"):
            B2XgbSteps().fit(X, np.array([0.0, 32.0]), np.array([1.0, math.inf]), ["gene_a"])

    def test_deterministic_given_seed(self, data: Synthetic) -> None:
        a = B2XgbSteps(seed=2).fit(data.X, data.lo, data.hi, FEATURE_NAMES).predict_log2(data.X)
        b = B2XgbSteps(seed=2).fit(data.X, data.lo, data.hi, FEATURE_NAMES).predict_log2(data.X)
        np.testing.assert_array_equal(a, b)

    def test_dense_and_sparse_inputs_give_identical_predictions(self, b2_fitted: B2XgbSteps, data: Synthetic) -> None:
        dense_fit = B2XgbSteps(seed=0, nthread=1).fit(data.dense, data.lo, data.hi, FEATURE_NAMES)
        np.testing.assert_array_equal(dense_fit.predict_log2(data.dense), b2_fitted.predict_log2(data.X))
        np.testing.assert_array_equal(b2_fitted.predict_log2(data.dense), b2_fitted.predict_log2(data.X))


# ----------------------------------------------------------------------- B1Lookup
class TestB1Lookup:
    def test_profile_medians_and_global_fallback(self, b1_fitted: B1Lookup, data: Synthetic) -> None:
        assert len(b1_fitted.table_) == 4
        y = base.label_point_log2_array(data.lo, data.hi)
        for key in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            rows = (data.f1 == key[0]) & (data.f2 == key[1])
            assert b1_fitted.table_[key] == pytest.approx(float(np.median(y[rows])))
            assert b1_fitted.counts_[key] == int(rows.sum())
        assert b1_fitted.global_median_ == pytest.approx(float(np.median(y)))

    def test_reasonable_ea_on_exact_rows(self, b1_fitted: B1Lookup, data: Synthetic) -> None:
        pred = b1_fitted.predict_log2(data.X)
        exact = data.exact
        assert essential_agreement(pred[exact], np.log2(data.hi[exact])) >= 0.85
        means = group_means(pred, data)
        assert means[(1, 0)] - means[(0, 0)] == pytest.approx(EFFECT_F1, abs=1.0)

    def test_unseen_profile_uses_global_median(self, data: Synthetic) -> None:
        # Fit without the (1, 1) profile so it is unseen at prediction time.
        keep = ~((data.f1 == 1) & (data.f2 == 1))
        m = B1Lookup().fit(data.X[np.flatnonzero(keep)], data.lo[keep], data.hi[keep], FEATURE_NAMES)
        pred = m.predict_log2(np.array([[1, 1]], dtype=np.int8))
        assert pred.tolist() == [m.global_median_]

    def test_int_and_float_encodings_share_a_key(self, b1_fitted: B1Lookup) -> None:
        a = b1_fitted.predict_log2(np.array([[1, 0]], dtype=np.int8))
        b = b1_fitted.predict_log2(np.array([[1.0, 0.0]]))
        c = b1_fitted.predict_log2(sp.csr_matrix(np.array([[1, 0]], dtype=np.int8)))
        assert a.tolist() == b.tolist() == c.tolist() == [b1_fitted.table_[(1, 0)]]

    def test_save_load_round_trip_is_exact(self, b1_fitted: B1Lookup, data: Synthetic, tmp_path: Path) -> None:
        b1_fitted.save(tmp_path / "b1")
        assert (tmp_path / "b1" / "lookup.json").exists()
        loaded = B1Lookup.load(tmp_path / "b1")
        np.testing.assert_array_equal(loaded.predict_log2(data.X), b1_fitted.predict_log2(data.X))
        assert loaded.table_ == b1_fitted.table_ and loaded.counts_ == b1_fitted.counts_
        assert loaded.global_median_ == b1_fitted.global_median_
        assert loaded.feature_names_ == FEATURE_NAMES

    def test_refuses_to_densify_a_wide_matrix(self) -> None:
        wide = sp.csr_matrix((5, B1Lookup.MAX_COLUMNS + 1), dtype=np.int8)
        names = [f"u_{i:06d}" for i in range(wide.shape[1])]
        with pytest.raises(ValueError, match="densify"):
            B1Lookup().fit(wide, np.full(5, 4.0), np.full(5, 8.0), names)

    def test_rejects_forbidden_feature(self, data: Synthetic) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            B1Lookup().fit(data.X, data.lo, data.hi, ["gene_a", "country"])
