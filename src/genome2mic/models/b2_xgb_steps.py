"""Baseline B2: multi-class XGBoost over integer log2 MIC steps (exact MICs only).

The contract lists ``b2_xgb_steps`` as a multi-class model over the doubling steps:
``multi:softprob`` with one class per step observed among the training rows that
have an exact MIC (``censor == 'interval'``, i.e. ``lo > 0 and hi < inf``). Censored
rows are dropped (and counted through :class:`~genome2mic.droplog.DropLog`) because
a class label needs a single step. Prediction is the argmax step, not the expected
value, so the output is always a step the training data contained.

``n_rounds`` is chosen by early stopping (``mlogloss``) on a seeded 20% holdout of
the exact rows and the final booster is refitted on all exact rows. Inputs are
converted to CSR first (see :func:`genome2mic.models.base.to_csr`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from genome2mic.droplog import DropLog
from genome2mic.models.base import (
    FeatureMatrix,
    ModelBundleMixin,
    check_feature_names,
    exact_mask,
    holdout_split,
    read_json,
    take_rows,
    to_csr,
    validate_intervals,
    write_json,
)

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bynode": 0.5,
}

HOLDOUT_NAME = "holdout"
# How far log2(mic_upper) may sit from an integer before the row is rejected as off-grid.
_STEP_TOL = 1e-6


class B2XgbSteps(ModelBundleMixin):
    """``multi:softprob`` over the integer log2 steps seen in exact training rows.

    Args:
        params: Overrides merged on top of :data:`DEFAULT_PARAMS` (``num_class``,
            ``seed`` and ``nthread`` are set by the class).
        name: Model id for the predictions table.
        max_rounds, early_stopping_rounds, holdout_fraction, fallback_rounds,
        min_rounds: Early-stopping controls, as in :class:`~genome2mic.models.xgb_aft.XgbAft`.
        seed, nthread: Determinism controls.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        name: str = "b2_xgb_steps",
        max_rounds: int = 500,
        early_stopping_rounds: int = 20,
        holdout_fraction: float = 0.2,
        fallback_rounds: int = 100,
        min_rounds: int = 5,
        seed: int = 0,
        nthread: int = 1,
    ) -> None:
        self.name = name
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.params["seed"] = int(seed)
        self.params["nthread"] = int(nthread)
        self.max_rounds = int(max_rounds)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.holdout_fraction = float(holdout_fraction)
        self.fallback_rounds = int(fallback_rounds)
        self.min_rounds = int(min_rounds)
        self.seed = int(seed)
        self.nthread = int(nthread)

        # fitted state
        self.booster_: xgb.Booster | None = None
        self.feature_names_: list[str] | None = None
        self.classes_: np.ndarray | None = None  # integer log2 steps, sorted
        self.n_rounds_: int | None = None
        self.n_train_: int | None = None
        self.n_exact_: int | None = None
        self.best_score_: float | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: FeatureMatrix,
        lo: np.ndarray,
        hi: np.ndarray,
        feature_names: list[str],
        groups: np.ndarray | None = None,
        droplog: DropLog | None = None,
    ) -> "B2XgbSteps":
        """Fit on the exact rows of ``(lo, hi]``; censored rows are dropped and counted."""
        matrix = to_csr(X)
        low, high = validate_intervals(lo, hi, matrix.shape[0])
        names = check_feature_names(feature_names, matrix.shape[1])
        n = matrix.shape[0]

        log = droplog if droplog is not None else DropLog(self.name)
        exact = exact_mask(low, high)
        log.drop(
            "censored row (B2 trains on exact MICs only)",
            int(n - exact.sum()),
            detail=f"{int(exact.sum())} exact rows kept of {n}",
        )
        if not exact.any():
            raise ValueError("B2XgbSteps needs at least one exact (interval-censored) row")

        steps_float = np.log2(high[exact])
        steps = np.rint(steps_float)
        if np.max(np.abs(steps_float - steps)) > _STEP_TOL:
            raise ValueError("mic_upper of exact rows must sit on the doubling grid")
        steps = steps.astype(np.int64)
        classes, y = np.unique(steps, return_inverse=True)
        X_exact = take_rows(matrix, np.flatnonzero(exact))
        g_exact = None if groups is None else np.asarray(groups)[exact]
        n_exact = int(exact.sum())

        self.classes_ = classes
        self.feature_names_ = names
        self.n_train_ = int(n)
        self.n_exact_ = n_exact
        self.best_score_ = None

        if len(classes) == 1:
            logger.warning("[%s] only one MIC step (%d) among exact rows; constant model", self.name, classes[0])
            self.booster_ = None
            self.n_rounds_ = 0
            return self

        params = {**self.params, "num_class": int(len(classes))}
        fit_idx, hold_idx = holdout_split(n_exact, self.holdout_fraction, self.seed, g_exact)
        if hold_idx.size == 0:
            logger.warning("[%s] only %d exact rows: skipping early stopping, %d rounds",
                           self.name, n_exact, self.fallback_rounds)
            n_rounds = self.fallback_rounds
        else:
            d_fit = xgb.DMatrix(take_rows(X_exact, fit_idx), label=y[fit_idx], feature_names=names)
            d_hold = xgb.DMatrix(take_rows(X_exact, hold_idx), label=y[hold_idx], feature_names=names)
            booster = xgb.train(
                params,
                d_fit,
                num_boost_round=self.max_rounds,
                evals=[(d_hold, HOLDOUT_NAME)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            n_rounds = max(self.min_rounds, int(booster.best_iteration) + 1)
            self.best_score_ = float(booster.best_score)
            logger.info("[%s] holdout mlogloss=%.4f at iteration %d",
                        self.name, self.best_score_, booster.best_iteration)

        d_all = xgb.DMatrix(X_exact, label=y, feature_names=names)
        self.booster_ = xgb.train(params, d_all, num_boost_round=n_rounds)
        self.params = params
        self.n_rounds_ = int(n_rounds)
        logger.info("[%s] fitted on %d exact rows, %d classes (steps %d..%d), n_rounds=%d",
                    self.name, n_exact, len(classes), classes[0], classes[-1], n_rounds)
        return self

    # -------------------------------------------------------------- predict
    def predict_log2(self, X: FeatureMatrix) -> np.ndarray:
        """Argmax step per row as a float log2 MIC."""
        matrix = to_csr(self._check_predict_input(X))
        assert self.classes_ is not None and self.feature_names_ is not None
        n = matrix.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if self.booster_ is None:
            return np.full(n, float(self.classes_[0]), dtype=np.float64)
        proba = self.booster_.predict(xgb.DMatrix(matrix, feature_names=self.feature_names_))
        proba = np.asarray(proba).reshape(n, len(self.classes_))
        return self.classes_[np.argmax(proba, axis=1)].astype(np.float64)

    def predict_proba(self, X: FeatureMatrix) -> np.ndarray:
        """Class probabilities, shape ``(n_rows, n_classes)``, columns in ``classes_`` order."""
        matrix = to_csr(self._check_predict_input(X))
        assert self.classes_ is not None and self.feature_names_ is not None
        n = matrix.shape[0]
        if self.booster_ is None:
            return np.ones((n, 1), dtype=np.float64)
        proba = self.booster_.predict(xgb.DMatrix(matrix, feature_names=self.feature_names_))
        return np.asarray(proba, dtype=np.float64).reshape(n, len(self.classes_))

    # ------------------------------------------------------------ persistence
    def save(self, dir: Path) -> None:
        """Write ``params.json`` and, unless the model is constant, ``model.ubj``."""
        self._require_fitted()
        assert self.classes_ is not None
        target = Path(dir)
        target.mkdir(parents=True, exist_ok=True)
        if self.booster_ is not None:
            self.booster_.save_model(str(target / self.MODEL_FILE))
        write_json(
            target / self.PARAMS_FILE,
            {
                "model_class": type(self).__name__,
                "name": self.name,
                "params": self.params,
                "classes": [int(c) for c in self.classes_],
                "n_rounds": self.n_rounds_,
                "max_rounds": self.max_rounds,
                "early_stopping_rounds": self.early_stopping_rounds,
                "holdout_fraction": self.holdout_fraction,
                "fallback_rounds": self.fallback_rounds,
                "min_rounds": self.min_rounds,
                "seed": self.seed,
                "nthread": self.nthread,
                "feature_names": self.feature_names_,
                "n_train": self.n_train_,
                "n_exact": self.n_exact_,
                "best_score": self.best_score_,
                "constant": self.booster_ is None,
                "xgboost_version": xgb.__version__,
            },
        )
        logger.info("[%s] saved model to %s", self.name, target)

    @classmethod
    def load(cls, dir: Path) -> "B2XgbSteps":
        """Read a bundle written by :meth:`save`."""
        source = Path(dir)
        meta = read_json(source / cls.PARAMS_FILE)
        if meta.get("model_class") not in (None, cls.__name__):
            raise ValueError(f"{source} holds a {meta.get('model_class')}, not {cls.__name__}")
        params = dict(meta["params"])
        model = cls(
            params,
            name=meta.get("name", "b2_xgb_steps"),
            max_rounds=meta.get("max_rounds", 500),
            early_stopping_rounds=meta.get("early_stopping_rounds", 20),
            holdout_fraction=meta.get("holdout_fraction", 0.2),
            fallback_rounds=meta.get("fallback_rounds", 100),
            min_rounds=meta.get("min_rounds", 5),
            seed=meta.get("seed", params.get("seed", 0)),
            nthread=meta.get("nthread", params.get("nthread", 1)),
        )
        model.params = params
        model.classes_ = np.asarray(meta["classes"], dtype=np.int64)
        model.feature_names_ = [str(n) for n in meta["feature_names"]]
        model.n_rounds_ = meta.get("n_rounds")
        model.n_train_ = meta.get("n_train")
        model.n_exact_ = meta.get("n_exact")
        model.best_score_ = meta.get("best_score")
        if meta.get("constant", False):
            model.booster_ = None
        else:
            booster = xgb.Booster()
            booster.load_model(str(source / cls.MODEL_FILE))
            model.booster_ = booster
        return model
