"""Training orchestrator: species x drug x model -> predictions table + model bundles.

Implements the ``train.py`` part of the models section in ``.context/DESIGN.md``:

1. Join labels (one species x drug pair at a time) with ``splits.parquet`` (split,
   fold, external_set, lolo_lineage), ``qc.parquet`` (QC pass only), the known-AMR
   table and, when the species has one, the unitig pattern matrix rows.
2. **CV**: for each fold ``f`` train on the other folds with *in-fold* feature
   selection (:func:`genome2mic.features.select.select_known` and
   :func:`~genome2mic.features.select.select_unitigs` see training rows only;
   for ``aft_known_unitig`` the unitig ranking target is the label residual after
   the known-AMR features, see :func:`_residual_target`), predict fold ``f`` ->
   out-of-fold predictions written with ``split = 'cv'``.
3. Conformal ``q`` per model from the OOF residuals on exact rows
   (:mod:`genome2mic.models.conformal`); the test set never calibrates anything.
4. Final fit on every train row (selection on all train rows) -> predictions for the
   ``test`` split (``external_set`` copied as an extra column). For every
   ``lolo_lineage`` of the pair: fit without that lineage, predict it,
   ``split = 'lolo_<lineage>'``.
5. ``pred_mic = round_up_to_step(2 ** pred_log2)`` (rule 9), ``band_low/high`` from
   ``q``, ``pred_sir`` from the call breakpoint, ``lab_sir`` = reported ``sir`` or,
   when absent, derived from the lab interval if unambiguous under the call breakpoint.
6. ``nearest_training_distance`` per prediction row: minimum Mash distance between
   the genome's sketch and the sketches of the genomes the model was fitted on
   (fold training genomes for CV rows, all train rows for test rows, the LOLO
   training set for LOLO rows). Lets the report plot accuracy by genetic distance.
7. ``b0_resfinder``: S/R calls parsed from ``data/interim/<gid>/resfinder/pheno_table.txt``
   (no MIC; ``pred_mic`` and bands null) for the same genomes.
8. Everything is stacked into ``results/preds_<SPECIES>_<drug>.parquet`` with a
   ``model`` column and ``run_id`` (sha1 of the training parameters + the splits
   file, 12 hex chars).
9. The final main model per pair (``aft_known_unitig``, or ``aft_known`` when the
   species has no unitig matrix) is saved as the bundle
   :mod:`genome2mic.predict.pipeline` loads::

       models/manifest.json                   model_version, run_id, created, species: {KPNEU: [drugs]}
       models/reference_sketches.npz          one sketch per species reference (ids = species keys)
       models/markers.fasta                   synthetic runs only (MarkerScan fallback)
       models/<SPECIES>/train_sketches.npz    sketches of the train-split genomes
       models/<SPECIES>/unitig_kmers.npz      copy of the frozen k-mer set (when unitigs exist)
       models/<SPECIES>/unitig_index.parquet  copy of the pattern index
       models/<SPECIES>/<drug>/model.ubj + params.json      (XgbAft.save)
       models/<SPECIES>/<drug>/features.json  model_class, known_columns, unitig_cols, class_by_column
       models/<SPECIES>/<drug>/conformal.json q, alpha, n_residuals
       models/<SPECIES>/<drug>/meta.json      free-form training record
       models/<SPECIES>/<drug>/importance.json [{feature, gain}, ...]

Feature matrices stay sparse (CSR) end to end; ``lineage_cluster`` is used only as
the ``groups=`` argument of the in-fold early-stopping holdout and for LOLO, never
as a column (every model also rejects it by name).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from genome2mic import sketch as sk
from genome2mic.config import Config
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.features import select
from genome2mic.io import read_parquet, write_parquet
from genome2mic.mic import GRID_MAX_EXPONENT, GRID_MIN_EXPONENT, round_up_to_step_array
from genome2mic.models import MODEL_CLASSES, make_model
from genome2mic.models.base import (
    label_point_log2_array,
    read_json,
    validate_intervals,
    write_json,
)
from genome2mic.models.conformal import DEFAULT_ALPHA, band, conformal_q, residual_steps, write_conformal
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE",
    "MODEL_VERSION",
    "TrainConfig",
    "PairData",
    "PairResult",
    "PREDS_COLUMNS",
    "compute_run_id",
    "derive_lab_sir",
    "finish_predictions",
    "load_pair",
    "parse_pheno_table",
    "resfinder_predictions",
    "train_pair",
    "write_shared_bundle",
    "run",
]

STAGE = "train"
MODEL_VERSION = "0.1.0"
SPLIT_CV = "cv"
SPLIT_TEST = "test"
SPLIT_TRAIN = "train"
LOLO_PREFIX = "lolo_"
MODEL_B0 = "b0_resfinder"
MODEL_B1 = "b1_lookup"
MODEL_B2 = "b2_xgb_steps"
MODEL_AFT_KNOWN = "aft_known"
MODEL_AFT_KNOWN_UNITIG = "aft_known_unitig"
MODEL_AFT_UNITIG_ONLY = "aft_unitig_only"
DEFAULT_MODELS: tuple[str, ...] = (MODEL_B1, MODEL_B2, MODEL_AFT_KNOWN, MODEL_AFT_KNOWN_UNITIG)
UNITIG_MODELS: frozenset[str] = frozenset({MODEL_AFT_KNOWN_UNITIG, MODEL_AFT_UNITIG_ONLY})
KNOWN_MODELS: frozenset[str] = frozenset({MODEL_B1, MODEL_B2, MODEL_AFT_KNOWN, MODEL_AFT_KNOWN_UNITIG})

PREDS_COLUMNS: tuple[str, ...] = (
    "genome_id",
    "species",
    "drug",
    "split",
    "pred_mic",
    "band_low",
    "band_high",
    "lab_lower",
    "lab_upper",
    "pred_sir",
    "lab_sir",
    "model",
    "run_id",
    "external_set",
    "nearest_training_distance",
)
"""Stage-10 contract columns plus the v0.2 additions ``external_set`` and ``nearest_training_distance``."""

PHENO_TABLE = Path("resfinder") / "pheno_table.txt"
SYNTHETIC_MARKER = "SYNTHETIC_DATA.md"
_FALLBACK_Q_STEPS = 2.0
"""Band half-width used when too few exact OOF residuals exist to certify coverage."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    """Knobs of the training stage (all recorded in ``meta.json`` and the run id).

    Attributes:
        models: Model ids to run (``b0_resfinder`` is added automatically when pheno
            tables exist; unitig models are skipped for species without a matrix).
        n_folds: Number of CV folds expected in ``splits.parquet``.
        min_count: Known-AMR rare-feature filter (training genomes per fold).
        top_k: Unitig columns kept per fold after the frequency window.
        min_freq, max_freq: Unitig training-frequency window.
        alpha: Conformal miscoverage (0.10 -> 90 % bands).
        nthread: xgboost threads per fit (pairs run sequentially).
        seed: Seed for the in-fold holdouts and xgboost sampling.
        max_rounds, early_stopping_rounds: Passed to the xgboost models.
        lolo: Run leave-one-lineage-out fits for the pair's ``lolo_lineage`` values.
        ablation: Also run ``aft_unitig_only``.
    """

    models: tuple[str, ...] = DEFAULT_MODELS
    n_folds: int = 5
    min_count: int = 5
    top_k: int = 2000
    min_freq: float = 0.01
    max_freq: float = 0.99
    alpha: float = DEFAULT_ALPHA
    nthread: int = 4
    seed: int = 7
    max_rounds: int = 400
    early_stopping_rounds: int = 20
    lolo: bool = True
    ablation: bool = False

    def effective_models(self, has_unitigs: bool) -> list[str]:
        out = [m for m in self.models if m in MODEL_CLASSES]
        unknown = [m for m in self.models if m not in MODEL_CLASSES and m != MODEL_B0]
        if unknown:
            raise ValueError(f"unknown model ids {unknown}; known: {sorted(MODEL_CLASSES)}")
        if self.ablation and MODEL_AFT_UNITIG_ONLY not in out:
            out.append(MODEL_AFT_UNITIG_ONLY)
        if not has_unitigs:
            out = [m for m in out if m not in UNITIG_MODELS]
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "n_folds": self.n_folds,
            "min_count": self.min_count,
            "top_k": self.top_k,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "alpha": self.alpha,
            "nthread": self.nthread,
            "seed": self.seed,
            "max_rounds": self.max_rounds,
            "early_stopping_rounds": self.early_stopping_rounds,
            "lolo": self.lolo,
            "ablation": self.ablation,
        }


def compute_run_id(train_config: TrainConfig, splits_path: Path) -> str:
    """``sha1(parameters JSON + splits.parquet bytes)`` truncated to 12 hex characters."""
    h = hashlib.sha1()
    h.update(json.dumps(train_config.as_dict(), sort_keys=True).encode("utf-8"))
    h.update(Path(splits_path).read_bytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@dataclass
class SpeciesData:
    """Per-species inputs shared by every drug of that species."""

    species: str
    known: pd.DataFrame
    """``known_amr.parquet`` rows of the species, indexed by ``genome_id``."""
    known_columns: list[str]
    class_by_column: dict[str, tuple[str | None, str | None]]
    sketch_ids: list[str]
    sketches: np.ndarray
    sketch_k: int
    distances: np.ndarray
    """Pairwise Mash distances between ``sketch_ids`` (symmetric, zero diagonal)."""
    unitigs: sp.csr_matrix | None
    unitig_rows: pd.DataFrame | None
    unitig_index: pd.DataFrame | None

    @property
    def has_unitigs(self) -> bool:
        return self.unitigs is not None

    def sketch_position(self, genome_ids: Sequence[str]) -> np.ndarray:
        pos = {g: i for i, g in enumerate(self.sketch_ids)}
        missing = [g for g in genome_ids if g not in pos]
        if missing:
            raise ContractViolation(
                f"{self.species}: {len(missing)} genome(s) have no sketch in sketches_{self.species}.npz: {missing[:5]}",
                STAGE,
            )
        return np.fromiter((pos[g] for g in genome_ids), dtype=np.int64, count=len(genome_ids))

    def unitig_position(self, genome_ids: Sequence[str]) -> np.ndarray:
        assert self.unitig_rows is not None
        pos = dict(zip(self.unitig_rows["genome_id"].astype(str), self.unitig_rows["row_index"].astype(int)))
        missing = [g for g in genome_ids if g not in pos]
        if missing:
            raise ContractViolation(
                f"{self.species}: {len(missing)} genome(s) missing from unitigs_{self.species}_rows.parquet: {missing[:5]}",
                STAGE,
            )
        return np.fromiter((pos[g] for g in genome_ids), dtype=np.int64, count=len(genome_ids))


@dataclass
class PairData:
    """Everything one species x drug pair needs for training, aligned by row."""

    species: str
    drug: str
    frame: pd.DataFrame
    """One row per labelled QC-passing genome: genome_id, split, fold, external_set,
    lolo_lineage, lineage_cluster, mic_lower, mic_upper, censor, sir."""
    X_known: pd.DataFrame
    """Known-AMR features aligned with ``frame`` (``genome_id``, ``species`` + feature columns)."""
    U: sp.csr_matrix | None
    """Unitig pattern rows aligned with ``frame`` (CSR int8) or ``None``."""
    lo: np.ndarray
    hi: np.ndarray
    y_point: np.ndarray
    sketch_pos: np.ndarray
    """Row position of each genome in ``SpeciesData.sketch_ids``."""

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def train_mask(self) -> np.ndarray:
        return (self.frame["split"].to_numpy(dtype=object) == SPLIT_TRAIN)

    @property
    def test_mask(self) -> np.ndarray:
        return (self.frame["split"].to_numpy(dtype=object) == SPLIT_TEST)

    @property
    def folds(self) -> np.ndarray:
        return _float_array(self.frame["fold"])

    @property
    def groups(self) -> np.ndarray:
        return self.frame["lineage_cluster"].to_numpy(dtype=object)


def _load_species_data(paths: Paths, species: str, known_all: pd.DataFrame, class_map: dict[str, tuple[str | None, str | None]]) -> SpeciesData:
    known = known_all.loc[known_all["species"] == species].copy()
    if known["genome_id"].duplicated().any():
        raise ContractViolation(f"{species}: duplicate genome_id in known_amr.parquet", STAGE)
    known_columns = select.known_feature_columns(known)
    known = known.set_index(known["genome_id"].astype(str), drop=False)

    ids, sketches, k = sk.load_sketches(paths.sketches(species))
    t0 = time.perf_counter()
    distances = sk.pairwise_distances(sketches, k=k)
    logger.info("%s: pairwise Mash distances for %d sketches in %.1fs", species, len(ids), time.perf_counter() - t0)

    unitigs = rows = index = None
    if paths.unitigs(species).is_file() and paths.unitig_rows(species).is_file():
        from genome2mic.features.unitigs import load_unitigs  # noqa: PLC0415

        unitigs, rows, index = load_unitigs(paths, species)
        logger.info("%s: unitig matrix %d rows x %d patterns", species, unitigs.shape[0], unitigs.shape[1])
    else:
        logger.warning("%s: no unitig matrix; unitig models are skipped for this species", species)
    return SpeciesData(
        species=species,
        known=known,
        known_columns=known_columns,
        class_by_column={c: class_map[c] for c in known_columns if c in class_map},
        sketch_ids=[str(g) for g in ids],
        sketches=sketches,
        sketch_k=k,
        distances=distances,
        unitigs=unitigs,
        unitig_rows=rows,
        unitig_index=index,
    )


def _class_map(paths: Paths) -> dict[str, tuple[str | None, str | None]]:
    """``column_name -> (class, subclass)`` from ``known_amr_columns.csv`` (empty if absent)."""
    if not paths.known_amr_columns.is_file():
        return {}
    table = pd.read_csv(paths.known_amr_columns, dtype=str, keep_default_na=False)
    out: dict[str, tuple[str | None, str | None]] = {}
    for _, row in table.iterrows():
        column = str(row.get("column_name", "")).strip()
        if not column:
            continue
        cls = str(row.get("class", "")).strip() or None
        sub = str(row.get("subclass", "")).strip() or None
        out[column] = (cls, sub)
    return out


def load_pair(
    species: str,
    drug: str,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    qc: pd.DataFrame,
    lineages: pd.DataFrame,
    sd: SpeciesData,
    droplog: DropLog,
) -> PairData:
    """Join the pair's labels with splits, QC, lineages, known-AMR and unitig rows.

    Every filter is counted in ``droplog``: no split row, QC fail / no QC row, no
    known-AMR row, no unitig row (only when the species has unitigs).
    """
    pair = labels.loc[(labels["species"] == species) & (labels["drug"] == drug)].copy()
    pair["genome_id"] = pair["genome_id"].astype(str)
    detail = f"{species} x {drug}"
    frame = pair.merge(
        splits[["genome_id", "split", "fold", "external_set", "lolo_lineage"]], on="genome_id", how="left"
    )
    frame = droplog.keep_where(frame, frame["split"].notna(), "label_without_split_row", detail)
    frame = frame.merge(qc[["genome_id", "qc_pass"]], on="genome_id", how="left")
    frame = droplog.keep_where(frame, frame["qc_pass"].notna(), "label_without_qc_row", detail)
    frame = droplog.keep_where(frame, frame["qc_pass"].astype(bool), "qc_fail", detail)
    frame = frame.merge(lineages[["genome_id", "lineage_cluster"]], on="genome_id", how="left")
    frame = droplog.keep_where(frame, frame["lineage_cluster"].notna(), "label_without_lineage_row", detail)
    has_known = frame["genome_id"].isin(sd.known.index)
    frame = droplog.keep_where(frame, has_known, "label_without_known_amr_row", detail)
    if sd.has_unitigs:
        assert sd.unitig_rows is not None
        has_u = frame["genome_id"].isin(set(sd.unitig_rows["genome_id"].astype(str)))
        frame = droplog.keep_where(frame, has_u, "label_without_unitig_row", detail)
    frame = frame.sort_values("genome_id", kind="stable").reset_index(drop=True)
    if frame["genome_id"].duplicated().any():
        raise ContractViolation(f"{detail}: duplicate genome_id after the join (labels PK broken)", STAGE)

    gids = frame["genome_id"].tolist()
    X_known = sd.known.loc[gids, ["genome_id", "species", *sd.known_columns]].reset_index(drop=True)
    U = None
    if sd.has_unitigs:
        assert sd.unitigs is not None
        U = sp.csr_matrix(sd.unitigs[sd.unitig_position(gids)], dtype=np.int8)
    lo, hi = validate_intervals(frame["mic_lower"].to_numpy(dtype=float), frame["mic_upper"].to_numpy(dtype=float))
    y_point = label_point_log2_array(lo, hi)
    logger.info(
        "%s: %d labelled QC-passing genomes (%d train, %d test), %d known-AMR columns%s",
        detail, len(frame), int((frame["split"] == SPLIT_TRAIN).sum()), int((frame["split"] == SPLIT_TEST).sum()),
        len(sd.known_columns), f", {U.shape[1]} unitig patterns" if U is not None else "",
    )
    return PairData(
        species=species, drug=drug, frame=frame, X_known=X_known, U=U, lo=lo, hi=hi, y_point=y_point,
        sketch_pos=sd.sketch_position(gids),
    )


# --------------------------------------------------------------------------- #
# Feature assembly (fold side)
# --------------------------------------------------------------------------- #


@dataclass
class FoldFeatures:
    """Feature selection outcome for one training set."""

    known_columns: list[str]
    unitig_cols: np.ndarray
    """Selected pattern column indices into ``PairData.U`` (empty when unused)."""

    def names(self, use_known: bool, use_unitigs: bool) -> list[str]:
        out: list[str] = []
        if use_known:
            out.extend(self.known_columns)
        if use_unitigs:
            out.extend(_pattern_name(int(c)) for c in self.unitig_cols)
        return out


def _pattern_name(col: int) -> str:
    return f"u_{col:06d}"


def _float_array(series: pd.Series) -> np.ndarray:
    """Nullable numeric column -> float64 array with NaN for missing."""
    return pd.to_numeric(series, errors="coerce").astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)


def _residual_target(pd_: PairData, train_idx: np.ndarray, known_cols: Sequence[str]) -> np.ndarray:
    """``y_point`` minus its least-squares fit on the fold's known-AMR features (training rows only).

    Ranking unitigs by correlation with this residual selects patterns that explain
    what the curated markers cannot, instead of the thousands of k-mer patterns that
    merely duplicate a known marker's presence vector (those dominate a marginal
    |corr| ranking and crowd out novel mechanisms). It is the fold-side stand-in
    for a conditional association test; nothing outside ``train_idx`` is used.
    A tiny ridge term keeps the solve stable when columns are collinear.
    """
    y = pd_.y_point[train_idx]
    if not known_cols:
        return y
    X = pd_.X_known.iloc[train_idx][list(known_cols)].to_numpy(dtype=np.float64)
    A = np.column_stack([np.ones(len(train_idx)), X])
    ridge = 1e-6 * np.eye(A.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + ridge, A.T @ y)
    return y - A @ beta


def _select_features(
    pd_: PairData,
    train_idx: np.ndarray,
    cfg: TrainConfig,
    droplog: DropLog,
    need_unitigs: bool,
    residualize: bool = True,
) -> FoldFeatures:
    """Fold-side selection: known-AMR rare filter, then unitig frequency window + top-k ranking.

    With ``residualize`` (the ``aft_known_unitig`` path) the unitig ranking target is
    :func:`_residual_target`; without it (``aft_unitig_only``) it is the raw label point.
    """
    known_cols = select.select_known(pd_.X_known, train_idx, min_count=cfg.min_count, droplog=droplog)
    unitig_cols = np.empty(0, dtype=np.int64)
    if need_unitigs and pd_.U is not None:
        target = _residual_target(pd_, train_idx, known_cols) if residualize else pd_.y_point[train_idx]
        unitig_cols = select.select_unitigs(
            pd_.U, train_idx, target, min_freq=cfg.min_freq, max_freq=cfg.max_freq, top_k=cfg.top_k, droplog=droplog
        )
    return FoldFeatures(known_columns=known_cols, unitig_cols=unitig_cols)


def _select_all(pd_: PairData, train_idx: np.ndarray, cfg: TrainConfig, droplog: DropLog, models: Sequence[str], has_unitigs: bool) -> dict[str, FoldFeatures]:
    """One feature selection per feature set, shared by the models that use it."""
    known = _select_features(pd_, train_idx, cfg, droplog, need_unitigs=False)
    out: dict[str, FoldFeatures] = {}
    cache: dict[bool, FoldFeatures] = {}
    for m in models:
        if m in UNITIG_MODELS and has_unitigs:
            residualize = m != MODEL_AFT_UNITIG_ONLY
            if residualize not in cache:
                cache[residualize] = _select_features(pd_, train_idx, cfg, droplog, need_unitigs=True, residualize=residualize)
            out[m] = cache[residualize]
        else:
            out[m] = known
    return out


def _matrix(pd_: PairData, feats: FoldFeatures, use_known: bool, use_unitigs: bool) -> sp.csr_matrix:
    """Sparse float32 CSR of the selected features for every row of the pair."""
    parts: list[sp.csr_matrix] = []
    if use_known:
        if feats.known_columns:
            dense = pd_.X_known[feats.known_columns].to_numpy(dtype=np.float32)
            parts.append(sp.csr_matrix(dense))
        else:
            parts.append(sp.csr_matrix((pd_.n, 0), dtype=np.float32))
    if use_unitigs:
        assert pd_.U is not None
        parts.append(sp.csr_matrix(pd_.U[:, feats.unitig_cols], dtype=np.float32))
    if not parts:
        return sp.csr_matrix((pd_.n, 0), dtype=np.float32)
    if len(parts) == 1:
        return sp.csr_matrix(parts[0], dtype=np.float32)
    return sp.csr_matrix(sp.hstack(parts, format="csr"), dtype=np.float32)


def _uses(model_id: str) -> tuple[bool, bool]:
    """``(use_known, use_unitigs)`` for a model id."""
    return model_id in KNOWN_MODELS, model_id in UNITIG_MODELS


def _new_model(model_id: str, cfg: TrainConfig) -> Any:
    if model_id == MODEL_B1:
        return make_model(MODEL_B1)
    if model_id == MODEL_B2:
        return make_model(MODEL_B2, seed=cfg.seed, nthread=cfg.nthread, max_rounds=cfg.max_rounds,
                          early_stopping_rounds=cfg.early_stopping_rounds)
    cls = MODEL_CLASSES[MODEL_AFT_KNOWN]
    return cls(name=model_id, seed=cfg.seed, nthread=cfg.nthread, max_rounds=cfg.max_rounds,
               early_stopping_rounds=cfg.early_stopping_rounds)


def _fit(model: Any, X: sp.csr_matrix, lo: np.ndarray, hi: np.ndarray, names: list[str], groups: np.ndarray | None, droplog: DropLog) -> Any:
    """Fit a model, passing ``groups``/``droplog`` only to the models that accept them."""
    if model.name == MODEL_B1:
        return model.fit(X, lo, hi, names)
    if model.name == MODEL_B2:
        return model.fit(X, lo, hi, names, groups=groups, droplog=droplog)
    return model.fit(X, lo, hi, names, groups=groups)


def _fit_predict(
    pd_: PairData,
    model_id: str,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    cfg: TrainConfig,
    droplog: DropLog,
    feats: FoldFeatures | None = None,
) -> tuple[np.ndarray, Any, FoldFeatures, list[str]]:
    """Select features on ``train_idx``, fit ``model_id`` and predict ``predict_idx``.

    Returns ``(pred_log2 for predict_idx, fitted model, features, feature names)``.
    """
    use_known, use_unitigs = _uses(model_id)
    if feats is None:
        feats = _select_features(pd_, train_idx, cfg, droplog, need_unitigs=use_unitigs)
    names = feats.names(use_known, use_unitigs)
    X = _matrix(pd_, feats, use_known, use_unitigs)
    if X.shape[1] == 0:
        raise ValueError(f"{pd_.species} x {pd_.drug} [{model_id}]: no features survived selection")
    model = _new_model(model_id, cfg)
    _fit(model, X[train_idx], pd_.lo[train_idx], pd_.hi[train_idx], names, pd_.groups[train_idx], droplog)
    pred = np.asarray(model.predict_log2(X[predict_idx]), dtype=np.float64) if predict_idx.size else np.empty(0)
    return pred, model, feats, names


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


def derive_lab_sir(sir: Any, lo: float, hi: float, bp: Any) -> str | None:
    """Reported ``sir`` when present; else the category the interval implies, else null.

    Under the call breakpoint ``bp`` (S if MIC <= s, R if MIC > r): an interval
    ``(lo, hi]`` is S when ``hi <= s``, R when ``lo >= r`` and I when
    ``s <= lo`` and ``hi <= r``; anything straddling a breakpoint is ambiguous.
    """
    if isinstance(sir, str) and sir in ("S", "I", "R"):
        return sir
    if bp is None:
        return None
    s, r = float(bp.s_breakpoint), float(bp.r_breakpoint)
    if hi <= s + 1e-9:
        return "S"
    if lo >= r - 1e-9:
        return "R"
    if lo >= s - 1e-9 and hi <= r + 1e-9:
        return "I"
    return None


def _pred_sir(pred_mic: np.ndarray, config: Config, species: str, drug: str) -> list[str | None]:
    bp = config.call_breakpoint(species, drug)
    if bp is None:
        logger.warning("%s x %s: no %s breakpoint; pred_sir is null", species, drug, config.call_standard)
        return [None] * len(pred_mic)
    return [config.sir_from_mic(float(m), bp) if np.isfinite(m) else None for m in pred_mic]


def finish_predictions(
    pd_: PairData,
    idx: np.ndarray,
    pred_log2: np.ndarray,
    q: float,
    split: str,
    model_id: str,
    run_id: str,
    nearest: np.ndarray,
    config: Config,
) -> pd.DataFrame:
    """Round up, band, classify and assemble the preds rows for ``idx``."""
    clipped = np.clip(pred_log2, GRID_MIN_EXPONENT, GRID_MAX_EXPONENT)
    n_clip = int((clipped != pred_log2).sum())
    if n_clip:
        logger.warning("%s x %s [%s/%s]: %d prediction(s) clipped to the MIC grid edges", pd_.species, pd_.drug, model_id, split, n_clip)
    pred_mic = round_up_to_step_array(2.0**clipped)
    band_low, band_high = band(pred_mic, q)
    bp = config.call_breakpoint(pd_.species, pd_.drug)
    sub = pd_.frame.iloc[idx]
    lab_sir = [derive_lab_sir(s, float(lo), float(hi), bp) for s, lo, hi in zip(sub["sir"], pd_.lo[idx], pd_.hi[idx])]
    frame = pd.DataFrame(
        {
            "genome_id": sub["genome_id"].to_numpy(dtype=object),
            "species": pd_.species,
            "drug": pd_.drug,
            "split": split,
            "pred_mic": pred_mic,
            "band_low": band_low,
            "band_high": band_high,
            "lab_lower": pd_.lo[idx],
            "lab_upper": pd_.hi[idx],
            "pred_sir": _pred_sir(pred_mic, config, pd_.species, pd_.drug),
            "lab_sir": lab_sir,
            "model": model_id,
            "run_id": run_id,
            "external_set": sub["external_set"].to_numpy(dtype=object) if split == SPLIT_TEST else None,
            "nearest_training_distance": np.asarray(nearest, dtype=np.float64),
        },
        columns=list(PREDS_COLUMNS),
    )
    return frame


def _nearest(sd: SpeciesData, pd_: PairData, predict_idx: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Min Mash distance from each predicted genome to the genomes the model was fitted on."""
    if train_idx.size == 0 or predict_idx.size == 0:
        return np.full(predict_idx.size, np.nan)
    block = sd.distances[np.ix_(pd_.sketch_pos[predict_idx], pd_.sketch_pos[train_idx])]
    return block.min(axis=1)


def _conformal(pred_log2_oof: np.ndarray, pd_: PairData, cfg: TrainConfig, droplog: DropLog, label: str) -> tuple[float, int]:
    """Conformal ``q`` from the OOF predictions of the train rows (exact rows only)."""
    clipped = np.clip(pred_log2_oof, GRID_MIN_EXPONENT, GRID_MAX_EXPONENT)
    pred_mic = round_up_to_step_array(2.0**clipped)
    mask = ~np.isnan(pred_mic)
    residuals = residual_steps(np.log2(pred_mic[mask]), pd_.lo[mask], pd_.hi[mask], droplog)
    if residuals.size == 0:
        logger.warning("%s: no exact OOF residuals; band half-width falls back to %.1f steps", label, _FALLBACK_Q_STEPS)
        return _FALLBACK_Q_STEPS, 0
    q = conformal_q(residuals, alpha=cfg.alpha)
    if math.isinf(q):
        q = max(_FALLBACK_Q_STEPS, float(residuals.max()))
        logger.warning(
            "%s: only %d exact OOF residuals cannot certify %.0f%% coverage; using finite fallback q=%.1f steps "
            "(the bundle requires a finite q)",
            label, residuals.size, 100 * (1 - cfg.alpha), q,
        )
    return float(q), int(residuals.size)


# --------------------------------------------------------------------------- #
# ResFinder baseline (b0)
# --------------------------------------------------------------------------- #


def parse_pheno_table(path: Path, config: Config) -> dict[str, str]:
    """``drug -> 'R' | 'S'`` from a ResFinder 4 ``pheno_table.txt``.

    ``WGS-predicted phenotype`` ``Resistant`` -> R; ``No resistance`` -> S. Drug
    names are normalized through ``drugs.yaml`` (ResFinder's ``piperacillin+tazobactam``
    becomes ``piperacillin-tazobactam``); unknown drugs are skipped.
    """
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        drug = config.normalize_drug(parts[0].strip().replace("+", "-"))
        if drug is None:
            continue
        pheno = parts[2].strip().lower()
        if pheno.startswith("resistant"):
            out[drug] = "R"
        elif pheno.startswith("no resistance"):
            out[drug] = "S"
    return out


def resfinder_predictions(paths: Paths, pd_: PairData, run_id: str, config: Config, droplog: DropLog, cache: dict[str, dict[str, str] | None]) -> pd.DataFrame | None:
    """``b0_resfinder`` rows (``pred_mic``/bands null) for the pair's train (as cv) and test genomes."""
    calls: list[str | None] = []
    for gid in pd_.frame["genome_id"]:
        if gid not in cache:
            table = paths.interim_dir(gid) / PHENO_TABLE
            cache[gid] = parse_pheno_table(table, config) if table.is_file() else None
        parsed = cache[gid]
        calls.append(None if parsed is None else parsed.get(pd_.drug))
    present = np.array([c is not None for c in calls], dtype=bool)
    droplog.drop("resfinder_pheno_table_missing_or_no_drug_row", int((~present).sum()), f"{pd_.species} x {pd_.drug}")
    if not present.any():
        return None
    bp = config.call_breakpoint(pd_.species, pd_.drug)
    split = np.where(pd_.train_mask, SPLIT_CV, np.where(pd_.test_mask, SPLIT_TEST, None))
    keep = present & (split != None)  # noqa: E711  (object array comparison)
    idx = np.flatnonzero(keep)
    sub = pd_.frame.iloc[idx]
    frame = pd.DataFrame(
        {
            "genome_id": sub["genome_id"].to_numpy(dtype=object),
            "species": pd_.species,
            "drug": pd_.drug,
            "split": split[idx],
            "pred_mic": np.nan,
            "band_low": np.nan,
            "band_high": np.nan,
            "lab_lower": pd_.lo[idx],
            "lab_upper": pd_.hi[idx],
            "pred_sir": [calls[i] for i in idx],
            "lab_sir": [derive_lab_sir(s, float(lo), float(hi), bp) for s, lo, hi in zip(sub["sir"], pd_.lo[idx], pd_.hi[idx])],
            "model": MODEL_B0,
            "run_id": run_id,
            "external_set": np.where(split[idx] == SPLIT_TEST, sub["external_set"].to_numpy(dtype=object), None),
            "nearest_training_distance": np.nan,
        },
        columns=list(PREDS_COLUMNS),
    )
    return frame


# --------------------------------------------------------------------------- #
# Per-pair training
# --------------------------------------------------------------------------- #


@dataclass
class PairResult:
    """Outputs of :func:`train_pair`."""

    species: str
    drug: str
    preds: pd.DataFrame
    main_model: str
    q_by_model: dict[str, float]
    n_residuals_by_model: dict[str, int]
    timings: dict[str, float] = field(default_factory=dict)


def _lolo_lineages(pd_: PairData) -> list[str]:
    values = pd_.frame["lolo_lineage"].dropna().astype(str).unique().tolist()
    return sorted(values)


def train_pair(
    paths: Paths,
    config: Config,
    cfg: TrainConfig,
    sd: SpeciesData,
    pd_: PairData,
    run_id: str,
    droplog: DropLog,
    pheno_cache: dict[str, dict[str, str] | None],
) -> PairResult:
    """CV, conformal calibration, final fit, LOLO and bundle for one species x drug pair."""
    label = f"{pd_.species} x {pd_.drug}"
    models = cfg.effective_models(sd.has_unitigs)
    train_all = np.flatnonzero(pd_.train_mask)
    test_idx = np.flatnonzero(pd_.test_mask)
    folds = pd_.folds
    fold_values = sorted({int(f) for f in folds[train_all] if not np.isnan(f)})
    if not fold_values:
        raise ContractViolation(f"{label}: train rows have no fold assignment", STAGE)
    if len(train_all) < 10:
        raise ValueError(f"{label}: only {len(train_all)} training rows; pairs_kept should have excluded this pair")

    preds: list[pd.DataFrame] = []
    q_by_model: dict[str, float] = {}
    n_res: dict[str, int] = {}
    timings: dict[str, float] = {}

    # --- CV: out-of-fold predictions, in-fold selection ------------------------
    oof: dict[str, np.ndarray] = {m: np.full(pd_.n, np.nan) for m in models}
    oof_nearest = np.full(pd_.n, np.nan)
    t0 = time.perf_counter()
    for f in fold_values:
        val_idx = np.flatnonzero(pd_.train_mask & (folds == f))
        fit_idx = np.flatnonzero(pd_.train_mask & (folds != f) & ~np.isnan(folds))
        if val_idx.size == 0 or fit_idx.size == 0:
            logger.warning("%s: fold %d has %d validation / %d fit rows; skipped", label, f, val_idx.size, fit_idx.size)
            continue
        oof_nearest[val_idx] = _nearest(sd, pd_, val_idx, fit_idx)
        selected = _select_all(pd_, fit_idx, cfg, droplog, models, sd.has_unitigs)
        for m in models:
            try:
                pred, _, _, _ = _fit_predict(pd_, m, fit_idx, val_idx, cfg, droplog, feats=selected[m])
            except ValueError as error:
                logger.warning("%s [%s] fold %d: %s; fold predictions are null", label, m, f, error)
                continue
            oof[m][val_idx] = pred
        logger.info("%s: fold %d done (%d fit, %d validation rows)", label, f, fit_idx.size, val_idx.size)
    timings["cv_s"] = time.perf_counter() - t0

    for m in models:
        q, n = _conformal(oof[m], pd_, cfg, droplog, f"{label} [{m}]")  # NaN outside train rows
        q_by_model[m] = q
        n_res[m] = n
        has = train_all[~np.isnan(oof[m][train_all])]
        if has.size:
            preds.append(finish_predictions(pd_, has, oof[m][has], q, SPLIT_CV, m, run_id, oof_nearest[has], config))
        logger.info("%s [%s]: conformal q=%.2f steps from %d exact OOF residuals", label, m, q, n)

    # --- Final fit on all train rows -> test (+ external) -------------------------
    t0 = time.perf_counter()
    main_model = MODEL_AFT_KNOWN_UNITIG if MODEL_AFT_KNOWN_UNITIG in models else MODEL_AFT_KNOWN
    selected = _select_all(pd_, train_all, cfg, droplog, models, sd.has_unitigs)
    test_nearest = _nearest(sd, pd_, test_idx, train_all)
    for m in models:
        pred, model, feats_used, names = _fit_predict(pd_, m, train_all, test_idx, cfg, droplog, feats=selected[m])
        if test_idx.size:
            preds.append(finish_predictions(pd_, test_idx, pred, q_by_model[m], SPLIT_TEST, m, run_id, test_nearest, config))
        if m == main_model:
            _write_pair_bundle(paths, config, sd, pd_, cfg, model, feats_used, names, q_by_model[m], n_res[m], run_id, train_all.size, test_idx.size)
    timings["final_s"] = time.perf_counter() - t0

    # --- LOLO -------------------------------------------------------------------
    t0 = time.perf_counter()
    if cfg.lolo:
        groups = pd_.groups
        for lineage in _lolo_lineages(pd_):
            in_lineage = groups == lineage
            fit_idx = np.flatnonzero(pd_.train_mask & ~in_lineage)
            predict_idx = np.flatnonzero(in_lineage)
            if fit_idx.size < 10 or predict_idx.size == 0:
                logger.warning("%s: LOLO %s skipped (%d fit rows, %d predict rows)", label, lineage, fit_idx.size, predict_idx.size)
                continue
            nearest = _nearest(sd, pd_, predict_idx, fit_idx)
            selected = _select_all(pd_, fit_idx, cfg, droplog, models, sd.has_unitigs)
            for m in models:
                try:
                    pred, _, _, _ = _fit_predict(pd_, m, fit_idx, predict_idx, cfg, droplog, feats=selected[m])
                except ValueError as error:
                    logger.warning("%s [%s] LOLO %s: %s; skipped", label, m, lineage, error)
                    continue
                preds.append(finish_predictions(pd_, predict_idx, pred, q_by_model[m], f"{LOLO_PREFIX}{lineage}", m, run_id, nearest, config))
            logger.info("%s: LOLO %s done (%d held-out genomes)", label, lineage, predict_idx.size)
    timings["lolo_s"] = time.perf_counter() - t0

    # --- b0_resfinder -----------------------------------------------------------
    b0 = resfinder_predictions(paths, pd_, run_id, config, droplog, pheno_cache)
    if b0 is not None:
        preds.append(b0)

    table = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=list(PREDS_COLUMNS))
    table = _typed_preds(table)
    _check_preds(table, pd_, label)
    write_parquet(table, paths.preds(pd_.species, pd_.drug))
    logger.info("%s: wrote %d prediction rows (%d models) to %s", label, len(table), table["model"].nunique(), paths.preds(pd_.species, pd_.drug))
    return PairResult(pd_.species, pd_.drug, table, main_model, q_by_model, n_res, timings)


def _typed_preds(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ("pred_mic", "band_low", "band_high", "lab_lower", "lab_upper", "nearest_training_distance"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("genome_id", "species", "drug", "split", "pred_sir", "lab_sir", "model", "run_id", "external_set"):
        out[col] = out[col].astype(object).where(out[col].notna(), None).astype("string")
    return out[list(PREDS_COLUMNS)]


def _check_preds(table: pd.DataFrame, pd_: PairData, label: str) -> None:
    """Stage-10 acceptance: unique (genome_id, model, split), bounds, rounding, splits."""
    if table.empty:
        raise ContractViolation(f"{label}: no predictions produced", STAGE)
    dup = table.duplicated(["genome_id", "model", "split"])
    if dup.any():
        raise ContractViolation(f"{label}: {int(dup.sum())} duplicate (genome_id, model, split) rows", STAGE)
    pm = table["pred_mic"].to_numpy(dtype=float)
    has = ~np.isnan(pm)
    if (pm[has] <= 0).any():
        raise ContractViolation(f"{label}: non-positive pred_mic", STAGE)
    steps = np.log2(pm[has])
    if np.abs(steps - np.rint(steps)).max(initial=0.0) > 1e-9:
        raise ContractViolation(f"{label}: pred_mic not on the doubling grid", STAGE)
    bl, bh = table["band_low"].to_numpy(dtype=float), table["band_high"].to_numpy(dtype=float)
    ok = has & ~np.isnan(bl)
    if ((bl[ok] > pm[ok]) | (bh[ok] < pm[ok])).any():
        raise ContractViolation(f"{label}: band does not contain pred_mic", STAGE)
    split_of = dict(zip(pd_.frame["genome_id"], pd_.frame["split"]))
    for eval_split, want in ((SPLIT_CV, SPLIT_TRAIN), (SPLIT_TEST, SPLIT_TEST)):
        rows = table.loc[table["split"] == eval_split, "genome_id"]
        bad = [g for g in rows if split_of.get(g) != want]
        if bad:
            raise ContractViolation(f"{label}: {len(bad)} {eval_split} rows belong to non-{want} genomes", STAGE)


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #


def _write_pair_bundle(
    paths: Paths,
    config: Config,
    sd: SpeciesData,
    pd_: PairData,
    cfg: TrainConfig,
    model: Any,
    feats: FoldFeatures,
    names: list[str],
    q: float,
    n_residuals: int,
    run_id: str,
    n_train: int,
    n_test: int,
) -> None:
    """Write ``models/<SPECIES>/<drug>/`` for the prediction pipeline."""
    target = paths.model_dir(pd_.species, pd_.drug)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    model.save(target)
    use_known, use_unitigs = _uses(model.name)
    known_columns = list(feats.known_columns) if use_known else []
    unitig_cols = [_pattern_name(int(c)) for c in feats.unitig_cols] if use_unitigs else []
    select.assert_no_forbidden(names, require_prefix=True)
    write_json(
        target / "features.json",
        {
            "model_class": model.name,
            "known_columns": known_columns,
            "unitig_cols": unitig_cols,
            "class_by_column": {c: list(sd.class_by_column[c]) for c in known_columns if c in sd.class_by_column},
            "feature_names": names,
        },
    )
    write_conformal(target / "conformal.json", q, cfg.alpha, n_residuals, {"unit": "doubling steps", "source": "out-of-fold exact residuals"})
    write_json(
        target / "meta.json",
        {
            "species": pd_.species,
            "drug": pd_.drug,
            "model": model.name,
            "run_id": run_id,
            "model_version": MODEL_VERSION,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "n_known_columns": len(known_columns),
            "n_unitig_cols": len(unitig_cols),
            "train_config": cfg.as_dict(),
            "call_standard": list(config.call_standard),
            "synthetic": (paths.raw_dir / SYNTHETIC_MARKER).is_file(),
        },
    )
    importance: list[dict[str, Any]] = []
    if hasattr(model, "feature_importance"):
        series = model.feature_importance("gain")
        importance = [{"feature": str(k), "gain": float(v)} for k, v in series.items() if float(v) > 0]
    (target / "importance.json").write_text(json.dumps(importance, indent=2) + "\n", encoding="utf-8")
    logger.info("%s x %s: bundle written to %s (%d known + %d unitig features, %d non-zero importances)",
                pd_.species, pd_.drug, target, len(known_columns), len(unitig_cols), len(importance))


def _synthetic_seed(paths: Paths) -> int:
    """Seed recorded in ``SYNTHETIC_DATA.md`` (default 7)."""
    text = (paths.raw_dir / SYNTHETIC_MARKER).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"seed\s*`?(\d+)`?", text)
    return int(match.group(1)) if match else 7


def write_shared_bundle(
    paths: Paths,
    config: Config,
    species_data: Mapping[str, SpeciesData],
    splits: pd.DataFrame,
    trained: Mapping[str, list[str]],
    run_id: str,
) -> None:
    """``manifest.json``, ``reference_sketches.npz``, per-species sketches and unitig copies, ``markers.fasta``."""
    models_dir = paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Reference sketches (species identification at prediction time).
    from genome2mic.qc import build_reference_sketches  # noqa: PLC0415

    refs = build_reference_sketches(paths, config)
    if refs is None:
        raise FileNotFoundError(
            f"no species reference genomes under {paths.references_dir} (or species.yaml reference_sketch); "
            "the prediction pipeline cannot identify species without models/reference_sketches.npz"
        )
    sk.save_sketches(models_dir / "reference_sketches.npz", list(refs.species), refs.sketches, k=refs.k)

    for species, sd in species_data.items():
        if species not in trained:
            continue
        sdir = models_dir / species
        sdir.mkdir(parents=True, exist_ok=True)
        train_ids = set(splits.loc[(splits["species"] == species) & (splits["split"] == SPLIT_TRAIN), "genome_id"].astype(str))
        keep = [i for i, g in enumerate(sd.sketch_ids) if g in train_ids]
        if not keep:
            raise ContractViolation(f"{species}: no training sketches to ship", STAGE)
        sk.save_sketches(sdir / "train_sketches.npz", [sd.sketch_ids[i] for i in keep], sd.sketches[keep], k=sd.sketch_k)
        if sd.has_unitigs:
            shutil.copyfile(paths.unitig_kmers(species), sdir / "unitig_kmers.npz")
            shutil.copyfile(paths.unitig_index(species), sdir / "unitig_index.parquet")
        logger.info("%s: shipped %d training sketches%s", species, len(keep), " + unitig k-mer set" if sd.has_unitigs else "")

    markers = models_dir / "markers.fasta"
    if (paths.raw_dir / SYNTHETIC_MARKER).is_file():
        from genome2mic.io import write_fasta  # noqa: PLC0415
        from genome2mic.synthetic import markers as mk  # noqa: PLC0415

        seqs = mk.marker_sequences(_synthetic_seed(paths))
        records = []
        for spec in mk.reportable_markers():
            header = (f"{spec.name} type={spec.marker_type} subtype={spec.subtype} class={spec.amr_class} "
                      f"subclass={spec.subclass} scope={spec.scope} name=\"{spec.element_name}\" synthetic=true")
            records.append((header, seqs[spec.name]))
        write_fasta(records, markers)
        logger.info("synthetic run: wrote %d marker sequences to %s (MarkerScan fallback)", len(records), markers)
    elif markers.is_file():
        logger.info("keeping existing %s", markers)

    manifest = {
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "species": {sp: sorted(drugs) for sp, drugs in trained.items()},
        "synthetic": (paths.raw_dir / SYNTHETIC_MARKER).is_file(),
        "note": "Predictions of in-vitro susceptibility, not prescribing advice.",
    }
    write_json(paths.models_manifest, manifest)
    logger.info("wrote %s: %s", paths.models_manifest, {k: len(v) for k, v in trained.items()})


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


def _read_inputs(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for required in (paths.labels, paths.splits, paths.qc, paths.known_amr, paths.lineages, paths.pairs_kept):
        if not required.is_file():
            raise FileNotFoundError(f"training input missing: {required}")
    labels = read_parquet(paths.labels)
    splits = read_parquet(paths.splits)
    qc = read_parquet(paths.qc, columns=["genome_id", "qc_pass"])
    known = read_parquet(paths.known_amr)
    lineages = read_parquet(paths.lineages, columns=["genome_id", "lineage_cluster"])
    pairs = pd.read_csv(paths.pairs_kept, dtype={"species": str, "drug": str})
    for frame in (labels, splits, qc, known, lineages):
        frame["genome_id"] = frame["genome_id"].astype(str)
    return labels, splits, qc, known, lineages, pairs


def run(
    paths: Paths,
    config: Config,
    *,
    train_config: TrainConfig | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Train every kept species x drug pair; write preds, bundles and ``drop_log_train.csv``.

    Args:
        paths: Project paths (reads the stage 2/3/5/6/7/8 files, writes ``results/``
            and ``models/``).
        config: Loaded project config.
        train_config: Training knobs (defaults: :class:`TrainConfig`).
        pairs: Optional subset of ``(species, drug)`` pairs; default = ``pairs_kept.csv``.

    Returns:
        One summary row per pair: species, drug, main_model, n_rows, q per model, timings.
    """
    cfg = train_config or TrainConfig()
    started = time.perf_counter()
    droplog = DropLog(STAGE)
    labels, splits, qc, known, lineages, pairs_kept = _read_inputs(paths)
    run_id = compute_run_id(cfg, paths.splits)
    logger.info("train: run_id=%s config=%s", run_id, cfg.as_dict())

    wanted = [(str(s), str(d)) for s, d in (pairs if pairs is not None else zip(pairs_kept["species"], pairs_kept["drug"]))]
    wanted = [(s, d) for s, d in wanted if s in config.species and config.normalize_drug(d) == d]
    if not wanted:
        raise ValueError("no species x drug pairs to train (pairs_kept.csv empty or out of scope)")

    class_map = _class_map(paths)
    species_data: dict[str, SpeciesData] = {}
    trained: dict[str, list[str]] = {}
    summaries: list[dict[str, Any]] = []
    pheno_cache: dict[str, dict[str, str] | None] = {}
    for species, drug in wanted:
        if species not in species_data:
            species_data[species] = _load_species_data(paths, species, known, class_map)
        sd = species_data[species]
        t0 = time.perf_counter()
        pd_ = load_pair(species, drug, labels, splits, qc, lineages, sd, droplog)
        result = train_pair(paths, config, cfg, sd, pd_, run_id, droplog, pheno_cache)
        trained.setdefault(species, []).append(drug)
        elapsed = time.perf_counter() - t0
        summaries.append(
            {
                "species": species,
                "drug": drug,
                "main_model": result.main_model,
                "n_rows": pd_.n,
                "n_train": int(pd_.train_mask.sum()),
                "n_test": int(pd_.test_mask.sum()),
                "n_preds": len(result.preds),
                "models": ",".join(sorted(result.preds["model"].unique())),
                "q_main": result.q_by_model.get(result.main_model),
                "seconds": round(elapsed, 1),
                **{f"q_{m}": q for m, q in result.q_by_model.items()},
            }
        )
        logger.info("%s x %s: trained in %.1fs (%s)", species, drug, elapsed, result.timings)

    write_shared_bundle(paths, config, species_data, splits, trained, run_id)
    droplog.write(paths.drop_log(STAGE))
    summary = pd.DataFrame(summaries)
    logger.info("train stage finished in %.1fs:\n%s", time.perf_counter() - started, summary.to_string(index=False))
    return summary


def load_manifest(paths: Paths) -> dict[str, Any]:
    """Read ``models/manifest.json``."""
    return read_json(paths.models_manifest)
