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
3. Conformal band per model from the OOF residuals on exact rows
   (:mod:`genome2mic.models.conformal`); the test set never calibrates anything.
   Default (``TrainConfig.band = "asym_tuned"``): asymmetric band whose upper level is
   chosen inside the calibration folds for call-level VME <= 1.5 %, with an
   active-call gate (:func:`_band_params`); CV rows use bands tuned and calibrated on
   the other folds only.
   "Exact" is ``lab_exact`` (:func:`genome2mic.mic.lab_exact_mask`): a one-step lab
   interval whose ``method`` is not ``disk`` -- a disk-diffusion ``I``-only row with
   a one-step ``I`` range (CLSI meropenem ``(1, 2]``) is not a measured MIC. The
   same mask selects B2's training rows and is written to the preds as ``lab_exact``.
4. Final fit on every train row (selection on all train rows) -> predictions for the
   ``test`` split (``external_set`` copied as an extra column). For every
   ``lolo_lineage`` of the pair (LOLO clusters are reserved as *train* clusters by
   :mod:`genome2mic.splits.make_splits`): fit on the train rows outside the lineage,
   predict the lineage's train rows, ``split = 'lolo_<lineage>'``. A lineage with no
   train rows for the pair (its fit set would equal the full train set, i.e. the
   test fit) or too few remaining fit rows is skipped, logged and counted.
5. ``pred_mic = round_up_to_step(2 ** pred_log2)`` (rule 9), ``band_low/high`` from
   ``q``, ``pred_sir`` from the call breakpoint, ``lab_sir`` = reported ``sir`` or,
   when absent, derived from the lab interval if unambiguous under the call breakpoint
   (as reported: the lab's own standard and year). ``lab_sir_rederived`` = the lab
   interval classified under the call breakpoint (:func:`rederive_lab_sir`), so
   categorical metrics can also be computed on one breakpoint table.
6. ``nearest_training_distance`` per prediction row: minimum Mash distance between
   the genome's sketch and the sketches of the genomes the model was fitted on
   (fold training genomes for CV rows, all train rows for test rows, the LOLO
   training set for LOLO rows). Lets the report plot accuracy by genetic distance.
   Computed exactly, once per species for all its drugs, block-wise over the query
   genomes (:func:`_species_nearest`); no n x n matrix is ever held.
7. ``b0_resfinder``: S/R calls parsed from ``data/interim/<gid>/resfinder/pheno_table.txt``
   (no MIC; ``pred_mic`` and bands null) for the same genomes.
8. Everything is stacked into ``results/preds_<SPECIES>_<drug>.parquet`` with a
   ``model`` column and ``run_id`` (sha1 of the training parameters + the splits
   file, 12 hex chars). Each write of test predictions appends one row to the
   append-only ``results/test_ledger.csv`` (``run_id, created_utc, species, drug,
   n_test_rows, inputs_sha1``) so the "test set touched once" rule can be checked.
   ``inputs_sha1`` (:func:`compute_inputs_sha1`) fingerprints what ``run_id`` does
   not: labels, known-AMR features, QC, lineages, the species' unitig files, the
   drug and breakpoint configs and the model / feature / MIC source code.
9. The final main model per pair (``aft_known_unitig``, or ``aft_known`` when the
   species has no unitig matrix) is saved as the bundle
   :mod:`genome2mic.predict.pipeline` loads::

       models/manifest.json                   model_version, run_id, run_ids, created, species: {KPNEU: [drugs]}
       models/reference_sketches.npz          one sketch per species reference (ids = species keys)
       models/markers.fasta                   synthetic runs only (MarkerScan fallback); removed otherwise
       models/<SPECIES>/train_sketches.npz    sketches of the train-split genomes
       models/<SPECIES>/unitig_kmers.npz      copy of the frozen k-mer set (when unitigs exist)
       models/<SPECIES>/unitig_index.parquet  copy of the pattern index
       models/<SPECIES>/<drug>/params.json + aft/ + b2/  (AftB2Select.save, v0.6: the in-fold choice of the
                                              AFT model, B2 or their average; model.ubj + params.json
                                              (XgbAft.save) when TrainConfig.model_select is off)
       models/<SPECIES>/<drug>/features.json  model_class, known_columns, unitig_cols, class_by_column,
                                              feature_names, unitig_kmer_set_sha1
       models/<SPECIES>/<drug>/conformal.json q, alpha, n_residuals, caps, q_up, q_low, alpha_up,
                                              alpha_low, active_gate_open (+ per-fold record,
                                              model_select: choice, per-fold choices and scores)
       models/<SPECIES>/<drug>/calibration.json  P(works) map (v0.6)
       models/<SPECIES>/<drug>/meta.json      free-form training record (incl. inputs_sha1)
       models/<SPECIES>/<drug>/importance.json [{feature, gain}, ...]

Feature matrices stay sparse (CSR) end to end; ``lineage_cluster`` is used only as
the ``groups=`` argument of the in-fold early-stopping holdout and for LOLO, never
as a column (every model also rejects it by name).

Sharding: ``run(pairs=...)`` (CLI ``train --species ... --drugs ...``) trains a
subset. A subset run merges its pairs into an existing ``models/manifest.json``
(``run_ids`` records which run produced each pair) and keeps the other pairs' rows
of ``drop_log_train.csv`` (every train drop-log ``detail`` starts with
``"<SPECIES> x <drug>"``); a run covering every kept pair overwrites both.
``models/<SPECIES>/unitig_kmers.npz`` is shared by every drug of the species, and each
unitig model records the set it was trained on (``features.json``
``unitig_kmer_set_sha1``, :meth:`genome2mic.features.unitigs.KmerSet.sha1`). A subset
run refuses to start (:class:`ContractViolation`) when it would replace that set with
a different one while drugs it does not retrain stay in the manifest -- their unitig
columns would silently index the new set.
Species are processed one at a time and their sketches/unitig matrices released
before the next species is loaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import time
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
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
from genome2mic.mic import GRID_MAX_EXPONENT, GRID_MIN_EXPONENT, lab_exact_mask, panel_caps_log2, round_up_to_step_array
from genome2mic.models import MODEL_CLASSES, make_model
from genome2mic.models import aft_b2_select as model_select
from genome2mic.models.base import (
    exact_mask,
    label_point_log2_array,
    read_json,
    validate_intervals,
    write_json,
)
from genome2mic.models import calibration as prob_cal
from genome2mic.models.conformal import (
    DEFAULT_ALPHA,
    DEFAULT_ALPHA_GRID,
    DEFAULT_ALPHA_LOW,
    DEFAULT_FOLD_P_THRESHOLD,
    DEFAULT_VME_TARGET,
    BandParams,
    asym_quantiles,
    calling_fold_vme,
    conformal_q,
    fold_call_vme_table,
    gate_calls,
    passing_levels,
    residual_steps,
    robust_q_low,
    robust_q_up,
    signed_residual_steps,
    vme_certified,
    symmetric_params,
    tune_band,
    write_conformal,
)
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE",
    "MODEL_VERSION",
    "TrainConfig",
    "PairData",
    "PairResult",
    "PREDS_COLUMNS",
    "LEDGER_COLUMNS",
    "FitGroup",
    "FitPlan",
    "PairNearest",
    "SpeciesData",
    "append_test_ledger",
    "check_subset_unitig_sets",
    "compute_inputs_sha1",
    "compute_run_id",
    "input_files",
    "derive_lab_sir",
    "rederive_lab_sir",
    "finish_predictions",
    "load_pair",
    "parse_pheno_table",
    "resfinder_predictions",
    "select_pairs",
    "train_pair",
    "write_shared_bundle",
    "write_species_bundle",
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
MODEL_SELECT = model_select.MODEL_ID
"""``aft_b2_select`` (v0.6): per pair, the AFT model, B2 or their average, chosen inside the
training folds (:func:`_select_model`). Derived from the fitted AFT and B2 models; never in
``TrainConfig.models``."""
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
    "lab_sir_rederived",
    "lab_exact",
    "call",
    "prob_works",
    "prob_tier",
)
"""Stage-10 contract columns plus the v0.2 additions ``external_set``,
``nearest_training_distance``, ``lab_sir_rederived`` and ``lab_exact`` (bool: the lab
result is an exact measured MIC, :func:`genome2mic.mic.lab_exact_mask`), the v0.4
``call`` (``likely_active`` / ``uncertain`` / ``likely_inactive`` or null; the
prediction pipeline's call rule and overrides, :func:`genome2mic.predict.rank.call_array`)
and the v0.6 ``prob_works`` (calibrated P(lab S under the call breakpoint); CV rows
cross-fitted on the other folds, :mod:`genome2mic.models.calibration`) and ``prob_tier``."""

LEDGER_COLUMNS: tuple[str, ...] = ("run_id", "created_utc", "species", "drug", "n_test_rows", "inputs_sha1")
"""Header of the append-only ``results/test_ledger.csv``."""
LEGACY_LEDGER_COLUMNS: tuple[str, ...] = LEDGER_COLUMNS[:5]
"""Header of ledgers written before ``inputs_sha1`` existed (upgraded in place on the next append)."""

INPUT_SOURCE_PATHS: tuple[str, ...] = ("models", "features", "mic.py")
"""Source under ``src/genome2mic/`` fingerprinted in ``inputs_sha1``: model code, feature
code and the MIC grid / rounding / exact-MIC rules that shape every prediction row."""
BUNDLE_UNITIG_KMERS = "unitig_kmers.npz"
"""Species-level copy of the frozen k-mer set in ``models/<SPECIES>/``."""

LOLO_SKIP_NO_TRAIN_ROWS = "lolo_lineage_without_train_rows"
"""Drop reason: the LOLO lineage has no train rows for the pair, so its fit set would
equal the full train set (the test fit) -- nothing would be left out."""
LOLO_SKIP_TOO_FEW_FIT_ROWS = "lolo_too_few_fit_rows"
"""Drop reason: fewer than ``_MIN_FIT_ROWS`` train rows remain outside the lineage."""
_MIN_FIT_ROWS = 10

PHENO_TABLE = Path("resfinder") / "pheno_table.txt"
SYNTHETIC_MARKER = "SYNTHETIC_DATA.md"
_FALLBACK_Q_STEPS = 2.0
BAND_ASYM_TUNED = "asym_tuned"
BAND_SYMMETRIC = "symmetric"
BAND_MODES = (BAND_ASYM_TUNED, BAND_SYMMETRIC)
"""Band half-width used when too few exact OOF residuals exist to certify coverage."""
_NEAREST_QUERY_BLOCK = 128
"""Query genomes per distance block in :func:`_species_nearest` (memory: block x n_ref float64)."""
_NEAREST_REF_BLOCK = 512
"""Reference rows compared per inner step of :class:`_MashBlocks` (memory only)."""


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
        nthread: xgboost threads per fit.
        exact_weight: Sample weight of exact-MIC training rows (``lab_exact``) relative
            to censored / S/I/R-only / disk rows in the AFT models (lever L7; 1.0 = off).
            Every row still trains; nothing is imputed. B1 and B2 are unaffected.
        band: ``"asym_tuned"`` (default): asymmetric cross-conformal band whose upper
            miscoverage level is chosen per pair from ``band_alpha_grid`` (largest first)
            as the first level whose nested cross-conformal call-level VME inside the
            training folds passes ``(n_vme + 1) / (n_lab_R + 1) <= band_vme_target``, both
            counts pooled only over the folds that issue ``likely_active`` calls; the
            bundle may only use a level that also passed in every CV fold that issues
            calls. Lower level fixed at ``band_alpha_low`` (widened by
            :func:`~genome2mic.models.conformal.robust_q_low` when leave-one-fold-out
            values disagree); if no level passes, ``likely_active`` calls are withheld
            (gate closed). ``"symmetric"``: the original ``+-q`` band at
            ``alpha`` with no gate.
        workers: Pairs of one species trained concurrently in spawned worker processes
            (1 = sequential). Not part of :meth:`as_dict` / the run id: every pair is
            fitted independently with its own seeds, so the outputs do not depend on it.
        seed: Seed for the in-fold holdouts and xgboost sampling.
        max_rounds, early_stopping_rounds: Passed to the xgboost models.
        lolo: Run leave-one-lineage-out fits for the pair's ``lolo_lineage`` values.
        ablation: Also run ``aft_unitig_only``.
        cv_only: Fit CV folds (OOF ``split='cv'``), conformal and the final bundle on all
            train rows, but never predict the test split or LOLO rows (test-split label
            rows are not even loaded) and never append to the test ledger.
        panel_cap: Clip each raw log2 prediction to the panel-edge caps of the rows the
            model was fitted on (:func:`genome2mic.mic.panel_caps_log2`) before rounding
            up and before the band; the final caps are stored in the bundle.
        model_select: (v0.6) Also emit ``aft_b2_select`` and ship it as the bundle: per pair
            the main AFT model, B2 or the average of their log2 predictions, chosen by
            :func:`_select_model` from out-of-fold calls and EA. CV fold ``f`` uses the
            choice made on the other folds only (nested); the bundle uses the choice made on
            every fold. Needs ``b2_xgb_steps`` and the AFT model in ``models``, the
            ``asym_tuned`` band and at least 3 CV folds; otherwise the AFT model ships.
        fold_gate_p: One-sided exact binomial p-value below which a fold counts as
            significantly above its target (v0.6): a calling CV fold whose call VME is
            significantly above ``band_vme_target`` closes the bundle's active-call gate
            (:func:`_oof_gate_check`), and a calibration fold that misses the band's upper
            end significantly more often than ``alpha_up`` widens it
            (:func:`~genome2mic.models.conformal.robust_q_up`).
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
    cv_only: bool = False
    panel_cap: bool = True
    workers: int = 1
    exact_weight: float = 2.0
    band: str = BAND_ASYM_TUNED
    band_alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    band_alpha_low: float = DEFAULT_ALPHA_LOW
    band_vme_target: float = DEFAULT_VME_TARGET
    fold_gate_p: float = DEFAULT_FOLD_P_THRESHOLD
    model_select: bool = True

    def effective_models(self, has_unitigs: bool) -> list[str]:
        if MODEL_SELECT in self.models:
            raise ValueError(f"{MODEL_SELECT} is derived from the fitted AFT and B2 models (TrainConfig.model_select); "
                             "do not list it in models")
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
            "cv_only": self.cv_only,
            "panel_cap": self.panel_cap,
            "exact_weight": float(self.exact_weight),
            "band": self.band,
            "band_alpha_grid": [float(a) for a in self.band_alpha_grid],
            "band_alpha_low": float(self.band_alpha_low),
            "band_vme_target": float(self.band_vme_target),
            "fold_gate_p": float(self.fold_gate_p),
            "model_select": bool(self.model_select),
        }


def compute_run_id(train_config: TrainConfig, splits_path: Path) -> str:
    """``sha1(parameters JSON + splits.parquet bytes)`` truncated to 12 hex characters.

    The run id names a *configuration*; it does not change when labels, features,
    configs or code change. :func:`compute_inputs_sha1` covers those.
    """
    h = hashlib.sha1()
    h.update(json.dumps(train_config.as_dict(), sort_keys=True).encode("utf-8"))
    h.update(Path(splits_path).read_bytes())
    return h.hexdigest()[:12]


def _file_sha1(path: Path) -> str:
    """Content sha1 of one file, streamed in 1 MiB chunks.

    ``.npz`` archives are hashed member by member (sorted member names + the
    decompressed ``.npy`` bytes), so zip metadata such as timestamps never changes
    the digest -- only the arrays do.
    """
    h = hashlib.sha1()
    if path.suffix == ".npz":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                h.update(name.encode("utf-8") + b"\0")
                with archive.open(name) as member:
                    for chunk in iter(lambda: member.read(1 << 20), b""):
                        h.update(chunk)
        return h.hexdigest()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files(paths: Paths, species: str) -> list[tuple[str, Path]]:
    """``(name, path)`` of every file :func:`compute_inputs_sha1` covers for one species.

    Names are root-independent (``data/labels.parquet``, ``configs/drugs.yaml``,
    ``src/genome2mic/models/train.py``), so the same inputs give the same fingerprint
    in any checkout or run root. Covered: ``labels.parquet``, ``known_amr.parquet``,
    ``qc.parquet``, ``lineages.parquet``; the species' unitig matrix, rows, index and
    k-mer set; ``configs/drugs.yaml`` and every ``configs/breakpoints/*.csv``; every
    ``.py`` file under ``src/genome2mic/models/`` and ``src/genome2mic/features/`` plus
    ``src/genome2mic/mic.py``. ``splits.parquet`` and the training parameters are in
    the ``run_id``. Per-genome tool outputs under ``data/interim`` (ResFinder tables
    of the ``b0_resfinder`` baseline) are not covered.
    """
    key = str(species).strip().upper()
    files: list[tuple[str, Path]] = [
        ("data/labels.parquet", paths.labels),
        ("data/known_amr.parquet", paths.known_amr),
        ("data/qc.parquet", paths.qc),
        ("data/lineages.parquet", paths.lineages),
    ]
    for path in (paths.unitigs(key), paths.unitig_rows(key), paths.unitig_index(key), paths.unitig_kmers(key)):
        files.append((f"data/{path.name}", path))
    files.append(("configs/drugs.yaml", paths.configs_dir / "drugs.yaml"))
    breakpoints = paths.configs_dir / "breakpoints"
    if breakpoints.is_dir():
        files.extend((f"configs/breakpoints/{csv_path.name}", csv_path) for csv_path in sorted(breakpoints.glob("*.csv")))
    package = Path(__file__).resolve().parents[1]
    for entry in INPUT_SOURCE_PATHS:
        target = package / entry
        sources = [target] if target.is_file() else sorted(
            path for path in target.rglob("*.py") if "__pycache__" not in path.parts
        )
        files.extend((f"src/genome2mic/{source.relative_to(package).as_posix()}", source) for source in sources)
    return files


def compute_inputs_sha1(paths: Paths, species: str, *, cache: dict[Path, str] | None = None) -> str:
    """Fingerprint (12 hex chars) of the inputs a species' test scoring depends on beyond ``run_id``.

    sha1 over one ``name TAB content-sha1`` line per file of :func:`input_files` (a missing
    file contributes ``absent``, so adding or removing one changes the value). The
    test ledger records it next to ``run_id``; the leakage check fails a pair scored
    under more than one ``(run_id, inputs_sha1)`` -- re-scoring after any change to
    labels, features, the unitig set, configs or model code shows up even though the
    run id is the same. ``cache`` (``path -> digest``) lets one run hash the shared
    files once for all species.
    """
    memo = cache if cache is not None else {}
    h = hashlib.sha1()
    for name, path in input_files(paths, species):
        if path.is_file():
            resolved = path.resolve()
            if resolved not in memo:
                memo[resolved] = _file_sha1(resolved)
            digest = memo[resolved]
        else:
            digest = "absent"
        h.update(f"{name}\t{digest}\n".encode("utf-8"))
    return h.hexdigest()[:12]


def _species_kmer_set_sha1(paths: Paths, species: str) -> str | None:
    """sha1 of the k-mer set this run would ship for ``species`` (``None``: no unitig features).

    Mirrors :func:`_load_species_data`: the species has unitig features when its
    matrix and rows files exist; the set is ``unitigs_<SPECIES>_kmers.npz``.
    """
    if not (paths.unitigs(species).is_file() and paths.unitig_rows(species).is_file()):
        return None
    kmers = paths.unitig_kmers(species)
    if not kmers.is_file():
        return None
    from genome2mic.features.unitigs import read_kmer_set_sha1  # noqa: PLC0415

    return read_kmer_set_sha1(kmers)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


@dataclass
class SpeciesData:
    """Per-species inputs shared by every drug of that species.

    Held only while the species' pairs train (``run`` drops it before loading the
    next species). Nearest-training distances are computed from ``sketches`` on
    demand (:func:`_species_nearest`); no pairwise distance matrix is stored.
    """

    species: str
    known: pd.DataFrame
    """``known_amr.parquet`` rows of the species, indexed by ``genome_id``."""
    known_columns: list[str]
    class_by_column: dict[str, tuple[str | None, str | None]]
    sketch_ids: list[str]
    sketches: np.ndarray
    sketch_k: int
    unitigs: sp.csr_matrix | None
    unitig_rows: pd.DataFrame | None
    unitig_index: pd.DataFrame | None
    unitig_kmer_set_sha1: str | None = None
    """:meth:`~genome2mic.features.unitigs.KmerSet.sha1` of ``unitigs_<SPECIES>_kmers.npz``
    (the set whose pattern columns the matrix holds); recorded in every unitig model's
    ``features.json``."""
    subclass_source: dict[str, Any] | None = None
    """Where the ``gene_`` column subclasses behind the training-time ``strong_subclasses``
    rule came from (AMRFinderPlus DB version and directory); ``None`` = no database, the
    subclass rule then only sees subclasses already in ``known_amr_columns.csv``."""

    @property
    def has_unitigs(self) -> bool:
        return self.unitigs is not None

    @property
    def has_sketches(self) -> bool:
        """False for an imported release without ``sketches_<SPECIES>.npz`` (distances are null)."""
        return bool(self.sketch_ids)

    def sketch_position(self, genome_ids: Sequence[str]) -> np.ndarray:
        if not self.has_sketches:
            return np.full(len(genome_ids), -1, dtype=np.int64)
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
    lab_exact: np.ndarray
    """Exact measured lab MIC per row (:func:`genome2mic.mic.lab_exact_mask`: one doubling
    step, method not ``disk``). Selects B2's training rows and the conformal residuals;
    written to the preds as ``lab_exact``."""

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

    if paths.sketches(species).is_file():
        ids, sketches, k = sk.load_sketches(paths.sketches(species))
        logger.info("%s: %d sketches (k=%d, s=%d)", species, len(ids), k, sketches.shape[1] if sketches.ndim == 2 else 0)
    else:
        ids, sketches, k = [], np.empty((0, 0), dtype=np.uint64), 0
        logger.warning(
            "%s: %s not found (e.g. an imported release ships no Mash sketches); nearest_training_distance "
            "is null for every prediction and the bundle ships no train_sketches.npz",
            species, paths.sketches(species),
        )

    unitigs = rows = index = None
    kmer_sha1: str | None = None
    if paths.unitigs(species).is_file() and paths.unitig_rows(species).is_file():
        from genome2mic.features.unitigs import load_unitigs  # noqa: PLC0415

        unitigs, rows, index = load_unitigs(paths, species)
        kmer_sha1 = _species_kmer_set_sha1(paths, species)
        logger.info(
            "%s: unitig matrix %d rows x %d patterns (k-mer set %s)",
            species, unitigs.shape[0], unitigs.shape[1], kmer_sha1[:12] if kmer_sha1 else "missing",
        )
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
        unitigs=unitigs,
        unitig_rows=rows,
        unitig_index=index,
        unitig_kmer_set_sha1=kmer_sha1,
    )


def _class_map(paths: Paths, subclass_tables: Any = None) -> dict[str, tuple[str | None, str | None]]:
    """``column_name -> (class, subclass)`` from ``known_amr_columns.csv`` (empty if absent).

    ``gene_`` columns get their AMRFinderPlus Subclass from
    :func:`genome2mic.features.subclass_map.column_subclasses`: a non-empty ``subclass``
    cell of the file wins; otherwise (the NCBI release ships none) the member symbols are
    looked up in ``subclass_tables`` (``AMRProt.fa`` alleles, then ``fam.tsv``). The
    subclass feeds the training-time ``strong_subclasses`` rule (:func:`pair_marker`).
    """
    if not paths.known_amr_columns.is_file():
        return {}
    from genome2mic.features.subclass_map import column_subclasses  # noqa: PLC0415

    table = pd.read_csv(paths.known_amr_columns, dtype=str, keep_default_na=False)
    subclasses = column_subclasses(table, subclass_tables)
    out: dict[str, tuple[str | None, str | None]] = {}
    for _, row in table.iterrows():
        column = str(row.get("column_name", "")).strip()
        if not column:
            continue
        cls = str(row.get("class", "")).strip() or None
        if column in out and out[column][0] and not cls:
            cls = out[column][0]
        sub = subclasses.get(column) or (str(row.get("subclass", "")).strip() or None)
        out[column] = (cls, sub)
    return out


def _subclass_tables(paths: Paths, amrfinder_db: Path | None) -> tuple[Any, dict[str, Any] | None]:
    """Load the AMRFinderPlus subclass tables for :func:`_class_map` (``(None, None)`` when unavailable)."""
    from genome2mic.features.subclass_map import load_subclass_tables  # noqa: PLC0415
    from genome2mic.predict.release_features import find_amrfinder_db  # noqa: PLC0415

    db = find_amrfinder_db(Path(amrfinder_db) if amrfinder_db is not None else None)
    if db is None or not (Path(db) / "fam.tsv").is_file():
        imported = paths.processed_dir.joinpath("IMPORTED_RELEASE.json").is_file()
        (logger.warning if imported else logger.info)(
            "train: no AMRFinderPlus database (%s); training-time calls apply strong_markers prefixes and only the "
            "subclasses already in known_amr_columns.csv%s. Pass --amrfinder-db DIR to apply strong_subclasses "
            "(e.g. every acquired carbapenemase for the carbapenems) as the prediction pipeline does",
            db or "amrfinder not on PATH",
            " (the imported release ships none, so the subclass rule is OFF)" if imported else "",
        )
        return None, None
    tables = load_subclass_tables(Path(db))
    source = {"amrfinder_db_version": tables.db_version, "amrfinder_db_dir": str(db),
              "lookup": "AMRProt.fa allele subclass, then fam.tsv node (raw symbol, then without allele suffix)"}
    logger.info("train: gene_ column subclasses from AMRFinderPlus DB %s (%s)", tables.db_version, db)
    return tables, source


def _subclass_by_column(sd: SpeciesData) -> dict[str, str | None]:
    """``gene_`` column -> AMRFinderPlus Subclass (``;``-joined members) for :func:`pair_marker`."""
    return {c: v[1] for c, v in sd.class_by_column.items() if c.startswith("gene_")}


class _PairDropLog(DropLog):
    """A :class:`DropLog` whose every record names its species x drug pair.

    ``detail`` becomes ``"<label>: <detail>"`` (or just ``<label>``), so the rows of
    one pair can be found again in ``drop_log_train.csv`` -- a sharded run replaces
    only its own pairs' rows (:func:`_write_drop_log`).
    """

    def __init__(self, stage: str, label: str) -> None:
        super().__init__(stage)
        self.label = label

    def drop(self, reason: str, n: int, detail: str | None = None) -> None:
        text = None if detail is None else str(detail)
        if not text:
            text = self.label
        elif not (text == self.label or text.startswith(f"{self.label}:")):
            text = f"{self.label}: {text}"
        super().drop(reason, n, text)


def _pair_label(species: str, drug: str) -> str:
    return f"{species} x {drug}"


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
    frame = _pair_frame(species, drug, labels, splits, qc, lineages, sd, droplog)
    return _pair_data(species, drug, frame, sd)


def _pair_frame(
    species: str,
    drug: str,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    qc: pd.DataFrame,
    lineages: pd.DataFrame,
    sd: SpeciesData,
    droplog: DropLog,
) -> pd.DataFrame:
    """The row-level join of :func:`load_pair` (no feature matrices), sorted by ``genome_id``."""
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
    return frame


def _pair_data(species: str, drug: str, frame: pd.DataFrame, sd: SpeciesData) -> PairData:
    """Feature matrices and label arrays for a joined pair frame (:func:`_pair_frame`)."""
    detail = _pair_label(species, drug)
    gids = frame["genome_id"].tolist()
    X_known = sd.known.loc[gids, ["genome_id", "species", *sd.known_columns]].reset_index(drop=True)
    U = None
    if sd.has_unitigs:
        assert sd.unitigs is not None
        U = sp.csr_matrix(sd.unitigs[sd.unitig_position(gids)], dtype=np.int8)
    lo, hi = validate_intervals(frame["mic_lower"].to_numpy(dtype=float), frame["mic_upper"].to_numpy(dtype=float))
    y_point = label_point_log2_array(lo, hi)
    lab_exact = _lab_exact(frame, lo, hi, detail)
    logger.info(
        "%s: %d labelled QC-passing genomes (%d train, %d test), %d known-AMR columns%s",
        detail, len(frame), int((frame["split"] == SPLIT_TRAIN).sum()), int((frame["split"] == SPLIT_TEST).sum()),
        len(sd.known_columns), f", {U.shape[1]} unitig patterns" if U is not None else "",
    )
    return PairData(
        species=species, drug=drug, frame=frame, X_known=X_known, U=U, lo=lo, hi=hi, y_point=y_point,
        sketch_pos=sd.sketch_position(gids), lab_exact=lab_exact,
    )


def _lab_exact(frame: pd.DataFrame, lo: np.ndarray, hi: np.ndarray, label: str) -> np.ndarray:
    """``mic.exact_interval_mask(lo, hi) & (method != 'disk')`` for the pair's rows.

    A frame without a ``method`` column (labels written before the column existed)
    gets the interval rule alone, with a warning.
    """
    if "method" not in frame.columns:
        logger.warning("%s: labels have no method column; lab_exact uses the one-step interval rule alone", label)
        return lab_exact_mask(lo, hi)
    exact = lab_exact_mask(lo, hi, frame["method"].to_numpy(dtype=object))
    n_disk = int((lab_exact_mask(lo, hi) & ~exact).sum())
    if n_disk:
        logger.info(
            "%s: %d disk-diffusion row(s) with a one-step lab interval are not exact MICs "
            "(left out of B2 training, conformal residuals and the exact-MIC metrics)", label, n_disk,
        )
    return exact


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


def _fit(
    model: Any,
    X: sp.csr_matrix,
    lo: np.ndarray,
    hi: np.ndarray,
    names: list[str],
    groups: np.ndarray | None,
    droplog: DropLog,
    exact_rows: np.ndarray | None = None,
    exact_weight: float = 1.0,
) -> Any:
    """Fit a model, passing ``groups``/``droplog``/``exact_rows`` only to the models that accept them.

    ``exact_rows`` (the fit rows' ``lab_exact``) restricts B2's exact-MIC training rows;
    the interval models (B1, AFT) use every interval, disk rows included. For the AFT
    models, ``exact_weight != 1`` up-weights the ``exact_rows`` (``TrainConfig.exact_weight``).
    """
    if model.name == MODEL_B1:
        return model.fit(X, lo, hi, names)
    if model.name == MODEL_B2:
        return model.fit(X, lo, hi, names, groups=groups, droplog=droplog, exact_rows=exact_rows)
    if exact_weight != 1.0 and exact_rows is not None:
        weight = np.where(np.asarray(exact_rows, dtype=bool), float(exact_weight), 1.0)
        return model.fit(X, lo, hi, names, groups=groups, sample_weight=weight)
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
    _fit(model, X[train_idx], pd_.lo[train_idx], pd_.hi[train_idx], names, pd_.groups[train_idx], droplog,
         exact_rows=pd_.lab_exact[train_idx], exact_weight=cfg.exact_weight)
    pred = np.asarray(model.predict_log2(X[predict_idx]), dtype=np.float64) if predict_idx.size else np.empty(0)
    return pred, model, feats, names


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #


def derive_lab_sir(sir: Any, lo: float, hi: float, bp: Any) -> str | None:
    """Reported ``sir`` when present; else the category the interval implies, else null.

    The fallback is :func:`rederive_lab_sir` under the call breakpoint ``bp``.
    """
    if isinstance(sir, str) and sir in ("S", "I", "R"):
        return sir
    return rederive_lab_sir(lo, hi, bp)


def rederive_lab_sir(lo: float, hi: float, bp: Any) -> str | None:
    """The lab interval ``(lo, hi]`` classified under the call breakpoint ``bp``; null if ambiguous.

    With S if MIC <= s and R if MIC > r: ``hi <= s`` -> ``S``; ``lo >= r`` -> ``R``;
    ``s <= lo`` and ``hi <= r`` -> ``I``; an interval straddling a breakpoint, or
    no breakpoint (``bp is None``), -> null. This is the ``lab_sir_rederived``
    column: unlike ``lab_sir`` (the lab's own standard and year), it puts every lab
    result on the same breakpoint table as ``pred_sir``.
    """
    if bp is None:
        return None
    s, r = float(bp.s_breakpoint), float(bp.r_breakpoint)
    lo, hi = float(lo), float(hi)
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


def apply_caps(pred_log2: np.ndarray, caps: tuple[float, float] | None) -> np.ndarray:
    """Clip raw log2 predictions to the panel-edge ``caps`` (no-op for ``None``); NaN stays NaN."""
    pred = np.asarray(pred_log2, dtype=np.float64)
    if caps is None:
        return pred
    return np.clip(pred, caps[0], caps[1])


def pair_marker(
    pd_: PairData, config: Config, subclass_by_column: Mapping[str, str | None] | None = None
) -> np.ndarray:
    """Override-2 strong-marker flag for every row of the pair (see :func:`pair_calls`).

    With ``subclass_by_column`` (:func:`_subclass_by_column`) the ``strong_subclasses``
    rule is applied too (acquired ``gene_`` columns whose every member carries e.g.
    ``CARBAPENEM``), as the pipeline does from the AMRFinderPlus detections.
    """
    from genome2mic.predict import rank  # noqa: PLC0415  (avoids a predict <-> models import cycle)

    return np.asarray(rank.strong_marker_mask(
        pd_.X_known, config.drugs.get(pd_.drug),
        exclude_columns=rank.intrinsic_columns(config.intrinsic_symbols(pd_.species)),
        subclass_by_column=subclass_by_column,
    ), dtype=bool)


def pair_calls(
    pd_: PairData,
    idx: np.ndarray,
    band_low: np.ndarray,
    band_high: np.ndarray,
    config: Config,
    marker_all: np.ndarray | None = None,
) -> np.ndarray:
    """Calls for rows ``idx`` with the prediction pipeline's rule and overrides (:func:`rank.call_array`).

    Override 2 uses the column-prefix rule on the genome's full known-AMR row (every
    column of the species, not only the model's selected features), minus the
    species' intrinsic genes (``intrinsic_markers.csv``), as the pipeline does.
    """
    from genome2mic.predict import rank  # noqa: PLC0415  (avoids a predict <-> models import cycle)

    bp = config.call_breakpoint(pd_.species, pd_.drug)
    if marker_all is not None:
        marker = np.asarray(marker_all, dtype=bool)[idx]
    else:
        marker = rank.strong_marker_mask(
            pd_.X_known.iloc[idx], config.drugs.get(pd_.drug),
            exclude_columns=rank.intrinsic_columns(config.intrinsic_symbols(pd_.species)),
        )
    return rank.call_array(
        band_low, band_high, bp,
        natural_resistance=config.is_naturally_resistant(pd_.species, pd_.drug),
        strong_marker=marker,
    )


def finish_predictions(
    pd_: PairData,
    idx: np.ndarray,
    pred_log2: np.ndarray,
    q: float | BandParams,
    split: str,
    model_id: str,
    run_id: str,
    nearest: np.ndarray,
    config: Config,
    caps: tuple[float, float] | None = None,
    marker_all: np.ndarray | None = None,
) -> pd.DataFrame:
    """Cap, round up, band, classify, call and assemble the preds rows for ``idx``.

    ``caps`` (:func:`genome2mic.mic.panel_caps_log2` of the fit rows) clips the raw
    log2 prediction before rounding up; ``None`` when the caller already capped.
    ``q`` is a :class:`BandParams` (asymmetric band + active-call gate) or a plain
    float (symmetric ``+-q`` band, gate open). ``marker_all`` is the pair's override-2
    flag per frame row (:func:`pair_marker`); computed (column rule only) when omitted.
    ``prob_works`` / ``prob_tier`` are left null here; :func:`train_pair` fills them
    (cross-fitted for CV rows, :func:`_add_probabilities`).
    """
    params = q if isinstance(q, BandParams) else symmetric_params(float(q), DEFAULT_ALPHA, 0)
    pred_log2 = apply_caps(pred_log2, caps)
    clipped = np.clip(pred_log2, GRID_MIN_EXPONENT, GRID_MAX_EXPONENT)
    n_clip = int((clipped != pred_log2).sum())
    if n_clip:
        logger.warning("%s x %s [%s/%s]: %d prediction(s) clipped to the MIC grid edges", pd_.species, pd_.drug, model_id, split, n_clip)
    pred_mic = round_up_to_step_array(2.0**clipped)
    band_low, band_high = params.band(pred_mic)
    bp = config.call_breakpoint(pd_.species, pd_.drug)
    sub = pd_.frame.iloc[idx]
    lab_sir = [derive_lab_sir(s, float(lo), float(hi), bp) for s, lo, hi in zip(sub["sir"], pd_.lo[idx], pd_.hi[idx])]
    rederived = [rederive_lab_sir(lo, hi, bp) for lo, hi in zip(pd_.lo[idx], pd_.hi[idx])]
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
            "lab_sir_rederived": rederived,
            "lab_exact": pd_.lab_exact[idx],
            "call": gate_calls(pair_calls(pd_, idx, band_low, band_high, config, marker_all), params.active_gate_open),
            "prob_works": np.nan,
            "prob_tier": None,
        },
        columns=list(PREDS_COLUMNS),
    )
    return frame


# --------------------------------------------------------------------------- #
# Fit plan and nearest-training distances
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FitGroup:
    """One fit of a pair: the rows a model is fitted on and the rows it predicts.

    Row indices refer to the pair frame (:func:`load_pair`).
    """

    split: str
    """Value of the preds ``split`` column: ``cv``, ``test`` or ``lolo_<lineage>``."""
    predict_idx: np.ndarray
    fit_idx: np.ndarray
    fold: int | None = None
    lineage: str | None = None


@dataclass(frozen=True)
class FitPlan:
    """Every fit :func:`train_pair` makes for one pair, decided from the frame alone."""

    cv: list[FitGroup]
    test: FitGroup
    lolo: list[FitGroup]
    skipped_folds: list[tuple[int, int, int]]
    """``(fold, n_validation, n_fit)`` of folds with nothing to fit or predict."""
    skipped_lolo: list[tuple[str, str, int]]
    """``(lineage, reason, n_rows of the lineage in the pair)`` of LOLO lineages not run."""

    def groups(self) -> list[FitGroup]:
        return [*self.cv, self.test, *self.lolo]


def _fit_plan(frame: pd.DataFrame, cfg: TrainConfig) -> FitPlan:
    """CV folds, the final test fit and the LOLO fits of a pair frame.

    LOLO (decision 1: LOLO clusters are train clusters): for each ``lolo_lineage``
    present in the frame, fit on the train rows outside the lineage and predict the
    lineage's *train* rows. Skipped (and reported in ``skipped_lolo``) when the
    lineage has no train rows for the pair -- the fit set would then equal the
    full train set and the "LOLO" rows would be a relabelled test evaluation -- or
    when fewer than ``_MIN_FIT_ROWS`` train rows remain.
    """
    split = frame["split"].astype(object).to_numpy()
    train_mask = split == SPLIT_TRAIN
    test_mask = split == SPLIT_TEST
    folds = _float_array(frame["fold"])
    train_all = np.flatnonzero(train_mask)
    fold_values = sorted({int(f) for f in folds[train_all] if not np.isnan(f)})

    cv: list[FitGroup] = []
    skipped_folds: list[tuple[int, int, int]] = []
    for f in fold_values:
        val_idx = np.flatnonzero(train_mask & (folds == f))
        fit_idx = np.flatnonzero(train_mask & (folds != f) & ~np.isnan(folds))
        if val_idx.size == 0 or fit_idx.size == 0:
            skipped_folds.append((f, int(val_idx.size), int(fit_idx.size)))
            continue
        cv.append(FitGroup(SPLIT_CV, val_idx, fit_idx, fold=f))
    test_idx = np.empty(0, dtype=np.int64) if cfg.cv_only else np.flatnonzero(test_mask)
    test = FitGroup(SPLIT_TEST, test_idx, train_all)

    lolo: list[FitGroup] = []
    skipped_lolo: list[tuple[str, str, int]] = []
    if cfg.lolo and not cfg.cv_only:
        clusters = frame["lineage_cluster"].astype(object).to_numpy()
        lineages = sorted(frame["lolo_lineage"].dropna().astype(str).unique().tolist())
        for lineage in lineages:
            in_lineage = clusters == lineage
            n_rows = int(in_lineage.sum())
            predict_idx = np.flatnonzero(in_lineage & train_mask)
            fit_idx = np.flatnonzero(train_mask & ~in_lineage)
            if predict_idx.size == 0 or fit_idx.size == train_all.size:
                skipped_lolo.append((lineage, LOLO_SKIP_NO_TRAIN_ROWS, n_rows))
            elif fit_idx.size < _MIN_FIT_ROWS:
                skipped_lolo.append((lineage, LOLO_SKIP_TOO_FEW_FIT_ROWS, n_rows))
            else:
                lolo.append(FitGroup(f"{LOLO_PREFIX}{lineage}", predict_idx, fit_idx, lineage=lineage))
    return FitPlan(cv=cv, test=test, lolo=lolo, skipped_folds=skipped_folds, skipped_lolo=skipped_lolo)


@dataclass
class PairNearest:
    """Nearest-training Mash distance per pair-frame row, one array per kind of fit.

    Each array has one entry per frame row, NaN where the row is not predicted by
    that kind of fit: ``cv`` on CV validation rows (to the other folds' train rows),
    ``test`` on test rows (to every train row), ``lolo[lineage]`` on the lineage's
    train rows (to the train rows outside it).
    """

    cv: np.ndarray
    test: np.ndarray
    lolo: dict[str, np.ndarray] = field(default_factory=dict)


def _mash_from_common(common: np.ndarray, denom: int, k: int) -> np.ndarray:
    """Mash distance ``-ln(2j / (1 + j)) / k`` from shared-hash counts over ``denom`` (``j = common / denom``)."""
    common = np.asarray(common, dtype=np.float64)
    j = common / float(denom)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = -np.log(2.0 * j / (1.0 + j)) / k
    d = np.where(common <= 0, 1.0, d)
    return np.where(common >= denom, 0.0, d)


class _MashBlocks:
    """Exact Mash distances from query sketches to reference sketches, a block of query rows at a time.

    Same estimate as :func:`genome2mic.sketch.pairwise_distances` and
    :func:`genome2mic.sketch.distances_to` (bit-identical results): hashes are
    replaced by dense, order-preserving ranks over the union of the two row sets,
    so "is x in a" and "how many of a are <= x" are table lookups. Memory is the
    rank matrix (int32, rows x s) plus one ``block x n_ref`` float64 result; no
    n x n matrix is formed.

    Args:
        sketches: ``(n, s)`` uint64 sketch matrix of the species (sorted unique rows).
        k: k-mer length of the sketches.
        query_rows, ref_rows: Row positions into ``sketches``.
        ref_block: Reference rows compared per inner step (memory only).
    """

    def __init__(
        self,
        sketches: np.ndarray,
        k: int,
        query_rows: np.ndarray,
        ref_rows: np.ndarray,
        *,
        ref_block: int = _NEAREST_REF_BLOCK,
    ) -> None:
        S = np.asarray(sketches, dtype=np.uint64)
        if S.ndim != 2 or S.shape[1] == 0:
            raise ValueError(f"sketches must be a non-empty (n, s) matrix; got shape {S.shape}")
        if S.shape[1] > 1 and not np.all(S[:, 1:] > S[:, :-1]):
            raise ValueError("every sketch row must be sorted ascending with no duplicates")
        query_rows = np.asarray(query_rows, dtype=np.int64)
        ref_rows = np.asarray(ref_rows, dtype=np.int64)
        both = np.unique(np.concatenate([query_rows, ref_rows]))
        uniq, inverse = np.unique(S[both], return_inverse=True)
        self._ranks = np.ascontiguousarray(inverse.reshape(both.size, S.shape[1]), dtype=np.int32)
        self._query = np.searchsorted(both, query_rows)
        self._ref = np.searchsorted(both, ref_rows)
        self._present = np.zeros(uniq.size, dtype=bool)
        self._cols = np.arange(1, S.shape[1] + 1, dtype=np.int32)
        self._s = int(S.shape[1])
        self._k = int(k)
        self._ref_block = max(1, int(ref_block))

    @property
    def n_query(self) -> int:
        return int(self._query.size)

    @property
    def n_ref(self) -> int:
        return int(self._ref.size)

    def block(self, start: int, stop: int) -> np.ndarray:
        """Distances ``(stop - start, n_ref)`` for query rows ``start:stop``."""
        out = np.empty((stop - start, self.n_ref), dtype=np.float64)
        present, s = self._present, self._s
        for i in range(start, stop):
            a = self._ranks[self._query[i]]
            present[a] = True
            n_a_le = np.cumsum(present, dtype=np.int32)  # |{x in a : rank(x) <= r}|
            for r0 in range(0, self.n_ref, self._ref_block):
                r1 = min(r0 + self._ref_block, self.n_ref)
                rest = self._ranks[self._ref[r0:r1]]
                in_a = present[rest]
                union_rank = n_a_le[rest]
                np.add(union_rank, self._cols, out=union_rank)
                np.subtract(union_rank, np.cumsum(in_a, axis=1, dtype=np.int32), out=union_rank)
                hit = union_rank <= s
                np.logical_and(hit, in_a, out=hit)
                out[i - start, r0:r1] = _mash_from_common(np.count_nonzero(hit, axis=1), s, self._k)
            present[a] = False
        return out


def _species_nearest(
    sd: SpeciesData,
    requests: Mapping[str, tuple[np.ndarray, FitPlan]],
    *,
    query_block: int = _NEAREST_QUERY_BLOCK,
) -> dict[str, PairNearest]:
    """Nearest-training distances for every fit of every requested drug of one species.

    One pass over the union of predicted genomes: each block of query genomes is
    compared with the union of fitted genomes once, then reduced to the minimum
    over each fit group's own fit set. Every drug of the species shares the pass,
    so the cost is about one rectangular query x reference distance computation
    per species, and memory stays at ``query_block x n_ref``.

    Args:
        sd: The species' sketches.
        requests: ``drug -> (sketch row of each pair-frame row, fit plan)``.
        query_block: Query genomes per block.
    """
    out: dict[str, PairNearest] = {}
    jobs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []  # (target, frame rows, query rows, ref rows)
    if not sd.has_sketches:
        for drug, (sketch_pos, plan) in requests.items():
            n = int(sketch_pos.size)
            out[drug] = PairNearest(
                cv=np.full(n, np.nan), test=np.full(n, np.nan),
                lolo={str(g.lineage): np.full(n, np.nan) for g in plan.lolo},
            )
        return out
    for drug, (sketch_pos, plan) in requests.items():
        n = int(sketch_pos.size)
        result = PairNearest(cv=np.full(n, np.nan), test=np.full(n, np.nan))
        for group in plan.groups():
            if group.split == SPLIT_CV:
                target = result.cv
            elif group.split == SPLIT_TEST:
                target = result.test
            else:
                target = result.lolo.setdefault(str(group.lineage), np.full(n, np.nan))
            if group.predict_idx.size and group.fit_idx.size:
                jobs.append((target, group.predict_idx, sketch_pos[group.predict_idx], sketch_pos[group.fit_idx]))
        out[drug] = result
    if not jobs:
        return out

    t0 = time.perf_counter()
    q_rows = np.unique(np.concatenate([j[2] for j in jobs]))
    r_rows = np.unique(np.concatenate([j[3] for j in jobs]))
    engine = _MashBlocks(sd.sketches, sd.sketch_k, q_rows, r_rows)
    prepared = []
    for target, frame_rows, query, ref in jobs:
        local = np.searchsorted(q_rows, query)
        order = np.argsort(local, kind="stable")
        prepared.append((target, frame_rows[order], local[order], np.searchsorted(r_rows, ref)))
    step = max(1, int(query_block))
    for start in range(0, engine.n_query, step):
        stop = min(start + step, engine.n_query)
        D = engine.block(start, stop)
        for target, frame_rows, local, ref_cols in prepared:
            a, b = np.searchsorted(local, [start, stop], side="left")
            if a == b:
                continue
            target[frame_rows[a:b]] = D[np.ix_(local[a:b] - start, ref_cols)].min(axis=1)
    logger.info(
        "%s: nearest-training distances for %d fit group(s) over %d drug(s): %d query x %d reference genomes in %.1fs",
        sd.species, len(jobs), len(requests), q_rows.size, r_rows.size, time.perf_counter() - t0,
    )
    return out


def _conformal(
    pred_log2_oof: np.ndarray,
    pd_: PairData,
    cfg: TrainConfig,
    droplog: DropLog,
    label: str,
    rows: np.ndarray | None = None,
) -> tuple[float, int]:
    """Conformal ``q`` from the (capped) OOF predictions of the train rows (``lab_exact`` rows only).

    ``rows`` (boolean mask) restricts the residuals, e.g. to the rows of the *other*
    folds for cross-conformal bands (:func:`train_pair`).
    """
    clipped = np.clip(pred_log2_oof, GRID_MIN_EXPONENT, GRID_MAX_EXPONENT)
    pred_mic = round_up_to_step_array(2.0**clipped)
    mask = ~np.isnan(pred_mic)
    if rows is not None:
        mask &= np.asarray(rows, dtype=bool)
    residuals = residual_steps(
        np.log2(pred_mic[mask]), pd_.lo[mask], pd_.hi[mask], droplog, exact_rows=pd_.lab_exact[mask]
    )
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


def _rounded_up_log2(pred_log2: np.ndarray) -> np.ndarray:
    """``log2`` of the grid-clipped, rounded-up MIC (what the preds file reports); NaN stays NaN."""
    clipped = np.clip(np.asarray(pred_log2, dtype=np.float64), GRID_MIN_EXPONENT, GRID_MAX_EXPONENT)
    return np.log2(round_up_to_step_array(2.0**clipped))


def _band_params(
    pred_log2_oof: np.ndarray,
    pd_: PairData,
    cfg: TrainConfig,
    config: Config,
    cal_folds: Sequence[int],
    label: str,
    marker_all: np.ndarray | None = None,
    lab_sir: np.ndarray | None = None,
    *,
    allowed_alphas: Sequence[float] | None = None,
) -> BandParams:
    """Tuned asymmetric band from the OOF predictions of ``cal_folds`` only (lever L5).

    The upper level is the narrowest level of
    :func:`genome2mic.models.conformal.passing_levels` (nested cross-conformal call-level
    VME inside ``cal_folds``, counted only over the folds that issue active calls); the
    half-widths are then calibrated on every exact residual of ``cal_folds``. The lower
    half-width goes through :func:`genome2mic.models.conformal.robust_q_low` (widened when
    its leave-one-fold-out values disagree by more than one step). Lab S/I/R is
    re-derived under the call breakpoint; calls use :func:`pair_calls` (the pipeline's
    rule and overrides). The caller passes the folds *other than* the one being banded
    (cross-conformal) or all folds for the bundle.

    ``allowed_alphas`` (bundle only, :func:`bundle_allowed_alphas`) keeps only the levels
    that also passed in every fold that issues calls; an empty sequence closes the gate.
    """
    p = _rounded_up_log2(pred_log2_oof)
    folds = pd_.folds
    bp = config.call_breakpoint(pd_.species, pd_.drug)
    callable_pair = bp is not None and not config.is_naturally_resistant(pd_.species, pd_.drug)
    if not callable_pair:
        lab = np.full(pd_.n, None, dtype=object)
    elif lab_sir is not None:
        lab = lab_sir
    else:
        lab = np.array([rederive_lab_sir(lo, hi, bp) for lo, hi in zip(pd_.lo, pd_.hi)], dtype=object)
    if marker_all is None and callable_pair:
        marker_all = pair_marker(pd_, config)
    passed: list[tuple[float, float]] = []
    if callable_pair:
        passed = passing_levels(
            p, pd_.lo, pd_.hi, pd_.lab_exact, lab, folds, cal_folds,
            lambda idx, low, high: pair_calls(pd_, idx, low, high, config, marker_all),
            grid=cfg.band_alpha_grid, alpha_low=cfg.band_alpha_low, target=cfg.band_vme_target,
        )
        usable = passed if allowed_alphas is None else [(a, u) for a, u in passed
                                                        if any(abs(a - b) < 1e-12 for b in allowed_alphas)]
        if usable:
            alpha_up, gate_open, ucb = usable[0][0], True, usable[0][1]
        else:
            alpha_up, gate_open, ucb = float(cfg.band_alpha_grid[-1]), False, None
    else:
        # No call breakpoint: nothing can be certified, so the gate ships closed (the pipeline's
        # call is 'uncertain' anyway). Natural resistance: the call is always likely_inactive.
        alpha_up, gate_open, ucb = 0.05, bp is not None, None
    cal_list = [int(f) for f in cal_folds]
    cal = np.isin(folds, cal_list) & ~np.isnan(p)
    signed = signed_residual_steps(p[cal], pd_.lo[cal], pd_.hi[cal], exact_rows=pd_.lab_exact[cal])
    q_up, q_low, certified = asym_quantiles(signed, alpha_up, cfg.band_alpha_low)
    by_fold = []
    for g in cal_list:
        rows = cal & (folds == g)
        by_fold.append(signed_residual_steps(p[rows], pd_.lo[rows], pd_.hi[rows], exact_rows=pd_.lab_exact[rows]))
    q_low, widened = robust_q_low(by_fold, q_low, cfg.band_alpha_low, cfg.alpha)
    if widened:
        logger.info("%s: leave-one-fold-out lower half-widths disagree by > 1 step; lower end widened to %.2f steps",
                    label, q_low)
    q_up_before = q_up
    q_up, up_widened = robust_q_up(by_fold, q_up, alpha_up, p_threshold=cfg.fold_gate_p)
    if up_widened:
        logger.info("%s: a calibration fold misses the upper end significantly more often than %.3g "
                    "(one-sided binomial p < %.2g); upper end widened from %.2f to %.2f steps",
                    label, alpha_up, cfg.fold_gate_p, q_up_before, q_up)
    if not gate_open:
        logger.info("%s: no upper level met call VME <= %.1f%% in the calling training folds%s; likely_active calls withheld",
                    label, 100 * cfg.band_vme_target, "" if allowed_alphas is None else " (and in every calling CV fold)")
    elif not certified:
        logger.warning("%s: %d exact residuals cannot certify the chosen upper level %.3g; likely_active calls withheld",
                       label, signed.size, alpha_up)
    return BandParams(
        q_up=q_up, q_low=q_low, alpha_up=alpha_up, alpha_low=cfg.band_alpha_low,
        active_gate_open=bool(gate_open and certified), n_residuals=int(signed.size), inner_vme_ucb=ucb,
        q_low_widened=widened, passing_alphas=tuple(a for a, _ in passed),
        allowed_alphas=None if allowed_alphas is None else tuple(float(a) for a in allowed_alphas),
        q_up_widened=up_widened,
    )


def _oof_gate_check(
    params: BandParams,
    cv_calls: Sequence[tuple[np.ndarray, np.ndarray]],
    pd_: PairData,
    lab_sir: np.ndarray | None,
    folds: np.ndarray,
    target: float,
    label: str,
    p_threshold: float = DEFAULT_FOLD_P_THRESHOLD,
) -> BandParams:
    """Close the bundle's active-call gate unless the out-of-fold CV calls pass the call-VME rules.

    Two rules, both on train-fold rows only:

    1. pooled: :func:`~genome2mic.models.conformal.vme_certified` on the VMEs and lab-R
       rows pooled over the CV folds that issued at least one ``likely_active`` call
       (:func:`~genome2mic.models.conformal.calling_fold_vme`);
    2. per fold: no calling fold's call VME may be significantly above ``target``
       (one-sided exact binomial ``P(X >= n_vme | n_lab_R, target) < p_threshold``, default 0.01,
       :func:`~genome2mic.models.conformal.fold_call_vme_table`). One bad fold is enough:
       pooling across folds must not hide a fold where the calls were unsafe.

    The per-fold table is recorded on the result (``fold_call_vme``, ``conformal.json``)
    whenever the pair is callable, also for a gate that was already closed. Uncallable
    pairs (``lab_sir`` None) are returned as is.
    """
    if lab_sir is None or not cv_calls:
        return params
    idx = np.concatenate([i for _, i in cv_calls])
    calls = np.concatenate([c for c, _ in cv_calls])
    table = tuple(fold_call_vme_table(calls, lab_sir[idx], folds[idx], target, p_threshold))
    params = replace(params, fold_call_vme=table)
    if not params.active_gate_open or params.inner_vme_ucb is None:
        return params
    k, n_r, n_folds = calling_fold_vme(calls, lab_sir[idx], folds[idx])
    ucb = (k + 1) / (n_r + 1)
    bad_folds = [r for r in table if r["significant"]]
    if bad_folds:
        logger.info("%s: calling fold(s) %s have call VME significantly above %.1f%% (%s); bundle likely_active "
                    "calls withheld", label, [r["fold"] for r in bad_folds], 100 * target,
                    ", ".join(f"fold {r['fold']}: {r['n_vme']}/{r['n_lab_r']}, p={r['p_value']:.2g}" for r in bad_folds))
        return replace(params, active_gate_open=False, oof_call_vme_ucb=ucb, fold_gate_closed=True)
    if n_folds and vme_certified(k, n_r, target):
        return replace(params, oof_call_vme_ucb=ucb)
    logger.info("%s: out-of-fold call VME over %d calling fold(s) is %d / %d lab R (UCB %.3f > %.3f); "
                "bundle likely_active calls withheld", label, n_folds, k, n_r, ucb, target)
    return replace(params, active_gate_open=False, oof_call_vme_ucb=ucb)


def bundle_allowed_alphas(params_cv: Mapping[int, BandParams], grid: Sequence[float]) -> tuple[float, ...] | None:
    """Upper levels the bundle may use: those that passed in every CV fold that issues calls.

    A fold "issues calls" when its cross-conformal gate opened. The result is the grid
    levels found in the ``passing_alphas`` of every such fold (grid order); empty when no
    fold opened its gate (the bundle's gate then stays closed) or the folds share no
    passing level. ``None`` when there are no folds (no restriction).
    """
    if not params_cv:
        return None
    calling = [p_ for p_ in params_cv.values() if p_.active_gate_open]
    if not calling:
        return ()
    return tuple(float(a) for a in grid
                 if all(any(abs(a - b) < 1e-12 for b in p_.passing_alphas) for p_ in calling))


# --------------------------------------------------------------------------- #
# aft_b2_select: in-fold choice of AFT, B2 or their average (v0.6)
# --------------------------------------------------------------------------- #

_MIN_SELECT_FOLDS = 3
"""Nested choice needs two inner folds per outer fold (one scored, one calibrating)."""


def select_score(
    calls: np.ndarray,
    idx: np.ndarray,
    pred_mic: np.ndarray,
    pd_: PairData,
    lab_sir: np.ndarray | None,
    target: float,
) -> dict[str, Any]:
    """Score one candidate on rows ``idx`` (its out-of-fold calls and rounded-up ``pred_mic``).

    ``vme_ok``: the gate's rule ``(n_vme + 1) / (n_lab_R + 1) <= target`` over the folds that
    issue ``likely_active`` calls (no calling fold = ok); ``active_s``: share of lab-S rows
    called ``likely_active``; ``ea``: essential agreement (+-1 step) on exact lab MICs. Lab
    S/I/R is the call-breakpoint re-derivation (``None`` = unknown, never counted).
    """
    calls = np.asarray(calls, dtype=object)
    idx = np.asarray(idx, dtype=np.int64)
    lab = np.full(idx.size, None, dtype=object) if lab_sir is None else np.asarray(lab_sir, dtype=object)[idx]
    k, n_r, n_cf = calling_fold_vme(calls, lab, pd_.folds[idx])
    ok = vme_certified(k, n_r, target) if n_cf else True
    is_s = np.array([v == "S" for v in lab], dtype=bool)
    active = np.array([c == "likely_active" for c in calls], dtype=bool)
    act = float((active & is_s).sum() / is_s.sum()) if is_s.any() else 0.0
    lo, hi = pd_.lo[idx], pd_.hi[idx]
    exact = np.asarray(pd_.lab_exact[idx], dtype=bool) & exact_mask(lo, hi)
    pm = np.asarray(pred_mic, dtype=np.float64)
    ea = float(np.mean(np.abs(np.log2(pm[exact]) - np.log2(hi[exact])) <= 1 + 1e-9)) if exact.any() else 0.0
    return {
        "vme_ok": bool(ok), "n_vme": int(k), "n_lab_r_calling": int(n_r), "n_calling_folds": int(n_cf),
        "active_s": act, "n_lab_s": int(is_s.sum()), "ea": ea, "n_exact": int(exact.sum()), "n": int(idx.size),
    }


def select_key(name: str, score: Mapping[str, Any]) -> tuple[Any, ...]:
    """Ranking key: call VME passes, then % lab S called active (to 1 pp), then EA (0.1 pp), then AFT > avg > B2."""
    act = float(score["active_s"]) if score["vme_ok"] else 0.0
    return (bool(score["vme_ok"]), round(act, 2), round(float(score["ea"]), 3), -model_select.CHOICES.index(name))


def choose_candidate(scores: Mapping[str, Mapping[str, Any]]) -> str:
    """The best candidate under :func:`select_key`."""
    if not scores:
        raise ValueError("no candidates to choose from")
    return max(scores, key=lambda c: select_key(c, scores[c]))


def _candidate_calls(
    cand: np.ndarray,
    pd_: PairData,
    cfg: TrainConfig,
    config: Config,
    cal_folds: Sequence[int],
    rows: np.ndarray,
    label: str,
    marker_all: np.ndarray,
    lab_sir: np.ndarray | None,
    params: BandParams | None = None,
) -> tuple[BandParams, np.ndarray, np.ndarray, np.ndarray]:
    """Band tuned on ``cal_folds`` (unless ``params`` is given) and the calls for ``rows`` -- train_pair's CV path."""
    if params is None:
        params = _band_params(cand, pd_, cfg, config, cal_folds, label, marker_all, lab_sir)
    idx = np.asarray(rows, dtype=np.int64)
    idx = idx[~np.isnan(cand[idx])]
    pred_mic = round_up_to_step_array(2.0 ** np.clip(cand[idx], GRID_MIN_EXPONENT, GRID_MAX_EXPONENT))
    low, high = params.band(pred_mic)
    calls = gate_calls(pair_calls(pd_, idx, low, high, config, marker_all), params.active_gate_open)
    return params, idx, pred_mic, calls


@dataclass
class SelectOutcome:
    """Result of :func:`_select_model` for one pair."""

    base_model: str
    candidates: dict[str, np.ndarray]
    """Capped log2 OOF prediction per candidate (``aft`` / ``b2`` / ``avg``)."""
    choice_by_fold: dict[int, str]
    """Candidate used for CV fold f, chosen on the other folds only."""
    params_cv: dict[int, BandParams]
    """Cross-conformal band of the chosen candidate per CV fold."""
    cv_frames: list[pd.DataFrame]
    cv_calls: list[tuple[np.ndarray, np.ndarray]]
    scores_by_fold: dict[int, dict[str, dict[str, Any]]]
    bundle_choice: str
    """What the bundle ships: ``rule_choice``, or ``aft`` after a fallback."""
    bundle_scores: dict[str, dict[str, Any]]
    bundle_params: BandParams
    n_residuals: int
    rule_choice: str = model_select.CHOICE_AFT
    """The rule's pick on every fold's cross-conformal calls."""
    fallback_from: str | None = None
    """Set when a non-AFT pick was not certified for likely-active calls and the bundle fell back to AFT."""
    selection_gate_check: dict[str, Any] = field(default_factory=dict)
    """The gate rules on the nested selection's out-of-fold calls (:func:`_gate_rules`)."""

    def as_json(self) -> dict[str, Any]:
        def clean(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
            return {c: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()} for c, s in scores.items()}

        return {
            "model": MODEL_SELECT,
            "base_model": self.base_model,
            "candidates": list(model_select.CHOICES),
            "choice": self.bundle_choice,
            "rule_choice": self.rule_choice,
            "fallback_from": self.fallback_from,
            "rule": "max over candidates of (call VME (k+1)/(n_lab_R+1) <= target over calling folds, "
                    "% lab S likely_active to 1 pp, EA to 0.1 pp, preference aft > avg > b2)",
            "gate_rule": "the shipped candidate's own out-of-fold calls must pass the pooled calling-fold UCB and the "
                         "per-fold binomial test (as the AFT bundle); a non-aft pick must also pass both on the "
                         "nested selection's out-of-fold calls, else the bundle falls back to aft when aft's own "
                         "gate is open",
            "selection_gate_check": dict(self.selection_gate_check),
            "bundle_scores": clean(self.bundle_scores),
            "choice_by_fold": {str(f): c for f, c in sorted(self.choice_by_fold.items())},
            "inner_scores_by_fold": {str(f): clean(s) for f, s in sorted(self.scores_by_fold.items())},
            "source": "out-of-fold CV predictions of the train folds only (test split never used)",
        }


def _select_model(
    pd_: PairData,
    cfg: TrainConfig,
    config: Config,
    plan: FitPlan,
    oof: Mapping[str, np.ndarray],
    base_model: str,
    params_cv: Mapping[str, Mapping[int, BandParams]],
    marker_all: np.ndarray,
    lab_sir: np.ndarray | None,
    run_id: str,
    nearest_cv: np.ndarray,
    label: str,
) -> SelectOutcome:
    """Choose AFT, B2 or their average per CV fold (nested) and for the bundle (all folds).

    Candidates are the capped log2 OOF predictions of the AFT model and B2 and their mean.
    For outer fold ``f`` every candidate is scored on the other folds ``g``: the band for
    ``g`` is tuned on the folds other than ``f`` and ``g`` (``_band_params``, the same
    nested level choice and gate as everywhere), the calls on ``g`` are pooled and scored
    with :func:`select_score` / :func:`select_key`. Fold ``f``'s rows then get the chosen
    candidate with that candidate's own cross-conformal band for ``f`` (calibrated on the
    other folds), so the ``aft_b2_select`` CV rows are an honest out-of-fold estimate of the
    whole procedure. The bundle repeats the choice on every fold's cross-conformal calls.

    Bundle gate: the chosen candidate's band on all folds (levels limited to those that passed
    in every calling fold), gated on the candidate's own out-of-fold calls by
    :func:`_oof_gate_check` (pooled UCB and per-fold test), exactly as the AFT bundle is. A
    pick other than AFT must also pass both rules on the nested selection's out-of-fold calls
    (:func:`_gate_rules`); when its gate ends up closed while AFT's own gate is open, the
    bundle falls back to AFT (the incumbent), so a likely-active call is never shipped on a
    weaker standard than the AFT bundle's and a deviation from it needs both certifications.
    """
    folds = pd_.folds
    cv_folds = sorted(int(g.fold) for g in plan.cv)
    rows_of = {int(g.fold): g.predict_idx for g in plan.cv}
    target = cfg.band_vme_target
    cands = {
        model_select.CHOICE_AFT: np.asarray(oof[base_model], dtype=np.float64),
        model_select.CHOICE_B2: np.asarray(oof[MODEL_B2], dtype=np.float64),
    }
    cands[model_select.CHOICE_AVG] = model_select.combine_log2(cands["aft"], cands["b2"], model_select.CHOICE_AVG)
    cands = {c: cands[c] for c in model_select.CHOICES}

    # Outer cross-conformal band + calls per candidate (aft / b2 reuse the per-model bands).
    outer_params: dict[str, dict[int, BandParams]] = {
        model_select.CHOICE_AFT: dict(params_cv.get(base_model, {})),
        model_select.CHOICE_B2: dict(params_cv.get(MODEL_B2, {})),
        model_select.CHOICE_AVG: {},
    }
    outer: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {c: {} for c in cands}
    for c, cand in cands.items():
        for f in cv_folds:
            others = [g for g in cv_folds if g != f]
            params, idx, pm, calls = _candidate_calls(cand, pd_, cfg, config, others, rows_of[f],
                                                      f"{label} [{MODEL_SELECT}:{c}] fold {f}", marker_all, lab_sir,
                                                      params=outer_params[c].get(f))
            outer_params[c][f] = params
            outer[c][f] = (idx, pm, calls)

    # Nested choice for each outer fold: score on the other folds only.
    choice_by_fold: dict[int, str] = {}
    scores_by_fold: dict[int, dict[str, dict[str, Any]]] = {}
    for f in cv_folds:
        inner = [g for g in cv_folds if g != f]
        scores: dict[str, dict[str, Any]] = {}
        for c, cand in cands.items():
            parts = []
            for g in inner:
                cal = [h for h in inner if h != g]
                _, idx, pm, calls = _candidate_calls(cand, pd_, cfg, config, cal, rows_of[g],
                                                     f"{label} [{MODEL_SELECT}:{c}] inner {f}/{g}", marker_all, lab_sir)
                parts.append((idx, pm, calls))
            scores[c] = select_score(np.concatenate([p[2] for p in parts]), np.concatenate([p[0] for p in parts]),
                                     np.concatenate([p[1] for p in parts]), pd_, lab_sir, target)
        choice_by_fold[f] = choose_candidate(scores)
        scores_by_fold[f] = scores

    frames: list[pd.DataFrame] = []
    sel_calls: list[tuple[np.ndarray, np.ndarray]] = []
    sel_params: dict[int, BandParams] = {}
    for f in cv_folds:
        c = choice_by_fold[f]
        sel_params[f] = outer_params[c][f]
        idx = rows_of[f][~np.isnan(cands[c][rows_of[f]])]
        if idx.size:
            frame = finish_predictions(pd_, idx, cands[c][idx], sel_params[f], SPLIT_CV, MODEL_SELECT, run_id,
                                       nearest_cv[idx], config, marker_all=marker_all)
            frames.append(frame)
            sel_calls.append((frame["call"].to_numpy(dtype=object), idx))

    # Bundle: the same rule on every fold's cross-conformal calls.
    bundle_scores = {}
    for c in cands:
        idx = np.concatenate([outer[c][f][0] for f in cv_folds])
        bundle_scores[c] = select_score(np.concatenate([outer[c][f][2] for f in cv_folds]), idx,
                                        np.concatenate([outer[c][f][1] for f in cv_folds]), pd_, lab_sir, target)
    rule_choice = choose_candidate(bundle_scores)
    has_bp = config.call_breakpoint(pd_.species, pd_.drug) is not None

    def bundle_params(c: str) -> BandParams:
        """Candidate c's bundle band, gated on c's own out-of-fold calls (the AFT bundle's standard)."""
        allowed = bundle_allowed_alphas(outer_params[c], cfg.band_alpha_grid)
        p_ = _band_params(cands[c], pd_, cfg, config, cv_folds, f"{label} [{MODEL_SELECT}:{c}] bundle",
                          marker_all, lab_sir, allowed_alphas=allowed)
        own = [(outer[c][f][2], outer[c][f][0]) for f in cv_folds]
        p_ = _oof_gate_check(p_, own, pd_, lab_sir, folds, target, f"{label} [{MODEL_SELECT}:{c}] bundle", cfg.fold_gate_p)
        return p_ if has_bp or not p_.active_gate_open else replace(p_, active_gate_open=False)

    # The selection's nested out-of-fold calls (the aft_b2_select CV rows) under the same two gate rules.
    sel_ok, sel_check = _gate_rules(sel_calls, pd_, lab_sir, folds, target, cfg.fold_gate_p)
    bundle_choice, fallback_from = rule_choice, None
    params = bundle_params(rule_choice)
    if rule_choice != model_select.CHOICE_AFT:
        if params.active_gate_open and not sel_ok:
            logger.info("%s [%s]: %s passes on its own out-of-fold calls but the nested selection's do not (%s); "
                        "its gate is closed", label, MODEL_SELECT, rule_choice, sel_check)
            params = replace(params, active_gate_open=False)
        if not params.active_gate_open:
            aft_params = bundle_params(model_select.CHOICE_AFT)
            if aft_params.active_gate_open:
                # Deviating from the incumbent was not certified: ship the AFT model, gated as before.
                logger.info("%s [%s]: %s not certified for likely-active calls; the bundle falls back to aft "
                            "(its own out-of-fold calls pass)", label, MODEL_SELECT, rule_choice)
                bundle_choice, fallback_from, params = model_select.CHOICE_AFT, rule_choice, aft_params
    logger.info("%s [%s]: fold choices %s; rule choice %s, bundle %s (gate %s); scores %s", label, MODEL_SELECT,
                {f: c for f, c in sorted(choice_by_fold.items())}, rule_choice, bundle_choice,
                "open" if params.active_gate_open else "closed",
                {c: (s["vme_ok"], round(s["active_s"], 3), round(s["ea"], 3)) for c, s in bundle_scores.items()})
    return SelectOutcome(
        base_model=base_model, candidates=cands, choice_by_fold=choice_by_fold, params_cv=sel_params, cv_frames=frames,
        cv_calls=sel_calls, scores_by_fold=scores_by_fold, bundle_choice=bundle_choice, bundle_scores=bundle_scores,
        bundle_params=params, n_residuals=int(params.n_residuals), rule_choice=rule_choice,
        fallback_from=fallback_from, selection_gate_check=sel_check,
    )


def _gate_rules(
    cv_calls: Sequence[tuple[np.ndarray, np.ndarray]],
    pd_: PairData,
    lab_sir: np.ndarray | None,
    folds: np.ndarray,
    target: float,
    p_threshold: float,
) -> tuple[bool, dict[str, Any]]:
    """The two :func:`_oof_gate_check` rules on out-of-fold calls, as ``(passed, record)``.

    Passes when no fold makes a likely-active call (nothing to certify), or when the pooled
    calling-fold UCB passes and no calling fold is significantly above ``target``.
    """
    if lab_sir is None or not cv_calls:
        return True, {"n_vme": 0, "n_lab_r_calling": 0, "n_calling_folds": 0, "ucb": None, "significant_folds": []}
    idx = np.concatenate([i for _, i in cv_calls])
    calls = np.concatenate([c for c, _ in cv_calls])
    k, n_r, n_cf = calling_fold_vme(calls, lab_sir[idx], folds[idx])
    table = fold_call_vme_table(calls, lab_sir[idx], folds[idx], target, p_threshold)
    significant = [int(r["fold"]) for r in table if r["significant"]]
    passed = n_cf == 0 or (vme_certified(k, n_r, target) and not significant)
    return bool(passed), {"n_vme": int(k), "n_lab_r_calling": int(n_r), "n_calling_folds": int(n_cf),
                          "ucb": (k + 1) / (n_r + 1) if n_cf else None, "significant_folds": significant}


def _select_calibration(
    select: SelectOutcome,
    pd_: PairData,
    config: Config,
    marker_all: np.ndarray,
) -> tuple[prob_cal.PairCalibration, dict[str, Any]]:
    """P(works) map for the bundle: fit on the chosen candidate's out-of-fold CV predictions.

    The bundle always applies the bundle choice, so its map is fit on that candidate's
    predictions (train-fold rows only); the per-fold record is cross-fitted the same way.
    """
    bp = config.call_breakpoint(pd_.species, pd_.drug)
    s_bp = float(bp.s_breakpoint) if bp is not None else None
    natural = config.is_naturally_resistant(pd_.species, pd_.drug)
    cand = select.candidates[select.bundle_choice]
    rows = np.flatnonzero(~np.isnan(pd_.folds) & ~np.isnan(cand))
    pred_mic = round_up_to_step_array(2.0 ** np.clip(cand[rows], GRID_MIN_EXPONENT, GRID_MAX_EXPONENT))
    lab = np.array([rederive_lab_sir(lo, hi, bp) for lo, hi in zip(pd_.lo[rows], pd_.hi[rows])], dtype=object) \
        if bp is not None else np.full(rows.size, None, dtype=object)
    works = prob_cal.works_from_sir(lab)
    marker = np.asarray(marker_all, dtype=bool)[rows]
    _, by_fold = prob_cal.cross_fit_probabilities(pred_mic, works, marker, pd_.folds[rows], s_bp, natural_resistance=natural)
    full = prob_cal.fit_pair_calibration(pred_mic, works, marker, s_bp, natural_resistance=natural)
    return full, prob_cal.summarize_fits(by_fold)


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
            "lab_sir_rederived": [rederive_lab_sir(lo, hi, bp) for lo, hi in zip(pd_.lo[idx], pd_.hi[idx])],
            "lab_exact": pd_.lab_exact[idx],
            "call": None,
            "prob_works": np.nan,
            "prob_tier": None,
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


def train_pair(
    paths: Paths,
    config: Config,
    cfg: TrainConfig,
    sd: SpeciesData,
    pd_: PairData,
    run_id: str,
    droplog: DropLog,
    pheno_cache: dict[str, dict[str, str] | None],
    *,
    plan: FitPlan | None = None,
    nearest: PairNearest | None = None,
    inputs_sha1: str | None = None,
) -> PairResult:
    """CV, conformal calibration, final fit, LOLO and bundle for one species x drug pair.

    ``plan``, ``nearest`` and ``inputs_sha1`` are computed here when not given
    (:func:`run` passes them so that one distance pass and one input fingerprint
    serve every drug of a species). Writes the preds file and appends the pair's
    test-row count, ``run_id`` and ``inputs_sha1`` to ``results/test_ledger.csv``.
    """
    label = _pair_label(pd_.species, pd_.drug)
    if inputs_sha1 is None:
        inputs_sha1 = compute_inputs_sha1(paths, pd_.species)
    models = cfg.effective_models(sd.has_unitigs)
    plan = plan if plan is not None else _fit_plan(pd_.frame, cfg)
    train_all = plan.test.fit_idx
    test_idx = plan.test.predict_idx
    if not plan.cv and not plan.skipped_folds:
        raise ContractViolation(f"{label}: train rows have no fold assignment", STAGE)
    if len(train_all) < _MIN_FIT_ROWS:
        raise ValueError(f"{label}: only {len(train_all)} training rows; pairs_kept should have excluded this pair")
    if nearest is None:
        nearest = _species_nearest(sd, {pd_.drug: (pd_.sketch_pos, plan)})[pd_.drug]

    preds: list[pd.DataFrame] = []
    q_by_model: dict[str, float] = {}
    n_res: dict[str, int] = {}
    timings: dict[str, float] = {}

    # --- CV: out-of-fold predictions, in-fold selection and panel caps -----------
    # Each fold's raw predictions are clipped to the panel caps of that fold's own fit
    # rows (never the validation rows), so oof[m] holds capped log2 predictions.
    oof: dict[str, np.ndarray] = {m: np.full(pd_.n, np.nan) for m in models}
    oof_nearest = nearest.cv
    fold_caps: dict[int, tuple[float, float] | None] = {}
    t0 = time.perf_counter()
    for f, n_val, n_fit in plan.skipped_folds:
        logger.warning("%s: fold %d has %d validation / %d fit rows; skipped", label, f, n_val, n_fit)
    for group in plan.cv:
        caps = panel_caps_log2(pd_.lo[group.fit_idx], pd_.hi[group.fit_idx]) if cfg.panel_cap else None
        fold_caps[int(group.fold)] = caps  # type: ignore[arg-type]
        selected = _select_all(pd_, group.fit_idx, cfg, droplog, models, sd.has_unitigs)
        for m in models:
            try:
                pred, _, _, _ = _fit_predict(pd_, m, group.fit_idx, group.predict_idx, cfg, droplog, feats=selected[m])
            except ValueError as error:
                logger.warning("%s [%s] fold %d: %s; fold predictions are null", label, m, group.fold, error)
                continue
            oof[m][group.predict_idx] = apply_caps(pred, caps)
        logger.info("%s: fold %d done (%d fit, %d validation rows, caps %s)", label, group.fold, group.fit_idx.size,
                    group.predict_idx.size, caps)
    timings["cv_s"] = time.perf_counter() - t0

    # --- Conformal: q for the bundle from every OOF residual; the CV rows' own bands are
    # cross-conformal (fold f's rows use q calibrated on the residuals of the other folds).
    # With cfg.band == "asym_tuned" the upper level and the active-call gate are tuned
    # inside the calibration folds too (nested: fold f's level never sees fold f).
    folds_arr = pd_.folds
    q_cv: dict[str, dict[int, float]] = {}
    params_cv: dict[str, dict[int, BandParams]] = {}
    params_by_model: dict[str, BandParams] = {}
    cv_calls: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    cv_folds = sorted(int(g.fold) for g in plan.cv)
    # Override 2 for every row, with the strong_subclasses rule when the subclass map is known.
    marker_all = pair_marker(pd_, config, _subclass_by_column(sd))
    band_marker = marker_all if cfg.band == BAND_ASYM_TUNED else None
    _bp = config.call_breakpoint(pd_.species, pd_.drug)
    band_lab = np.array([rederive_lab_sir(lo, hi, _bp) for lo, hi in zip(pd_.lo, pd_.hi)], dtype=object) \
        if cfg.band == BAND_ASYM_TUNED and _bp is not None else None
    for m in models:
        q, n = _conformal(oof[m], pd_, cfg, droplog, f"{label} [{m}]")  # NaN outside train rows
        q_cv[m] = {}
        params_cv[m] = {}
        quiet = DropLog(STAGE)
        for group in plan.cv:
            other = ~np.isnan(folds_arr) & (folds_arr != group.fold)
            if cfg.band == BAND_ASYM_TUNED:
                others = [f for f in cv_folds if f != int(group.fold)]
                params_f = _band_params(oof[m], pd_, cfg, config, others, f"{label} [{m}] fold {group.fold}",
                                        band_marker, band_lab)
            else:
                q_f, n_f = _conformal(oof[m], pd_, cfg, quiet, f"{label} [{m}] fold {group.fold}", rows=other)
                params_f = symmetric_params(q_f, cfg.alpha, n_f)
            q_cv[m][int(group.fold)] = params_f.q_up
            params_cv[m][int(group.fold)] = params_f
            has = group.predict_idx[~np.isnan(oof[m][group.predict_idx])]
            if has.size:
                cv_frame = finish_predictions(pd_, has, oof[m][has], params_f, SPLIT_CV, m, run_id, oof_nearest[has], config,
                                              marker_all=marker_all)
                preds.append(cv_frame)
                cv_calls.setdefault(m, []).append((cv_frame["call"].to_numpy(dtype=object), has))
        # Bundle: tuned on every fold, but never narrower than a level a fold validated
        # (conservative: only levels that passed in every calling fold; see bundle_allowed_alphas).
        if cfg.band == BAND_ASYM_TUNED:
            params_by_model[m] = _band_params(oof[m], pd_, cfg, config, cv_folds, f"{label} [{m}] bundle",
                                              band_marker, band_lab,
                                              allowed_alphas=bundle_allowed_alphas(params_cv[m], cfg.band_alpha_grid))
            params_by_model[m] = _oof_gate_check(params_by_model[m], cv_calls.get(m, []), pd_, band_lab, folds_arr,
                                                 cfg.band_vme_target, f"{label} [{m}] bundle", cfg.fold_gate_p)
        else:
            params_by_model[m] = symmetric_params(q, cfg.alpha, n)
        if _bp is None and params_by_model[m].active_gate_open:
            # No call breakpoint: nothing was certified, so the bundle ships with the gate closed.
            params_by_model[m] = replace(params_by_model[m], active_gate_open=False)
        q_by_model[m] = params_by_model[m].q_up
        n_res[m] = n
        bundle_params = params_by_model[m]
        logger.info(
            "%s [%s]: band %s q_up=%.2f q_low=%.2f (alpha_up %.3g, gate %s) from %d exact OOF residuals (bundle); "
            "cross-conformal fold (alpha_up, gate) %s",
            label, m, bundle_params.kind, bundle_params.q_up, bundle_params.q_low, bundle_params.alpha_up,
            "open" if bundle_params.active_gate_open else "closed", n,
            {f: (p_.alpha_up, p_.active_gate_open) for f, p_ in params_cv[m].items()},
        )

    # --- aft_b2_select: AFT, B2 or their average, chosen inside the training folds --
    base_model = MODEL_AFT_KNOWN_UNITIG if MODEL_AFT_KNOWN_UNITIG in models else MODEL_AFT_KNOWN
    select: SelectOutcome | None = None
    if cfg.model_select:
        why_not = (
            "band is not asym_tuned" if cfg.band != BAND_ASYM_TUNED
            else f"needs {base_model} and {MODEL_B2}" if base_model not in models or MODEL_B2 not in models
            else f"only {len(plan.cv)} CV fold(s)" if len(plan.cv) < _MIN_SELECT_FOLDS
            else None
        )
        if why_not is None:
            t_sel = time.perf_counter()
            select = _select_model(pd_, cfg, config, plan, oof, base_model, params_cv, marker_all, band_lab, run_id,
                                   oof_nearest, label)
            preds.extend(select.cv_frames)
            params_by_model[MODEL_SELECT] = select.bundle_params
            q_by_model[MODEL_SELECT] = select.bundle_params.q_up
            n_res[MODEL_SELECT] = select.n_residuals
            timings["select_s"] = time.perf_counter() - t_sel
        else:
            logger.info("%s: %s skipped (%s); %s ships", label, MODEL_SELECT, why_not, base_model)
    main_model = MODEL_SELECT if select is not None else base_model

    # --- Final fit on all train rows -> bundle (+ test unless cv_only) -------------
    t0 = time.perf_counter()
    final_caps = panel_caps_log2(pd_.lo[train_all], pd_.hi[train_all]) if cfg.panel_cap else None
    selected = _select_all(pd_, train_all, cfg, droplog, models, sd.has_unitigs)
    test_nearest = nearest.test[test_idx]
    final_fits: dict[str, tuple[Any, FoldFeatures, list[str], np.ndarray]] = {}
    for m in models:
        pred, model, feats_used, names = _fit_predict(pd_, m, train_all, test_idx, cfg, droplog, feats=selected[m])
        final_fits[m] = (model, feats_used, names, pred)
        if test_idx.size:
            preds.append(finish_predictions(pd_, test_idx, pred, params_by_model[m], SPLIT_TEST, m, run_id, test_nearest, config,
                                            caps=final_caps, marker_all=marker_all))
    if select is not None:
        base_fit, base_feats, base_names, base_pred = final_fits[base_model]
        b2_fit, _, _, b2_pred = final_fits[MODEL_B2]
        bundle_model = model_select.AftB2Select.from_parts(select.bundle_choice, base_fit, b2_fit, final_caps, base_names,
                                                          base_name=base_model)
        if test_idx.size:
            pred = model_select.combine_log2(apply_caps(base_pred, final_caps), apply_caps(b2_pred, final_caps),
                                             select.bundle_choice)
            preds.append(finish_predictions(pd_, test_idx, pred, select.bundle_params, SPLIT_TEST, MODEL_SELECT, run_id,
                                            test_nearest, config, caps=final_caps, marker_all=marker_all))
        _write_pair_bundle(
            paths, config, sd, pd_, cfg, bundle_model, base_feats, base_names, q_by_model[MODEL_SELECT],
            n_res[MODEL_SELECT], run_id, train_all.size, test_idx.size, inputs_sha1=inputs_sha1, caps=final_caps,
            q_cv={f: p_.q_up for f, p_ in select.params_cv.items()}, band_params=select.bundle_params,
            band_params_cv=select.params_cv, select_info=select.as_json(),
        )
    else:
        model, feats_used, names, _ = final_fits[main_model]
        _write_pair_bundle(
            paths, config, sd, pd_, cfg, model, feats_used, names, q_by_model[main_model], n_res[main_model], run_id,
            train_all.size, test_idx.size, inputs_sha1=inputs_sha1, caps=final_caps, q_cv=q_cv[main_model],
            band_params=params_by_model[main_model], band_params_cv=params_cv[main_model],
        )
    timings["final_s"] = time.perf_counter() - t0

    # --- LOLO (train lineages only; see _fit_plan) --------------------------------
    t0 = time.perf_counter()
    if cfg.lolo:
        for lineage, reason, n_rows in plan.skipped_lolo:
            logger.warning("%s: LOLO %s skipped (%s; %d row(s) of the lineage in this pair)", label, lineage, reason, n_rows)
        for reason in (LOLO_SKIP_NO_TRAIN_ROWS, LOLO_SKIP_TOO_FEW_FIT_ROWS):
            hits = [(lineage, n_rows) for lineage, r, n_rows in plan.skipped_lolo if r == reason]
            droplog.drop(
                reason,
                sum(n for _, n in hits),
                ", ".join(f"LOLO {lineage}" for lineage, _ in hits) if hits else "LOLO lineages",
            )
        for group in plan.lolo:
            lineage = str(group.lineage)
            lolo_caps = panel_caps_log2(pd_.lo[group.fit_idx], pd_.hi[group.fit_idx]) if cfg.panel_cap else None
            selected = _select_all(pd_, group.fit_idx, cfg, droplog, models, sd.has_unitigs)
            lolo_nearest = nearest.lolo[lineage][group.predict_idx]
            lolo_pred: dict[str, np.ndarray] = {}
            for m in models:
                try:
                    pred, _, _, _ = _fit_predict(pd_, m, group.fit_idx, group.predict_idx, cfg, droplog, feats=selected[m])
                except ValueError as error:
                    logger.warning("%s [%s] LOLO %s: %s; skipped", label, m, lineage, error)
                    continue
                lolo_pred[m] = pred
                preds.append(finish_predictions(pd_, group.predict_idx, pred, params_by_model[m], group.split, m, run_id, lolo_nearest, config,
                                                caps=lolo_caps, marker_all=marker_all))
            if select is not None:
                # The bundle's choice, applied to this lineage's own AFT / B2 fits (same rule as the bundle).
                missing = np.full(group.predict_idx.size, np.nan)
                pred = model_select.combine_log2(apply_caps(lolo_pred.get(base_model, missing), lolo_caps),
                                                 apply_caps(lolo_pred.get(MODEL_B2, missing), lolo_caps), select.bundle_choice)
                ok = ~np.isnan(pred)
                if ok.any():
                    preds.append(finish_predictions(pd_, group.predict_idx[ok], pred[ok], select.bundle_params, group.split,
                                                    MODEL_SELECT, run_id, lolo_nearest[ok], config, caps=lolo_caps,
                                                    marker_all=marker_all))
            logger.info("%s: LOLO %s done (%d fit, %d held-out train genomes)", label, lineage, group.fit_idx.size, group.predict_idx.size)
    timings["lolo_s"] = time.perf_counter() - t0

    # --- b0_resfinder (needs per-genome ResFinder output under data/interim) --------
    if paths.interim_root.is_dir():
        b0 = resfinder_predictions(paths, pd_, run_id, config, droplog, pheno_cache)
        if b0 is not None:
            if cfg.cv_only:
                b0 = b0.loc[b0["split"] != SPLIT_TEST]
            preds.append(b0)
    else:
        logger.info("%s: no %s (no per-genome tool output); b0_resfinder skipped", label, paths.interim_root)

    table = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=list(PREDS_COLUMNS))
    table, calibrations, fold_fits = _add_probabilities(table, pd_, config, marker_all)
    if select is not None and paths.model_dir(pd_.species, pd_.drug).is_dir():
        # The bundle always applies its own choice, so its map is fit on that candidate's OOF predictions.
        bundle_cal, bundle_fits = _select_calibration(select, pd_, config, marker_all)
        bundle_cal.save(
            paths.model_dir(pd_.species, pd_.drug) / prob_cal.CALIBRATION_FILE,
            extra={
                "model": main_model,
                "candidate": select.bundle_choice,
                "run_id": run_id,
                "call_standard": list(config.call_standard),
                "source": f"out-of-fold CV predictions of the bundle's candidate ({select.bundle_choice}) on every train "
                          "fold (test split never used)",
                "cross_fitted_by_fold": bundle_fits,
            },
        )
    elif main_model in calibrations and paths.model_dir(pd_.species, pd_.drug).is_dir():
        calibrations[main_model].save(
            paths.model_dir(pd_.species, pd_.drug) / prob_cal.CALIBRATION_FILE,
            extra={
                "model": main_model,
                "run_id": run_id,
                "call_standard": list(config.call_standard),
                "source": "out-of-fold CV predictions of this model on every train fold (test split never used)",
                "cross_fitted_by_fold": fold_fits.get(main_model, {}),
            },
        )
    table = _typed_preds(table)
    _check_preds(table, pd_, label)
    write_parquet(table, paths.preds(pd_.species, pd_.drug))
    logger.info("%s: wrote %d prediction rows (%d models) to %s", label, len(table), table["model"].nunique(), paths.preds(pd_.species, pd_.drug))
    n_test_rows = int((table["split"] == SPLIT_TEST).sum())
    if n_test_rows:
        append_test_ledger(paths, run_id, pd_.species, pd_.drug, n_test_rows, inputs_sha1=inputs_sha1)
    return PairResult(pd_.species, pd_.drug, table, main_model, q_by_model, n_res, timings)


def _add_probabilities(
    table: pd.DataFrame,
    pd_: PairData,
    config: Config,
    marker_all: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, prob_cal.PairCalibration], dict[str, dict[str, Any]]]:
    """Fill ``prob_works`` / ``prob_tier`` for every model with an MIC prediction.

    Per model: ``works`` = ``lab_sir_rederived == 'S'`` (call breakpoint; I/R = 0;
    straddling = unknown, skipped). CV rows of fold f get P from a calibration fit on the
    CV rows of the other folds only (:func:`~genome2mic.models.calibration.cross_fit_probabilities`);
    test and LOLO rows use the fit on every CV row, which is also what the bundle ships.
    Strong-marker rows (``marker_all``) use their own map or the 0.02 fallback; natural
    resistance gives 0. Pairs without a call breakpoint get null.

    Returns the table, the all-CV-row fit per model and a per-fold record per model.
    """
    out = table.copy()
    out["prob_works"] = np.nan
    out["prob_tier"] = None
    calibrations: dict[str, prob_cal.PairCalibration] = {}
    fold_fits: dict[str, dict[str, Any]] = {}
    if out.empty:
        return out, calibrations, fold_fits
    bp = config.call_breakpoint(pd_.species, pd_.drug)
    s_bp = float(bp.s_breakpoint) if bp is not None else None
    natural = config.is_naturally_resistant(pd_.species, pd_.drug)
    row_of = {str(g): i for i, g in enumerate(pd_.frame["genome_id"].astype(str))}
    folds = pd_.folds
    pred_all = pd.to_numeric(out["pred_mic"], errors="coerce").to_numpy(dtype=np.float64)
    split_all = out["split"].astype(object).to_numpy()
    model_all = out["model"].astype(object).to_numpy()
    prob = np.full(len(out), np.nan)
    for model in sorted({str(m) for m in model_all}):
        rows = np.flatnonzero(model_all == model)
        if np.isnan(pred_all[rows]).all():
            continue  # b0_resfinder: S/R only, no MIC
        pos = np.array([row_of[str(g)] for g in out["genome_id"].to_numpy()[rows]], dtype=np.int64)
        marker = np.asarray(marker_all, dtype=bool)[pos]
        cv = split_all[rows] == SPLIT_CV
        cv_rows = rows[cv]
        works = prob_cal.works_from_sir(out["lab_sir_rederived"].to_numpy(dtype=object)[cv_rows])
        p_cv, by_fold = prob_cal.cross_fit_probabilities(
            pred_all[cv_rows], works, marker[cv], folds[pos[cv]], s_bp, natural_resistance=natural,
        )
        prob[cv_rows] = p_cv
        full = prob_cal.fit_pair_calibration(pred_all[cv_rows], works, marker[cv], s_bp, natural_resistance=natural)
        other = rows[~cv]
        if other.size:
            prob[other] = full.predict(pred_all[other], marker[~cv])
        calibrations[model] = full
        fold_fits[model] = prob_cal.summarize_fits(by_fold)
    out["prob_works"] = prob
    out["prob_tier"] = prob_cal.prob_tier(prob)
    return out, calibrations, fold_fits


def _typed_preds(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ("pred_mic", "band_low", "band_high", "lab_lower", "lab_upper", "nearest_training_distance", "prob_works"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("genome_id", "species", "drug", "split", "pred_sir", "lab_sir", "model", "run_id", "external_set",
                "lab_sir_rederived", "prob_tier"):
        out[col] = out[col].astype(object).where(out[col].notna(), None).astype("string")
    if out["lab_exact"].isna().any():
        raise ContractViolation("lab_exact is null on some prediction rows", STAGE)
    out["lab_exact"] = out["lab_exact"].astype(bool)
    return out[list(PREDS_COLUMNS)]


def _check_preds(table: pd.DataFrame, pd_: PairData, label: str) -> None:
    """Stage-10 acceptance: unique (genome_id, model, split), bounds, rounding, splits.

    ``cv`` rows must be train genomes, ``test`` rows test genomes and
    ``lolo_<X>`` rows train genomes whose ``lolo_lineage`` is ``X``.
    """
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
        rows = table.loc[(table["split"] == eval_split).fillna(False).to_numpy(dtype=bool), "genome_id"]
        bad = [g for g in rows if split_of.get(g) != want]
        if bad:
            raise ContractViolation(f"{label}: {len(bad)} {eval_split} rows belong to non-{want} genomes", STAGE)
    lolo_of = {g: (None if pd.isna(v) else str(v)) for g, v in zip(pd_.frame["genome_id"], pd_.frame["lolo_lineage"])}
    for gid, split_value in zip(table["genome_id"], table["split"]):
        if not str(split_value).startswith(LOLO_PREFIX):
            continue
        lineage = str(split_value)[len(LOLO_PREFIX):]
        if split_of.get(gid) != SPLIT_TRAIN or lolo_of.get(gid) != lineage:
            raise ContractViolation(
                f"{label}: {split_value} row for {gid} (split={split_of.get(gid)}, lolo_lineage={lolo_of.get(gid)}); "
                "LOLO rows must be train genomes of that lineage",
                STAGE,
            )


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
    *,
    inputs_sha1: str | None = None,
    caps: tuple[float, float] | None = None,
    q_cv: Mapping[int, float] | None = None,
    band_params: BandParams | None = None,
    band_params_cv: Mapping[int, BandParams] | None = None,
    select_info: Mapping[str, Any] | None = None,
) -> None:
    """Write ``models/<SPECIES>/<drug>/`` for the prediction pipeline.

    ``select_info`` (``aft_b2_select`` bundles, :meth:`SelectOutcome.as_json`) is recorded
    as ``model_select`` in ``conformal.json`` and ``meta.json``.

    ``band_params`` (tuned on all OOF predictions) go to ``conformal.json``:
    ``q_up`` / ``q_low`` / ``alpha_up`` / ``alpha_low`` / ``active_gate_open``; ``q``
    repeats ``q_up`` for readers that only know the symmetric band. The pipeline builds
    the same asymmetric band and withholds ``likely_active`` when the gate is closed.

    ``caps`` (log2 mg/L, from all train rows) go to ``conformal.json``
    (``cap_low_log2``, ``cap_high_log2``) and ``meta.json``; the pipeline clips the raw
    prediction to them before rounding up, exactly as training did.

    ``features.json`` ``unitig_kmer_set_sha1`` names the k-mer set the model's unitig
    columns index (null for a model without unitig features); the prediction pipeline
    refuses a bundle whose species-level set differs.
    """
    target = paths.model_dir(pd_.species, pd_.drug)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    model.save(target)
    use_known, use_unitigs = _uses(getattr(model, "base_name", model.name))
    known_columns = list(feats.known_columns) if use_known else []
    unitig_cols = [_pattern_name(int(c)) for c in feats.unitig_cols] if use_unitigs else []
    select.assert_no_forbidden(names, require_prefix=True)
    kmer_set_sha1 = sd.unitig_kmer_set_sha1 if use_unitigs else None
    if use_unitigs and kmer_set_sha1 is None:
        raise ContractViolation(
            f"{pd_.species} x {pd_.drug}: {model.name} uses unitig features but {paths.unitig_kmers(pd_.species)} "
            "is missing; the bundle could not name (or ship) the k-mer set its columns index",
            STAGE,
        )
    write_json(
        target / "features.json",
        {
            "model_class": model.name,
            "known_columns": known_columns,
            "unitig_cols": unitig_cols,
            "class_by_column": {c: list(sd.class_by_column[c]) for c in known_columns if c in sd.class_by_column},
            "feature_names": names,
            "unitig_kmer_set_sha1": kmer_set_sha1,
        },
    )
    extra: dict[str, Any] = {"unit": "doubling steps", "source": "out-of-fold exact residuals"}
    if caps is not None:
        extra.update({"cap_low_log2": float(caps[0]), "cap_high_log2": float(caps[1])})
    if q_cv:
        extra["q_cross_conformal_by_fold"] = {str(k): float(v) for k, v in sorted(q_cv.items())}
    if band_params is not None:
        extra.update(band_params.as_json())
        extra["vme_target"] = float(cfg.band_vme_target)
        if band_params_cv:
            extra["band_cross_conformal_by_fold"] = {str(k): v.as_json() for k, v in sorted(band_params_cv.items())}
    if select_info is not None:
        extra["model_select"] = dict(select_info)
    write_conformal(target / "conformal.json", q, cfg.alpha, n_residuals, extra)
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
            "inputs_sha1": inputs_sha1,
            "unitig_kmer_set_sha1": kmer_set_sha1,
            "call_standard": list(config.call_standard),
            "panel_caps_log2": None if caps is None else [float(caps[0]), float(caps[1])],
            "strong_subclass_map": sd.subclass_source,
            "cv_only": bool(cfg.cv_only),
            "synthetic": (paths.raw_dir / SYNTHETIC_MARKER).is_file(),
            "model_select": None if select_info is None else {
                "choice": select_info.get("choice"), "base_model": select_info.get("base_model"),
                "choice_by_fold": select_info.get("choice_by_fold"),
            },
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


def write_species_bundle(paths: Paths, sd: SpeciesData, splits: pd.DataFrame) -> None:
    """``models/<SPECIES>/train_sketches.npz`` (+ unitig k-mer set and index copies) for one species."""
    species = sd.species
    sdir = paths.models_dir / species
    sdir.mkdir(parents=True, exist_ok=True)
    is_species = (splits["species"].astype(str) == species).to_numpy(dtype=bool)
    is_train = (splits["split"].astype(str) == SPLIT_TRAIN).to_numpy(dtype=bool)
    train_ids = set(splits.loc[is_species & is_train, "genome_id"].astype(str))
    keep = [i for i, g in enumerate(sd.sketch_ids) if g in train_ids]
    if not sd.has_sketches:
        logger.warning(
            "%s: no sketches available (imported release); models/%s/train_sketches.npz not written -- the "
            "prediction pipeline cannot compute nearest-training distances or load this species until it is added",
            species, species,
        )
    elif not keep:
        raise ContractViolation(f"{species}: no training sketches to ship", STAGE)
    else:
        sk.save_sketches(sdir / "train_sketches.npz", [sd.sketch_ids[i] for i in keep], sd.sketches[keep], k=sd.sketch_k)
    if sd.has_unitigs:
        current = _species_kmer_set_sha1(paths, species)
        if current is None or current != sd.unitig_kmer_set_sha1:
            raise ContractViolation(
                f"{species}: {paths.unitig_kmers(species)} changed while training (k-mer set "
                f"{(sd.unitig_kmer_set_sha1 or 'missing')[:12]} at load, {(current or 'missing')[:12]} now); "
                "the drug bundles would record a set that is not the one shipped",
                STAGE,
            )
        shutil.copyfile(paths.unitig_kmers(species), sdir / BUNDLE_UNITIG_KMERS)
        shutil.copyfile(paths.unitig_index(species), sdir / "unitig_index.parquet")
    logger.info("%s: shipped %d training sketches%s", species, len(keep), " + unitig k-mer set" if sd.has_unitigs else "")


def write_shared_bundle(
    paths: Paths,
    config: Config,
    trained: Mapping[str, Sequence[str]],
    run_id: str,
    *,
    full_run: bool = True,
    amrfinder_db: Path | None = None,
) -> None:
    """``manifest.json``, ``reference_sketches.npz`` and (synthetic runs only) ``markers.fasta``.

    Args:
        paths: Project paths.
        config: Loaded config (species references).
        trained: ``species -> drugs`` trained by this run.
        run_id: This run's id.
        full_run: The run covered every kept pair: the manifest is overwritten.
            Otherwise (a sharded/subset run) the pairs of an existing manifest that
            this run did not train are kept, with the run id that produced them in
            ``run_ids`` (pairs whose bundle directory is gone are dropped, logged).

    ``markers.fasta`` (the MarkerScan fallback) is written only for synthetic runs;
    on any other run a pre-existing file is deleted, so a real bundle can never
    fall back to the synthetic marker scan.
    """
    models_dir = paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    synthetic = (paths.raw_dir / SYNTHETIC_MARKER).is_file()

    # Reference sketches (species identification at prediction time).
    from genome2mic.qc import build_reference_sketches  # noqa: PLC0415

    refs = build_reference_sketches(paths, config)
    if refs is None and not paths.processed_dir.joinpath("IMPORTED_RELEASE.json").is_file():
        raise FileNotFoundError(
            f"no species reference genomes under {paths.references_dir} (or species.yaml reference_sketch); "
            "the prediction pipeline cannot identify species without models/reference_sketches.npz"
        )
    if refs is None:
        logger.warning(
            "imported release without species references: models/reference_sketches.npz not written; the "
            "prediction pipeline cannot identify species with this bundle until references are added"
        )
    else:
        sk.save_sketches(models_dir / "reference_sketches.npz", list(refs.species), refs.sketches, k=refs.k)

    markers = models_dir / "markers.fasta"
    if synthetic:
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
    elif markers.exists():
        markers.unlink()
        logger.warning(
            "non-synthetic run: removed stale %s (the MarkerScan fallback is for synthetic bundles only; "
            "real predictions need AMRFinderPlus)", markers,
        )

    species_map: dict[str, set[str]] = {sp: set(drugs) for sp, drugs in trained.items()}
    run_ids: dict[str, dict[str, str]] = {sp: {d: run_id for d in drugs} for sp, drugs in species_map.items()}
    if not full_run and paths.models_manifest.is_file():
        old = read_json(paths.models_manifest)
        old_run_ids = old.get("run_ids") if isinstance(old.get("run_ids"), dict) else {}
        kept = missing = 0
        for sp, drugs in (old.get("species") or {}).items():
            for drug in drugs or []:
                if drug in species_map.get(sp, set()):
                    continue
                if not paths.model_dir(sp, drug).is_dir():
                    missing += 1
                    logger.warning("manifest merge: %s x %s listed in the old manifest but %s is missing; dropped",
                                   sp, drug, paths.model_dir(sp, drug))
                    continue
                species_map.setdefault(sp, set()).add(drug)
                run_ids.setdefault(sp, {})[drug] = str((old_run_ids.get(sp) or {}).get(drug) or old.get("run_id"))
                kept += 1
        if bool(old.get("synthetic")) != synthetic:
            logger.warning("manifest merge: the existing manifest has synthetic=%s, this run synthetic=%s",
                           old.get("synthetic"), synthetic)
        logger.info("subset run: merged %d pair(s) from the existing manifest (%d dropped)", kept, missing)
    distinct = sorted({r for by_drug in run_ids.values() for r in by_drug.values()})
    if len(distinct) > 1:
        logger.warning("models/manifest.json now holds pairs from %d training runs: %s (see run_ids)", len(distinct), distinct)

    manifest = {
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "run_ids": {sp: {d: run_ids[sp][d] for d in sorted(drugs)} for sp, drugs in sorted(species_map.items())},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "species": {sp: sorted(drugs) for sp, drugs in sorted(species_map.items())},
        "synthetic": synthetic,
        "note": "Predictions of in-vitro susceptibility, not prescribing advice.",
    }
    imported = paths.processed_dir.joinpath("IMPORTED_RELEASE.json").is_file()
    if imported:
        # The release's known_amr columns follow the NCBI release rules, not features/known_amr.py:
        # the prediction pipeline must convert AMRFinderPlus output with predict/release_features.py.
        from genome2mic.predict import release_features  # noqa: PLC0415

        manifest["feature_naming"] = release_features.FEATURE_NAMING_NCBI_RELEASE
    write_json(paths.models_manifest, manifest)
    logger.info("wrote %s: %s", paths.models_manifest, {k: len(v) for k, v in manifest["species"].items()})
    if imported:
        try:
            release_features.write_specs_for_root(paths, config, amrfinder_db)
        except FileNotFoundError as error:
            logger.warning(
                "imported release: feature specs not written (%s); run `genome2mic release-feature-spec --root %s "
                "--amrfinder-db DIR` before predicting with this bundle", error, paths.root,
            )


# --------------------------------------------------------------------------- #
# Test ledger and drop log
# --------------------------------------------------------------------------- #


def _check_ledger_header(path: Path) -> bool:
    """Validate the header of an existing ledger; ``True`` when it is the legacy 5-column header.

    Raises :class:`ContractViolation` for any header other than :data:`LEDGER_COLUMNS`
    or :data:`LEGACY_LEDGER_COLUMNS` (an absent or empty file is fine).
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", newline="") as fh:
        header = tuple(h.strip() for h in next(csv.reader(fh), []))
    if header == LEDGER_COLUMNS:
        return False
    if header == LEGACY_LEDGER_COLUMNS:
        return True
    raise ContractViolation(
        f"{path} (test_ledger) has header {list(header)}, expected {list(LEDGER_COLUMNS)}; it is append-only, "
        "so it is never rewritten -- move the file aside if it is not a test ledger",
        STAGE,
    )


def _upgrade_legacy_ledger(path: Path) -> None:
    """Add the ``inputs_sha1`` column to a ledger written before it existed (once, atomically).

    Every existing row is kept, in order, with an empty ``inputs_sha1`` (unknown:
    the leakage check treats it as distinct from any recorded fingerprint). The new
    file is written next to the old one and moved over it, so a crash leaves either
    the old ledger or the upgraded one, never a truncated file.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    body = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    tmp = path.with_name(f"{path.name}.upgrade.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(LEDGER_COLUMNS)
        for row in body:
            writer.writerow([*row, ""])
    os.replace(tmp, path)
    logger.warning(
        "test ledger %s: upgraded the legacy header (added inputs_sha1; %d existing row(s) kept with an empty "
        "fingerprint, which the leakage check treats as unknown)", path, len(body),
    )


def append_test_ledger(
    paths: Paths,
    run_id: str,
    species: str,
    drug: str,
    n_test_rows: int,
    *,
    inputs_sha1: str,
) -> None:
    """Append one ``run_id,created_utc,species,drug,n_test_rows,inputs_sha1`` row to ``results/test_ledger.csv``.

    The file is created with its header when absent and never truncated (CLAUDE.md
    rule 8: every scoring of the test set leaves a trace; the leakage check fails a
    pair scored under more than one ``(run_id, inputs_sha1)``). A ledger with the
    legacy 5-column header is upgraded once (:func:`_upgrade_legacy_ledger`: rows
    kept, ``inputs_sha1`` empty); any other header stops the run. ``n_test_rows`` is
    the number of ``split == 'test'`` rows written to the pair's preds file (all
    models); ``inputs_sha1`` is :func:`compute_inputs_sha1` for the pair's species.
    """
    if not isinstance(inputs_sha1, str) or not inputs_sha1.strip():
        raise ValueError("inputs_sha1 must be a non-empty string (compute_inputs_sha1)")
    target = paths.test_ledger
    if _check_ledger_header(target):
        _upgrade_legacy_ledger(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fresh = not target.is_file() or target.stat().st_size == 0
    needs_newline = False
    if not fresh:
        with target.open("rb") as fh:
            fh.seek(-1, 2)
            needs_newline = fh.read(1) not in (b"\n", b"\r")
    with target.open("a", encoding="utf-8", newline="") as fh:
        if needs_newline:
            fh.write("\n")
        writer = csv.writer(fh, lineterminator="\n")
        if fresh:
            writer.writerow(LEDGER_COLUMNS)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        writer.writerow([run_id, created, species, drug, int(n_test_rows), inputs_sha1.strip()])
    logger.info(
        "test ledger: %s x %s scored on %d test row(s) under run %s, inputs %s -> %s",
        species, drug, n_test_rows, run_id, inputs_sha1, target,
    )


def _bundle_kmer_set_sha1(paths: Paths, species: str, drug: str, shipped: dict[str, str | None]) -> str | None:
    """The k-mer set an existing drug bundle was trained on (``None``: the model uses no unitig features).

    Read from ``features.json`` ``unitig_kmer_set_sha1``. A bundle written before
    that key existed but with unitig columns is assumed to match the species' shipped
    ``unitig_kmers.npz`` (``"unknown"`` when that file is missing too).
    """
    features_path = paths.model_dir(species, drug) / "features.json"
    features = read_json(features_path) if features_path.is_file() else {}
    recorded = features.get("unitig_kmer_set_sha1")
    if isinstance(recorded, str) and recorded:
        return recorded
    if not features.get("unitig_cols"):
        return None
    if species not in shipped:
        path = paths.models_dir / species / BUNDLE_UNITIG_KMERS
        if path.is_file():
            from genome2mic.features.unitigs import read_kmer_set_sha1  # noqa: PLC0415

            shipped[species] = read_kmer_set_sha1(path)
        else:
            shipped[species] = None
    return shipped[species] or "unknown"


def check_subset_unitig_sets(paths: Paths, by_species: Mapping[str, Sequence[str]]) -> None:
    """Refuse a subset run that would replace a species' shipped k-mer set under drugs it does not retrain.

    ``models/<SPECIES>/unitig_kmers.npz`` is shared by every drug of the species. For
    each species this run trains, the drugs listed in the existing
    ``models/manifest.json`` that the run does not retrain (and whose bundle
    directory exists, i.e. that the manifest merge keeps) must have been trained on
    the k-mer set this run ships; otherwise their unitig columns would silently index
    a different set. Nothing is checked when the run ships no set for the species
    (no unitig features: the shipped file is left alone) or there is no manifest.

    Raises:
        ContractViolation: naming the species, the new set and every conflicting drug
            with the set it was trained on.
    """
    if not paths.models_manifest.is_file():
        return
    manifest = read_json(paths.models_manifest)
    listed = manifest.get("species") if isinstance(manifest.get("species"), dict) else {}
    shipped: dict[str, str | None] = {}
    problems: list[str] = []
    for species, drugs in by_species.items():
        retrained = set(drugs)
        remaining = [
            str(d) for d in (listed.get(species) or [])
            if str(d) not in retrained and paths.model_dir(species, str(d)).is_dir()
        ]
        if not remaining:
            continue
        new = _species_kmer_set_sha1(paths, species)
        if new is None:
            continue
        conflicts = []
        for drug in sorted(remaining):
            old = _bundle_kmer_set_sha1(paths, species, drug, shipped)
            if old is not None and old != new:
                conflicts.append(f"{drug} (trained on {old[:12]})")
        if conflicts:
            problems.append(
                f"{species}: this subset run would replace models/{species}/{BUNDLE_UNITIG_KMERS} with k-mer set "
                f"{new[:12]}, but {len(conflicts)} drug model(s) it does not retrain stay in models/manifest.json "
                f"and index a different set: {', '.join(conflicts)}"
            )
    if problems:
        raise ContractViolation(
            "; ".join(problems) + ". Their predictions would silently change. Retrain every drug of the species "
            "in one run (train --species <SPECIES> without --drugs, or list all of its drugs) or run a full train.",
            STAGE,
        )


def _write_drop_log(paths: Paths, droplog: DropLog, labels: Sequence[str], *, full_run: bool) -> None:
    """Write ``drop_log_train.csv``; a subset run keeps the existing rows of the pairs it did not train.

    Rows belong to a pair when their ``detail`` is ``"<SPECIES> x <drug>"`` or starts
    with ``"<SPECIES> x <drug>:"`` (every record :class:`_PairDropLog` writes).
    """
    target = paths.drop_log(STAGE)
    if full_run or not target.is_file():
        droplog.write(target)
        return
    own = set(labels)
    old = pd.read_csv(target, dtype={"stage": "str", "reason": "str", "detail": "str"})
    mine = np.array([isinstance(d, str) and d.split(":", 1)[0] in own for d in old["detail"]], dtype=bool)
    merged = pd.concat([old.loc[~mine], droplog.to_frame()], ignore_index=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target, index=False)
    logger.info(
        "[%s] subset run: replaced %d drop-log row(s) of %d pair(s), kept %d row(s) of other pairs -> %s",
        STAGE, int(mine.sum()), len(own), int((~mine).sum()), target,
    )


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


def _in_scope(config: Config, pairs: Iterable[tuple[Any, Any]]) -> list[tuple[str, str]]:
    """``(species, drug)`` pairs whose species and (normalized) drug name are in the config, de-duplicated in order."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for s, d in pairs:
        pair = (str(s), str(d))
        if pair in seen or pair[0] not in config.species or config.normalize_drug(pair[1]) != pair[1]:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def select_pairs(
    paths: Paths,
    config: Config,
    *,
    species: Sequence[str] | None = None,
    drugs: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    """Kept, in-scope pairs from ``pairs_kept.csv`` restricted to ``species`` and/or ``drugs``.

    Species keys are upper-cased and drug names normalized through ``drugs.yaml``.
    Used by ``train --species ... --drugs ...`` to shard training.

    Raises:
        FileNotFoundError: ``pairs_kept.csv`` is missing.
        ValueError: the filters match no kept pair.
    """
    if not paths.pairs_kept.is_file():
        raise FileNotFoundError(f"training input missing: {paths.pairs_kept}")
    kept = pd.read_csv(paths.pairs_kept, dtype={"species": str, "drug": str})
    pairs = _in_scope(config, zip(kept["species"], kept["drug"]))
    if species:
        want_species = {str(sp).strip().upper() for sp in species}
        pairs = [p for p in pairs if p[0] in want_species]
    if drugs:
        want_drugs = {config.normalize_drug(str(d)) or str(d).strip().lower() for d in drugs}
        pairs = [p for p in pairs if p[1] in want_drugs]
    if not pairs:
        raise ValueError(
            f"no kept pair matches species={list(species) if species else 'any'} drugs={list(drugs) if drugs else 'any'}; "
            f"kept pairs: {_in_scope(config, zip(kept['species'], kept['drug']))}"
        )
    return pairs


def run(
    paths: Paths,
    config: Config,
    *,
    train_config: TrainConfig | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    amrfinder_db: Path | None = None,
) -> pd.DataFrame:
    """Train every kept species x drug pair; write preds, bundles and ``drop_log_train.csv``.

    Species are processed one at a time: the species' pair frames are joined,
    one distance pass computes every nearest-training distance of its drugs, the
    pairs train, the species' sketches/unitig copies are shipped to ``models/``,
    and the species' arrays are released before the next species is loaded.

    Args:
        paths: Project paths (reads the stage 2/3/5/6/7/8 files, writes ``results/``
            and ``models/``).
        config: Loaded project config.
        train_config: Training knobs (defaults: :class:`TrainConfig`).
        pairs: Optional subset of ``(species, drug)`` pairs; default = ``pairs_kept.csv``.
            A subset that is not every kept pair merges into the existing
            ``models/manifest.json`` and ``drop_log_train.csv`` instead of
            replacing them, and refuses to start when it would replace a
            species' shipped k-mer set under drugs it does not retrain
            (:func:`check_subset_unitig_sets`).
        amrfinder_db: AMRFinderPlus database directory (``fam.tsv``, ``AMRProt.fa``) used
            to give every ``gene_`` column its Subclass, so training-time calls apply the
            ``strong_subclasses`` override (e.g. any acquired carbapenemase for the
            carbapenems) exactly like the prediction pipeline. Default: the database next
            to ``amrfinder`` on ``PATH``; without one only the subclasses already in
            ``known_amr_columns.csv`` count (logged as a warning).

    Raises:
        ContractViolation: a subset run would swap the k-mer set under other drugs'
            bundles, or an input breaks a stage contract.

    Returns:
        One summary row per pair: species, drug, main_model, n_rows, q per model, timings.
    """
    cfg = train_config or TrainConfig()
    started = time.perf_counter()
    droplog = DropLog(STAGE)
    labels, splits, qc, known, lineages, pairs_kept = _read_inputs(paths)
    _check_ledger_header(paths.test_ledger)
    run_id = compute_run_id(cfg, paths.splits)
    logger.info("train: run_id=%s config=%s", run_id, cfg.as_dict())

    all_kept = _in_scope(config, zip(pairs_kept["species"], pairs_kept["drug"]))
    if pairs is None:
        wanted = all_kept
    else:
        requested = [(str(s), str(d)) for s, d in pairs]
        wanted = _in_scope(config, requested)
        dropped = sorted(set(requested) - set(wanted))
        if dropped:
            logger.warning("train: %d requested pair(s) out of scope (species/drug not in configs): %s", len(dropped), dropped)
    if not wanted:
        raise ValueError("no species x drug pairs to train (pairs_kept.csv empty or out of scope)")
    full_run = set(wanted) == set(all_kept)
    if not full_run:
        logger.info("train: subset run of %d of %d kept pair(s); manifest and drop log are merged", len(wanted), len(all_kept))

    by_species: dict[str, list[str]] = {}
    for species, drug in wanted:
        by_species.setdefault(species, []).append(drug)
    if not full_run:
        check_subset_unitig_sets(paths, by_species)

    subclass_tables, subclass_source = _subclass_tables(paths, amrfinder_db)
    class_map = _class_map(paths, subclass_tables)
    digest_cache: dict[Path, str] = {}
    trained: dict[str, list[str]] = {}
    summaries: list[dict[str, Any]] = []
    pheno_cache: dict[str, dict[str, str] | None] = {}
    for species, drugs in by_species.items():
        inputs_sha1 = compute_inputs_sha1(paths, species, cache=digest_cache)
        logger.info("%s: inputs_sha1=%s (labels, features, unitig set, configs, model code)", species, inputs_sha1)
        sd = _load_species_data(paths, species, known, class_map)
        sd.subclass_source = subclass_source
        logs: dict[str, _PairDropLog] = {}
        frames: dict[str, pd.DataFrame] = {}
        plans: dict[str, FitPlan] = {}
        for drug in drugs:
            logs[drug] = _PairDropLog(STAGE, _pair_label(species, drug))
            frame = _pair_frame(species, drug, labels, splits, qc, lineages, sd, logs[drug])
            if cfg.cv_only:
                # The test split is never touched: its label rows are not even loaded.
                is_train = frame["split"].astype(object).to_numpy() == SPLIT_TRAIN
                frame = logs[drug].keep_where(frame, is_train, "cv_only: non-train (test) rows not loaded", _pair_label(species, drug))
                frame = frame.reset_index(drop=True)
            frames[drug] = frame
            plans[drug] = _fit_plan(frames[drug], cfg)
        nearest = _species_nearest(
            sd, {drug: (sd.sketch_position(frames[drug]["genome_id"].tolist()), plans[drug]) for drug in drugs}
        )
        state = _PairJobState(
            paths=paths, config=config, cfg=cfg, sd=sd, species=species, frames=frames, plans=plans,
            nearest=nearest, logs=logs, run_id=run_id, inputs_sha1=inputs_sha1, pheno_cache=pheno_cache,
        )
        for drug, summary_row, pair_log in _run_pair_jobs(state, drugs, cfg.workers):
            droplog.extend(pair_log)
            trained.setdefault(species, []).append(drug)
            summaries.append(summary_row)
        write_species_bundle(paths, sd, splits)
        del sd, frames, plans, nearest
        logger.info("%s: done; species arrays released", species)

    write_shared_bundle(paths, config, trained, run_id, full_run=full_run, amrfinder_db=amrfinder_db)
    _write_drop_log(paths, droplog, [_pair_label(s, d) for s, d in wanted], full_run=full_run)
    summary = pd.DataFrame(summaries)
    logger.info("train stage finished in %.1fs:\n%s", time.perf_counter() - started, summary.to_string(index=False))
    return summary


@dataclass
class _PairJobState:
    """Everything one species' pair jobs read; sent once to each worker process."""

    paths: Paths
    config: Config
    cfg: TrainConfig
    sd: SpeciesData
    species: str
    frames: dict[str, pd.DataFrame]
    plans: dict[str, FitPlan]
    nearest: dict[str, PairNearest]
    logs: dict[str, _PairDropLog]
    run_id: str
    inputs_sha1: str
    pheno_cache: dict[str, dict[str, str] | None]


_JOB_STATE: _PairJobState | None = None


def _train_one_pair(state: _PairJobState, drug: str) -> tuple[str, dict[str, Any], _PairDropLog]:
    """Train one pair of ``state.species``; return its summary row and drop log."""
    species = state.species
    t0 = time.perf_counter()
    pd_ = _pair_data(species, drug, state.frames[drug], state.sd)
    result = train_pair(
        state.paths, state.config, state.cfg, state.sd, pd_, state.run_id, state.logs[drug], state.pheno_cache,
        plan=state.plans[drug], nearest=state.nearest[drug], inputs_sha1=state.inputs_sha1,
    )
    elapsed = time.perf_counter() - t0
    row = {
        "species": species,
        "drug": drug,
        "main_model": result.main_model,
        "n_rows": pd_.n,
        "n_train": int(pd_.train_mask.sum()),
        "n_test": int(pd_.test_mask.sum()),
        "n_preds": len(result.preds),
        "models": ",".join(sorted(result.preds["model"].unique())),
        "q_main": result.q_by_model.get(result.main_model),
        "inputs_sha1": state.inputs_sha1,
        "seconds": round(elapsed, 1),
        **{f"q_{m}": q for m, q in result.q_by_model.items()},
    }
    logger.info("%s x %s: trained in %.1fs (%s)", species, drug, elapsed, result.timings)
    return drug, row, state.logs[drug]


def _init_pair_worker(state: _PairJobState, log_level: int) -> None:
    """Worker initializer: receive the species state once and route logs to stderr."""
    global _JOB_STATE
    _JOB_STATE = state
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("genome2mic").setLevel(log_level)


def _pair_job_in_worker(drug: str) -> tuple[str, dict[str, Any], _PairDropLog]:
    """Worker entry point (state set by :func:`_init_pair_worker`)."""
    if _JOB_STATE is None:  # pragma: no cover - initializer always runs first
        raise RuntimeError("pair job state missing in worker")
    return _train_one_pair(_JOB_STATE, drug)


def _run_pair_jobs(
    state: _PairJobState, drugs: Sequence[str], workers: int
) -> Iterator[tuple[str, dict[str, Any], _PairDropLog]]:
    """Train ``drugs`` sequentially or in ``workers`` processes; yield results in ``drugs`` order.

    Workers are *spawned* (fresh interpreters), never forked: a fork after xgboost's
    OpenMP runtime has started in the parent deadlocks the child in libomp. The
    species state is pickled once per worker. Each pair writes its own preds file
    and bundle directory, so workers never write the same file; the species-level
    and shared outputs are written by the caller after every pair has finished.
    """
    n_workers = min(int(workers), len(drugs))
    if n_workers <= 1:
        for drug in drugs:
            yield _train_one_pair(state, drug)
        return
    logger.info("%s: training %d pair(s) in %d worker process(es)", state.species, len(drugs), n_workers)
    level = logging.getLogger("genome2mic").getEffectiveLevel()
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=mp.get_context("spawn"),
        initializer=_init_pair_worker, initargs=(state, level),
    ) as pool:
        futures = [pool.submit(_pair_job_in_worker, drug) for drug in drugs]
        for future in futures:
            yield future.result()


def load_manifest(paths: Paths) -> dict[str, Any]:
    """Read ``models/manifest.json``."""
    return read_json(paths.models_manifest)
