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
       models/<SPECIES>/<drug>/model.ubj + params.json      (XgbAft.save)
       models/<SPECIES>/<drug>/features.json  model_class, known_columns, unitig_cols, class_by_column,
                                              feature_names, unitig_kmer_set_sha1
       models/<SPECIES>/<drug>/conformal.json q, alpha, n_residuals
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
import os
import re
import shutil
import time
import zipfile
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
from genome2mic.mic import GRID_MAX_EXPONENT, GRID_MIN_EXPONENT, lab_exact_mask, round_up_to_step_array
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
)
"""Stage-10 contract columns plus the v0.2 additions ``external_set``,
``nearest_training_distance``, ``lab_sir_rederived`` and ``lab_exact`` (bool: the lab
result is an exact measured MIC, :func:`genome2mic.mic.lab_exact_mask`)."""

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

    ids, sketches, k = sk.load_sketches(paths.sketches(species))
    logger.info("%s: %d sketches (k=%d, s=%d)", species, len(ids), k, sketches.shape[1] if sketches.ndim == 2 else 0)

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
) -> Any:
    """Fit a model, passing ``groups``/``droplog``/``exact_rows`` only to the models that accept them.

    ``exact_rows`` (the fit rows' ``lab_exact``) restricts B2's exact-MIC training rows;
    the interval models (B1, AFT) use every interval, disk rows included.
    """
    if model.name == MODEL_B1:
        return model.fit(X, lo, hi, names)
    if model.name == MODEL_B2:
        return model.fit(X, lo, hi, names, groups=groups, droplog=droplog, exact_rows=exact_rows)
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
         exact_rows=pd_.lab_exact[train_idx])
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
    test = FitGroup(SPLIT_TEST, np.flatnonzero(test_mask), train_all)

    lolo: list[FitGroup] = []
    skipped_lolo: list[tuple[str, str, int]] = []
    if cfg.lolo:
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


def _conformal(pred_log2_oof: np.ndarray, pd_: PairData, cfg: TrainConfig, droplog: DropLog, label: str) -> tuple[float, int]:
    """Conformal ``q`` from the OOF predictions of the train rows (``lab_exact`` rows only)."""
    clipped = np.clip(pred_log2_oof, GRID_MIN_EXPONENT, GRID_MAX_EXPONENT)
    pred_mic = round_up_to_step_array(2.0**clipped)
    mask = ~np.isnan(pred_mic)
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

    # --- CV: out-of-fold predictions, in-fold selection ------------------------
    oof: dict[str, np.ndarray] = {m: np.full(pd_.n, np.nan) for m in models}
    oof_nearest = nearest.cv
    t0 = time.perf_counter()
    for f, n_val, n_fit in plan.skipped_folds:
        logger.warning("%s: fold %d has %d validation / %d fit rows; skipped", label, f, n_val, n_fit)
    for group in plan.cv:
        selected = _select_all(pd_, group.fit_idx, cfg, droplog, models, sd.has_unitigs)
        for m in models:
            try:
                pred, _, _, _ = _fit_predict(pd_, m, group.fit_idx, group.predict_idx, cfg, droplog, feats=selected[m])
            except ValueError as error:
                logger.warning("%s [%s] fold %d: %s; fold predictions are null", label, m, group.fold, error)
                continue
            oof[m][group.predict_idx] = pred
        logger.info("%s: fold %d done (%d fit, %d validation rows)", label, group.fold, group.fit_idx.size, group.predict_idx.size)
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
    test_nearest = nearest.test[test_idx]
    for m in models:
        pred, model, feats_used, names = _fit_predict(pd_, m, train_all, test_idx, cfg, droplog, feats=selected[m])
        if test_idx.size:
            preds.append(finish_predictions(pd_, test_idx, pred, q_by_model[m], SPLIT_TEST, m, run_id, test_nearest, config))
        if m == main_model:
            _write_pair_bundle(
                paths, config, sd, pd_, cfg, model, feats_used, names, q_by_model[m], n_res[m], run_id,
                train_all.size, test_idx.size, inputs_sha1=inputs_sha1,
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
            selected = _select_all(pd_, group.fit_idx, cfg, droplog, models, sd.has_unitigs)
            lolo_nearest = nearest.lolo[lineage][group.predict_idx]
            for m in models:
                try:
                    pred, _, _, _ = _fit_predict(pd_, m, group.fit_idx, group.predict_idx, cfg, droplog, feats=selected[m])
                except ValueError as error:
                    logger.warning("%s [%s] LOLO %s: %s; skipped", label, m, lineage, error)
                    continue
                preds.append(finish_predictions(pd_, group.predict_idx, pred, q_by_model[m], group.split, m, run_id, lolo_nearest, config))
            logger.info("%s: LOLO %s done (%d fit, %d held-out train genomes)", label, lineage, group.fit_idx.size, group.predict_idx.size)
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
    n_test_rows = int((table["split"] == SPLIT_TEST).sum())
    if n_test_rows:
        append_test_ledger(paths, run_id, pd_.species, pd_.drug, n_test_rows, inputs_sha1=inputs_sha1)
    return PairResult(pd_.species, pd_.drug, table, main_model, q_by_model, n_res, timings)


def _typed_preds(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for col in ("pred_mic", "band_low", "band_high", "lab_lower", "lab_upper", "nearest_training_distance"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ("genome_id", "species", "drug", "split", "pred_sir", "lab_sir", "model", "run_id", "external_set", "lab_sir_rederived"):
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
) -> None:
    """Write ``models/<SPECIES>/<drug>/`` for the prediction pipeline.

    ``features.json`` ``unitig_kmer_set_sha1`` names the k-mer set the model's unitig
    columns index (null for a model without unitig features); the prediction pipeline
    refuses a bundle whose species-level set differs.
    """
    target = paths.model_dir(pd_.species, pd_.drug)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    model.save(target)
    use_known, use_unitigs = _uses(model.name)
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
            "inputs_sha1": inputs_sha1,
            "unitig_kmer_set_sha1": kmer_set_sha1,
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


def write_species_bundle(paths: Paths, sd: SpeciesData, splits: pd.DataFrame) -> None:
    """``models/<SPECIES>/train_sketches.npz`` (+ unitig k-mer set and index copies) for one species."""
    species = sd.species
    sdir = paths.models_dir / species
    sdir.mkdir(parents=True, exist_ok=True)
    is_species = (splits["species"].astype(str) == species).to_numpy(dtype=bool)
    is_train = (splits["split"].astype(str) == SPLIT_TRAIN).to_numpy(dtype=bool)
    train_ids = set(splits.loc[is_species & is_train, "genome_id"].astype(str))
    keep = [i for i, g in enumerate(sd.sketch_ids) if g in train_ids]
    if not keep:
        raise ContractViolation(f"{species}: no training sketches to ship", STAGE)
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
    if refs is None:
        raise FileNotFoundError(
            f"no species reference genomes under {paths.references_dir} (or species.yaml reference_sketch); "
            "the prediction pipeline cannot identify species without models/reference_sketches.npz"
        )
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
    write_json(paths.models_manifest, manifest)
    logger.info("wrote %s: %s", paths.models_manifest, {k: len(v) for k, v in manifest["species"].items()})


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

    class_map = _class_map(paths)
    digest_cache: dict[Path, str] = {}
    trained: dict[str, list[str]] = {}
    summaries: list[dict[str, Any]] = []
    pheno_cache: dict[str, dict[str, str] | None] = {}
    for species, drugs in by_species.items():
        inputs_sha1 = compute_inputs_sha1(paths, species, cache=digest_cache)
        logger.info("%s: inputs_sha1=%s (labels, features, unitig set, configs, model code)", species, inputs_sha1)
        sd = _load_species_data(paths, species, known, class_map)
        logs: dict[str, _PairDropLog] = {}
        frames: dict[str, pd.DataFrame] = {}
        plans: dict[str, FitPlan] = {}
        for drug in drugs:
            logs[drug] = _PairDropLog(STAGE, _pair_label(species, drug))
            frames[drug] = _pair_frame(species, drug, labels, splits, qc, lineages, sd, logs[drug])
            plans[drug] = _fit_plan(frames[drug], cfg)
        nearest = _species_nearest(
            sd, {drug: (sd.sketch_position(frames[drug]["genome_id"].tolist()), plans[drug]) for drug in drugs}
        )
        for drug in drugs:
            t0 = time.perf_counter()
            pd_ = _pair_data(species, drug, frames[drug], sd)
            result = train_pair(
                paths, config, cfg, sd, pd_, run_id, logs[drug], pheno_cache,
                plan=plans[drug], nearest=nearest[drug], inputs_sha1=inputs_sha1,
            )
            droplog.extend(logs[drug])
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
                    "inputs_sha1": inputs_sha1,
                    "seconds": round(elapsed, 1),
                    **{f"q_{m}": q for m, q in result.q_by_model.items()},
                }
            )
            logger.info("%s x %s: trained in %.1fs (%s)", species, drug, elapsed, result.timings)
            del pd_, result
        write_species_bundle(paths, sd, splits)
        del sd, frames, plans, nearest
        logger.info("%s: done; species arrays released", species)

    write_shared_bundle(paths, config, trained, run_id, full_run=full_run)
    _write_drop_log(paths, droplog, [_pair_label(s, d) for s, d in wanted], full_run=full_run)
    summary = pd.DataFrame(summaries)
    logger.info("train stage finished in %.1fs:\n%s", time.perf_counter() - started, summary.to_string(index=False))
    return summary


def load_manifest(paths: Paths) -> dict[str, Any]:
    """Read ``models/manifest.json``."""
    return read_json(paths.models_manifest)
