"""Evaluation stage: ``results/preds_*.parquet`` -> ``metrics.parquet`` + ``metrics_by_distance.parquet``.

Also (v0.6): ``prob_summary`` (the P(drug works) table per species x model x split, danger
first), ``prob_calibration`` (probability bins vs the observed share that worked, per
species and pooled as ``ALL``) and ``call_vme_by_fold`` (out-of-fold call VME per CV fold
with its one-sided exact binomial p-value against 1.5 %), each as Parquet and CSV.

``run(paths, config)`` stacks every predictions file, calls
:func:`genome2mic.eval.metrics.summarize` (one row per species x drug x model x
evaluation set, **VME first**) and :func:`genome2mic.eval.metrics.by_distance_bin`
on the ``nearest_training_distance`` column written by the training stage, and
writes both tables as Parquet and as CSV (the CSVs are what the demo report copies).
``report.run`` is a separate stage (``genome2mic.eval.report``); ``run-all`` calls
both.

Everything here consumes the stage-10 predictions table only. The metrics describe
agreement with in-vitro lab results on (synthetic or real) genomes; they are not
clinical outcomes. EA, exact agreement and band coverage are over exact (one-step)
lab MICs (``n_exact`` / ``n_band``); the categorical metrics come as reported and,
when the preds carry ``lab_sir_rederived``, re-derived under the call breakpoint.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from genome2mic.droplog import DropLog
from genome2mic.eval import metrics
from genome2mic.io import write_csv, write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "evaluate"
DISTANCE_COLUMN = "nearest_training_distance"

__all__ = ["STAGE", "load_preds", "run", "vme_first"]


def load_preds(paths: Paths) -> pd.DataFrame:
    """Concatenate every ``results/preds_*.parquet`` (raises when there is none)."""
    files = sorted(paths.results_dir.glob("preds_*.parquet")) if paths.results_dir.is_dir() else []
    if not files:
        raise FileNotFoundError(f"no results/preds_*.parquet under {paths.results_dir}; run the train stage first")
    frames = [pd.read_parquet(f, engine="pyarrow") for f in files]
    preds = pd.concat(frames, ignore_index=True)
    logger.info("loaded %d prediction rows from %d files", len(preds), len(files))
    return preds


def vme_first(frame: pd.DataFrame) -> pd.DataFrame:
    """Assert the contract's 'VME first' rule on a metrics table (first metric column is ``vme_rate``)."""
    metric_cols = [c for c in frame.columns if c in metrics.METRIC_COLUMNS]
    if metric_cols and metric_cols[0] != "vme_rate":
        raise ValueError(f"metrics table must report vme_rate first, got {metric_cols[0]!r}")
    return frame


def run(paths: Paths, config: Any = None, *, bins: tuple[float, ...] = metrics.DEFAULT_DISTANCE_BINS) -> pd.DataFrame:
    """Write ``metrics.parquet``/``.csv`` and ``metrics_by_distance.parquet``/``.csv``; return the metrics table.

    Args:
        paths: Project paths.
        config: Unused (kept for the uniform ``run(paths, config)`` stage signature).
        bins: Mash-distance edges for the distance stratification.
    """
    started = time.perf_counter()
    log = DropLog(STAGE)
    preds = load_preds(paths)

    table = vme_first(metrics.summarize(preds, drop_log=log))
    write_parquet(table, paths.metrics)
    write_csv(table, _csv_path(paths.metrics))

    if DISTANCE_COLUMN in preds.columns:
        by_dist = vme_first(metrics.by_distance_bin(preds, preds[DISTANCE_COLUMN].to_numpy(dtype=float), bins=bins, drop_log=log))
    else:
        logger.warning("preds have no %s column; metrics_by_distance is empty", DISTANCE_COLUMN)
        by_dist = pd.DataFrame(columns=list(metrics.DISTANCE_COLUMNS))
    write_parquet(by_dist, paths.metrics_by_distance)
    write_csv(by_dist, _csv_path(paths.metrics_by_distance))

    # P(drug works) (v0.6): the per-species table, the calibration check and the per-fold call VME.
    prob = metrics.probability_summary(preds)
    write_parquet(prob, paths.prob_summary)
    write_csv(prob, _csv_path(paths.prob_summary))
    cal = metrics.calibration_table(preds)
    pooled = metrics.calibration_table(preds.assign(species="ALL")) if not preds.empty else cal
    cal = pd.concat([cal, pooled], ignore_index=True) if not cal.empty else pooled
    write_parquet(cal, paths.prob_calibration)
    write_csv(cal, _csv_path(paths.prob_calibration))
    if paths.splits.is_file():
        splits = pd.read_parquet(paths.splits, columns=["genome_id", "fold"]).drop_duplicates("genome_id")
        folds = pd.Series(splits["fold"].to_numpy(), index=splits["genome_id"].astype(str))
        by_fold = metrics.call_vme_by_fold(preds, folds)
    else:
        logger.warning("%s missing; call_vme_by_fold is empty", paths.splits)
        by_fold = pd.DataFrame(columns=list(metrics.FOLD_VME_COLUMNS))
    write_parquet(by_fold, paths.call_vme_by_fold)
    write_csv(by_fold, _csv_path(paths.call_vme_by_fold))
    n_sig = int(by_fold["significant"].sum()) if not by_fold.empty else 0
    if n_sig:
        logger.warning("%d calling CV fold(s) have call VME significantly above 1.5%% (binomial p < 0.01); "
                       "see %s", n_sig, paths.call_vme_by_fold)
    log.write(paths.drop_log(STAGE))

    headline_cols = ["species", "drug", "model", "vme_rate", "me_rate", "vme_rate_rederived", "essential_agreement", "n_exact", "categorical_agreement", "n"]
    headline = table.loc[table["split"] == "test", [c for c in headline_cols if c in table.columns]]
    logger.info(
        "evaluate finished in %.1fs: %d metric rows, %d distance-bin rows. Test set (VME first):\n%s",
        time.perf_counter() - started, len(table), len(by_dist), headline.to_string(index=False),
    )
    return table


def _csv_path(parquet: Path) -> Path:
    return parquet.with_suffix(".csv")
