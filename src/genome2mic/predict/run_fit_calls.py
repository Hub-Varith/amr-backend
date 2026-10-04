"""CLI: fit the probability calibration and the call thresholds from a run's out-of-fold predictions.

Writes into the run folder:
  probability_calibration.json   calibration curves, fitted on all out-of-fold rows
  call_thresholds.json           thresholds on the calibrated probability
  calls_report.csv               per species x drug, VME first
  confidence_levels_report.csv   per species x level: shown chance vs observed rate
Both reports score each fold with a calibration and thresholds fitted on the other folds.
Test rows are never read.

Example:
    python -m genome2mic.predict.run_fit_calls --run-dir models/multitask
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.models.model_artifact import ModelArtifact
from genome2mic.predict.call_thresholds import CallThresholds
from genome2mic.predict.confidence_levels import ConfidenceLevels
from genome2mic.predict.constants import (
    CALL_STANDARD,
    CALL_YEAR,
    CONFIDENCE_LEVEL_NAMES,
    ME_TARGET,
    MIN_PER_CLASS,
    VME_TARGET,
)
from genome2mic.predict.probability_calibrator import ProbabilityCalibrator
from genome2mic.predict.susceptibility_caller import SusceptibilityCaller

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Folder with model.pt, spec.json, preds_oof.parquet")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--vme-target", type=float, default=VME_TARGET)
    parser.add_argument("--me-target", type=float, default=ME_TARGET)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    artifact = ModelArtifact.load(args.run_dir)
    sigma_by_drug = dict(zip(artifact.drugs, np.exp(artifact.model.log_sigma.detach().numpy())))
    breakpoints = BreakpointTable.from_directory(args.configs_dir / "breakpoints")
    splits = pd.read_parquet(args.processed_dir / "splits.parquet")
    predictions = pd.read_parquet(args.run_dir / "preds_oof.parquet")
    predictions = predictions.merge(splits.loc[splits["split"] == "train", ["genome_id", "fold"]], on="genome_id")

    frames = []
    for (species, drug), group in predictions.groupby(["species", "drug"]):
        found = breakpoints.lookup(species, drug, CALL_STANDARD, CALL_YEAR)
        if found is None:
            continue
        s_breakpoint, r_breakpoint = found
        lab_susceptible = group["lab_upper"] <= s_breakpoint
        lab_resistant = group["lab_lower"] >= r_breakpoint
        decided = group[lab_susceptible | lab_resistant].copy()
        logger.info("%s %s: rows=%s lab category known=%s dropped (I or spans a breakpoint)=%s", species, drug,
                    len(group), len(decided), len(group) - len(decided))
        decided["lab_resistant"] = lab_resistant[decided.index]
        decided["p_raw"] = SusceptibilityCaller.probability_active(
            decided["mu_log2"].to_numpy(), sigma_by_drug[drug], s_breakpoint
        )
        frames.append(decided)
    rows = pd.concat(frames, ignore_index=True)

    # Final artifacts: fitted on every out-of-fold row.
    calibrator = ProbabilityCalibrator.fit(rows, MIN_PER_CLASS)
    rows["p_active"] = np.nan
    for (species, drug), group in rows.groupby(["species", "drug"]):
        rows.loc[group.index, "p_active"] = calibrator.calibrate(species, drug, group["p_raw"].to_numpy())
    thresholds = CallThresholds.fit(rows, args.vme_target, args.me_target, MIN_PER_CLASS)
    (args.run_dir / "probability_calibration.json").write_text(json.dumps(calibrator.to_dict()))
    (args.run_dir / "call_thresholds.json").write_text(json.dumps(thresholds.to_dict(), indent=2))

    # Honest score: nothing from the held-out fold is used to fit its calibration or thresholds.
    rows["held_out_p_active"] = np.nan
    rows["call"] = "uncertain"
    for fold in sorted(rows["fold"].unique()):
        held_out = rows["fold"] == fold
        fit_rows = rows[~held_out].copy()
        fold_calibrator = ProbabilityCalibrator.fit(fit_rows, MIN_PER_CLASS)
        for (species, drug), group in fit_rows.groupby(["species", "drug"]):
            fit_rows.loc[group.index, "p_active"] = fold_calibrator.calibrate(species, drug, group["p_raw"].to_numpy())
        fold_thresholds = CallThresholds.fit(fit_rows, args.vme_target, args.me_target, MIN_PER_CLASS)
        for (species, drug), group in rows[held_out].groupby(["species", "drug"]):
            pair = fold_thresholds.lookup(species, drug)
            if pair is None:
                continue
            p_active = fold_calibrator.calibrate(species, drug, group["p_raw"].to_numpy())
            rows.loc[group.index, "held_out_p_active"] = p_active
            rows.loc[group.index[p_active >= pair[0]], "call"] = "likely_active"
            rows.loc[group.index[p_active <= pair[1]], "call"] = "likely_inactive"
    scored = rows[rows["held_out_p_active"].notna()].copy()

    call_rows = []
    for (species, drug), group in scored.groupby(["species", "drug"]):
        resistant = group["lab_resistant"]
        committed = group["call"] != "uncertain"
        correct = ((group["call"] == "likely_active") & ~resistant) | ((group["call"] == "likely_inactive") & resistant)
        call_rows.append({
            "species": species,
            "drug": drug,
            "vme_rate": float(((group["call"] == "likely_active") & resistant).sum() / resistant.sum()),
            "me_rate": float(((group["call"] == "likely_inactive") & ~resistant).sum() / (~resistant).sum()),
            "committed_share": float(committed.mean()),
            "correct_when_committed": float(correct[committed].mean()) if committed.any() else float("nan"),
            "n": len(group),
            "n_lab_r": int(resistant.sum()),
            "n_lab_s": int((~resistant).sum()),
            "run_id": artifact.run_id,
        })
    calls_report = pd.DataFrame(call_rows)
    calls_report.to_csv(args.run_dir / "calls_report.csv", index=False)

    scored["confidence_level"] = scored["held_out_p_active"].map(ConfidenceLevels.level_for)
    level_rows = []
    for species, species_rows in scored.groupby("species"):
        for level in CONFIDENCE_LEVEL_NAMES:
            group = species_rows[species_rows["confidence_level"] == level]
            if group.empty:
                continue
            level_rows.append({
                "species": species,
                "confidence_level": level,
                "share_of_results": len(group) / len(species_rows),
                "mean_p_active": float(group["held_out_p_active"].mean()),
                "observed_active_rate": float((~group["lab_resistant"]).mean()),
                "n": len(group),
                "run_id": artifact.run_id,
            })
    levels_report = pd.DataFrame(level_rows)
    levels_report.to_csv(args.run_dir / "confidence_levels_report.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(calls_report.drop(columns="run_id").round(3).to_string(index=False))
        print(levels_report.drop(columns="run_id").round(3).to_string(index=False))
    logger.info("Calls fitted: pairs=%s scored rows=%s", len(calls_report), len(scored))


if __name__ == "__main__":
    main()
