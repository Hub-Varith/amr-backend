"""CLI: fit probability thresholds for calls from a run's out-of-fold predictions.

Writes call_thresholds.json (fitted on all out-of-fold rows) and call_thresholds_report.csv
(each fold scored with thresholds fitted on the other folds) into the run folder. Never reads test rows.

Example:
    python -m genome2mic.predict.run_fit_call_thresholds --run-dir models/multitask
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
from genome2mic.predict.constants import CALL_STANDARD, CALL_YEAR, ME_TARGET, MIN_PER_CLASS, VME_TARGET
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
        decided["p_active"] = SusceptibilityCaller.probability_active(
            decided["mu_log2"].to_numpy(), sigma_by_drug[drug], s_breakpoint
        )
        frames.append(decided)
    rows = pd.concat(frames, ignore_index=True)

    thresholds = CallThresholds.fit(rows, args.vme_target, args.me_target, MIN_PER_CLASS)
    (args.run_dir / "call_thresholds.json").write_text(json.dumps(thresholds.to_dict(), indent=2))

    # Honest score: each fold is called with thresholds fitted on the other folds.
    rows["call"] = "uncertain"
    for fold in sorted(rows["fold"].unique()):
        held_out = rows["fold"] == fold
        fold_thresholds = CallThresholds.fit(rows[~held_out], args.vme_target, args.me_target, MIN_PER_CLASS)
        for (species, drug), group in rows[held_out].groupby(["species", "drug"]):
            pair = fold_thresholds.lookup(species, drug)
            if pair is None:
                continue
            rows.loc[group.index[group["p_active"] >= pair[0]], "call"] = "likely_active"
            rows.loc[group.index[group["p_active"] <= pair[1]], "call"] = "likely_inactive"

    report_rows = []
    for (species, drug), group in rows.groupby(["species", "drug"]):
        if thresholds.lookup(species, drug) is None:
            continue
        resistant = group["lab_resistant"]
        committed = group["call"] != "uncertain"
        correct = ((group["call"] == "likely_active") & ~resistant) | ((group["call"] == "likely_inactive") & resistant)
        report_rows.append({
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
    report = pd.DataFrame(report_rows)
    report.to_csv(args.run_dir / "call_thresholds_report.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(report.drop(columns="run_id").round(3).to_string(index=False))
    logger.info("Call thresholds done: pairs=%s rows=%s", len(report), len(rows))


if __name__ == "__main__":
    main()
