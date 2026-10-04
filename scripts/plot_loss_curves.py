"""Reconstruct AFT CV diagnostics and plot them with ordinary Matplotlib.

Run from the repository root with the project environment::

    .venv/bin/python scripts/plot_loss_curves.py --root runs/hackathon5
    .venv/bin/python scripts/plot_loss_curves.py --plot-only

The second command redraws the saved CSV without fitting anything. Output defaults
to .context/charts/loss_curves_clean/ (PNG, PDF, SVG, CSV, JSON and a drop log).
Use --pair KPNEU:meropenem to select a single pair, or repeat --pair for several.

These are reconstructed diagnostics, not histories captured during the original
training run. The frozen train split, feature-selection helper, cluster holdout,
CSR representation, interval labels and weight policy match the AFT training code.
The scale is selected inside the fit folds using the early-stopping holdout only.
The plotted run continues to the round limit to show behavior after the selected
round. The held-out CV fold never controls scale or stopping. Test labels are not
loaded; model bundles and prediction files are never written.

These are predictions of in-vitro susceptibility, not prescribing advice.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xgboost as xgb

from genome2mic.droplog import DropLog
from genome2mic.features.select import select_known
from genome2mic.mic import lab_exact_mask
from genome2mic.models.base import holdout_split, to_csr, validate_intervals
from genome2mic.models.xgb_aft import XgbAft

DEFAULT_PAIRS = [
    ("KPNEU", "meropenem"), ("ECOLI", "ciprofloxacin"),
    ("SAUR", "oxacillin"), ("PAER", "ciprofloxacin"), ("ABAU", "imipenem"),
]
SPECIES_NAMES = {
    "KPNEU": "K. pneumoniae", "ECOLI": "E. coli", "SAUR": "S. aureus",
    "PAER": "P. aeruginosa", "ABAU": "A. baumannii",
}
DATASETS = [("train", "Training", "#1f77b4"),
            ("early_stop", "Early-stopping holdout", "#ff7f0e"),
            ("cv", "Held-out CV fold", "#2ca02c")]


def reconstruct_pair(root: Path, species: str, drug: str, fold: int,
                     rounds: int | None, log: DropLog) -> tuple[pd.DataFrame, dict]:
    """Replay one AFT fit fold using the saved training protocol."""
    bundle = root / "models" / species / drug
    top = json.loads((bundle / "params.json").read_text())
    if top.get("model_class") == "AftB2Select":
        if top.get("base_name") != "aft_known" or "aft" not in top["components"]:
            raise ValueError(f"{species}:{drug}: this diagnostic needs a known-AMR AFT component")
        aft_path = bundle / top["components"]["aft"] / "params.json"
    elif top.get("model_class") == "XgbAft" and top.get("name") == "aft_known":
        aft_path = bundle / "params.json"
    else:
        raise ValueError(f"{species}:{drug}: unsupported component {top.get('model_class')}")
    saved = json.loads(aft_path.read_text())
    meta = json.loads((bundle / "meta.json").read_text())
    cfg = meta["train_config"]
    limit = saved["max_rounds"] if rounds is None else rounds
    if limit < saved["max_rounds"]:
        raise ValueError("--rounds must cover the saved training round limit")

    processed = root / "data" / "processed"
    splits = pd.read_parquet(processed / "splits.parquet",
                             filters=[("species", "==", species), ("split", "==", "train")])
    if splits.empty or splits.genome_id.duplicated().any():
        raise ValueError(f"{species}: missing or duplicate training split rows")
    train_ids = splits.genome_id.tolist()
    only_train = [("genome_id", "in", train_ids)]
    # Filter at the Parquet read: no test labels enter this diagnostic.
    labels = pd.read_parquet(processed / "labels.parquet",
                             filters=only_train + [("species", "==", species), ("drug", "==", drug)],
                             columns=["genome_id", "mic_lower", "mic_upper", "method"])
    known = pd.read_parquet(processed / "known_amr.parquet", filters=only_train)
    qc = pd.read_parquet(processed / "qc.parquet", filters=only_train,
                         columns=["genome_id", "qc_pass"])
    lineages = pd.read_parquet(processed / "lineages.parquet", filters=only_train,
                               columns=["genome_id", "lineage_cluster"])
    frame = labels.merge(splits[["genome_id", "fold"]], on="genome_id", validate="one_to_one")
    frame = frame.merge(qc, on="genome_id", how="left", validate="one_to_one")
    frame = log.keep_where(frame, frame.qc_pass.notna(), "missing_qc", f"{species}:{drug}")
    frame = log.keep_where(frame, frame.qc_pass.astype(bool), "qc_fail", f"{species}:{drug}")
    frame = frame.merge(lineages, on="genome_id", how="left", validate="one_to_one")
    frame = log.keep_where(frame, frame.lineage_cluster.notna(), "missing_lineage", f"{species}:{drug}")
    frame = log.keep_where(frame, frame.genome_id.isin(known.genome_id), "missing_known_amr", f"{species}:{drug}")
    frame = frame.sort_values("genome_id", kind="stable").reset_index(drop=True)
    known = known.set_index("genome_id", drop=False).loc[frame.genome_id].reset_index(drop=True)
    fit_idx = np.flatnonzero((frame.fold != fold).to_numpy())
    cv_idx = np.flatnonzero((frame.fold == fold).to_numpy())
    if not len(fit_idx) or not len(cv_idx):
        raise ValueError(f"{species}:{drug}: fold {fold} needs fit and CV rows")
    names = select_known(known, fit_idx, min_count=cfg["min_count"], droplog=log)
    if not names:
        raise ValueError(f"{species}:{drug}: no features survive the training-fold filter")
    # Same sparse zero/missing semantics and prefix order as XgbAft.fit.
    matrix = to_csr(known[names].to_numpy(dtype=np.float32))
    lo, hi = validate_intervals(frame.mic_lower.to_numpy(float), frame.mic_upper.to_numpy(float))
    exact = lab_exact_mask(lo, hi, frame.method.to_numpy())
    weights = np.where(exact, float(cfg["exact_weight"]), 1.0)
    weights /= weights[fit_idx].mean()
    inner_fit, inner_hold = holdout_split(len(fit_idx), saved["holdout_fraction"],
                                        saved["seed"], frame.lineage_cluster.to_numpy()[fit_idx])
    if not len(inner_hold):
        raise ValueError(f"{species}:{drug}: too few clusters for an early-stopping holdout")
    train_idx, hold_idx = fit_idx[inner_fit], fit_idx[inner_hold]
    groups = frame.lineage_cluster.to_numpy()
    if set(groups[train_idx]) & set(groups[hold_idx]) or set(groups[fit_idx]) & set(groups[cv_idx]):
        raise ValueError(f"{species}:{drug}: a cluster crosses diagnostic partitions")

    def dmatrix(indices: np.ndarray) -> xgb.DMatrix:
        return XgbAft._dmatrix(matrix[indices], names, lo[indices], hi[indices], weights[indices])

    dtrain, dhold, dcv = map(dmatrix, [train_idx, hold_idx, cv_idx])
    params = {**saved["params"], "seed": saved["seed"], "nthread": saved["nthread"]}
    choices = []
    for scale in saved["scales"]:
        booster = xgb.train({**params, "aft_loss_distribution_scale": scale}, dtrain,
                            num_boost_round=saved["max_rounds"], evals=[(dhold, "early_stop")],
                            early_stopping_rounds=saved["early_stopping_rounds"], verbose_eval=False)
        choices.append({"scale": scale, "best_iteration": booster.best_iteration,
                        "best_score": booster.best_score})
    selected = min(choices, key=lambda item: (item["best_score"], abs(item["scale"] - 1.0)))
    history: dict = {}
    xgb.train({**params, "aft_loss_distribution_scale": selected["scale"]}, dtrain,
              num_boost_round=limit, evals=[(dtrain, "train"), (dhold, "early_stop"), (dcv, "cv")],
              evals_result=history, verbose_eval=False)
    chosen_round = max(saved["min_rounds"], selected["best_iteration"] + 1)
    # A continued run must reproduce the early-stopping run before its stop.
    assert np.isclose(history["early_stop"]["aft-nloglik"][selected["best_iteration"]],
                      selected["best_score"], rtol=1e-6)
    rows = []
    for dataset, _, _ in DATASETS:
        rows.extend({"species": species, "drug": drug, "fold": fold, "round": i,
                     "dataset": dataset, "aft_nloglik": float(value)}
                    for i, value in enumerate(history[dataset]["aft-nloglik"], start=1))
    record = {"species": species, "drug": drug, "fold": fold, "scale": selected["scale"],
              "selected_round": chosen_round, "diagnostic_rounds": limit,
              "n_train": len(train_idx), "n_early_stop": len(hold_idx), "n_cv": len(cv_idx),
              "n_features": len(names), "run_id": meta["run_id"], "bundle_choice": top.get("choice", "aft"),
              "tuning": choices,
              "params": {**params, "aft_loss_distribution_scale": selected["scale"]},
              "exact_weight": cfg["exact_weight"],
              "early_stopping_rounds": saved["early_stopping_rounds"],
              "holdout_fraction": saved["holdout_fraction"]}
    print(f"{species}:{drug} — {len(train_idx)} train / {len(hold_idx)} early-stop / "
          f"{len(cv_idx)} CV rows; scale {selected['scale']}, selected round {chosen_round}", flush=True)
    return pd.DataFrame(rows), record


def plot(history: pd.DataFrame, records: list[dict], output: Path) -> None:
    """Draw raw recorded values without smoothing or decorative overlays."""
    columns = min(3, len(records))
    rows = math.ceil(len(records) / columns)
    style = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .7, "xtick.labelsize": 8, "ytick.labelsize": 8,
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, figsize=(3.7 * columns, 3.05 * rows + .7),
                                 squeeze=False)
        for ax, record in zip(axes.flat, records):
            values = history.loc[(history.species == record["species"]) & (history.drug == record["drug"])]
            for dataset, label, color in DATASETS:
                series = values.loc[values.dataset == dataset].sort_values("round")
                ax.plot(series["round"], series.aft_nloglik, label=label, color=color, linewidth=1.35)
            ax.axvline(record["selected_round"], color="0.3", linestyle=":", linewidth=1)
            ax.set_title(f"{SPECIES_NAMES.get(record['species'], record['species'])}\n{record['drug']}")
            ax.text(.97, .95, f"Selected round: {record['selected_round']}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8)
            ax.set_xlabel("Boosting round")
            ax.set_ylabel("AFT negative log-likelihood")
            ax.grid(axis="y", color="0.9", linewidth=.5)
            ax.set_xlim(1, record["diagnostic_rounds"])
        for ax in list(axes.flat)[len(records):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=color, linewidth=1.35, label=label)
                   for _, label, color in DATASETS]
        handles.append(Line2D([], [], color="0.3", linestyle=":", linewidth=1, label="Selected round"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .095),
                   ncol=2 if columns == 1 else 4, frameon=False, fontsize=8)
        fig.text(.5, .045, "Reconstructed AFT diagnostics; weighted loss. Continued beyond the selected round.",
                 ha="center", fontsize=8)
        fig.text(.5, .017, "These are predictions of in-vitro susceptibility, not prescribing advice.",
                 ha="center", fontsize=7)
        fig.tight_layout(rect=(0, .15, 1, 1), h_pad=1.3, w_pad=1.4)
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(output / f"loss_curves.{extension}", dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/hackathon5"))
    parser.add_argument("--out-dir", type=Path, default=Path(".context/charts/loss_curves_clean"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--rounds", type=int, help="Diagnostic rounds (at least the saved training limit)")
    parser.add_argument("--pair", action="append", help="SPECIES:drug; may be repeated")
    parser.add_argument("--plot-only", action="store_true", help="Redraw saved CSV; do not fit or load labels")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        history = pd.read_csv(args.out_dir / "loss_history.csv")
        records = json.loads((args.out_dir / "loss_metadata.json").read_text())["pairs"]
    else:
        pairs = [tuple(value.split(":", 1)) for value in args.pair] if args.pair else DEFAULT_PAIRS
        if any(len(pair) != 2 or not all(pair) for pair in pairs) or len(set(pairs)) != len(pairs):
            parser.error("--pair must be unique SPECIES:drug values")
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        log = DropLog("loss_diagnostics")
        frames, records = [], []
        for species, drug in pairs:
            frame, record = reconstruct_pair(args.root, species, drug, args.fold, args.rounds, log)
            frames.append(frame)
            records.append(record)
        history = pd.concat(frames, ignore_index=True)
        history.to_csv(args.out_dir / "loss_history.csv", index=False)
        (args.out_dir / "loss_metadata.json").write_text(json.dumps({
            "kind": "reconstructed_aft_cv_diagnostics", "xgboost_version": xgb.__version__,
            "test_labels_loaded": False, "scale_selection": "early-stopping holdout within fitting folds",
            "note": "These are predictions of in-vitro susceptibility, not prescribing advice.",
            "pairs": records,
        }, indent=2) + "\n")
        log.write(args.out_dir / "drop_log.csv")
    plot(history, records, args.out_dir)
    print(f"Saved plots and recorded losses to {args.out_dir}")


if __name__ == "__main__":
    main()
