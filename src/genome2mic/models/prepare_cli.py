"""Validate the KPNEU release and describe the model. Never fits a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from genome2mic.data.known_amr_table import KnownAmrTable
from genome2mic.data.label_matrix import LabelMatrix
from genome2mic.models.train_config import TrainConfig


def prepare(processed: Path) -> dict:
    labels = pd.read_parquet(processed / "labels.parquet")
    known = KnownAmrTable.from_parquet(str(processed / "known_amr.parquet"))
    splits = pd.read_parquet(processed / "splits.parquet")
    pairs = pd.read_csv(processed / "pairs_kept.csv")
    split = splits[splits.species.eq("KPNEU")].copy()
    if split.genome_id.duplicated().any():
        raise ValueError("Duplicate split genome IDs")
    if not set(split.split).issubset({"train", "test"}):
        raise ValueError("Unexpected split names")
    # Check identifiers only for held-out rows; do not score or tune on their labels.
    known.rows(split.genome_id.to_numpy())
    train = split[split.split.eq("train")]
    if train.fold.isna().any() or set(train.fold) != set(range(5)):
        raise ValueError("Expected frozen training folds 0..4")
    allowed = pairs[pairs.species.eq("KPNEU")].drug
    fit_labels = labels[labels.species.eq("KPNEU") & labels.genome_id.isin(train.genome_id)
                        & labels.drug.isin(allowed)].copy()
    matrix = LabelMatrix.from_labels(fit_labels)
    if set(train.genome_id) != set(matrix.genome_ids):
        raise ValueError("Training IDs missing usable labels")
    columns_per_fold = {}
    for fold in sorted(train.fold.unique()):
        ids = train.loc[train.fold.ne(fold), "genome_id"].to_numpy()
        columns_per_fold[str(int(fold))] = len(known.frequent_columns(ids, np.full(len(ids), "KPNEU"), 5))
    config = TrainConfig(drugs=matrix.drugs, use_unitigs=False)
    return {
        "training_started": False,
        "release": (processed / "RELEASE").read_text().strip(),
        "species": "KPNEU", "train_genomes": len(train),
        "held_out_genomes": int(split.split.eq("test").sum()),
        "training_label_rows": len(fit_labels), "drug_heads": len(matrix.drugs),
        "available_amr_features": len(known.columns), "features_per_fit_fold": columns_per_fold,
        "train_genomes_per_fold": {str(int(k)): int(v) for k,v in train.fold.value_counts().items()},
        "training_labels_per_drug": matrix.label_counts().to_dict("records"),
        "training_censor_counts": fit_labels.censor.value_counts().to_dict(),
        "config": config.to_dict(),
        "input_sha256": {name: hashlib.sha256((processed/name).read_bytes()).hexdigest()
                         for name in ["labels.parquet", "known_amr.parquet", "splits.parquet", "pairs_kept.csv"]},
        "limitations": ["No raw DNA/read files in this release",
                        "Other four species lack feature rows in this release",
                        "Breakpoints and split provenance need validation before reliability claims",
                        "Existing conformal bands are provisional; calibration audit required"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.processed_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k,v in report.items() if k not in
                      {"config", "input_sha256", "training_labels_per_drug"}}, indent=2))

if __name__ == "__main__":
    main()
