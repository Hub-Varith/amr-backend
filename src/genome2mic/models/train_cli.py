"""Train the shared multi-drug model from the contract files.

Example:
    python -m genome2mic.models.train_cli --processed-dir data/processed --out-dir models/multitask
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from genome2mic.data.constants import SPECIES_KEYS
from genome2mic.data.known_amr_table import KnownAmrTable
from genome2mic.data.label_matrix import LabelMatrix
from genome2mic.data.unitig_store import UnitigStore
from genome2mic.models.cross_validation import CrossValidation
from genome2mic.models.train_config import TrainConfig

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, required=True, help="Folder with labels/known_amr/splits parquet files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where model.pt, spec.json and OOF predictions go")
    parser.add_argument("--species", nargs="*", default=None, help="Restrict to these species keys")
    parser.add_argument("--drugs", nargs="*", default=None, help="Restrict to these drugs")
    parser.add_argument("--no-unitigs", action="store_true", help="Known-AMR features only (ablation)")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--unitig-max-columns", type=int, default=20000)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    labels_frame = pd.read_parquet(arguments.processed_dir / "labels.parquet")
    splits = pd.read_parquet(arguments.processed_dir / "splits.parquet")
    if arguments.species:
        labels_frame = labels_frame[labels_frame["species"].isin(arguments.species)]
        splits = splits[splits["species"].isin(arguments.species)]
    pairs_path = arguments.processed_dir / "pairs_kept.csv"
    pairs_kept = pd.read_csv(pairs_path) if pairs_path.exists() else None
    if pairs_kept is not None:
        kept_key = pairs_kept["species"] + "|" + pairs_kept["drug"]
        before = len(labels_frame)
        labels_frame = labels_frame[(labels_frame["species"] + "|" + labels_frame["drug"]).isin(kept_key)]
        logger.info("Labels limited to pairs_kept", extra={"n_before": before, "n_after": len(labels_frame)})

    labels = LabelMatrix.from_labels(labels_frame, drugs=arguments.drugs)
    known = KnownAmrTable.from_parquet(str(arguments.processed_dir / "known_amr.parquet"))
    unitigs = UnitigStore.from_processed_dir(arguments.processed_dir, list(SPECIES_KEYS))
    config = TrainConfig(
        drugs=labels.drugs,
        use_unitigs=not arguments.no_unitigs,
        max_epochs=arguments.max_epochs,
        unitig_max_columns=arguments.unitig_max_columns,
    )

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    label_counts = labels.label_counts()
    label_counts.to_csv(arguments.out_dir / "label_counts.csv", index=False)
    print(label_counts.to_string(index=False))

    artifact, predictions, history = CrossValidation(labels, known, unitigs, splits, config, pairs_kept).run()
    artifact.save(arguments.out_dir)
    predictions.to_parquet(arguments.out_dir / "preds_oof.parquet", index=False)
    history.to_parquet(arguments.out_dir / "history.parquet", index=False)
    (arguments.out_dir / "conformal.json").write_text(json.dumps(artifact.conformal.to_records(), indent=2))
    print(f"Saved model to {arguments.out_dir} with run_id {artifact.run_id}")


if __name__ == "__main__":
    main()
