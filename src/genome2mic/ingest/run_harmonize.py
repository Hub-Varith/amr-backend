"""CLI: build labels.parquet, the counts table, and pairs_kept.csv from data/raw/ (stage 2)."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.ingest.harmonize import LabelHarmonizer
from genome2mic.validate.labels_checks import LabelsValidator

logger = logging.getLogger(__name__)


def main() -> None:
    """Harmonize raw AST, check the result, and write every stage 2 output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    species_config = yaml.safe_load((args.configs_dir / "species.yaml").read_text())
    drug_config = yaml.safe_load((args.configs_dir / "drugs.yaml").read_text())
    breakpoints = BreakpointTable.from_directory(args.configs_dir / "breakpoints")
    # Read every raw column as text so IDs like 573.10 keep their trailing zero.
    raw_tables = {
        name: pd.read_csv(args.raw_dir / f"{name}.csv", dtype=str, keep_default_na=False)
        for name in ("ast_bvbrc", "ast_ncbi", "meta_bvbrc", "meta_ncbi")
    }

    harmonizer = LabelHarmonizer(species_config, drug_config, breakpoints)
    labels, dropped = harmonizer.harmonize(
        bvbrc_ast=raw_tables["ast_bvbrc"],
        ncbi_ast=raw_tables["ast_ncbi"],
        bvbrc_meta=raw_tables["meta_bvbrc"],
        ncbi_meta=raw_tables["meta_ncbi"],
    )
    LabelsValidator.check(labels)

    counts = LabelHarmonizer.count_pairs(labels)
    pairs_kept = counts.loc[counts["kept"], ["species", "drug", "n_rows", "n_S", "n_nonsusceptible", "n_distinct_mic"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out_dir / "labels.parquet", index=False)
    dropped.to_parquet(args.out_dir / "dropped_labels.parquet", index=False)
    counts.to_csv(args.out_dir / "label_counts.csv", index=False)
    pairs_kept.to_csv(args.out_dir / "pairs_kept.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 400):
        logger.info("Drop reasons:\n%s", dropped["reason"].value_counts().to_string())
        logger.info("Label counts:\n%s", counts.to_string(index=False))
    logger.info("Stage 2 done: labels=%s genomes=%s pairs_kept=%s", len(labels), labels["genome_id"].nunique(),
                len(pairs_kept))


if __name__ == "__main__":
    main()
