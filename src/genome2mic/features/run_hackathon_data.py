"""CLI: provisional known_amr, lineages, and splits from NCBI Pathogen Detection. See docs/HACKATHON_DATA.md."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from genome2mic.features.ncbi_known_amr import NcbiKnownAmrBuilder
from genome2mic.splits.hackathon_splits import HackathonSplitMaker

logger = logging.getLogger(__name__)

SPLIT_SEED = 0


def main() -> None:
    """Build known_amr.parquet, known_amr_columns.csv, lineages.parquet, and splits.parquet for one species."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="KPNEU")
    parser.add_argument("--drugs", nargs="+", default=["ceftriaxone", "meropenem", "ciprofloxacin"],
                        help="The test set must hold R and S for each of these drugs")
    parser.add_argument("--amrfinder-db", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--pd-dir", type=Path, default=Path("data/raw/ncbi_pd"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    species_config = yaml.safe_load((args.configs_dir / "species.yaml").read_text())["species"][args.species]
    keep_variant = pd.read_csv(args.configs_dir / "keep_variant.csv")["family"].tolist()
    labels = pd.read_parquet(args.processed_dir / "labels.parquet")
    pairs_kept = pd.read_csv(args.processed_dir / "pairs_kept.csv")
    labels = labels.merge(pairs_kept[["species", "drug"]], on=["species", "drug"])
    labels = labels[labels["species"] == args.species]
    genomes = labels[["genome_id", "species", "biosample"]].drop_duplicates(subset="genome_id").dropna(subset=["biosample"])

    pd_metadata = pd.read_csv(args.pd_dir / "kleb.metadata.tsv", sep="\t", dtype=str,
                              usecols=["biosample_acc", "AMR_genotypes"], keep_default_na=False)
    clusters = pd.read_csv(args.pd_dir / "kleb.clusters.tsv", sep="\t", dtype=str)
    class_table = NcbiKnownAmrBuilder.load_class_table(args.amrfinder_db, species_config["amrfinder_organism"])

    features, column_map = NcbiKnownAmrBuilder(keep_variant, class_table).build(genomes, pd_metadata)
    covered_genomes = genomes[genomes["genome_id"].isin(features["genome_id"])]
    lineages = HackathonSplitMaker.lineages(covered_genomes, clusters)
    splits = HackathonSplitMaker.make_splits(lineages, labels, args.drugs, SPLIT_SEED)

    joined = splits.merge(lineages[["genome_id", "lineage_cluster"]], on="genome_id")
    if joined.groupby("lineage_cluster")["split"].nunique().max() > 1:
        raise ValueError("A lineage cluster spans train and test")
    if joined[joined["split"] == "train"].groupby("lineage_cluster")["fold"].nunique().max() > 1:
        raise ValueError("A lineage cluster spans two folds")

    features.to_parquet(args.processed_dir / "known_amr.parquet", index=False)
    column_map.to_csv(args.processed_dir / "known_amr_columns.csv", index=False)
    lineages.to_parquet(args.processed_dir / "lineages.parquet", index=False)
    splits.to_parquet(args.processed_dir / "splits.parquet", index=False)

    test_ids = set(splits.loc[splits["split"] == "test", "genome_id"])
    for drug in args.drugs:
        drug_labels = labels[labels["drug"] == drug]
        trainable = drug_labels[drug_labels["genome_id"].isin(features["genome_id"])]
        in_test = trainable["genome_id"].isin(test_ids)
        logger.info("%s: train rows=%s test rows=%s (test R=%s S=%s)", drug, int((~in_test).sum()), int(in_test.sum()),
                    int((trainable.loc[in_test, "sir"] == "R").sum()), int((trainable.loc[in_test, "sir"] == "S").sum()))
    logger.info("Hackathon data done: genomes=%s feature columns=%s", len(features), features.shape[1] - 2)


if __name__ == "__main__":
    main()
