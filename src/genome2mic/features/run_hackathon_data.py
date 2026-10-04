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
ALL_SPECIES = ["KPNEU", "ECOLI", "SAUR", "PAER", "ABAU"]


def main() -> None:
    """Build known_amr.parquet, known_amr_columns.csv, lineages.parquet, and splits.parquet for every species given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", default=ALL_SPECIES)
    parser.add_argument("--amrfinder-db", type=Path, required=True)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--pd-dir", type=Path, default=Path("data/raw/ncbi_pd"),
                        help="Holds <SPECIES>.metadata.tsv and <SPECIES>.clusters.tsv from `make ncbi-pd`")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    species_configs = yaml.safe_load((args.configs_dir / "species.yaml").read_text())["species"]
    keep_variant = pd.read_csv(args.configs_dir / "keep_variant.csv")["family"].tolist()
    labels = pd.read_parquet(args.processed_dir / "labels.parquet")
    pairs_kept = pd.read_csv(args.processed_dir / "pairs_kept.csv")
    labels = labels.merge(pairs_kept[["species", "drug"]], on=["species", "drug"])

    feature_tables, column_maps, lineage_tables, split_tables = [], [], [], []
    for species in args.species:
        species_config = species_configs[species]
        split_drugs = species_config["hackathon_split_drugs"]
        logger.info("Species %s started: split drugs %s", species, split_drugs)
        species_labels = labels[labels["species"] == species]
        genomes = (
            species_labels[["genome_id", "species", "biosample"]]
            .drop_duplicates(subset="genome_id")
            .dropna(subset=["biosample"])
        )
        pd_metadata = pd.read_csv(args.pd_dir / f"{species}.metadata.tsv", sep="\t", dtype=str,
                                  usecols=["biosample_acc", "AMR_genotypes"], keep_default_na=False)
        clusters = pd.read_csv(args.pd_dir / f"{species}.clusters.tsv", sep="\t", dtype=str)
        class_table = NcbiKnownAmrBuilder.load_class_table(args.amrfinder_db, species_config["amrfinder_organism"])

        features, column_map = NcbiKnownAmrBuilder(keep_variant, class_table).build(genomes, pd_metadata)
        covered_genomes = genomes[genomes["genome_id"].isin(features["genome_id"])]
        lineages = HackathonSplitMaker.lineages(covered_genomes, clusters)
        splits = HackathonSplitMaker.make_splits(lineages, species_labels, split_drugs, SPLIT_SEED)

        joined = splits.merge(lineages[["genome_id", "lineage_cluster"]], on="genome_id")
        if joined.groupby("lineage_cluster")["split"].nunique().max() > 1:
            raise ValueError(f"{species}: a lineage cluster spans train and test")
        if joined[joined["split"] == "train"].groupby("lineage_cluster")["fold"].nunique().max() > 1:
            raise ValueError(f"{species}: a lineage cluster spans two folds")

        test_ids = set(splits.loc[splits["split"] == "test", "genome_id"])
        for drug in split_drugs:
            drug_labels = species_labels[species_labels["drug"] == drug]
            trainable = drug_labels[drug_labels["genome_id"].isin(features["genome_id"])]
            in_test = trainable["genome_id"].isin(test_ids)
            logger.info("%s %s: train rows=%s test rows=%s (test R=%s S=%s)", species, drug,
                        int((~in_test).sum()), int(in_test.sum()),
                        int((trainable.loc[in_test, "sir"] == "R").sum()),
                        int((trainable.loc[in_test, "sir"] == "S").sum()))
        logger.info("Species %s done: labelled genomes=%s with features=%s test=%s", species, len(genomes),
                    len(features), len(test_ids))

        feature_tables.append(features)
        column_maps.append(column_map)
        lineage_tables.append(lineages)
        split_tables.append(splits)

    features, column_map = NcbiKnownAmrBuilder.combine(feature_tables, column_maps)
    lineages = pd.concat(lineage_tables, ignore_index=True)
    splits = pd.concat(split_tables, ignore_index=True)
    features.to_parquet(args.processed_dir / "known_amr.parquet", index=False)
    column_map.to_csv(args.processed_dir / "known_amr_columns.csv", index=False)
    lineages.to_parquet(args.processed_dir / "lineages.parquet", index=False)
    splits.to_parquet(args.processed_dir / "splits.parquet", index=False)
    logger.info("Hackathon data done: species=%s genomes=%s feature columns=%s", args.species, len(features),
                features.shape[1] - 2)


if __name__ == "__main__":
    main()
