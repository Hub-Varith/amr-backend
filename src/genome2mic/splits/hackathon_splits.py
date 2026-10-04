"""Provisional lineages and splits from NCBI SNP clusters. Hackathon only; see docs/HACKATHON_DATA.md.

NCBI SNP clusters keep near-identical isolates together but not whole lineages (e.g. ST258),
so test scores from these splits are optimistic. The frozen splits come from Mash/PopPUNK later.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_FOLDS = 5
TEST_SHARE_MIN = 0.15
TEST_SHARE_MAX = 0.20
MIN_TEST_PER_CLASS = 10
MAX_SEED_TRIES = 50
SPLIT_COLUMNS = ["genome_id", "species", "split", "fold", "external_set", "lolo_lineage"]


class HackathonSplitMaker:
    """Builds lineages.parquet and splits.parquet from NCBI Pathogen Detection SNP clusters."""

    @staticmethod
    def lineages(genomes: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
        """One row per genome. Genomes outside every SNP cluster form a cluster of their own."""
        cluster_by_biosample = (
            clusters.drop_duplicates(subset="biosample_acc").set_index("biosample_acc")["PDS_acc"].str.split(".").str[0]
        )
        snp_cluster = genomes["biosample"].map(cluster_by_biosample)
        lineage_cluster = np.where(
            snp_cluster.notna(),
            genomes["species"] + "_" + snp_cluster.fillna(""),
            genomes["species"] + "_SOLO_" + genomes["biosample"],
        )
        lineages = pd.DataFrame(
            {
                "genome_id": genomes["genome_id"],
                "species": genomes["species"],
                "lineage_cluster": lineage_cluster,
                "st": None,
                "cluster_method": "ncbi_snp_cluster",
            }
        )
        logger.info("Lineages: genomes=%s clusters=%s in_snp_cluster=%s", len(lineages),
                    lineages["lineage_cluster"].nunique(), int(snp_cluster.notna().sum()))
        return lineages

    @staticmethod
    def make_splits(lineages: pd.DataFrame, labels: pd.DataFrame, drugs: list[str], seed: int) -> pd.DataFrame:
        """Whole clusters go to test (15-20% of genomes) until every drug has R and S there; the rest get 5 folds."""
        cluster_sizes = lineages.groupby("lineage_cluster").size()
        n_genomes = len(lineages)
        for attempt_seed in range(seed, seed + MAX_SEED_TRIES):
            order = np.random.default_rng(attempt_seed).permutation(cluster_sizes.index.to_numpy())
            test_clusters = set()
            test_count = 0
            for cluster in order:
                if test_count >= TEST_SHARE_MIN * n_genomes:
                    break
                if test_count + cluster_sizes[cluster] > TEST_SHARE_MAX * n_genomes:
                    continue
                test_clusters.add(cluster)
                test_count += cluster_sizes[cluster]
            test_ids = set(lineages.loc[lineages["lineage_cluster"].isin(test_clusters), "genome_id"])
            test_labels = labels[labels["genome_id"].isin(test_ids) & labels["drug"].isin(drugs)]
            counts = test_labels.groupby(["drug", "sir"]).size()
            has_both = all(
                counts.get((drug, "R"), 0) >= MIN_TEST_PER_CLASS and counts.get((drug, "S"), 0) >= MIN_TEST_PER_CLASS
                for drug in drugs
            )
            if has_both:
                logger.info("Test set: seed=%s genomes=%s (%.1f%%)", attempt_seed, len(test_ids),
                            100 * len(test_ids) / n_genomes)
                break
            logger.info("Seed %s: test set lacks R or S for a drug, trying the next seed", attempt_seed)
        else:
            raise ValueError("No seed gave a test set with R and S for every drug")

        # GroupKFold's rule: biggest cluster first, into the fold that has the fewest genomes so far.
        train_sizes = cluster_sizes.drop(list(test_clusters)).sort_index().sort_values(ascending=False, kind="stable")
        fold_counts = [0] * N_FOLDS
        fold_by_cluster = {}
        for cluster, size in train_sizes.items():
            fold = int(np.argmin(fold_counts))
            fold_by_cluster[cluster] = fold
            fold_counts[fold] += size
        logger.info("Fold sizes: %s", fold_counts)

        splits = pd.DataFrame(
            {
                "genome_id": lineages["genome_id"],
                "species": lineages["species"],
                "split": np.where(lineages["lineage_cluster"].isin(test_clusters), "test", "train"),
                "fold": lineages["lineage_cluster"].map(fold_by_cluster).astype("Int64"),
                "external_set": None,
                "lolo_lineage": None,
            }
        )
        return splits[SPLIT_COLUMNS]
