"""Provisional splits: whole clusters per split and per fold, test holds R and S."""

import pandas as pd

from genome2mic.splits.hackathon_splits import HackathonSplitMaker


def toy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    genomes = pd.DataFrame(
        {
            "genome_id": [f"g{index}" for index in range(200)],
            "species": "KPNEU",
            "biosample": [f"SAMN{index}" for index in range(200)],
        }
    )
    # 40 twin clusters of 3 genomes; the other 80 genomes are on their own.
    clusters = pd.DataFrame(
        {"PDS_acc": [f"PDS{index // 3:06d}.1" for index in range(120)], "biosample_acc": genomes["biosample"][:120]}
    )
    labels = pd.DataFrame(
        {
            "genome_id": genomes["genome_id"],
            "species": "KPNEU",
            "drug": "meropenem",
            "sir": ["R" if index % 2 else "S" for index in range(200)],
        }
    )
    return HackathonSplitMaker.lineages(genomes, clusters), labels


def test_lineage_clusters_are_species_prefixed_and_singletons_get_their_own() -> None:
    lineages, _ = toy_data()
    clusters = lineages.set_index("genome_id")["lineage_cluster"]
    assert clusters["g0"] == clusters["g2"] == "KPNEU_PDS000000"
    assert clusters["g150"] == "KPNEU_SOLO_SAMN150"
    assert set(lineages["cluster_method"]) == {"ncbi_snp_cluster"}


def test_no_cluster_spans_two_splits_or_two_folds() -> None:
    lineages, labels = toy_data()
    splits = HackathonSplitMaker.make_splits(lineages, labels, ["meropenem"], seed=0)
    joined = splits.merge(lineages[["genome_id", "lineage_cluster"]], on="genome_id")
    assert joined.groupby("lineage_cluster")["split"].nunique().max() == 1
    train = joined[joined["split"] == "train"]
    assert train.groupby("lineage_cluster")["fold"].nunique().max() == 1
    assert set(train["fold"]) == {0, 1, 2, 3, 4}
    assert joined.loc[joined["split"] == "test", "fold"].isna().all()


def test_test_share_and_classes() -> None:
    lineages, labels = toy_data()
    splits = HackathonSplitMaker.make_splits(lineages, labels, ["meropenem"], seed=0)
    test_ids = set(splits.loc[splits["split"] == "test", "genome_id"])
    assert 0.15 <= len(test_ids) / len(splits) <= 0.20
    test_sir = labels.loc[labels["genome_id"].isin(test_ids), "sir"]
    assert (test_sir == "R").sum() >= 10 and (test_sir == "S").sum() >= 10
    assert list(splits.columns) == ["genome_id", "species", "split", "fold", "external_set", "lolo_lineage"]
