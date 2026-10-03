"""Tests for ``genome2mic.splits.make_splits``.

Inputs are built inline (no FASTAs needed): a lineages table with two species
and a dozen clusters of varying sizes, a labels table whose ``sir`` values are
arranged so that the test-set R/S check has real work to do (meropenem ``R`` for
KPNEU lives in one small cluster only), and a ``pairs_kept`` table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.io import write_csv, write_parquet
from genome2mic.paths import Paths
from genome2mic.splits import make_splits as ms

KPNEU_SIZES = {f"KPNEU_ML_{i:03d}": n for i, n in enumerate([20, 15, 12, 10, 8, 8, 6, 6, 5, 4, 3, 3], start=1)}
ECOLI_SIZES = {f"ECOLI_ML_{i:03d}": n for i, n in enumerate([12, 10, 8, 5, 3, 2], start=1)}
RARE_R_CLUSTER = "KPNEU_ML_011"  # the only KPNEU cluster with meropenem R
COUNTRIES = ["USA", "UK", "India", None]
FORBIDDEN_FEATURES = {"lineage_cluster", "st", "country", "year", "source", "isolation_source", "biosample"}


@dataclass
class FakeConfig:
    species: dict[str, object] = field(default_factory=lambda: {"KPNEU": object(), "ECOLI": object()})


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_lineages() -> pd.DataFrame:
    rows: list[tuple[str, str, str]] = []
    for species, sizes in (("KPNEU", KPNEU_SIZES), ("ECOLI", ECOLI_SIZES)):
        for cluster, n in sizes.items():
            for j in range(n):
                rows.append((f"{cluster}_g{j:02d}", species, cluster))
    return pd.DataFrame(
        {
            "genome_id": pd.array([r[0] for r in rows], dtype="str"),
            "species": pd.array([r[1] for r in rows], dtype="str"),
            "lineage_cluster": pd.array([r[2] for r in rows], dtype="str"),
            "st": pd.array(["NA"] * len(rows), dtype="str"),
            "cluster_method": pd.array(["mash_single_linkage"] * len(rows), dtype="str"),
        }
    )


def make_labels(lineages: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, (gid, species, cluster) in enumerate(
        zip(lineages["genome_id"], lineages["species"], lineages["lineage_cluster"])
    ):
        country = COUNTRIES[i % len(COUNTRIES)]
        year = None if i % 7 == 0 else 2015 + (i % 6)
        if species == "KPNEU":
            mero = "R" if cluster == RARE_R_CLUSTER else "S"
            cipro = "R" if i % 2 == 0 else ("I" if i % 5 == 0 else "S")
            rows.append(dict(genome_id=gid, species=species, drug="meropenem", sir=mero, country=country, year=year))
            rows.append(dict(genome_id=gid, species=species, drug="ciprofloxacin", sir=cipro, country=country, year=year))
            rows.append(dict(genome_id=gid, species=species, drug="colistin", sir="S", country=country, year=year))
        else:
            rows.append(
                dict(genome_id=gid, species=species, drug="meropenem", sir="R" if i % 3 == 0 else "S", country=country, year=year)
            )
    frame = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "genome_id": pd.array(frame["genome_id"].tolist(), dtype="str"),
            "biosample": pd.array([f"SAMN{i:08d}" for i in range(len(frame))], dtype="str"),
            "species": pd.array(frame["species"].tolist(), dtype="str"),
            "drug": pd.array(frame["drug"].tolist(), dtype="str"),
            "mic_lower": np.full(len(frame), 1.0),
            "mic_upper": np.full(len(frame), 2.0),
            "censor": pd.array(["interval"] * len(frame), dtype="str"),
            "sir": pd.array(frame["sir"].tolist(), dtype="str"),
            "raw_result": pd.array(["=2"] * len(frame), dtype="str"),
            "method": pd.array(["dilution"] * len(frame), dtype="str"),
            "standard": pd.array(["EUCAST"] * len(frame), dtype="str"),
            "standard_year": pd.array([2024] * len(frame), dtype="Int64"),
            "source": pd.array(["BVBRC"] * len(frame), dtype="str"),
            "isolation_source": pd.array(["blood"] * len(frame), dtype="str"),
            "country": pd.array(frame["country"].tolist(), dtype="str"),
            "year": pd.array(frame["year"].tolist(), dtype="Int64"),
        }
    )


def make_pairs_kept() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": ["KPNEU", "KPNEU", "ECOLI", "SAUR"],
            "drug": ["meropenem", "ciprofloxacin", "meropenem", "oxacillin"],
            "n": [100, 100, 40, 60],
            "n_nonsusceptible": [50, 50, 20, 30],
            "n_susceptible": [50, 50, 20, 30],
            "n_distinct_mic": [6, 6, 6, 6],
        }
    )


@pytest.fixture(scope="module")
def lineages() -> pd.DataFrame:
    return make_lineages()


@pytest.fixture(scope="module")
def labels(lineages: pd.DataFrame) -> pd.DataFrame:
    return make_labels(lineages)


@pytest.fixture(scope="module")
def pairs_kept() -> pd.DataFrame:
    return make_pairs_kept()


@pytest.fixture(scope="module")
def splits(lineages: pd.DataFrame, labels: pd.DataFrame, pairs_kept: pd.DataFrame) -> pd.DataFrame:
    return ms.build_splits(lineages, labels, pairs_kept, seed=7)


# --------------------------------------------------------------------------- #
# Contract shape and invariants
# --------------------------------------------------------------------------- #
def test_columns_dtypes_and_no_forbidden_columns(splits: pd.DataFrame) -> None:
    assert list(splits.columns) == list(ms.COLUMNS)
    assert str(splits["fold"].dtype) == "Int64"
    for col in ("genome_id", "species", "split", "external_set", "lolo_lineage"):
        assert str(splits[col].dtype) == "str"
    assert not (set(splits.columns) & FORBIDDEN_FEATURES)


def test_every_genome_once_and_split_values(splits: pd.DataFrame, lineages: pd.DataFrame) -> None:
    assert splits["genome_id"].is_unique
    assert set(splits["genome_id"]) == set(lineages["genome_id"])
    assert set(splits["split"]) == {ms.SPLIT_TRAIN, ms.SPLIT_TEST}


def test_no_cluster_in_two_splits_or_two_folds(splits: pd.DataFrame, lineages: pd.DataFrame) -> None:
    merged = splits.merge(lineages[["genome_id", "lineage_cluster"]], on="genome_id")
    assert (merged.groupby("lineage_cluster")["split"].nunique() == 1).all()
    train = merged[(merged["split"] == ms.SPLIT_TRAIN).to_numpy()]
    assert (train.groupby("lineage_cluster")["fold"].nunique() == 1).all()
    ms.check_invariants(splits, lineages)  # does not raise


def test_fold_is_null_exactly_on_test_and_uses_min_of_five_and_train_clusters(
    splits: pd.DataFrame, lineages: pd.DataFrame
) -> None:
    is_test = (splits["split"] == ms.SPLIT_TEST).to_numpy()
    assert np.array_equal(splits["fold"].isna().to_numpy(), is_test)
    merged = splits.merge(lineages[["genome_id", "lineage_cluster"]], on="genome_id")
    for species in ("KPNEU", "ECOLI"):
        train = merged[(merged["split"] == ms.SPLIT_TRAIN).to_numpy() & (merged["species"] == species).to_numpy()]
        n_train_clusters = train["lineage_cluster"].nunique()
        assert set(train["fold"].astype(int)) == set(range(min(5, n_train_clusters))), species
    # KPNEU has 12 clusters, so at least 8 stay in train and the full 5 folds are used.
    kpneu_train = merged[(merged["split"] == ms.SPLIT_TRAIN).to_numpy() & (merged["species"] == "KPNEU").to_numpy()]
    assert set(kpneu_train["fold"].astype(int)) == {0, 1, 2, 3, 4}


def test_test_fraction_within_range(splits: pd.DataFrame) -> None:
    for species in ("KPNEU", "ECOLI"):
        sub = splits[(splits["species"] == species).to_numpy()]
        frac = (sub["split"] == ms.SPLIT_TEST).mean()
        assert 0.15 <= frac <= 0.20, (species, frac)


# --------------------------------------------------------------------------- #
# Test-set R/S check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_test_set_has_r_and_s_for_every_kept_pair(
    lineages: pd.DataFrame, labels: pd.DataFrame, pairs_kept: pd.DataFrame, seed: int
) -> None:
    out = ms.build_splits(lineages, labels, pairs_kept, seed=seed)
    test_ids = set(out.loc[(out["split"] == ms.SPLIT_TEST).to_numpy(), "genome_id"])
    for species, drug in (("KPNEU", "meropenem"), ("KPNEU", "ciprofloxacin"), ("ECOLI", "meropenem")):
        sub = labels[(labels["species"] == species).to_numpy() & (labels["drug"] == drug).to_numpy()]
        sirs = set(sub.loc[sub["genome_id"].isin(test_ids).to_numpy(), "sir"])
        assert {"R", "S"} <= sirs, (seed, species, drug, sirs)
    ms.check_invariants(out, lineages)


def test_repair_swaps_in_the_missing_class_and_logs(
    lineages: pd.DataFrame, labels: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="genome2mic")
    lin = lineages[(lineages["species"] == "KPNEU").to_numpy()]
    cluster_of = dict(zip(lin["genome_id"], lin["lineage_cluster"]))
    labels_sp = labels[(labels["species"] == "KPNEU").to_numpy()]
    initial = {"KPNEU_ML_009", "KPNEU_ML_010", "KPNEU_ML_012"}  # 12 genomes, all meropenem S
    assert ms.missing_classes(labels_sp, {g for g, c in cluster_of.items() if c in initial}, ["meropenem"]) == [
        ("meropenem", "R")
    ]
    repaired, failing = ms.repair_test_clusters(
        initial, cluster_of, labels_sp, ["meropenem", "ciprofloxacin"], np.random.default_rng(0), species="KPNEU"
    )
    assert failing == []
    assert RARE_R_CLUSTER in repaired
    n_test = sum(KPNEU_SIZES[c] for c in repaired)
    assert 15 <= n_test <= 20
    assert "swap 1: +KPNEU_ML_011" in caplog.text
    assert "R/S check passed" in caplog.text


def test_unfixable_check_warns_without_raising(
    lineages: pd.DataFrame, labels: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="genome2mic")
    pairs = pd.DataFrame({"species": ["KPNEU"], "drug": ["colistin"]})  # colistin is S everywhere
    out = ms.build_splits(lineages, labels, pairs, seed=1)
    ms.check_invariants(out, lineages)
    assert "no cluster outside the test set carries colistin=R" in caplog.text
    assert "still failing" in caplog.text


def test_greedy_respects_bounds_and_whole_clusters() -> None:
    sizes = {"c1": 12, "c2": 10, "c3": 8, "c4": 5, "c5": 3, "c6": 2}  # 40 genomes -> 6..8 in test
    for seed in range(20):
        chosen = ms.greedy_test_clusters(sizes, np.random.default_rng(seed), 0.15, 0.20)
        n = sum(sizes[c] for c in chosen)
        assert 6 <= n <= 8, (seed, chosen)


def test_greedy_logs_when_clusters_too_large(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="genome2mic")
    chosen = ms.greedy_test_clusters({"big": 90, "other": 60}, np.random.default_rng(0), 0.15, 0.20)
    assert chosen == set()
    assert "below the 15% target" in caplog.text


def test_repair_warns_when_no_cluster_move_reaches_the_range(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="genome2mic")
    cluster_of = {f"big_g{i}": "big" for i in range(90)} | {f"other_g{i}": "other" for i in range(60)}
    labels_sp = pd.DataFrame(
        {
            "genome_id": pd.array(list(cluster_of), dtype="str"),
            "drug": pd.array(["meropenem"] * 150, dtype="str"),
            "sir": pd.array(["R" if i % 2 else "S" for i in range(150)], dtype="str"),
        }
    )
    test, failing = ms.repair_test_clusters(set(), cluster_of, labels_sp, ["meropenem"], np.random.default_rng(0))
    assert failing == []
    assert test == {"other"}  # the smaller cluster restores both classes; 40% is the best whole-cluster option
    assert "outside the 15-20% target" in caplog.text


def test_repair_replaces_an_oversized_cluster(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="genome2mic")
    lineages = make_lineages()
    lin = lineages[(lineages["species"] == "KPNEU").to_numpy()]
    cluster_of = dict(zip(lin["genome_id"], lin["lineage_cluster"]))
    labels_sp = make_labels(lineages)
    labels_sp = labels_sp[(labels_sp["species"] == "KPNEU").to_numpy()]
    # Start from the single 20-genome cluster: adding the R cluster overshoots to 23 and no single
    # removal keeps meropenem S, so the repair must replace KPNEU_ML_001 with a smaller cluster.
    test, failing = ms.repair_test_clusters(
        {"KPNEU_ML_001"}, cluster_of, labels_sp, ["meropenem", "ciprofloxacin"], np.random.default_rng(4), species="KPNEU"
    )
    assert failing == []
    assert RARE_R_CLUSTER in test
    assert 15 <= sum(KPNEU_SIZES[c] for c in test) <= 20
    assert "bring the test set inside the 15-20% range" in caplog.text


# --------------------------------------------------------------------------- #
# External sets and LOLO
# --------------------------------------------------------------------------- #
def test_external_sets_only_on_test_and_disjoint(splits: pd.DataFrame, labels: pd.DataFrame) -> None:
    is_test = (splits["split"] == ms.SPLIT_TEST).to_numpy()
    assert splits.loc[~is_test, "external_set"].isna().all()
    assert set(splits["external_set"].dropna()) <= set(ms.EXTERNAL_SETS)
    meta = ms.genome_metadata(labels).set_index("genome_id")
    for species in ("KPNEU", "ECOLI"):
        sub = splits[is_test & (splits["species"] == species).to_numpy()].set_index("genome_id")
        country = meta.loc[sub.index, "country"]
        counts = country.dropna().value_counts()
        top = sorted(counts[counts == counts.max()].index)[0]
        in_country = sub["external_set"] == ms.EXTERNAL_COUNTRY
        assert set(sub.index[in_country.to_numpy()]) == set(country.index[(country == top).to_numpy()])
        rest = sub[~in_country.to_numpy()]
        years = meta.loc[rest.index, "year"].dropna()
        latest = int(years.max())
        in_time = rest["external_set"] == ms.EXTERNAL_TIME
        expected_time = set(meta.loc[rest.index].index[(meta.loc[rest.index, "year"] == latest).fillna(False).to_numpy()])
        assert set(rest.index[in_time.to_numpy()]) == expected_time
        assert len(expected_time) > 0
        assert rest.loc[~in_time.to_numpy(), "external_set"].isna().all()


def test_external_sets_null_without_metadata() -> None:
    meta = pd.DataFrame(
        {"genome_id": pd.array(["a", "b"], dtype="str"), "country": pd.array([None, None], dtype="str"), "year": pd.array([None, None], dtype="Int64")}
    )
    ext = ms.mark_external_sets(meta)
    assert ext.isna().all()


def test_lolo_marks_the_two_largest_clusters(splits: pd.DataFrame, lineages: pd.DataFrame) -> None:
    merged = splits.merge(lineages[["genome_id", "lineage_cluster"]], on="genome_id")
    lolo = merged[merged["lolo_lineage"].notna().to_numpy()]
    assert (lolo["lolo_lineage"] == lolo["lineage_cluster"]).all()
    assert set(lolo["lolo_lineage"]) == {"KPNEU_ML_001", "KPNEU_ML_002", "ECOLI_ML_001", "ECOLI_ML_002"}
    for cluster in ("KPNEU_ML_001", "KPNEU_ML_002", "ECOLI_ML_001", "ECOLI_ML_002"):
        members = merged[(merged["lineage_cluster"] == cluster).to_numpy()]
        assert (members["lolo_lineage"] == cluster).all()  # every row of the cluster, train or test


# --------------------------------------------------------------------------- #
# Invariant checker and reproducibility
# --------------------------------------------------------------------------- #
def test_check_invariants_detects_cluster_in_two_splits(splits: pd.DataFrame, lineages: pd.DataFrame) -> None:
    bad = splits.copy()
    victim = bad.index[(bad["split"] == ms.SPLIT_TRAIN).to_numpy()][0]
    bad.loc[victim, "split"] = ms.SPLIT_TEST
    bad.loc[victim, "fold"] = pd.NA
    with pytest.raises(ContractViolation, match="both train and test"):
        ms.check_invariants(bad, lineages)


def test_check_invariants_detects_cluster_in_two_folds(splits: pd.DataFrame, lineages: pd.DataFrame) -> None:
    bad = splits.copy()
    train_idx = bad.index[(bad["split"] == ms.SPLIT_TRAIN).to_numpy()]
    victim = train_idx[0]
    bad.loc[victim, "fold"] = (int(bad.loc[victim, "fold"]) + 1) % 5
    with pytest.raises(ContractViolation, match="span two folds"):
        ms.check_invariants(bad, lineages)


def test_check_invariants_detects_fold_on_test_and_external_on_train(splits: pd.DataFrame, lineages: pd.DataFrame) -> None:
    bad = splits.copy()
    victim = bad.index[(bad["split"] == ms.SPLIT_TEST).to_numpy()][0]
    bad.loc[victim, "fold"] = 0
    with pytest.raises(ContractViolation, match="fold must be null"):
        ms.check_invariants(bad, lineages)
    bad = splits.copy()
    victim = bad.index[(bad["split"] == ms.SPLIT_TRAIN).to_numpy()][0]
    bad.loc[victim, "external_set"] = ms.EXTERNAL_COUNTRY
    with pytest.raises(ContractViolation, match="external_set"):
        ms.check_invariants(bad, lineages)


def test_seed_reproducible_and_species_independent(
    lineages: pd.DataFrame, labels: pd.DataFrame, pairs_kept: pd.DataFrame, splits: pd.DataFrame
) -> None:
    again = ms.build_splits(lineages, labels, pairs_kept, seed=7)
    pd.testing.assert_frame_equal(again, splits)
    other = ms.build_splits(lineages, labels, pairs_kept, seed=8)
    assert not other["split"].equals(splits["split"])
    # Dropping ECOLI must not change the KPNEU split for the same seed.
    only_k = lineages[(lineages["species"] == "KPNEU").to_numpy()]
    k_only = ms.build_splits(only_k, labels, pairs_kept, seed=7)
    k_full = splits[(splits["species"] == "KPNEU").to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(k_only, k_full)


def test_build_splits_logs_label_rows_without_lineage(
    lineages: pd.DataFrame, labels: pd.DataFrame, pairs_kept: pd.DataFrame
) -> None:
    extra = labels.iloc[[0]].assign(genome_id=pd.array(["GHOST"], dtype="str"))
    log = DropLog("splits")
    ms.build_splits(lineages, pd.concat([labels, extra], ignore_index=True), pairs_kept, seed=7, droplog=log)
    frame = log.to_frame().set_index("reason")
    assert frame.loc["label_rows_for_genomes_without_lineage", "n_dropped"] == 1
    assert frame.loc["kept_pairs_for_species_without_lineages", "n_dropped"] == 1  # the SAUR pair


# --------------------------------------------------------------------------- #
# run(): files and the frozen rule
# --------------------------------------------------------------------------- #
@pytest.fixture
def root(tmp_path: Path, lineages: pd.DataFrame, labels: pd.DataFrame, pairs_kept: pd.DataFrame) -> Paths:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    write_parquet(lineages, paths.lineages)
    write_parquet(labels, paths.labels)
    write_csv(pairs_kept, paths.pairs_kept)
    return paths


def test_run_writes_files_and_refuses_to_overwrite(root: Paths, caplog: pytest.LogCaptureFixture) -> None:
    first = ms.run(root, FakeConfig(), seed=7)
    assert root.splits.is_file()
    assert root.drop_log("splits").is_file()
    saved = pd.read_parquet(root.splits)
    pd.testing.assert_frame_equal(saved, first)
    assert str(saved["fold"].dtype) == "Int64"

    with pytest.raises(ContractViolation, match="frozen"):
        ms.run(root, FakeConfig(), seed=8)
    pd.testing.assert_frame_equal(pd.read_parquet(root.splits), first)  # untouched

    caplog.set_level(logging.WARNING, logger="genome2mic")
    forced = ms.run(root, FakeConfig(), seed=8, force=True)
    assert "overwriting frozen" in caplog.text
    pd.testing.assert_frame_equal(pd.read_parquet(root.splits), forced)
    assert not forced["split"].equals(first["split"])


def test_run_requires_inputs(tmp_path: Path, lineages: pd.DataFrame, labels: pd.DataFrame) -> None:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    write_parquet(lineages, paths.lineages)
    write_parquet(labels, paths.labels)
    with pytest.raises(FileNotFoundError, match="pairs_kept"):
        ms.run(paths, FakeConfig())


def test_run_drops_species_outside_config(root: Paths) -> None:
    out = ms.run(root, FakeConfig(species={"KPNEU": object()}), seed=7)
    assert set(out["species"]) == {"KPNEU"}
    drop = pd.read_csv(root.drop_log("splits")).set_index("reason")
    assert drop.loc["lineage_rows_species_not_in_config", "n_dropped"] == sum(ECOLI_SIZES.values())
