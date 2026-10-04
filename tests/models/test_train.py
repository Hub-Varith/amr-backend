"""Integration tests for the training orchestrator, the evaluation stage and the CLI.

A small seeded synthetic project (KPNEU only, 30 kb genomes) is generated once per
module and run through every stage with relaxed inclusion thresholds and tiny
xgboost budgets. All data is SYNTHETIC; the assertions check plumbing and contract
rules, never predictive quality.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from genome2mic import ingest, qc
from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.config import load_config
from genome2mic.droplog import DropLog
from genome2mic.eval import metrics
from genome2mic.eval import run as eval_run
from genome2mic.features import known_amr, unitigs
from genome2mic.models import train
from genome2mic.models.base import FORBIDDEN_FEATURES
from genome2mic.paths import Paths
from genome2mic.predict.pipeline import PredictionPipeline
from genome2mic.splits import lineages, make_splits
from genome2mic.synthetic import generate

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MODEL_IDS = {"b0_resfinder", "b1_lookup", "b2_xgb_steps", "aft_known", "aft_known_unitig"}


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Synth -> ingest -> qc -> known-amr -> lineages -> splits -> unitigs -> train -> evaluate."""
    logging.getLogger("genome2mic").setLevel(logging.WARNING)
    root = tmp_path_factory.mktemp("synthetic_project")
    generate.run(Paths(root=root, configs_dir=REPO_CONFIGS), seed=11, n_kpneu=110, n_ecoli=0, genome_length=30_000)
    paths = Paths(root=root, configs_dir=root / "configs")
    config = load_config(paths.configs_dir)
    ingest.run(paths, config, min_ns=5, min_s=5, min_levels=3)
    qc.run(paths, config)
    known_amr.run(paths, config)
    lineages.run(paths, config)
    make_splits.run(paths, config, seed=3)
    unitigs.run(paths, config)
    cfg = train.TrainConfig(top_k=150, min_count=3, nthread=1, max_rounds=25, early_stopping_rounds=5, lolo=True)
    summary = train.run(paths, config, train_config=cfg)
    table = eval_run.run(paths, config)
    pairs = pd.read_csv(paths.pairs_kept)
    return {"paths": paths, "config": config, "summary": summary, "metrics": table, "pairs": pairs, "cfg": cfg}


def _preds(project: dict) -> pd.DataFrame:
    return eval_run.load_preds(project["paths"])


# --------------------------------------------------------------------------- preds


def test_preds_written_for_every_kept_pair(project: dict) -> None:
    paths, pairs = project["paths"], project["pairs"]
    assert len(pairs) >= 2
    for species, drug in zip(pairs["species"], pairs["drug"]):
        assert paths.preds(species, drug).is_file()


def test_preds_columns_and_models(project: dict) -> None:
    preds = _preds(project)
    assert list(preds.columns) == list(train.PREDS_COLUMNS)
    assert set(preds["model"].unique()) == MODEL_IDS
    assert {"cv", "test"} <= set(preds["split"].unique())
    assert preds["run_id"].nunique() == 1
    assert preds["run_id"].str.len().eq(12).all()
    assert preds.duplicated(["genome_id", "drug", "model", "split"]).sum() == 0


def test_preds_rows_match_their_split(project: dict) -> None:
    preds = _preds(project)
    splits = pd.read_parquet(project["paths"].splits)
    assigned = dict(zip(splits["genome_id"].astype(str), splits["split"].astype(str)))
    cv = preds.loc[preds["split"] == "cv", "genome_id"]
    test = preds.loc[preds["split"] == "test", "genome_id"]
    assert all(assigned[g] == "train" for g in cv)
    assert all(assigned[g] == "test" for g in test)
    lolo = preds.loc[preds["split"].str.startswith("lolo_")]
    assert len(lolo) > 0
    # external_set is copied onto test rows only
    assert preds.loc[preds["split"] != "test", "external_set"].isna().all()


def test_lolo_rows_are_train_genomes_of_their_lineage(project: dict) -> None:
    """Decision 1: LOLO predicts train genomes of the lineage (never a relabelled test subset)."""
    preds = _preds(project)
    splits = pd.read_parquet(project["paths"].splits)
    assigned = dict(zip(splits["genome_id"].astype(str), splits["split"].astype(str)))
    lolo_of = dict(zip(splits["genome_id"].astype(str), splits["lolo_lineage"]))
    lolo = preds.loc[preds["split"].str.startswith("lolo_").fillna(False).to_numpy(dtype=bool)]
    assert len(lolo) > 0
    for split_value, gid in zip(lolo["split"], lolo["genome_id"]):
        assert assigned[gid] == "train"
        assert lolo_of[gid] == split_value.removeprefix("lolo_")
    test_ids = set(preds.loc[preds["split"] == "test", "genome_id"])
    assert not (set(lolo["genome_id"]) & test_ids)


def test_nearest_distance_equals_dense_reference(project: dict) -> None:
    """Block-wise nearest distances equal the old dense n x n computation exactly."""
    from genome2mic import sketch as sk

    paths = project["paths"]
    preds = _preds(project)
    ids, sketches, k = sk.load_sketches(paths.sketches("KPNEU"))
    pos = {g: i for i, g in enumerate(ids)}
    dense = sk.pairwise_distances(sketches, k=k)
    for drug, sub in preds.loc[preds["model"] == "b1_lookup"].groupby("drug"):
        train_ids = sub.loc[sub["split"] == "cv", "genome_id"].tolist()  # b1 predicts every train row
        cols = [pos[g] for g in train_ids]
        test = sub.loc[sub["split"] == "test"]
        expected = dense[np.ix_([pos[g] for g in test["genome_id"]], cols)].min(axis=1)
        np.testing.assert_array_equal(test["nearest_training_distance"].to_numpy(dtype=float), expected)
        for split_value, rows in sub.loc[sub["split"].str.startswith("lolo_")].groupby("split"):
            lineage = split_value.removeprefix("lolo_")
            in_lineage = set(rows["genome_id"])
            fit = [pos[g] for g in train_ids if g not in in_lineage]
            assert len(fit) < len(cols)  # a genuine refit: the lineage left the training set
            expected = dense[np.ix_([pos[g] for g in rows["genome_id"]], fit)].min(axis=1)
            np.testing.assert_array_equal(rows["nearest_training_distance"].to_numpy(dtype=float), expected)
            assert lineage


def test_lab_sir_rederived_column(project: dict) -> None:
    """Interface #5: the lab interval classified under the call breakpoint (null when ambiguous)."""
    preds = _preds(project)
    config = project["config"]
    assert "lab_sir_rederived" in preds.columns
    assert preds["lab_sir_rederived"].dropna().isin(["S", "I", "R"]).all()
    assert preds.loc[preds["model"] == "b0_resfinder", "lab_sir_rederived"].notna().any()
    for (species, drug), sub in preds.groupby(["species", "drug"]):
        bp = config.call_breakpoint(species, drug)
        got = sub["lab_sir_rederived"].astype(object).where(sub["lab_sir_rederived"].notna(), None).tolist()
        want = [train.rederive_lab_sir(lo, hi, bp) for lo, hi in zip(sub["lab_lower"], sub["lab_upper"])]
        assert got == want


def test_test_ledger_has_one_row_per_pair(project: dict) -> None:
    paths, pairs = project["paths"], project["pairs"]
    assert paths.test_ledger == paths.results_dir / "test_ledger.csv"
    ledger = pd.read_csv(paths.test_ledger, dtype={"run_id": str, "inputs_sha1": str})
    assert list(ledger.columns) == ["run_id", "created_utc", "species", "drug", "n_test_rows", "inputs_sha1"]
    assert len(ledger) == len(pairs)
    assert ledger["run_id"].nunique() == 1
    # One input fingerprint per species, equal to what compute_inputs_sha1 gives for the files on disk.
    for species, sub in ledger.groupby("species"):
        assert sub["inputs_sha1"].nunique() == 1
        assert sub["inputs_sha1"].iloc[0] == train.compute_inputs_sha1(paths, species)
    preds = _preds(project)
    for _, row in ledger.iterrows():
        n = int(((preds["species"] == row["species"]) & (preds["drug"] == row["drug"]) & (preds["split"] == "test")).sum())
        assert row["n_test_rows"] == n > 0


def test_train_drop_log_names_the_pair(project: dict) -> None:
    paths, pairs = project["paths"], project["pairs"]
    log = pd.read_csv(paths.drop_log("train"))
    labels = {f"{s} x {d}" for s, d in zip(pairs["species"], pairs["drug"])}
    assert log["detail"].map(lambda d: isinstance(d, str) and d.split(":", 1)[0] in labels).all()
    assert {"lolo_lineage_without_train_rows", "lolo_too_few_fit_rows"} <= set(log["reason"])


def test_pred_mic_on_grid_and_band_contains_it(project: dict) -> None:
    preds = _preds(project)
    model_rows = preds.loc[preds["model"] != "b0_resfinder"]
    pm = model_rows["pred_mic"].to_numpy(dtype=float)
    assert np.isfinite(pm).all() and (pm > 0).all()
    steps = np.log2(pm)
    assert np.allclose(steps, np.rint(steps))
    assert (model_rows["band_low"] <= model_rows["pred_mic"]).all()
    assert (model_rows["band_high"] >= model_rows["pred_mic"]).all()
    b0 = preds.loc[preds["model"] == "b0_resfinder"]
    assert len(b0) > 0 and b0["pred_mic"].isna().all() and b0["pred_sir"].isin(["S", "R"]).all()


def test_nearest_training_distance_present(project: dict) -> None:
    preds = _preds(project)
    model_rows = preds.loc[preds["model"] != "b0_resfinder"]
    dist = model_rows["nearest_training_distance"].to_numpy(dtype=float)
    assert np.isfinite(dist).all()
    assert (dist >= 0).all() and (dist <= 1).all()
    # a test genome's nearest training genome cannot be itself
    assert (preds.loc[preds["split"] == "test", "nearest_training_distance"].dropna() >= 0).all()


def test_lab_sir_filled_or_derived(project: dict) -> None:
    preds = _preds(project)
    assert preds["lab_sir"].dropna().isin(["S", "I", "R"]).all()
    assert preds["pred_sir"].dropna().isin(["S", "I", "R"]).all()
    assert preds["lab_sir"].notna().mean() > 0.8


# --------------------------------------------------------------------------- bundle


def test_bundle_layout(project: dict) -> None:
    paths, pairs = project["paths"], project["pairs"]
    models_dir = paths.models_dir
    assert (models_dir / "manifest.json").is_file()
    assert (models_dir / "reference_sketches.npz").is_file()
    assert (models_dir / "markers.fasta").is_file()
    manifest = json.loads((models_dir / "manifest.json").read_text())
    assert manifest["model_version"] and len(manifest["run_id"]) == 12
    for species, drugs in manifest["species"].items():
        assert (models_dir / species / "train_sketches.npz").is_file()
        assert (models_dir / species / "unitig_kmers.npz").is_file()
        assert (models_dir / species / "unitig_index.parquet").is_file()
        for drug in drugs:
            d = models_dir / species / drug
            for name in ("model.ubj", "params.json", "features.json", "conformal.json", "meta.json", "importance.json"):
                assert (d / name).is_file(), f"{d / name} missing"
            feats = json.loads((d / "features.json").read_text())
            assert feats["model_class"] == "aft_known_unitig"
            assert all(c.startswith("u_") for c in feats["unitig_cols"])
            assert not (set(feats["known_columns"]) & FORBIDDEN_FEATURES)
            assert all(c.startswith(("gene_", "point_", "n_class_")) for c in feats["known_columns"])
            conformal = json.loads((d / "conformal.json").read_text())
            assert np.isfinite(conformal["q"]) and conformal["q"] >= 0
            importance = json.loads((d / "importance.json").read_text())
            assert isinstance(importance, list) and all({"feature", "gain"} <= set(i) for i in importance)
    assert set(manifest["species"]) == set(pairs["species"].unique())
    assert sum(len(v) for v in manifest["species"].values()) == len(pairs)


def test_pipeline_loads_bundle_and_matches_test_predictions(project: dict) -> None:
    """The saved bundle reproduces the preds table's test-row pred_mic for the main model."""
    paths, config = project["paths"], project["config"]
    preds = _preds(project)
    test_rows = preds.loc[(preds["split"] == "test") & (preds["model"] == "aft_known_unitig")]
    assert len(test_rows) > 0
    gid = str(test_rows["genome_id"].iloc[0])
    pipeline = PredictionPipeline(paths.models_dir, paths.configs_dir, amrfinder_tsv=paths.interim_dir(gid) / "amrfinder.tsv")
    pipeline.load()
    assert pipeline.available_models() == {sp: sorted(d) for sp, d in json.loads(paths.models_manifest.read_text())["species"].items()}
    report = pipeline.run(paths.genome_fasta(gid), gid)
    PredictionReport.model_validate(report)
    assert report["species"] == "KPNEU"
    expected = test_rows.loc[test_rows["genome_id"] == gid].set_index("drug")["pred_mic"]
    got = {p["drug"]: p["pred_mic"] for p in report["predictions"] if p["pred_mic"] is not None}
    for drug, mic in expected.items():
        assert drug in got, f"{drug} not predicted by the bundle"
        assert got[drug] == pytest.approx(mic), f"{drug}: bundle {got[drug]} vs preds table {mic}"
    # nearest-training-distance from the bundle equals the preds table value for this test genome
    assert report["nearest_training_distance"] == pytest.approx(float(test_rows.loc[test_rows["genome_id"] == gid, "nearest_training_distance"].iloc[0]), abs=1e-9)


def test_pipeline_markerscan_fallback_validates(project: dict) -> None:
    paths = project["paths"]
    preds = _preds(project)
    gid = str(preds.loc[preds["split"] == "test", "genome_id"].iloc[0])
    pipeline = PredictionPipeline(paths.models_dir, paths.configs_dir)
    report = pipeline.run(paths.genome_fasta(gid), gid)
    PredictionReport.model_validate(report)
    assert report["disclaimer"]


# --------------------------------------------------------------------------- evaluate


def test_metrics_tables(project: dict) -> None:
    paths, table, pairs = project["paths"], project["metrics"], project["pairs"]
    assert paths.metrics.is_file() and paths.metrics.with_suffix(".csv").is_file()
    assert paths.metrics_by_distance.is_file() and paths.metrics_by_distance.with_suffix(".csv").is_file()
    metric_cols = [c for c in table.columns if c in metrics.METRIC_COLUMNS]
    assert metric_cols[0] == "vme_rate"
    for species, drug in zip(pairs["species"], pairs["drug"]):
        for model in MODEL_IDS:
            for split in ("cv", "test"):
                sub = table[(table["species"] == species) & (table["drug"] == drug) & (table["model"] == model) & (table["split"] == split)]
                assert len(sub) == 1, f"missing metrics row for {species} {drug} {model} {split}"
    by_dist = pd.read_parquet(paths.metrics_by_distance)
    assert len(by_dist) > 0 and "distance_bin" in by_dist.columns
    assert (by_dist["split"] == "test").any()


# --------------------------------------------------------------------------- units


class _Bp:
    def __init__(self, s: float, r: float) -> None:
        self.s_breakpoint, self.r_breakpoint = s, r


@pytest.mark.parametrize(
    ("sir", "lo", "hi", "bp", "expected"),
    [
        ("R", 0.0, 0.25, _Bp(2, 8), "R"),  # reported wins
        (None, 0.0, 0.25, _Bp(2, 8), "S"),  # <= 0.25 under S <= 2
        (None, 16.0, float("inf"), _Bp(2, 8), "R"),  # > 16 under R > 8
        (None, 2.0, 8.0, _Bp(2, 8), "I"),  # (2, 8] exactly the I window
        (None, 1.0, 4.0, _Bp(2, 8), None),  # straddles S
        (None, 4.0, 16.0, _Bp(2, 8), None),  # straddles R
        (None, 0.0, 4.0, None, None),  # no breakpoint
        (pd.NA, 0.0, 1.0, _Bp(2, 8), "S"),
    ],
)
def test_derive_lab_sir(sir, lo, hi, bp, expected) -> None:
    assert train.derive_lab_sir(sir, lo, hi, bp) == expected


def test_parse_pheno_table(tmp_path: Path) -> None:
    config = load_config(REPO_CONFIGS)
    table = tmp_path / "pheno_table.txt"
    table.write_text(
        "# comment\n"
        "# Antimicrobial\tClass\tWGS-predicted phenotype\tMatch\tGenetic background\n"
        "meropenem\tbeta-lactam\tResistant\t3\tblaKPC-2 (blaKPC-2_X)\n"
        "piperacillin+tazobactam\tbeta-lactam\tNo resistance\t0\t\n"
        "unobtainium\tother\tResistant\t3\t\n"
    )
    assert train.parse_pheno_table(table, config) == {"meropenem": "R", "piperacillin-tazobactam": "S"}


def test_run_id_depends_on_params_and_splits(tmp_path: Path) -> None:
    splits = tmp_path / "splits.parquet"
    splits.write_bytes(b"abc")
    a = train.compute_run_id(train.TrainConfig(), splits)
    b = train.compute_run_id(train.TrainConfig(top_k=10), splits)
    splits.write_bytes(b"abd")
    c = train.compute_run_id(train.TrainConfig(), splits)
    assert len(a) == 12 and a != b and a != c


def test_effective_models_without_unitigs() -> None:
    cfg = train.TrainConfig(ablation=True)
    assert "aft_unitig_only" in cfg.effective_models(True)
    assert cfg.effective_models(False) == ["b1_lookup", "b2_xgb_steps", "aft_known"]
    with pytest.raises(ValueError):
        train.TrainConfig(models=("nope",)).effective_models(True)


# --------------------------------------------------------------------------- cli


def test_cli_predict_prints_valid_report(project: dict, capsys: pytest.CaptureFixture[str]) -> None:
    from genome2mic.cli import main

    paths = project["paths"]
    preds = _preds(project)
    gid = str(preds.loc[preds["split"] == "test", "genome_id"].iloc[0])
    code = main([
        "predict", "--root", str(paths.root), "--configs-dir", str(paths.configs_dir), "--fasta", str(paths.genome_fasta(gid)),
        "--sample-id", "BC-TEST", "--amrfinder-tsv", str(paths.interim_dir(gid) / "amrfinder.tsv"), "-q",
    ])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    PredictionReport.model_validate(report)
    assert report["sample_id"] == "BC-TEST"


def test_cli_parser_has_every_stage() -> None:
    from genome2mic.cli import build_parser

    parser = build_parser()
    for stage in ("synth", "ingest", "qc", "known-amr", "lineages", "splits", "unitigs", "train", "evaluate", "report", "predict", "run-all"):
        ns = parser.parse_args([stage, "--root", "x"] + (["--fasta", "f.fa"] if stage == "predict" else []))
        assert ns.command == stage and ns.root == "x"


def test_cli_seeds_are_independent() -> None:
    """#38: --seed (synth), --split-seed and --train-seed land in distinct dests."""
    from genome2mic.cli import _train_config, build_parser

    parser = build_parser()
    ns = parser.parse_args(["run-all", "--root", "x", "--seed", "5", "--split-seed", "3", "--train-seed", "11"])
    assert (ns.synth_seed, ns.split_seed, ns.train_seed) == (5, 3, 11)
    assert not hasattr(ns, "seed")
    assert _train_config(ns).seed == 11
    ns = parser.parse_args(["run-all", "--root", "x", "--split-seed", "3"])
    assert (ns.synth_seed, ns.split_seed, ns.train_seed) == (7, 3, 7)
    assert parser.parse_args(["synth", "--root", "x", "--seed", "9"]).synth_seed == 9
    assert parser.parse_args(["splits", "--root", "x", "--split-seed", "4"]).split_seed == 4
    assert parser.parse_args(["train", "--root", "x", "--train-seed", "8"]).train_seed == 8


def test_cli_seeds_reach_their_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from genome2mic import cli

    seen: dict[str, object] = {}

    def record(name: str, attr: str):
        def fn(ns) -> int:
            seen[name] = getattr(ns, attr)
            return 0
        return fn

    for name, attr, field_name in (("cmd_synth", "synth", "synth_seed"), ("cmd_splits", "splits", "split_seed"),
                                   ("cmd_train", "train", "train_seed")):
        monkeypatch.setattr(cli, name, record(attr, field_name))
    for name in ("cmd_ingest", "cmd_qc", "cmd_known_amr", "cmd_lineages", "cmd_unitigs", "cmd_evaluate", "cmd_report"):
        monkeypatch.setattr(cli, name, lambda ns: 0)
    assert cli.main(["run-all", "--root", str(tmp_path), "--seed", "5", "--split-seed", "3", "--train-seed", "11", "-q"]) == 0
    assert seen == {"synth": 5, "splits": 3, "train": 11}


def test_cli_train_species_and_drugs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#32: `train --species/--drugs` passes the matching kept pairs to train.run(pairs=...)."""
    from genome2mic import cli

    parser = cli.build_parser()
    ns = parser.parse_args(["train", "--root", "x", "--species", "kpneu", "--drugs", "meropenem", "ciprofloxacin"])
    assert ns.train_species == ["kpneu"] and ns.train_drugs == ["meropenem", "ciprofloxacin"]
    ns = parser.parse_args(["run-all", "--root", "x", "--species", "KPNEU"])
    assert ns.train_species == ["KPNEU"] and ns.train_drugs is None

    paths = Paths(root=tmp_path, configs_dir=REPO_CONFIGS)
    paths.processed_dir.mkdir(parents=True)
    pd.DataFrame({"species": ["KPNEU", "KPNEU", "ECOLI"], "drug": ["meropenem", "ceftriaxone", "meropenem"]}).to_csv(
        paths.pairs_kept, index=False
    )
    captured: dict[str, object] = {}

    def fake_run(paths, config, *, train_config=None, pairs=None):
        captured["pairs"] = pairs
        return pd.DataFrame({"species": [], "drug": []})

    monkeypatch.setattr(train, "run", fake_run)
    base = ["train", "--root", str(tmp_path), "--configs-dir", str(REPO_CONFIGS), "-q"]
    assert cli.main([*base, "--species", "kpneu"]) == 0
    assert captured["pairs"] == [("KPNEU", "meropenem"), ("KPNEU", "ceftriaxone")]
    assert cli.main([*base, "--drugs", "meropenem"]) == 0
    assert captured["pairs"] == [("KPNEU", "meropenem"), ("ECOLI", "meropenem")]
    assert cli.main(base) == 0
    assert captured["pairs"] is None
    with pytest.raises(ValueError, match="no kept pair"):
        cli.main([*base, "--species", "SAUR"])


# --------------------------------------------------------------------------- subset runs (#32, #15)


@pytest.fixture(scope="module")
def subset_project(project: dict, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A copy of the project where one pair is retrained alone (sharded run)."""
    import shutil

    src: Paths = project["paths"]
    root = tmp_path_factory.mktemp("subset_project") / "root"
    shutil.copytree(src.root, root, symlinks=True)
    paths = Paths(root=root, configs_dir=root / "configs")
    before_manifest = json.loads(paths.models_manifest.read_text())
    before_log = pd.read_csv(paths.drop_log("train"))
    before_ledger = pd.read_csv(paths.test_ledger, dtype={"run_id": str})
    pairs = project["pairs"]
    pair = (str(pairs["species"].iloc[0]), str(pairs["drug"].iloc[0]))
    cfg = train.TrainConfig(top_k=150, min_count=3, nthread=1, max_rounds=20, early_stopping_rounds=5, lolo=True)
    train.run(paths, project["config"], train_config=cfg, pairs=[pair])
    return {
        "paths": paths, "pair": pair, "pairs": pairs, "before_manifest": before_manifest,
        "before_log": before_log, "before_ledger": before_ledger, "cfg": cfg,
    }


def test_subset_run_merges_manifest(subset_project: dict) -> None:
    paths, (species, drug) = subset_project["paths"], subset_project["pair"]
    before = subset_project["before_manifest"]
    after = json.loads(paths.models_manifest.read_text())
    assert after["species"] == before["species"]  # nothing lost
    assert after["run_ids"][species][drug] == after["run_id"] != before["run_id"]
    others = [(s, d) for s, drugs in after["species"].items() for d in drugs if (s, d) != (species, drug)]
    assert others and all(after["run_ids"][s][d] == before["run_id"] for s, d in others)


def test_subset_run_keeps_other_pairs_drop_log_rows(subset_project: dict) -> None:
    paths, (species, drug) = subset_project["paths"], subset_project["pair"]
    before, after = subset_project["before_log"], pd.read_csv(paths.drop_log("train"))
    label = f"{species} x {drug}"

    def rows_of(frame: pd.DataFrame, mine: bool) -> pd.DataFrame:
        owned = frame["detail"].map(lambda d: isinstance(d, str) and d.split(":", 1)[0] == label)
        return frame[owned == mine].reset_index(drop=True)

    pd.testing.assert_frame_equal(rows_of(after, False), rows_of(before, False))
    assert len(rows_of(after, True)) > 0


def test_subset_run_appends_to_ledger(subset_project: dict) -> None:
    paths, (species, drug) = subset_project["paths"], subset_project["pair"]
    before = subset_project["before_ledger"]
    after = pd.read_csv(paths.test_ledger, dtype={"run_id": str})
    assert len(after) == len(before) + 1
    pd.testing.assert_frame_equal(after.iloc[: len(before)].reset_index(drop=True), before)
    last = after.iloc[-1]
    assert (last["species"], last["drug"]) == (species, drug)
    assert last["run_id"] != before["run_id"].iloc[0]


def test_full_run_overwrites_manifest_and_drops_stale_markers(project: dict, tmp_path: Path) -> None:
    """A non-synthetic bundle never ships a markers.fasta (#15); a full run overwrites the manifest."""
    paths = Paths(root=tmp_path, configs_dir=project["paths"].configs_dir)
    src = project["paths"]
    (paths.raw_dir / "references").mkdir(parents=True)
    for ref in (src.raw_dir / "references").iterdir():
        (paths.raw_dir / "references" / ref.name).write_bytes(ref.read_bytes())
    paths.models_dir.mkdir(parents=True)
    (paths.models_dir / "markers.fasta").write_text(">stale\nACGT\n")
    paths.models_manifest.write_text(json.dumps({"model_version": "0", "run_id": "old", "species": {"ECOLI": ["meropenem"]}}))
    config = project["config"]
    train.write_shared_bundle(paths, config, {"KPNEU": ["meropenem"]}, "new_run_id_0", full_run=True)
    assert not (paths.models_dir / "markers.fasta").exists()  # no SYNTHETIC_DATA.md under this root
    manifest = json.loads(paths.models_manifest.read_text())
    assert manifest["species"] == {"KPNEU": ["meropenem"]} and manifest["synthetic"] is False
    assert manifest["run_ids"] == {"KPNEU": {"meropenem": "new_run_id_0"}}


# --------------------------------------------------------------------------- unit: ledger, re-derived S/I/R


def test_append_test_ledger_appends_and_never_truncates(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    train.append_test_ledger(paths, "aaaaaaaaaaaa", "KPNEU", "meropenem", 12, inputs_sha1="111111111111")
    train.append_test_ledger(paths, "aaaaaaaaaaaa", "KPNEU", "meropenem", 12, inputs_sha1="111111111111")
    train.append_test_ledger(paths, "bbbbbbbbbbbb", "ECOLI", "ceftriaxone", 7, inputs_sha1="222222222222")
    lines = paths.test_ledger.read_text().splitlines()
    assert lines[0] == "run_id,created_utc,species,drug,n_test_rows,inputs_sha1"
    assert len(lines) == 4
    ledger = pd.read_csv(paths.test_ledger, dtype={"run_id": str, "inputs_sha1": str})
    assert ledger["run_id"].tolist() == ["aaaaaaaaaaaa", "aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    assert ledger["n_test_rows"].tolist() == [12, 12, 7]
    assert ledger["inputs_sha1"].tolist() == ["111111111111", "111111111111", "222222222222"]
    assert pd.to_datetime(ledger["created_utc"]).notna().all()
    with pytest.raises(ValueError, match="inputs_sha1"):
        train.append_test_ledger(paths, "aaaaaaaaaaaa", "KPNEU", "meropenem", 12, inputs_sha1="")


def test_append_test_ledger_rejects_a_foreign_header(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    paths.results_dir.mkdir(parents=True)
    paths.test_ledger.write_text("a,b\n1,2\n")
    from genome2mic.errors import ContractViolation

    with pytest.raises(ContractViolation, match="test_ledger"):
        train.append_test_ledger(paths, "x" * 12, "KPNEU", "meropenem", 1, inputs_sha1="f" * 12)
    assert paths.test_ledger.read_text() == "a,b\n1,2\n"


def test_append_test_ledger_handles_missing_trailing_newline(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    paths.results_dir.mkdir(parents=True)
    paths.test_ledger.write_text(
        "run_id,created_utc,species,drug,n_test_rows,inputs_sha1\nr1,2026-01-01T00:00:00+00:00,KPNEU,meropenem,3,abc"
    )
    train.append_test_ledger(paths, "r2", "KPNEU", "meropenem", 4, inputs_sha1="def")
    ledger = pd.read_csv(paths.test_ledger)
    assert ledger["run_id"].tolist() == ["r1", "r2"] and ledger["inputs_sha1"].tolist() == ["abc", "def"]


def test_append_test_ledger_upgrades_a_legacy_ledger_keeping_every_row(tmp_path: Path) -> None:
    """A ledger written before inputs_sha1 existed gains the column once; no row is lost or changed."""
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    paths.results_dir.mkdir(parents=True)
    legacy = (
        "run_id,created_utc,species,drug,n_test_rows\n"
        "r1,2026-01-01T00:00:00+00:00,KPNEU,meropenem,3\n"
        "r1,2026-01-02T00:00:00+00:00,KPNEU,ceftriaxone,5"
    )
    paths.test_ledger.write_text(legacy)
    train.append_test_ledger(paths, "r2", "KPNEU", "meropenem", 4, inputs_sha1="0123456789ab")
    lines = paths.test_ledger.read_text().splitlines()
    assert lines[0] == "run_id,created_utc,species,drug,n_test_rows,inputs_sha1"
    assert lines[1:3] == [f"{line}," for line in legacy.splitlines()[1:]]
    assert lines[3].startswith("r2,") and lines[3].endswith(",KPNEU,meropenem,4,0123456789ab")
    assert not list(paths.results_dir.glob("*.tmp"))
    # The leakage check reads the upgraded ledger: the unknown fingerprint of r1 does not match r2's.
    from genome2mic.eval import leakage

    ledger = pd.read_csv(paths.test_ledger, dtype=str, keep_default_na=False)
    assert ledger["inputs_sha1"].tolist() == ["", "", "0123456789ab"]
    assert leakage.check_test_touched_once(paths).passed is False


# --------------------------------------------------------------------------- unit: input fingerprint (inputs_sha1)


def _fingerprint_tree(root: Path) -> Paths:
    paths = Paths(root=root, configs_dir=root / "configs")
    paths.processed_dir.mkdir(parents=True)
    (paths.configs_dir / "breakpoints").mkdir(parents=True)
    for target, text in (
        (paths.labels, "labels v1"), (paths.known_amr, "known v1"), (paths.qc, "qc v1"), (paths.lineages, "lin v1"),
        (paths.unitig_index("KPNEU"), "index v1"), (paths.unitig_rows("KPNEU"), "rows v1"),
        (paths.configs_dir / "drugs.yaml", "drugs: {}"), (paths.configs_dir / "breakpoints" / "eucast_2024.csv", "a,b"),
    ):
        target.write_text(text)
    np.savez_compressed(paths.unitigs("KPNEU"), data=np.arange(5, dtype=np.int8))
    return paths


def test_inputs_sha1_changes_with_every_input_but_not_with_the_root(tmp_path: Path) -> None:
    paths = _fingerprint_tree(tmp_path / "a")
    base = train.compute_inputs_sha1(paths, "KPNEU")
    assert len(base) == 12 and base == train.compute_inputs_sha1(paths, "kpneu")
    assert train.compute_inputs_sha1(_fingerprint_tree(tmp_path / "b"), "KPNEU") == base  # root-independent
    names = [name for name, _ in train.input_files(paths, "KPNEU")]
    assert {"data/labels.parquet", "data/known_amr.parquet", "configs/drugs.yaml",
            "configs/breakpoints/eucast_2024.csv", "src/genome2mic/models/train.py",
            "src/genome2mic/features/unitigs.py", "src/genome2mic/mic.py"} <= set(names)
    assert {f"data/{paths.unitigs('KPNEU').name}", f"data/{paths.unitig_index('KPNEU').name}"} <= set(names)

    seen = {base}
    for target in (paths.labels, paths.known_amr, paths.unitig_index("KPNEU"), paths.configs_dir / "drugs.yaml",
                   paths.configs_dir / "breakpoints" / "eucast_2024.csv"):
        target.write_text(target.read_text() + " changed")
        value = train.compute_inputs_sha1(paths, "KPNEU")
        assert value not in seen, target
        seen.add(value)
    np.savez_compressed(paths.unitigs("KPNEU"), data=np.arange(6, dtype=np.int8))
    assert train.compute_inputs_sha1(paths, "KPNEU") not in seen
    # A new breakpoint table or a removed unitig file changes it too.
    (paths.configs_dir / "breakpoints" / "clsi_2024.csv").write_text("x")
    assert train.compute_inputs_sha1(paths, "KPNEU") not in seen
    # Another species' unitig files are not this species' input.
    before = train.compute_inputs_sha1(paths, "KPNEU")
    paths.unitig_index("ECOLI").write_text("ecoli index")
    assert train.compute_inputs_sha1(paths, "KPNEU") == before != train.compute_inputs_sha1(paths, "ECOLI")


def test_inputs_sha1_covers_model_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fingerprint_tree(tmp_path)
    base = train.compute_inputs_sha1(paths, "KPNEU")
    real = train._file_sha1

    def edited(path: Path) -> str:
        return "edited" if path.name == "b2_xgb_steps.py" else real(path)

    monkeypatch.setattr(train, "_file_sha1", edited)
    assert train.compute_inputs_sha1(paths, "KPNEU") != base


def test_npz_inputs_are_hashed_by_content_not_zip_metadata(tmp_path: Path) -> None:
    import time
    import zipfile

    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez_compressed(a, x=np.arange(10))
    with zipfile.ZipFile(a) as src, zipfile.ZipFile(b, "w", compression=zipfile.ZIP_STORED) as dst:
        for info in src.infolist():
            clone = zipfile.ZipInfo(info.filename, date_time=time.localtime(time.time() + 86_400)[:6])
            dst.writestr(clone, src.read(info.filename))
    assert a.read_bytes() != b.read_bytes()
    assert train._file_sha1(a) == train._file_sha1(b)


@pytest.mark.parametrize(
    ("lo", "hi", "bp", "expected"),
    [
        (0.0, 0.25, _Bp(2, 8), "S"),  # <= 0.25 under S <= 2
        (1.0, 2.0, _Bp(2, 8), "S"),  # exact 2 == S breakpoint
        (2.0, 4.0, _Bp(2, 8), "I"),
        (4.0, 8.0, _Bp(2, 8), "I"),  # exact 8 == R breakpoint is still I (R is > r)
        (8.0, 16.0, _Bp(2, 8), "R"),
        (16.0, float("inf"), _Bp(2, 8), "R"),
        (1.0, 4.0, _Bp(2, 8), None),  # straddles S
        (4.0, 16.0, _Bp(2, 8), None),  # straddles R
        (0.0, float("inf"), _Bp(2, 8), None),
        (0.0, 4.0, None, None),  # no call breakpoint
        (0.5, 1.0, _Bp(1, 1), "S"),  # S == R breakpoint: no I window
        (1.0, 2.0, _Bp(1, 1), "R"),
    ],
)
def test_rederive_lab_sir(lo, hi, bp, expected) -> None:
    assert train.rederive_lab_sir(lo, hi, bp) == expected


# --------------------------------------------------------------------------- unit: fit plan (LOLO guard)


def _frame(rows: list[tuple[str, str, float | None, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genome_id": [r[0] for r in rows],
            "split": pd.array([r[1] for r in rows], dtype="string"),
            "fold": pd.array([r[2] for r in rows], dtype="Int64"),
            "lineage_cluster": [r[3] for r in rows],
            "lolo_lineage": pd.array([r[4] for r in rows], dtype="string"),
        }
    )


def test_fit_plan_lolo_guard() -> None:
    rows = []
    for i in range(30):
        cluster = f"C{i % 6}"
        rows.append((f"g{i:02d}", "train", i % 3, cluster, "C0" if cluster == "C0" else None))
    for i in range(30, 36):
        rows.append((f"g{i:02d}", "test", None, "T1", "T1"))  # an old splits file: LOLO on a test cluster
    frame = _frame(rows)
    plan = train._fit_plan(frame, train.TrainConfig())
    assert [g.fold for g in plan.cv] == [0, 1, 2]
    assert [g.lineage for g in plan.lolo] == ["C0"]
    lolo = plan.lolo[0]
    assert lolo.split == "lolo_C0"
    assert set(frame["genome_id"].iloc[lolo.predict_idx]) == {f"g{i:02d}" for i in range(30) if i % 6 == 0}
    assert not set(lolo.predict_idx) & set(lolo.fit_idx)
    assert len(lolo.fit_idx) == 25 < len(plan.test.fit_idx) == 30
    assert plan.skipped_lolo == [("T1", "lolo_lineage_without_train_rows", 6)]
    assert train._fit_plan(frame, train.TrainConfig(lolo=False)).lolo == []


def test_fit_plan_lolo_too_few_fit_rows() -> None:
    rows = [(f"g{i:02d}", "train", i % 2, "BIG" if i < 10 else "S", "BIG" if i < 10 else None) for i in range(14)]
    plan = train._fit_plan(_frame(rows), train.TrainConfig())
    assert plan.lolo == []
    assert plan.skipped_lolo == [("BIG", "lolo_too_few_fit_rows", 10)]


# --------------------------------------------------------------------------- unit: block-wise nearest distances


def _random_sketches(n: int, s: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = np.unique(rng.integers(1, 2**63, size=s * 6, dtype=np.uint64))
    base = rng.choice(pool, size=s * 2, replace=False)
    out = []
    for i in range(n):
        # related genomes: draw from a shared pool so distances spread over (0, 1)
        k = int(rng.integers(0, s))
        own = rng.choice(pool, size=s * 2, replace=False)
        merged = np.unique(np.concatenate([base[: s + k], own]))
        out.append(np.sort(merged)[:s])
    if n > 3:
        out[3] = out[2].copy()  # an identical pair (distance 0)
    return np.vstack(out).astype(np.uint64)


def test_mash_blocks_equal_pairwise_and_distances_to() -> None:
    from genome2mic import sketch as sk

    S = _random_sketches(40, 64, seed=1)
    dense = sk.pairwise_distances(S, k=21)
    query = np.array([0, 2, 5, 7, 11, 39])
    ref = np.array([1, 3, 4, 5, 9, 20, 21, 30])
    blocks = train._MashBlocks(S, 21, query, ref, ref_block=3)
    D = blocks.block(0, len(query))
    np.testing.assert_array_equal(D, dense[np.ix_(query, ref)])
    for i, q in enumerate(query):
        np.testing.assert_array_equal(D[i], sk.distances_to(S[q], S[ref], 21))
    assert D[1, 1] == 0.0  # rows 2 and 3 are identical sketches


def test_species_nearest_matches_dense_minima() -> None:
    from genome2mic import sketch as sk

    S = _random_sketches(60, 48, seed=2)
    dense = sk.pairwise_distances(S, k=21)
    sd = train.SpeciesData(
        species="KPNEU", known=pd.DataFrame(), known_columns=[], class_by_column={},
        sketch_ids=[f"g{i:02d}" for i in range(60)], sketches=S, sketch_k=21, unitigs=None, unitig_rows=None,
        unitig_index=None,
    )
    rows = []
    for i in range(50):
        cluster = f"C{i % 7}"
        rows.append((f"g{i:02d}", "train", i % 4, cluster, "C1" if cluster == "C1" else None))
    rows += [(f"g{i:02d}", "test", None, "T", None) for i in range(50, 60)]
    frame = _frame(rows)
    plan = train._fit_plan(frame, train.TrainConfig())
    # Second "drug": a subset of the genomes, so the fit sets differ between the two requests.
    frame_b = frame.iloc[::2].reset_index(drop=True)
    plan_b = train._fit_plan(frame_b, train.TrainConfig())
    pos_a = sd.sketch_position(frame["genome_id"].tolist())
    pos_b = sd.sketch_position(frame_b["genome_id"].tolist())
    out = train._species_nearest(sd, {"a": (pos_a, plan), "b": (pos_b, plan_b)}, query_block=7)
    for drug, pos, p, fr in (("a", pos_a, plan, frame), ("b", pos_b, plan_b, frame_b)):
        got = out[drug]
        assert np.isnan(got.test[fr["split"].to_numpy(dtype=object) == "train"]).all()
        assert np.isnan(got.cv[fr["split"].to_numpy(dtype=object) == "test"]).all()
        for g in p.cv:
            np.testing.assert_array_equal(got.cv[g.predict_idx], dense[np.ix_(pos[g.predict_idx], pos[g.fit_idx])].min(axis=1))
        np.testing.assert_array_equal(
            got.test[p.test.predict_idx], dense[np.ix_(pos[p.test.predict_idx], pos[p.test.fit_idx])].min(axis=1)
        )
        for g in p.lolo:
            np.testing.assert_array_equal(
                got.lolo[g.lineage][g.predict_idx], dense[np.ix_(pos[g.predict_idx], pos[g.fit_idx])].min(axis=1)
            )
    assert plan.lolo and plan_b.lolo
    assert not hasattr(sd, "distances")


# --------------------------------------------------------------------------- lab_exact (disk rows are not exact MICs)


def test_preds_lab_exact_follows_the_interval_and_method_rule(project: dict) -> None:
    from genome2mic.mic import lab_exact_mask

    paths = project["paths"]
    preds = _preds(project)
    labels = pd.read_parquet(paths.labels, columns=["genome_id", "drug", "method"])
    merged = preds.merge(labels.astype({"genome_id": str}), on=["genome_id", "drug"], how="left", validate="many_to_one")
    assert merged["method"].notna().all()
    assert preds["lab_exact"].dtype == bool
    want = lab_exact_mask(merged["lab_lower"].to_numpy(float), merged["lab_upper"].to_numpy(float), merged["method"].to_numpy(object))
    np.testing.assert_array_equal(preds["lab_exact"].to_numpy(dtype=bool), want)
    disk = (merged["method"] == "disk").to_numpy(dtype=bool)
    assert not preds["lab_exact"].to_numpy(dtype=bool)[disk].any()


def _disk_pair() -> SimpleNamespace:
    """Six train rows: four dilution exact MICs and two disk I-only rows whose I range is one step."""
    lo = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
    hi = np.array([8.0, 8.0, 4.0, 2.0, 2.0, 2.0])
    method = np.array(["dilution", "dilution", "gradient", "dilution", "disk", "disk"], dtype=object)
    frame = pd.DataFrame({"genome_id": [f"g{i}" for i in range(6)], "mic_lower": lo, "mic_upper": hi, "method": method})
    return SimpleNamespace(frame=frame, lo=lo, hi=hi, lab_exact=train._lab_exact(frame, lo, hi, "KPNEU x meropenem"))


def test_lab_exact_mask_of_a_pair_excludes_disk_rows() -> None:
    pair = _disk_pair()
    assert pair.lab_exact.tolist() == [True, True, True, True, False, False]
    no_method = pair.frame.drop(columns="method")
    assert train._lab_exact(no_method, pair.lo, pair.hi, "x").tolist() == [True] * 6


def test_conformal_residuals_skip_disk_rows() -> None:
    pair = _disk_pair()
    # Perfect predictions on the dilution rows; the disk rows are 5 steps off and must not widen q.
    oof = np.log2(pair.hi) + np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0])
    log = DropLog("train")
    q, n = train._conformal(oof, pair, train.TrainConfig(alpha=0.5), log, "KPNEU x meropenem")
    assert n == 4 and q == 0.0
    reasons = {r.reason: r.n_dropped for r in log.records}
    assert [v for k, v in reasons.items() if "not a measured MIC" in k] == [2]


def test_b2_fit_through_train_skips_disk_rows() -> None:
    pair = _disk_pair()
    X = sp.csr_matrix(np.array([[1], [1], [0], [0], [1], [1]], dtype=np.float32))
    model = train._new_model("b2_xgb_steps", train.TrainConfig(nthread=1, max_rounds=5, early_stopping_rounds=2))
    log = DropLog("train")
    train._fit(model, X, pair.lo, pair.hi, ["gene_a"], None, log, exact_rows=pair.lab_exact)
    assert model.n_exact_ == 4
    assert set(model.classes_.tolist()) == {1, 2, 3}


# --------------------------------------------------------------------------- _check_preds: LOLO rows (#5)


def _check_preds_inputs(split_value: str, genome_id: str) -> tuple[pd.DataFrame, SimpleNamespace]:
    frame = pd.DataFrame(
        {
            "genome_id": ["t1", "t2", "x1", "y1"],
            "split": pd.array(["train", "train", "test", "train"], dtype="string"),
            "lolo_lineage": pd.array(["C0", "C0", None, "C9"], dtype="string"),
        }
    )
    table = pd.DataFrame(
        {
            "genome_id": ["t1", "x1", genome_id],
            "model": "aft_known",
            "split": ["cv", "test", split_value],
            "pred_mic": [4.0, 4.0, 4.0],
            "band_low": [2.0, 2.0, 2.0],
            "band_high": [8.0, 8.0, 8.0],
        }
    )
    return table, SimpleNamespace(frame=frame)


def test_check_preds_accepts_lolo_rows_of_train_genomes_of_that_lineage() -> None:
    table, pair = _check_preds_inputs("lolo_C0", "t2")
    train._check_preds(table, pair, "KPNEU x meropenem")


def test_check_preds_rejects_a_lolo_row_on_a_test_genome() -> None:
    from genome2mic.errors import ContractViolation

    table, pair = _check_preds_inputs("lolo_C0", "x1")
    with pytest.raises(ContractViolation, match=r"lolo_C0 row for x1 \(split=test"):
        train._check_preds(table, pair, "KPNEU x meropenem")


def test_check_preds_rejects_a_lolo_row_on_a_train_genome_of_another_lineage() -> None:
    from genome2mic.errors import ContractViolation

    table, pair = _check_preds_inputs("lolo_C0", "y1")
    with pytest.raises(ContractViolation, match=r"lolo_C0 row for y1 \(split=train, lolo_lineage=C9\)"):
        train._check_preds(table, pair, "KPNEU x meropenem")


# --------------------------------------------------------------------------- shared k-mer set (#1)


def test_bundle_records_the_kmer_set_it_was_trained_on(project: dict) -> None:
    from genome2mic.features.unitigs import read_kmer_set_sha1

    paths = project["paths"]
    manifest = json.loads(paths.models_manifest.read_text())
    for species, drugs in manifest["species"].items():
        shipped = read_kmer_set_sha1(paths.models_dir / species / "unitig_kmers.npz")
        assert shipped == read_kmer_set_sha1(paths.unitig_kmers(species))
        for drug in drugs:
            feats = json.loads((paths.model_dir(species, drug) / "features.json").read_text())
            assert feats["unitig_kmer_set_sha1"] == shipped
            meta = json.loads((paths.model_dir(species, drug) / "meta.json").read_text())
            assert meta["inputs_sha1"] == train.compute_inputs_sha1(paths, species)


def _copy_project(project: dict, tmp_path: Path) -> Paths:
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(project["paths"].root, root, symlinks=True)
    return Paths(root=root, configs_dir=root / "configs")


def _other_kmer_set(paths: Paths, species: str) -> None:
    """Replace the processed k-mer set with a different one (a rebuilt unitig stage)."""
    from genome2mic.features import unitigs as ug

    old = ug.load_kmer_set(paths.unitig_kmers(species))
    keep = np.ones(old.n_kmers, dtype=bool)
    keep[0] = False
    ug.save_kmer_set(
        paths.unitig_kmers(species),
        ug.KmerSet(old.kmers[keep], old.pattern_col[keep], old.n_patterns, old.k, old.presence_fraction, species, old.backend),
    )


def test_pipeline_rejects_a_drug_trained_on_another_kmer_set(project: dict, tmp_path: Path) -> None:
    from genome2mic.predict.pipeline import BundleError

    paths = _copy_project(project, tmp_path)
    PredictionPipeline(paths.models_dir, paths.configs_dir).load()  # consistent bundle loads
    manifest = json.loads(paths.models_manifest.read_text())
    species = next(iter(manifest["species"]))
    drug = manifest["species"][species][0]
    features_path = paths.model_dir(species, drug) / "features.json"
    feats = json.loads(features_path.read_text())
    feats["unitig_kmer_set_sha1"] = "0" * 40
    features_path.write_text(json.dumps(feats))
    with pytest.raises(BundleError, match=rf"{drug} was trained on 000000000000"):
        PredictionPipeline(paths.models_dir, paths.configs_dir).load()


def test_pipeline_rejects_a_replaced_species_kmer_set(project: dict, tmp_path: Path) -> None:
    """The failure the subset guard prevents: the shared set swapped under existing drug models."""
    import shutil

    from genome2mic.predict.pipeline import BundleError

    paths = _copy_project(project, tmp_path)
    species = next(iter(json.loads(paths.models_manifest.read_text())["species"]))
    _other_kmer_set(paths, species)
    shutil.copyfile(paths.unitig_kmers(species), paths.models_dir / species / "unitig_kmers.npz")
    with pytest.raises(BundleError, match="trained on"):
        PredictionPipeline(paths.models_dir, paths.configs_dir).load()


def test_subset_run_refuses_to_replace_the_kmer_set_of_drugs_it_does_not_retrain(project: dict, tmp_path: Path) -> None:
    from genome2mic.errors import ContractViolation
    from genome2mic.features.unitigs import read_kmer_set_sha1

    paths = _copy_project(project, tmp_path)
    pairs = project["pairs"]
    species, drug = str(pairs["species"].iloc[0]), str(pairs["drug"].iloc[0])
    others = sorted(str(d) for s, d in zip(pairs["species"], pairs["drug"]) if s == species and d != drug)
    assert others
    shipped = read_kmer_set_sha1(paths.models_dir / species / "unitig_kmers.npz")
    manifest_before = paths.models_manifest.read_text()
    ledger_before = paths.test_ledger.read_text()

    # Same set: a subset run is allowed (guard passes without training anything here).
    train.check_subset_unitig_sets(paths, {species: [drug]})

    _other_kmer_set(paths, species)
    with pytest.raises(ContractViolation, match=rf"{species}: this subset run would replace") as info:
        train.run(paths, project["config"], train_config=project["cfg"], pairs=[(species, drug)])
    for other in others:
        assert f"{other} (trained on {shipped[:12]})" in str(info.value)
    # Nothing was written: the shared set, the manifest and the ledger are untouched.
    assert read_kmer_set_sha1(paths.models_dir / species / "unitig_kmers.npz") == shipped
    assert paths.models_manifest.read_text() == manifest_before
    assert paths.test_ledger.read_text() == ledger_before
    # Retraining every drug of the species together replaces the set for all of them: allowed.
    train.check_subset_unitig_sets(paths, {species: [drug, *others]})
    # A drug bundle written before unitig_kmer_set_sha1 existed is compared with the shipped set.
    legacy = paths.model_dir(species, others[0]) / "features.json"
    feats = json.loads(legacy.read_text())
    feats.pop("unitig_kmer_set_sha1")
    legacy.write_text(json.dumps(feats))
    with pytest.raises(ContractViolation, match=rf"{others[0]} \(trained on {shipped[:12]}\)"):
        train.check_subset_unitig_sets(paths, {species: [drug]})


def test_exact_weight_reaches_only_the_aft_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """TrainConfig.exact_weight up-weights lab_exact rows in the AFT fit; B1/B2 never see weights."""
    seen: dict[str, object] = {}

    class Spy:
        def __init__(self, name: str) -> None:
            self.name = name

        def fit(self, X, lo, hi, names, groups=None, droplog=None, exact_rows=None, sample_weight=None):  # noqa: ANN001
            seen[self.name] = sample_weight
            return self

    exact = np.array([True, False, True, False])
    for name in (train.MODEL_B2, train.MODEL_AFT_KNOWN):
        train._fit(Spy(name), None, None, None, [], None, DropLog("t"), exact_rows=exact, exact_weight=2.0)
    assert seen[train.MODEL_B2] is None
    assert list(seen[train.MODEL_AFT_KNOWN]) == [2.0, 1.0, 2.0, 1.0]
    train._fit(Spy("aft_known_unitig"), None, None, None, [], None, DropLog("t"), exact_rows=exact, exact_weight=1.0)
    assert seen["aft_known_unitig"] is None
    cfg = train.TrainConfig()
    assert cfg.exact_weight == 2.0 and cfg.as_dict()["exact_weight"] == 2.0 and cfg.as_dict()["band"] == "asym_tuned"


def test_bundle_levels_are_those_that_passed_in_every_calling_fold() -> None:
    from genome2mic.models.conformal import BandParams  # noqa: PLC0415

    grid = (0.08, 0.05, 0.025, 0.01, 0.005)

    def bp(passing: tuple[float, ...], gate: bool) -> BandParams:
        return BandParams(q_up=1.0, q_low=1.0, alpha_up=passing[0] if passing else 0.005, alpha_low=0.05,
                          active_gate_open=gate, n_residuals=10, passing_alphas=passing)

    assert train.bundle_allowed_alphas({}, grid) is None
    # Fold 0 passed at 0.08 and 0.025; fold 1 only at 0.025 and 0.01; fold 2 issued no calls (closed: ignored).
    folds = {0: bp((0.08, 0.025), True), 1: bp((0.025, 0.01), True), 2: bp((), False)}
    assert train.bundle_allowed_alphas(folds, grid) == (0.025,)
    assert train.bundle_allowed_alphas({0: bp((0.08,), True), 1: bp((0.01,), True)}, grid) == ()  # no shared level
    assert train.bundle_allowed_alphas({0: bp((), False), 1: bp((), False)}, grid) == ()  # no fold calls -> closed


def test_oof_gate_check_closes_the_bundle_gate_on_calling_fold_vme() -> None:
    from genome2mic.models.conformal import BandParams  # noqa: PLC0415

    params = BandParams(q_up=1.0, q_low=1.0, alpha_up=0.05, alpha_low=0.05, active_gate_open=True, n_residuals=10,
                        inner_vme_ucb=0.01)
    n = 200
    folds = np.repeat([0.0, 1.0], n // 2)
    lab = np.array(["R"] * 10 + ["S"] * 90 + ["R"] * 100, dtype=object)
    # Fold 0 calls with 1 VME among its 10 lab R; fold 1 makes no active call (100 lab R would dilute to 1 / 110).
    calls = np.array(["likely_active"] + ["uncertain"] * 9 + ["likely_active"] * 90 + ["uncertain"] * 100, dtype=object)
    out = train._oof_gate_check(params, [(calls, np.arange(n))], None, lab, folds, 0.015, "t")  # type: ignore[arg-type]
    assert not out.active_gate_open and out.oof_call_vme_ucb == pytest.approx(2 / 11)
    # Enough calling-fold lab R and no VME: the gate stays open.
    lab_ok = np.array(["R"] * 100 + ["S"] * 100, dtype=object)
    calls_ok = np.array(["uncertain"] * 100 + ["likely_active"] * 100, dtype=object)
    folds_ok = np.tile([0.0, 1.0], n // 2)
    ok = train._oof_gate_check(params, [(calls_ok, np.arange(n))], None, lab_ok, folds_ok, 0.015, "t")  # type: ignore[arg-type]
    assert ok.active_gate_open and ok.oof_call_vme_ucb == pytest.approx(1 / 101)
    # Uncallable pairs and closed gates are untouched.
    assert train._oof_gate_check(params, [(calls, np.arange(n))], None, None, folds, 0.015, "t") is params  # type: ignore[arg-type]
