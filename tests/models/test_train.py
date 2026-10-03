"""Integration tests for the training orchestrator, the evaluation stage and the CLI.

A small seeded synthetic project (KPNEU only, 30 kb genomes) is generated once per
module and run through every stage with relaxed inclusion thresholds and tiny
xgboost budgets. All data is SYNTHETIC; the assertions check plumbing and contract
rules, never predictive quality.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic import ingest, qc
from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.config import load_config
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
