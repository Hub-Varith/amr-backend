"""Imported data releases, ``train --cv-only``, panel-edge caps, cross-conformal CV bands,
the preds ``call`` column, and the OOF comparison harness.

Everything runs on a small SYNTHETIC project: a "release" directory is assembled from
its processed files (as a team release would ship them: no sketches, no unitigs, no
interim tool output, no species references) and imported into a fresh root.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from genome2mic import cli, ingest, qc
from genome2mic.config import Breakpoint, load_config
from genome2mic.errors import ContractViolation, ToolNotAvailable
from genome2mic.eval import leakage, oof_compare, report
from genome2mic.eval import run as eval_run
from genome2mic.features import known_amr
from genome2mic.ingest import release
from genome2mic.mic import panel_caps_log2
from genome2mic.models import train
from genome2mic.models.conformal import (
    asym_band,
    asym_quantiles,
    robust_q_low,
    conformal_q,
    gate_calls,
    signed_residual_steps,
    tune_band,
)
from genome2mic.paths import Paths
from genome2mic.predict import rank
from genome2mic.splits import lineages, make_splits
from genome2mic.synthetic import generate

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
KNOWN_MODELS = ("b1_lookup", "b2_xgb_steps", "aft_known")
SELECT_MODEL = "aft_b2_select"
"""Derived (v0.6): the in-fold choice of aft_known, B2 or their average; the shipped bundle."""
RELEASE_FILES = ("labels.parquet", "known_amr.parquet", "known_amr_columns.csv", "lineages.parquet",
                 "splits.parquet", "pairs_kept.csv", "label_counts.csv")


def _write_sums(directory: Path, names: list[str]) -> None:
    lines = [f"{hashlib.sha256((directory / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def release_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A release directory built from a synthetic project's processed files (+ RELEASE, SHA256SUMS)."""
    logging.getLogger("genome2mic").setLevel(logging.WARNING)
    root = tmp_path_factory.mktemp("release_source")
    generate.run(Paths(root=root, configs_dir=REPO_CONFIGS), seed=5, n_kpneu=90, n_ecoli=0, genome_length=30_000)
    paths = Paths(root=root, configs_dir=root / "configs")
    config = load_config(paths.configs_dir)
    ingest.run(paths, config, min_ns=5, min_s=5, min_levels=3)
    qc.run(paths, config)
    known_amr.run(paths, config)
    lineages.run(paths, config)
    make_splits.run(paths, config, seed=3)
    out = tmp_path_factory.mktemp("release") / "processed"
    out.mkdir()
    for name in RELEASE_FILES:
        shutil.copyfile(paths.processed_dir / name, out / name)
    (out / "RELEASE").write_text("release=test-release\ncreated_utc=2026-01-01T00:00:00Z\n", encoding="utf-8")
    _write_sums(out, [*RELEASE_FILES, "RELEASE"])
    return out


@pytest.fixture(scope="module")
def imported(release_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """import-release -> train --cv-only (known models) -> evaluate -> report on a fresh root."""
    root = tmp_path_factory.mktemp("imported") / "root"
    paths = Paths(root=root, configs_dir=REPO_CONFIGS)
    config = load_config(REPO_CONFIGS)
    summary = release.run(paths, config, release_dir=release_dir)
    cfg = train.TrainConfig(models=KNOWN_MODELS, min_count=3, nthread=1, max_rounds=25, early_stopping_rounds=5, cv_only=True)
    trained = train.run(paths, config, train_config=cfg)
    metrics_table = eval_run.run(paths, config)
    out = report.run(paths, config, make_figures=False)
    return {"paths": paths, "config": config, "summary": summary, "trained": trained, "cfg": cfg,
            "metrics": metrics_table, "report": out}


# --------------------------------------------------------------------------- import-release


def test_import_copies_files_byte_for_byte_and_writes_qc(imported: dict, release_dir: Path) -> None:
    paths: Paths = imported["paths"]
    for name in (*RELEASE_FILES, "RELEASE", "SHA256SUMS"):
        assert (paths.processed_dir / name).read_bytes() == (release_dir / name).read_bytes(), name
    qc_table = pd.read_parquet(paths.qc)
    known = pd.read_parquet(paths.known_amr, columns=["genome_id"])
    assert sorted(qc_table["genome_id"]) == sorted(known["genome_id"].astype(str))
    assert qc_table["qc_pass"].dtype == bool and qc_table["qc_pass"].all()
    assert qc_table["n_contigs"].isna().all() and qc_table["qc_fail_reason"].isna().all()
    record = json.loads((paths.processed_dir / release.IMPORTED_RELEASE_FILE).read_text())
    assert record["release"] == "test-release" and record["sha256_verified"] is True
    assert "QC not assessed: release ships no assemblies" in record["notes"]
    assert "no Mash sketches: nearest_training_distance unavailable" in record["notes"]
    assert "no unitig matrix" in record["notes"]


def test_import_refuses_a_checksum_mismatch(release_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    shutil.copytree(release_dir, bad)
    labels = pd.read_parquet(bad / "labels.parquet")
    labels.iloc[:-1].to_parquet(bad / "labels.parquet")
    paths = Paths(root=tmp_path / "root", configs_dir=REPO_CONFIGS)
    with pytest.raises(release.ReleaseError, match="labels.parquet"):
        release.run(paths, load_config(REPO_CONFIGS), release_dir=bad)
    assert not paths.processed_dir.exists(), "nothing may be written when verification fails"


def test_import_refuses_unlisted_required_file_and_unknown_drug(release_dir: Path, tmp_path: Path) -> None:
    config = load_config(REPO_CONFIGS)
    unlisted = tmp_path / "unlisted"
    shutil.copytree(release_dir, unlisted)
    _write_sums(unlisted, ["labels.parquet", "known_amr.parquet", "lineages.parquet", "pairs_kept.csv"])
    with pytest.raises(release.ReleaseError, match="splits.parquet"):
        release.run(Paths(root=tmp_path / "r1", configs_dir=REPO_CONFIGS), config, release_dir=unlisted)

    odd = tmp_path / "odd"
    shutil.copytree(release_dir, odd)
    pairs = pd.read_csv(odd / "pairs_kept.csv")
    pairs.loc[0, "drug"] = "unobtainium"
    pairs.to_csv(odd / "pairs_kept.csv", index=False)
    _write_sums(odd, [*RELEASE_FILES, "RELEASE"])
    with pytest.raises(release.ReleaseError, match="unobtainium"):
        release.run(Paths(root=tmp_path / "r2", configs_dir=REPO_CONFIGS), config, release_dir=odd)


def test_import_refuses_to_replace_different_frozen_splits(release_dir: Path, tmp_path: Path) -> None:
    paths = Paths(root=tmp_path / "root", configs_dir=REPO_CONFIGS)
    paths.processed_dir.mkdir(parents=True)
    splits = pd.read_parquet(release_dir / "splits.parquet")
    splits.iloc[::-1].reset_index(drop=True).to_parquet(paths.splits)
    with pytest.raises(ContractViolation, match="frozen"):
        release.run(paths, load_config(REPO_CONFIGS), release_dir=release_dir)
    # Re-importing the same release over identical splits is fine.
    shutil.copyfile(release_dir / "splits.parquet", paths.splits)
    release.run(paths, load_config(REPO_CONFIGS), release_dir=release_dir)


def test_s3_release_runs_aws_s3_sync_then_imports(release_dir: Path, tmp_path: Path) -> None:
    """The --s3-release path with a fake ``aws`` (nothing touches S3)."""
    calls: list[dict] = []

    def fake_aws(command, check, env):  # noqa: ANN001 - subprocess.run signature
        calls.append({"command": list(command), "env": env})
        shutil.copytree(release_dir, Path(command[-1]), dirs_exist_ok=True)
        return SimpleNamespace(returncode=0)

    paths = Paths(root=tmp_path / "root", configs_dir=REPO_CONFIGS)
    summary = release.run(
        paths, load_config(REPO_CONFIGS), s3_release="test-release", aws_profile="g2m",
        runner=fake_aws, which=lambda tool: "/usr/bin/aws",
    )
    assert calls[0]["command"] == [
        "aws", "s3", "sync", "s3://g2m-data-v1/releases/test-release/", str(paths.data_dir / "release_test-release"),
    ]
    assert calls[0]["env"]["AWS_PROFILE"] == "g2m"
    assert summary.release == "test-release"
    assert summary.source_dir == (paths.data_dir / "release_test-release").resolve()
    assert (paths.processed_dir / "labels.parquet").read_bytes() == (release_dir / "labels.parquet").read_bytes()


def test_s3_release_errors(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path / "root", configs_dir=REPO_CONFIGS)
    config = load_config(REPO_CONFIGS)
    with pytest.raises(ToolNotAvailable):
        release.run(paths, config, s3_release="x", which=lambda tool: None)
    with pytest.raises(release.ReleaseError, match="exit code 1"):
        release.run(paths, config, s3_release="x", which=lambda tool: "/usr/bin/aws",
                    runner=lambda command, check, env: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="invalid release name"):
        release.run(paths, config, s3_release="../x", which=lambda tool: "/usr/bin/aws")
    with pytest.raises(ValueError, match="exactly one"):
        release.run(paths, config)


def test_cli_import_release_parses_both_sources() -> None:
    parser = cli.build_parser()
    ns = parser.parse_args(["import-release", "--root", "r", "--release-dir", "d"])
    assert ns.release_dir == "d" and ns.s3_release is None
    ns = parser.parse_args(["import-release", "--root", "r", "--s3-release", "name", "--aws-profile", "g2m"])
    assert ns.s3_release == "name" and ns.aws_profile == "g2m"
    with pytest.raises(SystemExit):
        parser.parse_args(["import-release", "--root", "r"])
    ns = parser.parse_args(["train", "--root", "r", "--cv-only"])
    assert cli._train_config(ns).cv_only is True and cli._train_config(ns).lolo is False


# --------------------------------------------------------------------------- train --cv-only


def _all_preds(imported: dict) -> pd.DataFrame:
    return eval_run.load_preds(imported["paths"])


def test_cv_only_writes_no_test_or_lolo_rows_and_no_ledger(imported: dict) -> None:
    paths: Paths = imported["paths"]
    preds = _all_preds(imported)
    assert set(preds["split"]) == {"cv"}
    assert set(preds["model"]) == {*KNOWN_MODELS, SELECT_MODEL}
    assert not paths.test_ledger.exists()
    splits = pd.read_parquet(paths.splits)
    test_ids = set(splits.loc[splits["split"] == "test", "genome_id"].astype(str))
    assert test_ids and not (set(preds["genome_id"]) & test_ids)
    log = pd.read_csv(paths.drop_log("train"))
    assert (log["reason"] == "cv_only: non-train (test) rows not loaded").any()
    check = leakage.check_test_touched_once(paths)
    assert check.passed is True and "test set not scored" in check.detail


def test_release_run_has_null_distances_and_a_bundle_without_sketches(imported: dict) -> None:
    paths: Paths = imported["paths"]
    preds = _all_preds(imported)
    assert preds["nearest_training_distance"].isna().all()
    manifest = json.loads(paths.models_manifest.read_text())
    assert manifest["species"]["KPNEU"]
    assert not (paths.models_dir / "reference_sketches.npz").exists()
    assert not (paths.models_dir / "KPNEU" / "train_sketches.npz").exists()
    drug = manifest["species"]["KPNEU"][0]
    meta = json.loads((paths.model_dir("KPNEU", drug) / "meta.json").read_text())
    conformal = json.loads((paths.model_dir("KPNEU", drug) / "conformal.json").read_text())
    assert meta["cv_only"] is True and meta["n_test"] == 0
    assert meta["call_standard"] == ["CLSI", "2024"]
    assert conformal["cap_low_log2"] <= conformal["cap_high_log2"]
    assert meta["panel_caps_log2"] == [conformal["cap_low_log2"], conformal["cap_high_log2"]]
    assert set(conformal["q_cross_conformal_by_fold"]) == {"0", "1", "2", "3", "4"} or conformal["q_cross_conformal_by_fold"]


def test_final_caps_come_from_all_train_rows(imported: dict) -> None:
    paths: Paths = imported["paths"]
    labels = pd.read_parquet(paths.labels)
    splits = pd.read_parquet(paths.splits)
    for drug in json.loads(paths.models_manifest.read_text())["species"]["KPNEU"]:
        conformal = json.loads((paths.model_dir("KPNEU", drug) / "conformal.json").read_text())
        pair = labels.loc[(labels["species"] == "KPNEU") & (labels["drug"] == drug)].merge(splits[["genome_id", "split"]], on="genome_id")
        pair = pair.loc[pair["split"] == "train"]
        expected = panel_caps_log2(pair["mic_lower"].to_numpy(float), pair["mic_upper"].to_numpy(float))
        assert (conformal["cap_low_log2"], conformal["cap_high_log2"]) == expected, drug


def test_cv_predictions_respect_their_fold_caps(imported: dict) -> None:
    """Each fold's raw predictions are clipped to the caps of that fold's own fit rows."""
    paths: Paths = imported["paths"]
    preds = _all_preds(imported)
    labels = pd.read_parquet(paths.labels)
    splits = pd.read_parquet(paths.splits)
    frame = preds.merge(splits[["genome_id", "fold"]], on="genome_id")
    for (drug, fold), block in frame.groupby(["drug", "fold"]):
        pair = labels.loc[(labels["species"] == "KPNEU") & (labels["drug"] == drug)].merge(splits, on="genome_id")
        fit = pair.loc[(pair["split"] == "train") & (pair["fold"] != fold)]
        lo, hi = panel_caps_log2(fit["mic_lower"].to_numpy(float), fit["mic_upper"].to_numpy(float))
        steps = np.log2(block["pred_mic"].to_numpy(float))
        assert steps.min() >= np.floor(lo) - 1e-9 and steps.max() <= np.ceil(hi) + 1e-9, (drug, fold)


def _lab_obj(series: pd.Series) -> np.ndarray:
    return series.astype(object).where(series.notna(), None).to_numpy()


def test_cv_bands_are_tuned_cross_conformal(imported: dict) -> None:
    """Fold f's band (and active-call gate) is re-derived from the other folds' OOF rows only.

    Upper level: :func:`tune_band` on the other folds (nested cross-conformal call VME);
    half-widths: signed exact residuals of the other folds; band: :func:`asym_band`.
    """
    paths: Paths = imported["paths"]
    config = imported["config"]
    preds = _all_preds(imported)
    splits = pd.read_parquet(paths.splits)
    known = pd.read_parquet(paths.known_amr).set_index("genome_id")
    frame = preds.loc[preds["model"] == "aft_known"].merge(splits[["genome_id", "fold"]], on="genome_id")
    checked = gated = 0
    for drug, block in frame.groupby("drug"):
        block = block.reset_index(drop=True)
        bp = config.call_breakpoint("KPNEU", drug)
        nat = config.is_naturally_resistant("KPNEU", drug)
        pred = block["pred_mic"].to_numpy(float)
        p = np.log2(pred)
        lo, hi = block["lab_lower"].to_numpy(float), block["lab_upper"].to_numpy(float)
        exact = block["lab_exact"].to_numpy(bool)
        lab = _lab_obj(block["lab_sir_rederived"])
        folds = block["fold"].to_numpy(float)
        marker = np.asarray(rank.strong_marker_mask(known.loc[block["genome_id"]], config.drugs.get(drug)), dtype=bool)

        def calls_fn(idx, low, high, bp=bp, nat=nat, marker=marker):
            return rank.call_array(low, high, bp, natural_resistance=nat, strong_marker=marker[idx])

        all_folds = sorted(int(f) for f in np.unique(folds))
        for fold in all_folds:
            others = [f for f in all_folds if f != fold]
            a, gate, _ = tune_band(p, lo, hi, exact, lab, folds, others, calls_fn, callable_pair=bp is not None and not nat)
            cal = np.isin(folds, others)
            r = signed_residual_steps(p[cal], lo[cal], hi[cal], exact_rows=exact[cal])
            q_up, q_low, certified = asym_quantiles(r, a, 0.05)
            by_fold = [signed_residual_steps(p[folds == g], lo[folds == g], hi[folds == g], exact_rows=exact[folds == g])
                       for g in others]
            q_low, _ = robust_q_low(by_fold, q_low, 0.05, 0.10)
            rows = np.where(folds == fold)[0]
            low, high = asym_band(pred[rows], q_up, q_low)
            assert np.array_equal(low, block["band_low"].to_numpy(float)[rows]), (drug, fold)
            assert np.array_equal(high, block["band_high"].to_numpy(float)[rows]), (drug, fold)
            expected = gate_calls(calls_fn(rows, low, high), gate and certified)
            assert list(_lab_obj(block["call"])[rows]) == list(expected), (drug, fold)
            gated += int(not (gate and certified))
            checked += 1
    assert checked > 0


def test_bundle_carries_the_tuned_band_and_the_pipeline_reads_it(imported: dict) -> None:
    from genome2mic.predict.pipeline import PredictionPipeline  # noqa: PLC0415

    paths: Paths = imported["paths"]
    found = 0
    for conf_path in sorted(paths.models_dir.glob("KPNEU/*/conformal.json")):
        conf = json.loads(conf_path.read_text())
        assert conf["band_kind"] == "asymmetric_tuned"
        assert conf["q"] == conf["q_up"] and conf["q_low"] >= 0 and isinstance(conf["active_gate_open"], bool)
        assert conf["alpha_up"] in train.TrainConfig().band_alpha_grid or conf["alpha_up"] == 0.05
        assert set(conf["band_cross_conformal_by_fold"]) == set(conf["q_cross_conformal_by_fold"])
        # The bundle is never narrower than a level a fold validated, and stays closed when no fold opened.
        calling = [f for f in conf["band_cross_conformal_by_fold"].values() if f["active_gate_open"]]
        shared = [a for a in train.TrainConfig().band_alpha_grid if all(a in f["passing_alpha_up"] for f in calling)]
        if conf["inner_call_vme_ucb"] is not None:  # callable pair whose bundle gate opened
            assert calling and conf["alpha_up"] in shared and conf["alpha_up"] in conf["passing_alpha_up"]
            assert conf["allowed_alpha_up"] == shared
            if conf["active_gate_open"]:
                assert conf["oof_call_vme_ucb"] is not None and conf["oof_call_vme_ucb"] <= 0.015 + 1e-12
        elif conf["allowed_alpha_up"] is not None and not calling:
            assert conf["active_gate_open"] is False
        found += 1
    assert found
    # The release ships no reference sketches, so load the drug bundles one by one.
    predictor = PredictionPipeline(paths.models_dir, REPO_CONFIGS)
    predictor.config = imported["config"]
    for conf_path in sorted(paths.models_dir.glob("KPNEU/*/conformal.json")):
        drug = conf_path.parent.name
        bundle = predictor._load_drug("KPNEU", drug, conf_path.parent, False)
        conf = json.loads(conf_path.read_text())
        assert bundle.q == conf["q_up"] and bundle.q_low == conf["q_low"]
        assert bundle.active_gate_open is conf["active_gate_open"]


def test_symmetric_band_option_changes_the_run_id(imported: dict) -> None:
    paths: Paths = imported["paths"]
    tuned = train.TrainConfig(cv_only=True)
    sym = train.TrainConfig(cv_only=True, band="symmetric")
    assert train.compute_run_id(tuned, paths.splits) != train.compute_run_id(sym, paths.splits)


def test_call_column_follows_the_pipeline_rule(imported: dict) -> None:
    config = imported["config"]
    preds = _all_preds(imported)
    assert "call" in preds.columns
    known = pd.read_parquet(imported["paths"].known_amr).set_index("genome_id")
    splits = pd.read_parquet(imported["paths"].splits)[["genome_id", "fold"]]
    preds = preds.merge(splits, on="genome_id")
    for (drug, model, fold), block in preds.groupby(["drug", "model", "fold"]):
        bp = config.call_breakpoint("KPNEU", drug)
        marker = rank.strong_marker_mask(known.loc[block["genome_id"]], config.drugs.get(drug))
        expected = rank.call_array(
            block["band_low"].to_numpy(float), block["band_high"].to_numpy(float), bp,
            natural_resistance=config.is_naturally_resistant("KPNEU", drug), strong_marker=marker,
        )
        got = block["call"].astype(object).where(block["call"].notna(), None).to_numpy()
        # The fold's active-call gate either is open (rule as is) or withholds every likely_active.
        assert list(got) in (list(expected), list(gate_calls(expected, False))), (drug, model, fold)
    ampicillin = preds.loc[preds["drug"] == "ampicillin", "call"]
    if len(ampicillin):
        assert (ampicillin == "likely_inactive").all(), "KPNEU ampicillin: natural resistance override"


def test_metrics_and_report_on_cv_only_release(imported: dict) -> None:
    table: pd.DataFrame = imported["metrics"]
    assert set(table["split"]) == {"cv"}
    assert table["n_call"].gt(0).any()
    text = imported["report"].report_path.read_text(encoding="utf-8")
    assert "test set not scored" in text.lower()
    assert report.TEST_NOT_SCORED_TEXT in text
    assert "## Data release" in text and "test-release" in text
    assert report.CALL_TEXT in text
    assert "not available for this release" in text or "Figures skipped" in text


# --------------------------------------------------------------------------- compare-oof


def _fake_reference(imported: dict, tmp_path: Path, *, shift: float = 1.0) -> Path:
    preds = _all_preds(imported)
    ref = preds.loc[preds["model"] == "aft_known", ["genome_id", "species", "drug", "pred_mic", "band_low", "band_high",
                                                    "lab_lower", "lab_upper"]].copy()
    ref["pred_mic"] = ref["pred_mic"] * 2.0**shift
    ref["band_low"] = ref["band_low"] * 2.0**shift
    ref["band_high"] = ref["band_high"] * 2.0**shift
    ref["split"] = "oof"
    ref["model"] = "multitask_aft"
    ref["mu_log2"] = np.log2(ref["pred_mic"])
    path = tmp_path / "ref_oof.parquet"
    ref.to_parquet(path)
    return path


def test_compare_oof_scores_every_model_on_identical_rows(imported: dict, tmp_path: Path) -> None:
    paths: Paths = imported["paths"]
    ref_path = _fake_reference(imported, tmp_path)
    result = oof_compare.run(paths, imported["config"], develop_preds=ref_path, out_dir=tmp_path / "out")
    table = result.table
    assert set(table["model"]) == {*KNOWN_MODELS, SELECT_MODEL, "develop:multitask_aft"}
    assert list(table.columns[3:5]) == ["vme_rate", "vme_rate_rederived"]
    per_pair = table.groupby(["species", "drug"])["n"].nunique()
    assert (per_pair == 1).all(), "every model must be scored on the same rows"
    assert (tmp_path / "out" / "oof_compare.csv").is_file()
    md = (tmp_path / "out" / "oof_compare.md").read_text(encoding="utf-8")
    assert "identical rows" in md and "not prescribing advice" in md
    assert set(result.summary["model"]) == set(table["model"])


def test_compare_oof_drops_rows_whose_lab_result_differs(imported: dict, tmp_path: Path) -> None:
    ref_path = _fake_reference(imported, tmp_path, shift=0.0)
    ref = pd.read_parquet(ref_path)
    ref.loc[ref.index[0], "lab_upper"] = ref.loc[ref.index[0], "lab_upper"] * 4
    ref.loc[ref.index[0], "lab_lower"] = ref.loc[ref.index[0], "lab_upper"] / 2
    ref.to_parquet(ref_path)
    result = oof_compare.run(imported["paths"], imported["config"], develop_preds=ref_path, out_dir=tmp_path / "o2")
    rows = pd.read_csv(tmp_path / "o2" / "oof_compare_rows.csv")
    reason = rows.loc[rows["reason"] == "lab interval differs between our preds and the reference preds", "n_dropped"]
    assert int(reason.iloc[0]) == 1
    # Same-model reference without a shift: identical point predictions -> identical EA.
    table = result.table.set_index(["drug", "model"])
    for drug in result.table["drug"].unique():
        a, b = table.loc[(drug, "aft_known")], table.loc[(drug, "develop:multitask_aft")]
        if not np.isnan(a["essential_agreement"]):
            assert a["essential_agreement"] == pytest.approx(b["essential_agreement"])


def test_compare_oof_with_an_explicit_breakpoint_table(imported: dict, tmp_path: Path) -> None:
    ref_path = _fake_reference(imported, tmp_path)
    bp_csv = tmp_path / "clsi_2019.csv"
    bp_csv.write_text(
        "species,drug,s_breakpoint,r_breakpoint,version,site\n"
        "KPNEU,meropenem,1,2,M100 29th ed,bloodstream\n"
        "KPNEU,meropenem,1,2,M100 29th ed,bloodstream\n",
        encoding="utf-8",
    )
    result = oof_compare.run(imported["paths"], imported["config"], develop_preds=ref_path,
                             out_dir=tmp_path / "o3", breakpoints_csv=bp_csv)
    assert result.table["breakpoints"].str.startswith("clsi_2019.csv").all()
    others = result.table.loc[result.table["drug"] != "meropenem"]
    assert others["n_lab_r_rederived"].eq(0).all(), "no breakpoint row -> no re-derived categories"
    bp = oof_compare.load_breakpoint_csv(bp_csv)
    assert bp[("KPNEU", "meropenem")] == Breakpoint("KPNEU", "meropenem", 1.0, 2.0, "CLSI", "2019")
    bad = tmp_path / "bad.csv"
    bad.write_text("species,drug,s_breakpoint,r_breakpoint\nKPNEU,meropenem,1,2\nKPNEU,meropenem,2,8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        oof_compare.load_breakpoint_csv(bad)


# --------------------------------------------------------------------------- pure helpers


def test_panel_caps_log2() -> None:
    # Panel <=0.25 ... >32: finite bounds 0.25..32 -> caps one step beyond: 0.125 and 64.
    lo = np.array([0.0, 0.25, 16.0, 32.0])
    hi = np.array([0.25, 0.5, 32.0, np.inf])
    assert panel_caps_log2(lo, hi) == (-3.0, 6.0)
    # Off-grid bounds are widened to the grid first.
    assert panel_caps_log2(np.array([0.0, 3.0]), np.array([0.3, np.inf])) == (-3.0, 3.0)
    # No finite bound at all: the reference grid.
    from genome2mic.mic import GRID_MAX_EXPONENT, GRID_MIN_EXPONENT

    assert panel_caps_log2(np.array([0.0]), np.array([np.inf])) == (float(GRID_MIN_EXPONENT), float(GRID_MAX_EXPONENT))


def test_apply_caps_and_call_array() -> None:
    assert train.apply_caps(np.array([-9.0, 0.0, 9.0, np.nan]), (-3.0, 6.0)).tolist()[:3] == [-3.0, 0.0, 6.0]
    assert np.isnan(train.apply_caps(np.array([np.nan]), (-3.0, 6.0))[0])
    bp = Breakpoint("KPNEU", "meropenem", 1.0, 2.0, "CLSI", "2024")
    low = [0.25, 0.5, 4.0, np.nan, 0.25]
    high = [1.0, 2.0, 16.0, np.nan, 1.0]
    calls = rank.call_array(low, high, bp, strong_marker=[False, False, False, False, True])
    assert list(calls) == ["likely_active", "uncertain", "likely_inactive", None, "likely_inactive"]
    assert list(rank.call_array(low, high, None)) == [None] * 5
    assert list(rank.call_array(low, high, None, strong_marker=[True, False, False, False, False])) == [
        "likely_inactive", None, None, None, None]
    assert set(rank.call_array(low, high, bp, natural_resistance=True)) == {"likely_inactive"}


def test_cli_compare_oof_end_to_end(imported: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref_path = _fake_reference(imported, tmp_path)
    paths: Paths = imported["paths"]
    code = cli.main([
        "compare-oof", "--root", str(paths.root), "--configs-dir", str(REPO_CONFIGS), "-q",
        "--develop-preds", str(ref_path), "--out", str(tmp_path / "cli_out"),
    ])
    assert code == 0
    assert (tmp_path / "cli_out" / "oof_compare.md").is_file()
    assert "compare-oof:" in capsys.readouterr().out


# --------------------------------------------------------------------------- --workers


def test_workers_give_identical_preds_to_a_sequential_run(imported: dict, release_dir: Path, tmp_path: Path) -> None:
    """Pairs trained in worker processes produce identical predictions (and never deadlock after
    the parent process has already run xgboost, as the ``imported`` fixture has)."""
    paths = Paths(root=tmp_path / "root", configs_dir=REPO_CONFIGS)
    config = imported["config"]
    release.run(paths, config, release_dir=release_dir)
    cfg = train.TrainConfig(**{**imported["cfg"].__dict__, "workers": 2})
    assert train.compute_run_id(cfg, paths.splits) == train.compute_run_id(imported["cfg"], paths.splits)
    summary = train.run(paths, config, train_config=cfg)
    seq = imported["trained"]
    assert summary[["species", "drug", "n_rows"]].equals(seq[["species", "drug", "n_rows"]])
    for species, drug in zip(summary["species"], summary["drug"]):
        a = pd.read_parquet(imported["paths"].preds(species, drug))
        b = pd.read_parquet(paths.preds(species, drug))
        pd.testing.assert_frame_equal(a, b)
    a_log = pd.read_csv(imported["paths"].processed_dir / "drop_log_train.csv")
    b_log = pd.read_csv(paths.processed_dir / "drop_log_train.csv")
    pd.testing.assert_frame_equal(a_log, b_log)


def test_model_select_off_ships_the_aft_model(release_dir: Path, tmp_path: Path) -> None:
    """``TrainConfig(model_select=False)`` (CLI ``--no-model-select``): no aft_b2_select rows; the AFT bundle ships."""
    paths = Paths(root=tmp_path / "root", configs_dir=REPO_CONFIGS)
    config = load_config(REPO_CONFIGS)
    release.run(paths, config, release_dir=release_dir)
    drug = pd.read_csv(paths.pairs_kept)["drug"].iloc[0]
    cfg = train.TrainConfig(models=KNOWN_MODELS, min_count=3, nthread=1, max_rounds=25, early_stopping_rounds=5,
                            cv_only=True, model_select=False)
    train.run(paths, config, train_config=cfg, pairs=[("KPNEU", drug)])
    preds = pd.read_parquet(paths.preds("KPNEU", drug))
    assert set(preds["model"]) == set(KNOWN_MODELS)
    d = paths.model_dir("KPNEU", drug)
    assert json.loads((d / "features.json").read_text())["model_class"] == "aft_known"
    assert (d / "model.ubj").is_file() and not (d / "aft").exists()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["model_select"] is None and meta["train_config"]["model_select"] is False
    assert "model_select" not in json.loads((d / "conformal.json").read_text())
