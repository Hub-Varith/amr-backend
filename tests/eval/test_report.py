"""Tests for ``genome2mic.eval.report``, ``eval.figures`` and ``eval.leakage``.

A tiny fake ``results/`` + ``data/processed/`` + ``models/`` tree (2 models x 1
drug, KPNEU) is built in ``tmp_path``. Every number is made up for the test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic.api.constants import DISCLAIMER
from genome2mic.droplog import DropLog
from genome2mic.eval import figures, leakage, report
from genome2mic.paths import Paths

SPECIES = "KPNEU"
DRUG = "meropenem"
MODELS = ("aft_known", "aft_known_unitig")
N_TRAIN = 60
N_TEST = 20
MIN_PNG_BYTES = 1024


# --------------------------------------------------------------------------- #
# Fixture: fake tree
# --------------------------------------------------------------------------- #


def _genome_ids() -> list[str]:
    return [f"573.{1000 + i}" for i in range(N_TRAIN + N_TEST)]


def _write_splits_and_lineages(paths: Paths) -> None:
    gids = _genome_ids()
    split = ["train"] * N_TRAIN + ["test"] * N_TEST
    # 4 genomes per cluster -> 15 train clusters (3 per fold), 5 test clusters.
    cluster_idx = [i // 4 for i in range(len(gids))]
    fold = [c % 5 if s == "train" else None for c, s in zip(cluster_idx, split, strict=True)]
    splits = pd.DataFrame(
        {
            "genome_id": gids,
            "species": SPECIES,
            "split": split,
            "fold": pd.array(fold, dtype="Int64"),
            "external_set": pd.array([None] * N_TRAIN + ["country_holdout"] * 5 + [None] * (N_TEST - 5), dtype="str"),
            "lolo_lineage": pd.array([None] * len(gids), dtype="str"),
        }
    )
    splits.to_parquet(paths.splits, index=False)
    lineages = pd.DataFrame(
        {
            "genome_id": gids,
            "species": SPECIES,
            "lineage_cluster": [f"{SPECIES}_ML_{c + 1:03d}" for c in cluster_idx],
            "st": pd.array(["NA"] * len(gids), dtype="str"),
            "cluster_method": "mash_single_linkage",
        }
    )
    lineages.to_parquet(paths.lineages, index=False)


def _write_labels(paths: Paths, rng: np.random.Generator) -> pd.DataFrame:
    gids = _genome_ids()
    steps = rng.integers(-5, 7, size=len(gids))  # log2 MIC steps 0.03 .. 64
    upper = 2.0 ** steps
    censor = np.where(steps <= -5, "left", np.where(steps >= 6, "right", "interval"))
    lower = np.where(censor == "left", 0.0, upper / 2)
    upper = np.where(censor == "right", np.inf, upper)
    sir = np.where(steps >= 4, "R", np.where(steps >= 2, "I", "S"))
    labels = pd.DataFrame(
        {
            "genome_id": gids,
            "biosample": [f"SAMN{i:08d}" for i in range(len(gids))],
            "species": SPECIES,
            "drug": DRUG,
            "mic_lower": lower,
            "mic_upper": upper,
            "censor": censor,
            "sir": sir,
            "raw_result": [f"={u:g}" for u in upper],
            "method": "dilution",
            "standard": "EUCAST",
            "standard_year": pd.array([2024] * len(gids), dtype="Int64"),
            "source": "BVBRC",
            "isolation_source": "blood",
            "country": "Testland",
            "year": pd.array([2021] * len(gids), dtype="Int64"),
        }
    )
    labels.to_parquet(paths.labels, index=False)
    return labels


def _write_preds(paths: Paths, labels: pd.DataFrame, rng: np.random.Generator) -> None:
    frames = []
    for model_i, model in enumerate(MODELS):
        noise = rng.integers(-1 - (1 - model_i), 2 + (1 - model_i), size=len(labels))  # weaker model noisier
        lab_step = np.where(np.isfinite(labels["mic_upper"]), np.log2(labels["mic_upper"].replace(np.inf, 64.0)), 6.0)
        pred_step = lab_step + noise
        pred = 2.0**pred_step
        band_low = pred / 2
        band_high = pred * 2
        pred_sir = np.where(pred > 8, "R", np.where(pred > 2, "I", "S"))
        split = ["cv"] * N_TRAIN + ["test"] * N_TEST
        frames.append(
            pd.DataFrame(
                {
                    "genome_id": labels["genome_id"],
                    "species": SPECIES,
                    "drug": DRUG,
                    "split": split,
                    "pred_mic": pred,
                    "band_low": band_low,
                    "band_high": band_high,
                    "lab_lower": labels["mic_lower"],
                    "lab_upper": labels["mic_upper"],
                    "pred_sir": pred_sir,
                    "lab_sir": labels["sir"],
                    "model": model,
                    "run_id": "deadbeef0123",
                    "external_set": pd.array([None] * N_TRAIN + ["country_holdout"] * 5 + [None] * (N_TEST - 5), dtype="str"),
                    "nearest_training_distance": np.where(np.array(split) == "test", rng.uniform(0, 0.03, len(labels)), np.nan),
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_parquet(paths.preds(SPECIES, DRUG), index=False)


def _write_metrics(paths: Paths) -> None:
    rows = []
    for model, vme, me, ea in (("aft_known", 0.04, 0.05, 0.80), ("aft_known_unitig", 0.012, 0.025, 0.92)):
        for split, n in (("test", N_TEST), ("cv", N_TRAIN), ("country_holdout", 5)):
            rows.append(
                {
                    "species": SPECIES,
                    "drug": DRUG,
                    "model": model,
                    "split": split,
                    "vme_rate": vme,
                    "me_rate": me,
                    "mine_rate": 0.10,
                    "categorical_agreement": 0.88,
                    "essential_agreement": ea,
                    "exact_agreement": ea - 0.3,
                    "auroc": 0.93,
                    "band_coverage": 0.91,
                    "band_width_steps": 2.0,
                    "n": n,
                    "n_exact": n - 2,
                    "n_cat": n,
                    "n_lab_r": n // 3,
                    "n_lab_s": n - n // 3,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(paths.metrics, index=False)
    by_distance = pd.DataFrame(
        [
            {"species": SPECIES, "drug": DRUG, "model": m, "distance_bin": b, "essential_agreement": ea, "vme_rate": vme, "n": n}
            for m in MODELS
            for b, ea, vme, n in (("(0, 0.005]", 0.95, 0.0, 8), ("(0.005, 0.02]", 0.90, 0.02, 7), ("(0.02, 0.05]", 0.80, 0.05, 5))
        ]
    )
    by_distance.to_parquet(paths.metrics_by_distance, index=False)


def _write_processed_small_files(paths: Paths) -> None:
    pd.DataFrame(
        [
            {"species": SPECIES, "drug": DRUG, "n": 80, "n_R": 25, "n_S": 45, "n_I": 10, "n_exact": 70, "n_censored": 10, "n_distinct_mic": 9},
            {"species": SPECIES, "drug": "ciprofloxacin", "n": 70, "n_R": 30, "n_S": 38, "n_I": 2, "n_exact": 60, "n_censored": 10, "n_distinct_mic": 8},
        ]
    ).to_csv(paths.label_counts, index=False)
    pd.DataFrame(
        [{"species": SPECIES, "drug": DRUG, "n": 80, "n_nonsusceptible": 35, "n_susceptible": 45, "n_distinct_mic": 9}]
    ).to_csv(paths.pairs_kept, index=False)
    log = DropLog("ingest")
    log.drop("evidence != Laboratory Method", 12)
    log.drop("unknown drug", 3, detail="cefepime/tazo")
    log.write(paths.drop_log("ingest"))
    pd.DataFrame(
        [
            {"column_name": "gene_blakpc_2", "source_symbol": "blaKPC-2", "class": "BETA-LACTAM", "subclass": "CARBAPENEM", "n_genomes_present": 20},
            {"column_name": "gene_blactx_m", "source_symbol": "blaCTX-M", "class": "BETA-LACTAM", "subclass": "CEPHALOSPORIN", "n_genomes_present": 30},
            {"column_name": "point_ompk36_d135dgd", "source_symbol": "ompK36_D135DGD", "class": "BETA-LACTAM", "subclass": "BETA-LACTAM", "n_genomes_present": 15},
            {"column_name": "n_class_beta_lactam", "source_symbol": None, "class": "BETA-LACTAM", "subclass": None, "n_genomes_present": 80},
        ]
    ).to_csv(paths.known_amr_columns, index=False)
    pd.DataFrame(
        {
            "col_index": [0, 1, 2],
            "pattern_id": ["u_000000", "u_000001", "u_000002"],
            "n_unitigs": [3, 1, 12],
            "unitig_sequences": [["ACGTACGTACGTACGTACGTACGTACGTACG", "TTGACC", "GGCATT"], ["CCCGGGAAATTT"], ["A" * 31] * 12],
            "train_frequency": [0.23, 0.51, 0.05],
        }
    ).to_parquet(paths.unitig_index(SPECIES), index=False)
    gids = _genome_ids()
    pd.DataFrame(
        {
            "row_index": range(len(gids)),
            "genome_id": gids,
            "split": ["train"] * N_TRAIN + ["test"] * N_TEST,
            "role": ["built"] * N_TRAIN + ["queried"] * N_TEST,
        }
    ).to_parquet(paths.unitig_rows(SPECIES), index=False)


def _write_models(paths: Paths) -> None:
    model_dir = paths.model_dir(SPECIES, DRUG)
    model_dir.mkdir(parents=True)
    importance = [
        {"feature": "gene_blakpc_2", "gain": 120.5},
        {"feature": "u_000000", "gain": 40.2},
        {"feature": "point_ompk36_d135dgd", "gain": 33.0},
        {"feature": "n_class_beta_lactam", "gain": 12.0},
        {"feature": "u_000002", "gain": 9.5},
        {"feature": "gene_blactx_m", "gain": 4.0},
    ]
    (model_dir / "importance.json").write_text(json.dumps(importance), encoding="utf-8")
    (model_dir / "features.json").write_text(
        json.dumps({"known_columns": ["gene_blakpc_2", "gene_blactx_m", "point_ompk36_d135dgd", "n_class_beta_lactam"], "unitig_cols": ["u_000000", "u_000002"]}),
        encoding="utf-8",
    )


@pytest.fixture
def fake_paths(tmp_path: Path) -> Paths:
    """A complete fake tree: 2 models x 1 drug for KPNEU, synthetic marker present."""
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs-that-do-not-exist")
    paths.processed_dir.mkdir(parents=True)
    paths.results_dir.mkdir(parents=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    (paths.raw_dir / "SYNTHETIC_DATA.md").write_text("# Synthetic data\nEverything here is simulated.\n")
    rng = np.random.default_rng(7)
    _write_splits_and_lineages(paths)
    labels = _write_labels(paths, rng)
    _write_preds(paths, labels, rng)
    _write_metrics(paths)
    _write_processed_small_files(paths)
    _write_models(paths)
    return paths


def _header_cells(table_line: str) -> list[str]:
    return [c.strip() for c in table_line.strip().strip("|").split("|")]


def _metrics_table_headers(text: str) -> list[list[str]]:
    """Header cells of every table that carries a VME column, in document order."""
    headers = [_header_cells(line) for line in text.splitlines() if line.startswith("|") and "VME" in line and "Model" in line]
    assert headers, "no metrics table header with a VME column found"
    return headers


def _first_metrics_table_header(text: str) -> list[str]:
    return _metrics_table_headers(text)[0]


# --------------------------------------------------------------------------- #
# report.run
# --------------------------------------------------------------------------- #


def test_run_writes_report_with_disclaimer_banner_and_checklist(fake_paths: Paths) -> None:
    output = report.run(fake_paths, None)

    assert output.report_path == fake_paths.report_md
    assert output.report_path.is_file()
    text = output.report_path.read_text(encoding="utf-8")
    assert DISCLAIMER in text
    # Disclaimer must be near the top: within the first 30 lines.
    assert any(DISCLAIMER in line for line in text.splitlines()[:30])
    assert "SYNTHETIC DATA" in text.splitlines()[2]
    assert "figures commonly used in AST device evaluation" in text
    assert "## Leakage checklist" in text
    assert len(output.checks) == len(leakage.CHECKS)
    assert fake_paths.drop_log("report").is_file()
    # No claim that the targets were met, no clinical validation claims.
    lowered = text.lower()
    assert "clinically validated" not in lowered
    assert "targets met" not in lowered


def test_first_metrics_table_lists_vme_before_me(fake_paths: Paths) -> None:
    report.run(fake_paths, None)
    text = fake_paths.report_md.read_text(encoding="utf-8")

    headers = _metrics_table_headers(text)
    for header in headers:  # every table: VME strictly before ME
        vme_idx = next(i for i, c in enumerate(header) if c.startswith("VME"))
        me_idx = next(i for i, c in enumerate(header) if c.startswith("ME"))
        assert vme_idx < me_idx, header
    # The first per-pair table (Model | n | VME ...) has VME as the first metric after the keys.
    per_pair = next(h for h in headers if h[0] == "Model")
    assert per_pair[:3] == ["Model", "n", per_pair[2]] and per_pair[2].startswith("VME")
    assert re.search(r"\| 1\.2% \(n=6\) \|", text), "VME rate should render as a percentage with its denominator"


def test_report_orders_test_before_cv_before_external(fake_paths: Paths) -> None:
    report.run(fake_paths, None)
    text = fake_paths.report_md.read_text(encoding="utf-8")

    i_test = text.index("#### Test set")
    i_cv = text.index("#### Cross-validation")
    i_ext = text.index("#### External and leave-one-lineage-out sets")
    assert i_test < i_cv < i_ext
    assert "country holdout" in text


def test_report_embeds_every_figure_relative_to_report(fake_paths: Paths) -> None:
    output = report.run(fake_paths, None)
    text = fake_paths.report_md.read_text(encoding="utf-8")

    expected = {
        f"vme_me_by_model_{SPECIES}_{DRUG}",
        f"ea_ca_by_model_{SPECIES}_{DRUG}",
        f"mic_confusion_{SPECIES}_{DRUG}",
        f"feature_importance_{SPECIES}_{DRUG}",
        f"band_coverage_{SPECIES}",
        f"accuracy_vs_distance_{SPECIES}",
        f"label_counts_{SPECIES}",
        f"lineage_clusters_{SPECIES}",
    }
    assert set(output.figures) == expected
    for stem, path in output.figures.items():
        assert path.is_file() and path.stat().st_size > MIN_PNG_BYTES
        assert f"](figures/{stem}.png)" in text


def test_run_on_empty_tree_still_writes_report(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path / "empty", configs_dir=tmp_path / "nope")
    output = report.run(paths, None)

    text = output.report_path.read_text(encoding="utf-8")
    assert DISCLAIMER in text
    assert "SYNTHETIC DATA" not in text
    assert "not found" in text
    assert output.figures == {}
    assert all(c["passed"] is None for c in output.checks)
    assert "NOT RUN" in text


def test_metrics_table_handles_missing_columns_and_nulls() -> None:
    frame = pd.DataFrame({"model": ["b0_resfinder", "aft_known"], "vme_rate": [0.05, None], "me_rate": [np.nan, 0.1], "n": [10, 12]})
    table = report.metrics_table(frame)
    header = _header_cells(table.splitlines()[0])
    assert header[2].startswith("VME") and header[3].startswith("ME")
    assert "5.0%" in table and "10.0%" in table
    assert "—" in table


# --------------------------------------------------------------------------- #
# Individual figure functions
# --------------------------------------------------------------------------- #


def _loaded(fake_paths: Paths) -> report.ReportInputs:
    return report.load_inputs(fake_paths)


def _assert_png(path: Path | None) -> None:
    assert path is not None
    assert path.suffix == ".png"
    assert path.is_file()
    assert path.stat().st_size > MIN_PNG_BYTES
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_fig_vme_me_by_model(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    _assert_png(figures.vme_me_by_model(inputs.metrics, SPECIES, DRUG, tmp_path / "f1.png"))
    assert figures.vme_me_by_model(inputs.metrics, "ECOLI", DRUG, tmp_path / "none.png") is None


def test_fig_ea_ca_by_model(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    _assert_png(figures.ea_ca_by_model(inputs.metrics, SPECIES, DRUG, tmp_path / "f2.png", split="cv"))


def test_fig_mic_confusion_logs_excluded_rows(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    log = DropLog("report")
    _assert_png(figures.mic_confusion(inputs.preds, SPECIES, DRUG, tmp_path / "f3.png", droplog=log))
    assert len(log) == 1
    sub = inputs.preds[(inputs.preds["model"] == "aft_known_unitig") & (inputs.preds["split"] == "test")]
    n_censored = int(((sub["lab_lower"] <= 0) | ~np.isfinite(sub["lab_upper"])).sum())
    assert log.records[0].n_dropped == n_censored
    # Falls back to another model when the main model is absent.
    only_known = inputs.preds[inputs.preds["model"] == "aft_known"]
    _assert_png(figures.mic_confusion(only_known, SPECIES, DRUG, tmp_path / "f3b.png"))


def test_fig_band_coverage(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    _assert_png(figures.band_coverage(inputs.metrics, SPECIES, tmp_path / "f4.png"))


def test_fig_accuracy_vs_distance(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    _assert_png(figures.accuracy_vs_distance(inputs.metrics_by_distance, SPECIES, tmp_path / "f5.png"))


def test_fig_feature_importance_labels_unitigs_and_known_columns(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    _assert_png(
        figures.feature_importance(
            inputs.importances[(SPECIES, DRUG)],
            SPECIES,
            DRUG,
            tmp_path / "f6.png",
            unitig_index=inputs.unitig_index[SPECIES],
            known_columns=inputs.known_columns,
        )
    )
    label = figures.feature_label("u_000000", inputs.unitig_index[SPECIES], inputs.known_columns)
    assert label.startswith("u_000000 (3 unitigs, train freq 0.23, ")
    assert figures.feature_label("gene_blakpc_2", inputs.unitig_index[SPECIES], inputs.known_columns) == (
        "gene_blakpc_2 (blaKPC-2, BETA-LACTAM/CARBAPENEM)"
    )
    assert figures.feature_label("gene_unknown", None, None) == "gene_unknown"


def test_fig_label_counts(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    _assert_png(figures.label_counts(inputs.label_counts, SPECIES, tmp_path / "f7.png"))


def test_fig_lineage_clusters(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    _assert_png(figures.lineage_clusters(inputs.lineages, inputs.splits, SPECIES, tmp_path / "f8.png"))
    _assert_png(figures.lineage_clusters(inputs.lineages, None, SPECIES, tmp_path / "f8b.png"))


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #


def test_leakage_checks_pass_on_clean_tree(fake_paths: Paths) -> None:
    results = leakage.run_checks(fake_paths)

    assert [r["check"] for r in results] == [c(fake_paths).check for c in leakage.CHECKS]
    assert all(r["passed"] is True for r in results), results


def test_leakage_detects_cluster_in_two_splits(fake_paths: Paths) -> None:
    splits = pd.read_parquet(fake_paths.splits)
    splits.loc[splits["genome_id"] == "573.1000", "split"] = "test"  # cluster 001 now spans train+test
    splits.to_parquet(fake_paths.splits, index=False)

    result = leakage.check_clusters_within_splits(fake_paths)
    assert result.passed is False
    assert "KPNEU_ML_001" in result.detail


def test_leakage_detects_cluster_in_two_folds(fake_paths: Paths) -> None:
    splits = pd.read_parquet(fake_paths.splits)
    splits.loc[splits["genome_id"] == "573.1000", "fold"] = 4
    splits.to_parquet(fake_paths.splits, index=False)

    result = leakage.check_clusters_within_splits(fake_paths)
    assert result.passed is False and "two folds" in result.detail


def test_leakage_detects_forbidden_feature(fake_paths: Paths) -> None:
    columns = pd.read_csv(fake_paths.known_amr_columns)
    columns.loc[len(columns)] = ["lineage_cluster", None, None, None, 80]
    columns.to_csv(fake_paths.known_amr_columns, index=False)

    result = leakage.check_feature_names(fake_paths)
    assert result.passed is False
    assert "lineage_cluster" in result.detail
    assert leakage.is_forbidden_feature("country_usa")
    assert leakage.is_forbidden_feature("year_2019")
    assert not leakage.is_forbidden_feature("gene_stx2")
    assert leakage.classify_features(["gene_a", "mystery", "fold"]) == (["fold"], ["mystery"])


def test_leakage_detects_forbidden_importance_feature(fake_paths: Paths) -> None:
    path = fake_paths.model_dir(SPECIES, DRUG) / "importance.json"
    content = json.loads(path.read_text())
    content.append({"feature": "st", "gain": 99.0})
    path.write_text(json.dumps(content))

    result = leakage.check_feature_names(fake_paths)
    assert result.passed is False and "importance.json" in result.detail


def test_leakage_detects_unitig_built_on_test_genome(fake_paths: Paths) -> None:
    rows = pd.read_parquet(fake_paths.unitig_rows(SPECIES))
    rows.loc[rows["genome_id"] == "573.1079", "role"] = "built"  # a test genome
    rows.to_parquet(fake_paths.unitig_rows(SPECIES), index=False)

    result = leakage.check_unitig_rows_built_on_train(fake_paths)
    assert result.passed is False and "573.1079" in result.detail


def test_leakage_unitig_check_not_run_without_role_column(fake_paths: Paths) -> None:
    rows = pd.read_parquet(fake_paths.unitig_rows(SPECIES)).drop(columns="role")
    rows.to_parquet(fake_paths.unitig_rows(SPECIES), index=False)

    result = leakage.check_unitig_rows_built_on_train(fake_paths)
    assert result.passed is None and "role" in result.detail


def test_leakage_detects_duplicate_biosample(fake_paths: Paths) -> None:
    labels = pd.read_parquet(fake_paths.labels)
    labels.loc[labels["genome_id"] == "573.1001", "biosample"] = "SAMN00000000"  # same as 573.1000
    labels.to_parquet(fake_paths.labels, index=False)

    result = leakage.check_biosample_dedup(fake_paths)
    assert result.passed is False and "SAMN00000000" in result.detail


def test_leakage_detects_preds_row_in_wrong_split(fake_paths: Paths) -> None:
    preds = pd.read_parquet(fake_paths.preds(SPECIES, DRUG))
    preds.loc[preds["genome_id"] == "573.1000", "split"] = "test"  # a train genome scored as test
    preds.to_parquet(fake_paths.preds(SPECIES, DRUG), index=False)

    result = leakage.check_preds_splits(fake_paths)
    assert result.passed is False and "573.1000" in result.detail


def test_leakage_test_touched_once_flags_two_run_ids(fake_paths: Paths) -> None:
    preds = pd.read_parquet(fake_paths.preds(SPECIES, DRUG))
    preds.loc[preds.index[:3], "run_id"] = "other_run_id"
    preds.to_parquet(fake_paths.preds(SPECIES, DRUG), index=False)

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is False and "2 run_ids" in result.detail


def test_leakage_failure_shows_in_report(fake_paths: Paths) -> None:
    splits = pd.read_parquet(fake_paths.splits)
    splits.loc[splits["genome_id"] == "573.1000", "split"] = "test"
    splits.to_parquet(fake_paths.splits, index=False)

    output = report.run(fake_paths, None, make_figures=False)
    text = output.report_path.read_text(encoding="utf-8")
    failed = {c["check"] for c in output.checks if c["passed"] is False}
    # The moved genome breaks three checks: its cluster spans splits, its cv preds row is now a
    # test genome, and the unitig rows still mark it as built on train.
    assert failed == {
        "no lineage cluster in two splits or two folds",
        "prediction rows match their split",
        "unitig patterns built on train genomes only",
    }
    n_failed = len(failed)
    assert "**FAIL**" in text
    assert f"{n_failed} failed" in text
    assert output.figures == {} and "Figures skipped" in text
