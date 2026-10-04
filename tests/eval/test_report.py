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
                    "n_band": n - 3,
                    "n_cat": n,
                    "n_lab_r": n // 3,
                    "n_lab_s": n - n // 3,
                    "vme_rate_rederived": vme / 2,
                    "me_rate_rederived": 0.333,
                    "mine_rate_rederived": 0.05,
                    "categorical_agreement_rederived": 0.777,
                    "n_cat_rederived": n - 1,
                    "n_lab_r_rederived": n // 4,
                    "n_lab_s_rederived": n - 1 - n // 4,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(paths.metrics, index=False)
    by_distance = pd.DataFrame(
        [
            {"species": SPECIES, "drug": DRUG, "model": m, "distance_bin": b, "vme_rate": vme, "essential_agreement": ea, "n": n}
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


LEDGER_COLUMNS = ["run_id", "created_utc", "species", "drug", "n_test_rows"]


def _ledger_path(paths: Paths) -> Path:
    return paths.results_dir / "test_ledger.csv"


def _write_ledger(paths: Paths, rows: list[tuple[str, ...]]) -> None:
    """``rows`` = ``[(run_id, species, drug), ...]`` written as the train stage's append-only ledger.

    Three-element rows write the legacy 5-column ledger (before ``inputs_sha1``);
    four-element rows ``(run_id, species, drug, inputs_sha1)`` write the current one.
    """
    with_inputs = any(len(row) == 4 for row in rows)
    columns = [*LEDGER_COLUMNS, "inputs_sha1"] if with_inputs else LEDGER_COLUMNS
    records = []
    for i, (r, s, d, *rest) in enumerate(rows):
        record = {"run_id": r, "created_utc": f"2026-01-0{i + 1}T00:00:00Z", "species": s, "drug": d, "n_test_rows": N_TEST}
        if with_inputs:
            record["inputs_sha1"] = rest[0] if rest else ""
        records.append(record)
    pd.DataFrame(records, columns=columns).to_csv(_ledger_path(paths), index=False)


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
    _write_ledger(paths, [("deadbeef0123", SPECIES, DRUG)])
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
    assert header[2].startswith("VME") and header[3].startswith("Call VME")
    assert header[6].startswith("ME")
    assert "5.0%" in table and "10.0%" in table
    assert "—" in table


def test_ea_exact_and_band_coverage_render_with_their_own_denominators() -> None:
    frame = pd.DataFrame(
        {
            "model": ["aft_known_unitig"],
            "n": [20],
            "vme_rate": [0.0],
            "n_lab_r": [6],
            "essential_agreement": [0.9],
            "exact_agreement": [0.5],
            "n_exact": [10],
            "band_coverage": [0.875],
            "n_band": [8],
        }
    )
    table = report.metrics_table(frame)
    header = _header_cells(table.splitlines()[0])
    row = _header_cells(table.splitlines()[2])
    cells = dict(zip(header, row, strict=True))
    ea_header = next(h for h in header if h.startswith("EA"))
    band_header = next(h for h in header if h.startswith("Band coverage"))
    assert "exact lab MICs" in ea_header and "exact lab MICs" in band_header
    assert cells[ea_header] == "90.0% (n=10)"
    assert cells["Exact agreement % (exact lab MICs)"] == "50.0% (n=10)"
    assert cells[band_header] == "87.5% (n=8)"  # n_band, not the 20 rows of the group


def test_report_explains_that_ea_and_coverage_use_exact_rows(fake_paths: Paths) -> None:
    report.run(fake_paths, None, make_figures=False)
    text = fake_paths.report_md.read_text(encoding="utf-8")
    how_to_read = text[text.index("## How to read this report") : text.index("## Headline")]
    assert report.EXACT_ROWS_TEXT in how_to_read
    headline = text[text.index("## Headline") :].split("\n\n")[1]
    header = _header_cells(headline.splitlines()[0])
    assert "EA % (exact lab MICs)" in header
    # aft_known_unitig test row: EA 0.92 over n_exact = 18.
    assert "92.0% (n=18)" in headline


def test_cv_band_coverage_is_footnoted_as_cross_conformal(fake_paths: Paths) -> None:
    """CV bands are cross-conformal (each fold calibrated on the other folds' residuals)."""
    report.run(fake_paths, None, make_figures=False)
    text = fake_paths.report_md.read_text(encoding="utf-8")
    test_section = text[text.index("#### Test set") : text.index("#### Cross-validation")]
    cv_section = text[text.index("#### Cross-validation") : text.index("#### External and leave-one-lineage-out sets")]
    other_section = text[text.index("#### External and leave-one-lineage-out sets") :]

    assert report.CV_COVERAGE_NOTE in cv_section
    assert report.CV_COVERAGE_NOTE not in test_section and report.CV_COVERAGE_NOTE not in other_section
    cv_header = next(_header_cells(line) for line in cv_section.splitlines() if line.startswith("| Model") and "VME" in line)
    test_header = next(_header_cells(line) for line in test_section.splitlines() if line.startswith("| Model") and "VME" in line)
    cv_band = next(h for h in cv_header if h.startswith("Band coverage"))
    test_band = next(h for h in test_header if h.startswith("Band coverage"))
    assert "cross-conformal" in cv_band and "cross-conformal" not in test_band
    assert "calibrated on the out-of-fold residuals of the other folds only" in report.CV_COVERAGE_NOTE


def test_report_shows_as_reported_then_rederived_categorical_metrics(fake_paths: Paths) -> None:
    report.run(fake_paths, None, make_figures=False)
    text = fake_paths.report_md.read_text(encoding="utf-8")

    assert report.REDERIVED_TEXT in text
    test_section = text[text.index("#### Test set") : text.index("#### Cross-validation")]
    i_reported = test_section.index("| VME % (predicted S, lab R; of lab R) |")
    i_rederived = test_section.index("| VME % re-derived")
    assert i_reported < i_rederived
    # Re-derived rates carry their own denominators: aft_known_unitig 0.006 of n_lab_r_rederived 5.
    assert "0.6% (n=5)" in test_section
    rederived_header = next(
        _header_cells(line) for line in test_section.splitlines() if line.startswith("| Model") and "re-derived" in line
    )
    assert rederived_header[1].startswith("VME"), "VME first in the re-derived table too"

    headline = text[text.index("## Headline") :].split("\n\n")[1]
    header = _header_cells(headline.splitlines()[0])
    first_metric = header[3]
    assert first_metric.startswith("VME") and "as reported" in first_metric
    assert header.index(first_metric) < header.index("VME % re-derived (of lab R)")


def test_rederived_table_absent_for_old_metrics_tables() -> None:
    frame = pd.DataFrame({"model": ["aft_known"], "vme_rate": [0.05], "n": [10]})
    assert report.rederived_metrics_table(frame) is None


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
# Synthetic marking of figures
# --------------------------------------------------------------------------- #


def test_figure_png_carries_synthetic_metadata_only_when_synthetic(fake_paths: Paths, tmp_path: Path) -> None:
    inputs = _loaded(fake_paths)
    marked = figures.vme_me_by_model(inputs.metrics, SPECIES, DRUG, tmp_path / "s.png", synthetic=True)
    plain = figures.vme_me_by_model(inputs.metrics, SPECIES, DRUG, tmp_path / "p.png")
    assert marked is not None and plain is not None
    assert figures.SYNTHETIC_MARK.encode() in marked.read_bytes()
    assert figures.SYNTHETIC_MARK.encode() not in plain.read_bytes()


def test_synthetic_watermark_is_drawn_on_every_figure_type(fake_paths: Paths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _loaded(fake_paths)
    drawn: list[list[str]] = []
    original = figures.plt.Figure.savefig

    def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        drawn.append([t.get_text() for t in self.texts])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(figures.plt.Figure, "savefig", spy)
    calls = [
        lambda: figures.vme_me_by_model(inputs.metrics, SPECIES, DRUG, tmp_path / "1.png", synthetic=True),
        lambda: figures.ea_ca_by_model(inputs.metrics, SPECIES, DRUG, tmp_path / "2.png", synthetic=True),
        lambda: figures.mic_confusion(inputs.preds, SPECIES, DRUG, tmp_path / "3.png", synthetic=True),
        lambda: figures.band_coverage(inputs.metrics, SPECIES, tmp_path / "4.png", synthetic=True),
        lambda: figures.accuracy_vs_distance(inputs.metrics_by_distance, SPECIES, tmp_path / "5.png", synthetic=True),
        lambda: figures.feature_importance(inputs.importances[(SPECIES, DRUG)], SPECIES, DRUG, tmp_path / "6.png", synthetic=True),
        lambda: figures.label_counts(inputs.label_counts, SPECIES, tmp_path / "7.png", synthetic=True),
        lambda: figures.lineage_clusters(inputs.lineages, inputs.splits, SPECIES, tmp_path / "8.png", synthetic=True),
    ]
    for call in calls:
        _assert_png(call())
    assert len(drawn) == len(calls)
    for texts in drawn:
        assert figures.SYNTHETIC_MARK in texts and figures.SYNTHETIC_NOTE in texts


def test_report_figures_are_marked_synthetic_on_a_synthetic_run(fake_paths: Paths) -> None:
    output = report.run(fake_paths, None)
    assert output.figures
    for path in output.figures.values():
        assert figures.SYNTHETIC_MARK.encode() in path.read_bytes(), path.name


def test_report_figures_are_unmarked_on_a_real_run(fake_paths: Paths) -> None:
    (fake_paths.raw_dir / "SYNTHETIC_DATA.md").unlink()
    output = report.run(fake_paths, None)
    assert output.figures
    for path in output.figures.values():
        assert figures.SYNTHETIC_MARK.encode() not in path.read_bytes(), path.name


# --------------------------------------------------------------------------- #
# Unitig index: only the displayed patterns are read
# --------------------------------------------------------------------------- #


def test_load_inputs_reads_only_the_displayed_unitig_patterns(fake_paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_patterns = 300
    table = pa.table(
        {
            "col_index": pa.array(np.arange(n_patterns), pa.int64()),
            "pattern_id": pa.array([f"u_{i:06d}" for i in range(n_patterns)], pa.string()),
            "n_unitigs": pa.array(np.full(n_patterns, 2), pa.int64()),
            "unitig_sequences": pa.array([[f"ACGT{i}", f"TTGG{i}"] for i in range(n_patterns)], pa.list_(pa.string())),
            "train_frequency": pa.array(np.linspace(0.02, 0.98, n_patterns), pa.float64()),
        }
    )
    pq.write_table(table, fake_paths.unitig_index(SPECIES), row_group_size=50)
    # 21 features outrank u_000005, so it falls outside the top 20 and must not be read.
    importance_path = fake_paths.model_dir(SPECIES, DRUG) / "importance.json"
    importance = json.loads(importance_path.read_text())
    importance += [{"feature": f"gene_extra_{i}", "gain": 2.0 + i / 100} for i in range(15)]
    importance.append({"feature": "u_000005", "gain": 0.01})
    importance_path.write_text(json.dumps(importance))

    calls: list[dict[str, object]] = []
    original = report.pq.read_table

    def spy(source, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"source": str(source), **kwargs})
        return original(source, *args, **kwargs)

    monkeypatch.setattr(report.pq, "read_table", spy)
    inputs = report.load_inputs(fake_paths)

    index = inputs.unitig_index[SPECIES]
    assert sorted(index["pattern_id"].astype(str)) == ["u_000000", "u_000002"]
    index_calls = [c for c in calls if c["source"] == str(fake_paths.unitig_index(SPECIES))]
    assert index_calls, "the unitig index must be read with pyarrow column selection"
    assert all(c.get("columns") for c in index_calls)
    label = figures.feature_label("u_000002", index, None)
    assert label == "u_000002 (2 unitigs, train freq 0.03, ACGT2)"


def test_unitig_index_without_displayed_patterns_reads_no_rows(fake_paths: Paths) -> None:
    (fake_paths.model_dir(SPECIES, DRUG) / "importance.json").write_text(json.dumps([{"feature": "gene_blakpc_2", "gain": 1.0}]))
    inputs = report.load_inputs(fake_paths)
    assert SPECIES in inputs.unitig_index
    assert inputs.unitig_index[SPECIES].empty
    assert "pattern_id" in inputs.unitig_index[SPECIES].columns


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


def test_leakage_test_touched_once_passes_with_one_run_id_per_pair(fake_paths: Paths) -> None:
    # Re-running the identical configuration appends the same run_id again: fine.
    _write_ledger(fake_paths, [("deadbeef0123", SPECIES, DRUG), ("deadbeef0123", SPECIES, DRUG), ("aaaa11112222", SPECIES, "ciprofloxacin")])

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is True, result.detail
    assert "test_ledger.csv" in result.detail


def test_leakage_test_touched_once_flags_two_run_ids_for_a_pair(fake_paths: Paths) -> None:
    _write_ledger(fake_paths, [("other_run_id", SPECIES, DRUG), ("deadbeef0123", SPECIES, DRUG), ("aaaa11112222", SPECIES, "ciprofloxacin")])

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is False
    assert f"{SPECIES} x {DRUG}" in result.detail
    assert "deadbeef0123" in result.detail and "other_run_id" in result.detail
    assert "ciprofloxacin" not in result.detail


def test_leakage_test_touched_once_fails_without_a_ledger(fake_paths: Paths) -> None:
    _ledger_path(fake_paths).unlink()

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is False and "no test ledger: cannot verify" in result.detail


def test_leakage_test_touched_once_fails_when_preds_come_from_an_unrecorded_run(fake_paths: Paths) -> None:
    _write_ledger(fake_paths, [("cafe00000000", SPECIES, DRUG)])

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is False
    assert "deadbeef0123" in result.detail and "not in the test ledger" in result.detail


def test_leakage_test_ledger_run_ids_are_read_as_text(fake_paths: Paths) -> None:
    # A 12-hex-digit hash can look like a number ("0012e4567890" parses as a float).
    preds = pd.read_parquet(fake_paths.preds(SPECIES, DRUG))
    preds["run_id"] = "0012e4567890"
    preds.to_parquet(fake_paths.preds(SPECIES, DRUG), index=False)
    _write_ledger(fake_paths, [("0012e4567890", SPECIES, DRUG)])

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is True, result.detail


def test_leakage_test_touched_once_passes_with_one_run_id_and_inputs_per_pair(fake_paths: Paths) -> None:
    # The identical configuration on identical inputs, scored twice: still one scoring configuration.
    _write_ledger(
        fake_paths,
        [("deadbeef0123", SPECIES, DRUG, "1111aaaa2222"), ("deadbeef0123", SPECIES, DRUG, "1111aaaa2222"), ("aaaa11112222", SPECIES, "ciprofloxacin", "3333bbbb4444")],
    )

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is True, result.detail
    assert "inputs_sha1" in result.check


def test_leakage_test_touched_once_flags_changed_inputs_under_the_same_run_id(fake_paths: Paths) -> None:
    # Same TrainConfig + splits (same run_id), but labels / features / configs / model code changed in between.
    _write_ledger(
        fake_paths,
        [("deadbeef0123", SPECIES, DRUG, "1111aaaa2222"), ("deadbeef0123", SPECIES, DRUG, "9999ffff0000"), ("aaaa11112222", SPECIES, "ciprofloxacin", "3333bbbb4444")],
    )

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is False
    assert f"{SPECIES} x {DRUG}" in result.detail
    assert "1111aaaa2222" in result.detail and "9999ffff0000" in result.detail
    assert "ciprofloxacin" not in result.detail


def test_leakage_test_touched_once_treats_a_missing_fingerprint_as_unknown(fake_paths: Paths) -> None:
    # A row written before inputs_sha1 existed cannot show the inputs were unchanged: a pair with
    # such a row and a fingerprinted row of the same run_id fails.
    _write_ledger(fake_paths, [("deadbeef0123", SPECIES, DRUG, ""), ("deadbeef0123", SPECIES, DRUG, "1111aaaa2222")])

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is False
    assert "1111aaaa2222" in result.detail and "no inputs_sha1" in result.detail


def test_leakage_test_touched_once_still_reads_a_legacy_ledger(fake_paths: Paths) -> None:
    _write_ledger(fake_paths, [("deadbeef0123", SPECIES, DRUG), ("deadbeef0123", SPECIES, DRUG)])
    assert "inputs_sha1" not in _ledger_path(fake_paths).read_text().splitlines()[0]

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is True, result.detail


def test_leakage_test_ledger_without_required_columns_fails(fake_paths: Paths) -> None:
    pd.DataFrame({"run_id": ["deadbeef0123"]}).to_csv(_ledger_path(fake_paths), index=False)

    result = leakage.check_test_touched_once(fake_paths)
    assert result.passed is False and "species" in result.detail


def _add_lolo_rows(paths: Paths, preds_lineage: str, genome_ids: list[str], splits_lineage: str | None) -> None:
    """Append ``lolo_<preds_lineage>`` preds rows for ``genome_ids``; mark them in splits as ``splits_lineage``."""
    splits = pd.read_parquet(paths.splits)
    if splits_lineage is not None:
        lolo = splits["lolo_lineage"].astype("str")
        lolo[splits["genome_id"].isin(genome_ids)] = splits_lineage
        splits["lolo_lineage"] = lolo
    splits.to_parquet(paths.splits, index=False)
    preds = pd.read_parquet(paths.preds(SPECIES, DRUG))
    extra = preds[(preds["genome_id"].isin(genome_ids)) & (preds["model"] == MODELS[0])].copy()
    extra["split"] = f"lolo_{preds_lineage}"
    pd.concat([preds, extra], ignore_index=True).to_parquet(paths.preds(SPECIES, DRUG), index=False)


TRAIN_CLUSTER = "KPNEU_ML_001"  # genomes 573.1000 - 573.1003, all train
TRAIN_CLUSTER_GIDS = ["573.1000", "573.1001", "573.1002", "573.1003"]
TEST_CLUSTER = "KPNEU_ML_016"  # genomes 573.1060 - 573.1063, all test
TEST_CLUSTER_GIDS = ["573.1060", "573.1061", "573.1062", "573.1063"]


def test_leakage_lolo_rows_on_a_train_lineage_pass(fake_paths: Paths) -> None:
    _add_lolo_rows(fake_paths, TRAIN_CLUSTER, TRAIN_CLUSTER_GIDS, TRAIN_CLUSTER)

    result = leakage.check_preds_splits(fake_paths)
    assert result.passed is True, result.detail
    assert "lolo" in result.detail


def test_leakage_lolo_rows_on_test_genomes_fail(fake_paths: Paths) -> None:
    _add_lolo_rows(fake_paths, TEST_CLUSTER, TEST_CLUSTER_GIDS, TEST_CLUSTER)

    result = leakage.check_preds_splits(fake_paths)
    assert result.passed is False
    assert "573.1060" in result.detail and "train" in result.detail


def test_leakage_lolo_rows_with_the_wrong_lineage_fail(fake_paths: Paths) -> None:
    _add_lolo_rows(fake_paths, "KPNEU_ML_002", TRAIN_CLUSTER_GIDS, TRAIN_CLUSTER)

    result = leakage.check_preds_splits(fake_paths)
    assert result.passed is False
    assert "lolo_KPNEU_ML_002" in result.detail and "573.1000" in result.detail


def test_leakage_lolo_rows_unmarked_in_splits_fail(fake_paths: Paths) -> None:
    _add_lolo_rows(fake_paths, TRAIN_CLUSTER, TRAIN_CLUSTER_GIDS, None)  # splits.lolo_lineage stays null

    result = leakage.check_preds_splits(fake_paths)
    assert result.passed is False and "573.1000" in result.detail


def test_leakage_lolo_rows_without_lolo_column_fail(fake_paths: Paths) -> None:
    _add_lolo_rows(fake_paths, TRAIN_CLUSTER, TRAIN_CLUSTER_GIDS, TRAIN_CLUSTER)
    pd.read_parquet(fake_paths.splits).drop(columns="lolo_lineage").to_parquet(fake_paths.splits, index=False)

    result = leakage.check_preds_splits(fake_paths)
    assert result.passed is False and "lolo_lineage" in result.detail


def test_leakage_section_explains_what_is_and_is_not_verified(fake_paths: Paths) -> None:
    report.run(fake_paths, None, make_figures=False)
    text = fake_paths.report_md.read_text(encoding="utf-8")
    section = text[text.index("## Leakage checklist") :]
    # Finding 10: CV / LOLO genomes help build the unitig set; only test genomes are queried.
    assert report.UNITIG_BUILD_NOTE in section
    assert "only test genomes are queried" in report.UNITIG_BUILD_NOTE
    # Finding 12: rule 8 is checked against the ledger, with its limits stated.
    assert "test_ledger.csv" in section


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


# --------------------------------------------------------------------------- #
# Demo copy (make demo-copy)
# --------------------------------------------------------------------------- #


def _write_metric_csvs(paths: Paths) -> None:
    pd.read_parquet(paths.metrics).to_csv(paths.metrics.with_suffix(".csv"), index=False)
    pd.read_parquet(paths.metrics_by_distance).to_csv(paths.metrics_by_distance.with_suffix(".csv"), index=False)


def test_demo_copy_labels_every_artefact_synthetic(fake_paths: Paths, tmp_path: Path) -> None:
    from genome2mic.eval import demo

    report.run(fake_paths, None)
    _write_metric_csvs(fake_paths)
    out = tmp_path / "synthetic_demo"
    (out / "figures").mkdir(parents=True)
    (out / "figures" / "stale_from_an_old_run.png").write_bytes(b"old")

    written = demo.copy_demo(fake_paths, out)

    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in readme and "simulated" in readme.lower()
    assert "`synthetic`" in readme  # explains the CSV marker column
    assert DISCLAIMER in readme
    assert (out / "report.md").read_text(encoding="utf-8") == fake_paths.report_md.read_text(encoding="utf-8")
    for name in ("metrics.csv", "metrics_by_distance.csv"):
        copied = pd.read_csv(out / name)
        source = pd.read_csv(fake_paths.results_dir / name)
        assert copied.columns[0] == "synthetic" and copied["synthetic"].astype(bool).all(), name
        pd.testing.assert_frame_equal(copied.drop(columns="synthetic"), source)
        metric_cols = [c for c in copied.columns if c.endswith("_rate") or c.endswith("_agreement")]
        assert metric_cols[0] == "vme_rate", name
    pngs = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert pngs == sorted(p.name for p in fake_paths.figures_dir.glob("*.png"))
    assert "stale_from_an_old_run.png" not in pngs
    for png in (out / "figures").glob("*.png"):
        assert figures.SYNTHETIC_MARK.encode() in png.read_bytes(), png.name
    assert out / "README.md" in written


def test_demo_copy_refuses_a_non_synthetic_root(fake_paths: Paths, tmp_path: Path) -> None:
    from genome2mic.eval import demo

    report.run(fake_paths, None, make_figures=False)
    _write_metric_csvs(fake_paths)
    (fake_paths.raw_dir / "SYNTHETIC_DATA.md").unlink()
    with pytest.raises(ValueError, match="SYNTHETIC_DATA.md"):
        demo.copy_demo(fake_paths, tmp_path / "demo")
    assert not (tmp_path / "demo").exists()


def test_demo_cli_entry_point(fake_paths: Paths, tmp_path: Path) -> None:
    from genome2mic.eval import demo

    report.run(fake_paths, None, make_figures=False)
    _write_metric_csvs(fake_paths)
    out = tmp_path / "demo"
    assert demo.main(["--root", str(fake_paths.root), "--out", str(out)]) == 0
    assert (out / "README.md").is_file() and (out / "metrics.csv").is_file()


def test_makefile_demo_copy_uses_the_labelling_copier() -> None:
    makefile = (Path(__file__).resolve().parents[2] / "Makefile").read_text(encoding="utf-8")
    recipe = makefile[makefile.index("demo-copy:") :].split("\n\n")[0]
    assert "genome2mic.eval.demo" in recipe
