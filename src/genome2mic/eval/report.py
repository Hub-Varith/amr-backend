"""Build ``results/report.md`` and ``results/figures/*.png`` from the pipeline outputs.

Inputs (every one optional -- a missing file becomes a note in the report, never a
crash):

* ``results/preds_*.parquet``            stage 10 predictions (+ optional ``external_set``,
                                         ``nearest_training_distance``)
* ``results/metrics.parquet``            stage 11 metrics, VME first
* ``results/metrics_by_distance.parquet`` EA / VME by nearest-training-distance bin
* ``data/processed/label_counts.csv``, ``pairs_kept.csv``, ``drop_log_*.csv``,
  ``lineages.parquet``, ``splits.parquet``, ``known_amr_columns.csv``,
  ``unitigs_<SPECIES>_index.parquet``, ``labels.parquet`` (fallback for counts)
* ``models/<SPECIES>/<drug>/importance.json``  ``[{feature, gain}, ...]``

Report layout: synthetic-data banner (when ``data/raw/SYNTHETIC_DATA.md`` exists),
the verbatim API disclaimer, how to read the tables, the data counts, per species x
drug metrics tables (test, then CV, then external / LOLO; VME column first), the
figures, and the automated leakage checklist from :mod:`genome2mic.eval.leakage`.

Wording rules: the metric targets are *figures commonly used in AST device
evaluation*, not thresholds met and not regulatory limits; nothing here implies a
dose, a treatment decision or clinical validation.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genome2mic.api.constants import DISCLAIMER
from genome2mic.droplog import DropLog
from genome2mic.eval import figures, leakage
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "report"
SYNTHETIC_MARKER = "SYNTHETIC_DATA.md"

RATE_COLUMNS: tuple[str, ...] = (
    "vme_rate",
    "me_rate",
    "mine_rate",
    "categorical_agreement",
    "essential_agreement",
    "exact_agreement",
    "band_coverage",
)
"""Fraction-valued metric columns rendered as percentages."""

TARGETS_TEXT = (
    "Targets quoted on the figures and below -- EA >= 90 %, CA >= 90 %, VME <= 1.5 %, "
    "ME <= 3 % -- are figures commonly used in AST device evaluation. They are listed "
    "as reference points only; this report does not claim they were met, and they are "
    "not regulatory thresholds."
)
"""Fixed wording about the target figures."""

PRIMARY_SPLITS: tuple[str, ...] = ("test", "cv")

SYNTHETIC_BANNER = (
    "> **SYNTHETIC DATA.** `data/raw/SYNTHETIC_DATA.md` is present: every genome, lab "
    "result, model and number in this report was produced from simulated data. These "
    "figures demonstrate that the pipeline runs end to end. They say nothing about "
    "real-world performance and must not be quoted as such."
)


# --------------------------------------------------------------------------- #
# Loaded inputs
# --------------------------------------------------------------------------- #


@dataclass
class ReportInputs:
    """Every table the report may use, ``None`` when the file was absent or unreadable."""

    metrics: pd.DataFrame | None = None
    metrics_by_distance: pd.DataFrame | None = None
    preds: pd.DataFrame | None = None
    label_counts: pd.DataFrame | None = None
    pairs_kept: pd.DataFrame | None = None
    drop_logs: dict[str, pd.DataFrame] = field(default_factory=dict)
    lineages: pd.DataFrame | None = None
    splits: pd.DataFrame | None = None
    known_columns: pd.DataFrame | None = None
    unitig_index: dict[str, pd.DataFrame] = field(default_factory=dict)
    importances: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    synthetic: bool = False
    found: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def pairs(self) -> list[tuple[str, str]]:
        """Sorted ``(species, drug)`` pairs seen in metrics, preds or importances."""
        pairs: set[tuple[str, str]] = set(self.importances)
        for frame in (self.metrics, self.preds):
            if frame is not None and {"species", "drug"}.issubset(frame.columns):
                sub = frame[["species", "drug"]].dropna().drop_duplicates()
                pairs.update((str(s), str(d)) for s, d in sub.itertuples(index=False))
        return sorted(pairs)

    def species(self) -> list[str]:
        """Sorted species keys seen in any input."""
        keys: set[str] = {s for s, _ in self.pairs()}
        keys.update(self.unitig_index)
        for frame in (self.label_counts, self.lineages, self.splits, self.metrics_by_distance):
            if frame is not None and "species" in frame.columns:
                keys.update(str(s) for s in frame["species"].dropna().unique())
        return sorted(keys)


@dataclass
class ReportOutput:
    """What :func:`run` produced."""

    report_path: Path
    figures: dict[str, Path]
    checks: list[dict[str, object]]
    notes: list[str]


def _load(
    inputs: ReportInputs,
    label: str,
    path: Path,
    reader: Callable[[Path], Any],
) -> Any:
    """Read ``path`` with ``reader``; record a note and return ``None`` on absence or error."""
    if not path.exists():
        inputs.found[label] = False
        inputs.notes.append(f"`{label}` not found at `{path}`; related sections omitted.")
        logger.warning("%s missing: %s", label, path)
        return None
    try:
        value = reader(path)
    except Exception as error:  # noqa: BLE001 - every input is optional
        inputs.found[label] = False
        inputs.notes.append(f"`{label}` at `{path}` could not be read ({error}); related sections omitted.")
        logger.warning("%s unreadable: %s (%s)", label, path, error)
        return None
    inputs.found[label] = True
    return value


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_importance(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    if isinstance(content, dict):
        content = content.get("importance", content.get("features", []))
    return [dict(item) for item in content if isinstance(item, dict)]


def _counts_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Fallback count table (``label_counts.csv`` shape) computed from ``labels.parquet``."""
    needed = {"species", "drug", "mic_lower", "mic_upper"}
    if not needed.issubset(labels.columns):
        raise ValueError(f"labels.parquet lacks {sorted(needed - set(labels.columns))}")
    sir = labels["sir"].astype("str") if "sir" in labels.columns else pd.Series([None] * len(labels), index=labels.index, dtype="str")
    lower = pd.to_numeric(labels["mic_lower"], errors="coerce")
    upper = pd.to_numeric(labels["mic_upper"], errors="coerce")
    exact = (lower > 0) & np.isfinite(upper)
    work = labels.assign(
        _r=(sir == "R").astype(int),
        _s=(sir == "S").astype(int),
        _i=(sir == "I").astype(int),
        _exact=exact.astype(int),
        _mic=np.where(exact, upper, np.nan),
    )
    grouped = work.groupby(["species", "drug"], sort=True)
    counts = grouped.agg(
        n=("genome_id", "size") if "genome_id" in work.columns else ("_r", "size"),
        n_R=("_r", "sum"),
        n_S=("_s", "sum"),
        n_I=("_i", "sum"),
        n_exact=("_exact", "sum"),
        n_distinct_mic=("_mic", "nunique"),
    ).reset_index()
    return counts.assign(n_censored=counts["n"] - counts["n_exact"])


def load_inputs(paths: Paths) -> ReportInputs:
    """Load every optional input under ``paths`` (see the module docstring)."""
    inputs = ReportInputs()
    inputs.synthetic = (paths.raw_dir / SYNTHETIC_MARKER).is_file()
    inputs.metrics = _load(inputs, "metrics.parquet", paths.metrics, _read_parquet)
    inputs.metrics_by_distance = _load(inputs, "metrics_by_distance.parquet", paths.metrics_by_distance, _read_parquet)

    preds_files = sorted(paths.results_dir.glob("preds_*.parquet")) if paths.results_dir.is_dir() else []
    frames: list[pd.DataFrame] = []
    for file in preds_files:
        frame = _load(inputs, file.name, file, _read_parquet)
        if frame is not None:
            frames.append(frame)
    if frames:
        inputs.preds = pd.concat(frames, ignore_index=True)
        inputs.found["preds_*.parquet"] = True
    else:
        inputs.found["preds_*.parquet"] = False
        inputs.notes.append("No `results/preds_*.parquet` found; MIC confusion figures omitted.")

    inputs.label_counts = _load(inputs, "label_counts.csv", paths.label_counts, _read_csv)
    if inputs.label_counts is None and paths.labels.is_file():
        try:
            inputs.label_counts = _counts_from_labels(_read_parquet(paths.labels))
            inputs.notes.append("`label_counts.csv` missing; counts recomputed from `labels.parquet`.")
        except Exception as error:  # noqa: BLE001
            inputs.notes.append(f"`labels.parquet` could not be summarised ({error}).")
    inputs.pairs_kept = _load(inputs, "pairs_kept.csv", paths.pairs_kept, _read_csv)
    if paths.processed_dir.is_dir():
        for file in sorted(paths.processed_dir.glob("drop_log_*.csv")):
            stage = file.stem.removeprefix("drop_log_")
            if stage == STAGE:
                continue  # this stage's own log from a previous run
            frame = _load(inputs, file.name, file, _read_csv)
            if frame is not None:
                inputs.drop_logs[stage] = frame
    if not inputs.drop_logs:
        inputs.notes.append("No `data/processed/drop_log_*.csv` found; drop-log section omitted.")
    inputs.lineages = _load(inputs, "lineages.parquet", paths.lineages, _read_parquet)
    inputs.splits = _load(inputs, "splits.parquet", paths.splits, _read_parquet)
    inputs.known_columns = _load(inputs, "known_amr_columns.csv", paths.known_amr_columns, _read_csv)
    if paths.processed_dir.is_dir():
        for file in sorted(paths.processed_dir.glob("unitigs_*_index.parquet")):
            species = file.stem.removeprefix("unitigs_").removesuffix("_index")
            frame = _load(inputs, file.name, file, _read_parquet)
            if frame is not None:
                inputs.unitig_index[species] = frame
    if paths.models_dir.is_dir():
        for file in sorted(paths.models_dir.glob("*/*/importance.json")):
            species, drug = file.parent.parent.name, file.parent.name
            content = _load(inputs, f"models/{species}/{drug}/importance.json", file, _read_importance)
            if content is not None:
                inputs.importances[(species, drug)] = content
    if not inputs.importances:
        inputs.notes.append("No `models/<SPECIES>/<drug>/importance.json` found; feature-importance figures omitted.")
    return inputs


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def build_figures(paths: Paths, inputs: ReportInputs, droplog: DropLog | None = None) -> tuple[dict[str, Path], list[str]]:
    """Write every figure the inputs allow. Returns ``(figures by file stem, notes)``."""
    out: dict[str, Path] = {}
    notes: list[str] = []
    figures_dir = paths.figures_dir

    def attempt(stem: str, func: Callable[[], Path | None]) -> None:
        try:
            result = func()
        except Exception as error:  # noqa: BLE001 - a figure must never sink the report
            logger.warning("figure %s failed: %s", stem, error, exc_info=True)
            notes.append(f"Figure `{stem}.png` not drawn: {error}")
            return
        if result is None:
            notes.append(f"Figure `{stem}.png` not drawn: not enough input rows.")
        else:
            out[stem] = result

    for species, drug in inputs.pairs():
        stem = f"vme_me_by_model_{species}_{drug}"
        attempt(stem, lambda s=stem, sp=species, d=drug: figures.vme_me_by_model(inputs.metrics, sp, d, figures_dir / f"{s}.png"))
        stem = f"ea_ca_by_model_{species}_{drug}"
        attempt(stem, lambda s=stem, sp=species, d=drug: figures.ea_ca_by_model(inputs.metrics, sp, d, figures_dir / f"{s}.png"))
        stem = f"mic_confusion_{species}_{drug}"
        attempt(stem, lambda s=stem, sp=species, d=drug: figures.mic_confusion(inputs.preds, sp, d, figures_dir / f"{s}.png", droplog=droplog))
        stem = f"feature_importance_{species}_{drug}"
        attempt(
            stem,
            lambda s=stem, sp=species, d=drug: figures.feature_importance(
                inputs.importances.get((sp, d)),
                sp,
                d,
                figures_dir / f"{s}.png",
                unitig_index=inputs.unitig_index.get(sp),
                known_columns=inputs.known_columns,
            ),
        )
    for species in inputs.species():
        stem = f"band_coverage_{species}"
        attempt(stem, lambda s=stem, sp=species: figures.band_coverage(inputs.metrics, sp, figures_dir / f"{s}.png"))
        stem = f"accuracy_vs_distance_{species}"
        attempt(stem, lambda s=stem, sp=species: figures.accuracy_vs_distance(inputs.metrics_by_distance, sp, figures_dir / f"{s}.png"))
        stem = f"label_counts_{species}"
        attempt(stem, lambda s=stem, sp=species: figures.label_counts(inputs.label_counts, sp, figures_dir / f"{s}.png"))
        stem = f"lineage_clusters_{species}"
        attempt(stem, lambda s=stem, sp=species: figures.lineage_clusters(inputs.lineages, inputs.splits, sp, figures_dir / f"{s}.png"))
    return out, notes


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #


def _isna(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell(value: Any) -> str:
    if _isna(value):
        return "—"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(_cell(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def frame_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    """A DataFrame as a Markdown table (truncated with a note after ``max_rows``)."""
    shown = frame.head(max_rows)
    rows = [[_fmt_value(v) for v in row] for row in shown.itertuples(index=False)]
    text = md_table([str(c) for c in frame.columns], rows)
    if len(frame) > max_rows:
        text += f"\n\n_{len(frame) - max_rows} more rows not shown._"
    return text


def _fmt_value(value: Any) -> str:
    if _isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        if float(value) == int(value) and abs(float(value)) < 1e12:
            return str(int(value))
        return f"{float(value):.4g}"
    return str(value)


def fmt_pct(rate: Any, n: Any = None) -> str:
    """``0.0123, 83`` -> ``1.2% (n=83)``; null rate -> ``—``."""
    if _isna(rate):
        return "—"
    text = f"{float(rate) * 100:.1f}%"
    if n is not None and not _isna(n):
        text += f" (n={int(n)})"
    return text


def _fmt_float(value: Any, digits: int = 2) -> str:
    return "—" if _isna(value) else f"{float(value):.{digits}f}"


def _get(row: Mapping[str, Any], key: str) -> Any:
    return row[key] if key in row else None


METRICS_HEADERS: tuple[str, ...] = (
    "Model",
    "n",
    "VME % (predicted S, lab R; of lab R)",
    "ME % (predicted R, lab S; of lab S)",
    "Minor error % (of categorised)",
    "CA % (same S/I/R)",
    "EA % (within +/-1 step; exact MICs)",
    "Exact agreement %",
    "AUROC (R vs S)",
    "Band coverage % (90% band)",
    "Band width (doubling steps)",
)
"""Column order of every metrics table: VME first after the key columns."""


def metrics_table(metrics: pd.DataFrame) -> str:
    """One Markdown table for the rows of one species x drug x split, VME first, rates as % with n."""
    frame = metrics.copy()
    if "model" in frame.columns:
        order = figures.order_models(frame["model"])
        rank = {m: i for i, m in enumerate(order)}
        frame = frame.assign(_rank=frame["model"].astype("str").map(rank)).sort_values("_rank").drop(columns="_rank")
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            [
                _get(record, "model"),
                _fmt_value(_get(record, "n")),
                fmt_pct(_get(record, "vme_rate"), _get(record, "n_lab_r")),
                fmt_pct(_get(record, "me_rate"), _get(record, "n_lab_s")),
                fmt_pct(_get(record, "mine_rate"), _get(record, "n_cat")),
                fmt_pct(_get(record, "categorical_agreement"), _get(record, "n_cat")),
                fmt_pct(_get(record, "essential_agreement"), _get(record, "n_exact")),
                fmt_pct(_get(record, "exact_agreement"), _get(record, "n_exact")),
                _fmt_float(_get(record, "auroc"), 3),
                fmt_pct(_get(record, "band_coverage"), _get(record, "n")),
                _fmt_float(_get(record, "band_width_steps"), 2),
            ]
        )
    return md_table(METRICS_HEADERS, rows)


def _image(report_path: Path, figure: Path, caption: str) -> str:
    rel = os.path.relpath(figure, report_path.parent)
    return f"![{caption}]({rel.replace(os.sep, '/')})"


def _call_breakpoint_line(config: Any, species: str, drug: str) -> str | None:
    if config is None:
        return None
    try:
        bp = config.call_breakpoint(species, drug)
    except Exception:  # noqa: BLE001 - config shape is another module's
        return None
    if bp is None:
        return f"_No call breakpoint for {species} x {drug} under the configured call standard._"
    return (
        f"Call breakpoint ({bp.standard} {bp.version}, bloodstream): S if MIC <= {figures.fmt_mic(bp.s_breakpoint)} mg/L, "
        f"R if MIC > {figures.fmt_mic(bp.r_breakpoint)} mg/L."
    )


# --------------------------------------------------------------------------- #
# Markdown document
# --------------------------------------------------------------------------- #


def render_markdown(
    paths: Paths,
    inputs: ReportInputs,
    figure_paths: Mapping[str, Path],
    checks: Sequence[Mapping[str, object]],
    notes: Sequence[str],
    config: Any = None,
    droplog: DropLog | None = None,
) -> str:
    """Assemble the full report text."""
    report_path = paths.report_md
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = ["# genome2mic results report", ""]
    if inputs.synthetic:
        parts += [SYNTHETIC_BANNER, ""]
    parts += [f"> **Scope note.** {DISCLAIMER}", ""]
    parts += [
        f"Generated {now} from `{paths.root.resolve()}`. Predictions of in-vitro MIC (mg/L) per "
        "species x drug; the system ranks, a clinician decides.",
        "",
        "## How to read this report",
        "",
        "- **VME** (very major error: predicted S, lab R) is listed first in every table and figure; "
        "it is the error that would harm a patient. Its denominator is the number of lab-R rows.",
        "- **ME** (major error: predicted R, lab S) uses the lab-S rows as denominator; "
        "**minor error** has exactly one side I; **CA** is same S/I/R; **EA** is within +/-1 doubling step "
        "of an exact lab MIC; **exact agreement** is the same doubling step.",
        "- Rates are shown as percentages with their denominator `n`. Band coverage is the share of lab MICs "
        "inside the 90 % conformal band; band width is in doubling steps.",
        f"- {TARGETS_TEXT}",
        "- `test set` = lineage-held-out genomes scored once at the end; `CV` = out-of-fold predictions on "
        "the train split; external and leave-one-lineage-out (LOLO) sets are listed separately.",
        "",
    ]
    if inputs.synthetic:
        parts += ["_All numbers below come from synthetic data (see the banner above)._", ""]

    parts += _headline_section(inputs)
    parts += _data_section(inputs)
    parts += _results_sections(inputs, figure_paths, report_path, config)
    parts += _species_figures_section(inputs, figure_paths, report_path)
    parts += _leakage_section(checks)
    parts += _notes_section(inputs, notes, droplog)
    return "\n".join(parts).rstrip() + "\n"


def _headline_section(inputs: ReportInputs) -> list[str]:
    metrics = inputs.metrics
    if metrics is None or not {"species", "drug", "model", "split"}.issubset(metrics.columns):
        return []
    test = figures.subset(metrics, split="test")
    if test.empty:
        return []
    rows = []
    for species, drug in sorted({(str(s), str(d)) for s, d in test[["species", "drug"]].drop_duplicates().itertuples(index=False)}):
        pair = figures.subset(test, species=species, drug=drug)
        model = figures.pick_model(pair["model"])
        if model is None:
            continue
        record = figures.subset(pair, model=model).iloc[0].to_dict()
        rows.append(
            [
                species,
                drug,
                model,
                fmt_pct(_get(record, "vme_rate"), _get(record, "n_lab_r")),
                fmt_pct(_get(record, "me_rate"), _get(record, "n_lab_s")),
                fmt_pct(_get(record, "categorical_agreement"), _get(record, "n_cat")),
                fmt_pct(_get(record, "essential_agreement"), _get(record, "n_exact")),
                _fmt_value(_get(record, "n")),
            ]
        )
    if not rows:
        return []
    return [
        "## Headline: main model on the test set (VME first)",
        "",
        md_table(
            ["Species", "Drug", "Model", "VME % (of lab R)", "ME % (of lab S)", "CA %", "EA % (exact MICs)", "n"],
            rows,
        ),
        "",
    ]


def _data_section(inputs: ReportInputs) -> list[str]:
    parts = ["## Data", ""]
    parts += ["### Label counts (species x drug, all lab-measured labels)", ""]
    if inputs.label_counts is not None and not inputs.label_counts.empty:
        parts += [frame_table(inputs.label_counts), ""]
    else:
        parts += ["_`label_counts.csv` not available._", ""]
    parts += ["### Pairs kept (>= 50 non-susceptible, >= 50 susceptible, >= 4 distinct MIC levels)", ""]
    if inputs.pairs_kept is not None and not inputs.pairs_kept.empty:
        parts += [frame_table(inputs.pairs_kept), ""]
    else:
        parts += ["_`pairs_kept.csv` not available._", ""]
    parts += ["### Drop logs (every filter, with its count)", ""]
    if inputs.drop_logs:
        for stage, frame in inputs.drop_logs.items():
            total = int(pd.to_numeric(frame["n_dropped"], errors="coerce").fillna(0).sum()) if "n_dropped" in frame.columns else None
            heading = f"#### Stage `{stage}`" + (f" — {total} rows dropped" if total is not None else "")
            parts += [heading, "", frame_table(frame), ""]
    else:
        parts += ["_No drop logs available._", ""]
    return parts


def _results_sections(
    inputs: ReportInputs,
    figure_paths: Mapping[str, Path],
    report_path: Path,
    config: Any,
) -> list[str]:
    parts = ["## Results by species and drug", ""]
    pairs = inputs.pairs()
    if not pairs:
        parts += ["_No `metrics.parquet`, `preds_*.parquet` or importance files found; nothing to report._", ""]
        return parts
    metrics = inputs.metrics
    for species, drug in pairs:
        parts += [f"### {species} x {drug}", ""]
        bp_line = _call_breakpoint_line(config, species, drug)
        if bp_line:
            parts += [bp_line, ""]
        if metrics is None or not {"species", "drug", "split"}.issubset(metrics.columns):
            parts += ["_`metrics.parquet` not available; metrics tables omitted._", ""]
            parts += _preds_summary(inputs, species, drug)
        else:
            pair = figures.subset(metrics, species=species, drug=drug)
            for split, title in (("test", "Test set (lineage-held-out genomes, scored once)"), ("cv", "Cross-validation (out-of-fold, train split)")):
                parts += [f"#### {title}", ""]
                rows = figures.subset(pair, split=split)
                parts += [metrics_table(rows) if not rows.empty else f"_No `{split}` rows in metrics.parquet for this pair._", ""]
            others = [s for s in pair["split"].astype("str").dropna().unique() if s not in PRIMARY_SPLITS]
            if others:
                parts += ["#### External and leave-one-lineage-out sets", ""]
                for split in sorted(others):
                    parts += [f"**{figures.split_label(split)}** (`{split}`)", "", metrics_table(figures.subset(pair, split=split)), ""]
        for stem, caption in (
            (f"vme_me_by_model_{species}_{drug}", "VME and ME by model, test set"),
            (f"ea_ca_by_model_{species}_{drug}", "EA and CA by model, test set"),
            (f"mic_confusion_{species}_{drug}", "MIC confusion heatmap, test set, exact lab MICs"),
            (f"feature_importance_{species}_{drug}", "Top-20 gain importances, final fit on train"),
        ):
            if stem in figure_paths:
                parts += [_image(report_path, figure_paths[stem], caption), ""]
            else:
                parts += [f"_Figure `{stem}.png` not available._", ""]
    return parts


def _preds_summary(inputs: ReportInputs, species: str, drug: str) -> list[str]:
    preds = figures.subset(inputs.preds, species=species, drug=drug)
    if preds.empty or not {"model", "split"}.issubset(preds.columns):
        return []
    counts = preds.groupby(["model", "split"], sort=True).size().reset_index(name="rows")
    return ["Prediction rows available (metrics not computed here):", "", frame_table(counts), ""]


def _species_figures_section(inputs: ReportInputs, figure_paths: Mapping[str, Path], report_path: Path) -> list[str]:
    species_keys = inputs.species()
    if not species_keys:
        return []
    parts = ["## Species-level figures", ""]
    for species in species_keys:
        parts += [f"### {species}", ""]
        for stem, caption in (
            (f"band_coverage_{species}", "Conformal band coverage and width per drug, test set"),
            (f"accuracy_vs_distance_{species}", "EA and VME by distance to the nearest training genome, test set"),
            (f"label_counts_{species}", "Lab label counts per drug"),
            (f"lineage_clusters_{species}", "Lineage cluster sizes, train vs test"),
        ):
            if stem in figure_paths:
                parts += [_image(report_path, figure_paths[stem], caption), ""]
            else:
                parts += [f"_Figure `{stem}.png` not available._", ""]
    return parts


def _leakage_section(checks: Sequence[Mapping[str, object]]) -> list[str]:
    status = {True: "PASS", False: "**FAIL**", None: "NOT RUN"}
    rows = [[c.get("check"), status.get(c.get("passed"), "NOT RUN"), c.get("detail")] for c in checks]  # type: ignore[arg-type]
    n_fail = sum(1 for c in checks if c.get("passed") is False)
    n_skip = sum(1 for c in checks if c.get("passed") is None)
    summary = f"{len(checks) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} not run."
    return [
        "## Leakage checklist (automated)",
        "",
        "Checks from `DATA_CONTRACT.md` section 4 that can be verified from the files on disk. "
        "`NOT RUN` means the input needed for that check is missing. Fold-internal feature "
        "selection and calibration-on-validation-only are enforced in the training code and are "
        "not re-verifiable from outputs.",
        "",
        summary,
        "",
        md_table(["Check", "Result", "Detail"], rows),
        "",
    ]


def _notes_section(inputs: ReportInputs, notes: Sequence[str], droplog: DropLog | None) -> list[str]:
    parts = ["## Inputs and notes", ""]
    rows = [[label, "found" if ok else "missing"] for label, ok in sorted(inputs.found.items())]
    if rows:
        parts += [md_table(["Input", "Status"], rows), ""]
    all_notes = [*inputs.notes, *notes]
    if all_notes:
        parts += ["\n".join(f"- {n}" for n in all_notes), ""]
    if droplog is not None and len(droplog):
        parts += ["### Rows excluded while drawing figures", "", frame_table(droplog.to_frame()), ""]
    parts += [f"> {DISCLAIMER}", ""]
    return parts


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(paths: Paths, config: Any = None, *, make_figures: bool = True) -> ReportOutput:
    """Write ``results/report.md`` (and figures) from whatever outputs exist under ``paths``.

    Args:
        paths: Project paths; every input is optional.
        config: Loaded :class:`genome2mic.config.Config` for breakpoint lines in the
            report. ``None`` is accepted (lines are omitted); loading is attempted from
            ``paths.configs_dir`` when ``None`` and the directory exists.
        make_figures: Set ``False`` to skip the PNGs (text report only).

    Returns:
        :class:`ReportOutput` with the report path, figure paths, checklist and notes.
    """
    if config is None and paths.configs_dir.is_dir():
        try:
            from genome2mic.config import load_config

            config = load_config(paths.configs_dir)
        except Exception as error:  # noqa: BLE001 - the report must not depend on configs
            logger.warning("configs at %s not loadable (%s); breakpoint lines omitted", paths.configs_dir, error)
    droplog = DropLog(STAGE)
    inputs = load_inputs(paths)
    figure_paths: dict[str, Path] = {}
    notes: list[str] = []
    if make_figures:
        figure_paths, notes = build_figures(paths, inputs, droplog)
    else:
        notes.append("Figures skipped (`make_figures=False`).")
    checks = leakage.run_checks(paths)
    text = render_markdown(paths, inputs, figure_paths, checks, notes, config, droplog)
    report_path = paths.report_md
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    droplog.write(paths.drop_log(STAGE))
    logger.info(
        "wrote %s (%d figures, %d leakage checks, %d notes)%s",
        report_path,
        len(figure_paths),
        len(checks),
        len(notes) + len(inputs.notes),
        " [SYNTHETIC DATA]" if inputs.synthetic else "",
    )
    return ReportOutput(report_path=report_path, figures=figure_paths, checks=checks, notes=[*inputs.notes, *notes])


__all__ = [
    "METRICS_HEADERS",
    "RATE_COLUMNS",
    "ReportInputs",
    "ReportOutput",
    "build_figures",
    "fmt_pct",
    "frame_table",
    "load_inputs",
    "md_table",
    "metrics_table",
    "render_markdown",
    "run",
]
