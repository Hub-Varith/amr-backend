"""Matplotlib figures for ``results/report.md``.

Eight figure types (``.context/DESIGN.md`` section *eval*), each a pure function
that takes already-loaded tables and an output path and returns the written path,
or ``None`` when the inputs do not contain enough rows to draw anything. Nothing in
this module reads from disk, so the report layer owns every "file missing" decision.

Style rules shared by every figure:

* Agg backend (no display needed), 150 dpi PNG.
* Okabe-Ito colorblind-safe palette (:data:`PALETTE`).
* Every title states which evaluation set the figure shows (``test set``, ``CV``,
  ``train split`` ...); every axis label carries its unit.
* Target lines are labelled in the legend and described as figures commonly used
  in AST device evaluation -- never as thresholds that were met.
* Rate columns from ``metrics.parquet`` are fractions in ``[0, 1]`` per
  ``DATA_CONTRACT.md`` stage 11 and are drawn as percentages.
* VME is always drawn first / left-most (CLAUDE.md rule 10).
* EA and band coverage are annotated with their own denominators (``n_exact``,
  ``n_band``: exact lab MICs only, see :mod:`genome2mic.eval.metrics`).
* ``synthetic=True`` (every figure function) stamps a diagonal ``SYNTHETIC DATA``
  watermark and a note line on the figure and writes the same note into the PNG
  ``Description`` metadata, so a PNG copied out of the report still says what it is.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from genome2mic.droplog import DropLog  # noqa: E402
from genome2mic.eval.metrics import exact_lab_mask  # noqa: E402

logger = logging.getLogger(__name__)

DPI = 150
"""Output resolution of every PNG."""

PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
"""Okabe-Ito colorblind-safe palette."""

GREY = "#7F7F7F"

TARGET_VME_PCT = 1.5
TARGET_ME_PCT = 3.0
TARGET_EA_PCT = 90.0
TARGET_CA_PCT = 90.0
TARGET_COVERAGE_PCT = 90.0
TARGET_NOTE = "figure commonly used in AST device evaluation"
"""Wording appended to every target-line label."""

MAIN_MODEL = "aft_known_unitig"
MODEL_ORDER: tuple[str, ...] = (
    "b0_resfinder",
    "b1_lookup",
    "b2_xgb_steps",
    "aft_known",
    "aft_unitig_only",
    "aft_known_unitig",
    "multitask_nn",
)
"""Display order for models; unknown models are appended alphabetically."""

TOP_K_FEATURES = 20
"""Number of gain importances drawn by :func:`feature_importance` (and labelled by the report)."""

SYNTHETIC_MARK = "SYNTHETIC DATA"
"""Watermark text stamped on every figure of a synthetic run."""

SYNTHETIC_NOTE = (
    "SYNTHETIC DATA: simulated genomes and lab results. Shows that the pipeline runs; "
    "says nothing about real-world performance."
)
"""Note line and PNG ``Description`` metadata of every figure of a synthetic run."""

CLUSTER_SIZE_BINS: tuple[tuple[int, float, str], ...] = (
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 50, "21-50"),
    (51, math.inf, "51+"),
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def split_label(split: str) -> str:
    """Human wording for a ``split`` value used in titles (``test`` -> ``test set``)."""
    key = str(split)
    if key == "test":
        return "test set"
    if key == "cv":
        return "CV (out-of-fold)"
    if key.startswith("lolo_"):
        return f"leave-one-lineage-out ({key[5:]})"
    return key.replace("_", " ")


def order_models(models: Iterable[Any]) -> list[str]:
    """Stable display order: :data:`MODEL_ORDER` first, then unknown names alphabetically."""
    seen = list(dict.fromkeys(str(m) for m in models if not _isna(m)))
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(seen, key=lambda m: (rank.get(m, len(MODEL_ORDER)), m))


def model_color(model: str) -> str:
    """Palette colour for a model name (grey for models outside :data:`MODEL_ORDER`)."""
    if model in MODEL_ORDER:
        return PALETTE[MODEL_ORDER.index(model) % len(PALETTE)]
    return GREY


def fmt_mic(mic: float) -> str:
    """Compact MIC tick label in mg/L (``8``, ``0.25``, ``0.0625``)."""
    return f"{float(mic):g}"


def subset(frame: pd.DataFrame | None, **filters: Any) -> pd.DataFrame:
    """Rows of ``frame`` where every ``column == value`` filter holds.

    ``None`` filter values are ignored. A missing filter column yields an empty
    frame (and a warning) rather than an exception, so callers can treat "no rows"
    uniformly.
    """
    if frame is None:
        return pd.DataFrame()
    mask = np.ones(len(frame), dtype=bool)
    for column, value in filters.items():
        if value is None:
            continue
        if column not in frame.columns:
            logger.warning("column %r missing from %d-row frame; cannot filter on it", column, len(frame))
            return frame.iloc[0:0]
        eq = frame[column].astype("str") == str(value)
        mask &= eq.fillna(False).to_numpy(dtype=bool)
    return frame.loc[mask]


def _isna(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _float_col(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    """Column as float64 array; a missing column yields all-``default``."""
    if column not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _pct(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def _fmt_n(value: float) -> str:
    return "n/a" if _isna(value) else f"{int(value)}"


def _denominator(frame: pd.DataFrame, column: str, fallback: str = "n") -> np.ndarray:
    """Denominator column as floats; ``fallback`` when ``column`` is absent (older metrics tables)."""
    return _float_col(frame, column if column in frame.columns else fallback)


def mark_synthetic(fig: plt.Figure) -> None:
    """Stamp a diagonal :data:`SYNTHETIC_MARK` watermark and a :data:`SYNTHETIC_NOTE` line on ``fig``.

    The note sits just above the figure's top edge; ``bbox_inches='tight'`` keeps it
    in the saved PNG.
    """
    fig.text(
        0.5,
        0.5,
        SYNTHETIC_MARK,
        ha="center",
        va="center",
        rotation=25,
        fontsize=40,
        fontweight="bold",
        color=PALETTE[3],
        alpha=0.18,
        zorder=100,
    )
    fig.text(0.5, 1.0, SYNTHETIC_NOTE, ha="center", va="bottom", fontsize=7, fontweight="bold", color=PALETTE[3])


def _save(fig: plt.Figure, out_path: Path, synthetic: bool = False) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, str] | None = None
    if synthetic:
        mark_synthetic(fig)
        metadata = {"Description": SYNTHETIC_NOTE}
    fig.savefig(target, dpi=DPI, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    logger.info("wrote figure %s%s", target, " [SYNTHETIC DATA]" if synthetic else "")
    return target


def _annotate_bars(ax: plt.Axes, bars: Any, labels: Sequence[str], heights: np.ndarray) -> None:
    """Write ``labels`` just above each bar (``n/a`` text where the height is null)."""
    for bar, label, height in zip(bars, labels, heights, strict=True):
        x = bar.get_x() + bar.get_width() / 2
        if _isna(height):
            ax.annotate("n/a", (x, 0), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=GREY)
            continue
        ax.annotate(label, (x, height), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)


def _grouped_bars(
    ax: plt.Axes,
    categories: Sequence[str],
    series: Mapping[str, np.ndarray],
    colors: Mapping[str, str],
    n_labels: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Grouped bar chart: one group per category, one bar per series key."""
    n_series = max(len(series), 1)
    width = 0.8 / n_series
    x = np.arange(len(categories))
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        heights = np.asarray(values, dtype=float)
        bars = ax.bar(x + offset, np.nan_to_num(heights, nan=0.0), width, color=colors[name], label=name)
        labels = list(n_labels[name]) if n_labels and name in n_labels else [""] * len(categories)
        _annotate_bars(ax, bars, labels, heights)
    ax.set_xticks(x, [str(c) for c in categories], rotation=20, ha="right")


def _ylim_pct(ax: plt.Axes, values: Iterable[float], floor: float) -> None:
    finite = [v for v in values if not _isna(v)]
    top = max([floor, *finite]) if finite else floor
    ax.set_ylim(0, top * 1.25 + 1)


def _legend_below(ax: plt.Axes, ncol: int = 2, pad: float = 0.32) -> None:
    """Legend under the x axis so it never covers bars or annotations (kept by ``bbox_inches='tight'``)."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -pad), ncol=ncol, fontsize=7, frameon=False)


# --------------------------------------------------------------------------- #
# 1. VME / ME by model
# --------------------------------------------------------------------------- #


def vme_me_by_model(
    metrics: pd.DataFrame | None,
    species: str,
    drug: str,
    out_path: Path,
    split: str = "test",
    *,
    synthetic: bool = False,
) -> Path | None:
    """VME and ME rates (%) per model with the 1.5 % / 3 % target lines.

    VME (predicted S, lab R) is drawn first because it is the error that harms a
    patient. Bars are annotated with the denominator (``n_lab_r`` / ``n_lab_s``).
    """
    sub = subset(metrics, species=species, drug=drug, split=split)
    if sub.empty or "model" not in sub.columns:
        logger.warning("vme_me_by_model: no %s rows for %s %s", split, species, drug)
        return None
    models = order_models(sub["model"])
    sub = sub.drop_duplicates("model")
    sub = sub.set_index(sub["model"].astype("str")).loc[models]
    vme = _pct(_float_col(sub, "vme_rate"))
    me = _pct(_float_col(sub, "me_rate"))
    n_r = _float_col(sub, "n_lab_r")
    n_s = _float_col(sub, "n_lab_s")

    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(models) + 2.5), 4.2))
    _grouped_bars(
        ax,
        models,
        {"VME (predicted S, lab R)": vme, "ME (predicted R, lab S)": me},
        {"VME (predicted S, lab R)": PALETTE[3], "ME (predicted R, lab S)": PALETTE[0]},
        {
            "VME (predicted S, lab R)": [f"n R={_fmt_n(v)}" for v in n_r],
            "ME (predicted R, lab S)": [f"n S={_fmt_n(v)}" for v in n_s],
        },
    )
    ax.axhline(TARGET_VME_PCT, color=PALETTE[3], ls="--", lw=1, label=f"VME target {TARGET_VME_PCT:g}% ({TARGET_NOTE})")
    ax.axhline(TARGET_ME_PCT, color=PALETTE[0], ls=":", lw=1, label=f"ME target {TARGET_ME_PCT:g}% ({TARGET_NOTE})")
    _ylim_pct(ax, [*vme, *me], TARGET_ME_PCT)
    ax.set_xlabel("Model")
    ax.set_ylabel("Error rate (%; VME of lab-R rows, ME of lab-S rows)")
    ax.set_title(f"VME and ME by model, {split_label(split)}: {species} {drug}")
    _legend_below(ax)
    return _save(fig, out_path, synthetic)


# --------------------------------------------------------------------------- #
# 2. EA / CA by model
# --------------------------------------------------------------------------- #


def ea_ca_by_model(
    metrics: pd.DataFrame | None,
    species: str,
    drug: str,
    out_path: Path,
    split: str = "test",
    *,
    synthetic: bool = False,
) -> Path | None:
    """Essential and categorical agreement (%) per model with the 90 % target line.

    EA bars are annotated with ``n_exact`` (rows with an exact lab MIC: EA's
    denominator), CA bars with ``n_cat``.
    """
    sub = subset(metrics, species=species, drug=drug, split=split)
    if sub.empty or "model" not in sub.columns:
        logger.warning("ea_ca_by_model: no %s rows for %s %s", split, species, drug)
        return None
    models = order_models(sub["model"])
    sub = sub.drop_duplicates("model")
    sub = sub.set_index(sub["model"].astype("str")).loc[models]
    ea = _pct(_float_col(sub, "essential_agreement"))
    ca = _pct(_float_col(sub, "categorical_agreement"))
    n_exact = _float_col(sub, "n_exact")
    n_cat = _float_col(sub, "n_cat")
    ea_label = "EA (within +/-1 doubling step; exact lab MICs)"

    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(models) + 2.5), 4.2))
    _grouped_bars(
        ax,
        models,
        {ea_label: ea, "CA (same S/I/R)": ca},
        {ea_label: PALETTE[2], "CA (same S/I/R)": PALETTE[4]},
        {
            ea_label: [f"n={_fmt_n(v)}" for v in n_exact],
            "CA (same S/I/R)": [f"n={_fmt_n(v)}" for v in n_cat],
        },
    )
    ax.axhline(TARGET_EA_PCT, color=GREY, ls="--", lw=1, label=f"EA / CA target {TARGET_EA_PCT:g}% ({TARGET_NOTE})")
    ax.set_ylim(0, 112)
    ax.set_xlabel("Model")
    ax.set_ylabel("Agreement (% of evaluated rows; n above each bar)")
    ax.set_title(f"EA and CA by model, {split_label(split)}: {species} {drug}")
    _legend_below(ax)
    return _save(fig, out_path, synthetic)


# --------------------------------------------------------------------------- #
# 3. MIC confusion heatmap
# --------------------------------------------------------------------------- #


def pick_model(available: Iterable[Any], preferred: str = MAIN_MODEL) -> str | None:
    """``preferred`` if present, else the best-ranked model in :data:`MODEL_ORDER` (reversed)."""
    models = order_models(available)
    if not models:
        return None
    if preferred in models:
        return preferred
    return models[-1]


def mic_confusion(
    preds: pd.DataFrame | None,
    species: str,
    drug: str,
    out_path: Path,
    model: str = MAIN_MODEL,
    split: str = "test",
    droplog: DropLog | None = None,
    *,
    synthetic: bool = False,
) -> Path | None:
    """Heatmap of lab doubling step x predicted doubling step on exact lab MICs.

    Only rows with an exact lab MIC (one doubling step and, when the preds carry
    it, ``lab_exact`` true: :func:`genome2mic.eval.metrics.exact_lab_mask`) and a
    non-null ``pred_mic`` are drawn -- the same rows as EA in ``metrics.parquet``; the excluded count is
    recorded in ``droplog`` when given. Cells within +/-1 step of the diagonal
    (the essential-agreement band) are outlined.
    """
    sub = subset(preds, species=species, drug=drug, split=split)
    if sub.empty or "model" not in sub.columns:
        logger.warning("mic_confusion: no %s rows for %s %s", split, species, drug)
        return None
    chosen = pick_model(sub["model"], model)
    if chosen is None:
        return None
    if chosen != model:
        logger.warning("mic_confusion: model %r absent for %s %s; using %r", model, species, drug, chosen)
    sub = subset(sub, model=chosen).reset_index(drop=True)
    lab_hi = _float_col(sub, "lab_upper")
    pred = _float_col(sub, "pred_mic")
    with np.errstate(invalid="ignore"):
        has_lab = {"lab_lower", "lab_upper"} <= set(sub.columns)
        exact = (exact_lab_mask(sub) if has_lab else np.zeros(len(sub), dtype=bool)) & np.isfinite(pred) & (pred > 0)
    n_dropped = int(len(sub) - exact.sum())
    if droplog is not None:
        droplog.drop(
            "mic_confusion: lab result not an exact MIC (interval wider than one doubling step, or lab_exact false), or null prediction",
            n_dropped,
            detail=f"{species} {drug} {chosen} {split}",
        )
    if exact.sum() == 0:
        logger.warning("mic_confusion: no exact rows for %s %s (%s, %s)", species, drug, chosen, split)
        return None

    lab_step = np.rint(np.log2(lab_hi[exact])).astype(int)
    pred_step = np.rint(np.log2(pred[exact])).astype(int)
    lo = int(min(lab_step.min(), pred_step.min()))
    hi = int(max(lab_step.max(), pred_step.max()))
    steps = np.arange(lo, hi + 1)
    matrix = np.zeros((len(steps), len(steps)), dtype=int)
    np.add.at(matrix, (lab_step - lo, pred_step - lo), 1)
    n_total = int(matrix.sum())
    within = sum(int(matrix[i, j]) for i in range(len(steps)) for j in range(len(steps)) if abs(i - j) <= 1)
    ea_pct = 100.0 * within / n_total

    size = max(4.5, 0.55 * len(steps) + 2)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    image = ax.imshow(matrix, cmap="Blues", origin="lower")
    labels = [fmt_mic(2.0**k) for k in steps]
    ax.set_xticks(np.arange(len(steps)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(steps)), labels)
    threshold = matrix.max() * 0.6 if matrix.max() > 0 else 1
    for i in range(len(steps)):
        for j in range(len(steps)):
            count = matrix[i, j]
            if count:
                ax.text(j, i, str(count), ha="center", va="center", fontsize=7, color="white" if count > threshold else "black")
        ax.add_patch(Rectangle((i - 1.5, i - 0.5), 3, 1, fill=False, edgecolor=PALETTE[3], lw=1.0, clip_on=True))
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(-0.5, len(steps) - 0.5)
    ax.set_xlabel("Predicted MIC (mg/L, rounded up to doubling step)")
    ax.set_ylabel("Lab MIC (mg/L, exact results only)")
    ax.set_title(
        f"MIC confusion, {split_label(split)}: {species} {drug}\n"
        f"{chosen}, exact lab MICs only, n={n_total}, EA (outlined +/-1-step band) = {ea_pct:.1f}%",
        fontsize=10,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Genomes (count)")
    return _save(fig, out_path, synthetic)


# --------------------------------------------------------------------------- #
# 4. Band coverage and width
# --------------------------------------------------------------------------- #


def band_coverage(
    metrics: pd.DataFrame | None,
    species: str,
    out_path: Path,
    split: str = "test",
    *,
    synthetic: bool = False,
) -> Path | None:
    """Conformal band coverage (%) against the 90 % nominal level, and mean band width, per drug.

    Coverage is over exact lab MICs (the rows the conformal ``q`` is calibrated on);
    bars are annotated with that denominator, ``n_band`` (``n`` for older tables).
    """
    sub = subset(metrics, species=species, split=split)
    if sub.empty or "drug" not in sub.columns or "model" not in sub.columns:
        logger.warning("band_coverage: no %s rows for %s", split, species)
        return None
    coverage_all = _float_col(sub, "band_coverage")
    sub = sub.loc[~np.isnan(coverage_all)]
    if sub.empty:
        logger.warning("band_coverage: band_coverage is null for every %s row of %s", split, species)
        return None
    drugs = sorted(sub["drug"].astype("str").unique())
    models = order_models(sub["model"])
    coverage: dict[str, np.ndarray] = {}
    width: dict[str, np.ndarray] = {}
    n_labels: dict[str, list[str]] = {}
    for model in models:
        rows = subset(sub, model=model).drop_duplicates("drug")
        rows = rows.set_index(rows["drug"].astype("str")).reindex(drugs)
        coverage[model] = _pct(_float_col(rows, "band_coverage"))
        width[model] = _float_col(rows, "band_width_steps")
        n_labels[model] = [f"n={_fmt_n(v)}" for v in _denominator(rows, "n_band")]
    colors = {m: model_color(m) for m in models}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 2.2 * len(drugs) * max(len(models), 1) / 2 + 6), 4.2))
    _grouped_bars(ax1, drugs, coverage, colors, n_labels)
    ax1.axhline(TARGET_COVERAGE_PCT, color=GREY, ls="--", lw=1, label=f"nominal 90% band level ({TARGET_NOTE})")
    ax1.set_ylim(0, 112)
    ax1.set_xlabel("Drug")
    ax1.set_ylabel("Band coverage (% of exact lab MICs inside the 90% band)")
    ax1.set_title(f"Conformal band coverage, {split_label(split)}: {species}", fontsize=10)
    _legend_below(ax1, ncol=2)
    _grouped_bars(ax2, drugs, width, colors)
    ax2.set_xlabel("Drug")
    ax2.set_ylabel("Mean band width (doubling steps)")
    ax2.set_title(f"Mean conformal band width, {split_label(split)}: {species}", fontsize=10)
    finite_width = [v for arr in width.values() for v in arr if not _isna(v)]
    ax2.set_ylim(0, (max(finite_width) if finite_width else 1.0) * 1.3 + 0.2)
    _legend_below(ax2, ncol=min(len(models), 3) or 1)
    return _save(fig, out_path, synthetic)


# --------------------------------------------------------------------------- #
# 5. Accuracy vs distance to nearest training genome
# --------------------------------------------------------------------------- #


def _bin_sort_key(label: str) -> tuple[float, str]:
    text = str(label)
    for token in text.replace("(", " ").replace("[", " ").replace("]", " ").replace(")", " ").replace(",", " ").split():
        try:
            return (float(token), text)
        except ValueError:
            continue
    return (math.inf, text)


def accuracy_vs_distance(
    metrics_by_distance: pd.DataFrame | None,
    species: str,
    out_path: Path,
    model: str | None = None,
    *,
    synthetic: bool = False,
) -> Path | None:
    """EA and VME (%) per nearest-training-distance bin, one line per drug.

    Distance bins are the ``distance_bin`` labels of ``metrics_by_distance.parquet``
    ordered by their lower edge. If the frame carries a ``split`` column only the
    ``test`` rows are used; otherwise all rows are assumed to be test rows (the
    distance to the nearest *training* genome is only meaningful off the train set).
    EA points are annotated with ``n_exact`` and VME points with ``n_lab_r`` (their
    denominators; ``n`` for older tables without those columns).
    """
    sub = subset(metrics_by_distance, species=species)
    if "split" in sub.columns and not sub.empty:
        sub = subset(sub, split="test")
    if sub.empty or "distance_bin" not in sub.columns or "drug" not in sub.columns:
        logger.warning("accuracy_vs_distance: no rows for %s", species)
        return None
    chosen = pick_model(sub["model"], model or MAIN_MODEL) if "model" in sub.columns else None
    if chosen is not None:
        sub = subset(sub, model=chosen)
    bins = sorted(sub["distance_bin"].astype("str").unique(), key=_bin_sort_key)
    x = np.arange(len(bins))
    drugs = sorted(sub["drug"].astype("str").unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, drug in enumerate(drugs):
        rows = subset(sub, drug=drug).drop_duplicates("distance_bin")
        rows = rows.set_index(rows["distance_bin"].astype("str")).reindex(bins)
        ea = _pct(_float_col(rows, "essential_agreement"))
        vme = _pct(_float_col(rows, "vme_rate"))
        n_ea = _denominator(rows, "n_exact")
        n_vme = _denominator(rows, "n_lab_r")
        color = PALETTE[i % len(PALETTE)]
        ax1.plot(x, ea, marker="o", color=color, label=drug)
        ax2.plot(x, vme, marker="o", color=color, label=drug)
        for xi, (e, v, count_ea, count_vme) in enumerate(zip(ea, vme, n_ea, n_vme, strict=True)):
            if not _isna(e):
                ax1.annotate(f"n={_fmt_n(count_ea)}", (xi, e), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6)
            if not _isna(v):
                ax2.annotate(f"n={_fmt_n(count_vme)}", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6)
    model_note = f" ({chosen})" if chosen else ""
    for ax, ylabel, target, target_label, title in (
        (ax1, "EA (% of exact lab MICs within +/-1 doubling step)", TARGET_EA_PCT, f"EA target {TARGET_EA_PCT:g}% ({TARGET_NOTE})", "Essential agreement"),
        (ax2, "VME (% of lab-R rows predicted S)", TARGET_VME_PCT, f"VME target {TARGET_VME_PCT:g}% ({TARGET_NOTE})", "Very major errors"),
    ):
        ax.axhline(target, color=GREY, ls="--", lw=1, label=target_label)
        ax.set_xticks(x, bins, rotation=20, ha="right")
        ax.set_xlabel("Mash distance to nearest training genome (bin)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        _legend_below(ax, ncol=2)
    ax1.set_ylim(0, 112)
    ax2.set_ylim(bottom=0)
    fig.suptitle(f"Accuracy by distance to the nearest training genome, test set: {species}{model_note}", fontsize=11)
    return _save(fig, out_path, synthetic)


# --------------------------------------------------------------------------- #
# 6. Feature importance
# --------------------------------------------------------------------------- #


def _sequence_list(value: Any) -> list[str]:
    if value is None or isinstance(value, float):
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return []


def feature_label(
    feature: str,
    unitig_index: pd.DataFrame | None = None,
    known_columns: pd.DataFrame | None = None,
) -> str:
    """Readable label for a feature column.

    ``u_000017`` -> ``u_000017 (3 unitigs, train freq 0.23, ACGT...)`` via the unitig
    index; ``gene_blakpc_2`` -> ``gene_blakpc_2 (blaKPC-2, BETA-LACTAM/CARBAPENEM)`` via
    ``known_amr_columns.csv``. Unknown features are returned unchanged.
    """
    name = str(feature)
    if name.startswith("u_") and unitig_index is not None and "pattern_id" in unitig_index.columns:
        rows = unitig_index.loc[unitig_index["pattern_id"].astype("str") == name]
        if not rows.empty:
            row = rows.iloc[0]
            parts: list[str] = []
            if "n_unitigs" in rows.columns and not _isna(row["n_unitigs"]):
                parts.append(f"{int(row['n_unitigs'])} unitig{'s' if int(row['n_unitigs']) != 1 else ''}")
            if "train_frequency" in rows.columns and not _isna(row["train_frequency"]):
                parts.append(f"train freq {float(row['train_frequency']):.2f}")
            seqs = _sequence_list(row["unitig_sequences"]) if "unitig_sequences" in rows.columns else []
            if seqs:
                first = seqs[0]
                parts.append(first if len(first) <= 16 else first[:13] + "...")
            return f"{name} ({', '.join(parts)})" if parts else name
    if known_columns is not None and "column_name" in known_columns.columns:
        rows = known_columns.loc[known_columns["column_name"].astype("str") == name]
        if not rows.empty:
            row = rows.iloc[0]
            parts = []
            if "source_symbol" in rows.columns and not _isna(row["source_symbol"]):
                parts.append(str(row["source_symbol"]))
            klass = str(row["class"]) if "class" in rows.columns and not _isna(row["class"]) else ""
            subclass = str(row["subclass"]) if "subclass" in rows.columns and not _isna(row["subclass"]) else ""
            if klass or subclass:
                parts.append("/".join(p for p in (klass, subclass) if p))
            return f"{name} ({', '.join(parts)})" if parts else name
    return name


def _feature_kind(feature: str) -> str:
    for prefix, kind in (("gene_", "acquired gene (gene_)"), ("point_", "point mutation (point_)"), ("n_class_", "class count (n_class_)"), ("u_", "unitig pattern (u_)")):
        if feature.startswith(prefix):
            return kind
    return "other"


_KIND_COLORS = {
    "acquired gene (gene_)": PALETTE[0],
    "point mutation (point_)": PALETTE[1],
    "class count (n_class_)": PALETTE[2],
    "unitig pattern (u_)": PALETTE[4],
    "other": GREY,
}


def top_features(
    importance: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    top_k: int = TOP_K_FEATURES,
) -> pd.DataFrame:
    """The ``top_k`` ``{feature, gain}`` rows by gain (highest first); empty when unusable.

    The single ranking used by :func:`feature_importance` and by the report when it
    decides which unitig patterns it needs labels for.
    """
    if importance is None:
        return pd.DataFrame(columns=["feature", "gain"])
    frame = importance if isinstance(importance, pd.DataFrame) else pd.DataFrame(list(importance))
    if frame.empty or "feature" not in frame.columns or "gain" not in frame.columns:
        return pd.DataFrame(columns=["feature", "gain"])
    frame = frame.assign(gain=pd.to_numeric(frame["gain"], errors="coerce")).dropna(subset=["gain"])
    return frame.sort_values("gain", ascending=False, kind="stable").head(top_k)


def feature_importance(
    importance: Sequence[Mapping[str, Any]] | pd.DataFrame | None,
    species: str,
    drug: str,
    out_path: Path,
    unitig_index: pd.DataFrame | None = None,
    known_columns: pd.DataFrame | None = None,
    top_k: int = TOP_K_FEATURES,
    model: str = MAIN_MODEL,
    *,
    synthetic: bool = False,
) -> Path | None:
    """Horizontal bar chart of the top-``top_k`` gain importances of the main model.

    ``importance`` is the content of ``models/<SPECIES>/<drug>/importance.json``
    (a list of ``{feature, gain}``). Unitig ids are mapped to their pattern via the
    unitig index; known-AMR columns are labelled with their source symbol.
    """
    if importance is None:
        return None
    frame = top_features(importance, top_k)
    if frame.empty:
        logger.warning("feature_importance: no usable {feature, gain} rows for %s %s", species, drug)
        return None
    features = [str(f) for f in frame["feature"]]
    gains = frame["gain"].to_numpy(dtype=float)
    labels = [feature_label(f, unitig_index, known_columns) for f in features]
    kinds = [_feature_kind(f) for f in features]
    colors = [_KIND_COLORS[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(features) + 1.8))
    y = np.arange(len(features))[::-1]
    ax.barh(y, gains, color=colors)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("Gain (XGBoost total loss reduction attributed to the feature, model units)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top-{len(features)} gain importances, final fit on the train split: {species} {drug} ({model})", fontsize=10)
    present = [k for k in _KIND_COLORS if k in kinds]
    ax.legend(handles=[Patch(color=_KIND_COLORS[k], label=k) for k in present], fontsize=7, loc="lower right")
    return _save(fig, out_path, synthetic)


# --------------------------------------------------------------------------- #
# 7. Label counts
# --------------------------------------------------------------------------- #


def label_counts(
    counts: pd.DataFrame | None,
    species: str,
    out_path: Path,
    *,
    synthetic: bool = False,
) -> Path | None:
    """Stacked S/I/R counts and exact-vs-censored counts per drug (all labelled genomes).

    ``counts`` is ``label_counts.csv`` (``species, drug, n, n_R, n_S, n_I, n_exact,
    n_censored, n_distinct_mic``).
    """
    sub = subset(counts, species=species)
    if sub.empty or "drug" not in sub.columns:
        logger.warning("label_counts: no rows for %s", species)
        return None
    sub = sub.drop_duplicates("drug")
    order = np.argsort(-_float_col(sub, "n", default=0.0), kind="stable")
    sub = sub.iloc[order]
    drugs = [str(d) for d in sub["drug"]]
    x = np.arange(len(drugs))
    n_s = np.nan_to_num(_float_col(sub, "n_S"))
    n_i = np.nan_to_num(_float_col(sub, "n_I"))
    n_r = np.nan_to_num(_float_col(sub, "n_R"))
    n_exact = np.nan_to_num(_float_col(sub, "n_exact"))
    n_cens = np.nan_to_num(_float_col(sub, "n_censored"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 1.2 * len(drugs) + 6), 4.2))
    ax1.bar(x, n_s, color=PALETTE[0], label="S (susceptible)")
    ax1.bar(x, n_i, bottom=n_s, color=PALETTE[1], label="I")
    ax1.bar(x, n_r, bottom=n_s + n_i, color=PALETTE[3], label="R (resistant)")
    for xi, total in zip(x, n_s + n_i + n_r, strict=True):
        ax1.annotate(f"n={int(total)}", (xi, total), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax1.set_xticks(x, drugs, rotation=25, ha="right")
    ax1.set_xlabel("Drug")
    ax1.set_ylabel("Genomes with a lab result (count)")
    ax1.set_title(f"Lab S/I/R label counts, all labelled genomes: {species}", fontsize=10)
    _legend_below(ax1, ncol=3)
    ax2.bar(x, n_exact, color=PALETTE[2], label="exact MIC (one doubling step)")
    ax2.bar(x, n_cens, bottom=n_exact, color=PALETTE[5], label="censored or multi-step (<=, >, S/I/R-only)")
    for xi, total in zip(x, n_exact + n_cens, strict=True):
        ax2.annotate(f"n={int(total)}", (xi, total), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax2.set_xticks(x, drugs, rotation=25, ha="right")
    ax2.set_xlabel("Drug")
    ax2.set_ylabel("Genomes with a lab result (count)")
    ax2.set_title(f"Exact vs censored MIC labels, all labelled genomes: {species}", fontsize=10)
    _legend_below(ax2, ncol=2)
    return _save(fig, out_path, synthetic)


# --------------------------------------------------------------------------- #
# 8. Lineage cluster sizes
# --------------------------------------------------------------------------- #


def lineage_clusters(
    lineages: pd.DataFrame | None,
    splits: pd.DataFrame | None,
    species: str,
    out_path: Path,
    *,
    synthetic: bool = False,
) -> Path | None:
    """Distribution of lineage-cluster sizes, train vs test.

    Joins ``lineages.parquet`` with ``splits.parquet`` on ``genome_id``. A cluster
    that appears in more than one split is drawn as ``mixed (leak)`` so the
    violation is visible; the leakage checklist reports it formally. Without
    ``splits`` the distribution is drawn for all genomes.
    """
    sub = subset(lineages, species=species)
    if sub.empty or "lineage_cluster" not in sub.columns or "genome_id" not in sub.columns:
        logger.warning("lineage_clusters: no rows for %s", species)
        return None
    if splits is not None and {"genome_id", "split"}.issubset(splits.columns):
        joined = sub[["genome_id", "lineage_cluster"]].merge(splits[["genome_id", "split"]], on="genome_id", how="left")
        joined = joined.assign(split=joined["split"].astype("str").fillna("unsplit"))
    else:
        joined = sub[["genome_id", "lineage_cluster"]].assign(split="all genomes (no splits.parquet)")
    per_cluster = joined.groupby("lineage_cluster").agg(size=("genome_id", "size"), splits=("split", lambda s: sorted(set(s))))
    per_cluster = per_cluster.assign(group=per_cluster["splits"].map(lambda s: s[0] if len(s) == 1 else "mixed (leak)"))
    groups = sorted(per_cluster["group"].unique(), key=lambda g: ({"train": 0, "test": 1}.get(g, 2), g))
    labels = [b[2] for b in CLUSTER_SIZE_BINS]

    def _bin(size: int) -> str:
        for low, high, label in CLUSTER_SIZE_BINS:
            if low <= size <= high:
                return label
        return labels[-1]

    per_cluster = per_cluster.assign(size_bin=per_cluster["size"].map(_bin))
    series = {
        g: np.array([int(((per_cluster["group"] == g) & (per_cluster["size_bin"] == b)).sum()) for b in labels], dtype=float)
        for g in groups
    }
    colors = {g: {"train": PALETTE[0], "test": PALETTE[1], "mixed (leak)": PALETTE[3]}.get(g, GREY) for g in groups}
    summary = "; ".join(
        f"{g}: {int((per_cluster['group'] == g).sum())} clusters, {int(per_cluster.loc[per_cluster['group'] == g, 'size'].sum())} genomes"
        for g in groups
    )

    fig, ax = plt.subplots(figsize=(8, 4.2))
    _grouped_bars(ax, labels, series, colors)
    ax.set_xlabel("Genomes per lineage cluster (count)")
    ax.set_ylabel("Lineage clusters (count)")
    ax.set_title(f"Lineage cluster sizes, train vs test: {species}\n{summary}", fontsize=9)
    _legend_below(ax, ncol=min(len(groups), 3) or 1, pad=0.2)
    return _save(fig, out_path, synthetic)


__all__ = [
    "DPI",
    "MAIN_MODEL",
    "MODEL_ORDER",
    "PALETTE",
    "SYNTHETIC_MARK",
    "SYNTHETIC_NOTE",
    "TOP_K_FEATURES",
    "TARGET_CA_PCT",
    "TARGET_COVERAGE_PCT",
    "TARGET_EA_PCT",
    "TARGET_ME_PCT",
    "TARGET_VME_PCT",
    "accuracy_vs_distance",
    "band_coverage",
    "ea_ca_by_model",
    "feature_importance",
    "feature_label",
    "fmt_mic",
    "label_counts",
    "lineage_clusters",
    "mark_synthetic",
    "mic_confusion",
    "model_color",
    "order_models",
    "pick_model",
    "split_label",
    "subset",
    "top_features",
    "vme_me_by_model",
]
