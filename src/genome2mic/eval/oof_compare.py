"""Out-of-fold comparison harness: our ``split == 'cv'`` preds vs another model's OOF preds.

``python -m genome2mic compare-oof --root <root> --develop-preds <parquet> [--breakpoints <csv>] --out <dir>``

Both sides are scored on **identical rows and identical breakpoints**:

1. Our preds: every ``results/preds_*.parquet`` row with ``split == 'cv'`` and a
   prediction (``b0_resfinder`` has none and is left out). Theirs: the reference OOF
   table (columns ``genome_id, species, drug, pred_mic, band_low, band_high,
   lab_lower, lab_upper, model``; ``split`` other than ``oof``/``cv`` is dropped).
2. Rows are keyed by ``(species, genome_id, drug)``; only keys present for the
   reference model **and every one of our models** are kept, and keys whose lab
   interval differs between the two tables are dropped (counted). So every model is
   scored on exactly the same lab results.
3. One breakpoint table for everyone: the call standard (``configs/drugs.yaml``) by
   default, or ``--breakpoints`` (``species, drug, s_breakpoint, r_breakpoint``; any
   other column ignored; identical duplicate rows allowed). ``pred_sir`` is
   re-classified from ``pred_mic``, ``lab_sir`` is the reported ``sir`` from
   ``labels.parquet`` (else the interval's category under the same table),
   ``lab_sir_rederived`` the lab interval under the table, and ``call`` comes from
   each model's own band with the prediction pipeline's rule and overrides
   (:func:`genome2mic.predict.rank.call_array`; natural resistance and the
   strong-marker column rule on the release's known-AMR table) -- applied
   identically to the reference model's bands.
4. Metrics via :func:`genome2mic.eval.metrics.summarize`: VME / call-level metrics / ME
   / CA (as reported and re-derived), EA and exact agreement on exact (one-step,
   measured) lab MICs, band coverage on the same rows, mean band width, and the n's.

Outputs ``oof_compare.csv`` (VME first) and ``oof_compare.md`` (+ ``oof_compare_rows.csv``
with the per-key row-count bookkeeping). Every number is out-of-fold on the train
split; the test split is never read.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genome2mic.config import Breakpoint, Config
from genome2mic.droplog import DropLog
from genome2mic.eval import metrics
from genome2mic.eval.run import load_preds
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "compare_oof"
OUR_SPLIT = "cv"
REFERENCE_SPLITS: tuple[str, ...] = ("oof", "cv")
KEY: tuple[str, ...] = ("species", "genome_id", "drug")
VME_TARGET = 0.015
START_DRUGS: tuple[str, ...] = ("meropenem", "ciprofloxacin", "ceftriaxone")

OUTPUT_COLUMNS: tuple[str, ...] = (
    "species", "drug", "model",
    "vme_rate", "vme_rate_rederived",
    "call_vme_rate", "call_vme_rate_rederived",
    "call_me_rate", "call_me_rate_rederived",
    "active_call_rate_s", "active_call_rate_s_rederived",
    "uncertain_rate", "uncertain_rate_rederived",
    "me_rate", "me_rate_rederived",
    "categorical_agreement", "categorical_agreement_rederived",
    "essential_agreement", "exact_agreement", "n_exact",
    "band_coverage", "n_band", "band_width_steps",
    "n", "n_lab_r", "n_lab_s", "n_lab_r_rederived", "n_lab_s_rederived",
    "n_call_lab_r_rederived", "n_call_lab_s_rederived",
    "breakpoints",
)
"""Column order of ``oof_compare.csv``: VME first (raw, then call-level), then ME, CA, EA, bands, n's."""


@dataclass
class CompareResult:
    table: pd.DataFrame
    summary: pd.DataFrame
    csv_path: Path
    md_path: Path
    n_rows: int


# --------------------------------------------------------------------------- #
# Breakpoints
# --------------------------------------------------------------------------- #


def load_breakpoint_csv(path: Path) -> dict[tuple[str, str], Breakpoint]:
    """``(species, drug) -> Breakpoint`` from a standalone breakpoint CSV (develop or ours).

    Required columns ``species, drug, s_breakpoint, r_breakpoint`` (same convention as
    ours: S if MIC <= s, R if MIC > r). Identical duplicate rows are collapsed;
    conflicting duplicates raise ``ValueError``. ``standard``/``version`` are taken
    from the file name (``clsi_2019.csv`` -> ``CLSI``, ``2019``) when it has that form.
    """
    path = Path(path)
    stem = path.stem.split("_")
    standard = stem[0].upper() if len(stem) == 2 else path.stem
    version = stem[1] if len(stem) == 2 else ""
    out: dict[tuple[str, str], Breakpoint] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ("species", "drug", "s_breakpoint", "r_breakpoint") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}")
        for line_no, row in enumerate(reader, start=2):
            key = (row["species"].strip(), row["drug"].strip())
            s, r = float(row["s_breakpoint"]), float(row["r_breakpoint"])
            if not (s > 0 and r > 0 and s <= r):
                raise ValueError(f"{path}: line {line_no}: need 0 < s_breakpoint <= r_breakpoint, got {s}, {r}")
            bp = Breakpoint(species=key[0], drug=key[1], s_breakpoint=s, r_breakpoint=r, standard=standard, version=version)
            if key in out and (out[key].s_breakpoint, out[key].r_breakpoint) != (s, r):
                raise ValueError(f"{path}: conflicting rows for {key}")
            out[key] = bp
    if not out:
        raise ValueError(f"{path}: no breakpoint rows")
    return out


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def _ours(paths: Paths) -> pd.DataFrame:
    preds = load_preds(paths)
    preds = preds.loc[(preds["split"].astype("str") == OUR_SPLIT) & preds["pred_mic"].notna()].copy()
    if preds.empty:
        raise ValueError(f"no split == '{OUR_SPLIT}' rows with a prediction under {paths.results_dir}")
    for column in ("genome_id", "species", "drug", "model"):
        preds[column] = preds[column].astype(str)
    if "lab_exact" not in preds.columns:
        raise ValueError("our preds have no lab_exact column (re-run train)")
    return preds


def _reference(path: Path) -> pd.DataFrame:
    ref = pd.read_parquet(path)
    needed = ["genome_id", "species", "drug", "pred_mic", "band_low", "band_high", "lab_lower", "lab_upper", "model"]
    missing = [c for c in needed if c not in ref.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    if "split" in ref.columns:
        keep = ref["split"].astype("str").isin(REFERENCE_SPLITS)
        if (~keep).any():
            logger.warning("%s: %d row(s) with split not in %s ignored", path, int((~keep).sum()), REFERENCE_SPLITS)
        ref = ref.loc[keep]
    ref = ref.loc[ref["pred_mic"].notna()].copy()
    for column in ("genome_id", "species", "drug", "model"):
        ref[column] = ref[column].astype(str)
    if ref.duplicated(["model", *KEY]).any():
        raise ValueError(f"{path}: duplicate (model, species, genome_id, drug) rows")
    ref["model"] = "develop:" + ref["model"]
    return ref


def _intersection(ours: pd.DataFrame, ref: pd.DataFrame, droplog: DropLog) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keys present for the reference model and every one of our models, with equal lab intervals."""
    keys = None
    for model, block in [*ours.groupby("model"), *ref.groupby("model")]:
        current = set(map(tuple, block[list(KEY)].to_numpy()))
        keys = current if keys is None else keys & current
        logger.info("compare-oof: %s has %d (species, genome_id, drug) keys", model, len(current))
    keys = keys or set()
    key_frame = pd.DataFrame(sorted(keys), columns=list(KEY))
    ours_k = ours.merge(key_frame, on=list(KEY), how="inner")
    ref_k = ref.merge(key_frame, on=list(KEY), how="inner")
    droplog.drop("our cv row not in the intersection", int(len(ours) - len(ours_k)))
    droplog.drop("reference oof row not in the intersection", int(len(ref) - len(ref_k)))

    # Lab intervals must agree (same lab result) between the two tables.
    lab_ours = ours_k.drop_duplicates(list(KEY))[[*KEY, "lab_lower", "lab_upper", "lab_exact"]]
    lab_ref = ref_k.drop_duplicates(list(KEY))[[*KEY, "lab_lower", "lab_upper"]]
    both = lab_ours.merge(lab_ref, on=list(KEY), suffixes=("", "_ref"))
    same = np.isclose(both["lab_lower"], both["lab_lower_ref"], rtol=1e-9, atol=1e-12) & (
        np.isclose(both["lab_upper"], both["lab_upper_ref"], rtol=1e-9, atol=1e-12)
        | (np.isinf(both["lab_upper"]) & np.isinf(both["lab_upper_ref"]))
    )
    bad = both.loc[~same, list(KEY)]
    droplog.drop("lab interval differs between our preds and the reference preds", int(len(bad)))
    if len(bad):
        logger.warning("compare-oof: %d key(s) with different lab intervals dropped, e.g. %s", len(bad), bad.head(3).to_dict("records"))
        bad_keys = set(map(tuple, bad.to_numpy()))
        ours_k = ours_k.loc[[k not in bad_keys for k in map(tuple, ours_k[list(KEY)].to_numpy())]]
        ref_k = ref_k.loc[[k not in bad_keys for k in map(tuple, ref_k[list(KEY)].to_numpy())]]
    # The reference rows carry no lab_exact; take ours (same lab result, same labels row).
    exact = lab_ours.set_index(list(KEY))["lab_exact"]
    ref_k = ref_k.drop(columns=[c for c in ("lab_exact",) if c in ref_k.columns])
    ref_k = ref_k.join(exact, on=list(KEY))
    return ours_k.reset_index(drop=True), ref_k.reset_index(drop=True)


def _reported_sir(paths: Paths, keys: pd.DataFrame) -> pd.Series:
    """The labels' reported ``sir`` per key (null where the lab gave no category)."""
    labels = pd.read_parquet(paths.labels, columns=["genome_id", "species", "drug", "sir"])
    for column in KEY:
        labels[column] = labels[column].astype(str)
    labels = labels.drop_duplicates(list(KEY))
    merged = keys[list(KEY)].merge(labels, on=list(KEY), how="left")
    return merged["sir"].astype(object).where(merged["sir"].notna(), None)


def _rescore(
    block: pd.DataFrame,
    paths: Paths,
    config: Config,
    breakpoints: Mapping[tuple[str, str], Breakpoint] | None,
    known: pd.DataFrame,
) -> pd.DataFrame:
    """pred_sir / lab_sir / lab_sir_rederived / call recomputed under one breakpoint table."""
    from genome2mic.models.train import derive_lab_sir, rederive_lab_sir  # noqa: PLC0415
    from genome2mic.predict import rank  # noqa: PLC0415

    out = block[[*KEY, "model", "pred_mic", "band_low", "band_high", "lab_lower", "lab_upper", "lab_exact"]].copy()
    out["split"] = OUR_SPLIT
    out["reported_sir"] = _reported_sir(paths, out).to_numpy(dtype=object)
    pred_sir = np.full(len(out), None, dtype=object)
    lab_sir = np.full(len(out), None, dtype=object)
    rederived = np.full(len(out), None, dtype=object)
    call = np.full(len(out), None, dtype=object)
    for (species, drug), idx in out.groupby(["species", "drug"]).indices.items():
        bp = breakpoints.get((species, drug)) if breakpoints is not None else config.call_breakpoint(species, drug)
        sub = out.iloc[idx]
        lo = sub["lab_lower"].to_numpy(dtype=float)
        hi = sub["lab_upper"].to_numpy(dtype=float)
        if bp is not None:
            pred_sir[idx] = [config.sir_from_mic(float(m), bp) for m in sub["pred_mic"].to_numpy(dtype=float)]
        lab_sir[idx] = [derive_lab_sir(s, a, b, bp) for s, a, b in zip(sub["reported_sir"], lo, hi)]
        rederived[idx] = [rederive_lab_sir(a, b, bp) for a, b in zip(lo, hi)]
        rows = known.reindex(sub["genome_id"].tolist())
        marker = rank.strong_marker_mask(
            rows.fillna(0), config.drugs.get(drug),
            exclude_columns=rank.intrinsic_columns(config.intrinsic_symbols(species)),
        )
        call[idx] = rank.call_array(
            sub["band_low"].to_numpy(dtype=float), sub["band_high"].to_numpy(dtype=float), bp,
            natural_resistance=config.is_naturally_resistant(species, drug), strong_marker=marker,
        )
    out["pred_sir"] = pred_sir
    out["lab_sir"] = lab_sir
    out["lab_sir_rederived"] = rederived
    out["call"] = call
    out["lab_exact"] = out["lab_exact"].astype(bool)
    return out.drop(columns=["reported_sir"])


# --------------------------------------------------------------------------- #
# Summary and Markdown
# --------------------------------------------------------------------------- #


def summarize_models(table: pd.DataFrame, *, target: float = VME_TARGET) -> pd.DataFrame:
    """Per model across drugs: median EA, drugs meeting call-level VME (re-derived) <= target, median active calls."""
    rows = []
    for model, block in table.groupby("model", sort=True):
        with_bp = block.loc[block["n_call_lab_r_rederived"] > 0]
        rows.append(
            {
                "model": model,
                "n_drugs": int(block["drug"].nunique()),
                "median_ea": float(block["essential_agreement"].median()),
                "n_drugs_with_call_breakpoint": int(with_bp["drug"].nunique()),
                "n_drugs_call_vme_ok": int((with_bp["call_vme_rate_rederived"] <= target + 1e-12).sum()),
                "median_call_vme_rederived": float(with_bp["call_vme_rate_rederived"].median()) if len(with_bp) else math.nan,
                "median_active_call_rate_s": float(with_bp["active_call_rate_s_rederived"].median()) if len(with_bp) else math.nan,
                "median_raw_vme_rederived": float(with_bp["vme_rate_rederived"].median()) if len(with_bp) else math.nan,
                "median_band_coverage": float(block["band_coverage"].median()),
                "median_band_width_steps": float(block["band_width_steps"].median()),
                "n_rows": int(block["n"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _pct(value: Any, n: Any = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    text = f"{100 * float(value):.1f}%"
    return f"{text} (n={int(n)})" if n is not None and not pd.isna(n) else text


def _num(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _md(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


DETAIL_HEADERS: tuple[str, ...] = (
    "Drug", "Model", "Call VME % re-derived (of lab R)", "Raw VME % re-derived", "Raw VME % as reported",
    "Active calls % of lab S (re-derived)", "Uncertain % (re-derived)", "Call ME % re-derived", "ME % re-derived",
    "CA % re-derived", "EA % (exact MICs)", "Exact agreement %", "Band coverage % (exact MICs)", "Band width (steps)", "n",
)


def _detail_rows(table: pd.DataFrame) -> list[list[str]]:
    rows = []
    for record in table.to_dict(orient="records"):
        rows.append([
            record["drug"], record["model"],
            _pct(record["call_vme_rate_rederived"], record["n_call_lab_r_rederived"]),
            _pct(record["vme_rate_rederived"], record["n_lab_r_rederived"]),
            _pct(record["vme_rate"], record["n_lab_r"]),
            _pct(record["active_call_rate_s_rederived"], record["n_call_lab_s_rederived"]),
            _pct(record["uncertain_rate_rederived"]),
            _pct(record["call_me_rate_rederived"], record["n_call_lab_s_rederived"]),
            _pct(record["me_rate_rederived"], record["n_lab_s_rederived"]),
            _pct(record["categorical_agreement_rederived"]),
            _pct(record["essential_agreement"], record["n_exact"]),
            _pct(record["exact_agreement"], record["n_exact"]),
            _pct(record["band_coverage"], record["n_band"]),
            _num(record["band_width_steps"]),
            int(record["n"]),
        ])
    return rows


def render_markdown(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    breakpoints_label: str,
    develop_preds: Path,
    root: Path,
    row_log: DropLog,
) -> str:
    from genome2mic.api.constants import DISCLAIMER  # noqa: PLC0415

    drops = row_log.to_frame()
    parts = [
        "# Out-of-fold comparison (identical rows, identical breakpoints)",
        "",
        f"> {DISCLAIMER}",
        "",
        f"- Run root: `{root}`; reference OOF preds: `{develop_preds}`.",
        f"- Breakpoints for every model: **{breakpoints_label}** (unverified tables entered from memory; "
        "S/I/R and call metrics are provisional until docs/BREAKPOINT_VERIFICATION.md is completed).",
        "- Every number is cross-validation (out-of-fold) on the release's frozen train folds. The test split "
        "was not read. Rows: the (species, genome_id, drug) keys predicted by every model, with the same lab result.",
        "- VME first. **Call VME** = lab R called likely active (band upper end <= S breakpoint, after the "
        "natural-resistance and strong-marker overrides; the same rule is applied to every model's own band). "
        "Raw VME = predicted S (point MIC) and lab R. Re-derived = lab MIC re-classified under the breakpoints "
        "above; as reported = the lab's own S/I/R.",
        "- EA, exact agreement and band coverage use exact (one-step, measured) lab MICs only.",
        "- Choosing among a few candidate configurations by these OOF scores adds a little optimism; the final "
        "honest number is one test-set run, done later.",
        "",
        "## Summary per model (all drugs)",
        "",
        _md(
            ["Model", "Drugs", "Median EA", f"Drugs with call VME re-derived <= {100 * VME_TARGET:.1f}% (of drugs with a breakpoint)",
             "Median call VME re-derived", "Median active calls % of lab S", "Median raw VME re-derived",
             "Median band coverage", "Median band width (steps)", "Rows"],
            [[r["model"], r["n_drugs"], _pct(r["median_ea"]),
              f"{r['n_drugs_call_vme_ok']} / {r['n_drugs_with_call_breakpoint']}",
              _pct(r["median_call_vme_rederived"]), _pct(r["median_active_call_rate_s"]),
              _pct(r["median_raw_vme_rederived"]), _pct(r["median_band_coverage"]),
              _num(r["median_band_width_steps"]), r["n_rows"]] for r in summary.to_dict(orient="records")],
        ),
        "",
    ]
    start = table.loc[table["drug"].isin(START_DRUGS)]
    if not start.empty:
        parts += ["## Start drugs", "", _md(DETAIL_HEADERS, _detail_rows(start)), ""]
    parts += ["## Every drug", "", _md(DETAIL_HEADERS, _detail_rows(table)), ""]
    if not drops.empty:
        parts += ["## Row bookkeeping", "", _md(list(drops.columns), drops.astype(str).to_numpy().tolist()), ""]
    parts += [f"> {DISCLAIMER}", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def compare(
    paths: Paths,
    config: Config,
    develop_preds: Path,
    *,
    breakpoints_csv: Path | None = None,
    droplog: DropLog | None = None,
) -> tuple[pd.DataFrame, str]:
    """The comparison table (one row per species x drug x model) and its breakpoint label."""
    log = droplog if droplog is not None else DropLog(STAGE)
    ours = _ours(paths)
    ref = _reference(Path(develop_preds))
    ours_k, ref_k = _intersection(ours, ref, log)
    if ours_k.empty:
        raise ValueError("no (species, genome_id, drug) key is shared by the reference and every one of our models")
    if breakpoints_csv is not None:
        breakpoints = load_breakpoint_csv(Path(breakpoints_csv))
        label = f"{Path(breakpoints_csv).name} ({len(breakpoints)} species x drug rows)"
    else:
        breakpoints = None
        label = f"call standard {config.call_standard[0]} {config.call_standard[1]} (configs/breakpoints)"
    known = pd.read_parquet(paths.known_amr)
    known["genome_id"] = known["genome_id"].astype(str)
    known = known.drop_duplicates("genome_id").set_index("genome_id")
    feature_cols = [c for c in known.columns if c.startswith(("gene_", "point_"))]
    known = known[feature_cols]
    scored = pd.concat(
        [_rescore(ours_k, paths, config, breakpoints, known), _rescore(ref_k, paths, config, breakpoints, known)],
        ignore_index=True,
    )
    summary = metrics.summarize(scored, drop_log=log)
    summary["breakpoints"] = label
    table = summary[[c for c in OUTPUT_COLUMNS if c in summary.columns]].sort_values(["species", "drug", "model"]).reset_index(drop=True)
    return table, label


def run(
    paths: Paths,
    config: Config,
    *,
    develop_preds: Path,
    out_dir: Path,
    breakpoints_csv: Path | None = None,
) -> CompareResult:
    """Write ``oof_compare.csv``, ``oof_compare.md`` and ``oof_compare_rows.csv`` under ``out_dir``."""
    log = DropLog(STAGE)
    table, label = compare(paths, config, develop_preds, breakpoints_csv=breakpoints_csv, droplog=log)
    summary = summarize_models(table)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "oof_compare.csv"
    md_path = out_dir / "oof_compare.md"
    table.to_csv(csv_path, index=False)
    summary.to_csv(out_dir / "oof_compare_summary.csv", index=False)
    log.to_frame().to_csv(out_dir / "oof_compare_rows.csv", index=False)
    md_path.write_text(
        render_markdown(table, summary, breakpoints_label=label, develop_preds=Path(develop_preds), root=paths.root, row_log=log),
        encoding="utf-8",
    )
    logger.info("compare-oof: %d rows (%d models) -> %s", len(table), table["model"].nunique(), out_dir)
    return CompareResult(table=table, summary=summary, csv_path=csv_path, md_path=md_path, n_rows=int(table["n"].sum()))


__all__ = [
    "CompareResult",
    "OUTPUT_COLUMNS",
    "compare",
    "load_breakpoint_csv",
    "render_markdown",
    "run",
    "summarize_models",
]
