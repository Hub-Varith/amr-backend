"""Automated leakage checklist (``DATA_CONTRACT.md`` section 4).

:func:`run_checks` runs every check against the files under a :class:`Paths` root
and returns ``[{check, passed, detail}, ...]``. ``passed`` is ``True`` / ``False``
when the check ran, and ``None`` when an input file it needs is missing or lacks
the needed columns (the report prints that as *NOT RUN*). A check that raises is
recorded as ``False`` with the exception in ``detail`` -- a leakage check that
cannot finish must not look like a pass.

Checks:

1. **Feature names** -- no forbidden column (``lineage_cluster``, ``st``, ``country``,
   ``year``, ``source``, ``isolation_source``, ``biosample``, ``split``, ``fold``)
   and no un-prefixed column among ``known_amr_columns.csv``, the
   ``known_amr.parquet`` schema, ``models/*/*/features.json`` and
   ``models/*/*/importance.json``.
2. **Clusters within splits** -- no ``lineage_cluster`` in two ``split`` values or
   two ``fold`` values (``lineages.parquet`` joined to ``splits.parquet``).
3. **Unitigs built on train only** -- every ``role == 'built'`` row of
   ``unitigs_<SPECIES>_rows.parquet`` has ``split == 'train'`` (cross-checked
   against ``splits.parquet`` when present).
4. **Biosample de-duplication** -- no ``biosample`` maps to two ``genome_id`` values
   in ``labels.parquet``.
5. **Prediction rows match their split** -- ``cv`` rows are train genomes, ``test``
   rows are test genomes, every genome is in ``splits.parquet``.
6. **Test set touched once** (informational) -- one ``run_id`` per preds file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"lineage_cluster", "st", "country", "year", "source", "isolation_source", "biosample", "split", "fold"}
)
"""Column names that must never enter a feature matrix (CLAUDE.md rule 2, DESIGN conventions)."""

ALLOWED_FEATURE_PREFIXES: tuple[str, ...] = ("gene_", "point_", "n_class_", "u_")
"""The only prefixes a feature column may carry."""

ID_COLUMNS: frozenset[str] = frozenset({"genome_id", "species"})
"""Key columns of ``known_amr.parquet`` that are not features."""

ALLOWED_UNITIG_ROLES: frozenset[str] = frozenset({"built", "queried"})

_MAX_EXAMPLES = 5


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one leakage check.

    Attributes:
        check: Short name of the check.
        passed: ``True`` / ``False`` when the check ran; ``None`` when it could not
            run because an input is missing.
        detail: Counts, offending keys (first few) or the reason it did not run.
    """

    check: str
    passed: bool | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        """``{"check": ..., "passed": ..., "detail": ...}``."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def is_forbidden_feature(name: str) -> bool:
    """True if ``name`` is a forbidden column or derived from one (``country_usa``, ``year_2019``)."""
    key = str(name).strip().lower()
    if key in FORBIDDEN_FEATURES:
        return True
    return any(key.startswith(f"{forbidden}_") for forbidden in FORBIDDEN_FEATURES)


def has_allowed_prefix(name: str) -> bool:
    """True if ``name`` starts with one of :data:`ALLOWED_FEATURE_PREFIXES`."""
    return str(name).startswith(ALLOWED_FEATURE_PREFIXES)


def classify_features(names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return ``(forbidden, unprefixed)`` feature names, each de-duplicated and sorted."""
    forbidden: set[str] = set()
    unprefixed: set[str] = set()
    for raw in names:
        name = str(raw)
        if is_forbidden_feature(name):
            forbidden.add(name)
        elif not has_allowed_prefix(name):
            unprefixed.add(name)
    return sorted(forbidden), sorted(unprefixed)


def _examples(items: Iterable[object], limit: int = _MAX_EXAMPLES) -> str:
    listed = [str(i) for i in items]
    shown = ", ".join(listed[:limit])
    return shown + (f", ... (+{len(listed) - limit} more)" if len(listed) > limit else "")


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def _read_columns(path: Path, wanted: Iterable[str]) -> pd.DataFrame | None:
    """Read ``wanted`` columns from a Parquet file; ``None`` if any is missing."""
    available = set(_parquet_columns(path))
    wanted_list = list(wanted)
    missing = [c for c in wanted_list if c not in available]
    if missing:
        logger.warning("%s lacks columns %s", path, missing)
        return None
    return pd.read_parquet(path, engine="pyarrow", columns=wanted_list)


def _json_list(path: Path) -> list[object]:
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    return content if isinstance(content, list) else []


def _json_dict(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        content = json.load(handle)
    return content if isinstance(content, dict) else {}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_feature_names(paths: Paths) -> CheckResult:
    """No forbidden or un-prefixed column among known-AMR, model feature and importance lists."""
    name = "no forbidden columns among features"
    sources: dict[str, list[str]] = {}
    if paths.known_amr_columns.is_file():
        table = pd.read_csv(paths.known_amr_columns)
        if "column_name" in table.columns:
            sources["known_amr_columns.csv"] = [str(c) for c in table["column_name"].dropna()]
    if paths.known_amr.is_file():
        sources["known_amr.parquet"] = [c for c in _parquet_columns(paths.known_amr) if c not in ID_COLUMNS]
    if paths.models_dir.is_dir():
        for features_json in sorted(paths.models_dir.glob("*/*/features.json")):
            content = _json_dict(features_json)
            names: list[str] = []
            for key in ("known_columns", "unitig_cols"):
                value = content.get(key)
                if isinstance(value, list):
                    names.extend(str(v) for v in value)
            sources[str(features_json.relative_to(paths.models_dir))] = names
        for importance_json in sorted(paths.models_dir.glob("*/*/importance.json")):
            names = [
                str(item["feature"])
                for item in _json_list(importance_json)
                if isinstance(item, dict) and "feature" in item
            ]
            sources[str(importance_json.relative_to(paths.models_dir))] = names
    if not sources:
        return CheckResult(name, None, "no known_amr_columns.csv, known_amr.parquet or models/*/*/{features,importance}.json found")
    problems: list[str] = []
    n_features = 0
    for source, names in sources.items():
        n_features += len(names)
        forbidden, unprefixed = classify_features(names)
        if forbidden:
            problems.append(f"{source}: forbidden {_examples(forbidden)}")
        if unprefixed:
            problems.append(f"{source}: no allowed prefix {_examples(unprefixed)}")
    if problems:
        return CheckResult(name, False, "; ".join(problems))
    return CheckResult(name, True, f"{n_features} feature names across {len(sources)} source(s); none forbidden, all prefixed")


def check_clusters_within_splits(paths: Paths) -> CheckResult:
    """No lineage cluster appears in two splits or in two folds."""
    name = "no lineage cluster in two splits or two folds"
    if not paths.lineages.is_file() or not paths.splits.is_file():
        return CheckResult(name, None, "lineages.parquet or splits.parquet missing")
    lineages = _read_columns(paths.lineages, ["genome_id", "lineage_cluster"])
    splits = _read_columns(paths.splits, ["genome_id", "split", "fold"])
    if lineages is None or splits is None:
        return CheckResult(name, None, "required columns missing (genome_id, lineage_cluster / split, fold)")
    joined = lineages.merge(splits, on="genome_id", how="inner")
    if joined.empty:
        return CheckResult(name, None, "lineages and splits share no genome_id")
    grouped = joined.groupby("lineage_cluster", sort=False)
    n_splits = grouped["split"].nunique(dropna=True)
    n_folds = grouped["fold"].nunique(dropna=True)
    span_splits = sorted(n_splits.index[n_splits > 1].astype(str))
    span_folds = sorted(n_folds.index[n_folds > 1].astype(str))
    detail = f"{joined['lineage_cluster'].nunique()} clusters over {len(joined)} genomes"
    if span_splits or span_folds:
        parts = []
        if span_splits:
            parts.append(f"{len(span_splits)} cluster(s) in two splits: {_examples(span_splits)}")
        if span_folds:
            parts.append(f"{len(span_folds)} cluster(s) in two folds: {_examples(span_folds)}")
        return CheckResult(name, False, f"{detail}; " + "; ".join(parts))
    return CheckResult(name, True, f"{detail}; every cluster is in one split and one fold")


def check_unitig_rows_built_on_train(paths: Paths) -> CheckResult:
    """Every ``role == 'built'`` unitig row is a ``train`` genome."""
    name = "unitig patterns built on train genomes only"
    rows_files = sorted(paths.processed_dir.glob("unitigs_*_rows.parquet")) if paths.processed_dir.is_dir() else []
    if not rows_files:
        return CheckResult(name, None, "no unitigs_<SPECIES>_rows.parquet found")
    splits: pd.DataFrame | None = None
    if paths.splits.is_file():
        splits = _read_columns(paths.splits, ["genome_id", "split"])
    problems: list[str] = []
    not_run: list[str] = []
    summary: list[str] = []
    for rows_file in rows_files:
        columns = set(_parquet_columns(rows_file))
        if not {"genome_id", "split", "role"}.issubset(columns):
            not_run.append(f"{rows_file.name} lacks a role/split/genome_id column")
            continue
        rows = pd.read_parquet(rows_file, engine="pyarrow", columns=["genome_id", "split", "role"])
        role = rows["role"].astype("str")
        unknown_roles = sorted(set(role.dropna()) - ALLOWED_UNITIG_ROLES)
        if unknown_roles:
            problems.append(f"{rows_file.name}: unknown role values {_examples(unknown_roles)}")
        built = rows.loc[role == "built"]
        bad = built.loc[built["split"].astype("str") != "train", "genome_id"]
        if len(bad):
            problems.append(f"{rows_file.name}: {len(bad)} built row(s) not marked train: {_examples(bad)}")
        if splits is not None and not built.empty:
            merged = built[["genome_id"]].merge(splits, on="genome_id", how="left")
            missing = merged.loc[merged["split"].isna(), "genome_id"]
            non_train = merged.loc[merged["split"].notna() & (merged["split"].astype("str") != "train"), "genome_id"]
            if len(missing):
                problems.append(f"{rows_file.name}: {len(missing)} built genome(s) absent from splits.parquet: {_examples(missing)}")
            if len(non_train):
                problems.append(f"{rows_file.name}: {len(non_train)} built genome(s) are not train in splits.parquet: {_examples(non_train)}")
        summary.append(f"{rows_file.name}: {len(built)} built, {int((role == 'queried').sum())} queried")
    if problems:
        return CheckResult(name, False, "; ".join(problems + not_run))
    if not summary:
        return CheckResult(name, None, "; ".join(not_run))
    detail = "; ".join(summary)
    if not_run:
        detail += "; NOT CHECKED: " + "; ".join(not_run)
    if splits is None:
        detail += " (splits.parquet not available for cross-check)"
    return CheckResult(name, True, detail)


def check_biosample_dedup(paths: Paths) -> CheckResult:
    """No biosample maps to two genome_id values in labels.parquet."""
    name = "labels de-duplicated by biosample"
    if not paths.labels.is_file():
        return CheckResult(name, None, "labels.parquet missing")
    labels = _read_columns(paths.labels, ["genome_id", "biosample"])
    if labels is None:
        return CheckResult(name, None, "labels.parquet lacks genome_id/biosample")
    with_sample = labels.dropna(subset=["biosample"])
    per_sample = with_sample.groupby("biosample")["genome_id"].nunique()
    duplicated = sorted(per_sample.index[per_sample > 1].astype(str))
    detail = f"{len(per_sample)} biosamples over {with_sample['genome_id'].nunique()} genomes ({labels['biosample'].isna().sum()} rows without biosample)"
    if duplicated:
        return CheckResult(name, False, f"{detail}; {len(duplicated)} biosample(s) with two genome_ids: {_examples(duplicated)}")
    return CheckResult(name, True, detail)


def _preds_files(paths: Paths) -> list[Path]:
    if not paths.results_dir.is_dir():
        return []
    return sorted(paths.results_dir.glob("preds_*.parquet"))


def check_preds_splits(paths: Paths) -> CheckResult:
    """Every preds row's genome has the matching split in splits.parquet."""
    name = "prediction rows match their split"
    files = _preds_files(paths)
    if not files:
        return CheckResult(name, None, "no results/preds_*.parquet found")
    if not paths.splits.is_file():
        return CheckResult(name, None, "splits.parquet missing")
    splits = _read_columns(paths.splits, ["genome_id", "split"])
    if splits is None:
        return CheckResult(name, None, "splits.parquet lacks genome_id/split")
    expected = {"cv": "train", "test": "test"}
    problems: list[str] = []
    n_rows = 0
    for file in files:
        preds = _read_columns(file, ["genome_id", "split"])
        if preds is None:
            problems.append(f"{file.name}: lacks genome_id/split")
            continue
        n_rows += len(preds)
        merged = preds.merge(splits.rename(columns={"split": "split_assigned"}), on="genome_id", how="left")
        unknown = merged.loc[merged["split_assigned"].isna(), "genome_id"].unique()
        if len(unknown):
            problems.append(f"{file.name}: {len(unknown)} genome(s) not in splits.parquet: {_examples(unknown)}")
        eval_split = merged["split"].astype("str")
        want = eval_split.map(expected)
        mismatch = merged.loc[want.notna() & merged["split_assigned"].notna() & (want != merged["split_assigned"].astype("str"))]
        if len(mismatch):
            problems.append(f"{file.name}: {len(mismatch)} row(s) whose genome is in the wrong split: {_examples(mismatch['genome_id'].unique())}")
    if problems:
        return CheckResult(name, False, "; ".join(problems))
    return CheckResult(name, True, f"{n_rows} rows across {len(files)} preds file(s); cv rows are train genomes, test rows are test genomes")


def check_test_touched_once(paths: Paths) -> CheckResult:
    """Informational: one ``run_id`` per preds file (the test set was scored by one configuration)."""
    name = "test set touched once (informational: one run_id per preds file)"
    files = _preds_files(paths)
    if not files:
        return CheckResult(name, None, "no results/preds_*.parquet found")
    multi: list[str] = []
    summary: list[str] = []
    for file in files:
        preds = _read_columns(file, ["run_id"])
        if preds is None:
            multi.append(f"{file.name}: no run_id column")
            continue
        run_ids = sorted(preds["run_id"].dropna().astype(str).unique())
        if len(run_ids) != 1:
            multi.append(f"{file.name}: {len(run_ids)} run_ids ({_examples(run_ids)})")
        else:
            summary.append(f"{file.name}: {run_ids[0]}")
    if multi:
        return CheckResult(name, False, "; ".join(multi))
    return CheckResult(name, True, "; ".join(summary))


CHECKS: tuple[Callable[[Paths], CheckResult], ...] = (
    check_feature_names,
    check_clusters_within_splits,
    check_unitig_rows_built_on_train,
    check_biosample_dedup,
    check_preds_splits,
    check_test_touched_once,
)
"""Every check in report order."""


def run_checks(paths: Paths) -> list[dict[str, object]]:
    """Run every check and return ``[{check, passed, detail}, ...]``.

    A check that raises is recorded as ``passed=False`` with the error in
    ``detail`` so a broken input cannot pass silently.
    """
    results: list[dict[str, object]] = []
    for check in CHECKS:
        try:
            result = check(paths)
        except Exception as error:  # noqa: BLE001 - the checklist must always complete
            logger.warning("leakage check %s raised: %s", check.__name__, error, exc_info=True)
            result = CheckResult(check.__name__.replace("check_", "").replace("_", " "), False, f"error: {error}")
        status = {True: "PASS", False: "FAIL", None: "NOT RUN"}[result.passed]
        logger.info("leakage check [%s] %s: %s", status, result.check, result.detail)
        results.append(result.to_dict())
    return results


__all__ = [
    "ALLOWED_FEATURE_PREFIXES",
    "CHECKS",
    "CheckResult",
    "FORBIDDEN_FEATURES",
    "check_biosample_dedup",
    "check_clusters_within_splits",
    "check_feature_names",
    "check_preds_splits",
    "check_test_touched_once",
    "check_unitig_rows_built_on_train",
    "classify_features",
    "has_allowed_prefix",
    "is_forbidden_feature",
    "run_checks",
]
