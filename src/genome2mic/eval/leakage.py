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
   rows are test genomes, every genome is in ``splits.parquet``, and every
   ``lolo_<X>`` row is a train genome whose ``splits.lolo_lineage`` is ``X`` (LOLO
   lineages are train clusters; a test cluster would only relabel test rows).
6. **Test set touched once** -- the append-only ``results/test_ledger.csv`` (one row
   per time ``train`` scored the test rows of a pair: ``run_id, created_utc, species,
   drug, n_test_rows, inputs_sha1``) holds one distinct ``(run_id, inputs_sha1)`` per
   species x drug, and every preds file's test rows come from a ``run_id`` recorded
   there. ``run_id`` fingerprints the training parameters and ``splits.parquet``;
   ``inputs_sha1`` fingerprints what else the scoring depended on (labels, features,
   the unitig set, configs, model code; see ``models.train.compute_inputs_sha1``), so
   re-scoring after any of those changed fails even under the same ``run_id``.
   Re-running the identical configuration on identical inputs is allowed. A ledger
   written before ``inputs_sha1`` existed still parses (the fingerprint is then
   unknown: such a row and a fingerprinted row of the same pair count as two). No
   ledger while preds exist is a FAIL: the rule cannot be verified. The ledger proves
   that the test rows of a pair were only ever scored by one configuration on one set
   of inputs; it cannot prove that nobody looked at test metrics before choosing that
   configuration.
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

LOLO_PREFIX = "lolo_"
"""``split`` prefix of leave-one-lineage-out prediction rows (``lolo_<lineage>``)."""

TEST_LEDGER_NAME = "test_ledger.csv"
"""Append-only ledger written by ``train`` under ``results/`` each time it scores test rows."""

TEST_LEDGER_COLUMNS: tuple[str, ...] = ("run_id", "created_utc", "species", "drug", "n_test_rows", "inputs_sha1")
"""Columns of the test ledger (the check needs ``run_id``, ``species`` and ``drug``; ``inputs_sha1``
is absent from ledgers written before it existed and is then treated as unknown)."""

LEDGER_INPUTS_COLUMN = "inputs_sha1"

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


def _text(series: pd.Series) -> pd.Series:
    """``series`` as plain strings with nulls as ``""`` (NA-safe for ``==`` / ``!=`` masks)."""
    return series.astype("str").fillna("")


def _lolo_problems(file_name: str, merged: pd.DataFrame, has_lolo_column: bool) -> tuple[list[str], int]:
    """Problems with the ``lolo_<X>`` rows of one preds file (merged with splits), and their count.

    Every ``lolo_<X>`` row must be a train genome whose ``splits.lolo_lineage`` is ``X``.
    """
    eval_split = _text(merged["split"])
    lolo = eval_split.str.startswith(LOLO_PREFIX).to_numpy(dtype=bool)
    n_lolo = int(lolo.sum())
    if n_lolo == 0:
        return [], 0
    if not has_lolo_column:
        return [f"{file_name}: {n_lolo} lolo row(s) but splits.parquet has no lolo_lineage column to validate them"], n_lolo
    rows = merged.loc[lolo]
    lineage = _text(rows["split"]).str.slice(len(LOLO_PREFIX))
    assigned_lineage = _text(rows["lolo_lineage"])
    assigned_split = _text(rows["split_assigned"])
    known = assigned_split != ""
    problems: list[str] = []
    not_train = rows.loc[known & (assigned_split != "train")]
    if len(not_train):
        problems.append(
            f"{file_name}: {len(not_train)} lolo row(s) whose genome is not a train genome "
            f"(LOLO lineages must be train clusters): {_examples(not_train['genome_id'].unique())}"
        )
    wrong = rows.loc[known & (assigned_lineage != lineage)]
    if len(wrong):
        pairs = sorted({f"{g} in {s} (splits.lolo_lineage={a or 'null'})" for g, s, a in zip(wrong["genome_id"], _text(wrong["split"]), _text(wrong["lolo_lineage"]), strict=True)})
        problems.append(f"{file_name}: {len(wrong)} lolo row(s) whose genome is not marked with that lineage: {_examples(pairs)}")
    return problems, n_lolo


def check_preds_splits(paths: Paths) -> CheckResult:
    """Every preds row's genome has the matching split in splits.parquet.

    ``cv`` rows must be train genomes, ``test`` rows test genomes, and ``lolo_<X>``
    rows train genomes with ``splits.lolo_lineage == X``.
    """
    name = "prediction rows match their split"
    files = _preds_files(paths)
    if not files:
        return CheckResult(name, None, "no results/preds_*.parquet found")
    if not paths.splits.is_file():
        return CheckResult(name, None, "splits.parquet missing")
    has_lolo_column = "lolo_lineage" in _parquet_columns(paths.splits)
    splits = _read_columns(paths.splits, ["genome_id", "split", *(["lolo_lineage"] if has_lolo_column else [])])
    if splits is None:
        return CheckResult(name, None, "splits.parquet lacks genome_id/split")
    expected = {"cv": "train", "test": "test"}
    problems: list[str] = []
    n_rows = 0
    n_lolo_total = 0
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
        lolo_problems, n_lolo = _lolo_problems(file.name, merged, has_lolo_column)
        problems.extend(lolo_problems)
        n_lolo_total += n_lolo
    if problems:
        return CheckResult(name, False, "; ".join(problems))
    return CheckResult(
        name,
        True,
        f"{n_rows} rows across {len(files)} preds file(s); cv rows are train genomes, test rows are test genomes, "
        f"{n_lolo_total} lolo row(s) are train genomes of their held-out lineage",
    )


def ledger_file(paths: Paths) -> Path:
    """``<root>/results/test_ledger.csv`` (written by ``train``; read here directly)."""
    return paths.results_dir / TEST_LEDGER_NAME


def _preds_test_run_ids(file: Path) -> dict[tuple[str, str], set[str]] | None:
    """``{(species, drug): {run_id, ...}}`` over the ``split == 'test'`` rows of one preds file."""
    preds = _read_columns(file, ["species", "drug", "split", "run_id"])
    if preds is None:
        return None
    test = preds.loc[_text(preds["split"]) == "test"]
    out: dict[tuple[str, str], set[str]] = {}
    for species, drug, run_id in zip(_text(test["species"]), _text(test["drug"]), _text(test["run_id"]), strict=True):
        out.setdefault((species, drug), set()).add(run_id)
    return out


def _scoring_label(run_id: str, inputs_sha1: str) -> str:
    """``run_id/inputs_sha1`` for messages; a missing fingerprint is spelled out."""
    return f"{run_id}/{inputs_sha1}" if inputs_sha1 else f"{run_id}/(no inputs_sha1)"


def check_test_touched_once(paths: Paths) -> CheckResult:
    """One ``(run_id, inputs_sha1)`` per species x drug in ``results/test_ledger.csv``, matching the preds files.

    FAIL when a pair was scored on the test set by two configurations or on two sets
    of inputs (more than one distinct ``(run_id, inputs_sha1)``; names the pair and
    each ``run_id/inputs_sha1``), when the ledger is absent although preds exist ("no
    test ledger: cannot verify"), when it lacks ``run_id``/``species``/``drug``, or
    when a preds file's test rows carry a ``run_id`` the ledger never recorded for
    that pair. A ledger without the ``inputs_sha1`` column (written before it
    existed) parses with every fingerprint empty; an empty fingerprint is "unknown"
    and is distinct from any recorded one. NOT RUN when there is neither a ledger nor
    any preds file (nothing was scored).
    """
    name = "test set touched once (one run_id and inputs_sha1 per species x drug in results/test_ledger.csv)"
    ledger_path = ledger_file(paths)
    files = _preds_files(paths)
    if not ledger_path.is_file():
        if not files:
            return CheckResult(name, None, f"no {TEST_LEDGER_NAME} and no results/preds_*.parquet: the test set was never scored")
        scored = [_preds_test_run_ids(f) for f in files]
        if all(s is not None and not s for s in scored):
            # A cv-only run (train --cv-only): no preds file holds a single test row.
            return CheckResult(
                name, True,
                f"test set not scored: no split == 'test' rows in {len(files)} preds file(s) and no {TEST_LEDGER_NAME}",
            )
        return CheckResult(name, False, f"no test ledger: cannot verify (expected {ledger_path})")
    ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    missing = [c for c in ("run_id", "species", "drug") if c not in ledger.columns]
    if missing:
        return CheckResult(name, False, f"{TEST_LEDGER_NAME} lacks columns {missing}: cannot verify")
    legacy = LEDGER_INPUTS_COLUMN not in ledger.columns
    if legacy:
        ledger = ledger.assign(**{LEDGER_INPUTS_COLUMN: ""})
    ledger = ledger.loc[ledger["run_id"].str.strip() != ""]
    recorded: dict[tuple[str, str], set[str]] = {}
    scorings: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for species, drug, run_id, inputs in zip(
        ledger["species"], ledger["drug"], ledger["run_id"], ledger[LEDGER_INPUTS_COLUMN], strict=True
    ):
        pair = (species.strip(), drug.strip())
        recorded.setdefault(pair, set()).add(run_id.strip())
        scorings.setdefault(pair, set()).add((run_id.strip(), str(inputs).strip()))

    problems: list[str] = []
    multi = {pair: keys for pair, keys in sorted(scorings.items()) if len(keys) > 1}
    for (species, drug), keys in multi.items():
        run_ids = {r for r, _ in keys}
        what = f"{len(run_ids)} run_ids" if len(run_ids) > 1 else f"one run_id on {len(keys)} different inputs"
        problems.append(
            f"{species} x {drug} scored on the test set by {what} "
            f"({', '.join(_scoring_label(r, i) for r, i in sorted(keys))})"
        )
    for file in files:
        by_pair = _preds_test_run_ids(file)
        if by_pair is None:
            problems.append(f"{file.name}: lacks species/drug/split/run_id")
            continue
        for (species, drug), ids in sorted(by_pair.items()):
            unrecorded = sorted(ids - recorded.get((species, drug), set()))
            if unrecorded:
                problems.append(f"{file.name}: test rows of {species} x {drug} from run_id(s) {', '.join(unrecorded)} not in the test ledger")
    if problems:
        if any("(no inputs_sha1)" in problem for problem in problems):
            problems.append(
                "(no inputs_sha1) = a ledger row written before input fingerprints were recorded: "
                "it cannot show the inputs were unchanged"
            )
        return CheckResult(name, False, "; ".join(problems))
    per_pair = [f"{s} x {d}: {_scoring_label(*next(iter(keys)))}" for (s, d), keys in sorted(scorings.items())]
    note = " (legacy ledger without inputs_sha1: input changes under the same run_id cannot be detected)" if legacy else ""
    return CheckResult(
        name,
        True,
        f"{TEST_LEDGER_NAME}: {len(ledger)} scoring(s) of {len(recorded)} pair(s), one run_id and inputs_sha1 each"
        f" ({_examples(per_pair) if per_pair else 'none'}); {len(files)} preds file(s) match it{note}",
    )


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
    "LEDGER_INPUTS_COLUMN",
    "LOLO_PREFIX",
    "TEST_LEDGER_COLUMNS",
    "TEST_LEDGER_NAME",
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
    "ledger_file",
    "run_checks",
]
