"""Stage 7 -- frozen train/test splits by lineage cluster (``splits.parquet``).

Construction (``DATA_CONTRACT.md`` stage 7, per species, seeded):

1. Reserve the ``n_lolo`` (default two) largest clusters for leave-one-lineage-out
   runs. Reserved clusters are never test candidates, so they always stay
   ``train`` and a LOLO fit (train minus the lineage) is a genuine refit rather
   than a relabelled copy of the test evaluation.
2. Shuffle the remaining lineage clusters and assign *whole clusters* to ``test``
   until 15-20 % of the species' genomes (all of them, reserved clusters
   included) are held out (never splitting a cluster).
3. For every kept species x drug pair (``pairs_kept.csv``) verify the test set
   contains both a resistant (``sir == 'R'``) and a susceptible (``sir == 'S'``)
   genome. If not, swap clusters between train and test (bounded number of
   attempts, every swap logged; reserved clusters are never swapped in). If a
   class exists nowhere outside the test set and the reserved clusters, the check
   is logged as unfixable and the run continues with a warning.
4. ``GroupKFold(n_splits=5)`` on the remaining genomes, grouped by cluster, gives
   ``fold`` (0-4; null for test rows, stored as pandas ``Int64``).
5. External sets, defined on test rows only and kept disjoint:
   ``country_holdout`` = the most common test-set country,
   ``time_holdout`` = the latest test-set year among rows not already in the
   country hold-out.
6. ``lolo_lineage``: every row of a reserved cluster carries its own cluster id
   (train rows only, by construction).

Invariants (raise :class:`~genome2mic.errors.ContractViolation`): no cluster in
two splits; no cluster in two folds; ``fold`` is null exactly on test rows;
external sets only on test rows; ``lolo_lineage`` only on train rows and equal to
the row's own cluster.

Text columns may arrive as pd.NA-based ``string`` dtype (``labels.parquet`` as
ingest writes it); every comparison on them goes through :func:`_true`, which
treats a missing value as ``False`` instead of letting ``pd.NA`` reach a mask.

Freezing: :func:`run` refuses to overwrite an existing ``splits.parquet`` unless
``force=True`` (``CLAUDE.md`` rule 7 -- changing splits invalidates every result
produced before the change).

Nothing produced here is a feature. ``split``, ``fold``, ``external_set`` and
``lolo_lineage`` exist only to select rows for training and evaluation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.io import read_csv, read_parquet, write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

__all__ = [
    "STAGE",
    "SPLIT_TRAIN",
    "SPLIT_TEST",
    "EXTERNAL_COUNTRY",
    "EXTERNAL_TIME",
    "EXTERNAL_SOURCE",
    "EXTERNAL_SETS",
    "COLUMNS",
    "DEFAULT_SEED",
    "DEFAULT_TEST_RANGE",
    "DEFAULT_N_FOLDS",
    "DEFAULT_MAX_SWAPS",
    "DEFAULT_N_LOLO",
    "species_rng",
    "greedy_test_clusters",
    "missing_classes",
    "cluster_class_presence",
    "missing_from_presence",
    "repair_test_clusters",
    "assign_folds",
    "genome_metadata",
    "mark_external_sets",
    "lolo_clusters",
    "build_splits",
    "check_invariants",
    "run",
]

STAGE = "splits"
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
EXTERNAL_COUNTRY = "country_holdout"
EXTERNAL_TIME = "time_holdout"
EXTERNAL_SOURCE = "source_holdout"
EXTERNAL_SETS: tuple[str, ...] = (EXTERNAL_COUNTRY, EXTERNAL_TIME, EXTERNAL_SOURCE)
COLUMNS: tuple[str, ...] = ("genome_id", "species", "split", "fold", "external_set", "lolo_lineage")

DEFAULT_SEED = 7
DEFAULT_TEST_RANGE: tuple[float, float] = (0.15, 0.20)
DEFAULT_N_FOLDS = 5
DEFAULT_MAX_SWAPS = 100
DEFAULT_N_LOLO = 2

RESISTANT = "R"
SUSCEPTIBLE = "S"
_CHECKED_CLASSES: tuple[str, ...] = (RESISTANT, SUSCEPTIBLE)

_LABEL_COLUMNS_REQUIRED: tuple[str, ...] = ("genome_id", "species", "drug", "sir")
_LABEL_COLUMNS_OPTIONAL: tuple[str, ...] = ("country", "year")
_PAIRS_COLUMNS_REQUIRED: tuple[str, ...] = ("species", "drug")


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #
def _true(mask: pd.Series) -> np.ndarray:
    """Plain ``bool`` ndarray from a comparison result; missing (``pd.NA``/NaN) -> ``False``.

    Comparing a pd.NA-based ``string`` column yields a nullable boolean
    (``boolean`` / ``bool[pyarrow]``) whose ``.to_numpy()`` is an object array
    holding ``pd.NA``; ``.loc`` refuses to mask with it. Every mask built from a
    text column in this module goes through here.
    """
    return np.asarray(mask.to_numpy(dtype=bool, na_value=False), dtype=bool)


# --------------------------------------------------------------------------- #
# Randomness
# --------------------------------------------------------------------------- #
def species_rng(seed: int, species: str) -> np.random.Generator:
    """A generator seeded by ``(seed, species)`` so species are independent.

    Adding or removing one species therefore never changes another species'
    split for the same seed.
    """
    return np.random.default_rng([int(seed), int.from_bytes(str(species).encode("utf-8"), "big")])


# --------------------------------------------------------------------------- #
# Test-set construction
# --------------------------------------------------------------------------- #
def greedy_test_clusters(
    sizes: Mapping[str, int],
    rng: np.random.Generator,
    lo: float = DEFAULT_TEST_RANGE[0],
    hi: float = DEFAULT_TEST_RANGE[1],
    *,
    reserved: Collection[str] = (),
) -> set[str]:
    """Pick whole clusters for ``test`` until ``lo..hi`` of the genomes are held out.

    Clusters are visited in a seeded random order; a cluster is taken when it
    fits under the ``hi`` bound and the walk stops once ``lo`` is reached. If the
    walk ends below ``lo`` (large clusters left no room), one chosen cluster is
    swapped for an unchosen one when that lands inside the bounds; otherwise the
    shortfall is logged and the caller accepts the smaller test set.

    Args:
        sizes: ``cluster_id -> number of genomes``.
        rng: Seeded generator.
        lo, hi: Target range as fractions of the species' genome count.
        reserved: Clusters that may never enter ``test`` (the LOLO clusters).
            They still count towards the species' genome total, and the seeded
            visiting order of the other clusters is the same as without them.
    """
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"need 0 <= lo <= hi <= 1; got lo={lo}, hi={hi}")
    names = sorted(sizes)
    n_total = sum(int(sizes[c]) for c in names)
    if n_total == 0:
        return set()
    lo_n, hi_n = lo * n_total, hi * n_total
    blocked = set(reserved)
    order = [c for c in (names[i] for i in rng.permutation(len(names))) if c not in blocked]

    chosen: set[str] = set()
    n_test = 0
    for cluster in order:
        if n_test >= lo_n:
            break
        if n_test + sizes[cluster] <= hi_n:
            chosen.add(cluster)
            n_test += sizes[cluster]

    if n_test < lo_n:
        # One-for-one swap that lands inside the bounds, in seeded order.
        unchosen = [c for c in order if c not in chosen]
        for out in [c for c in order if c in chosen]:
            for inc in unchosen:
                candidate = n_test - sizes[out] + sizes[inc]
                if lo_n <= candidate <= hi_n:
                    chosen.discard(out)
                    chosen.add(inc)
                    n_test = candidate
                    break
            if n_test >= lo_n:
                break
    if n_test < lo_n:
        logger.info(
            "[%s] greedy pass ended with %d of %d genomes (%.1f%%) in test, below the %.0f%% target; "
            "the repair step will try cluster replacements",
            STAGE,
            n_test,
            n_total,
            100.0 * n_test / n_total,
            100.0 * lo,
        )
    return chosen


def missing_classes(labels_sp: pd.DataFrame, genome_ids: set[str], drugs: Sequence[str]) -> list[tuple[str, str]]:
    """``(drug, class)`` pairs for which no genome in ``genome_ids`` carries that ``sir``.

    ``class`` is ``'R'`` or ``'S'`` (as reported in ``labels.parquet``; ``I`` and
    null count for neither). An empty list means every drug has both classes.
    """
    if labels_sp.empty or not drugs:
        return [(d, c) for d in drugs for c in _CHECKED_CLASSES]
    sub = labels_sp[_true(labels_sp["genome_id"].isin(genome_ids))]
    out: list[tuple[str, str]] = []
    for drug in drugs:
        present = set(sub.loc[_true(sub["drug"] == drug), "sir"].dropna().astype(str))
        out.extend((drug, cls) for cls in _CHECKED_CLASSES if cls not in present)
    return out


def cluster_class_presence(labels_sp: pd.DataFrame, cluster_of: Mapping[str, str]) -> dict[str, frozenset[tuple[str, str]]]:
    """``cluster -> {(drug, class), ...}`` for the checked classes its genomes carry.

    Computed once per species so the repair loop can score a candidate test set
    by a set union over its clusters instead of re-scanning the label table
    (:func:`missing_from_presence` on these sets equals :func:`missing_classes`
    on the clusters' genomes). Label rows of genomes outside ``cluster_of`` and
    null ``sir`` values are ignored, exactly as in :func:`missing_classes`.
    """
    if labels_sp.empty:
        return {}
    sir = labels_sp["sir"]
    checked = _true(sir.isin(list(_CHECKED_CLASSES)))
    sub = labels_sp.loc[checked, ["genome_id", "drug", "sir"]]
    presence: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for gid, drug, cls in zip(sub["genome_id"].astype(str), sub["drug"].astype(object), sub["sir"].astype(str)):
        cluster = cluster_of.get(gid)
        if cluster is None or drug is None or pd.isna(drug):
            continue
        presence[cluster].add((str(drug), cls))
    return {c: frozenset(v) for c, v in presence.items()}


def missing_from_presence(
    presence: Mapping[str, frozenset[tuple[str, str]]], clusters: Collection[str], drugs: Sequence[str]
) -> list[tuple[str, str]]:
    """:func:`missing_classes` for the genomes of ``clusters``, from :func:`cluster_class_presence`.

    Same order as :func:`missing_classes`: drugs as given, then ``R`` before ``S``.
    """
    present: set[tuple[str, str]] = set()
    for cluster in clusters:
        present |= presence.get(cluster, frozenset())
    return [(d, c) for d in drugs for c in _CHECKED_CLASSES if (d, c) not in present]


def repair_test_clusters(
    test: set[str],
    cluster_of: Mapping[str, str],
    labels_sp: pd.DataFrame,
    drugs: Sequence[str],
    rng: np.random.Generator,
    *,
    lo: float = DEFAULT_TEST_RANGE[0],
    hi: float = DEFAULT_TEST_RANGE[1],
    max_swaps: int = DEFAULT_MAX_SWAPS,
    max_candidates: int = 25,
    species: str | None = None,
    reserved: Collection[str] = (),
) -> tuple[set[str], list[tuple[str, str]]]:
    """Swap clusters between train and test until every kept drug has R and S in test.

    Hill climbing on the pair ``(number of failing checks, size violation)`` where
    the size violation is how many genomes the test set lies outside ``lo..hi``.
    While a check fails, an attempt picks one failing ``(drug, class)``, chooses
    (seeded) a train cluster carrying the missing class and scores every way of
    bringing it in: add only, or add plus removing one current test cluster.
    Once every check passes but the size is outside the range, attempts score
    single removals/additions and one-for-one replacements against a seeded
    sample of at most ``max_candidates`` train clusters. The best option is kept
    only when it strictly improves the pair, so the loop terminates; attempts
    are also bounded by ``max_swaps``. Every kept move is logged at INFO; a final
    state outside the size range or with failing checks is logged at WARNING.
    The size bounds are soft; the class check is what matters.

    Clusters in ``reserved`` (the LOLO clusters) are never brought into test;
    they still count towards the species' genome total. Which ``(drug, class)``
    each cluster carries is computed once (:func:`cluster_class_presence`), so
    scoring an option costs a set union over its clusters, not a label scan.

    Returns:
        ``(test_clusters, still_failing)``.
    """
    members: dict[str, set[str]] = defaultdict(set)
    for gid, cluster in cluster_of.items():
        members[cluster].add(gid)
    sizes = {c: len(g) for c, g in members.items()}
    n_total = sum(sizes.values())
    lo_n, hi_n = lo * n_total, hi * n_total
    tag = f"{species}: " if species else ""
    blocked = set(reserved)
    presence = cluster_class_presence(labels_sp, cluster_of)
    carriers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cluster, classes in presence.items():
        for check in classes:
            carriers[check].add(cluster)

    def size_of(clusters: set[str]) -> int:
        return sum(sizes[c] for c in clusters)

    def evaluate(clusters: set[str]) -> tuple[list[tuple[str, str]], float]:
        n = size_of(clusters)
        return missing_from_presence(presence, clusters, drugs), max(0.0, lo_n - n) + max(0.0, n - hi_n)

    test = set(test)
    if test & blocked:
        raise ValueError(f"reserved clusters cannot start in test: {sorted(test & blocked)[:5]}")
    failing, violation = evaluate(test)
    unfixable: set[tuple[str, str]] = set()
    attempts = n_swaps = 0
    while attempts < max_swaps:
        open_checks = [f for f in failing if f not in unfixable]
        if not open_checks and violation == 0:
            break
        attempts += 1
        options: list[tuple[str | None, str | None, set[str]]]
        if open_checks:
            drug, cls = open_checks[int(rng.integers(len(open_checks)))]
            candidates = sorted(carriers.get((drug, cls), set()) - test - blocked)
            if not candidates:
                unfixable.add((drug, cls))
                logger.warning(
                    "[%s] %sno cluster outside the test set carries %s=%s%s; cannot repair this check",
                    STAGE,
                    tag,
                    drug,
                    cls,
                    " (LOLO-reserved clusters excluded)" if blocked else "",
                )
                continue
            add = candidates[int(rng.integers(len(candidates)))]
            options = [(add, None, test | {add})] + [(add, c, (test - {c}) | {add}) for c in sorted(test)]
            reason = f"to restore {drug}={cls} in test"
        else:
            others = sorted(set(sizes) - test - blocked)
            sample = [others[i] for i in rng.permutation(len(others))[: max(1, int(max_candidates))]]
            current_test = sorted(test)
            if size_of(test) > hi_n:
                options = [(None, c, test - {c}) for c in current_test]
                options += [(d, c, (test - {c}) | {d}) for c in current_test for d in sample if sizes[d] < sizes[c]]
            else:
                options = [(d, None, test | {d}) for d in sample]
                options += [(d, c, (test - {c}) | {d}) for c in current_test for d in sample if sizes[d] > sizes[c]]
            reason = f"to bring the test set inside the {100 * lo:.0f}-{100 * hi:.0f}% range"
            if not options:
                break

        scored = []
        for add, remove, trial in options:
            trial_failing, trial_violation = evaluate(trial)
            scored.append((len(trial_failing), trial_violation, float(rng.random()), add, remove, trial, trial_failing))
        scored.sort(key=lambda s: s[:3])
        n_fail, trial_violation, _, add, remove, trial, trial_failing = scored[0]
        if (n_fail, trial_violation) < (len(failing), violation):
            test, failing, violation = trial, trial_failing, trial_violation
            n_swaps += 1
            logger.info(
                "[%s] %sswap %d: %s%s %s (%d check(s) still failing, test size %d)",
                STAGE,
                tag,
                n_swaps,
                f"+{add}" if add is not None else "",
                f" -{remove}" if remove is not None else "",
                reason,
                len(failing),
                size_of(test),
            )
        elif not open_checks:
            break  # size outside the range and no move keeps every class; reported below
        else:
            logger.debug("[%s] %sno improving move for %s (best would leave %d failing)", STAGE, tag, reason, n_fail)

    if violation > 0:
        logger.warning(
            "[%s] %stest set holds %d of %d genomes (%.1f%%), outside the %.0f-%.0f%% target; no whole-cluster "
            "move keeps every required class, keeping it as is",
            STAGE,
            tag,
            size_of(test),
            n_total,
            100.0 * size_of(test) / max(n_total, 1),
            100 * lo,
            100 * hi,
        )
    if failing:
        logger.warning(
            "[%s] %stest-set R/S check: %d check(s) still failing after %d attempt(s)/%d swap(s): %s",
            STAGE,
            tag,
            len(failing),
            attempts,
            n_swaps,
            ", ".join(f"{d}:{c}" for d, c in failing),
        )
    else:
        logger.info(
            "[%s] %stest-set R/S check passed for %d drug(s) after %d attempt(s)/%d swap(s)",
            STAGE,
            tag,
            len(drugs),
            attempts,
            n_swaps,
        )
    return test, list(failing)


# --------------------------------------------------------------------------- #
# Folds, external sets, LOLO
# --------------------------------------------------------------------------- #
def assign_folds(groups: Sequence[str], n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """``GroupKFold`` fold numbers for train rows grouped by cluster.

    Uses ``min(n_folds, n_clusters)`` splits; with fewer than two clusters every
    row goes to fold 0 and a warning is logged (no cross-validation possible).
    """
    labels = np.asarray(list(groups), dtype=object)
    folds = np.zeros(labels.shape[0], dtype=np.int64)
    if labels.shape[0] == 0:
        return folds
    n_groups = len(set(labels.tolist()))
    n_splits = min(int(n_folds), n_groups)
    if n_splits < 2:
        logger.warning("[%s] only %d cluster(s) in train; every row goes to fold 0", STAGE, n_groups)
        return folds
    if n_splits < n_folds:
        logger.warning("[%s] only %d train cluster(s); using %d folds instead of %d", STAGE, n_groups, n_splits, n_folds)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    for fold, (_, idx) in enumerate(splitter.split(np.zeros((labels.shape[0], 1)), groups=labels)):
        folds[idx] = fold
    return folds


def genome_metadata(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per genome: ``genome_id, country, year`` (evaluation metadata only).

    ``country`` is the most common non-null value across the genome's label rows
    (ties alphabetical); ``year`` is the latest non-null year. Missing stays null.
    """
    if labels.empty:
        return pd.DataFrame(
            {
                "genome_id": pd.array([], dtype="str"),
                "country": pd.array([], dtype="str"),
                "year": pd.array([], dtype="Int64"),
            }
        )
    rows: list[tuple[str, str | None, int | None]] = []
    for gid, grp in labels.groupby("genome_id", sort=True):
        countries = grp["country"].dropna().astype(str) if "country" in grp else pd.Series([], dtype="str")
        country = None
        if len(countries):
            counts = countries.value_counts()
            country = sorted(counts[counts == counts.max()].index)[0]
        year: int | None = None
        if "year" in grp:
            years = pd.to_numeric(grp["year"], errors="coerce").dropna()
            if len(years):
                year = int(years.max())
        rows.append((str(gid), country, year))
    return pd.DataFrame(
        {
            "genome_id": pd.array([r[0] for r in rows], dtype="str"),
            "country": pd.array([r[1] for r in rows], dtype="str"),
            "year": pd.array([r[2] for r in rows], dtype="Int64"),
        }
    )


def mark_external_sets(test_meta: pd.DataFrame, species: str | None = None) -> pd.Series:
    """External hold-out label per **test** genome (index ``genome_id``, ``str`` dtype).

    ``country_holdout``: rows whose country is the most common test-set country
    (ties alphabetical). ``time_holdout``: rows not already in the country
    hold-out whose year is the latest among those rows (so the two sets are
    disjoint). Null everywhere else, and for every genome when the metadata is
    entirely missing.
    """
    ids = test_meta["genome_id"].astype(str).tolist()
    ext = pd.Series(pd.array([None] * len(ids), dtype="str"), index=ids, name="external_set")
    tag = f"{species}: " if species else ""
    if not ids:
        return ext

    country = pd.Series(test_meta["country"].to_numpy(dtype=object), index=ids)
    counts = country.dropna().astype(str).value_counts()
    if len(counts):
        top = sorted(counts[counts == counts.max()].index)[0]
        mask = _true(country.astype(object) == top)
        ext[mask] = EXTERNAL_COUNTRY
        logger.info("[%s] %s%s = %r on %d test genome(s)", STAGE, tag, EXTERNAL_COUNTRY, top, int(mask.sum()))
    else:
        logger.info("[%s] %sno country metadata in test; %s not assigned", STAGE, tag, EXTERNAL_COUNTRY)

    year = pd.to_numeric(pd.Series(test_meta["year"].to_numpy(dtype=object), index=ids), errors="coerce")
    free = ext.isna().to_numpy(dtype=bool)
    free_years = year[free].dropna()
    if len(free_years):
        latest = int(free_years.max())
        mask = free & _true(year == latest)
        ext[mask] = EXTERNAL_TIME
        logger.info("[%s] %s%s = %d on %d test genome(s)", STAGE, tag, EXTERNAL_TIME, latest, int(mask.sum()))
    else:
        logger.info("[%s] %sno year metadata outside the country hold-out; %s not assigned", STAGE, tag, EXTERNAL_TIME)
    return ext


def lolo_clusters(lineages_sp: pd.DataFrame, n_lolo: int = DEFAULT_N_LOLO) -> list[str]:
    """The ``n_lolo`` largest clusters of one species (size desc, then id).

    :func:`build_splits` reserves these *before* choosing test clusters, so they
    are always ``train`` and their LOLO fit leaves the lineage out of a training
    set that would otherwise include it.
    """
    counts = lineages_sp["lineage_cluster"].astype(str).value_counts()
    ordered = sorted(counts.index, key=lambda c: (-int(counts[c]), c))
    return ordered[: max(0, int(n_lolo))]


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_splits(
    lineages: pd.DataFrame,
    labels: pd.DataFrame,
    pairs_kept: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    test_range: tuple[float, float] = DEFAULT_TEST_RANGE,
    n_folds: int = DEFAULT_N_FOLDS,
    max_swaps: int = DEFAULT_MAX_SWAPS,
    n_lolo: int = DEFAULT_N_LOLO,
    droplog: DropLog | None = None,
) -> pd.DataFrame:
    """Pure (no I/O) construction of the splits table; see the module docstring.

    Args:
        lineages: Stage 6 table (``genome_id, species, lineage_cluster, ...``).
        labels: Stage 2 table; needs ``genome_id, species, drug, sir`` and, for
            external sets, ``country, year`` (missing -> all null, logged).
        pairs_kept: ``species, drug`` pairs passing the inclusion rule.
        seed: Base seed; each species derives its own generator from it.
        test_range: ``(lo, hi)`` fraction of genomes to hold out per species.
        n_folds: ``GroupKFold`` splits on train rows.
        max_swaps: Bound on cluster-swap attempts for the R/S check.
        n_lolo: Largest clusters per species to reserve for leave-one-lineage-out.
            They are excluded from test selection (always ``train``) and carry
            their cluster id in ``lolo_lineage``.
        droplog: Where filter counts go (a fresh ``DropLog("splits")`` if ``None``).

    Raises:
        ContractViolation: on malformed inputs or a broken invariant.
    """
    log = droplog if droplog is not None else DropLog(STAGE)
    lo, hi = float(test_range[0]), float(test_range[1])
    _validate_inputs(lineages, labels, pairs_kept)
    labels = _ensure_optional_label_columns(labels)

    lineage_ids = set(lineages["genome_id"].astype(str))
    labels_in = log.keep_where(
        labels,
        _true(labels["genome_id"].astype(str).isin(lineage_ids)),
        "label_rows_for_genomes_without_lineage",
        detail="genome failed QC or has no lineage row; excluded from the test-set R/S check",
    )
    species_present = set(lineages["species"].astype(str))
    pairs = log.keep_where(
        pairs_kept,
        _true(pairs_kept["species"].astype(str).isin(species_present)),
        "kept_pairs_for_species_without_lineages",
    )
    meta = genome_metadata(labels_in).set_index("genome_id")

    frames: list[pd.DataFrame] = []
    for species, lin_sp in lineages.groupby("species", sort=True):
        species_key = str(species)
        rng = species_rng(seed, species_key)
        gids = sorted(lin_sp["genome_id"].astype(str).tolist())
        cluster_of = dict(zip(lin_sp["genome_id"].astype(str), lin_sp["lineage_cluster"].astype(str)))
        sizes = {c: int(n) for c, n in lin_sp["lineage_cluster"].astype(str).value_counts().items()}
        drugs = sorted(set(pairs.loc[_true(pairs["species"] == species_key), "drug"].dropna().astype(str)))
        labels_sp = labels_in[_true(labels_in["species"] == species_key)]

        # Decision 1 (LOLO): reserve the largest clusters first; they never enter test.
        lolo = set(lolo_clusters(lin_sp, n_lolo))
        test_clusters = greedy_test_clusters(sizes, rng, lo, hi, reserved=lolo)
        test_clusters, still_failing = repair_test_clusters(
            test_clusters, cluster_of, labels_sp, drugs, rng, lo=lo, hi=hi, max_swaps=max_swaps, species=species_key,
            reserved=lolo,
        )
        test_genomes = {g for g, c in cluster_of.items() if c in test_clusters}
        if not test_genomes:
            logger.warning(
                "[%s] %s: no test cluster could be chosen (%d cluster(s), %d reserved for LOLO); "
                "this species has no held-out test set",
                STAGE, species_key, len(sizes), len(lolo),
            )

        split = [SPLIT_TEST if g in test_genomes else SPLIT_TRAIN for g in gids]
        train_positions = [i for i, s in enumerate(split) if s == SPLIT_TRAIN]
        train_folds = assign_folds([cluster_of[gids[i]] for i in train_positions], n_folds, rng)
        fold_values: list[int | None] = [None] * len(gids)
        for pos, fold in zip(train_positions, train_folds):
            fold_values[pos] = int(fold)

        test_ids = [g for g in gids if g in test_genomes]
        test_meta = meta.reindex(test_ids).reset_index().rename(columns={"index": "genome_id"})
        test_meta["genome_id"] = pd.array(test_ids, dtype="str")
        ext = mark_external_sets(test_meta, species=species_key)
        external = [ext.get(g) if g in test_genomes else None for g in gids]
        external = [None if (v is None or pd.isna(v)) else str(v) for v in external]

        lolo_values = [cluster_of[g] if cluster_of[g] in lolo else None for g in gids]

        frames.append(
            pd.DataFrame(
                {
                    "genome_id": pd.array(gids, dtype="str"),
                    "species": pd.array([species_key] * len(gids), dtype="str"),
                    "split": pd.array(split, dtype="str"),
                    "fold": pd.array(fold_values, dtype="Int64"),
                    "external_set": pd.array(external, dtype="str"),
                    "lolo_lineage": pd.array(lolo_values, dtype="str"),
                },
                columns=list(COLUMNS),
            )
        )
        n_test = len(test_ids)
        logger.info(
            "[%s] %s: %d genomes, %d clusters -> test %d genomes (%.1f%%) in %d clusters, "
            "train %d genomes in %d clusters over %d fold(s); lolo clusters (train, reserved) %s; R/S checks failing: %d",
            STAGE,
            species_key,
            len(gids),
            len(sizes),
            n_test,
            100.0 * n_test / max(len(gids), 1),
            len(test_clusters),
            len(gids) - n_test,
            len(sizes) - len(test_clusters),
            len(set(train_folds.tolist())) if len(train_folds) else 0,
            sorted(lolo),
            len(still_failing),
        )

    if frames:
        splits = pd.concat(frames, ignore_index=True)
    else:
        splits = pd.DataFrame(
            {
                "genome_id": pd.array([], dtype="str"),
                "species": pd.array([], dtype="str"),
                "split": pd.array([], dtype="str"),
                "fold": pd.array([], dtype="Int64"),
                "external_set": pd.array([], dtype="str"),
                "lolo_lineage": pd.array([], dtype="str"),
            },
            columns=list(COLUMNS),
        )
    check_invariants(splits, lineages, n_folds=n_folds)
    return splits


def _validate_inputs(lineages: pd.DataFrame, labels: pd.DataFrame, pairs_kept: pd.DataFrame) -> None:
    for col in ("genome_id", "species", "lineage_cluster"):
        if col not in lineages.columns:
            raise ContractViolation(f"lineages lacks required column {col!r}", stage=STAGE)
    if lineages["genome_id"].duplicated().any():
        raise ContractViolation("duplicate genome_id in lineages", stage=STAGE)
    if lineages[["genome_id", "species", "lineage_cluster"]].isna().any().any():
        raise ContractViolation("null genome_id/species/lineage_cluster in lineages", stage=STAGE)
    per_cluster_species = lineages.groupby("lineage_cluster")["species"].nunique()
    if (per_cluster_species > 1).any():
        raise ContractViolation(
            f"lineage clusters spanning species: {per_cluster_species[per_cluster_species > 1].index.tolist()[:5]}",
            stage=STAGE,
        )
    for col in _LABEL_COLUMNS_REQUIRED:
        if col not in labels.columns:
            raise ContractViolation(f"labels lacks required column {col!r}", stage=STAGE)
    for col in _PAIRS_COLUMNS_REQUIRED:
        if col not in pairs_kept.columns:
            raise ContractViolation(f"pairs_kept lacks required column {col!r}", stage=STAGE)


def _ensure_optional_label_columns(labels: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _LABEL_COLUMNS_OPTIONAL if c not in labels.columns]
    if not missing:
        return labels
    logger.warning("[%s] labels lack %s; external hold-out sets depending on them stay null", STAGE, missing)
    out = labels.copy()
    for col in missing:
        out[col] = pd.array([None] * len(out), dtype="str" if col == "country" else "Int64")
    return out


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #
def check_invariants(splits: pd.DataFrame, lineages: pd.DataFrame, n_folds: int = DEFAULT_N_FOLDS) -> None:
    """Raise :class:`ContractViolation` if the splits table breaks the contract.

    Checks: exact columns; ``genome_id`` unique and equal to the lineage set;
    ``split`` in {train, test}; ``fold`` null exactly on test rows and in
    ``0..n_folds-1`` on train rows; no cluster in two splits; no cluster in two
    folds; ``external_set`` only on test rows with known values; ``lolo_lineage``
    only on train rows (LOLO clusters are reserved before test selection) and
    equal to the row's own cluster when set.
    """
    if list(splits.columns) != list(COLUMNS):
        raise ContractViolation(f"splits columns {list(splits.columns)} != {list(COLUMNS)}", stage=STAGE)
    gid = splits["genome_id"].astype(str)
    if gid.duplicated().any():
        raise ContractViolation(f"duplicate genome_id in splits: {gid[gid.duplicated()].tolist()[:5]}", stage=STAGE)
    lineage_ids = set(lineages["genome_id"].astype(str))
    if set(gid) != lineage_ids:
        missing = sorted(lineage_ids - set(gid))[:5]
        extra = sorted(set(gid) - lineage_ids)[:5]
        raise ContractViolation(f"splits/lineages genome sets differ (missing {missing}, extra {extra})", stage=STAGE)

    if splits["split"].isna().any():
        raise ContractViolation("null split values", stage=STAGE)
    split = splits["split"].astype(str)
    bad_split = sorted(set(split) - {SPLIT_TRAIN, SPLIT_TEST})
    if bad_split:
        raise ContractViolation(f"unknown split values {bad_split}", stage=STAGE)

    is_test = _true(split == SPLIT_TEST)
    fold_na = splits["fold"].isna().to_numpy(dtype=bool)
    if not np.array_equal(fold_na, is_test):
        raise ContractViolation("fold must be null exactly on test rows", stage=STAGE)
    train_folds = splits.loc[~is_test, "fold"]
    if len(train_folds) and ((train_folds < 0) | (train_folds >= int(n_folds))).any():
        raise ContractViolation(f"train folds must lie in 0..{int(n_folds) - 1}", stage=STAGE)

    merged = splits.merge(lineages[["genome_id", "species", "lineage_cluster"]], on="genome_id", suffixes=("", "_lin"))
    if _true(merged["species"].astype(str) != merged["species_lin"].astype(str)).any():
        raise ContractViolation("species differs between splits and lineages for some genomes", stage=STAGE)
    per_cluster_split = merged.groupby("lineage_cluster")["split"].nunique()
    spanning = per_cluster_split[per_cluster_split > 1].index.tolist()
    if spanning:
        raise ContractViolation(f"{len(spanning)} cluster(s) appear in both train and test: {spanning[:5]}", stage=STAGE)
    train_rows = merged[_true(merged["split"].astype(str) == SPLIT_TRAIN)]
    per_cluster_fold = train_rows.groupby("lineage_cluster")["fold"].nunique()
    spanning_folds = per_cluster_fold[per_cluster_fold > 1].index.tolist()
    if spanning_folds:
        raise ContractViolation(f"{len(spanning_folds)} cluster(s) span two folds: {spanning_folds[:5]}", stage=STAGE)

    ext = splits["external_set"]
    ext_set = ext.notna().to_numpy(dtype=bool)
    if np.any(ext_set & ~is_test):
        raise ContractViolation("external_set must be null on train rows", stage=STAGE)
    bad_ext = sorted(set(ext.dropna().astype(str)) - set(EXTERNAL_SETS))
    if bad_ext:
        raise ContractViolation(f"unknown external_set values {bad_ext}", stage=STAGE)

    lolo = merged["lolo_lineage"]
    has_lolo = lolo.notna().to_numpy(dtype=bool)
    if np.any(has_lolo & _true(lolo.astype(str) != merged["lineage_cluster"].astype(str))):
        raise ContractViolation("lolo_lineage must equal the row's own lineage_cluster when set", stage=STAGE)
    merged_test = _true(merged["split"].astype(str) == SPLIT_TEST)
    if np.any(has_lolo & merged_test):
        on_test = sorted(set(lolo[has_lolo & merged_test].astype(str)))
        raise ContractViolation(
            f"lolo_lineage must be null on test rows (LOLO clusters are reserved for train); found {on_test[:5]}",
            stage=STAGE,
        )


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
def run(
    paths: Paths,
    config: Any,
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    test_range: tuple[float, float] = DEFAULT_TEST_RANGE,
    n_folds: int = DEFAULT_N_FOLDS,
    max_swaps: int = DEFAULT_MAX_SWAPS,
    n_lolo: int = DEFAULT_N_LOLO,
) -> pd.DataFrame:
    """Read lineages, labels and kept pairs; write the frozen ``splits.parquet``.

    Args:
        paths: Project paths.
        config: Loaded :class:`genome2mic.config.Config`; only ``.species`` is
            used to drop lineage rows of out-of-scope species (``None`` skips it).
        seed: Base random seed (recorded in the log).
        force: Overwrite an existing ``splits.parquet``. Without it an existing
            file raises :class:`ContractViolation` (rule 7: splits are frozen).
        test_range, n_folds, max_swaps, n_lolo: See :func:`build_splits`.

    Raises:
        ContractViolation: existing frozen file without ``force``; broken invariant.
        FileNotFoundError: a required input is missing.
    """
    if paths.splits.exists():
        if not force:
            raise ContractViolation(
                f"{paths.splits} already exists and splits are frozen (CLAUDE.md rule 7). "
                "Fix the upstream problem instead of regenerating; pass force=True only if you "
                "accept that every result produced so far becomes invalid.",
                stage=STAGE,
            )
        logger.warning("[%s] force=True: overwriting frozen %s -- every existing result is now invalid", STAGE, paths.splits)

    for required, hint in (
        (paths.lineages, "run the lineages stage first"),
        (paths.labels, "run the ingest stage first"),
        (paths.pairs_kept, "run the ingest stage first (it writes pairs_kept.csv)"),
    ):
        if not required.is_file():
            raise FileNotFoundError(f"{required} not found; {hint}")

    droplog = DropLog(STAGE)
    lineages = read_parquet(paths.lineages)
    labels = read_parquet(paths.labels)
    pairs_kept = read_csv(paths.pairs_kept)

    if config is not None:
        known = {str(k).upper() for k in getattr(config, "species", {})}
        in_config = _true(lineages["species"].astype(str).isin(known))
        unknown = sorted(set(lineages.loc[~in_config, "species"].astype(str)))
        lineages = droplog.keep_where(
            lineages, in_config, "lineage_rows_species_not_in_config", detail=", ".join(unknown) if unknown else None
        )

    logger.info(
        "[%s] building splits with seed=%d, test range %.0f-%.0f%%, %d folds, max %d swaps, %d lolo cluster(s)/species",
        STAGE,
        seed,
        100 * test_range[0],
        100 * test_range[1],
        n_folds,
        max_swaps,
        n_lolo,
    )
    splits = build_splits(
        lineages,
        labels,
        pairs_kept,
        seed=seed,
        test_range=test_range,
        n_folds=n_folds,
        max_swaps=max_swaps,
        n_lolo=n_lolo,
        droplog=droplog,
    )
    write_parquet(splits, paths.splits)
    droplog.write(paths.drop_log(STAGE))
    logger.info(
        "[%s] wrote %d row(s) (%d test, %d train) to %s -- this file is now frozen",
        STAGE,
        len(splits),
        int((splits["split"] == SPLIT_TEST).sum()),
        int((splits["split"] == SPLIT_TRAIN).sum()),
        paths.splits,
    )
    return splits
