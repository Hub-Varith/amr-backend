"""Stage 5 -- known-AMR features from AMRFinderPlus output (``known_amr.parquet``).

One row per QC-passing genome (from ``qc.parquet``), wide, ``int8``, zero-filled:

* ``gene_<FAMILY>``       0/1 -- an acquired resistance gene of that family is present
* ``point_<GENE>_<MUT>``  0/1 -- that resistance mutation is present
* ``n_class_<CLASS>``     count of AMRFinderPlus ``Type == AMR`` hits in that class

Parsing (:func:`parse_amrfinder`)
    Both header variants are accepted: AMRFinderPlus 4.x (``Element symbol``,
    ``Type``, ``Subtype``) and 3.x (``Gene symbol``, ``Element type``,
    ``Element subtype``). Only ``Type == AMR`` rows are kept; ``STRESS`` and
    ``VIRULENCE`` elements are counted and dropped. Within ``AMR``, ``Subtype ==
    POINT`` becomes a ``point_`` column and ``Subtype == AMR`` a ``gene_`` column;
    any other subtype (e.g. ``AMR-SUSCEPTIBLE``) is counted and dropped. ``NA`` and
    empty cells are read as null.

Family rule (:func:`family_of`)
    The family of an acquired gene is its symbol with the trailing allele number
    removed: the last ``-<digits>`` group, optionally with a ``.<digits>`` sub-number
    (``blaCTX-M-15 -> blaCTX-M``, ``blaOXA-1 -> blaOXA``, ``mcr-1.1 -> mcr``).
    Symbols without such a suffix are unchanged (``mecA``, ``tet(A)``,
    ``aac(6')-Ib-cr``, ``qnrB1``). Exception: when the symbol matches a prefix from
    ``configs/keep_variant.csv`` the exact variant is kept, because the variant
    changes the drug answer (``blaKPC-2 -> blaKPC-2``, ``blaOXA-48 -> blaOXA-48``).
    A prefix matches at a token boundary only: ``blaOXA-48`` matches ``blaOXA-48``
    and ``blaOXA-48-like`` but not ``blaOXA-484``, and ``blaKPC`` matches
    ``blaKPC-2``. Point mutations never go through the family rule.

Column names (:func:`column_name`)
    ``<prefix>`` + lower-cased symbol with every run of non-alphanumeric characters
    replaced by one ``_`` and leading/trailing ``_`` removed:
    ``gene_blactx_m``, ``gene_blakpc_2``, ``point_ompk36_d135dgd``,
    ``n_class_beta_lactam``. Two distinct source symbols that sanitise to the same
    column are merged (logical OR) and both are listed in the mapping file.

The rare-feature filter (``< min_count`` training genomes) is **not** applied here:
it belongs inside each training fold. :func:`filter_rare` is the fold-side helper.

The column mapping ``known_amr_columns.csv`` carries the contract's columns
(``column_name, source_symbol, class, subclass, n_genomes_present``) plus
``member_symbols``: the ``;``-joined original symbols that collapsed into the
column, for audit.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from genome2mic.config import Config
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.io import read_parquet, write_csv, write_parquet
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

STAGE = "known_amr"
AMRFINDER_TSV_NAME = "amrfinder.tsv"

PREFIX_GENE = "gene_"
PREFIX_POINT = "point_"
PREFIX_CLASS = "n_class_"
FEATURE_PREFIXES: tuple[str, ...] = (PREFIX_GENE, PREFIX_POINT, PREFIX_CLASS)
ID_COLUMNS: tuple[str, ...] = ("genome_id", "species")

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {"lineage_cluster", "st", "country", "year", "source", "isolation_source", "biosample", "split", "fold"}
)
"""Metadata that must never enter a feature matrix (CLAUDE.md rule 2)."""

TYPE_AMR = "AMR"
SUBTYPE_AMR = "AMR"
SUBTYPE_POINT = "POINT"

PARSED_COLUMNS: tuple[str, ...] = ("symbol", "type", "subtype", "class", "subclass", "method", "coverage", "identity")
"""Columns returned by :func:`parse_amrfinder` (lower-case, version-independent)."""

COLUMNS_FILE_COLUMNS: tuple[str, ...] = (
    "column_name",
    "source_symbol",
    "class",
    "subclass",
    "n_genomes_present",
    "member_symbols",
)

INT8_MAX = 127

_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("Element symbol", "Gene symbol"),
    "type": ("Type", "Element type"),
    "subtype": ("Subtype", "Element subtype"),
    "class": ("Class",),
    "subclass": ("Subclass",),
    "method": ("Method",),
    "coverage": ("% Coverage of reference", "% Coverage of reference sequence"),
    "identity": ("% Identity to reference", "% Identity to reference sequence"),
}
_REQUIRED_FIELDS: tuple[str, ...] = ("symbol", "type", "subtype")
_NULL_TOKENS: frozenset[str] = frozenset({"", "NA", "N/A", "-", "."})
_VARIANT_SUFFIX_RE = re.compile(r"-\d+(?:\.\d+)?$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_COLUMN_RE = re.compile(r"^[a-z0-9_]+$")
_ID_PREVIEW = 10


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def _matches_prefix(symbol: str, prefix: str) -> bool:
    """True if ``symbol`` is ``prefix`` or continues it at a token boundary."""
    if symbol == prefix:
        return True
    if symbol.startswith(prefix):
        return not symbol[len(prefix)].isalnum()
    return False


def family_of(symbol: str, keep_variant_prefixes: Iterable[str]) -> str:
    """Gene family of an acquired-gene symbol (see the module docstring for the rule).

    >>> family_of("blaCTX-M-15", ("blaKPC",))
    'blaCTX-M'
    >>> family_of("blaKPC-2", ("blaKPC",))
    'blaKPC-2'
    """
    text = str(symbol).strip()
    if not text:
        raise ValueError("gene symbol must be a non-empty string")
    for prefix in keep_variant_prefixes:
        if _matches_prefix(text, str(prefix).strip()):
            return text
    return _VARIANT_SUFFIX_RE.sub("", text) or text


def column_name(prefix: str, symbol: str) -> str:
    """Sanitised feature column name: ``prefix`` + lower-case alphanumeric body.

    >>> column_name("gene_", "blaCTX-M")
    'gene_blactx_m'
    >>> column_name("point_", "ompK36_D135DGD")
    'point_ompk36_d135dgd'
    """
    if prefix not in FEATURE_PREFIXES:
        raise ValueError(f"prefix must be one of {FEATURE_PREFIXES}, got {prefix!r}")
    body = _NON_ALNUM_RE.sub("_", str(symbol).strip().lower()).strip("_")
    if not body:
        raise ValueError(f"symbol {symbol!r} has no alphanumeric characters")
    return prefix + body


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns of ``frame`` that carry a feature prefix, in frame order."""
    return [c for c in frame.columns if str(c).startswith(FEATURE_PREFIXES)]


# --------------------------------------------------------------------------- #
# AMRFinderPlus parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AmrHit:
    """One ``Type == AMR`` row of an AMRFinderPlus table."""

    symbol: str
    subtype: str
    amr_class: str | None
    subclass: str | None
    method: str | None
    coverage: float | None
    identity: float | None


@dataclass
class ParseCounts:
    """Row-level filter counts for one or many AMRFinderPlus tables."""

    n_rows: int = 0
    n_non_amr: int = 0
    n_other_subtype: int = 0
    n_no_symbol: int = 0

    def update(self, other: "ParseCounts") -> None:
        self.n_rows += other.n_rows
        self.n_non_amr += other.n_non_amr
        self.n_other_subtype += other.n_other_subtype
        self.n_no_symbol += other.n_no_symbol


def _clean(cell: str | None) -> str | None:
    if cell is None:
        return None
    text = cell.strip()
    return None if text in _NULL_TOKENS else text


def _as_float(cell: str | None) -> float | None:
    text = _clean(cell)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _resolve_header(header: Sequence[str], path: Path) -> dict[str, int]:
    """Map our field names to column positions using either header variant."""
    positions = {name.strip(): i for i, name in enumerate(header)}
    index: dict[str, int] = {}
    for field_name, aliases in _HEADER_ALIASES.items():
        for alias in aliases:
            if alias in positions:
                index[field_name] = positions[alias]
                break
    missing = [f for f in _REQUIRED_FIELDS if f not in index]
    if missing:
        raise ValueError(
            f"{path}: not an AMRFinderPlus table; missing columns for {missing} "
            f"(accepted names: {[_HEADER_ALIASES[m] for m in missing]}); header was {list(header)}"
        )
    return index


def _parse_rows(path: Path) -> tuple[list[AmrHit], ParseCounts]:
    counts = ParseCounts()
    hits: list[AmrHit] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path}: empty file; AMRFinderPlus always writes a header line")
        index = _resolve_header(header, path)

        def cell(row: Sequence[str], name: str) -> str | None:
            position = index.get(name)
            if position is None or position >= len(row):
                return None
            return row[position]

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            counts.n_rows += 1
            if _clean(cell(row, "type")) != TYPE_AMR:
                counts.n_non_amr += 1
                continue
            subtype = _clean(cell(row, "subtype"))
            if subtype not in (SUBTYPE_AMR, SUBTYPE_POINT):
                counts.n_other_subtype += 1
                continue
            symbol = _clean(cell(row, "symbol"))
            if symbol is None:
                counts.n_no_symbol += 1
                continue
            hits.append(
                AmrHit(
                    symbol=symbol,
                    subtype=subtype,
                    amr_class=_clean(cell(row, "class")),
                    subclass=_clean(cell(row, "subclass")),
                    method=_clean(cell(row, "method")),
                    coverage=_as_float(cell(row, "coverage")),
                    identity=_as_float(cell(row, "identity")),
                )
            )
    return hits, counts


def parse_amrfinder(tsv_path: Path) -> pd.DataFrame:
    """Read one AMRFinderPlus table and return its ``Type == AMR`` rows.

    Columns: :data:`PARSED_COLUMNS`. ``subtype`` is ``AMR`` or ``POINT``. Nulls for
    empty / ``NA`` cells. Raises ``ValueError`` for an empty file or a header that is
    not an AMRFinderPlus header. Dropped-row counts are logged at DEBUG; the stage
    builder aggregates them into its ``DropLog``.
    """
    hits, counts = _parse_rows(Path(tsv_path))
    logger.debug(
        "%s: %d rows, %d AMR hits kept, %d non-AMR, %d other subtype, %d without symbol",
        tsv_path,
        counts.n_rows,
        len(hits),
        counts.n_non_amr,
        counts.n_other_subtype,
        counts.n_no_symbol,
    )
    return pd.DataFrame(
        {
            "symbol": pd.array([h.symbol for h in hits], dtype="str"),
            "type": pd.array([TYPE_AMR] * len(hits), dtype="str"),
            "subtype": pd.array([h.subtype for h in hits], dtype="str"),
            "class": pd.array([h.amr_class for h in hits], dtype="str"),
            "subclass": pd.array([h.subclass for h in hits], dtype="str"),
            "method": pd.array([h.method for h in hits], dtype="str"),
            "coverage": pd.array([h.coverage for h in hits], dtype="Float64"),
            "identity": pd.array([h.identity for h in hits], dtype="Float64"),
        },
        columns=list(PARSED_COLUMNS),
    )


# --------------------------------------------------------------------------- #
# Building the wide table
# --------------------------------------------------------------------------- #


@dataclass
class _ColumnMeta:
    sources: set[str] = field(default_factory=set)
    members: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    subclasses: set[str] = field(default_factory=set)

    def add(self, source: str, member: str, amr_class: str | None, subclass: str | None) -> None:
        self.sources.add(source)
        self.members.add(member)
        if amr_class is not None:
            self.classes.add(amr_class)
        if subclass is not None:
            self.subclasses.add(subclass)


def _join(values: Iterable[str]) -> str | None:
    items = sorted(set(values))
    return ";".join(items) if items else None


def _qc_passing(qc: pd.DataFrame, log: DropLog) -> pd.DataFrame:
    for column in ("genome_id", "qc_pass"):
        if column not in qc.columns:
            raise ContractViolation(f"qc.parquet is missing column {column!r}", stage=STAGE)
    if qc["qc_pass"].isna().any():
        raise ContractViolation("qc.parquet has null qc_pass", stage=STAGE)
    passing = log.keep_where(qc, qc["qc_pass"].astype(bool), "qc_fail", detail="qc_pass == False; excluded from known_amr")
    ids = passing["genome_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ContractViolation("qc.parquet genome_id must be non-null and unique", stage=STAGE)
    return passing


def _species_by_genome(passing: pd.DataFrame, paths: Paths) -> dict[str, str]:
    """``genome_id -> species`` for QC-passing genomes, from ``qc.parquet`` or ``labels.parquet``."""
    ids = [str(g) for g in passing["genome_id"]]
    species: dict[str, str] = {}
    if "species" in passing.columns:
        for gid, sp in zip(ids, passing["species"], strict=True):
            if not pd.isna(sp):
                species[gid] = str(sp)
    missing = [gid for gid in ids if gid not in species]
    if missing and paths.labels.is_file():
        labels = read_parquet(paths.labels, columns=["genome_id", "species"]).drop_duplicates()
        for gid, sp in zip(labels["genome_id"], labels["species"], strict=True):
            key = str(gid)
            if key in species or pd.isna(sp):
                continue
            if key in missing:
                species[key] = str(sp)
        missing = [gid for gid in ids if gid not in species]
    if missing:
        raise ContractViolation(
            f"{len(missing)} QC-passing genome(s) have no species in qc.parquet or labels.parquet: "
            f"{missing[:_ID_PREVIEW]}",
            stage=STAGE,
        )
    return species


def build(
    paths: Paths,
    config: Config,
    *,
    droplog: DropLog | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the known-AMR feature table and its column mapping (not written).

    Reads ``qc.parquet`` (QC-passing genomes and their species) and
    ``data/interim/<genome_id>/amrfinder.tsv`` for each. Genomes without a table
    get all-zero features and are counted in the drop log (rows kept). Returns
    ``(features, columns)`` after :func:`validate_known_amr`.
    """
    log = droplog if droplog is not None else DropLog(STAGE)
    if not paths.qc.is_file():
        raise FileNotFoundError(f"qc.parquet not found at {paths.qc}; run the qc stage first")
    qc = read_parquet(paths.qc)
    passing = _qc_passing(qc, log)
    species = _species_by_genome(passing, paths)
    genome_ids = sorted(species)
    keep = tuple(config.keep_variant_prefixes())
    logger.info("known_amr: %d QC-passing genomes, %d keep-variant prefixes", len(genome_ids), len(keep))

    present: defaultdict[str, set[str]] = defaultdict(set)
    class_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    meta: defaultdict[str, _ColumnMeta] = defaultdict(_ColumnMeta)
    parse_counts = ParseCounts()
    missing_tsv: list[str] = []
    n_hits = 0
    n_no_class = 0

    for gid in genome_ids:
        tsv = paths.interim_dir(gid) / AMRFINDER_TSV_NAME
        if not tsv.is_file():
            missing_tsv.append(gid)
            continue
        hits, counts = _parse_rows(tsv)
        parse_counts.update(counts)
        n_hits += len(hits)
        for hit in hits:
            if hit.subtype == SUBTYPE_POINT:
                source = hit.symbol
                column = column_name(PREFIX_POINT, source)
            else:
                source = family_of(hit.symbol, keep)
                column = column_name(PREFIX_GENE, source)
            present[column].add(gid)
            meta[column].add(source, hit.symbol, hit.amr_class, hit.subclass)
            if hit.amr_class is None:
                n_no_class += 1
                continue
            class_column = column_name(PREFIX_CLASS, hit.amr_class)
            class_counts[class_column][gid] += 1
            meta[class_column].add(hit.amr_class, hit.amr_class, hit.amr_class, None)

    log.drop(
        "amrfinder.tsv missing -> features zero-filled (rows kept)",
        len(missing_tsv),
        detail=", ".join(missing_tsv[:_ID_PREVIEW]) + (" ..." if len(missing_tsv) > _ID_PREVIEW else "") or None,
    )
    log.drop("amrfinder rows with Type != AMR (STRESS/VIRULENCE)", parse_counts.n_non_amr)
    log.drop("amrfinder AMR rows with Subtype not in {AMR, POINT}", parse_counts.n_other_subtype)
    log.drop("amrfinder AMR rows without a symbol", parse_counts.n_no_symbol)
    log.drop("amrfinder AMR hits without a Class (kept as gene_/point_, not counted in n_class_)", n_no_class)
    if missing_tsv:
        logger.warning("%d genome(s) have no %s; features zero-filled: %s", len(missing_tsv), AMRFINDER_TSV_NAME, missing_tsv[:_ID_PREVIEW])

    for column, info in meta.items():
        if column.startswith(PREFIX_GENE) and len(info.sources) > 1:
            logger.warning("column %s merges distinct families %s (sanitised names collide)", column, sorted(info.sources))

    gene_cols = sorted(c for c in present if c.startswith(PREFIX_GENE))
    point_cols = sorted(c for c in present if c.startswith(PREFIX_POINT))
    class_cols = sorted(class_counts)
    columns = gene_cols + point_cols + class_cols
    row_of = {gid: i for i, gid in enumerate(genome_ids)}
    matrix = np.zeros((len(genome_ids), len(columns)), dtype=np.int8)
    for j, column in enumerate(columns):
        if column in class_counts:
            for gid, count in class_counts[column].items():
                if count > INT8_MAX:
                    raise ContractViolation(f"{column} = {count} for {gid} exceeds int8; contract requires small ints", stage=STAGE)
                matrix[row_of[gid], j] = count
        else:
            for gid in present[column]:
                matrix[row_of[gid], j] = 1

    features = pd.DataFrame(matrix, columns=columns)
    features.insert(0, "species", pd.array([species[g] for g in genome_ids], dtype="str"))
    features.insert(0, "genome_id", pd.array(genome_ids, dtype="str"))

    n_present = (matrix > 0).sum(axis=0)
    columns_df = pd.DataFrame(
        {
            "column_name": pd.array(columns, dtype="str"),
            "source_symbol": pd.array([_join(meta[c].sources) for c in columns], dtype="str"),
            "class": pd.array([_join(meta[c].classes) for c in columns], dtype="str"),
            "subclass": pd.array([_join(meta[c].subclasses) for c in columns], dtype="str"),
            "n_genomes_present": np.asarray(n_present, dtype=np.int64),
            "member_symbols": pd.array([_join(meta[c].members) for c in columns], dtype="str"),
        },
        columns=list(COLUMNS_FILE_COLUMNS),
    )
    logger.info(
        "known_amr: %d AMR hits -> %d gene_, %d point_, %d n_class_ columns over %d genomes",
        n_hits,
        len(gene_cols),
        len(point_cols),
        len(class_cols),
        len(genome_ids),
    )
    validate_known_amr(features, columns_df, expected_genome_ids=genome_ids)
    return features, columns_df


def validate_known_amr(
    features: pd.DataFrame,
    columns: pd.DataFrame,
    *,
    expected_genome_ids: Iterable[str] | None = None,
) -> None:
    """Contract acceptance checks for stage 5; raises ``ContractViolation``.

    * one row per QC-passing genome (``genome_id`` unique; equals ``expected_genome_ids`` if given)
    * ``species`` non-null
    * every non-id column has a feature prefix and a sanitised name; none is a forbidden field
    * feature columns are ``int8``, no nulls, ``gene_``/``point_`` in {0, 1}, ``n_class_`` in 0..127
    * mapping non-empty, with the required columns, naming exactly the feature columns
    """
    for column in ID_COLUMNS:
        if column not in features.columns:
            raise ContractViolation(f"known_amr is missing column {column!r}", stage=STAGE)
    ids = features["genome_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ContractViolation("known_amr genome_id must be non-null and unique", stage=STAGE)
    if expected_genome_ids is not None:
        expected = {str(g) for g in expected_genome_ids}
        actual = {str(g) for g in ids}
        if expected != actual:
            raise ContractViolation(
                f"known_amr must have one row per QC-passing genome; missing {sorted(expected - actual)[:_ID_PREVIEW]}, "
                f"extra {sorted(actual - expected)[:_ID_PREVIEW]}",
                stage=STAGE,
            )
    if features["species"].isna().any():
        raise ContractViolation("known_amr species must be non-null", stage=STAGE)

    feature_cols = [c for c in features.columns if c not in ID_COLUMNS]
    forbidden = sorted(set(feature_cols) & FORBIDDEN_FEATURES)
    if forbidden:
        raise ContractViolation(f"forbidden metadata in feature matrix: {forbidden}", stage=STAGE)
    bad_names = [c for c in feature_cols if not str(c).startswith(FEATURE_PREFIXES) or not _COLUMN_RE.match(str(c))]
    if bad_names:
        raise ContractViolation(f"feature columns must be sanitised and prefixed {FEATURE_PREFIXES}: {bad_names[:_ID_PREVIEW]}", stage=STAGE)
    if not feature_cols:
        raise ContractViolation("known_amr has no feature columns; no AMRFinderPlus hit in any genome", stage=STAGE)

    block = features[feature_cols]
    if block.isna().any().any():
        raise ContractViolation("known_amr feature columns contain nulls", stage=STAGE)
    wrong_dtype = [c for c in feature_cols if block[c].dtype != np.int8]
    if wrong_dtype:
        raise ContractViolation(f"feature columns must be int8: {wrong_dtype[:_ID_PREVIEW]}", stage=STAGE)
    values = block.to_numpy(dtype=np.int16)
    if (values < 0).any():
        raise ContractViolation("feature columns contain negative values", stage=STAGE)
    binary = [c for c in feature_cols if c.startswith((PREFIX_GENE, PREFIX_POINT))]
    if binary and (block[binary].to_numpy(dtype=np.int16) > 1).any():
        raise ContractViolation("gene_/point_ columns must be 0/1", stage=STAGE)

    missing_cols = [c for c in COLUMNS_FILE_COLUMNS if c not in columns.columns]
    if missing_cols:
        raise ContractViolation(f"column mapping is missing {missing_cols}", stage=STAGE)
    if len(columns) == 0:
        raise ContractViolation("column mapping is empty", stage=STAGE)
    mapped = {str(c) for c in columns["column_name"]}
    if mapped != set(feature_cols):
        raise ContractViolation("column mapping does not name exactly the feature columns", stage=STAGE)


def run(paths: Paths, config: Config) -> pd.DataFrame:
    """Build and write ``known_amr.parquet``, ``known_amr_columns.csv`` and the drop log."""
    log = DropLog(STAGE)
    features, columns = build(paths, config, droplog=log)
    write_parquet(features, paths.known_amr)
    write_csv(columns, paths.known_amr_columns)
    log.write(paths.drop_log(STAGE))
    logger.info("wrote %s (%d x %d) and %s (%d columns)", paths.known_amr, *features.shape, paths.known_amr_columns, len(columns))
    return features


# --------------------------------------------------------------------------- #
# Fold-side helper
# --------------------------------------------------------------------------- #


def filter_rare(
    X: pd.DataFrame,
    train_mask: np.ndarray | pd.Series | Sequence[bool],
    min_count: int = 5,
    *,
    droplog: DropLog | None = None,
) -> list[str]:
    """Feature columns present (``> 0``) in at least ``min_count`` **training** genomes.

    Only rows where ``train_mask`` is true are counted, so the filter can run inside a
    fold without seeing validation or test genomes. ``train_mask`` is a boolean array
    aligned positionally with ``X`` (or a boolean Series on ``X``'s index). Returns
    the kept columns in ``X`` order; raises ``ContractViolation`` if a forbidden
    metadata column is among the feature columns.
    """
    if not isinstance(min_count, (int, np.integer)) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError(f"min_count must be a positive integer, got {min_count!r}")
    if isinstance(train_mask, pd.Series):
        mask = train_mask.reindex(X.index).fillna(False).to_numpy(dtype=bool)
    else:
        mask = np.asarray(train_mask)
        if mask.dtype != bool:
            raise ValueError("train_mask must be boolean")
    if mask.shape != (len(X),):
        raise ValueError(f"train_mask has shape {mask.shape}, expected ({len(X)},)")

    columns = feature_columns(X)
    forbidden = sorted(FORBIDDEN_FEATURES & set(map(str, X.columns)) - set(ID_COLUMNS))
    if forbidden:
        raise ContractViolation(f"forbidden metadata among candidate features: {forbidden}", stage=STAGE)
    if not columns:
        return []
    counts = (X.loc[mask, columns].to_numpy(dtype=np.int64) > 0).sum(axis=0)
    kept = [c for c, n in zip(columns, counts, strict=True) if n >= min_count]
    n_dropped = len(columns) - len(kept)
    detail = f"min_count={min_count} over {int(mask.sum())} training genomes"
    if droplog is not None:
        droplog.drop("rare known-AMR feature columns", n_dropped, detail=detail)
    else:
        logger.info("filter_rare: kept %d of %d feature columns (%s)", len(kept), len(columns), detail)
    return kept


__all__ = [
    "AMRFINDER_TSV_NAME",
    "COLUMNS_FILE_COLUMNS",
    "FEATURE_PREFIXES",
    "FORBIDDEN_FEATURES",
    "ID_COLUMNS",
    "PARSED_COLUMNS",
    "PREFIX_CLASS",
    "PREFIX_GENE",
    "PREFIX_POINT",
    "AmrHit",
    "ParseCounts",
    "build",
    "column_name",
    "family_of",
    "feature_columns",
    "filter_rare",
    "parse_amrfinder",
    "run",
    "validate_known_amr",
]
