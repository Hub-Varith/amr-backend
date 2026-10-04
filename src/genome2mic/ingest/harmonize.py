"""Stage 2 of the pipeline: raw AST rows -> ``labels.parquet`` (DATA_CONTRACT.md).

Both readers (:mod:`genome2mic.ingest.bvbrc`, :mod:`genome2mic.ingest.ncbi_ast`)
emit the **common raw schema** (:data:`RAW_COLUMNS`). :func:`harmonize` turns that
into the stage-2 contract table (:data:`LABEL_COLUMNS`) in a fixed order of steps,
each of which records its drop count in a :class:`~genome2mic.droplog.DropLog`:

1. keep ``evidence == 'Laboratory Method'`` (NCBI rows are lab by definition);
2. normalize the drug (unknown -> drop) and the species (unknown -> drop);
3. classify the typing method (``dilution`` / ``gradient`` / ``disk``; unknown -> drop);
4. check the MIC unit on numeric rows (``mg/L`` and its spellings; anything else -> drop);
5. convert every result to one interval ``(mic_lower, mic_upper]``:
   the numeric path for dilution / gradient rows with a value, the S/I/R path for disk
   rows and for rows without a value. A disk value is a zone diameter, never an MIC.
   Numeric path: a combination MIC (``16/4`` for piperacillin/tazobactam) uses its
   first, primary-agent component; a decimal rendering of a power of two (``0.016``,
   ``0.008``, ``0.12``) is snapped to that power (:func:`genome2mic.mic.snap_reported_mic`)
   before the interval rule; both are counted in the drop log (counts, not drops) and
   the reported text stays in ``raw_result``.
   S/I/R path: needs the breakpoint table for exactly the row's (standard,
   standard_year) and a row for (species, drug) in it. A null or unknown standard, a
   null year, a year without a table, or a missing pair drops the row (never guess;
   there is no fallback to the latest table);
6. de-duplicate by ``biosample`` across sources: rows of the same biosample are
   re-keyed to the BV-BRC ``genome_id``;
7. resolve duplicate (``genome_id``, ``drug``) results: overlapping intervals ->
   intersection; adjacent steps -> the higher (safer) one; S vs R or more than one
   step apart -> the pair is dropped. The merged row's ``method``, ``standard``,
   ``standard_year`` and ``source`` come as one block from a row that supports the
   merged bounds (see :func:`_merge_group`);
8. acceptance checks (:func:`check_labels`) raise :class:`ContractViolation`.

Nothing here imputes a label: a row that cannot be converted is dropped and counted.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genome2mic import mic as micmod
from genome2mic.config import STANDARDS, Config, normalize_standard
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

RAW_COLUMNS: tuple[str, ...] = (
    "genome_id",
    "biosample",
    "species",
    "antibiotic_raw",
    "sir_raw",
    "sign",
    "value",
    "unit",
    "method_raw",
    "standard",
    "standard_year",
    "evidence",
    "source",
    "isolation_source",
    "country",
    "year",
)
"""Common raw schema returned by every reader. Text is ``object`` with ``None`` for
missing, ``value`` is the raw measurement (string or number), ``standard_year`` and
``year`` are nullable ``Int64``."""

RAW_INT_COLUMNS: tuple[str, ...] = ("standard_year", "year")

LABEL_COLUMNS: tuple[str, ...] = (
    "genome_id",
    "biosample",
    "species",
    "drug",
    "mic_lower",
    "mic_upper",
    "censor",
    "sir",
    "raw_result",
    "method",
    "standard",
    "standard_year",
    "source",
    "isolation_source",
    "country",
    "year",
)
"""Stage-2 contract columns, in order."""

LABEL_TEXT_COLUMNS: tuple[str, ...] = (
    "genome_id",
    "biosample",
    "species",
    "drug",
    "censor",
    "sir",
    "raw_result",
    "method",
    "standard",
    "source",
    "isolation_source",
    "country",
)
LABEL_FLOAT_COLUMNS: tuple[str, ...] = ("mic_lower", "mic_upper")
LABEL_INT_COLUMNS: tuple[str, ...] = ("standard_year", "year")
LABEL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "genome_id",
    "species",
    "drug",
    "mic_lower",
    "mic_upper",
    "censor",
    "raw_result",
    "method",
    "source",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "genome_id",
    "biosample",
    "species",
    "source",
    "isolation_source",
    "country",
    "year",
)
"""Columns of the optional ``data/raw/genome_metadata.csv``."""

METHOD_DILUTION = "dilution"
METHOD_GRADIENT = "gradient"
METHOD_DISK = "disk"
METHODS: tuple[str, ...] = (METHOD_DILUTION, METHOD_GRADIENT, METHOD_DISK)
CENSORS: tuple[str, ...] = (micmod.CENSOR_INTERVAL, micmod.CENSOR_LEFT, micmod.CENSOR_RIGHT)
SIR_VALUES: tuple[str, ...] = ("S", "I", "R")
SOURCE_BVBRC = "BVBRC"
SOURCE_NCBI = "NCBI"
SOURCES: tuple[str, ...] = (SOURCE_BVBRC, SOURCE_NCBI)
LAB_EVIDENCE = "Laboratory Method"

DEFAULT_SPECIES_NAMES: Mapping[str, str] = {
    "ECOLI": "Escherichia coli",
    "KPNEU": "Klebsiella pneumoniae",
    "SAUR": "Staphylococcus aureus",
    "PAER": "Pseudomonas aeruginosa",
    "ABAU": "Acinetobacter baumannii",
}
"""Species key -> binomial name (DATA_CONTRACT.md section 0). ``ingest.run`` passes
the names from ``species.yaml`` instead so the configs stay the single source."""


class Reason:
    """Drop-log reasons, one per filter, in pipeline order."""

    NO_GENOME_ID = "missing genome_id"
    EVIDENCE = "evidence != Laboratory Method"
    UNKNOWN_DRUG = "unknown drug"
    UNKNOWN_SPECIES = "unknown species"
    UNKNOWN_METHOD = "unknown laboratory typing method"
    BAD_SIGN = "unknown measurement sign"
    BAD_VALUE = "unparseable or non-positive MIC value"
    UNKNOWN_UNIT = "unknown MIC unit"
    NO_RESULT = "no MIC value and no S/I/R"
    UNKNOWN_SIR = "unknown S/I/R value on S/I/R-only row"
    NULL_STANDARD = "null or unknown standard on S/I/R-only row"
    NULL_YEAR = "S/I/R-only row with null standard_year"
    NO_TABLE_FOR_YEAR = "no breakpoint table for standard_year"
    NO_BREAKPOINT = "no breakpoint for species x drug x standard on S/I/R-only row"
    INVALID_I = "'I' reported but the standard has no I category (S == R)"
    COMBINATION = (
        "combination MIC 'x/y': primary-agent value x used, full text kept in raw_result (count, not a drop)"
    )
    SNAPPED = (
        "decimal rendering of a power of two snapped to the doubling grid, e.g. 0.016 -> 2^-6 "
        "(count, not a drop)"
    )
    REKEYED = "biosample de-dup: rows re-keyed to the BV-BRC genome_id (count, not a drop)"
    DUP_MERGED = "duplicate (genome_id, drug): extra rows merged into one interval"
    DUP_SR = "duplicate (genome_id, drug): S vs R conflict, pair dropped"
    DUP_FAR = "duplicate (genome_id, drug): intervals more than 1 step apart, pair dropped"


# Row-level conversion reasons, in the order they are written to the drop log.
_ROW_REASONS: tuple[str, ...] = (
    Reason.BAD_SIGN,
    Reason.BAD_VALUE,
    Reason.UNKNOWN_UNIT,
    Reason.NO_RESULT,
    Reason.UNKNOWN_SIR,
    Reason.NULL_STANDARD,
    Reason.NULL_YEAR,
    Reason.NO_TABLE_FOR_YEAR,
    Reason.NO_BREAKPOINT,
    Reason.INVALID_I,
)

# Method keywords (contract "Method filter"). Checked in this order; the first
# category with a keyword contained in the lower-cased method wins.
METHOD_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (METHOD_GRADIENT, ("etest", "e-test", "e test", "gradient", "mic strip", "strip")),
    (METHOD_DISK, ("disk", "disc", "kirby", "bauer", "zone")),
    (METHOD_DILUTION, ("broth", "microdilution", "agar dilution", "dilution")),
)

# Accepted MIC unit spellings after lower-casing, removing spaces and mapping the
# micro sign to ``u``. All are mg/L (µg/mL converts 1:1).
_ACCEPTED_UNITS: frozenset[str] = frozenset({"mg/l", "ug/ml", "mcg/ml"})
MIC_UNIT = "mg/L"

_BVBRC_ID_RE = re.compile(r"^\d+\.\d+$")
_EMBEDDED_SIGN_RE = re.compile(r"^(<=|>=|=<|=>|==|≤|≥|<|>|=)\s*(.*)$")
_YEAR_RE = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)")
_WS_RE = re.compile(r"\s+")
_MISSING_TEXT: frozenset[str] = frozenset(
    {
        "",
        "missing",
        "not collected",
        "not applicable",
        "not provided",
        "not available",
        "restricted access",
        "unknown",
        "na",
        "n/a",
        "none",
        "null",
    }
)
_ADJACENT_TOL = 1e-9
_MAX_DETAIL_ITEMS = 5


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #


def is_missing(value: Any) -> bool:
    """True for ``None``, ``pd.NA``, float NaN and blank strings."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def clean_text(value: Any) -> str | None:
    """Stripped string, or ``None`` for a missing / blank value."""
    if is_missing(value):
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def coerce_int(value: Any) -> int | None:
    """Integer from ``2016``, ``"2016"``, ``2016.0`` or ``"2016.0"``; ``None`` otherwise."""
    if is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(round(number))


def year_from_text(value: Any) -> int | None:
    """First four-digit year (1800-2099) in a date-like string, else ``None``.

    ``"2016"``, ``"2016-03-01"``, ``"2016.0"`` and ``"19/03/2016"`` all give 2016;
    ``"missing"`` / ``"not collected"`` give ``None``.
    """
    text = clean_text(value)
    if text is None or text.lower() in _MISSING_TEXT:
        return None
    match = _YEAR_RE.search(text)
    return int(match.group(1)) if match else None


def country_from_text(value: Any) -> str | None:
    """Country part of an NCBI ``geo_loc_name`` (``"USA: California"`` -> ``"USA"``).

    NCBI placeholders (``missing``, ``not collected``, ...) and blanks give ``None``.
    """
    text = clean_text(value)
    if text is None or text.lower() in _MISSING_TEXT:
        return None
    country = text.split(":", 1)[0].strip()
    if not country or country.lower() in _MISSING_TEXT:
        return None
    return country


def species_key_from_organism(organism: Any, species_names: Mapping[str, str] | None = None) -> str | None:
    """Map an organism name to a species key, or ``None`` if it is out of scope.

    Matches the binomial (``Klebsiella pneumoniae``) or its abbreviated genus form
    (``K. pneumoniae``) as the whole name or as its leading words, case- and
    whitespace-insensitive, so strain suffixes (``subsp. pneumoniae KPNIH1``,
    ``O157:H7``) are accepted. ``Klebsiella oxytoca`` and a bare genus are not.
    """
    text = clean_text(organism)
    if text is None:
        return None
    low = _WS_RE.sub(" ", text.lower())
    names = DEFAULT_SPECIES_NAMES if species_names is None else species_names
    for key, name in names.items():
        binomial = _WS_RE.sub(" ", name.strip().lower())
        candidates = [binomial]
        genus, _, rest = binomial.partition(" ")
        if rest:
            candidates.append(f"{genus[0]}. {rest}")
        for candidate in candidates:
            if low == candidate or low.startswith(candidate + " "):
                return key
    return None


def classify_method(method_raw: Any) -> str | None:
    """``dilution`` / ``gradient`` / ``disk`` from the lab method text, else ``None``.

    Follows the contract's keyword table exactly (broth, microdilution, agar
    dilution -> ``dilution``; Etest, gradient, MIC strip -> ``gradient``; disk,
    Kirby-Bauer, zone -> ``disk``). Anything else -- including automated systems
    named without their method -- is unknown and the caller drops the row.
    """
    text = clean_text(method_raw)
    if text is None:
        return None
    low = text.lower()
    for method, keywords in METHOD_KEYWORDS:
        if any(keyword in low for keyword in keywords):
            return method
    return None


def normalize_unit(unit: Any) -> str | None:
    """``"mg/L"`` for an accepted MIC unit spelling (blank assumes mg/L), else ``None``.

    Accepts ``mg/L``, ``mg/l``, ``µg/mL``, ``μg/mL``, ``ug/mL``, ``mcg/mL`` in any case.
    """
    text = clean_text(unit)
    if text is None:
        return MIC_UNIT
    low = text.lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    return MIC_UNIT if low in _ACCEPTED_UNITS else None


def parse_measurement(sign: Any, value: Any) -> tuple[str, float | None]:
    """Canonical ``(sign, value)`` for a raw measurement.

    The sign may be blank (meaning ``=``) or embedded in the value (``">32"``,
    ``"<= 0.25"``). A combination-drug MIC ``"x/y"`` (``"16/4"``, ``"<=8/4"``,
    ``"8 / 4"``: piperacillin/tazobactam, amoxicillin/clavulanate, ...) gives the first,
    primary-agent component ``x``; both components must be positive numbers. The
    number is returned as reported (no snapping to the grid). Returns ``(sign, None)``
    when the value is missing. Raises ``ValueError`` for an unknown or conflicting sign
    and for a value that is not a finite positive number.
    """
    op, number, _ = _parse_measurement(sign, value)
    return op, number


def _parse_measurement(sign: Any, value: Any) -> tuple[str, float | None, str | None]:
    """:func:`parse_measurement` plus the cleaned value text for ``raw_result``."""
    op = micmod.normalize_sign(None if is_missing(sign) else str(sign))
    if is_missing(value):
        return op, None, None
    if isinstance(value, str):
        text = value.strip()
        match = _EMBEDDED_SIGN_RE.match(text)
        if match:
            embedded = micmod.normalize_sign(match.group(1))
            if op != "=" and embedded != op:
                raise ValueError(f"conflicting signs {sign!r} and {match.group(1)!r}")
            op = embedded
            text = match.group(2).strip()
        number = _primary_component(text, value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"MIC value {value!r} is not a number") from None
        text = format(number, "g")
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"MIC value must be a finite positive number, got {value!r}")
    return op, number, text


def _primary_component(text: str, value: Any) -> float:
    """``float(text)``, or the first component of a combination MIC ``"x/y"``.

    The partner agent's fixed concentration (``/4`` tazobactam, ``/2`` clavulanate,
    ``/76`` sulfamethoxazole) must be a positive number too, so ``"N/A"``, ``"8/"`` and
    ``"8/4/2"`` are rejected. Raises ``ValueError`` with "is not a number".
    """
    message = f"MIC value {value!r} is not a number"
    parts = [part.strip() for part in text.split("/")]
    if len(parts) > 2 or not all(parts):
        raise ValueError(message)
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        raise ValueError(message) from None
    if len(numbers) == 2 and not (math.isfinite(numbers[1]) and numbers[1] > 0):
        raise ValueError(message)
    return numbers[0]


def _normalize_evidence(value: Any) -> str | None:
    text = clean_text(value)
    return _WS_RE.sub(" ", text).lower() if text else None


def _top_detail(values: Iterable[Any], limit: int = _MAX_DETAIL_ITEMS) -> str | None:
    """``"a (3), b (1)"`` for the most frequent values, or ``None`` when empty."""
    counts: dict[str, int] = {}
    for value in values:
        key = "<null>" if is_missing(value) else str(value).strip()
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    text = ", ".join(f"{key} ({n})" for key, n in top)
    if len(counts) > limit:
        text += f", ... {len(counts) - limit} more"
    return text


# --------------------------------------------------------------------------- #
# Column coercion
# --------------------------------------------------------------------------- #


def text_object(values: pd.Series | Sequence[Any]) -> pd.Series:
    """Object series of stripped strings with ``None`` for missing / blank values."""
    index = values.index if isinstance(values, pd.Series) else None
    items = values.tolist() if isinstance(values, pd.Series) else list(values)
    return pd.Series([clean_text(v) for v in items], index=index, dtype=object)


def int64_nullable(values: pd.Series | Sequence[Any]) -> pd.Series:
    """Nullable ``Int64`` series via :func:`coerce_int` (unparseable -> ``<NA>``)."""
    index = values.index if isinstance(values, pd.Series) else None
    items = values.tolist() if isinstance(values, pd.Series) else list(values)
    return pd.Series(pd.array([coerce_int(v) for v in items], dtype="Int64"), index=index)


def _value_object(values: pd.Series) -> pd.Series:
    """The raw measurement column: strings stripped, blanks -> ``None``, numbers kept."""
    out: list[Any] = []
    for v in values.tolist():
        if is_missing(v):
            out.append(None)
        elif isinstance(v, str):
            out.append(v.strip())
        else:
            out.append(v)
    return pd.Series(out, index=values.index, dtype=object)


def finalize_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce any frame into the common raw schema (:data:`RAW_COLUMNS`).

    Missing columns are added as all-null; extra columns are dropped; text becomes
    ``object`` with ``None``; ``standard_year`` / ``year`` become ``Int64``. The
    index is reset.
    """
    n = len(frame)
    columns: dict[str, pd.Series] = {}
    for column in RAW_COLUMNS:
        series = frame[column] if column in frame.columns else pd.Series([None] * n, index=frame.index, dtype=object)
        if column in RAW_INT_COLUMNS:
            columns[column] = int64_nullable(series)
        elif column == "value":
            columns[column] = _value_object(series)
        else:
            columns[column] = text_object(series)
    out = pd.DataFrame(columns, columns=list(RAW_COLUMNS), index=frame.index)
    return out.reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], what: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{what}: missing columns {missing}; found {list(frame.columns)}")


# --------------------------------------------------------------------------- #
# Raw CSV reading shared by the readers
# --------------------------------------------------------------------------- #


def normalize_header(name: Any) -> str:
    """``"#BioSample"`` -> ``biosample``; ``"Lab Typing Method"`` -> ``lab_typing_method``."""
    text = str(name).strip().lstrip("#").lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def read_csv_text(path: Path) -> pd.DataFrame:
    """Read a raw CSV with every column as text and normalized header names.

    Blank fields stay ``""`` here (``keep_default_na=False``) so that the string
    ``"NA"`` is not silently turned into a missing value; :func:`text_object`
    converts blanks to ``None`` later. Duplicate normalized headers keep the first.
    """
    frame = pd.read_csv(Path(path), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    renamed: list[str] = []
    seen: set[str] = set()
    keep: list[int] = []
    for position, column in enumerate(frame.columns):
        key = normalize_header(column)
        if key in seen:
            logger.warning("%s: duplicate column %r after normalization; keeping the first", path, column)
            continue
        seen.add(key)
        renamed.append(key)
        keep.append(position)
    out = frame.iloc[:, keep]
    out.columns = renamed
    return out


def select_columns(
    frame: pd.DataFrame,
    aliases: Mapping[str, Sequence[str]],
    required: Iterable[str],
    what: str,
) -> dict[str, pd.Series]:
    """Pick one source column per canonical name from an alias table.

    ``aliases`` maps a canonical name to the normalized header names that may hold
    it, in preference order. A canonical name in ``required`` with no match raises
    ``ValueError``; an optional one is returned as an all-``None`` series.
    """
    out: dict[str, pd.Series] = {}
    missing_required: list[str] = []
    required_set = set(required)
    for canonical, candidates in aliases.items():
        source = next((c for c in candidates if c in frame.columns), None)
        if source is None:
            if canonical in required_set:
                missing_required.append(canonical)
            out[canonical] = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        else:
            out[canonical] = text_object(frame[source])
    if missing_required:
        raise ValueError(
            f"{what}: required columns {missing_required} not found (accepted names: "
            f"{ {c: list(aliases[c]) for c in missing_required} }); found {list(frame.columns)}"
        )
    return out


def unknown_organism_summary(organism: pd.Series, species: pd.Series) -> str | None:
    """``"Klebsiella oxytoca (3), ..."`` for rows whose organism mapped to no species."""
    unmapped = organism[species.isna()]
    if unmapped.empty:
        return None
    return _top_detail(unmapped.tolist())


# --------------------------------------------------------------------------- #
# Genome metadata (optional data/raw/genome_metadata.csv)
# --------------------------------------------------------------------------- #


def read_genome_metadata(path: Path) -> pd.DataFrame:
    """Read ``genome_metadata.csv`` into :data:`METADATA_COLUMNS`.

    ``genome_id`` and ``biosample`` are required; the other columns are optional and
    all-null when absent. Text is ``object`` / ``None``; ``year`` is ``Int64``.
    """
    frame = read_csv_text(path)
    _require_columns(frame, ("genome_id", "biosample"), f"{path}")
    out: dict[str, pd.Series] = {}
    for column in METADATA_COLUMNS:
        series = frame[column] if column in frame.columns else pd.Series([None] * len(frame), dtype=object)
        out[column] = int64_nullable(series) if column == "year" else text_object(series)
    meta = pd.DataFrame(out, columns=list(METADATA_COLUMNS)).reset_index(drop=True)
    logger.info("read genome metadata for %d genomes from %s", meta["genome_id"].nunique(), path)
    return meta


def apply_metadata(frame: pd.DataFrame, metadata: pd.DataFrame | None, on: str) -> pd.DataFrame:
    """Fill ``biosample``, ``species``, ``isolation_source``, ``country``, ``year`` from metadata.

    ``on`` is the join key: ``genome_id`` for BV-BRC rows, ``biosample`` for NCBI rows.
    Only null cells are filled -- values the raw file already has are never
    overridden -- with one exception: when joining on ``biosample`` and the
    metadata maps it to a ``genome_id``, that id replaces the reader's fallback
    ``NCBI_<biosample>`` id (the contract prefers the BV-BRC id).
    """
    if metadata is None or frame.empty or on not in frame.columns:
        return frame
    meta = metadata[metadata[on].notna()].drop_duplicates(subset=[on], keep="first").set_index(on)
    out = frame.copy()
    key = text_object(out[on])
    if on == "biosample" and "genome_id" in meta.columns:
        mapped = key.map(meta["genome_id"])
        has_id = mapped.notna().to_numpy()
        out["genome_id"] = pd.Series(
            np.where(has_id, mapped.to_numpy(dtype=object), out["genome_id"].to_numpy(dtype=object)),
            index=out.index,
            dtype=object,
        )
        if has_id.any():
            logger.info("genome_metadata: %d rows keyed to the genome_id of their biosample", int(has_id.sum()))
    for column in ("biosample", "species", "isolation_source", "country", "year"):
        if column == on or column not in meta.columns:
            continue
        mapped = key.map(meta[column])
        current = out[column] if column in out.columns else pd.Series([None] * len(out), index=out.index, dtype=object)
        current_obj = current.astype(object)
        fill = current_obj.isna().to_numpy()
        out[column] = pd.Series(
            np.where(fill, mapped.to_numpy(dtype=object), current_obj.to_numpy(dtype=object)),
            index=out.index,
            dtype=object,
        )
    return out


# --------------------------------------------------------------------------- #
# Harmonize
# --------------------------------------------------------------------------- #


def harmonize(raw: pd.DataFrame, config: Config, droplog: DropLog) -> pd.DataFrame:
    """Turn common-raw-schema rows into the stage-2 label table.

    See the module docstring for the step order. Every filter writes one record
    to ``droplog`` (zero counts included). Raises ``ValueError`` if ``raw`` lacks a
    column of :data:`RAW_COLUMNS` and :class:`ContractViolation` if the result
    fails the acceptance checks.
    """
    _require_columns(raw, RAW_COLUMNS, "harmonize input")
    frame = finalize_raw(raw)
    n_in = len(frame)
    frame = frame.assign(species=frame["species"].map(lambda s: s.upper() if isinstance(s, str) else None))

    # 0. a label without a genome is useless
    frame = droplog.keep_where(frame, frame["genome_id"].notna(), Reason.NO_GENOME_ID)

    # 1. lab-measured results only (contract rule 5)
    evidence = frame["evidence"].map(_normalize_evidence)
    is_lab = (evidence == LAB_EVIDENCE.lower()).to_numpy()
    ncbi_unlabelled = (frame["source"].eq(SOURCE_NCBI) & frame["evidence"].isna()).to_numpy()
    frame = droplog.keep_where(
        frame,
        is_lab | ncbi_unlabelled,
        Reason.EVIDENCE,
        detail=_top_detail(frame.loc[~(is_lab | ncbi_unlabelled), "evidence"].tolist()),
    )
    frame = frame.assign(evidence=LAB_EVIDENCE)

    # 2. drug and species
    drug = frame["antibiotic_raw"].map(config.normalize_drug)
    unknown_drugs = frame.loc[drug.isna(), "antibiotic_raw"].tolist()
    frame = frame.assign(drug=drug)
    frame = droplog.keep_where(frame, frame["drug"].notna(), Reason.UNKNOWN_DRUG, detail=_top_detail(unknown_drugs))
    known_species = frame["species"].isin(list(config.species)).to_numpy()
    frame = droplog.keep_where(
        frame,
        known_species,
        Reason.UNKNOWN_SPECIES,
        detail=_top_detail(frame.loc[~known_species, "species"].tolist()),
    )

    # 3. typing method
    method = frame["method_raw"].map(classify_method)
    unknown_methods = frame.loc[method.isna(), "method_raw"].tolist()
    frame = frame.assign(method=method)
    frame = droplog.keep_where(frame, frame["method"].notna(), Reason.UNKNOWN_METHOD, detail=_top_detail(unknown_methods))

    # 4 + 5. unit check and interval conversion, row by row
    converted, reasons, notes = _convert_rows(frame, config)
    for reason in _ROW_REASONS:
        detail = _top_detail(notes.missing_tables) if reason == Reason.NO_TABLE_FOR_YEAR else None
        droplog.drop(reason, int(sum(1 for r in reasons if r == reason)), detail=detail)
    # counts, not drops: kept numeric rows whose reported text was reinterpreted
    droplog.drop(Reason.COMBINATION, len(notes.combination), detail=_top_detail(notes.combination))
    droplog.drop(Reason.SNAPPED, len(notes.snapped), detail=_top_detail(notes.snapped))
    keep = np.array([r is None for r in reasons], dtype=bool)
    labels = pd.DataFrame(
        {column: [row[column] for row, ok in zip(converted, keep) if ok] for column in (*LABEL_COLUMNS, "evidence")},
        columns=[*LABEL_COLUMNS, "evidence"],
    )
    labels = _finalize_labels(labels, keep_evidence=True)

    # 6. one genome per biosample across sources
    labels = dedupe_biosamples(labels, droplog)

    # 7. one row per (genome_id, drug)
    labels = resolve_duplicates(labels, droplog)

    # 8. acceptance checks, then the contract columns only
    labels = _finalize_labels(labels, keep_evidence=True)
    check_labels(labels, config, stage=droplog.stage)
    labels = labels.loc[:, list(LABEL_COLUMNS)]
    labels = labels.sort_values(["genome_id", "drug"], kind="stable").reset_index(drop=True)

    logger.info(
        "harmonized %d raw rows into %d labels: %d genomes, %d species x drug pairs, %d rows dropped",
        n_in,
        len(labels),
        labels["genome_id"].nunique(),
        len(labels.drop_duplicates(["species", "drug"])) if not labels.empty else 0,
        droplog.total(),
    )
    return labels


@dataclass
class _ConversionNotes:
    """What :func:`_convert_rows` saw besides the per-row drop reasons (for the drop log)."""

    n_unknown_sir_numeric: int = 0
    combination: list[str] = field(default_factory=list)
    """Reported text of kept combination MICs (``16/4``)."""
    snapped: list[str] = field(default_factory=list)
    """Reported text of kept values that :func:`genome2mic.mic.snap_reported_mic` moved."""
    missing_tables: list[str] = field(default_factory=list)
    """``"<standard> <year>"`` of S/I/R-only rows dropped for :attr:`Reason.NO_TABLE_FOR_YEAR`."""


@dataclass(frozen=True)
class _RowOutcome:
    """Result of converting one raw row."""

    reason: str | None
    label: dict[str, Any] | None = None
    sir_ignored: bool = False
    combination: str | None = None
    snapped: str | None = None
    missing_table: str | None = None


def _convert_rows(
    frame: pd.DataFrame, config: Config
) -> tuple[list[dict[str, Any]], list[str | None], _ConversionNotes]:
    """Steps 4-5 for every row: ``(label dicts, per-row drop reason or None, notes)``."""
    converted: list[dict[str, Any]] = []
    reasons: list[str | None] = []
    notes = _ConversionNotes()
    for row in frame.to_dict("records"):
        outcome = _convert_row(row, config)
        reasons.append(outcome.reason)
        converted.append(outcome.label if outcome.label is not None else {})
        if outcome.missing_table is not None:
            notes.missing_tables.append(outcome.missing_table)
        if outcome.reason is not None:
            continue
        notes.n_unknown_sir_numeric += int(outcome.sir_ignored)
        if outcome.combination is not None:
            notes.combination.append(outcome.combination)
        if outcome.snapped is not None:
            notes.snapped.append(outcome.snapped)
    if notes.n_unknown_sir_numeric:
        logger.info(
            "%d numeric rows carried an unrecognized S/I/R value; the MIC was kept and sir set to null",
            notes.n_unknown_sir_numeric,
        )
    return converted, reasons, notes


def _convert_row(row: Mapping[str, Any], config: Config) -> _RowOutcome:
    """Convert one raw row (steps 4-5) into a label dict or a drop reason."""
    method = row["method"]
    sir_raw = row["sir_raw"]
    standard = normalize_standard(row["standard"])
    year = row["standard_year"]
    year_int = None if pd.isna(year) else int(year)

    sir_code: str | None = None
    sir_unknown = False
    if not is_missing(sir_raw):
        try:
            sir_code = micmod.normalize_sir(str(sir_raw))
        except ValueError:
            sir_unknown = True

    if method == METHOD_DISK:
        # A disk value is a zone diameter in mm. It is never an MIC.
        op, number, text = "=", None, clean_text(row["value"])
    else:
        try:
            op, number, text = _parse_measurement(row["sign"], row["value"])
        except ValueError as error:
            message = str(error)
            reason = Reason.BAD_SIGN if "sign" in message else Reason.BAD_VALUE
            return _RowOutcome(reason)

    base = {
        "genome_id": row["genome_id"],
        "biosample": row["biosample"],
        "species": row["species"],
        "drug": row["drug"],
        "method": method,
        "standard": standard,
        "standard_year": year_int,
        "source": row["source"],
        "isolation_source": row["isolation_source"],
        "country": row["country"],
        "year": None if pd.isna(row["year"]) else int(row["year"]),
        "evidence": LAB_EVIDENCE,
    }

    if method != METHOD_DISK and number is not None:
        # numeric path
        if normalize_unit(row["unit"]) is None:
            return _RowOutcome(Reason.UNKNOWN_UNIT)
        on_grid = micmod.snap_reported_mic(number)
        lo, hi, censor = micmod.interval_from_result(op, on_grid)
        label = {
            **base,
            "mic_lower": lo,
            "mic_upper": hi,
            "censor": censor,
            "sir": sir_code,
            "raw_result": f"{op}{text}",
        }
        return _RowOutcome(
            None,
            label,
            sir_ignored=sir_unknown,
            combination=text if text is not None and "/" in text else None,
            snapped=text if on_grid != number else None,
        )

    # S/I/R path (disk rows, and rows without a value): the breakpoint table must match
    # the row's standard and standard_year exactly (DATA_CONTRACT stage 2; no fallback).
    if is_missing(sir_raw):
        return _RowOutcome(Reason.NO_RESULT)
    if sir_code is None:
        return _RowOutcome(Reason.UNKNOWN_SIR)
    if standard is None:
        return _RowOutcome(Reason.NULL_STANDARD)
    if year_int is None:
        return _RowOutcome(Reason.NULL_YEAR)
    if not config.has_breakpoint_table(standard, year_int):
        return _RowOutcome(Reason.NO_TABLE_FOR_YEAR, missing_table=f"{standard} {year_int}")
    breakpoint = config.breakpoint(row["species"], row["drug"], standard, year_int)
    if breakpoint is None:
        return _RowOutcome(Reason.NO_BREAKPOINT)
    try:
        lo, hi, censor = micmod.interval_from_sir(sir_code, breakpoint)
    except ValueError:
        return _RowOutcome(Reason.INVALID_I)
    raw_result = sir_code if text is None else f"{sir_code} zone={text}"
    label = {
        **base,
        "mic_lower": lo,
        "mic_upper": hi,
        "censor": censor,
        "sir": sir_code,
        "raw_result": raw_result,
    }
    return _RowOutcome(None, label)


def _finalize_labels(frame: pd.DataFrame, *, keep_evidence: bool = False) -> pd.DataFrame:
    """Enforce the label dtypes: float64 bounds, Int64 years, object text with ``None``."""
    columns = [*LABEL_COLUMNS, "evidence"] if keep_evidence else list(LABEL_COLUMNS)
    n = len(frame)
    out: dict[str, pd.Series] = {}
    for column in columns:
        series = frame[column] if column in frame.columns else pd.Series([None] * n, index=frame.index, dtype=object)
        if column in LABEL_FLOAT_COLUMNS:
            out[column] = pd.to_numeric(series, errors="coerce").astype("float64")
        elif column in LABEL_INT_COLUMNS:
            out[column] = int64_nullable(series)
        else:
            out[column] = text_object(series)
    return pd.DataFrame(out, columns=columns, index=frame.index)


# --------------------------------------------------------------------------- #
# Step 6: biosample de-duplication
# --------------------------------------------------------------------------- #


def _canonical_genome_id(candidates: Iterable[tuple[str, str | None]]) -> str:
    """Pick the genome_id to keep for one biosample.

    Preference: a BV-BRC-format id (``573.2002``), then an id carried by BV-BRC
    rows, then the lexically smallest id (deterministic).
    """

    def rank(item: tuple[str, str | None]) -> tuple[int, int, str]:
        genome_id, source = item
        return (0 if _BVBRC_ID_RE.match(genome_id) else 1, 0 if source == SOURCE_BVBRC else 1, genome_id)

    return min(candidates, key=rank)[0]


def dedupe_biosamples(labels: pd.DataFrame, droplog: DropLog) -> pd.DataFrame:
    """Re-key rows so that one biosample maps to exactly one ``genome_id``.

    BV-BRC and NCBI share many isolates; the same genome under two ids would land
    in both train and test. Rows with a null biosample are left alone. The number
    of re-keyed rows is written to the drop log under :attr:`Reason.REKEYED` (it
    is a count, not a drop).
    """
    if labels.empty:
        droplog.drop(Reason.REKEYED, 0)
        return labels
    with_bs = labels[labels["biosample"].notna()]
    pairs = with_bs[["biosample", "genome_id", "source"]].drop_duplicates()
    per_bs = pairs.groupby("biosample", sort=False)["genome_id"].nunique()
    multi = per_bs[per_bs > 1].index.tolist()
    if not multi:
        droplog.drop(Reason.REKEYED, 0)
        return labels
    mapping: dict[tuple[str, str], str] = {}
    for biosample in multi:
        rows = pairs[pairs["biosample"] == biosample]
        canonical = _canonical_genome_id(zip(rows["genome_id"].tolist(), rows["source"].tolist()))
        for genome_id in rows["genome_id"].unique().tolist():
            if genome_id != canonical:
                mapping[(biosample, genome_id)] = canonical
    new_ids = [
        mapping.get((bs, gid), gid) if bs is not None else gid
        for bs, gid in zip(labels["biosample"].tolist(), labels["genome_id"].tolist())
    ]
    changed = int(sum(1 for old, new in zip(labels["genome_id"].tolist(), new_ids) if old != new))
    droplog.drop(
        Reason.REKEYED,
        changed,
        detail=f"{len(multi)} biosamples had more than one genome_id",
    )
    return labels.assign(genome_id=pd.Series(new_ids, index=labels.index, dtype=object))


# --------------------------------------------------------------------------- #
# Step 7: duplicate (genome_id, drug) resolution
# --------------------------------------------------------------------------- #

_MERGE_SR = "sr"
_MERGE_FAR = "far"
_MERGE_OK = "ok"


_PROVENANCE_COLUMNS: tuple[str, ...] = ("method", "standard", "standard_year", "source")
"""Columns that describe *how* a result was measured. A merged row takes all of them,
as one block, from its representative and never fills them from another row."""


def _representative_index(rows: list[dict[str, Any]], lows: np.ndarray, highs: np.ndarray,
                          new_lo: float, new_hi: float) -> int:
    """Row whose metadata describes the merged interval ``(new_lo, new_hi]``.

    1. Prefer rows whose own interval *equals* the merged interval (they support the
       bounds on their own); if none does (a partial-overlap intersection), all rows
       are candidates.
    2. Among the candidates prefer an MIC method (``dilution`` / ``gradient``) over
       ``disk``.
    3. Then the highest reported step (:func:`genome2mic.mic.label_point_log2`); ties
       keep the first row.

    Consequence: a merged one-step interval is never labelled ``disk`` when an MIC row
    reported that cell or contributed one of its bounds. It stays ``disk`` only when
    the disk row alone produced it -- a one-step ``I`` range such as CLSI meropenem
    ``(1, 2]`` that the other rows did not tighten -- because no dilution or gradient
    row measured it.
    """
    points = [micmod.label_point_log2(lo, hi) for lo, hi in zip(lows, highs)]
    equal = [i for i in range(len(rows)) if lows[i] == new_lo and highs[i] == new_hi]
    candidates = equal or list(range(len(rows)))
    return max(candidates, key=lambda i: (rows[i]["method"] != METHOD_DISK, points[i]))


def _merge_group(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    """Fold several results for one (genome_id, drug) into one label.

    * reported S and R in the same group -> ``("sr", None)``;
    * intervals that overlap -> their intersection;
    * disjoint but adjacent steps (``(4,8]`` and ``(8,16]``) -> the higher one;
    * further apart -> ``("far", None)``.

    ``method``, ``standard``, ``standard_year`` and ``source`` are copied as one block
    from the representative row chosen by :func:`_representative_index` (a row that
    supports the merged bounds, MIC methods before disk), so an exact MIC merged with a
    disk ``S`` keeps the dilution row's provenance. Genome-level columns (biosample,
    isolation source, country, year) are filled from the other rows when the
    representative has a null. ``sir`` is the single reported category when the rows
    agree, else the representative's (or, if it reported none, that of the highest-step
    row that did). ``raw_result`` joins every distinct original.
    """
    sirs = {row["sir"] for row in rows if row["sir"] is not None}
    if "S" in sirs and "R" in sirs:
        return _MERGE_SR, None
    lows = np.array([row["mic_lower"] for row in rows], dtype=float)
    highs = np.array([row["mic_upper"] for row in rows], dtype=float)
    inter_lo, inter_hi = float(lows.max()), float(highs.min())
    if inter_lo < inter_hi:
        new_lo, new_hi = inter_lo, inter_hi
    else:
        gap = math.log2(inter_lo) - math.log2(inter_hi)
        if gap > _ADJACENT_TOL:
            return _MERGE_FAR, None
        highest = max(range(len(rows)), key=lambda i: (lows[i], highs[i]))
        new_lo, new_hi = float(lows[highest]), float(highs[highest])
    representative = rows[_representative_index(rows, lows, highs, new_lo, new_hi)]
    merged = dict(representative)
    for column in merged:
        if column in _PROVENANCE_COLUMNS:
            continue
        if merged[column] is None or (isinstance(merged[column], float) and math.isnan(merged[column])):
            for row in rows:
                if row[column] is not None and not (isinstance(row[column], float) and math.isnan(row[column])):
                    merged[column] = row[column]
                    break
    merged["mic_lower"], merged["mic_upper"] = new_lo, new_hi
    merged["censor"] = micmod.censor_of(new_lo, new_hi)
    if len(sirs) == 1:
        merged["sir"] = next(iter(sirs))
    elif representative["sir"] is not None:
        merged["sir"] = representative["sir"]
    else:
        reported = [row for row in rows if row["sir"] is not None]
        if reported:
            top = max(reported, key=lambda r: micmod.label_point_log2(r["mic_lower"], r["mic_upper"]))
            merged["sir"] = top["sir"]
        else:
            merged["sir"] = None
    seen: list[str] = []
    for row in rows:
        if row["raw_result"] not in seen:
            seen.append(row["raw_result"])
    merged["raw_result"] = " | ".join(seen)
    return _MERGE_OK, merged


def resolve_duplicates(labels: pd.DataFrame, droplog: DropLog) -> pd.DataFrame:
    """Enforce one row per (``genome_id``, ``drug``) with the contract's duplicate rules.

    Writes three drop-log records: rows merged away, rows dropped for an S vs R
    conflict, rows dropped for results more than one doubling step apart.
    """
    key = ["genome_id", "drug"]
    if labels.empty:
        for reason in (Reason.DUP_MERGED, Reason.DUP_SR, Reason.DUP_FAR):
            droplog.drop(reason, 0)
        return labels
    is_dup = labels.duplicated(key, keep=False).to_numpy()
    singles = labels.loc[~is_dup]
    duplicates = labels.loc[is_dup]
    merged_rows: list[dict[str, Any]] = []
    n_merged = n_sr = n_far = 0
    pairs_sr: list[str] = []
    pairs_far: list[str] = []
    for (genome_id, drug), group in duplicates.groupby(key, sort=False):
        rows = _records_with_none(group)
        outcome, merged = _merge_group(rows)
        if outcome == _MERGE_SR:
            n_sr += len(rows)
            pairs_sr.append(f"{genome_id}/{drug}")
        elif outcome == _MERGE_FAR:
            n_far += len(rows)
            pairs_far.append(f"{genome_id}/{drug}")
        else:
            assert merged is not None
            merged_rows.append(merged)
            n_merged += len(rows) - 1
    droplog.drop(Reason.DUP_MERGED, n_merged, detail=f"{len(merged_rows)} pairs" if merged_rows else None)
    droplog.drop(Reason.DUP_SR, n_sr, detail=_pairs_detail(pairs_sr))
    droplog.drop(Reason.DUP_FAR, n_far, detail=_pairs_detail(pairs_far))
    if not merged_rows:
        return singles
    merged_frame = pd.DataFrame(merged_rows, columns=list(labels.columns))
    return pd.concat([singles, merged_frame], ignore_index=True)


def _pairs_detail(pairs: list[str]) -> str | None:
    if not pairs:
        return None
    shown = ", ".join(pairs[:_MAX_DETAIL_ITEMS])
    more = f", ... {len(pairs) - _MAX_DETAIL_ITEMS} more" if len(pairs) > _MAX_DETAIL_ITEMS else ""
    return f"{len(pairs)} pairs: {shown}{more}"


def _records_with_none(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """``to_dict('records')`` with every missing value as ``None`` (bounds stay floats)."""
    records = []
    for record in frame.to_dict("records"):
        cleaned = {}
        for column, value in record.items():
            if column in LABEL_FLOAT_COLUMNS:
                cleaned[column] = float(value)
            elif column in LABEL_INT_COLUMNS:
                cleaned[column] = None if pd.isna(value) else int(value)
            else:
                cleaned[column] = None if is_missing(value) else value
        records.append(cleaned)
    return records


# --------------------------------------------------------------------------- #
# Step 8: acceptance checks
# --------------------------------------------------------------------------- #


def check_labels(labels: pd.DataFrame, config: Config | None = None, stage: str = "ingest") -> None:
    """Acceptance checks for the stage-2 table; raise :class:`ContractViolation` on failure.

    Checks: all contract columns present; no nulls in required columns; category
    values valid; ``mic_lower < mic_upper``; ``censor == 'left'`` iff ``mic_lower == 0``;
    ``censor == 'right'`` iff ``mic_upper == inf``; (``genome_id``, ``drug``) unique;
    if an ``evidence`` column is present every value is ``Laboratory Method``; with a
    ``config``, species and drugs must be known.
    """

    def fail(message: str) -> None:
        raise ContractViolation(message, stage=stage)

    missing = [column for column in LABEL_COLUMNS if column not in labels.columns]
    if missing:
        fail(f"labels missing columns {missing}")

    for column in LABEL_REQUIRED_COLUMNS:
        n_null = int(labels[column].isna().sum())
        if n_null:
            fail(f"null values in required column {column}: {n_null} rows")

    if "evidence" in labels.columns:
        bad = labels["evidence"].astype(object) != LAB_EVIDENCE
        if bool(bad.any()):
            fail(f"{int(bad.sum())} rows with evidence != '{LAB_EVIDENCE}'")

    for column, allowed, nullable in (
        ("censor", CENSORS, False),
        ("sir", SIR_VALUES, True),
        ("method", METHODS, False),
        ("standard", STANDARDS, True),
        ("source", SOURCES, False),
    ):
        values = labels[column]
        ok = values.isin(list(allowed)) | (values.isna() if nullable else False)
        if not bool(ok.all()):
            offenders = sorted({str(v) for v in values[~ok].tolist()})[:_MAX_DETAIL_ITEMS]
            fail(f"invalid {column} values {offenders}; allowed {list(allowed)}")

    lo = pd.to_numeric(labels["mic_lower"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(labels["mic_upper"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(lo).any() or np.isnan(hi).any():
        fail("mic_lower / mic_upper must be numeric and non-null")
    if (lo < 0).any() or np.isinf(lo).any():
        fail(f"mic_lower must be finite and >= 0: {int(((lo < 0) | np.isinf(lo)).sum())} rows")
    bad = ~(lo < hi)
    if bad.any():
        fail(f"mic_lower < mic_upper violated on {int(bad.sum())} rows")
    censor = labels["censor"].to_numpy(dtype=object)
    is_left = censor == micmod.CENSOR_LEFT
    is_right = censor == micmod.CENSOR_RIGHT
    bad = is_left != (lo == 0)
    if bad.any():
        fail(f"censor == 'left' <=> mic_lower == 0 violated on {int(bad.sum())} rows")
    bad = is_right != np.isinf(hi)
    if bad.any():
        fail(f"censor == 'right' <=> mic_upper == inf violated on {int(bad.sum())} rows")

    duplicated = labels.duplicated(["genome_id", "drug"], keep=False)
    if bool(duplicated.any()):
        examples = labels.loc[duplicated, ["genome_id", "drug"]].drop_duplicates().head(_MAX_DETAIL_ITEMS)
        fail(
            f"(genome_id, drug) must be unique; {int(duplicated.sum())} rows in "
            f"{len(labels.loc[duplicated, ['genome_id', 'drug']].drop_duplicates())} duplicated keys, "
            f"e.g. {examples.to_dict('records')}"
        )

    if config is not None:
        unknown_species = sorted(set(labels["species"].dropna().tolist()) - set(config.species))
        if unknown_species:
            fail(f"unknown species {unknown_species}")
        unknown_drugs = sorted(set(labels["drug"].dropna().tolist()) - set(config.drugs))
        if unknown_drugs:
            fail(f"unknown drug names {unknown_drugs}")


# --------------------------------------------------------------------------- #
# Count table and pair inclusion
# --------------------------------------------------------------------------- #

COUNT_COLUMNS: tuple[str, ...] = (
    "species",
    "drug",
    "n",
    "n_R",
    "n_S",
    "n_I",
    "n_exact",
    "n_censored",
    "n_distinct_mic",
)
PAIRS_KEPT_COLUMNS: tuple[str, ...] = (
    "species",
    "drug",
    "n",
    "n_nonsusceptible",
    "n_susceptible",
    "n_distinct_mic",
)


def count_table(labels: pd.DataFrame) -> pd.DataFrame:
    """Per species x drug: ``n, n_R, n_S, n_I, n_exact, n_censored, n_distinct_mic``.

    ``n_R`` / ``n_S`` / ``n_I`` count the *reported* ``sir``. ``n_exact`` counts rows
    with an exact measured MIC: the interval pins the MIC to one doubling step
    (``mic_upper == 2 * mic_lower``) and the method is not disk diffusion
    (:func:`genome2mic.mic.lab_exact_mask`; a disk ``I``-only row whose ``I`` range is
    one step, e.g. CLSI meropenem ``(1, 2]``, is not an MIC). ``n_censored = n -
    n_exact``: left- and right-censored rows, multi-step intervals such as an
    ``I``-only ``(2, 8]`` (EUCAST meropenem), which have ``censor == 'interval'`` but do
    not pin the MIC, and those one-step disk rows. ``n_distinct_mic`` is the number
    of distinct reported steps (:func:`genome2mic.mic.label_point_log2`: the upper
    bound for interval and left-censored rows, the step above the edge for
    right-censored rows), so panel-edge readings count as levels.
    """
    int_columns = [column for column in COUNT_COLUMNS if column not in ("species", "drug")]
    if labels.empty:
        return pd.DataFrame(
            {
                "species": pd.Series(dtype=object),
                "drug": pd.Series(dtype=object),
                **{column: pd.Series(dtype="int64") for column in int_columns},
            },
            columns=list(COUNT_COLUMNS),
        )
    lo = pd.to_numeric(labels["mic_lower"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(labels["mic_upper"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        point = np.where(np.isfinite(hi), np.log2(hi), np.log2(lo) + 1.0)
    sir = labels["sir"].astype(object)
    work = pd.DataFrame(
        {
            "species": labels["species"].astype(object).to_numpy(),
            "drug": labels["drug"].astype(object).to_numpy(),
            "is_R": (sir == "R").to_numpy(),
            "is_S": (sir == "S").to_numpy(),
            "is_I": (sir == "I").to_numpy(),
            "is_exact": micmod.lab_exact_mask(lo, hi, labels["method"].to_numpy(dtype=object) if "method" in labels.columns else None),
            "point": np.round(point, 6),
        }
    )
    grouped = work.groupby(["species", "drug"], sort=True)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "n_R": grouped["is_R"].sum(),
            "n_S": grouped["is_S"].sum(),
            "n_I": grouped["is_I"].sum(),
            "n_exact": grouped["is_exact"].sum(),
            "n_distinct_mic": grouped["point"].nunique(),
        }
    ).reset_index()
    out["n_censored"] = out["n"] - out["n_exact"]
    out = out.loc[:, list(COUNT_COLUMNS)]
    for column in int_columns:
        out[column] = out[column].astype("int64")
    return out.reset_index(drop=True)


def pairs_kept(
    counts: pd.DataFrame,
    min_ns: int = 50,
    min_s: int = 50,
    min_levels: int = 4,
) -> pd.DataFrame:
    """Species x drug pairs passing the inclusion rule (contract stage 2).

    Keep a pair when it has at least ``min_ns`` non-susceptible rows (``n_R + n_I``),
    at least ``min_s`` susceptible rows (``n_S``) and at least ``min_levels`` distinct
    MIC levels. Columns: ``species, drug, n, n_nonsusceptible, n_susceptible,
    n_distinct_mic``.
    """
    _require_columns(counts, COUNT_COLUMNS, "pairs_kept input")
    table = pd.DataFrame(
        {
            "species": counts["species"].astype(object).to_numpy(),
            "drug": counts["drug"].astype(object).to_numpy(),
            "n": counts["n"].astype("int64").to_numpy(),
            "n_nonsusceptible": (counts["n_R"].astype("int64") + counts["n_I"].astype("int64")).to_numpy(),
            "n_susceptible": counts["n_S"].astype("int64").to_numpy(),
            "n_distinct_mic": counts["n_distinct_mic"].astype("int64").to_numpy(),
        },
        columns=list(PAIRS_KEPT_COLUMNS),
    )
    keep = (
        (table["n_nonsusceptible"] >= min_ns)
        & (table["n_susceptible"] >= min_s)
        & (table["n_distinct_mic"] >= min_levels)
    )
    kept = table.loc[keep].sort_values(["species", "drug"], kind="stable").reset_index(drop=True)
    dropped = table.loc[~keep]
    logger.info(
        "pair inclusion (>= %d non-susceptible, >= %d susceptible, >= %d MIC levels): kept %d of %d pairs%s",
        min_ns,
        min_s,
        min_levels,
        len(kept),
        len(table),
        "" if dropped.empty else "; excluded " + ", ".join(f"{s}/{d}" for s, d in zip(dropped["species"], dropped["drug"])),
    )
    return kept


__all__ = [
    "COUNT_COLUMNS",
    "DEFAULT_SPECIES_NAMES",
    "LABEL_COLUMNS",
    "LAB_EVIDENCE",
    "METADATA_COLUMNS",
    "METHODS",
    "PAIRS_KEPT_COLUMNS",
    "RAW_COLUMNS",
    "Reason",
    "SOURCES",
    "apply_metadata",
    "check_labels",
    "classify_method",
    "count_table",
    "country_from_text",
    "dedupe_biosamples",
    "finalize_raw",
    "harmonize",
    "normalize_header",
    "normalize_unit",
    "pairs_kept",
    "parse_measurement",
    "read_csv_text",
    "read_genome_metadata",
    "resolve_duplicates",
    "select_columns",
    "species_key_from_organism",
    "unknown_organism_summary",
    "year_from_text",
]
