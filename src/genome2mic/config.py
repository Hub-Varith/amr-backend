"""Load and query the project configs under ``configs/``.

Files read (see ``DATA_CONTRACT.md`` section 3 and ``.context/DESIGN.md``):

- ``species.yaml``            species keys, AMRFinderPlus organism names, QC sizes
- ``drugs.yaml``              drug names, synonyms, spectrum tiers, strong markers
- ``breakpoints/<std>_<year>.csv``  S / R MIC breakpoints per standard version
- ``natural_resistance.csv``  species x drug pairs that are always inactive
- ``keep_variant.csv``        gene families whose exact variant number is kept

Breakpoint convention (contract): ``S`` if ``mic <= s_breakpoint``, ``R`` if
``mic > r_breakpoint``, ``I`` otherwise. CLSI tables publish ``R >= x``; the CSV
stores ``x / 2`` so the same rule applies.

This module never drops data rows itself; it either loads a config or raises
``ConfigError``. ``Config.normalize_drug`` returns ``None`` for an unknown drug and
``Config.breakpoint`` returns ``None`` unless a table exists for exactly the row's
``(standard, standard_year)`` and has a row for the pair (no fallback to the latest
table). The caller (the ingest stage) owns the row filter and logs the dropped count
through ``DropLog``.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SPECIES_FILE = "species.yaml"
DRUGS_FILE = "drugs.yaml"
BREAKPOINTS_DIR = "breakpoints"
NATURAL_RESISTANCE_FILE = "natural_resistance.csv"
KEEP_VARIANT_FILE = "keep_variant.csv"
INTRINSIC_MARKERS_FILE = "intrinsic_markers.csv"

BREAKPOINT_COLUMNS: tuple[str, ...] = (
    "species",
    "drug",
    "s_breakpoint",
    "r_breakpoint",
    "version",
    "site",
    "note",
)
BREAKPOINT_SITE = "bloodstream"
NATURAL_RESISTANCE_COLUMNS: tuple[str, ...] = ("species", "drug", "note")
KEEP_VARIANT_COLUMNS: tuple[str, ...] = ("family_prefix", "note")
INTRINSIC_MARKERS_COLUMNS: tuple[str, ...] = ("species", "symbol", "family", "note")

STANDARDS: tuple[str, ...] = ("EUCAST", "CLSI")
STANDARD_ALIASES: Mapping[str, str] = {"NCCLS": "CLSI"}
SPECTRUM_TIERS: tuple[int, ...] = (1, 2, 3, 4)
SIR_VALUES: tuple[str, ...] = ("S", "I", "R")

_BREAKPOINT_FILE_RE = re.compile(r"^(eucast|clsi)_(\d{4})\.csv$")
_SEPARATOR_RE = re.compile(r"[\s/_\-]+")
_STRONG_MARKER_RE = re.compile(r"^(gene|point)_[a-z0-9_]+$")


class ConfigError(ValueError):
    """A config file is missing, malformed, or internally inconsistent."""


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpeciesConfig:
    """One species in scope (``species.yaml``)."""

    key: str
    name: str
    amrfinder_organism: str
    expected_genome_size: int
    size_tolerance: float
    reference_accession: str
    reference_sketch: str | None = None


@dataclass(frozen=True)
class DrugConfig:
    """One drug in the panel (``drugs.yaml``)."""

    name: str
    synonyms: tuple[str, ...]
    spectrum_tier: int
    strong_markers: tuple[str, ...]
    strong_subclasses: tuple[str, ...] = ()
    """AMRFinderPlus ``Subclass`` values (upper case) that force ``likely_inactive`` when an
    acquired gene (Subtype ``AMR``, never ``POINT``) carries them, e.g. ``CARBAPENEM``."""
    strong_marker_exceptions: tuple[str, ...] = ()
    """Column-name prefixes that never trigger the override even though a ``strong_markers``
    prefix matches them (e.g. ``gene_mcr_9`` under ``gene_mcr``: mcr-9 often leaves
    colistin MICs susceptible)."""

    def is_strong_column(self, column: str) -> bool:
        """``column`` matches a ``strong_markers`` prefix and no ``strong_marker_exceptions`` prefix.

        Matching is token-aware (:func:`column_prefix_matches`): ``gene_blaoxa_48`` matches
        ``gene_blaoxa_48`` but never ``gene_blaoxa_485`` (an OXA-50-family enzyme intrinsic
        to *P. aeruginosa*, not a carbapenemase).
        """
        text = str(column)
        if not any(column_prefix_matches(text, prefix) for prefix in self.strong_markers):
            return False
        return not any(column_prefix_matches(text, prefix) for prefix in self.strong_marker_exceptions)


@dataclass(frozen=True)
class Breakpoint:
    """S / R MIC breakpoints (mg/L) for one species x drug under one standard version.

    ``S`` if ``mic <= s_breakpoint``; ``R`` if ``mic > r_breakpoint``; ``I`` otherwise.
    """

    species: str
    drug: str
    s_breakpoint: float
    r_breakpoint: float
    standard: str
    version: str


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def column_prefix_matches(column: str, prefix: str) -> bool:
    """Token-aware column-prefix match used by every strong-marker / exception prefix.

    ``column == prefix`` always matches. Otherwise ``column`` must start with ``prefix`` and:

    * a prefix ending in a **digit** names one allele number, so the next character must be
      ``_`` (or the column ends): ``gene_blaoxa_48`` matches ``gene_blaoxa_48`` and
      ``gene_blaoxa_48_like`` but not ``gene_blaoxa_485`` / ``_486`` / ``_488``;
      ``gene_mcr_1`` would not match ``gene_mcr_10``;
    * a prefix ending in a **letter** (or ``_``) names a family, so any continuation
      matches: ``gene_blakpc`` -> ``gene_blakpc_2``; ``gene_rmtb`` -> ``gene_rmtb1``.
    """
    text, head = str(column), str(prefix)
    if not head:
        return False
    if text == head:
        return True
    if not text.startswith(head):
        return False
    if head[-1].isdigit():
        return text[len(head)] == "_"
    return True


def normalize_name(raw: Any) -> str | None:
    """Canonicalize a drug spelling without synonym mapping.

    Lowercase, strip, and collapse any run of whitespace, ``/``, ``_`` or ``-`` to a
    single ``-``. Returns ``None`` for null, non-string or empty input.

    >>> normalize_name(" Piperacillin / Tazobactam ")
    'piperacillin-tazobactam'
    """
    if raw is None or not isinstance(raw, str):
        return None
    text = _SEPARATOR_RE.sub("-", raw.strip().lower()).strip("-")
    return text or None


def normalize_standard(raw: Any) -> str | None:
    """Return ``EUCAST`` / ``CLSI`` for a raw standard string, else ``None``.

    Null, non-string, empty and unknown standards all map to ``None`` because the
    contract says an unknown standard must never be guessed.
    """
    if raw is None or not isinstance(raw, str):
        return None
    text = raw.strip().upper()
    if not text:
        return None
    text = STANDARD_ALIASES.get(text, text)
    return text if text in STANDARDS else None


def sir_from_mic(mic: float, bp: Breakpoint) -> str:
    """Return ``"S"``, ``"I"`` or ``"R"`` for an MIC under one breakpoint.

    ``S`` if ``mic <= s_breakpoint``; ``R`` if ``mic > r_breakpoint``; ``I`` otherwise.
    ``mic == s_breakpoint`` is ``S``; ``mic == r_breakpoint`` is ``I`` when ``s < r``
    (and ``S`` when ``s == r``). ``inf`` is ``R``. Raises ``ValueError`` for a MIC that
    is not a positive number.
    """
    try:
        value = float(mic)
    except (TypeError, ValueError) as error:
        raise ValueError(f"MIC must be a positive number, got {mic!r}") from error
    if math.isnan(value) or value <= 0:
        raise ValueError(f"MIC must be a positive number, got {mic!r}")
    if value <= bp.s_breakpoint:
        return "S"
    if value > bp.r_breakpoint:
        return "R"
    return "I"


def _coerce_year(year: Any) -> int | None:
    """Return ``year`` as an int, or ``None`` for null / NaN / unparseable values."""
    if year is None or isinstance(year, bool):
        return None
    try:
        value = int(year)
    except (TypeError, ValueError, OverflowError):
        return None
    return value


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """All project configs, validated and cross-referenced.

    Build with :func:`load_config`. Lookups never raise for unknown drugs,
    species or standards; they return ``None`` / ``False`` and leave the drop
    decision and its ``DropLog`` count to the caller.
    """

    species: dict[str, SpeciesConfig]
    drugs: dict[str, DrugConfig]
    call_standard: tuple[str, str]
    qc_max_contigs: int
    qc_max_mash_distance: float
    breakpoints: dict[tuple[str, str], dict[tuple[str, str], Breakpoint]] = field(repr=False)
    natural_resistance: frozenset[tuple[str, str]] = field(repr=False)
    keep_variant: tuple[str, ...] = field(repr=False)
    synonym_map: dict[str, str] = field(repr=False)
    intrinsic_markers: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    # -- drugs ------------------------------------------------------------- #

    def normalize_drug(self, raw: Any) -> str | None:
        """Map a raw drug spelling to its normalized name, or ``None`` if unknown.

        Lowercase, strip, collapse whitespace / ``/`` / ``_`` / ``-`` runs to ``-``,
        then resolve synonyms. The caller logs ``None`` results as dropped rows.
        """
        key = normalize_name(raw)
        if key is None:
            return None
        if key in self.drugs:
            return key
        return self.synonym_map.get(key)

    def drug_names(self) -> tuple[str, ...]:
        """All normalized drug names in ``drugs.yaml`` order."""
        return tuple(self.drugs)

    def species_keys(self) -> tuple[str, ...]:
        """All species keys in ``species.yaml`` order."""
        return tuple(self.species)

    # -- breakpoints ------------------------------------------------------- #

    def standards(self) -> dict[str, tuple[str, ...]]:
        """Available table versions per standard, oldest first."""
        out: dict[str, list[str]] = {}
        for standard, version in self.breakpoints:
            out.setdefault(standard, []).append(version)
        return {standard: tuple(sorted(versions, key=int)) for standard, versions in out.items()}

    def latest_version(self, standard: str) -> str | None:
        """Latest table version for a standard, or ``None`` if none is loaded."""
        versions = self.standards().get(standard)
        return versions[-1] if versions else None

    def has_breakpoint_table(self, standard: str | None, year: int | None) -> bool:
        """True when a breakpoint table is loaded for exactly ``(standard, year)``.

        ``standard`` goes through :func:`normalize_standard` (``NCCLS`` -> ``CLSI``);
        a null year, a null or unknown standard, or a year without a CSV give ``False``.
        The ingest stage uses this to tell "no table for that year" apart from "the
        table has no row for this species x drug" when it logs a dropped row.
        """
        std = normalize_standard(standard)
        year_int = _coerce_year(year)
        if std is None or year_int is None:
            return False
        return (std, str(year_int)) in self.breakpoints

    def breakpoint(
        self,
        species: str,
        drug: str,
        standard: str | None,
        year: int | None,
    ) -> Breakpoint | None:
        """Look up the breakpoint for ``species`` x ``drug`` under exactly ``(standard, year)``.

        DATA_CONTRACT stage 2: an S/I/R-only row needs "the breakpoint table matching
        that row's standard and standard_year ... Do not guess." So there is no
        fallback: a null year, a year without a loaded table, a null or unknown
        standard, or a table without a row for the pair all return ``None``. Breakpoints
        change between versions (EUCAST ciprofloxacin in 2017, CLSI fluoroquinolones in
        2019, ...), and converting an old S/I/R with a newer table would fabricate a
        tighter interval. Add ``configs/breakpoints/<std>_<year>.csv`` for the years in
        the data instead. :meth:`call_breakpoint` passes the explicit ``call_standard``
        version, whose table :func:`load_config` guarantees.
        """
        if not self.has_breakpoint_table(standard, year):
            return None
        std = normalize_standard(standard)
        version = str(_coerce_year(year))
        return self.breakpoints[(std, version)].get((species, drug))  # type: ignore[index]

    def call_breakpoint(self, species: str, drug: str) -> Breakpoint | None:
        """Breakpoint under ``call_standard`` (used for ``pred_sir`` and the call)."""
        standard, version = self.call_standard
        return self.breakpoint(species, drug, standard, _coerce_year(version))

    def breakpoint_rows(self) -> tuple[Breakpoint, ...]:
        """Every loaded breakpoint row, in table then file order."""
        return tuple(bp for table in self.breakpoints.values() for bp in table.values())

    def sir_from_mic(self, mic: float, bp: Breakpoint) -> str:
        """See :func:`sir_from_mic`."""
        return sir_from_mic(mic, bp)

    # -- overrides --------------------------------------------------------- #

    def is_naturally_resistant(self, species: str, drug: str) -> bool:
        """True if ``natural_resistance.csv`` lists the pair (drug may be un-normalized)."""
        if (species, drug) in self.natural_resistance:
            return True
        normalized = self.normalize_drug(drug)
        return normalized is not None and (species, normalized) in self.natural_resistance

    def intrinsic_symbols(self, species: str | None) -> frozenset[str]:
        """Lower-cased AMRFinderPlus symbols intrinsic to ``species`` (``intrinsic_markers.csv``).

        These chromosomal genes never trigger the strong-marker override (e.g. the
        OXA-51-like genes every *A. baumannii* carries are reported with Subclass
        ``CARBAPENEM``). They stay model features.
        """
        if species is None:
            return frozenset()
        return self.intrinsic_markers.get(species, frozenset())

    def is_intrinsic_marker(self, species: str | None, symbol: str) -> bool:
        """True if ``symbol`` is an intrinsic chromosomal gene of ``species`` (case-insensitive)."""
        return str(symbol).strip().lower() in self.intrinsic_symbols(species)

    def keep_variant_prefixes(self) -> tuple[str, ...]:
        """Gene-family prefixes whose exact variant number is kept as a feature."""
        return self.keep_variant


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_config(configs_dir: Path | str) -> Config:
    """Load, validate and cross-reference every config file under ``configs_dir``.

    Raises ``ConfigError`` on a missing file, a wrong shape, an unknown species or
    drug reference, a duplicate synonym, an invalid tier, or a breakpoint row with
    ``s_breakpoint > r_breakpoint``.
    """
    root = Path(configs_dir)
    if not root.is_dir():
        raise ConfigError(f"configs dir not found: {root}")

    species, qc_max_contigs, qc_max_mash_distance = _load_species(root / SPECIES_FILE)
    drugs, call_standard, synonym_map = _load_drugs(root / DRUGS_FILE)
    breakpoints = _load_breakpoints(root / BREAKPOINTS_DIR, species, drugs)
    natural_resistance = _load_natural_resistance(root / NATURAL_RESISTANCE_FILE, species, drugs)
    keep_variant = _load_keep_variant(root / KEEP_VARIANT_FILE)
    intrinsic_markers = _load_intrinsic_markers(root / INTRINSIC_MARKERS_FILE, species)

    if call_standard not in breakpoints:
        raise ConfigError(
            f"{DRUGS_FILE}: call_standard {call_standard} has no breakpoint table; "
            f"available: {sorted(breakpoints)}"
        )

    config = Config(
        species=species,
        drugs=drugs,
        call_standard=call_standard,
        qc_max_contigs=qc_max_contigs,
        qc_max_mash_distance=qc_max_mash_distance,
        breakpoints=breakpoints,
        natural_resistance=natural_resistance,
        keep_variant=keep_variant,
        synonym_map=synonym_map,
        intrinsic_markers=intrinsic_markers,
    )
    logger.info(
        "Loaded configs from %s: %d species, %d drugs, %d synonyms, %d breakpoint tables "
        "(%d rows), %d natural-resistance pairs, %d keep-variant prefixes, call standard %s %s",
        root,
        len(species),
        len(drugs),
        len(synonym_map),
        len(breakpoints),
        sum(len(table) for table in breakpoints.values()),
        len(natural_resistance),
        len(keep_variant),
        call_standard[0],
        call_standard[1],
    )
    return config


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path.name}: invalid YAML: {error}") from error
    if not isinstance(content, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    return content


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: '{key}' must be a non-empty string")
    return value.strip()


def _require_number(mapping: Mapping[str, Any], key: str, where: str, *, minimum: float = 0.0) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: '{key}' must be a number")
    if not math.isfinite(value) or value <= minimum:
        raise ConfigError(f"{where}: '{key}' must be > {minimum}, got {value}")
    return float(value)


def _load_species(path: Path) -> tuple[dict[str, SpeciesConfig], int, float]:
    content = _read_yaml_mapping(path)
    raw_species = _require_mapping(content.get("species"), f"{path.name}: species")
    if not raw_species:
        raise ConfigError(f"{path.name}: no species defined")
    species: dict[str, SpeciesConfig] = {}
    for key, raw in raw_species.items():
        where = f"{path.name}: species.{key}"
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z]{3,6}", key):
            raise ConfigError(f"{where}: species key must be 3-6 uppercase letters")
        entry = _require_mapping(raw, where)
        size = _require_number(entry, "expected_genome_size", where)
        if size != int(size):
            raise ConfigError(f"{where}: expected_genome_size must be an integer number of bp")
        tolerance = _require_number(entry, "size_tolerance", where)
        if tolerance >= 1:
            raise ConfigError(f"{where}: size_tolerance must be a fraction < 1")
        sketch = entry.get("reference_sketch")
        if sketch is not None and (not isinstance(sketch, str) or not sketch.strip()):
            raise ConfigError(f"{where}: reference_sketch must be null or a path string")
        species[key] = SpeciesConfig(
            key=key,
            name=_require_str(entry, "name", where),
            amrfinder_organism=_require_str(entry, "amrfinder_organism", where),
            expected_genome_size=int(size),
            size_tolerance=tolerance,
            reference_accession=_require_str(entry, "reference_accession", where),
            reference_sketch=sketch.strip() if isinstance(sketch, str) else None,
        )
    qc = _require_mapping(content.get("qc"), f"{path.name}: qc")
    max_contigs = _require_number(qc, "max_contigs", f"{path.name}: qc")
    if max_contigs != int(max_contigs):
        raise ConfigError(f"{path.name}: qc.max_contigs must be an integer")
    max_mash = _require_number(qc, "max_mash_distance", f"{path.name}: qc")
    if max_mash >= 1:
        raise ConfigError(f"{path.name}: qc.max_mash_distance must be < 1")
    return species, int(max_contigs), max_mash


def _load_drugs(path: Path) -> tuple[dict[str, DrugConfig], tuple[str, str], dict[str, str]]:
    content = _read_yaml_mapping(path)
    where_cs = f"{path.name}: call_standard"
    raw_cs = _require_mapping(content.get("call_standard"), where_cs)
    standard = normalize_standard(raw_cs.get("standard"))
    if standard is None:
        raise ConfigError(f"{where_cs}: standard must be one of {STANDARDS}")
    version_raw = raw_cs.get("version")
    version = _coerce_year(version_raw)
    if version is None or not isinstance(version_raw, (str, int)) or isinstance(version_raw, bool):
        raise ConfigError(f"{where_cs}: version must be a 4-digit year string, got {version_raw!r}")
    call_standard = (standard, str(version))

    raw_drugs = _require_mapping(content.get("drugs"), f"{path.name}: drugs")
    if not raw_drugs:
        raise ConfigError(f"{path.name}: no drugs defined")

    drugs: dict[str, DrugConfig] = {}
    synonym_map: dict[str, str] = {}
    for name, raw in raw_drugs.items():
        where = f"{path.name}: drugs.{name}"
        if not isinstance(name, str) or normalize_name(name) != name:
            raise ConfigError(f"{where}: drug key must already be normalized (lowercase, hyphenated)")
        entry = _require_mapping(raw, where)
        tier = entry.get("spectrum_tier")
        if isinstance(tier, bool) or not isinstance(tier, int) or tier not in SPECTRUM_TIERS:
            raise ConfigError(f"{where}: spectrum_tier must be one of {SPECTRUM_TIERS}, got {tier!r}")
        synonyms = _string_list(entry.get("synonyms", []), f"{where}: synonyms")
        markers = _string_list(entry.get("strong_markers", []), f"{where}: strong_markers")
        for marker in markers:
            if not _STRONG_MARKER_RE.fullmatch(marker):
                raise ConfigError(
                    f"{where}: strong marker {marker!r} must be a gene_/point_ feature column name"
                )
        subclasses = _string_list(entry.get("strong_subclasses", []), f"{where}: strong_subclasses")
        exceptions = _string_list(entry.get("strong_marker_exceptions", []), f"{where}: strong_marker_exceptions")
        for exception in exceptions:
            if not _STRONG_MARKER_RE.fullmatch(exception):
                raise ConfigError(
                    f"{where}: strong marker exception {exception!r} must be a gene_/point_ feature column name"
                )
            if not any(column_prefix_matches(exception, prefix) for prefix in markers):
                raise ConfigError(
                    f"{where}: strong marker exception {exception!r} does not narrow any strong_markers prefix"
                )
        drugs[name] = DrugConfig(
            name=name,
            synonyms=tuple(synonyms),
            spectrum_tier=tier,
            strong_markers=tuple(markers),
            strong_subclasses=tuple(s.strip().upper() for s in subclasses if s.strip()),
            strong_marker_exceptions=tuple(exceptions),
        )

    # Synonym index, built after all names are known so collisions are detected.
    for name, drug in drugs.items():
        for synonym in drug.synonyms:
            key = normalize_name(synonym)
            if key is None:
                raise ConfigError(f"{path.name}: drugs.{name}: empty synonym")
            if key in drugs and key != name:
                raise ConfigError(f"{path.name}: drugs.{name}: synonym {synonym!r} is another drug's name")
            owner = synonym_map.get(key)
            if owner is not None and owner != name:
                raise ConfigError(
                    f"{path.name}: synonym {synonym!r} maps to both {owner!r} and {name!r}"
                )
            synonym_map[key] = name
    return drugs, call_standard, synonym_map


def _string_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigError(f"{where}: must be a list of non-empty strings")
    return [item.strip() for item in value]


def _read_csv_rows(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in required_columns if column not in columns]
        if missing:
            raise ConfigError(f"{path.name}: missing columns {missing}; found {list(columns)}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ConfigError(f"{path.name}: line {line_no} has the wrong number of fields")
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return rows


def _parse_mic(value: str, where: str) -> float:
    try:
        mic = float(value)
    except ValueError as error:
        raise ConfigError(f"{where}: MIC {value!r} is not a number") from error
    if not math.isfinite(mic) or mic <= 0:
        raise ConfigError(f"{where}: MIC must be a finite positive number, got {value!r}")
    return mic


def _load_breakpoints(
    directory: Path,
    species: Mapping[str, SpeciesConfig],
    drugs: Mapping[str, DrugConfig],
) -> dict[tuple[str, str], dict[tuple[str, str], Breakpoint]]:
    if not directory.is_dir():
        raise ConfigError(f"breakpoints dir not found: {directory}")
    tables: dict[tuple[str, str], dict[tuple[str, str], Breakpoint]] = {}
    for path in sorted(directory.glob("*.csv")):
        match = _BREAKPOINT_FILE_RE.match(path.name)
        if match is None:
            raise ConfigError(
                f"{path.name}: breakpoint files must be named <eucast|clsi>_<year>.csv"
            )
        standard = match.group(1).upper()
        version = match.group(2)
        table: dict[tuple[str, str], Breakpoint] = {}
        for line_no, row in enumerate(_read_csv_rows(path, BREAKPOINT_COLUMNS), start=2):
            where = f"{path.name}: line {line_no}"
            if row["species"] not in species:
                raise ConfigError(f"{where}: unknown species {row['species']!r}")
            if row["drug"] not in drugs:
                raise ConfigError(f"{where}: unknown drug {row['drug']!r} (must be a normalized name)")
            if row["version"] != version:
                raise ConfigError(f"{where}: version {row['version']!r} does not match filename")
            if row["site"] != BREAKPOINT_SITE:
                raise ConfigError(f"{where}: site must be {BREAKPOINT_SITE!r}, got {row['site']!r}")
            if not row["note"]:
                raise ConfigError(f"{where}: note must not be empty")
            s_bp = _parse_mic(row["s_breakpoint"], f"{where}: s_breakpoint")
            r_bp = _parse_mic(row["r_breakpoint"], f"{where}: r_breakpoint")
            if s_bp > r_bp:
                raise ConfigError(f"{where}: s_breakpoint {s_bp} > r_breakpoint {r_bp}")
            key = (row["species"], row["drug"])
            if key in table:
                raise ConfigError(f"{where}: duplicate row for {key}")
            table[key] = Breakpoint(
                species=key[0],
                drug=key[1],
                s_breakpoint=s_bp,
                r_breakpoint=r_bp,
                standard=standard,
                version=version,
            )
        if not table:
            raise ConfigError(f"{path.name}: no breakpoint rows")
        tables[(standard, version)] = table
        logger.info("Loaded breakpoint table %s %s: %d rows", standard, version, len(table))
    if not tables:
        raise ConfigError(f"no breakpoint tables found in {directory}")
    return tables


def _load_natural_resistance(
    path: Path,
    species: Mapping[str, SpeciesConfig],
    drugs: Mapping[str, DrugConfig],
) -> frozenset[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for line_no, row in enumerate(_read_csv_rows(path, NATURAL_RESISTANCE_COLUMNS), start=2):
        where = f"{path.name}: line {line_no}"
        if row["species"] not in species:
            raise ConfigError(f"{where}: unknown species {row['species']!r}")
        if row["drug"] not in drugs:
            raise ConfigError(f"{where}: unknown drug {row['drug']!r} (must be a normalized name)")
        key = (row["species"], row["drug"])
        if key in pairs:
            raise ConfigError(f"{where}: duplicate row for {key}")
        pairs.add(key)
    return frozenset(pairs)


def _load_intrinsic_markers(path: Path, species: Mapping[str, SpeciesConfig]) -> dict[str, frozenset[str]]:
    """``species -> lower-cased symbols`` from ``intrinsic_markers.csv``; optional file (absent -> empty)."""
    if not path.is_file():
        logger.warning("%s not found: no intrinsic-gene exclusion from the strong-marker override", path.name)
        return {}
    out: dict[str, set[str]] = {}
    for line_no, row in enumerate(_read_csv_rows(path, INTRINSIC_MARKERS_COLUMNS), start=2):
        where = f"{path.name}: line {line_no}"
        if row["species"] not in species:
            raise ConfigError(f"{where}: unknown species {row['species']!r}")
        symbol = (row["symbol"] or "").strip().lower()
        if not symbol:
            raise ConfigError(f"{where}: empty symbol")
        bucket = out.setdefault(row["species"], set())
        if symbol in bucket:
            raise ConfigError(f"{where}: duplicate symbol {row['symbol']!r} for {row['species']}")
        bucket.add(symbol)
    logger.info("Loaded %s: %s", path.name, {k: len(v) for k, v in sorted(out.items())})
    return {k: frozenset(v) for k, v in out.items()}


def _load_keep_variant(path: Path) -> tuple[str, ...]:
    prefixes: list[str] = []
    for line_no, row in enumerate(_read_csv_rows(path, KEEP_VARIANT_COLUMNS), start=2):
        where = f"{path.name}: line {line_no}"
        prefix = row["family_prefix"]
        if not prefix:
            raise ConfigError(f"{where}: empty family_prefix")
        if prefix in prefixes:
            raise ConfigError(f"{where}: duplicate family_prefix {prefix!r}")
        prefixes.append(prefix)
    if not prefixes:
        raise ConfigError(f"{path.name}: no keep-variant prefixes")
    return tuple(prefixes)


__all__ = [
    "BREAKPOINT_COLUMNS",
    "BREAKPOINT_SITE",
    "Breakpoint",
    "Config",
    "ConfigError",
    "DrugConfig",
    "SPECTRUM_TIERS",
    "STANDARDS",
    "SpeciesConfig",
    "column_prefix_matches",
    "load_config",
    "normalize_name",
    "normalize_standard",
    "sir_from_mic",
]
