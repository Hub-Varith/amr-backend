"""Known-AMR detection for one new genome and its conversion to a feature row.

Backends, tried in this order by :func:`detect` (``.context/DESIGN.md`` predict step 3):

1. :class:`AmrFinderCli` -- runs ``amrfinder`` when it is on ``PATH``.
2. :class:`PrecomputedAmrFinder` -- reads an AMRFinderPlus TSV given explicitly or
   found as a sidecar ``<fasta>.amrfinder.tsv`` next to the genome.
3. :class:`MarkerScan` -- exact substring search for the marker sequences in
   ``models/markers.fasta`` (written by the synthetic run). Headers are
   AMRFinder-style: ``>blaKPC-2 BETA-LACTAM CARBAPENEM AMR`` =
   ``symbol class subclass subtype``, or ``key=value`` tokens
   (``type=AMR subtype=AMR class=... subclass=...``). The prediction pipeline only
   passes ``markers.fasta`` for bundles whose ``manifest.json`` says
   ``synthetic: true``: an exact full-length substring match misses any marker
   split across contigs or carrying one SNP, so a real bundle never falls back to it.
4. Otherwise :class:`genome2mic.errors.ToolNotAvailable`.

Every backend returns the same table (:data:`DETECTION_COLUMNS`), one row per hit,
so the rest of the pipeline never knows which backend ran.

Rows must match training exactly (``features/known_amr.py``):

* AMRFinderPlus tables are read by the training parser itself, so the row filter is
  identical: ``Type == AMR`` (exact; blank, ``STRESS`` and ``VIRULENCE`` are dropped),
  ``Subtype`` in ``{AMR, POINT}`` (``AMR-SUSCEPTIBLE`` etc. are dropped), a non-null
  symbol; ``NA`` cells are null. Each drop is counted in the ``DropLog``.
* ``markers.fasta`` records honour ``type=`` / ``subtype=`` the same way.
* ``family_of`` collapses a symbol to its gene family unless its prefix is in
  ``keep_variant.csv``, and ``column_name`` turns ``prefix + family`` into a column
  such as ``gene_blactx_m``. Those helpers are imported from
  ``genome2mic.features.known_amr``; local implementations of the same documented
  rule remain as a fallback (a warning is logged once if the import fails).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from genome2mic.config import Config
from genome2mic.droplog import DropLog
from genome2mic.errors import ToolNotAvailable
from genome2mic.features import known_amr as _training_amr
from genome2mic.io import read_fasta

logger = logging.getLogger(__name__)

__all__ = [
    "DETECTION_COLUMNS",
    "SUBTYPE_AMR",
    "SUBTYPE_POINT",
    "AmrDetector",
    "AmrFinderCli",
    "KnownAmrRow",
    "Marker",
    "MarkerScan",
    "PrecomputedAmrFinder",
    "column_name",
    "detect",
    "family_of",
    "feature_vector",
    "known_amr_row",
    "markers_from_frame",
    "parse_amrfinder_tsv",
    "parse_marker_header",
    "sidecar_path",
]

SUBTYPE_AMR = "AMR"
SUBTYPE_POINT = "POINT"
TYPE_AMR = "AMR"
RESISTANCE_SUBTYPES: frozenset[str] = frozenset({SUBTYPE_AMR, SUBTYPE_POINT})
"""Subtypes that become ``gene_`` / ``point_`` features (training drops every other one)."""
NON_AMR_TYPES: frozenset[str] = frozenset({"STRESS", "VIRULENCE"})

DETECTION_COLUMNS: tuple[str, ...] = (
    "symbol",
    "type",
    "subtype",
    "class",
    "subclass",
    "method",
    "coverage",
    "identity",
    "backend",
)
"""Columns of the table every detector returns (AMRFinderPlus names, snake_case)."""

SIDECAR_SUFFIX = ".amrfinder.tsv"

AMRFINDER_THREADS = 4
"""Threads for the default :class:`AmrFinderCli` backend."""

TsvParser = Callable[..., pd.DataFrame]
"""``(tsv_path, droplog, backend=...) -> detection table`` (:data:`DETECTION_COLUMNS`)."""

PREFIX_GENE = "gene_"
PREFIX_POINT = "point_"
PREFIX_CLASS = "n_class_"

_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")
_VARIANT_SUFFIX_RE = re.compile(r"-\d+$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_POINT_SYMBOL_RE = re.compile(r"^[A-Za-z0-9()'\-]+_[A-Za-z]\d+[A-Za-z]+$")
# Header tokens: ``key="value with spaces"`` or any run of non-space characters
# (``aac(6')-Ib-cr`` contains a quote, so shlex cannot be used).
_HEADER_TOKEN_RE = re.compile(r'[^\s=]+="[^"]*"|\S+')
_HEADER_NULLS = frozenset({"", "NA", "N/A", "-", ".", "NAN", "NONE", "<NA>"})


# --------------------------------------------------------------------------- #
# Marker record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Marker:
    """One detected known-AMR determinant.

    Attributes:
        symbol: AMRFinder element symbol (``blaKPC-2``, ``gyrA_S83L``).
        subtype: ``AMR`` (acquired gene) or ``POINT`` (resistance mutation). A
            ``markers.fasta`` header may carry another subtype (``VIRULENCE``,
            ``AMR-SUSCEPTIBLE``); such markers are never features.
        amr_class: AMRFinder ``Class`` (``BETA-LACTAM``), or ``None`` if unreported.
        subclass: AMRFinder ``Subclass`` (``CARBAPENEM``), or ``None``.
        method: AMRFinder method (``EXACTX``) or the scan method.
        backend: Which detector produced the hit.
        column: Known-AMR feature column (``gene_blakpc_2``); set by :func:`known_amr_row`.
        element_type: AMRFinder ``Type`` (``AMR``, ``STRESS``, ``VIRULENCE``).
    """

    symbol: str
    subtype: str
    amr_class: str | None = None
    subclass: str | None = None
    method: str | None = None
    backend: str | None = None
    column: str | None = None
    element_type: str = TYPE_AMR

    @property
    def is_resistance_feature(self) -> bool:
        """True if training would turn this hit into a ``gene_`` / ``point_`` feature."""
        return self.element_type == TYPE_AMR and self.subtype in RESISTANCE_SUBTYPES


def _clean_text(value: object) -> str | None:
    """Null-safe string cleaning: ``None`` for null / blank / ``nan`` / ``NA``."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "<NA>"}:
        return None
    return text


def markers_from_frame(frame: pd.DataFrame) -> tuple[Marker, ...]:
    """Convert a detection table into :class:`Marker` records (one per kept row).

    The built-in backends already apply the training row filter; rows from a custom
    detector are held to the same rule here: a non-null ``type`` other than ``AMR`` or
    a non-null ``subtype`` outside ``{AMR, POINT}`` is skipped (and counted in the log).
    A null ``subtype`` is read as ``AMR`` (hand-built tables).
    """
    out: list[Marker] = []
    n_skipped = 0
    for record in frame.to_dict("records"):
        symbol = _clean_text(record.get("symbol"))
        if symbol is None:
            continue
        element_type = (_clean_text(record.get("type")) or TYPE_AMR).upper()
        subtype = (_clean_text(record.get("subtype")) or SUBTYPE_AMR).upper()
        if element_type != TYPE_AMR or subtype not in RESISTANCE_SUBTYPES:
            n_skipped += 1
            continue
        out.append(
            Marker(
                symbol=symbol,
                subtype=subtype,
                amr_class=_clean_text(record.get("class")),
                subclass=_clean_text(record.get("subclass")),
                method=_clean_text(record.get("method")),
                backend=_clean_text(record.get("backend")),
            )
        )
    if n_skipped:
        logger.info("Skipped %d detection row(s) that are not Type AMR with Subtype AMR/POINT", n_skipped)
    return tuple(out)


def _empty_detections() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series([], dtype="str") for column in DETECTION_COLUMNS})


def _frame_from_markers(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=list(DETECTION_COLUMNS))
    if frame.empty:
        return _empty_detections()
    for column in ("coverage", "identity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in DETECTION_COLUMNS:
        if column not in ("coverage", "identity"):
            frame[column] = frame[column].astype("str")
    return frame


# --------------------------------------------------------------------------- #
# AMRFinderPlus TSV parsing
# --------------------------------------------------------------------------- #


def parse_amrfinder_tsv(
    path: Path,
    droplog: DropLog | None = None,
    backend: str = "precomputed",
) -> pd.DataFrame:
    """Parse an AMRFinderPlus 4.x (or 3.x) TSV into :data:`DETECTION_COLUMNS`.

    The rows are read by the stage-5 training parser (``features.known_amr``), so the
    filter is exactly the one the models were trained with: ``Type == AMR`` (exact
    match; blank, ``STRESS``, ``VIRULENCE`` dropped), ``Subtype`` in ``{AMR, POINT}``
    (``AMR-SUSCEPTIBLE`` etc. dropped) and a non-null symbol; ``NA`` cells are null.
    Both header variants are accepted. Drop counts go to ``droplog`` under
    ``amrfinder_type_not_amr``, ``amrfinder_subtype_not_amr_or_point`` and
    ``amrfinder_no_symbol``.

    Raises:
        ValueError: empty file or a header that is not an AMRFinderPlus header.
    """
    source = Path(path)
    # The private row reader is used on purpose: it is the single definition of the
    # training row filter and it returns the per-reason counts the drop log needs.
    hits, counts = _training_amr._parse_rows(source)
    drops = (
        ("amrfinder_type_not_amr", counts.n_non_amr),
        ("amrfinder_subtype_not_amr_or_point", counts.n_other_subtype),
        ("amrfinder_no_symbol", counts.n_no_symbol),
    )
    for reason, n_dropped in drops:
        if droplog is not None:
            droplog.drop(reason, n_dropped, detail=str(source))
        elif n_dropped:
            logger.info("Dropped %d AMRFinder row(s) from %s: %s", n_dropped, source, reason)
    frame = _frame_from_markers(
        {
            "symbol": hit.symbol,
            "type": TYPE_AMR,
            "subtype": hit.subtype,
            "class": hit.amr_class,
            "subclass": hit.subclass,
            "method": hit.method,
            "coverage": hit.coverage,
            "identity": hit.identity,
            "backend": backend,
        }
        for hit in hits
    )
    logger.info("Parsed %d AMR hit(s) from %d row(s) of %s", len(frame), counts.n_rows, source)
    return frame


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class AmrDetector(Protocol):
    """A known-AMR detection backend."""

    name: str

    def available(self, fasta: Path) -> bool:
        """True if this backend can run for ``fasta`` right now."""
        ...

    def detect(self, fasta: Path, species: str | None, config: Config, droplog: DropLog | None = None) -> pd.DataFrame:
        """Return the detection table (:data:`DETECTION_COLUMNS`)."""
        ...


class AmrFinderCli:
    """Run the ``amrfinder`` command-line tool (nucleotide mode, ``--plus``)."""

    name = "amrfinder_cli"

    def __init__(self, executable: str = "amrfinder", threads: int = 1, parser: TsvParser | None = None) -> None:
        self.executable = executable
        self.threads = int(threads)
        self.parser: TsvParser = parser if parser is not None else parse_amrfinder_tsv

    def available(self, fasta: Path) -> bool:
        return shutil.which(self.executable) is not None

    def detect(self, fasta: Path, species: str | None, config: Config, droplog: DropLog | None = None) -> pd.DataFrame:
        exe = shutil.which(self.executable)
        if exe is None:
            raise ToolNotAvailable(self.executable, hint="Install AMRFinderPlus or provide <fasta>.amrfinder.tsv.")
        with tempfile.TemporaryDirectory(prefix="g2m_amrfinder_") as tmp:
            out = Path(tmp) / "amrfinder.tsv"
            command = [exe, "-n", str(fasta), "--plus", "--threads", str(self.threads), "-o", str(out)]
            if species is not None and species in config.species:
                command += ["-O", config.species[species].amrfinder_organism]
            logger.info("Running %s", " ".join(command))
            subprocess.run(command, check=True, capture_output=True, text=True)
            return self.parser(out, droplog, backend=self.name)


def sidecar_path(fasta: Path) -> Path:
    """``<fasta>.amrfinder.tsv`` next to the genome (``genome.fasta.amrfinder.tsv``)."""
    fasta = Path(fasta)
    return fasta.with_name(fasta.name + SIDECAR_SUFFIX)


class PrecomputedAmrFinder:
    """Read an AMRFinderPlus TSV given explicitly or found as a sidecar file."""

    name = "precomputed"

    def __init__(self, tsv_path: Path | None = None, parser: TsvParser | None = None) -> None:
        self.tsv_path = Path(tsv_path) if tsv_path is not None else None
        self.parser: TsvParser = parser if parser is not None else parse_amrfinder_tsv

    def resolve(self, fasta: Path) -> Path | None:
        """The TSV to read: the explicit path if given, else the sidecar if it exists."""
        if self.tsv_path is not None:
            return self.tsv_path
        candidate = sidecar_path(fasta)
        return candidate if candidate.is_file() else None

    def available(self, fasta: Path) -> bool:
        path = self.resolve(fasta)
        return path is not None and path.is_file()

    def detect(self, fasta: Path, species: str | None, config: Config, droplog: DropLog | None = None) -> pd.DataFrame:
        path = self.resolve(fasta)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"No AMRFinder TSV for {fasta} (looked for {sidecar_path(fasta)})")
        return self.parser(path, droplog, backend=self.name)


def _header_value(value: str | None) -> str | None:
    """Upper-cased header value, ``None`` for ``NA`` / blank placeholders."""
    if value is None:
        return None
    text = value.strip().strip('"').strip()
    return None if text.upper() in _HEADER_NULLS else text.upper()


def parse_marker_header(header: str) -> Marker:
    """Parse a ``markers.fasta`` header into a :class:`Marker`.

    Whitespace-separated positional form ``symbol [class [subclass [subtype]]]``, e.g.
    ``blaKPC-2 BETA-LACTAM CARBAPENEM AMR`` or ``gyrA_S83L QUINOLONE QUINOLONE POINT``.
    ``key=value`` tokens (``type=AMR``, ``class=BETA-LACTAM``, ``subtype=POINT``,
    ``name="quoted text"``) are also accepted and win over positional ones.

    * ``type=`` sets :attr:`Marker.element_type` (default ``AMR``). A positional
      subtype slot holding ``STRESS`` / ``VIRULENCE`` is read as the type.
    * Without an explicit subtype, a ``gene_Mutation``-shaped symbol is ``POINT``,
      anything else ``AMR``. An explicit other subtype (``AMR-SUSCEPTIBLE``) is kept
      as is, so :attr:`Marker.is_resistance_feature` is False, as in training.
    * ``NA`` class / subclass values are null; a missing subclass defaults to the class.
    """
    tokens = _HEADER_TOKEN_RE.findall(header.strip().lstrip(">"))
    if not tokens:
        raise ValueError("empty marker header")
    symbol = tokens[0]
    positional: list[str] = []
    keyed: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" in token:
            key, _, value = token.partition("=")
            keyed[key.strip().lower()] = value
        else:
            positional.append(token)
    amr_class = _header_value(keyed["class"]) if "class" in keyed else _header_value(positional[0] if positional else None)
    if "subclass" in keyed:
        subclass = _header_value(keyed["subclass"])
    else:
        subclass = _header_value(positional[1]) if len(positional) > 1 else amr_class
    element_type = _header_value(keyed.get("type"))
    subtype = _header_value(keyed["subtype"]) if "subtype" in keyed else _header_value(positional[2] if len(positional) > 2 else None)
    if element_type is None and subtype in NON_AMR_TYPES:
        element_type = subtype  # the Type was written in the subtype slot
    if subtype is None:
        subtype = SUBTYPE_POINT if _POINT_SYMBOL_RE.match(symbol) else SUBTYPE_AMR
    return Marker(
        symbol=symbol,
        subtype=subtype,
        amr_class=amr_class,
        subclass=subclass,
        method="EXACT_SUBSTRING",
        backend=MarkerScan.name,
        element_type=element_type or TYPE_AMR,
    )


def _revcomp(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


class MarkerScan:
    """Exact substring search for marker sequences from ``models/markers.fasta``.

    A marker is present when its sequence (or its reverse complement) occurs
    verbatim in any contig. Contigs are joined with ``N`` so a match cannot span two
    of them. This is the fallback for SYNTHETIC bundles only (the pipeline passes
    ``markers.fasta`` only when ``manifest.json`` has ``synthetic: true``); it only
    knows the markers the bundle ships and misses any marker split across contigs or
    carrying a single SNP. Records that training would drop (``type`` other than
    ``AMR``, ``subtype`` outside ``{AMR, POINT}``) are skipped and counted.
    """

    name = "marker_scan"

    def __init__(self, markers_fasta: Path) -> None:
        self.markers_fasta = Path(markers_fasta)
        self._markers: list[tuple[Marker, str, str]] | None = None
        self._skipped: tuple[int, tuple[str, ...]] = (0, ())

    def available(self, fasta: Path) -> bool:
        return self.markers_fasta.is_file()

    def markers(self, droplog: DropLog | None = None) -> list[tuple[Marker, str, str]]:
        """``(marker, sequence, reverse_complement)`` for every usable AMR record (cached).

        The skip counts are written to ``droplog`` every call (``marker_records_unusable``,
        ``marker_records_not_amr``) so each prediction's drop log shows them.
        """
        if self._markers is None:
            loaded: list[tuple[Marker, str, str]] = []
            n_bad = 0
            not_amr: list[str] = []
            for header, seq in read_fasta(self.markers_fasta):
                clean = "".join(seq.split()).upper()
                if not clean or not header.strip():
                    n_bad += 1
                    continue
                try:
                    marker = parse_marker_header(header)
                except ValueError:
                    n_bad += 1
                    continue
                if not marker.is_resistance_feature:
                    not_amr.append(marker.symbol)
                    continue
                loaded.append((marker, clean, _revcomp(clean)))
            if n_bad:
                logger.warning("Skipped %d unusable record(s) in %s", n_bad, self.markers_fasta)
            if not_amr:
                logger.info("Skipped %d non-AMR marker record(s) in %s: %s", len(not_amr), self.markers_fasta, not_amr)
            logger.info("Loaded %d marker sequence(s) from %s", len(loaded), self.markers_fasta)
            self._markers = loaded
            self._skipped = (n_bad, tuple(not_amr))
        if droplog is not None:
            n_bad, not_amr_symbols = self._skipped
            droplog.drop("marker_records_unusable", n_bad, detail=str(self.markers_fasta))
            droplog.drop(
                "marker_records_not_amr",
                len(not_amr_symbols),
                detail=", ".join(not_amr_symbols) if not_amr_symbols else None,
            )
        return self._markers

    def detect(self, fasta: Path, species: str | None, config: Config, droplog: DropLog | None = None) -> pd.DataFrame:
        genome = "N".join(seq for _, seq in read_fasta(fasta))
        rows: list[dict[str, object]] = []
        for marker, seq, rc in self.markers(droplog):
            if seq in genome or rc in genome:
                rows.append(
                    {
                        "symbol": marker.symbol,
                        "type": marker.element_type,
                        "subtype": marker.subtype,
                        "class": marker.amr_class,
                        "subclass": marker.subclass,
                        "method": marker.method,
                        "coverage": 100.0,
                        "identity": 100.0,
                        "backend": self.name,
                    }
                )
        logger.info("MarkerScan (exact-match fallback, not AMRFinderPlus) found %d marker(s) in %s", len(rows), fasta)
        return _frame_from_markers(rows)


def detect(
    fasta: Path,
    species: str | None,
    config: Config,
    *,
    amrfinder_tsv: Path | None = None,
    markers_fasta: Path | None = None,
    detectors: Sequence[AmrDetector] | None = None,
    droplog: DropLog | None = None,
    parser: TsvParser | None = None,
) -> pd.DataFrame:
    """Detect known AMR determinants in ``fasta`` with the first available backend.

    Order: an explicitly given ``amrfinder_tsv`` wins; otherwise ``amrfinder`` on
    ``PATH``, then a sidecar ``<fasta>.amrfinder.tsv``, then :class:`MarkerScan` on
    ``markers_fasta`` (pass it only for synthetic bundles). ``detectors`` replaces
    that list entirely (tests, custom deployments). ``parser`` replaces the AMRFinderPlus
    TSV reader of the default backends (bundles trained on an imported NCBI release use
    ``release_features.read_amrfinder_rows``, which keeps every ``Type == AMR`` row).
    Raises :class:`ToolNotAvailable` with the ways to provide a backend when none can run.
    """
    fasta = Path(fasta)
    if detectors is None:
        chain: list[AmrDetector] = []
        if amrfinder_tsv is not None:
            chain.append(PrecomputedAmrFinder(amrfinder_tsv, parser=parser))
        chain.append(AmrFinderCli(threads=AMRFINDER_THREADS, parser=parser))
        chain.append(PrecomputedAmrFinder(parser=parser))
        if markers_fasta is not None:
            chain.append(MarkerScan(markers_fasta))
        detectors = chain
    for detector in detectors:
        if detector.available(fasta):
            logger.info("Known-AMR detection for %s via %s", fasta.name, detector.name)
            frame = detector.detect(fasta, species, config, droplog)
            return frame.reset_index(drop=True)
    hint = (
        "No known-AMR backend for this genome. Install AMRFinderPlus so `amrfinder` is on PATH, "
        "or provide a precomputed AMRFinderPlus TSV (`genome2mic predict --amrfinder-tsv FILE`, "
        f"or {sidecar_path(fasta).name} next to the genome)."
    )
    if markers_fasta is None:
        hint += (
            " The markers.fasta exact-match fallback is only used for synthetic model bundles "
            "(manifest.json synthetic: true)."
        )
    raise ToolNotAvailable("amrfinder", hint=hint)


# --------------------------------------------------------------------------- #
# Symbol -> feature column (must match features/known_amr.py)
# --------------------------------------------------------------------------- #


def _local_family_of(symbol: str, keep_variant_prefixes: Sequence[str]) -> str:
    """Gene family of ``symbol``: strip the trailing ``-<digits>`` unless the prefix is kept.

    ``blaCTX-M-15`` -> ``blaCTX-M``; ``blaKPC-2`` -> ``blaKPC-2`` (``blaKPC`` is kept);
    ``aac(6')-Ib-cr`` -> unchanged (no trailing digit group).
    """
    text = symbol.strip()
    for prefix in keep_variant_prefixes:
        if text == prefix or text.startswith(prefix):
            return text
    return _VARIANT_SUFFIX_RE.sub("", text) or text


def _local_column_name(prefix: str, symbol: str) -> str:
    """``prefix`` + lower-cased ``symbol`` with non-alphanumerics collapsed to ``_``."""
    body = _NON_ALNUM_RE.sub("_", symbol.strip().lower()).strip("_")
    return f"{prefix}{body}"


_HELPERS: tuple[Callable[..., str], Callable[..., str]] | None = None


def _known_amr_helpers() -> tuple[Callable[..., str], Callable[..., str]]:
    """``(column_name, family_of)`` from ``features.known_amr`` if present, else local."""
    global _HELPERS
    if _HELPERS is None:
        try:
            from genome2mic.features import known_amr as _ka  # noqa: PLC0415

            _HELPERS = (_ka.column_name, _ka.family_of)
            logger.debug("Using genome2mic.features.known_amr naming helpers")
        except ImportError:
            _HELPERS = (_local_column_name, _local_family_of)
            logger.warning(
                "genome2mic.features.known_amr is not importable; using the local "
                "column_name/family_of implementation of the same rule"
            )
    return _HELPERS


def column_name(prefix: str, symbol: str) -> str:
    """Feature column for ``prefix`` + ``symbol`` (``gene_`` + ``blaCTX-M`` -> ``gene_blactx_m``)."""
    return _known_amr_helpers()[0](prefix, symbol)


def family_of(symbol: str, keep_variant_prefixes: Sequence[str]) -> str:
    """Gene family for ``symbol`` honouring ``keep_variant.csv`` prefixes."""
    return _known_amr_helpers()[1](symbol, keep_variant_prefixes)


@dataclass(frozen=True)
class KnownAmrRow:
    """Known-AMR features of one genome before alignment to a model's columns.

    Attributes:
        values: every derived column -> value (``gene_``/``point_`` 0/1, ``n_class_`` counts).
        symbols_by_column: present ``gene_`` / ``point_`` column -> detected symbols.
        markers: detections with their ``column`` filled in.
    """

    values: dict[str, int]
    symbols_by_column: dict[str, tuple[str, ...]]
    markers: tuple[Marker, ...]


def known_amr_row(detections: pd.DataFrame, config: Config) -> KnownAmrRow:
    """Turn a detection table into feature columns with the training naming rules.

    * ``AMR`` subtype -> ``gene_<family>`` (family via ``keep_variant.csv``), value 1.
    * ``POINT`` subtype -> ``point_<symbol>``, value 1.
    * every hit with a ``Class`` -> ``n_class_<class>`` += 1.
    """
    keep = config.keep_variant_prefixes()
    values: dict[str, int] = {}
    symbols: dict[str, list[str]] = {}
    markers: list[Marker] = []
    for marker in markers_from_frame(detections):
        if marker.subtype == SUBTYPE_POINT:
            column = column_name(PREFIX_POINT, marker.symbol)
        else:
            column = column_name(PREFIX_GENE, family_of(marker.symbol, keep))
        values[column] = 1
        bucket = symbols.setdefault(column, [])
        if marker.symbol not in bucket:
            bucket.append(marker.symbol)
        if marker.amr_class is not None:
            class_column = column_name(PREFIX_CLASS, marker.amr_class)
            values[class_column] = values.get(class_column, 0) + 1
        markers.append(replace(marker, column=column))
    logger.info(
        "Known-AMR row: %d marker(s) -> %d feature column(s)",
        len(markers),
        len(values),
    )
    return KnownAmrRow(
        values=values,
        symbols_by_column={column: tuple(found) for column, found in symbols.items()},
        markers=tuple(markers),
    )


def feature_vector(
    row: KnownAmrRow,
    known_columns: Sequence[str],
    droplog: DropLog | None = None,
) -> np.ndarray:
    """Align a :class:`KnownAmrRow` to a model's ``known_columns`` (zero-filled).

    Detected columns that the model never saw in training cannot be used and are
    counted in ``droplog`` under ``known_amr_column_not_in_training``.
    """
    wanted = set(known_columns)
    unseen = sorted(column for column in row.values if column not in wanted)
    if droplog is not None:
        droplog.drop(
            "known_amr_column_not_in_training",
            len(unseen),
            detail=", ".join(unseen) if unseen else None,
        )
    elif unseen:
        logger.info("%d detected known-AMR column(s) not in training features: %s", len(unseen), unseen)
    return np.array([row.values.get(column, 0) for column in known_columns], dtype=np.float32)
