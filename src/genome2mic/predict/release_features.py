"""Prediction-side known-AMR features with the NCBI data-release naming rules.

Bundles trained on an imported data release (``import-release``) learned from the
release's ``known_amr.parquet``. Those columns were built by develop's
``genome2mic.features.ncbi_known_amr.NcbiKnownAmrBuilder`` from NCBI Pathogen Detection's
``AMR_genotypes`` strings (AMRFinderPlus run by NCBI), **not** by our
``features/known_amr.py``. The rules differ (which subtypes count, which alleles collapse,
where the class comes from), so a genome scored by such a bundle must be converted with the
release rules or the model reads silently wrong inputs. This module is that converter.

Release rules (develop ``docs/HACKATHON_DATA.md`` "Feature rules", ``ncbi_known_amr.py`` at fca4513):

* One hit per ``AMR_genotypes`` item ``symbol[=TAG]``. ``MISTRANSLATION`` (internal stop
  codon) is skipped; partial / HMM / end-of-contig hits count as present.
* ``=POINT`` -> ``point_<symbol>``; everything else -> ``gene_<family>``.
* Only ``bla`` symbols collapse to the family (trailing ``-<digits>`` removed), except the
  families in the release's keep-variant list (``blaKPC``, ``blaNDM``, ``blaVIM``,
  ``blaIMP``, ``blaGES``, ``blaOXA``), which keep the allele.
* Column names: lower case, every run of non-alphanumerics -> one ``_``, stripped.
* ``n_class_<class>`` counts hits (every hit, also repeated ones) per AMRFinderPlus class,
  where the class comes from the AMRFinderPlus database tables (``fam.tsv``,
  ``AMRProt-mutation.tsv``, ``AMR_DNA-<organism>.tsv``) looked up by the raw symbol, then
  by the symbol without its allele suffix; a multi-class ``A/B`` entry counts once in each.
  The ``Class`` column of an AMRFinderPlus report is **not** used (it differs for e.g.
  frameshift ``POINT_DISRUPT`` hits that the tables do not list).
* Values are clipped to 127 (int8).

An AMRFinderPlus TSV (our own run on a new genome) is mapped to ``AMR_genotypes`` items the
way NCBI Pathogen Detection writes them: ``Type == AMR`` rows only (STRESS / VIRULENCE are
separate NCBI columns); ``Method`` ``POINT*`` or ``Subtype`` ``POINT*`` -> ``POINT``;
``INTERNAL_STOP`` -> ``MISTRANSLATION``; ``PARTIAL_CONTIG_END*`` -> ``PARTIAL_END_OF_CONTIG``;
``PARTIAL*`` -> ``PARTIAL``; ``HMM`` -> ``HMM``; anything else (``EXACT*``, ``ALLELE*``,
``BLAST*``) -> complete (no tag).

The bundle carries the rules and the class table it was trained with
(``models/<SPECIES>/feature_spec.json``, written by :func:`write_feature_specs`), so a
different local AMRFinderPlus database cannot change the class counts unnoticed.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from genome2mic.droplog import DropLog
from genome2mic.features import known_amr as _training_amr
from genome2mic.predict import amr_detect

logger = logging.getLogger(__name__)

__all__ = [
    "FEATURE_NAMING_NCBI_RELEASE",
    "FEATURE_SPEC_FILE",
    "RELEASE_KEEP_VARIANT_FAMILIES",
    "ReleaseFeatureSpec",
    "amrfinder_items",
    "column_name",
    "find_amrfinder_db",
    "genotype_string",
    "load_class_table",
    "parse_genotypes",
    "read_amrfinder_rows",
    "release_known_amr_row",
    "release_values",
    "write_feature_specs",
    "write_specs_for_root",
]

FEATURE_NAMING_NCBI_RELEASE = "ncbi_release"
FEATURE_SPEC_FILE = "feature_spec.json"
SPEC_VERSION = 1

RELEASE_KEEP_VARIANT_FAMILIES: tuple[str, ...] = ("blaKPC", "blaNDM", "blaVIM", "blaIMP", "blaGES", "blaOXA")
"""develop ``configs/keep_variant.csv`` (fca4513), used to build the release's ``known_amr.parquet``."""

SKIPPED_TAGS: frozenset[str] = frozenset({"MISTRANSLATION"})
TAG_POINT = "POINT"
TAG_MISTRANSLATION = "MISTRANSLATION"
TAG_PARTIAL = "PARTIAL"
TAG_PARTIAL_END = "PARTIAL_END_OF_CONTIG"
TAG_HMM = "HMM"
VALUE_CAP = 127

ALLELE_SUFFIX = re.compile(r"-\d+$")
NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------------- #
# Spec (rules + class table) stored with the bundle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReleaseFeatureSpec:
    """Everything needed to rebuild a release ``known_amr`` row for one species."""

    organism: str | None
    class_by_symbol: Mapping[str, str]
    keep_variant_families: tuple[str, ...] = RELEASE_KEEP_VARIANT_FAMILIES
    skipped_tags: frozenset[str] = SKIPPED_TAGS
    amrfinder_db_version: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_json(self) -> dict[str, Any]:
        return {
            "spec_version": SPEC_VERSION,
            "feature_naming": FEATURE_NAMING_NCBI_RELEASE,
            "organism": self.organism,
            "keep_variant_families": list(self.keep_variant_families),
            "skipped_tags": sorted(self.skipped_tags),
            "amrfinder_db_version": self.amrfinder_db_version,
            "notes": list(self.notes),
            "class_by_symbol": dict(sorted(self.class_by_symbol.items())),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ReleaseFeatureSpec:
        if data.get("feature_naming") != FEATURE_NAMING_NCBI_RELEASE:
            raise ValueError(f"feature spec is not {FEATURE_NAMING_NCBI_RELEASE!r}: {data.get('feature_naming')!r}")
        classes = data.get("class_by_symbol")
        if not isinstance(classes, dict):
            raise ValueError("feature spec: 'class_by_symbol' must be an object")
        return cls(
            organism=data.get("organism"),
            class_by_symbol={str(k): str(v) for k, v in classes.items() if v},
            keep_variant_families=tuple(data.get("keep_variant_families") or RELEASE_KEEP_VARIANT_FAMILIES),
            skipped_tags=frozenset(data.get("skipped_tags") or SKIPPED_TAGS),
            amrfinder_db_version=data.get("amrfinder_db_version"),
            notes=tuple(data.get("notes") or ()),
        )

    @classmethod
    def load(cls, path: Path) -> ReleaseFeatureSpec:
        return cls.from_json(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1) + "\n")

    def class_of(self, symbol: str) -> str | None:
        """Class by the raw symbol, then by the symbol without its allele suffix (release rule)."""
        return self.class_by_symbol.get(symbol) or self.class_by_symbol.get(ALLELE_SUFFIX.sub("", symbol))


def load_class_table(amrfinder_db_dir: Path, organism: str | None) -> dict[str, str]:
    """Symbol -> AMRFinderPlus class from the database tables (develop's ``load_class_table``).

    ``AMR_DNA-<organism>.tsv`` is optional: the database has none for some organisms
    (Pseudomonas aeruginosa), which the release build treated as an empty table.
    """
    db = Path(amrfinder_db_dir)
    families = pd.read_csv(db / "fam.tsv", sep="\t", dtype=str)
    class_by_symbol = dict(zip(families["#node_id"], families["class"].fillna("")))
    mutation_files = ["AMRProt-mutation.tsv"]
    if organism:
        mutation_files.append(f"AMR_DNA-{organism}.tsv")
    for mutation_file in mutation_files:
        path = db / mutation_file
        if not path.is_file():
            logger.info("%s not in the AMRFinderPlus database; treated as an empty table", path.name)
            continue
        mutations = pd.read_csv(path, sep="\t", dtype=str)
        class_by_symbol.update(zip(mutations["standard_mutation_symbol"], mutations["class"].fillna("")))
    return {symbol: amr_class for symbol, amr_class in class_by_symbol.items() if amr_class}


def find_amrfinder_db(explicit: Path | None = None) -> Path | None:
    """The AMRFinderPlus database dir: explicit, else ``<amrfinder prefix>/share/amrfinderplus/data/latest``."""
    import shutil  # noqa: PLC0415

    if explicit is not None:
        return Path(explicit)
    exe = shutil.which("amrfinder")
    if exe is None:
        return None
    candidate = Path(exe).resolve().parent.parent / "share" / "amrfinderplus" / "data" / "latest"
    return candidate if (candidate / "fam.tsv").is_file() else None


def _db_version(db_dir: Path) -> str | None:
    version = Path(db_dir) / "version.txt"
    if version.is_file():
        return version.read_text().strip() or None
    resolved = Path(db_dir).resolve()
    return resolved.name or None


def write_feature_specs(
    models_dir: Path,
    amrfinder_db_dir: Path,
    organisms: Mapping[str, str],
    *,
    species: Iterable[str] | None = None,
    expected_db_version: str | None = None,
) -> list[Path]:
    """Write ``models/<SPECIES>/feature_spec.json`` and mark ``manifest.json`` ``feature_naming: ncbi_release``.

    Args:
        models_dir: bundle root holding ``manifest.json``.
        amrfinder_db_dir: AMRFinderPlus database the class table is read from.
        organisms: species key -> AMRFinderPlus ``-O`` organism.
        species: species to write (default: every species in the manifest).
        expected_db_version: the release's ``tool_versions.json`` ``amrfinder_db``; a
            different local database is logged as a warning and recorded in the spec notes.
    """
    models_dir = Path(models_dir)
    manifest_path = models_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    keys = list(species) if species is not None else list(manifest.get("species", {}))
    db_version = _db_version(amrfinder_db_dir)
    notes: list[str] = []
    if expected_db_version and db_version and expected_db_version != db_version:
        message = (
            f"class table from AMRFinderPlus DB {db_version}, release features were built with "
            f"{expected_db_version}: n_class_ counts may differ for symbols whose class changed"
        )
        logger.warning(message)
        notes.append(message)
    written = []
    for key in keys:
        organism = organisms.get(key)
        spec = ReleaseFeatureSpec(
            organism=organism,
            class_by_symbol=load_class_table(amrfinder_db_dir, organism),
            amrfinder_db_version=db_version,
            notes=tuple(notes),
        )
        out = models_dir / key / FEATURE_SPEC_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        spec.save(out)
        written.append(out)
        logger.info("Wrote %s (%d class symbols, organism %s)", out, len(spec.class_by_symbol), organism)
    manifest["feature_naming"] = FEATURE_NAMING_NCBI_RELEASE
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return written


# --------------------------------------------------------------------------- #
# AMR_genotypes items -> feature values
# --------------------------------------------------------------------------- #


def column_name(prefix: str, symbol: str) -> str:
    """Release naming: lower case, every run of non-alphanumerics -> one ``_``, stripped."""
    return prefix + NON_ALPHANUMERIC.sub("_", symbol.lower()).strip("_")


def family_for(symbol: str, keep_variant_families: Sequence[str]) -> str:
    """Collapse ``bla`` alleles to their family unless the family keeps its variant."""
    if not symbol.startswith("bla"):
        return symbol
    if any(symbol.startswith(kept) for kept in keep_variant_families):
        return symbol
    return ALLELE_SUFFIX.sub("", symbol)


def parse_genotypes(genotypes: str, skipped_tags: Iterable[str] = SKIPPED_TAGS) -> list[tuple[str, str]]:
    """``AMR_genotypes`` string -> ``(kind, symbol)`` pairs; kind is ``gene`` or ``point``."""
    skipped = frozenset(skipped_tags)
    hits = []
    for item in str(genotypes).split(","):
        symbol, _, tag = item.strip().partition("=")
        if not symbol or tag in skipped:
            continue
        hits.append(("point" if tag == TAG_POINT else "gene", symbol))
    return hits


def hit_column(kind: str, symbol: str, spec: ReleaseFeatureSpec) -> str:
    if kind == "point":
        return column_name("point_", symbol)
    return column_name("gene_", family_for(symbol, spec.keep_variant_families))


def release_values(hits: Sequence[tuple[str, str]], spec: ReleaseFeatureSpec) -> dict[str, int]:
    """Feature values for one genome: ``gene_``/``point_`` presence (0/1) and ``n_class_`` hit counts."""
    values: dict[str, int] = {}
    for kind, symbol in hits:
        values[hit_column(kind, symbol, spec)] = 1
        amr_class = spec.class_of(symbol)
        if amr_class:
            for name in amr_class.split("/"):
                column = column_name("n_class_", name)
                values[column] = min(values.get(column, 0) + 1, VALUE_CAP)
    return values


# --------------------------------------------------------------------------- #
# AMRFinderPlus TSV -> AMR_genotypes items
# --------------------------------------------------------------------------- #


def tag_for(method: str | None, subtype: str | None) -> str:
    """NCBI Pathogen Detection tag for one AMRFinderPlus row (``""`` = complete)."""
    method = (method or "").strip().upper()
    subtype = (subtype or "").strip().upper()
    if method.startswith("POINT") or subtype.startswith("POINT"):
        return TAG_POINT
    if method.startswith("INTERNAL_STOP"):
        return TAG_MISTRANSLATION
    if method.startswith("PARTIAL_CONTIG_END"):
        return TAG_PARTIAL_END
    if method.startswith("PARTIAL"):
        return TAG_PARTIAL
    if method.startswith("HMM"):
        return TAG_HMM
    return ""


def read_amrfinder_rows(
    path: Path,
    droplog: DropLog | None = None,
    backend: str = "precomputed",
) -> pd.DataFrame:
    """Every ``Type == AMR`` row of an AMRFinderPlus 3.x/4.x TSV as :data:`amr_detect.DETECTION_COLUMNS`.

    Unlike :func:`amr_detect.parse_amrfinder_tsv` no subtype is dropped (``POINT_DISRUPT``
    rows are ``POINT`` items in NCBI's ``AMR_genotypes``). Non-AMR rows are counted in
    ``droplog`` under ``amrfinder_type_not_amr``.
    """
    source = Path(path)
    rows: list[dict[str, object]] = []
    n_rows = n_non_amr = n_no_symbol = 0
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{source}: empty file; AMRFinderPlus always writes a header line")
        index = _training_amr._resolve_header(header, source)

        def cell(row: Sequence[str], name: str) -> str | None:
            position = index.get(name)
            if position is None or position >= len(row):
                return None
            return amr_detect._clean_text(row[position])

        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            n_rows += 1
            if (cell(row, "type") or "").upper() != "AMR":
                n_non_amr += 1
                continue
            symbol = cell(row, "symbol")
            if symbol is None:
                n_no_symbol += 1
                continue
            rows.append({
                "symbol": symbol,
                "type": "AMR",
                "subtype": cell(row, "subtype"),
                "class": cell(row, "class"),
                "subclass": cell(row, "subclass"),
                "method": cell(row, "method"),
                "coverage": cell(row, "coverage"),
                "identity": cell(row, "identity"),
                "backend": backend,
            })
    for reason, count in (("amrfinder_type_not_amr", n_non_amr), ("amrfinder_no_symbol", n_no_symbol)):
        if droplog is not None:
            droplog.drop(reason, count, detail=str(source))
        elif count:
            logger.info("Dropped %d AMRFinder row(s) from %s: %s", count, source, reason)
    frame = pd.DataFrame(rows, columns=list(amr_detect.DETECTION_COLUMNS))
    for column in ("coverage", "identity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    logger.info("Read %d AMR row(s) of %d from %s (release feature rules)", len(frame), n_rows, source)
    return frame


def amrfinder_items(detections: pd.DataFrame) -> list[tuple[str, str]]:
    """``(symbol, tag)`` per ``Type == AMR`` detection row, as NCBI writes ``AMR_genotypes``."""
    items = []
    for record in detections.to_dict("records"):
        symbol = amr_detect._clean_text(record.get("symbol"))
        element_type = (amr_detect._clean_text(record.get("type")) or "AMR").upper()
        if symbol is None or element_type != "AMR":
            continue
        items.append((symbol, tag_for(amr_detect._clean_text(record.get("method")), amr_detect._clean_text(record.get("subtype")))))
    return items


def genotype_string(items: Sequence[tuple[str, str]]) -> str:
    """``(symbol, tag)`` items -> an ``AMR_genotypes``-style string (complete hits carry no tag)."""
    return ",".join(f"{symbol}={tag}" if tag else symbol for symbol, tag in items)


def release_known_amr_row(detections: pd.DataFrame, spec: ReleaseFeatureSpec) -> amr_detect.KnownAmrRow:
    """Detection table -> :class:`amr_detect.KnownAmrRow` with the release rules.

    ``markers`` keep the AMRFinderPlus ``Subclass`` of each kept hit (the strong-marker
    override reads it); skipped (``MISTRANSLATION``) hits are neither features nor markers.
    """
    values: dict[str, int] = {}
    symbols: dict[str, list[str]] = {}
    markers: list[amr_detect.Marker] = []
    n_skipped = 0
    for record in detections.to_dict("records"):
        symbol = amr_detect._clean_text(record.get("symbol"))
        element_type = (amr_detect._clean_text(record.get("type")) or "AMR").upper()
        if symbol is None or element_type != "AMR":
            continue
        tag = tag_for(amr_detect._clean_text(record.get("method")), amr_detect._clean_text(record.get("subtype")))
        if tag in spec.skipped_tags:
            n_skipped += 1
            continue
        kind = "point" if tag == TAG_POINT else "gene"
        for column, value in release_values([(kind, symbol)], spec).items():
            if column.startswith("n_class_"):
                values[column] = min(values.get(column, 0) + value, VALUE_CAP)
            else:
                values[column] = 1
        column = hit_column(kind, symbol, spec)
        bucket = symbols.setdefault(column, [])
        if symbol not in bucket:
            bucket.append(symbol)
        markers.append(
            amr_detect.Marker(
                symbol=symbol,
                subtype=amr_detect.SUBTYPE_POINT if kind == "point" else amr_detect.SUBTYPE_AMR,
                amr_class=amr_detect._clean_text(record.get("class")),
                subclass=amr_detect._clean_text(record.get("subclass")),
                method=amr_detect._clean_text(record.get("method")),
                backend=amr_detect._clean_text(record.get("backend")),
                column=column,
            )
        )
    if n_skipped:
        logger.info("Release rules: skipped %d hit(s) tagged %s", n_skipped, sorted(spec.skipped_tags))
    logger.info("Known-AMR row (release rules): %d marker(s) -> %d feature column(s)", len(markers), len(values))
    return amr_detect.KnownAmrRow(
        values=values,
        symbols_by_column={column: tuple(found) for column, found in symbols.items()},
        markers=tuple(markers),
    )


def write_specs_for_root(paths: Any, config: Any, amrfinder_db: Path | None = None) -> list[Path]:
    """:func:`write_feature_specs` for ``<root>/models`` of an imported release.

    The database is ``amrfinder_db`` or the one next to ``amrfinder`` on ``PATH``; the
    release's ``tool_versions.json`` ``amrfinder_db`` is the expected version.

    Raises:
        FileNotFoundError: no AMRFinderPlus database could be found.
    """
    db = find_amrfinder_db(amrfinder_db)
    if db is None or not (Path(db) / "fam.tsv").is_file():
        raise FileNotFoundError(
            f"AMRFinderPlus database not found ({db or 'amrfinder not on PATH'}); pass --amrfinder-db"
        )
    expected = None
    versions = Path(paths.processed_dir) / "tool_versions.json"
    if versions.is_file():
        expected = json.loads(versions.read_text()).get("amrfinder_db")
    organisms = {key: sp.amrfinder_organism for key, sp in config.species.items()}
    return write_feature_specs(Path(paths.models_dir), Path(db), organisms, expected_db_version=expected)
