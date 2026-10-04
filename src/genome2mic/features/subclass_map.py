"""AMRFinderPlus ``Subclass`` of every known-AMR ``gene_`` column, for training-time calls.

At prediction time the strong-marker override reads the ``Subclass`` AMRFinderPlus
reports for each detected acquired gene (``drugs.yaml`` ``strong_subclasses``, e.g.
``CARBAPENEM`` for the carbapenems). Training-time calls only see the feature table, so
each ``gene_`` column needs the subclass of the symbol(s) it was built from. The release
``known_amr_columns.csv`` carries ``source_symbol`` but no subclass, so it is looked up
in the AMRFinderPlus database (the version the release was built with, 2026-08-07.1):

1. ``AMRProt.fa`` headers -- the allele-level table (field 4 = gene symbol, field 8 =
   Subclass). This is where AMRFinderPlus takes the Subclass it reports for an allele
   hit: ``blaOXA-23`` -> ``CARBAPENEM`` while ``blaOXA-1`` -> ``CEPHALOSPORIN``.
2. ``fam.tsv`` by ``#node_id``: the raw symbol, then the symbol without its allele suffix
   (the release naming rule used for ``n_class_``; covers family-level symbols such as
   ``blaKPC`` / ``blaNDM`` / ``blaIMI``).

``fam.tsv`` alone is not enough: it lists families, not alleles, and gives the generic
``blaOXA`` / ``blaOXA-48_fam`` nodes the Subclass ``BETA-LACTAM``, so every OXA
carbapenemase allele (OXA-23, -24/40, -48, -58, -181, ...) and GES-5 would be missed.

A column built from several symbols (release ``source_symbol`` ``a,b``; our own
``member_symbols`` ``a;b``) gets the ``;``-joined set of its members' subclasses, with
``UNKNOWN`` for a member the database does not list, so
:func:`genome2mic.predict.rank.subclass_is_strong` only fires when every member is
strong. A non-empty ``subclass`` already in ``known_amr_columns.csv`` (our own
AMRFinderPlus runs) is kept as is.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["SubclassTables", "UNKNOWN_SUBCLASS", "column_subclasses", "load_subclass_tables"]

UNKNOWN_SUBCLASS = "UNKNOWN"
ALLELE_SUFFIX = re.compile(r"-\d+$")


@dataclass(frozen=True)
class SubclassTables:
    """Symbol -> Subclass from one AMRFinderPlus database."""

    allele: Mapping[str, str]
    family: Mapping[str, str]
    db_version: str | None = None
    db_dir: str | None = None

    def subclass_of(self, symbol: str) -> str | None:
        text = str(symbol).strip()
        if not text:
            return None
        return (
            self.allele.get(text)
            or self.family.get(text)
            or self.family.get(ALLELE_SUFFIX.sub("", text))
        )


def _merge(existing: str | None, new: str) -> str:
    if not existing:
        return new
    tokens = sorted(set(existing.split("/")) | set(new.split("/")))
    return "/".join(tokens)


def load_subclass_tables(db_dir: Path) -> SubclassTables:
    """Read ``AMRProt.fa`` headers and ``fam.tsv`` of an AMRFinderPlus database directory.

    Raises:
        FileNotFoundError: ``fam.tsv`` is missing (``AMRProt.fa`` is optional but logged).
    """
    db = Path(db_dir)
    fam_path = db / "fam.tsv"
    if not fam_path.is_file():
        raise FileNotFoundError(f"{fam_path} not found (AMRFinderPlus database directory expected)")
    fam = pd.read_csv(fam_path, sep="\t", dtype=str, keep_default_na=False)
    family = {
        str(node).strip(): str(sub).strip()
        for node, sub in zip(fam["#node_id"], fam["subclass"])
        if str(node).strip() and str(sub).strip()
    }
    allele: dict[str, str] = {}
    prot = db / "AMRProt.fa"
    if prot.is_file():
        with prot.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith(">"):
                    continue
                fields = line[1:].rstrip("\n").split("|")
                if len(fields) < 9:
                    continue
                symbol, subclass = fields[3].strip(), fields[7].strip()
                if symbol and subclass:
                    allele[symbol] = _merge(allele.get(symbol), subclass)
    else:
        logger.warning("%s not found: allele-level subclasses unavailable (fam.tsv families only)", prot)
    version_file = db / "version.txt"
    version = version_file.read_text().strip() if version_file.is_file() else Path(db).resolve().name
    logger.info("AMRFinderPlus DB %s: %d allele and %d family subclasses", version, len(allele), len(family))
    return SubclassTables(allele=allele, family=family, db_version=version or None, db_dir=str(db))


def _members(row: Mapping[str, object]) -> list[str]:
    members = str(row.get("member_symbols") or "").strip()
    if members and members.lower() not in {"nan", "none", "<na>"}:
        return [m.strip() for m in members.split(";") if m.strip()]
    source = str(row.get("source_symbol") or "").strip()
    if not source or source.lower() in {"nan", "none", "<na>"}:
        return []
    return [m.strip() for m in re.split(r"[;,]", source) if m.strip()]


def column_subclasses(columns_table: pd.DataFrame, tables: SubclassTables | None) -> dict[str, str | None]:
    """``gene_`` column -> ``;``-joined member subclasses (see the module docstring).

    Rows of every species are pooled per column (a column's member set is the union).
    A non-empty ``subclass`` cell already in the table wins. Without ``tables`` only those
    cells are used.
    """
    if columns_table is None or columns_table.empty or "column_name" not in columns_table.columns:
        return {}
    members: dict[str, set[str]] = {}
    given: dict[str, set[str]] = {}
    for row in columns_table.to_dict("records"):
        column = str(row.get("column_name") or "").strip()
        if not column.startswith("gene_"):
            continue
        members.setdefault(column, set()).update(_members(row))
        sub = str(row.get("subclass") or "").strip() if "subclass" in row else ""
        if sub and sub.lower() not in {"nan", "none", "<na>"}:
            given.setdefault(column, set()).update(s for s in sub.split(";") if s.strip())
    out: dict[str, str | None] = {}
    for column, syms in members.items():
        if column in given:
            out[column] = ";".join(sorted(given[column]))
            continue
        if tables is None or not syms:
            out[column] = None
            continue
        values = {tables.subclass_of(s) or UNKNOWN_SUBCLASS for s in syms}
        out[column] = ";".join(sorted(values))
    return out


def strong_columns_report(subclass_by_column: Mapping[str, str | None], wanted: Iterable[str]) -> list[str]:
    """Columns whose every member carries one of ``wanted`` (for logging / meta.json)."""
    from genome2mic.predict.rank import subclass_is_strong  # noqa: PLC0415

    return sorted(c for c, s in subclass_by_column.items() if subclass_is_strong(s, wanted))
