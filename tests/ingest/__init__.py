"""Shared helpers for the ingest tests.

Builds a tiny, self-contained ``configs/`` directory in a temporary path so the
tests do not depend on the content of the real configs. Breakpoints mirror the
published EUCAST / CLSI values for *K. pneumoniae* so the expected intervals in
the tests read like the contract's examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from genome2mic.config import Config, load_config
from genome2mic.droplog import DropLog

# The common raw schema every reader returns (DESIGN.md "ingest"). Kept here as
# an independent copy so a test can assert the module constant did not drift.
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

# Stage 2 contract columns, in order (DATA_CONTRACT.md stage 2).
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

SPECIES_YAML = """\
species:
  KPNEU:
    name: Klebsiella pneumoniae
    amrfinder_organism: Klebsiella_pneumoniae
    expected_genome_size: 5500000
    size_tolerance: 0.20
    reference_accession: NC_016845.1
    reference_sketch: null
  ECOLI:
    name: Escherichia coli
    amrfinder_organism: Escherichia
    expected_genome_size: 5000000
    size_tolerance: 0.20
    reference_accession: NC_000913.3
    reference_sketch: null
qc:
  max_contigs: 500
  max_mash_distance: 0.05
"""

DRUGS_YAML = """\
call_standard: {standard: EUCAST, version: "2024"}
drugs:
  ampicillin:
    synonyms: [AMP]
    spectrum_tier: 1
    strong_markers: []
  ceftriaxone:
    synonyms: [CRO, ceftriaxone sodium]
    spectrum_tier: 2
    strong_markers: [gene_blactx_m]
  ciprofloxacin:
    synonyms: [CIP]
    spectrum_tier: 2
    strong_markers: []
  meropenem:
    synonyms: [MEM, mero, meropenem trihydrate]
    spectrum_tier: 4
    strong_markers: [gene_blakpc]
  piperacillin-tazobactam:
    synonyms: [TZP, piperacillin/tazobactam]
    spectrum_tier: 3
    strong_markers: []
"""

# EUCAST v14.0 (2024) Enterobacterales, bloodstream. ECOLI x ceftriaxone is left
# out on purpose: the "no breakpoint -> drop" test relies on it.
EUCAST_2024 = """\
species,drug,s_breakpoint,r_breakpoint,version,site,note
KPNEU,meropenem,2,8,2024,bloodstream,test table
KPNEU,ceftriaxone,1,2,2024,bloodstream,test table
KPNEU,ciprofloxacin,0.25,0.5,2024,bloodstream,test table
KPNEU,ampicillin,8,8,2024,bloodstream,test table; S == R so there is no I category
ECOLI,meropenem,2,8,2024,bloodstream,test table
ECOLI,ciprofloxacin,0.25,0.5,2024,bloodstream,test table
"""

# EUCAST v6.0 (2016): ciprofloxacin S<=0.5 / R>1 for Enterobacteriaceae. Lets a
# test prove that ``standard_year`` selects the matching table.
EUCAST_2016 = """\
species,drug,s_breakpoint,r_breakpoint,version,site,note
KPNEU,meropenem,2,8,2016,bloodstream,test table
KPNEU,ciprofloxacin,0.5,1,2016,bloodstream,test table; pre-2017 ciprofloxacin breakpoints
"""

# CLSI M100 (2024). Published R>=x is stored as x/2 (DESIGN.md).
CLSI_2024 = """\
species,drug,s_breakpoint,r_breakpoint,version,site,note
KPNEU,meropenem,1,2,2024,bloodstream,test table; published R>=4 stored as 2
KPNEU,ceftriaxone,1,2,2024,bloodstream,test table; published R>=4 stored as 2
KPNEU,ciprofloxacin,0.25,0.5,2024,bloodstream,test table; published R>=1 stored as 0.5
KPNEU,ampicillin,8,16,2024,bloodstream,test table; published R>=32 stored as 16
"""

NATURAL_RESISTANCE_CSV = "species,drug,note\nKPNEU,ampicillin,chromosomal SHV beta-lactamase\n"
KEEP_VARIANT_CSV = "family_prefix,note\nblaKPC,carbapenemase\n"


def write_configs(root: Path) -> Path:
    """Write the tiny test configs under ``root/configs`` and return that directory."""
    configs = root / "configs"
    (configs / "breakpoints").mkdir(parents=True, exist_ok=True)
    (configs / "species.yaml").write_text(SPECIES_YAML, encoding="utf-8")
    (configs / "drugs.yaml").write_text(DRUGS_YAML, encoding="utf-8")
    (configs / "breakpoints" / "eucast_2024.csv").write_text(EUCAST_2024, encoding="utf-8")
    (configs / "breakpoints" / "eucast_2016.csv").write_text(EUCAST_2016, encoding="utf-8")
    (configs / "breakpoints" / "clsi_2024.csv").write_text(CLSI_2024, encoding="utf-8")
    (configs / "natural_resistance.csv").write_text(NATURAL_RESISTANCE_CSV, encoding="utf-8")
    (configs / "keep_variant.csv").write_text(KEEP_VARIANT_CSV, encoding="utf-8")
    return configs


def make_config(root: Path) -> Config:
    """Write the test configs under ``root`` and load them."""
    return load_config(write_configs(root))


def raw_row(**overrides: Any) -> dict[str, Any]:
    """One row of the common raw schema: a lab-measured ``= 8`` meropenem result.

    Every column has a sensible default so a test only states what it is about.
    """
    row: dict[str, Any] = {
        "genome_id": "573.1",
        "biosample": None,
        "species": "KPNEU",
        "antibiotic_raw": "meropenem",
        "sir_raw": None,
        "sign": "=",
        "value": "8",
        "unit": "mg/L",
        "method_raw": "Broth dilution",
        "standard": "EUCAST",
        "standard_year": 2024,
        "evidence": "Laboratory Method",
        "source": "BVBRC",
        "isolation_source": None,
        "country": None,
        "year": None,
    }
    unknown = set(overrides) - set(row)
    if unknown:
        raise KeyError(f"not raw-schema columns: {sorted(unknown)}")
    row.update(overrides)
    return row


def make_raw(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Common-raw-schema frame from ``raw_row`` dicts (object dtype, ``None`` = missing)."""
    frame = pd.DataFrame(rows, columns=list(RAW_COLUMNS), dtype=object)
    return frame.astype(object)


def log_counts(droplog: DropLog) -> dict[str, int]:
    """``{reason: total n_dropped}`` for a drop log."""
    counts: dict[str, int] = {}
    for record in droplog.records:
        counts[record.reason] = counts.get(record.reason, 0) + record.n_dropped
    return counts
