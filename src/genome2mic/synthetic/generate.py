"""Seeded generator for the complete fake raw layer (DESIGN.md "synth" stage).

``run(paths, seed, n_kpneu, n_ecoli, genome_length)`` writes under ``paths.root``:

- ``configs/``                       copy of the real configs, ``expected_genome_size``
                                     overridden to the synthetic genome size
- ``data/raw/ast_bvbrc.csv``         BV-BRC-style AST export (messy, see below)
- ``data/raw/ast_ncbi.csv``          NCBI AST-browser-style export
- ``data/raw/genome_metadata.csv``   ``genome_id, biosample, species, source,
                                     isolation_source, country, year``
- ``data/raw/genomes/<gid>.fasta``   3-40 contigs per genome
- ``data/raw/references/<SP>.fasta`` one synthetic reference per species (all five)
- ``data/interim/<gid>/``            ``amrfinder.tsv``, ``mash.tsv``, ``mlst.tsv``,
                                     ``resfinder/pheno_table.txt``
- ``data/processed/drop_log_synth.csv``  counts of the mess the ingest/QC stages are
                                     expected to drop (cross-check for their drop logs)
- ``models/markers.fasta``           reportable marker sequences for the predict
                                     pipeline's exact-match ``MarkerScan`` fallback
- ``data/raw/SYNTHETIC_DATA.md``     states that everything here is simulated

Genome model
------------
Per species one random core of ``genome_length`` bp. ~25 lineages per species, each
the core plus ~0.3% lineage SNPs and 2-4 lineage-specific accessory blocks; each
genome adds ~0.02% private SNPs, its planted markers (fixed DNA blocks from
:mod:`genome2mic.synthetic.markers`) and is split into 3-40 contigs, never cutting
through a marker. 4-6 near-identical pairs per species (<= 1 private SNP apart)
exercise the outbreak rule; ~4% of genomes deliberately fail exactly one QC rule.

Label model
-----------
``log2 MIC = base + sum(marker effects) + lineage effect N(0, 0.3) + N(0, 0.6)``,
rounded to the doubling grid, censored at the panel edges, reported S/I/R under the
row's standard using the real breakpoint tables. KPNEU and ECOLI x {meropenem,
ceftriaxone, ciprofloxacin} are built to pass the 50/50 inclusion rule; gentamicin
and piperacillin-tazobactam have fewer rows and ECOLI x gentamicin is built to fail.

All random choices come from one ``numpy.random.default_rng(seed)`` consumed in a
fixed order, so a seed reproduces every byte. Never iterate a ``set`` of strings
here: Python string hashing is randomized per process.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from genome2mic.config import Config, load_config, sir_from_mic
from genome2mic.droplog import DropLog
from genome2mic.io import write_fasta
from genome2mic.paths import Paths
from genome2mic.synthetic import markers as mk

logger = logging.getLogger(__name__)

__all__ = ["run", "main", "SPECIES_ORDER", "LABELLED_SPECIES"]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SPECIES_ORDER: tuple[str, ...] = ("ECOLI", "KPNEU", "SAUR", "PAER", "ABAU")
LABELLED_SPECIES: tuple[str, ...] = ("KPNEU", "ECOLI")
SPECIES_NAMES: dict[str, str] = {
    "ECOLI": "Escherichia coli",
    "KPNEU": "Klebsiella pneumoniae",
    "SAUR": "Staphylococcus aureus",
    "PAER": "Pseudomonas aeruginosa",
    "ABAU": "Acinetobacter baumannii",
}
BVBRC_TAXON: dict[str, str] = {"KPNEU": "573", "ECOLI": "562"}
OTHER_SPECIES: dict[str, str] = {"KPNEU": "ECOLI", "ECOLI": "KPNEU"}

N_LINEAGES = 25
LINEAGE_SNP_RATE = 0.003
PRIVATE_SNP_RATE = 0.0002
DISTANT_SNP_RATE = 0.08
ACCESSORY_BLOCKS = (2, 4)  # inclusive
ACCESSORY_LENGTH = (1000, 3000)  # bp at genome_length 60 kb; scaled with genome_length
CONTIGS = (3, 40)  # inclusive
FRAGMENTED_CONTIGS = (520, 600)
QC_FAIL_FRACTION = 0.04
QC_FAIL_KINDS: tuple[str, ...] = ("too_many_contigs", "wrong_size", "wrong_species", "too_distant")
NEAR_IDENTICAL_PAIRS = (4, 6)  # inclusive, per species
OLD_HEADER_FRACTION = 0.05
BVBRC_FRACTION = 0.65
OVERLAP_FRACTION = 0.15
CONFLICT_FRACTION = 0.25  # of overlap genomes get one conflicting NCBI row
COMPUTATIONAL_FRACTION = 0.06  # extra BV-BRC rows with evidence Computational Prediction
UNKNOWN_DRUG_FRACTION = 0.015
WITHIN_SOURCE_DUPLICATE_FRACTION = 0.03
MASH_SKETCH_SIZE = 1000
MASH_K = 21

DRUG_COVERAGE: dict[str, dict[str, float]] = {
    "KPNEU": {"meropenem": 0.92, "ceftriaxone": 0.90, "ciprofloxacin": 0.90,
              "gentamicin": 0.55, "piperacillin-tazobactam": 0.55},
    "ECOLI": {"meropenem": 0.92, "ceftriaxone": 0.90, "ciprofloxacin": 0.90,
              "gentamicin": 0.25, "piperacillin-tazobactam": 0.80},
}
PLANNED_TO_FAIL_INCLUSION: tuple[tuple[str, str], ...] = (("ECOLI", "gentamicin"),)

LINEAGE_WEIGHTS = (0.18, 0.12) + (0.06,) * 6 + (0.34 / 17,) * 17
TIER_OF_INDEX = ("high", "high") + ("medium",) * 6 + ("low",) * 17
HIGH_RISK_ST: dict[str, tuple[str, str]] = {"KPNEU": ("258", "11"), "ECOLI": ("131", "410")}
ST_POOL: dict[str, tuple[str, ...]] = {
    "KPNEU": ("15", "147", "307", "101", "14", "17", "23", "37", "45", "48", "65", "86", "231",
              "340", "392", "405", "512", "29", "35", "36", "39", "107", "152", "16", "20", "25"),
    "ECOLI": ("10", "69", "73", "95", "38", "405", "648", "1193", "12", "127", "141", "144", "167",
              "354", "393", "457", "617", "44", "58", "88", "101", "117", "155", "162", "224", "361"),
}
KNOWN_PROFILES: dict[str, tuple[int, ...]] = {
    "258": (3, 3, 1, 1, 1, 1, 79),
    "11": (3, 3, 1, 1, 1, 1, 4),
    "131": (53, 40, 47, 13, 36, 28, 29),
    "410": (6, 4, 12, 1, 20, 18, 7),
}
MLST_SCHEME: dict[str, tuple[str, tuple[str, ...]]] = {
    "KPNEU": ("klebsiella", ("gapA", "infB", "mdh", "pgi", "phoE", "rpoB", "tonB")),
    "ECOLI": ("ecoli", ("adk", "fumC", "gyrB", "icd", "mdh", "purA", "recA")),
}

COUNTRIES: tuple[str, ...] = ("USA", "United Kingdom", "India", "China", "Brazil", "Germany", "Thailand",
                              "South Africa", "Italy", "Spain", "Australia", "Vietnam", "Pakistan",
                              "Kenya", "Colombia")
COUNTRY_WEIGHTS = (0.22, 0.09, 0.11, 0.09, 0.06, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03)
US_STATES: tuple[str, ...] = ("Texas", "California", "New York", "Illinois", "Georgia", "Pennsylvania")
YEARS: tuple[int, ...] = tuple(range(2010, 2024))
YEAR_WEIGHTS = tuple(0.3 + 0.1 * i for i in range(len(YEARS)))
ISOLATION_SOURCES: tuple[str, ...] = ("blood", "urine", "sputum", "wound", "rectal swab", "cerebrospinal fluid")
ISOLATION_WEIGHTS = (0.80, 0.07, 0.05, 0.04, 0.03, 0.01)
NCBI_ISOLATION_SPELLING: dict[str, tuple[str, ...]] = {
    "blood": ("blood", "Blood", "blood culture", "whole blood"),
}

STANDARDS: tuple[str, ...] = ("EUCAST", "CLSI")
BVBRC_METHODS: tuple[str, ...] = ("Broth dilution", "Broth microdilution", "Agar dilution", "Etest",
                                  "Disk diffusion", "Vitek 2", "no_value")
BVBRC_METHOD_WEIGHTS = (0.52, 0.15, 0.05, 0.07, 0.13, 0.04, 0.04)
NCBI_METHODS: tuple[str, ...] = ("broth microdilution", "agar dilution", "MIC strip", "Etest",
                                 "disk diffusion", "automated system", "no_value")
NCBI_METHOD_WEIGHTS = (0.60, 0.06, 0.05, 0.05, 0.14, 0.05, 0.05)
UNITS: tuple[str, ...] = ("mg/L", "µg/mL", "ug/mL", "mg/l", "")
UNIT_WEIGHTS = (0.55, 0.22, 0.10, 0.06, 0.07)
NAME_STYLES: tuple[str, ...] = ("full", "abbrev", "messy")
NAME_STYLE_WEIGHTS = (0.5, 0.3, 0.2)
UNKNOWN_DRUG_NAMES: tuple[str, ...] = ("Cefiderocol", "Fosfomycin", "Ceftolozane/tazobactam", "Nitrofurantoin")
BVBRC_YEAR_NULL_BELOW = 0.08
BVBRC_YEAR_DRAWN = (0.15, 0.27)
"""BV-BRC ``testing_standard_year`` from one uniform draw ``u`` per lab: ``u < 0.08`` blank;
``0.15 <= u < 0.27`` a year in 2016-2023 (no breakpoint table: S/I/R-only rows are dropped
by ingest); otherwise ``2024`` (the shipped tables). A year integer is drawn exactly when
``u >= 0.15`` so the random stream (and every genome) matches earlier generator versions."""
COMBINATION_DRUGS: dict[str, str] = {"piperacillin-tazobactam": "4"}
"""Drug -> fixed partner concentration written as ``x/<partner>`` on BV-BRC MIC rows whose
lab spells drug names in the ``abbrev`` or ``messy`` style (ingest keeps ``x``)."""
DISK_METHODS: tuple[str, ...] = ("Disk diffusion", "disk diffusion")
DROP_NULL_YEAR = "S/I/R-only row with null standard_year"
DROP_NO_TABLE_FOR_YEAR = "S/I/R-only row with no breakpoint table for standard_year"

DRUG_SPELLINGS: dict[str, dict[str, tuple[str, ...]]] = {
    "meropenem": {"full": ("meropenem", "Meropenem"), "abbrev": ("MEM", "MERO"),
                  "messy": ("meropenem ", "Meropenem trihydrate", "MEROPENEM")},
    "ceftriaxone": {"full": ("ceftriaxone", "Ceftriaxone"), "abbrev": ("CRO",),
                    "messy": ("ceftriaxone ", "Ceftriaxone sodium", "CEFTRIAXONE")},
    "ciprofloxacin": {"full": ("ciprofloxacin", "Ciprofloxacin"), "abbrev": ("CIP",),
                      "messy": ("ciprofloxacin ", "Cipro", "ciprofloxacin hydrochloride")},
    "gentamicin": {"full": ("gentamicin", "Gentamicin"), "abbrev": ("GEN", "CN"),
                   "messy": ("gentamycin", "Gentamicin sulfate", "gentamicin ")},
    "piperacillin-tazobactam": {"full": ("piperacillin-tazobactam", "Piperacillin/tazobactam"),
                                "abbrev": ("TZP", "PIP/TAZ"),
                                "messy": ("piperacillin/tazobactam ", "Piperacillin-Tazobactam",
                                          "Piperacillin tazobactam")},
}
MIC_TEXT: dict[int, str] = {
    -7: "0.008", -6: "0.015", -5: "0.03", -4: "0.06", -3: "0.125", -2: "0.25", -1: "0.5",
    0: "1", 1: "2", 2: "4", 3: "8", 4: "16", 5: "32", 6: "64", 7: "128", 8: "256", 9: "512",
}
SIR_WORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "BVBRC": {"S": ("Susceptible", "Susceptible", "Susceptible", "susceptible", "SUSCEPTIBLE"),
              "I": ("Intermediate", "Intermediate", "Intermediate", "intermediate", "Intermediate "),
              "R": ("Resistant", "Resistant", "Resistant", "resistant", "RESISTANT")},
    "NCBI": {"S": ("susceptible",), "I": ("intermediate",), "R": ("resistant",)},
}

AMRFINDER_COLUMNS_NEW: tuple[str, ...] = (
    "Protein identifier", "Contig id", "Start", "Stop", "Strand", "Element symbol", "Element name",
    "Scope", "Type", "Subtype", "Class", "Subclass", "Method", "Target length",
    "Reference sequence length", "% Coverage of reference", "% Identity to reference",
    "Alignment length", "Closest reference accession", "Closest reference name", "HMM accession",
    "HMM description",
)
AMRFINDER_COLUMNS_OLD: tuple[str, ...] = tuple(
    {"Element symbol": "Gene symbol", "Element name": "Sequence name"}.get(c, c) for c in AMRFINDER_COLUMNS_NEW
)
BVBRC_COLUMNS: tuple[str, ...] = (
    "genome_id", "genome_name", "taxon_id", "biosample_accession", "antibiotic", "resistant_phenotype",
    "measurement_sign", "measurement_value", "measurement_unit", "laboratory_typing_method",
    "testing_standard", "testing_standard_year", "evidence", "isolation_source", "isolation_country",
    "collection_year",
)
NCBI_COLUMNS: tuple[str, ...] = (
    "biosample", "organism", "antibiotic", "resistance_phenotype", "measurement_sign", "measurement",
    "measurement_units", "laboratory_typing_method", "testing_standard", "isolation_source",
    "geo_loc_name", "collection_date",
)
METADATA_COLUMNS: tuple[str, ...] = ("genome_id", "biosample", "species", "source", "isolation_source",
                                     "country", "year")

_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
_CODE = np.zeros(256, dtype=np.uint8)
for _i, _b in enumerate(b"ACGT"):
    _CODE[_b] = _i


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

OriginalRecord = tuple[np.ndarray, list[tuple[str, int, int]], int, int, int, np.ndarray]
"""``(sequence, marker_spans, n_snps_vs_reference, extra_bases, backbone_length, contig_cuts)``
of a genome whose near-identical partner is built later."""


@dataclass
class Lineage:
    """One synthetic lineage: a backbone (core + SNPs) with accessory blocks."""

    species: str
    index: int
    tier: str
    st: str
    alleles: tuple[int, ...]
    backbone: np.ndarray  # uint8 ASCII codes
    accessory: list[tuple[int, np.ndarray]]  # (offset into backbone, block)
    effects: dict[str, float]  # per drug lineage effect (log2 steps)


@dataclass
class SpeciesModel:
    """Core sequence, lineages and fixed marker loci for one species."""

    species: str
    core: np.ndarray
    lineages: list[Lineage] = field(default_factory=list)
    fixed_loci: dict[str, int] = field(default_factory=dict)


@dataclass
class LabProfile:
    """How one laboratory reports results (shared by all rows of a genome in one source)."""

    standard: str  # the standard the lab really used
    report_standard: str  # what the export says (may be "" or "missing")
    standard_year: str  # BV-BRC only; may be ""
    method: str
    unit: str
    name_style: str
    panel_shrink_top: int  # 0 or 1: narrower panel on the high side


@dataclass
class GenomePlan:
    """Everything decided about one genome before its sequence is built."""

    index: int
    species: str
    genome_id: str
    biosample: str
    source: str  # primary source: BVBRC / NCBI
    in_both: bool
    lineage: Lineage
    qc_fail: str | None
    pair_of: str | None  # genome_id of the near-identical original
    markers: list[str]  # present blocks, incl. hidden ones
    country: str
    year: int
    isolation_source: str
    drugs: list[str]
    old_header: bool
    lab: LabProfile
    second_lab: LabProfile | None  # NCBI re-report of a BV-BRC biosample
    truth: dict[str, int] = field(default_factory=dict)  # drug -> true log2 step
    seq_species: str = ""  # species of the actual sequence (differs for wrong_species)
    seq_lineage: Lineage | None = None
    total_length: int = 0
    n_contigs: int = 0
    n_snps_vs_reference: int = 0
    extra_bases: int = 0
    backbone_length: int = 0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _random_dna_array(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` random bases as uint8 ASCII codes."""
    return _BASES[rng.integers(0, 4, size=int(n))]


def _apply_snps(rng: np.random.Generator, seq: np.ndarray, n: int, forbidden: np.ndarray | None = None) -> int:
    """Mutate ``n`` distinct positions of ``seq`` in place to a *different* base.

    ``forbidden`` is an optional boolean mask of positions that must not change.
    Returns the number of positions actually mutated.
    """
    if n <= 0 or seq.size == 0:
        return 0
    if forbidden is None:
        candidates = seq.size
        positions = rng.choice(candidates, size=min(int(n), candidates), replace=False)
    else:
        allowed = np.flatnonzero(~forbidden)
        if allowed.size == 0:
            return 0
        positions = allowed[rng.choice(allowed.size, size=min(int(n), allowed.size), replace=False)]
    shift = rng.integers(1, 4, size=positions.size).astype(np.uint8)
    seq[positions] = _BASES[(_CODE[seq[positions]] + shift) % 4]
    return int(positions.size)


def _choice(rng: np.random.Generator, items: Sequence[Any], weights: Sequence[float] | None = None) -> Any:
    """Weighted choice of one item (weights normalized)."""
    if weights is None:
        return items[int(rng.integers(0, len(items)))]
    p = np.asarray(weights, dtype=np.float64)
    return items[int(rng.choice(len(items), p=p / p.sum()))]


def _mic_text(step: int, rng: np.random.Generator) -> str:
    """Lab-style text for a doubling step, with occasional ``0.12`` / ``8.0`` spellings."""
    if step == -3 and rng.random() < 0.2:
        return "0.12"
    text = MIC_TEXT.get(step)
    if text is None:
        return format(math.ldexp(1.0, step), "g")
    if step >= 0 and rng.random() < 0.08:
        return f"{text}.0"
    return text


def _mash_distance_from_kmers(core_len: int, backbone_len: int, n_snps: int, extra_bases: int,
                              k: int = MASH_K) -> float:
    """Mash distance a k-mer sketch would report between a genome and its species core.

    Analytic approximation: a k-mer of the backbone survives when none of its ``k``
    bases is a SNP (``(1 - rate) ** k``); only ``min(backbone, core)`` bases of the
    backbone can match the core (truncation loses, duplication adds nothing new);
    inserted blocks add k-mers to the genome only. ``d = -ln(2j / (1 + j)) / k``.
    """
    if core_len <= 0 or backbone_len <= 0:
        return 1.0
    rate = min(1.0, n_snps / backbone_len)
    intact = (1.0 - rate) ** k
    matchable = min(backbone_len, core_len)
    shared = matchable * intact
    union = core_len + matchable * (1.0 - intact) + extra_bases
    j = shared / union if union else 0.0
    if j <= 0.0:
        return 1.0
    if j >= 1.0:
        return 0.0
    return -math.log(2.0 * j / (1.0 + j)) / k


def _shared_hashes(distance: float, k: int = MASH_K, s: int = MASH_SKETCH_SIZE) -> str:
    """Mash ``shared-hashes`` text (``"985/1000"``) implied by a distance."""
    j = 1.0 / (2.0 * math.exp(k * distance) - 1.0) if distance < 1.0 else 0.0
    shared = max(1, int(round(j * s))) if distance < 1.0 else 0
    return f"{shared}/{s}"


def _sir(config: Config, species: str, drug: str, standard: str, mic: float) -> str:
    """S/I/R under the 2024 table of ``standard``; falls back to EUCAST if the pair is missing."""
    bp = config.breakpoint(species, drug, standard, 2024)
    if bp is None:
        bp = config.breakpoint(species, drug, "EUCAST", 2024)
    if bp is None:
        raise ValueError(f"no breakpoint for {species} x {drug} under {standard}")
    return sir_from_mic(mic, bp)


def _wrap(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


# --------------------------------------------------------------------------- #
# Species models and lineages
# --------------------------------------------------------------------------- #


def _build_species_models(rng: np.random.Generator, genome_length: int) -> dict[str, SpeciesModel]:
    """Random core per species (all five) plus lineages for the labelled species."""
    models: dict[str, SpeciesModel] = {}
    for species in SPECIES_ORDER:
        models[species] = SpeciesModel(species=species, core=_random_dna_array(rng, genome_length))
    scale = genome_length / 60_000.0
    for species in LABELLED_SPECIES:
        model = models[species]
        fixed_names = [m.name for m in mk.markers_for_species(species) if m.locus == "fixed"]
        loci = np.sort(rng.choice(np.arange(1, genome_length), size=len(fixed_names), replace=False))
        model.fixed_loci = {name: int(pos) for name, pos in zip(fixed_names, loci)}
        model.lineages = _build_lineages(rng, species, model.core, scale)
    return models


def _build_lineages(rng: np.random.Generator, species: str, core: np.ndarray, scale: float) -> list[Lineage]:
    """``N_LINEAGES`` lineages: SNP backbone, accessory blocks, ST, lineage MIC effects."""
    pool = list(ST_POOL[species])
    others = [pool[i] for i in rng.choice(len(pool), size=N_LINEAGES - 2, replace=False)]
    sts = list(HIGH_RISK_ST[species]) + others
    lineages: list[Lineage] = []
    n_snps = max(1, int(round(LINEAGE_SNP_RATE * core.size)))
    for index in range(N_LINEAGES):
        backbone = core.copy()
        _apply_snps(rng, backbone, n_snps)
        n_blocks = int(rng.integers(ACCESSORY_BLOCKS[0], ACCESSORY_BLOCKS[1] + 1))
        accessory: list[tuple[int, np.ndarray]] = []
        for _ in range(n_blocks):
            length = max(100, int(round(float(rng.integers(ACCESSORY_LENGTH[0], ACCESSORY_LENGTH[1] + 1)) * scale)))
            offset = int(rng.integers(1, core.size))
            accessory.append((offset, _random_dna_array(rng, length)))
        st = sts[index]
        profile = KNOWN_PROFILES.get(st)
        alleles = profile if profile is not None else tuple(int(a) for a in rng.integers(1, 120, size=7))
        effects = {drug: float(rng.normal(0.0, mk.LINEAGE_SD)) for drug in mk.DRUGS}
        lineages.append(Lineage(species, index, TIER_OF_INDEX[index], st, alleles, backbone, accessory, effects))
    return lineages


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def _lab_profile(rng: np.random.Generator, source: str) -> LabProfile:
    """Draw one laboratory's reporting conventions."""
    if source == "BVBRC":
        standard = _choice(rng, STANDARDS, (0.45, 0.55))
        report_standard = standard
        u = rng.random()
        drawn = int(rng.integers(2016, 2024)) if u >= BVBRC_YEAR_DRAWN[0] else None
        if u < BVBRC_YEAR_NULL_BELOW:
            year = ""
        elif drawn is not None and u < BVBRC_YEAR_DRAWN[1]:
            year = str(drawn)
        else:
            year = "2024"
        method = _choice(rng, BVBRC_METHODS, BVBRC_METHOD_WEIGHTS)
    else:
        standard = _choice(rng, STANDARDS, (0.30, 0.70))
        u = rng.random()
        report_standard = standard if u < 0.86 else ("" if u < 0.94 else "missing")
        year = ""
        method = _choice(rng, NCBI_METHODS, NCBI_METHOD_WEIGHTS)
    unit = _choice(rng, UNITS, UNIT_WEIGHTS)
    style = _choice(rng, NAME_STYLES, NAME_STYLE_WEIGHTS)
    shrink = 1 if rng.random() < 0.3 else 0
    return LabProfile(standard, report_standard, year, method, unit, style, shrink)


def _draw_markers(rng: np.random.Generator, species: str, tier: str) -> list[str]:
    """Present blocks for one genome given its lineage tier (fixed marker order)."""
    tier_idx = mk.TIERS.index(tier)
    table = mk.PRESENCE[species]
    present: list[str] = []
    for spec in mk.markers_for_species(species):
        name = spec.name
        if name in mk.CONDITIONAL_ON:
            parent, p_if, p_else = mk.CONDITIONAL_ON[name]
            p = p_if if parent in present else p_else
        else:
            p = table[name][tier_idx]
        if name in mk.EXCLUSIVE_WITH and mk.EXCLUSIVE_WITH[name] in present:
            continue
        if rng.random() < p:
            present.append(name)
    return present


def _unique_biosamples(rng: np.random.Generator, n: int) -> list[str]:
    """``n`` distinct ``SAMN########`` accessions."""
    out: list[str] = []
    seen: set[int] = set()
    while len(out) < n:
        for value in rng.integers(10_000_000, 99_999_999, size=max(8, 2 * (n - len(out)))):
            v = int(value)
            if v not in seen:
                seen.add(v)
                out.append(f"SAMN{v:08d}")
                if len(out) == n:
                    break
    return out


def _plan_species(
    rng: np.random.Generator,
    species: str,
    n: int,
    models: dict[str, SpeciesModel],
    biosamples: list[str],
    first_bvbrc_number: int,
) -> list[GenomePlan]:
    """Decide ids, sources, lineages, markers, QC failures, pairs and labs for ``n`` genomes."""
    model = models[species]
    weights = np.asarray(LINEAGE_WEIGHTS) * rng.uniform(0.7, 1.3, size=N_LINEAGES)
    weights /= weights.sum()
    lineage_idx = rng.choice(N_LINEAGES, size=n, p=weights)

    # ~4% fail QC; at tiny sizes still plant every kind once (n // 5 caps it for n < 20).
    n_fail = max(int(round(QC_FAIL_FRACTION * n)), min(len(QC_FAIL_KINDS), n // 5))
    fail_kind: dict[int, str] = {}
    if n_fail:
        fail_positions = np.sort(rng.choice(n, size=n_fail, replace=False))
        for j, pos in enumerate(fail_positions):
            fail_kind[int(pos)] = QC_FAIL_KINDS[j % len(QC_FAIL_KINDS)]

    normal = [i for i in range(n) if i not in fail_kind]
    n_pairs = min(int(rng.integers(NEAR_IDENTICAL_PAIRS[0], NEAR_IDENTICAL_PAIRS[1] + 1)), len(normal) // 2)
    pair_members = np.sort(rng.choice(len(normal), size=2 * n_pairs, replace=False))
    partner_of: dict[int, int] = {}
    for j in range(n_pairs):
        original, partner = normal[int(pair_members[2 * j])], normal[int(pair_members[2 * j + 1])]
        partner_of[partner] = original

    is_bvbrc = rng.random(n) < BVBRC_FRACTION
    bvbrc_positions = [i for i in range(n) if is_bvbrc[i]]
    n_both = min(int(round(OVERLAP_FRACTION * n)), len(bvbrc_positions))
    both = set(int(bvbrc_positions[k]) for k in rng.choice(len(bvbrc_positions), size=n_both, replace=False))
    n_old = max(1, int(round(OLD_HEADER_FRACTION * n))) if n else 0
    old_header = set(int(x) for x in rng.choice(n, size=min(n_old, n), replace=False))

    plans: list[GenomePlan] = []
    bvbrc_number = first_bvbrc_number
    for i in range(n):
        biosample = biosamples[i]
        source = "BVBRC" if is_bvbrc[i] else "NCBI"
        if source == "BVBRC":
            genome_id = f"{BVBRC_TAXON[species]}.{bvbrc_number}"
            bvbrc_number += 1
        else:
            genome_id = f"NCBI_{biosample}"
        original = partner_of.get(i)
        if original is not None:
            lineage = plans[original].lineage
            markers = list(plans[original].markers)
            qc_fail = None
        else:
            lineage = model.lineages[int(lineage_idx[i])]
            markers = _draw_markers(rng, species, lineage.tier)
            qc_fail = fail_kind.get(i)
        country = _choice(rng, COUNTRIES, COUNTRY_WEIGHTS)
        year = int(_choice(rng, YEARS, YEAR_WEIGHTS))
        isolation = _choice(rng, ISOLATION_SOURCES, ISOLATION_WEIGHTS)
        coverage = DRUG_COVERAGE[species]
        drugs = [d for d in mk.DRUGS if rng.random() < coverage[d]]
        lab = _lab_profile(rng, source)
        second = _lab_profile(rng, "NCBI") if i in both else None
        plans.append(GenomePlan(
            index=i, species=species, genome_id=genome_id, biosample=biosample, source=source,
            in_both=i in both, lineage=lineage, qc_fail=qc_fail,
            pair_of=plans[original].genome_id if original is not None else None,
            markers=markers, country=country, year=year, isolation_source=isolation, drugs=drugs,
            old_header=i in old_header, lab=lab, second_lab=second,
        ))
    return plans


def _draw_truth(rng: np.random.Generator, plan: GenomePlan) -> None:
    """True log2 MIC step for every drug (not only the ones with a lab result)."""
    present = mk.present_set(plan.markers, plan.species)
    for drug in mk.DRUGS:
        value = (mk.DRUG_BASE_LOG2[drug] + mk.effect_sum(present, drug) + plan.lineage.effects[drug]
                 + float(rng.normal(0.0, mk.NOISE_SD)))
        plan.truth[drug] = int(math.floor(value + 0.5))


# --------------------------------------------------------------------------- #
# Genome assembly
# --------------------------------------------------------------------------- #


def _assemble(
    rng: np.random.Generator,
    plan: GenomePlan,
    models: dict[str, SpeciesModel],
    seqs: dict[str, np.ndarray],
    originals: dict[str, "OriginalRecord"],
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Full genome sequence (uint8 ASCII) and marker spans ``(name, start, end)`` (0-based, end exclusive).

    ``originals`` caches ``(sequence, spans, n_snps_vs_reference, extra_bases,
    backbone_length, contig_cuts)`` of genomes whose near-identical partner is still
    to be built.
    """
    if plan.pair_of is not None:
        seq, spans, n_snps, extra, backbone_len, _ = originals[plan.pair_of]
        seq = seq.copy()
        forbidden = np.zeros(seq.size, dtype=bool)
        for _, s, e in spans:
            forbidden[s:e] = True
        n = int(rng.integers(0, 2))  # 0 or 1 private SNP
        plan.n_snps_vs_reference = n_snps + _apply_snps(rng, seq, n, forbidden)
        plan.extra_bases = extra
        plan.backbone_length = backbone_len
        plan.seq_species = plan.species
        plan.seq_lineage = plan.lineage
        return seq, list(spans)

    seq_species = plan.species
    lineage = plan.lineage
    if plan.qc_fail == "wrong_species":
        seq_species = OTHER_SPECIES[plan.species]
        other = models[seq_species].lineages
        lineage = other[int(rng.integers(0, len(other)))]
    plan.seq_species = seq_species
    plan.seq_lineage = lineage
    model = models[seq_species]

    if plan.qc_fail == "too_distant":
        backbone = model.core.copy()
        n_snps = int(round(DISTANT_SNP_RATE * backbone.size))
        _apply_snps(rng, backbone, n_snps)
        plan.n_snps_vs_reference = n_snps
    else:
        backbone = lineage.backbone.copy()
        plan.n_snps_vs_reference = max(1, int(round(LINEAGE_SNP_RATE * backbone.size)))
    if plan.qc_fail == "wrong_size":
        if plan.index % 2 == 0:
            backbone = backbone[: int(0.45 * backbone.size)].copy()
        else:
            backbone = np.concatenate([backbone, backbone[: int(0.7 * backbone.size)]])
    n_private = max(1, int(round(PRIVATE_SNP_RATE * backbone.size)))
    plan.n_snps_vs_reference += _apply_snps(rng, backbone, n_private)
    plan.backbone_length = int(backbone.size)

    # Insertions: (offset, order, name, block). Accessory blocks are unnamed.
    inserts: list[tuple[int, int, str | None, np.ndarray]] = []
    order = 0
    for offset, block in lineage.accessory:
        inserts.append((min(offset, backbone.size), order, None, block))
        order += 1
    loci = models[plan.species].fixed_loci
    for name in plan.markers:
        spec_locus = mk.marker_by_name()[name].locus
        if spec_locus == "fixed":
            offset = min(loci[name], backbone.size)
        else:
            offset = int(rng.integers(1, backbone.size))
        inserts.append((offset, order, name, seqs[name]))
        order += 1
    inserts.sort(key=lambda t: (t[0], t[1]))

    pieces: list[np.ndarray] = []
    spans: list[tuple[str, int, int]] = []
    prev = 0
    length = 0
    extra = 0
    for offset, _, name, block in inserts:
        if offset > prev:
            pieces.append(backbone[prev:offset])
            length += offset - prev
            prev = offset
        if name is not None:
            spans.append((name, length, length + block.size))
        pieces.append(block)
        length += block.size
        extra += block.size
    pieces.append(backbone[prev:])
    plan.extra_bases = extra
    return np.concatenate(pieces), spans


def _cut_positions(rng: np.random.Generator, length: int, spans: list[tuple[str, int, int]], n_contigs: int) -> np.ndarray:
    """Sorted contig boundaries that never fall inside a marker span.

    Contigs are kept at least ``min_gap`` bp long (200 bp, or less when ``n_contigs``
    is large relative to the genome, as for the deliberately fragmented genomes).
    Picks are topped up in a few rounds so the requested count is reached whenever
    the allowed positions permit it.
    """
    min_gap = max(1, min(200, length // (4 * max(n_contigs, 1))))
    allowed = np.ones(length + 1, dtype=bool)
    allowed[: min_gap] = False
    allowed[max(length + 1 - min_gap, 0):] = False
    for _, s, e in spans:
        allowed[s + 1 : e] = False
    candidates = np.flatnonzero(allowed)
    k = min(max(n_contigs - 1, 0), candidates.size)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    chosen: list[int] = []
    blocked = np.zeros(length + 1, dtype=bool)
    for _ in range(4):
        need = k - len(chosen)
        if need <= 0:
            break
        pool = candidates[~blocked[candidates]]
        if pool.size == 0:
            break
        picks = np.sort(pool[rng.choice(pool.size, size=min(need, pool.size), replace=False)])
        for p in picks.tolist():
            if not blocked[p]:
                chosen.append(p)
                blocked[max(0, p - min_gap + 1) : p + min_gap] = True
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _contig_coords(cuts: np.ndarray, start: int, end: int) -> tuple[int, int, int]:
    """``(contig_index, start_1based, stop_1based)`` of a span given the cut positions."""
    idx = int(np.searchsorted(cuts, start, side="right"))
    contig_start = 0 if idx == 0 else int(cuts[idx - 1])
    return idx, start - contig_start + 1, end - contig_start


# --------------------------------------------------------------------------- #
# Interim tool outputs
# --------------------------------------------------------------------------- #


def _write_amrfinder(path: Path, plan: GenomePlan, spans: list[tuple[str, int, int]], cuts: np.ndarray,
                     seqs: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    """AMRFinderPlus 4.x table; ~5% of genomes use the older ``Gene symbol`` header."""
    by_name = mk.marker_by_name()
    header = AMRFINDER_COLUMNS_OLD if plan.old_header else AMRFINDER_COLUMNS_NEW
    lines = ["\t".join(header)]
    for name, s, e in spans:
        spec = by_name[name]
        if not spec.reportable:
            continue
        contig_idx, start, stop = _contig_coords(cuts, s, e)
        aa_len = seqs[name].size // 3
        if spec.subtype == "POINT":
            method, identity = "POINTX", "99.70"
        else:
            u = rng.random()
            if u < 0.85:
                method, identity = "EXACTX", "100.00"
            elif u < 0.95:
                method, identity = "ALLELEX", "100.00"
            else:
                method, identity = "BLASTX", f"{rng.uniform(98.5, 99.9):.2f}"
        row = (
            "NA", f"contig_{contig_idx + 1}", str(start), str(stop), "+", spec.name, spec.element_name,
            spec.scope, spec.marker_type, spec.subtype, spec.amr_class, spec.subclass, method, str(aa_len),
            str(aa_len), "100.00", identity, str(aa_len), spec.accession, spec.element_name, "NA", "NA",
        )
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mash(path: Path, plan: GenomePlan, genome_length: int, rng: np.random.Generator, root_rel_fasta: str) -> None:
    """``mash dist`` style lines (no header): reference, query, distance, p-value, shared-hashes."""
    own = _mash_distance_from_kmers(genome_length, plan.backbone_length, plan.n_snps_vs_reference, plan.extra_bases)
    own = max(1e-4, own * float(rng.uniform(0.95, 1.05)))
    lines = []
    for species in SPECIES_ORDER:
        if species == plan.seq_species:
            d = own
            p = "0"
        else:
            d = float(rng.uniform(0.30, 0.50))
            p = f"{rng.uniform(1e-6, 1e-2):.3g}"
        lines.append(f"{species}\t{root_rel_fasta}\t{d:.6g}\t{p}\t{_shared_hashes(d)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mlst(path: Path, plan: GenomePlan, root_rel_fasta: str) -> None:
    """``mlst`` style line (no header): file, scheme, ST, allele(number) columns."""
    lineage = plan.seq_lineage
    assert lineage is not None
    scheme, loci = MLST_SCHEME[plan.seq_species]
    if plan.qc_fail == "too_distant":
        st = "-"
        alleles = [f"{locus}(~{a})" for locus, a in zip(loci, lineage.alleles)]
    else:
        st = lineage.st
        alleles = [f"{locus}({a})" for locus, a in zip(loci, lineage.alleles)]
    path.write_text("\t".join([root_rel_fasta, scheme, st, *alleles]) + "\n", encoding="utf-8")


def _write_pheno_table(path: Path, plan: GenomePlan) -> None:
    """ResFinder 4 phenotype table derived from the planted acquired genes only."""
    by_name = mk.marker_by_name()
    acquired = [n for n in plan.markers if by_name[n].reportable and by_name[n].subtype == "AMR"]
    lines = [
        "# ResFinder phenotype results (SYNTHETIC: generated by genome2mic.synthetic, not a real run).",
        f"# Sample: {plan.genome_id}.fasta",
        "# ",
        "# The phenotype 'No resistance' should be interpreted with",
        "# caution, as it only means that nothing in the used",
        "# database indicate resistance, but resistance could exist",
        "# from 'unknown' or not yet implemented sources.",
        "# ",
        "# The 'Match' column stores one of the integers 0, 1, 2, 3.",
        "#      0: No match found",
        "#      1: Match < 100% ID AND match length < ref length",
        "#      2: Match = 100% ID AND match length < ref length",
        "#      3: Match = 100% ID AND match length = ref length",
        "# If several hits causing the same resistance are found,",
        "# the highest number will be stored in the 'Match' column.",
        "",
        "# Antimicrobial\tClass\tWGS-predicted phenotype\tMatch\tGenetic background",
    ]
    for antimicrobial, cls in mk.RESFINDER_ANTIMICROBIALS:
        hits = [n for n in acquired if antimicrobial in mk.RESFINDER_CALLS.get(n, ())]
        if hits:
            background = ", ".join(f"{n} ({n}_{by_name[n].accession})" for n in hits)
            lines.append(f"{antimicrobial}\t{cls}\tResistant\t3\t{background}")
        else:
            lines.append(f"{antimicrobial}\t{cls}\tNo resistance\t0\t")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_genome(
    rng: np.random.Generator,
    paths: Paths,
    plan: GenomePlan,
    models: dict[str, SpeciesModel],
    seqs: dict[str, np.ndarray],
    originals: dict[str, "OriginalRecord"],
    genome_length: int,
    keep_for_pair: bool,
) -> None:
    """Build one genome, write its FASTA and the four interim tool outputs."""
    seq, spans = _assemble(rng, plan, models, seqs, originals)
    if plan.pair_of is not None:
        # Re-submitted assemblies share their contig boundaries. Every boundary costs
        # k-1 k-mers, so distinct cuts alone would push a pair's sketch distance well
        # above the 1e-4 outbreak threshold on these small genomes.
        cuts = originals[plan.pair_of][5]
    else:
        if plan.qc_fail == "too_many_contigs":
            n_contigs = int(rng.integers(FRAGMENTED_CONTIGS[0], FRAGMENTED_CONTIGS[1] + 1))
        else:
            n_contigs = int(rng.integers(CONTIGS[0], CONTIGS[1] + 1))
        cuts = _cut_positions(rng, seq.size, spans, n_contigs)
    if keep_for_pair:
        originals[plan.genome_id] = (seq, spans, plan.n_snps_vs_reference, plan.extra_bases,
                                     plan.backbone_length, cuts)
    bounds = [0, *cuts.tolist(), seq.size]
    text = seq.tobytes().decode("ascii")
    records = []
    for i in range(len(bounds) - 1):
        piece = text[bounds[i] : bounds[i + 1]]
        records.append((f"contig_{i + 1} length={len(piece)} synthetic=true genome={plan.genome_id}", piece))
    write_fasta(records, paths.genome_fasta(plan.genome_id))
    plan.total_length = seq.size
    plan.n_contigs = len(records)

    interim = paths.interim_dir(plan.genome_id)
    (interim / "resfinder").mkdir(parents=True, exist_ok=True)
    rel_fasta = f"data/raw/genomes/{plan.genome_id}.fasta"
    _write_amrfinder(interim / "amrfinder.tsv", plan, spans, cuts, seqs, rng)
    _write_mash(interim / "mash.tsv", plan, genome_length, rng, rel_fasta)
    _write_mlst(interim / "mlst.tsv", plan, rel_fasta)
    _write_pheno_table(interim / "resfinder" / "pheno_table.txt", plan)


# --------------------------------------------------------------------------- #
# Label rows
# --------------------------------------------------------------------------- #


def _row_from_step(
    rng: np.random.Generator,
    plan: GenomePlan,
    drug: str,
    step: int,
    lab: LabProfile,
    source: str,
    config: Config,
    force_numeric: bool = False,
) -> dict[str, Any]:
    """One lab result for ``drug`` at true log2 ``step`` as reported by ``lab``.

    Returns the common fields plus ``sir`` (canonical), ``drop`` (expected-drop
    reason or ``None``) and ``sir_only`` (bool).
    """
    lo, hi = mk.PANELS[drug]
    hi -= lab.panel_shrink_top
    spellings = DRUG_SPELLINGS[drug][lab.name_style]
    antibiotic = spellings[int(rng.integers(0, len(spellings)))]
    method = lab.method
    drop: str | None = None
    sir_only = (method in ("Disk diffusion", "disk diffusion", "no_value")) and not force_numeric
    sign = value = unit = ""
    if sir_only:
        mic_for_sir = math.ldexp(1.0, step)
        if method == "no_value":
            method = "Broth dilution" if source == "BVBRC" else "broth microdilution"
        standard = lab.report_standard
        if source == "BVBRC" and rng.random() < 0.2:
            standard = ""
    else:
        if method == "no_value":
            method = "Broth dilution" if source == "BVBRC" else "broth microdilution"
        if step <= lo:
            if rng.random() < 0.8:
                sign, value = "<=", _mic_text(lo, rng)
            else:
                sign, value = "<", _mic_text(lo + 1, rng)
            mic_for_sir = math.ldexp(1.0, lo)
        elif step > hi:
            if rng.random() < 0.75:
                sign, value = ">", _mic_text(hi, rng)
            else:
                sign, value = ">=", _mic_text(hi + 1, rng)
            mic_for_sir = math.ldexp(1.0, hi + 1)
        else:
            if source == "NCBI":
                sign = "=="
            else:
                sign = "=" if rng.random() < 0.6 else ""
            value = _mic_text(step, rng)
            mic_for_sir = math.ldexp(1.0, step)
        unit = lab.unit
        standard = lab.report_standard
        partner = COMBINATION_DRUGS.get(drug)
        if partner is not None and source == "BVBRC" and lab.name_style != "full":
            value = f"{value}/{partner}"
        if method in ("Vitek 2", "automated system"):
            drop = "unknown typing method"
    if sir_only or method in DISK_METHODS:
        # Ingest reads these on the S/I/R path (a disk value is a zone diameter, even the
        # forced-numeric conflict rows of a disk lab), so predict its drop reason here.
        drop = _sir_path_drop(config, standard, lab.standard_year)
    sir = _sir(config, plan.species, drug, lab.standard, mic_for_sir)
    words = SIR_WORDS[source][sir]
    return {
        "antibiotic": antibiotic,
        "phenotype": words[int(rng.integers(0, len(words)))],
        "sign": sign,
        "value": value,
        "unit": unit,
        "method": method,
        "standard": standard,
        "standard_year": lab.standard_year if standard else "",
        "sir": sir,
        "sir_only": sir_only,
        "step": step,
        "drop": drop,
    }


def _sir_path_drop(config: Config, standard: str, standard_year: str) -> str | None:
    """Drop reason ingest will log for an S/I/R-path row, or ``None`` if it converts.

    Mirrors ``harmonize``: a blank/``missing`` standard, then a blank year (NCBI exports
    never have one), then a year without a breakpoint table. Only the 2024 tables ship.
    """
    if standard in ("", "missing"):
        return "S/I/R-only row with blank standard"
    if not standard_year:
        return DROP_NULL_YEAR
    if not config.has_breakpoint_table(standard, int(standard_year)):
        return DROP_NO_TABLE_FOR_YEAR
    return None


def _conflict_step(config: Config, plan: GenomePlan, drug: str, sir: str, lab: LabProfile) -> int:
    """A log2 step whose category under ``lab`` is on the other side of the S/R divide."""
    bp = config.breakpoint(plan.species, drug, lab.standard, 2024) or config.breakpoint(plan.species, drug, "EUCAST", 2024)
    assert bp is not None
    lo, hi = mk.PANELS[drug]
    if sir == "S":
        return min(int(math.ceil(math.log2(bp.r_breakpoint))) + 2, hi + 1)
    return max(int(math.floor(math.log2(bp.s_breakpoint))) - 2, lo)


def _genome_name(rng: np.random.Generator, plan: GenomePlan) -> str:
    base = SPECIES_NAMES[plan.species]
    if plan.species == "KPNEU" and rng.random() < 0.3:
        base = "Klebsiella pneumoniae subsp. pneumoniae"
    tag = f"SYN-{plan.species[:2]}-{plan.index:04d}"
    return f"{base} strain {tag}" if rng.random() < 0.8 else f"{base} {tag}"


def _bvbrc_row(plan: GenomePlan, r: dict[str, Any], genome_name: str, evidence: str = "Laboratory Method") -> list[str]:
    return [
        plan.genome_id, genome_name, BVBRC_TAXON[plan.species], plan.biosample, r["antibiotic"], r["phenotype"],
        r["sign"], r["value"], r["unit"], r["method"], r["standard"], r["standard_year"], evidence,
        plan.isolation_source if plan.index % 3 else plan.isolation_source.capitalize(), plan.country,
        str(plan.year),
    ]


def _ncbi_row(rng: np.random.Generator, plan: GenomePlan, r: dict[str, Any]) -> list[str]:
    organism = SPECIES_NAMES[plan.species]
    if plan.species == "KPNEU" and rng.random() < 0.2:
        organism = "Klebsiella pneumoniae subsp. pneumoniae"
    u = rng.random()
    if u < 0.6:
        date = f"{plan.year}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}"
    elif u < 0.75:
        date = f"{plan.year}-{int(rng.integers(1, 13)):02d}"
    elif u < 0.9:
        date = str(plan.year)
    else:
        date = "missing"
    geo = plan.country
    if plan.country == "USA" and rng.random() < 0.7:
        geo = f"USA: {US_STATES[int(rng.integers(0, len(US_STATES)))]}"
    isolation = plan.isolation_source
    variants = NCBI_ISOLATION_SPELLING.get(isolation)
    if variants is not None:
        isolation = variants[int(rng.integers(0, len(variants)))]
    return [
        plan.biosample, organism, r["antibiotic"], r["phenotype"], r["sign"], r["value"], r["unit"],
        r["method"], r["standard"], isolation, geo, date,
    ]


def _build_label_rows(
    rng: np.random.Generator, plans: list[GenomePlan], config: Config, droplog: DropLog
) -> tuple[list[list[str]], list[list[str]], dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    """BV-BRC rows, NCBI rows, per-pair truth counts and planted-drop counts."""
    bvbrc: list[list[str]] = []
    ncbi: list[list[str]] = []
    counts: dict[tuple[str, str], dict[str, Any]] = {}
    planted = {
        "evidence == Computational Prediction": 0,
        "S/I/R-only row with blank standard": 0,
        DROP_NULL_YEAR: 0,
        DROP_NO_TABLE_FOR_YEAR: 0,
        "unknown antibiotic name": 0,
        "unknown typing method": 0,
        "conflicting cross-source duplicate (genome x drug pairs)": 0,
        "within-source duplicate rows (consistent)": 0,
        "biosamples present in both sources": 0,
    }
    for plan in plans:
        genome_name = _genome_name(rng, plan)
        primary = plan.source
        for drug in plan.drugs:
            step = plan.truth[drug]
            r = _row_from_step(rng, plan, drug, step, plan.lab, primary, config)
            if r["drop"] is not None:
                planted[r["drop"]] += 1
            else:
                c = counts.setdefault((plan.species, drug), {"n": 0, "n_nonS": 0, "n_S": 0, "levels": set()})
                c["n"] += 1
                c["n_nonS" if r["sir"] != "S" else "n_S"] += 1
                if not r["sir_only"]:
                    lo, hi = mk.PANELS[drug]
                    c["levels"].add(min(max(r["step"], lo), hi + 1))
            if primary == "BVBRC":
                bvbrc.append(_bvbrc_row(plan, r, genome_name))
                if rng.random() < WITHIN_SOURCE_DUPLICATE_FRACTION:
                    dup_lab = LabProfile(plan.lab.standard, plan.lab.standard, plan.lab.standard_year,
                                         "Disk diffusion", "", plan.lab.name_style, 0)
                    r2 = _row_from_step(rng, plan, drug, step, dup_lab, "BVBRC", config)
                    bvbrc.append(_bvbrc_row(plan, r2, genome_name))
                    planted["within-source duplicate rows (consistent)"] += 1
                    if r2["drop"] is not None:
                        planted[r2["drop"]] += 1
            else:
                ncbi.append(_ncbi_row(rng, plan, r))
        if primary == "BVBRC":
            for drug in mk.DRUGS:
                if rng.random() < COMPUTATIONAL_FRACTION:
                    true_sir = _sir(config, plan.species, drug, plan.lab.standard, math.ldexp(1.0, plan.truth[drug]))
                    pred = true_sir if rng.random() < 0.85 else ("S" if true_sir != "S" else "R")
                    r = {"antibiotic": DRUG_SPELLINGS[drug]["full"][0], "phenotype": SIR_WORDS["BVBRC"][pred][0],
                         "sign": "", "value": "", "unit": "", "method": "", "standard": "", "standard_year": ""}
                    bvbrc.append(_bvbrc_row(plan, r, genome_name, evidence="Computational Prediction"))
                    planted["evidence == Computational Prediction"] += 1
        if rng.random() < UNKNOWN_DRUG_FRACTION:
            name = UNKNOWN_DRUG_NAMES[int(rng.integers(0, len(UNKNOWN_DRUG_NAMES)))]
            r = {"antibiotic": name, "phenotype": "Susceptible" if primary == "BVBRC" else "susceptible",
                 "sign": "=" if primary == "BVBRC" else "==", "value": "1", "unit": "mg/L",
                 "method": "Broth dilution" if primary == "BVBRC" else "broth microdilution",
                 "standard": plan.lab.standard, "standard_year": plan.lab.standard_year}
            (bvbrc if primary == "BVBRC" else ncbi).append(
                _bvbrc_row(plan, r, genome_name) if primary == "BVBRC" else _ncbi_row(rng, plan, r))
            planted["unknown antibiotic name"] += 1
        if plan.in_both and plan.second_lab is not None:
            planted["biosamples present in both sources"] += 1
            second = plan.second_lab
            drugs = [d for d in plan.drugs if rng.random() < 0.8] or plan.drugs[:1]
            conflict_drug = drugs[int(rng.integers(0, len(drugs)))] if rng.random() < CONFLICT_FRACTION else None
            for drug in drugs:
                step = plan.truth[drug]
                if drug == conflict_drug:
                    true_sir = _sir(config, plan.species, drug, second.standard, math.ldexp(1.0, step))
                    step = _conflict_step(config, plan, drug, true_sir, second)
                    r = _row_from_step(rng, plan, drug, step, second, "NCBI", config, force_numeric=True)
                    planted["conflicting cross-source duplicate (genome x drug pairs)"] += 1
                else:
                    if rng.random() < 0.3:
                        step += int(rng.integers(0, 2)) * 2 - 1  # +-1 step: intersection rule
                    r = _row_from_step(rng, plan, drug, step, second, "NCBI", config)
                if r["drop"] is not None:
                    planted[r["drop"]] += 1
                ncbi.append(_ncbi_row(rng, plan, r))
    for reason, n in planted.items():
        if reason == "biosamples present in both sources" or reason.startswith("within-source"):
            droplog.drop(f"planted: {reason} (not a drop; ingest should merge)", n)
        else:
            droplog.drop(f"planted expected_drop: {reason}", n)
    return bvbrc, ncbi, counts, planted


def _log_inclusion(counts: dict[tuple[str, str], dict[str, Any]], droplog: DropLog) -> list[dict[str, Any]]:
    """Log the planned count table and which pairs fail the 50/50/4 inclusion rule."""
    table: list[dict[str, Any]] = []
    failing: list[str] = []
    for species in LABELLED_SPECIES:
        for drug in mk.DRUGS:
            c = counts.get((species, drug), {"n": 0, "n_nonS": 0, "n_S": 0, "levels": set()})
            passes = c["n_nonS"] >= 50 and c["n_S"] >= 50 and len(c["levels"]) >= 4
            row = {"species": species, "drug": drug, "n": c["n"], "n_nonsusceptible": c["n_nonS"],
                   "n_susceptible": c["n_S"], "n_distinct_mic": len(c["levels"]), "passes_inclusion": passes}
            table.append(row)
            planned_fail = (species, drug) in PLANNED_TO_FAIL_INCLUSION
            level = logging.INFO if passes or planned_fail else logging.WARNING
            logger.log(level, "[synth] planned labels %s x %s: n=%d non-S=%d S=%d levels=%d -> %s%s",
                       species, drug, c["n"], c["n_nonS"], c["n_S"], len(c["levels"]),
                       "passes" if passes else "FAILS inclusion rule",
                       " (planned to fail)" if planned_fail and not passes else "")
            if not passes:
                failing.append(f"{species} x {drug}")
    droplog.drop("planted expected_drop: species x drug pairs failing the 50/50 inclusion rule",
                 len(failing), detail="; ".join(failing) if failing else None)
    return table


# --------------------------------------------------------------------------- #
# Other outputs
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, header: Sequence[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def _write_configs(src: Path, dst: Path, expected_sizes: dict[str, int]) -> None:
    """Copy the real configs and override ``expected_genome_size`` per species."""
    if not src.is_dir():
        raise FileNotFoundError(f"source configs dir not found: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    species_file = dst / "species.yaml"
    with species_file.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    for key, entry in content["species"].items():
        if key in expected_sizes:
            entry["expected_genome_size"] = int(expected_sizes[key])
            entry["reference_sketch"] = None
    with species_file.open("w", encoding="utf-8") as handle:
        handle.write("# SYNTHETIC copy of the project configs written by genome2mic.synthetic.generate.\n"
                     "# expected_genome_size is overridden to the synthetic genome size so QC size rules work.\n")
        yaml.safe_dump(content, handle, sort_keys=False)


def _write_markers_fasta(path: Path, seqs: dict[str, np.ndarray]) -> None:
    """Reportable marker sequences for the prediction pipeline's exact-match fallback."""
    records = []
    for spec in mk.reportable_markers():
        header = (f"{spec.name} type={spec.marker_type} subtype={spec.subtype} class={spec.amr_class} "
                  f"subclass={spec.subclass} scope={spec.scope} name=\"{spec.element_name}\" synthetic=true")
        records.append((header, seqs[spec.name].tobytes().decode("ascii")))
    write_fasta(records, path)


def _write_readme(path: Path, summary: dict[str, Any], table: list[dict[str, Any]]) -> None:
    by_name = mk.marker_by_name()
    effects_rows = []
    for drug in mk.DRUGS:
        parts = ", ".join(f"{m} {e:+g}" for m, e in mk.MARKER_EFFECTS[drug].items() if e)
        effects_rows.append(f"| {drug} | {mk.DRUG_BASE_LOG2[drug]:+g} ({math.ldexp(1.0, int(mk.DRUG_BASE_LOG2[drug])):g} mg/L) "
                            f"| {parts} |")
    counts_rows = [f"| {r['species']} | {r['drug']} | {r['n']} | {r['n_nonsusceptible']} | {r['n_susceptible']} | "
                   f"{r['n_distinct_mic']} | {'yes' if r['passes_inclusion'] else 'NO'} |" for r in table]
    reportable = ", ".join(f"`{m.name}`" for m in mk.reportable_markers())
    sizes = ", ".join(f"{k}: {v:,} bp" for k, v in summary["expected_genome_size"].items())
    text = f"""# SYNTHETIC DATA -- nothing in this directory is real

Every file under this root was **simulated** by `genome2mic.synthetic.generate`
(seed `{summary['seed']}`, `n_kpneu={summary['n_per_species'].get('KPNEU', 0)}`,
`n_ecoli={summary['n_per_species'].get('ECOLI', 0)}`, `genome_length={summary['genome_length']}`).
There are no real genomes, isolates, laboratories, patients or lab results here.
Any metric computed on this data describes the simulator, not bacteria. Never
present such metrics as real, and never use anything here for a clinical purpose.
Predictions built on it are not prescribing advice.

## What is simulated

- **Genomes** (`data/raw/genomes/<genome_id>.fasta`): per species one random ACGT
  core of {summary['genome_length']:,} bp; ~{N_LINEAGES} lineages per species, each with
  ~{LINEAGE_SNP_RATE:.1%} lineage SNPs and {ACCESSORY_BLOCKS[0]}-{ACCESSORY_BLOCKS[1]} accessory blocks;
  ~{PRIVATE_SNP_RATE:.2%} private SNPs per genome; {CONTIGS[0]}-{CONTIGS[1]} contigs. Markers are never split
  across contigs and are always inserted on the forward strand.
- **References** (`data/raw/references/<SPECIES>.fasta`): the species cores, for
  all five species. Accessions in `configs/species.yaml` are not real for this data.
- **Near-identical pairs** ({summary['n_near_identical_pairs']} pairs, <= 1 SNP apart, distinct biosamples)
  exercise the outbreak rule in lineage clustering.
- **Deliberate QC failures** (~{QC_FAIL_FRACTION:.0%}, one rule each): >500 contigs; size outside
  +/-20%; wrong species (built from the other species' core but labelled as this
  one; its `mash.tsv` names the other species); ~{DISTANT_SNP_RATE:.0%} divergence (`mash.tsv`
  distance > 0.05, `mlst.tsv` ST `-`).
- **Markers** (`src/genome2mic/synthetic/markers.py`): fixed random DNA blocks of
  600-1200 bp planted when present: {reportable}. Point mutations are independent
  blocks, not single-base changes. Presence probabilities depend on the lineage
  tier (high-risk lineages KPNEU ST258/ST11 and ECOLI ST131/ST410 carry more).
  Two blocks are hidden from every annotation file: `{mk.CRYPTIC_NAME}` (~15% of
  genomes, +3 ciprofloxacin steps -- only unitigs can find it) and the loss of the
  core `{mk.OMPK35_NAME}` block (+1 meropenem step).
- **MIC model**: `log2 MIC = base + sum(effects) + lineage N(0, {mk.LINEAGE_SD}) + N(0, {mk.NOISE_SD})`,
  rounded to the doubling grid, censored at the panel edges (`<=` lowest well, `>`
  highest well), S/I/R reported under the row's standard using the 2024 breakpoint
  tables in `configs/breakpoints/`. Every lab's S/I/R is derived from the 2024 tables,
  but only rows that *say* 2024 can be converted back: an S/I/R-only row needs the
  table for exactly its `standard_year`, so 2016-2023 and blank years are dropped.
  `ompK36_D135DGD` adds {mk.OMPK36_SYNERGY_EXTRA:+g} extra meropenem steps
  when a carbapenemase or ESBL is present. Gentamicin is a synthetic simplification
  (only `aac(6')-Ib-cr` drives it).

| drug | base log2 (MIC) | marker effects (log2 steps) |
| --- | --- | --- |
{chr(10).join(effects_rows)}

## Planned label counts (before the ingest stage's filters)

| species | drug | n | non-S | S | distinct MIC | passes 50/50/4 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(counts_rows)}

`ECOLI x gentamicin` is built to fail the inclusion rule on purpose.

## Realism planted in the raw AST files

- `ast_bvbrc.csv` (BV-BRC export columns) and `ast_ncbi.csv` (NCBI AST browser
  columns). ~{BVBRC_FRACTION:.0%} of genomes are BV-BRC, ~{1 - BVBRC_FRACTION:.0%} NCBI-only, and ~{OVERLAP_FRACTION:.0%} of
  biosamples appear in both files (consistent values, +/-1-step values, and some
  deliberate S-vs-R conflicts that the ingest stage must drop).
- Antibiotic synonyms, abbreviations, casing and trailing spaces; units `mg/L`,
  `µg/mL`, `ug/mL`, `mg/l` and blank; signs `=`, `==`, blank, `<=`, `<`, `>`, `>=`;
  `0.12`/`0.125` and `8.0` spellings; `evidence == Computational Prediction` rows
  (BV-BRC) that must be filtered; disk-diffusion S/I/R-only rows under EUCAST and
  CLSI; S/I/R-only rows with a blank or `missing` standard (must be dropped);
  BV-BRC `testing_standard_year` mostly `2024`, some 2016-2023 (no breakpoint table:
  S/I/R-only rows must be dropped) and some blank, while NCBI exports have no year
  column (NCBI S/I/R-only rows must be dropped: no table can be matched);
  piperacillin-tazobactam MICs written as `x/4` by some BV-BRC labs (ingest uses `x`);
  rows with a method but no value (S/I/R path); unknown methods (`Vitek 2`,
  `automated system`) and unknown drug names that must be dropped and logged.
- `data/processed/drop_log_synth.csv` records how many of each were planted so the
  ingest and QC drop logs can be cross-checked.
- `genome_metadata.csv` maps every genome to its biosample, species, primary
  source, isolation source, country and year. It carries no lineage or ST.

## Interim tool outputs (`data/interim/<genome_id>/`)

- `amrfinder.tsv`: AMRFinderPlus 4.x columns with the `Element symbol` /
  `Element name` header; ~{OLD_HEADER_FRACTION:.0%} of files use the older `Gene symbol` /
  `Sequence name` names. `iutA` rows have `Type == VIRULENCE` and must be ignored by
  the known-AMR parser. Hidden blocks never appear.
- `mash.tsv`: one line per species reference in `mash dist` format
  (reference, query, distance, p-value, shared-hashes), **no header**. The
  reference column is the species key. Own-species distances follow the k-mer model
  of the planted SNPs; other species are reported at ~0.3-0.5 as Mash would.
- `mlst.tsv`: `mlst` format (file, scheme, ST, allele columns), **no header**.
- `resfinder/pheno_table.txt`: ResFinder 4 phenotype table derived from the planted
  acquired genes only (point mutations and hidden blocks are not used), so it is a
  plausible but imperfect baseline. Antimicrobial spelling follows ResFinder
  (`piperacillin+tazobactam`).

## Configs

`configs/` is a copy of the project configs with `expected_genome_size` overridden
to the median synthetic genome length ({sizes}).

## Also written

- `models/markers.fasta`: the reportable marker sequences, for the prediction
  pipeline's exact-substring `MarkerScan` fallback when AMRFinderPlus is absent.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(
    paths: Paths,
    seed: int = 7,
    n_kpneu: int = 700,
    n_ecoli: int = 300,
    genome_length: int = 60_000,
    *,
    config: Config | None = None,
) -> dict[str, Any]:
    """Write the complete synthetic raw layer under ``paths.root``.

    Args:
        paths: ``paths.root`` receives ``configs/``, ``data/`` and ``models/``;
            ``paths.configs_dir`` is the *source* of the real configs (breakpoints are
            read from it). It must not be ``<root>/configs`` itself.
        seed: Seed for ``numpy.random.default_rng``; fixes every byte of the output.
        n_kpneu: Number of K. pneumoniae genomes.
        n_ecoli: Number of E. coli genomes.
        genome_length: Core length in bp per species (real genomes are 100x larger;
            60 kb keeps generation under ~90 s).
        config: Optional pre-loaded config (otherwise loaded from ``paths.configs_dir``).

    Returns:
        Summary dict: counts, genome ids, QC-fail plan, near-identical pairs, header
        variants, expected genome sizes, label count table, planted-drop counts.
    """
    started = time.perf_counter()
    if n_kpneu < 0 or n_ecoli < 0 or genome_length < 2000:
        raise ValueError("n_kpneu/n_ecoli must be >= 0 and genome_length >= 2000")
    root = Path(paths.root)
    src_configs = Path(paths.configs_dir)
    dst_configs = root / "configs"
    if src_configs.resolve() == dst_configs.resolve():
        raise ValueError(
            f"refusing to overwrite the source configs at {src_configs}; use a separate --root "
            "(e.g. runs/synthetic) so <root>/configs is distinct from --configs-dir"
        )
    if config is None:
        config = load_config(src_configs)
    logger.info("[synth] SYNTHETIC data generation: seed=%d n_kpneu=%d n_ecoli=%d genome_length=%d root=%s",
                seed, n_kpneu, n_ecoli, genome_length, root)

    rng = np.random.default_rng(seed)
    droplog = DropLog("synth")
    seqs = {name: np.frombuffer(s.encode("ascii"), dtype=np.uint8) for name, s in mk.marker_sequences(seed).items()}
    models = _build_species_models(rng, genome_length)

    for species in SPECIES_ORDER:
        header = f"{species}_synthetic_reference length={genome_length} synthetic=true species={SPECIES_NAMES[species]}"
        write_fasta([(header, models[species].core.tobytes().decode("ascii"))], paths.reference_fasta(species))

    sizes = {"KPNEU": n_kpneu, "ECOLI": n_ecoli}
    biosamples = _unique_biosamples(rng, n_kpneu + n_ecoli)
    plans: list[GenomePlan] = []
    offset = 0
    for species in LABELLED_SPECIES:
        n = sizes[species]
        plans.extend(_plan_species(rng, species, n, models, biosamples[offset : offset + n], 1000))
        offset += n
    for plan in plans:
        _draw_truth(rng, plan)

    originals_needed = {p.pair_of for p in plans if p.pair_of is not None}
    originals: dict[str, OriginalRecord] = {}
    for plan in plans:
        _write_genome(rng, paths, plan, models, seqs, originals, genome_length, plan.genome_id in originals_needed)

    bvbrc_rows, ncbi_rows, counts, planted = _build_label_rows(rng, plans, config, droplog)
    _write_csv(paths.raw_ast("bvbrc"), BVBRC_COLUMNS, bvbrc_rows)
    _write_csv(paths.raw_ast("ncbi"), NCBI_COLUMNS, ncbi_rows)
    _write_csv(paths.genome_metadata, METADATA_COLUMNS, [
        [p.genome_id, p.biosample, p.species, p.source, p.isolation_source, p.country, str(p.year)] for p in plans
    ])

    expected: dict[str, int] = {}
    for species in SPECIES_ORDER:
        lengths = [p.total_length for p in plans if p.species == species and p.qc_fail is None]
        expected[species] = int(np.median(lengths)) if lengths else int(genome_length)
    _write_configs(src_configs, dst_configs, expected)
    _write_markers_fasta(paths.models_dir / "markers.fasta", seqs)

    qc_fail = {p.genome_id: p.qc_fail for p in plans if p.qc_fail is not None}
    for kind in QC_FAIL_KINDS:
        droplog.drop(f"planted expected_drop: QC-fail genome ({kind})", sum(1 for v in qc_fail.values() if v == kind))
    table = _log_inclusion(counts, droplog)
    droplog.write(paths.drop_log("synth"))

    summary: dict[str, Any] = {
        "synthetic": True,
        "seed": seed,
        "genome_length": genome_length,
        "n_genomes": len(plans),
        "n_per_species": {sp: sum(1 for p in plans if p.species == sp) for sp in LABELLED_SPECIES},
        "n_per_source": {src: sum(1 for p in plans if p.source == src) for src in ("BVBRC", "NCBI")},
        "genome_ids": [p.genome_id for p in plans],
        "qc_fail": qc_fail,
        "near_identical_pairs": [(p.pair_of, p.genome_id) for p in plans if p.pair_of is not None],
        "n_near_identical_pairs": sum(1 for p in plans if p.pair_of is not None),
        "old_header_genomes": [p.genome_id for p in plans if p.old_header],
        "n_bvbrc_rows": len(bvbrc_rows),
        "n_ncbi_rows": len(ncbi_rows),
        "planted": planted,
        "expected_genome_size": expected,
        "label_counts": table,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    _write_readme(paths.raw_dir / "SYNTHETIC_DATA.md", summary, table)
    logger.info("[synth] wrote %d SYNTHETIC genomes (%d BV-BRC rows, %d NCBI rows, %d QC-fail, %d pairs) in %.1f s",
                len(plans), len(bvbrc_rows), len(ncbi_rows), len(qc_fail), summary["n_near_identical_pairs"],
                summary["elapsed_s"])
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for manual runs: ``python -m genome2mic.synthetic.generate --root runs/synthetic``."""
    parser = argparse.ArgumentParser(description="Generate the SYNTHETIC raw data layer (nothing real).")
    parser.add_argument("--root", default="runs/synthetic", help="output root (default runs/synthetic)")
    parser.add_argument("--configs-dir", default="configs", help="source configs dir (default configs)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-kpneu", type=int, default=700)
    parser.add_argument("--n-ecoli", type=int, default=300)
    parser.add_argument("--genome-length", type=int, default=60_000)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    summary = run(Paths(root=Path(args.root), configs_dir=Path(args.configs_dir)), seed=args.seed,
                  n_kpneu=args.n_kpneu, n_ecoli=args.n_ecoli, genome_length=args.genome_length)
    print(f"SYNTHETIC data written to {args.root}: {summary['n_genomes']} genomes, "
          f"{summary['n_bvbrc_rows']} BV-BRC rows, {summary['n_ncbi_rows']} NCBI rows, "
          f"{len(summary['qc_fail'])} planted QC failures, {summary['n_near_identical_pairs']} near-identical pairs, "
          f"{summary['elapsed_s']} s")
    for row in summary["label_counts"]:
        print(f"  {row['species']:<6} {row['drug']:<24} n={row['n']:<4} non-S={row['n_nonsusceptible']:<4} "
              f"S={row['n_susceptible']:<4} levels={row['n_distinct_mic']:<3} "
              f"{'passes' if row['passes_inclusion'] else 'FAILS inclusion rule'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
