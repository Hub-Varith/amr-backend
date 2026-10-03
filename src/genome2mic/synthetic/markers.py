"""Planted resistance markers for the synthetic data generator.

A *marker* is a fixed DNA block that the generator inserts into a synthetic genome
when the genome "carries" it. Each marker has:

- an AMRFinderPlus-style annotation (element symbol, class, subclass, subtype) used
  to write ``data/interim/<gid>/amrfinder.tsv``;
- lineage-tier presence probabilities per species (``high`` / ``medium`` / ``low``
  risk lineages), so resistance and lineage are correlated as they are in nature;
- additive log2 MIC effects per drug (:data:`MARKER_EFFECTS`).

Two blocks are deliberately **hidden** from the fake AMRFinderPlus output:

- :data:`CRYPTIC_NAME` (``cryptic_cip_block``): an unknown mechanism that raises the
  ciprofloxacin MIC. Only the unitig model can learn it; known-AMR-only models
  cannot. It must never appear in ``amrfinder.tsv`` or ``pheno_table.txt``.
- :data:`OMPK35_NAME` (``ompK35``): a KPNEU porin present in the core. Its *loss* is
  the resistance signal (pseudo-marker :data:`OMPK35_LOSS`), and a loss has nothing
  to report, so it is partly hidden in the same way.

Sequences are random ACGT strings drawn with ``numpy.random.default_rng`` from the
run seed (:func:`marker_sequences`), so a seed fixes every sequence bit for bit.
Point mutations (``gyrA_S83L`` ...) are modelled as independent blocks rather than
as single-base changes inside a core gene: this keeps "marker listed in
amrfinder.tsv" equivalent to "marker sequence literally present in the FASTA" even
for double mutants, which is what the exact-substring ``MarkerScan`` fallback in the
prediction pipeline relies on. Accessions are plausible-looking but synthetic.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CARBAPENEMASES",
    "CONDITIONAL_ON",
    "CRYPTIC_NAME",
    "DRUGS",
    "DRUG_BASE_LOG2",
    "ESBLS",
    "EXCLUSIVE_WITH",
    "LINEAGE_SD",
    "MARKERS",
    "MARKER_EFFECTS",
    "MarkerSpec",
    "NOISE_SD",
    "OMPK35_LOSS",
    "OMPK35_NAME",
    "OMPK36_SYNERGY_EXTRA",
    "PANELS",
    "PRESENCE",
    "RESFINDER_ANTIMICROBIALS",
    "RESFINDER_CALLS",
    "TIERS",
    "effect_sum",
    "marker_by_name",
    "marker_sequences",
    "markers_for_species",
    "present_set",
    "random_dna",
    "reportable_markers",
]

TIERS: tuple[str, ...] = ("high", "medium", "low")
"""Lineage risk tiers; index into the presence-probability triples."""

CRYPTIC_NAME = "cryptic_cip_block"
"""Hidden ciprofloxacin block. Never reported by the fake AMRFinderPlus/ResFinder."""

OMPK35_NAME = "ompK35"
"""Core KPNEU porin block; its absence is the signal."""

OMPK35_LOSS = "ompK35_loss"
"""Pseudo-marker added to a KPNEU genome's present set when :data:`OMPK35_NAME` is absent."""

_MARKER_SALT = 0x5EED_A5A5


@dataclass(frozen=True)
class MarkerSpec:
    """One planted block and how the fake tools annotate it.

    Attributes:
        name: AMRFinderPlus element symbol (or the internal name for hidden blocks).
        element_name: AMRFinderPlus ``Element name``.
        marker_type: ``AMR`` or ``VIRULENCE`` (AMRFinderPlus ``Type``).
        subtype: ``AMR`` (acquired gene), ``POINT`` (mutation) or ``VIRULENCE``.
        amr_class: AMRFinderPlus ``Class`` (``NA`` for virulence rows).
        subclass: AMRFinderPlus ``Subclass``.
        scope: AMRFinderPlus ``Scope`` (``core`` or ``plus``).
        accession: Synthetic ``Closest reference accession``.
        reportable: Whether the block appears in ``amrfinder.tsv``.
        locus: ``fixed`` (same chromosomal offset in every genome of the species) or
            ``mobile`` (random offset per genome, like a plasmid gene).
        species: Species keys that can carry the block.
        resfinder_class: ResFinder ``Class`` column, or ``None`` when ResFinder
            would not list it (point mutations and hidden blocks).
    """

    name: str
    element_name: str
    marker_type: str
    subtype: str
    amr_class: str
    subclass: str
    scope: str
    accession: str
    reportable: bool
    locus: str
    species: tuple[str, ...]
    resfinder_class: str | None = None


_ENTERO = ("KPNEU", "ECOLI")
_KP = ("KPNEU",)

MARKERS: tuple[MarkerSpec, ...] = (
    MarkerSpec("blaKPC-2", "carbapenem-hydrolyzing class A beta-lactamase KPC-2", "AMR", "AMR",
               "BETA-LACTAM", "CARBAPENEM", "core", "WP_004199234.1", True, "mobile", _ENTERO, "beta-lactam"),
    MarkerSpec("blaKPC-3", "carbapenem-hydrolyzing class A beta-lactamase KPC-3", "AMR", "AMR",
               "BETA-LACTAM", "CARBAPENEM", "core", "WP_032495166.1", True, "mobile", _ENTERO, "beta-lactam"),
    MarkerSpec("blaNDM-1", "subclass B1 metallo-beta-lactamase NDM-1", "AMR", "AMR",
               "BETA-LACTAM", "CARBAPENEM", "core", "WP_004201164.1", True, "mobile", _ENTERO, "beta-lactam"),
    MarkerSpec("blaOXA-48", "OXA-48 family carbapenem-hydrolyzing class D beta-lactamase OXA-48", "AMR", "AMR",
               "BETA-LACTAM", "CARBAPENEM", "core", "WP_012322961.1", True, "mobile", _ENTERO, "beta-lactam"),
    MarkerSpec("blaCTX-M-15", "extended-spectrum class A beta-lactamase CTX-M-15", "AMR", "AMR",
               "BETA-LACTAM", "CEPHALOSPORIN", "core", "WP_000239590.1", True, "mobile", _ENTERO, "beta-lactam"),
    MarkerSpec("blaCTX-M-27", "extended-spectrum class A beta-lactamase CTX-M-27", "AMR", "AMR",
               "BETA-LACTAM", "CEPHALOSPORIN", "core", "WP_000239576.1", True, "mobile", _ENTERO, "beta-lactam"),
    MarkerSpec("blaSHV-11", "class A broad-spectrum beta-lactamase SHV-11", "AMR", "AMR",
               "BETA-LACTAM", "BETA-LACTAM", "core", "WP_004176269.1", True, "fixed", _KP, "beta-lactam"),
    MarkerSpec("blaTEM-1", "class A broad-spectrum beta-lactamase TEM-1", "AMR", "AMR",
               "BETA-LACTAM", "BETA-LACTAM", "core", "WP_000027057.1", True, "mobile", _ENTERO, "beta-lactam"),
    MarkerSpec("qnrB1", "quinolone resistance pentapeptide repeat protein QnrB1", "AMR", "AMR",
               "QUINOLONE", "QUINOLONE", "core", "WP_012954695.1", True, "mobile", _ENTERO, "quinolone"),
    MarkerSpec("aac(6')-Ib-cr", "fluoroquinolone-acetylating aminoglycoside 6'-N-acetyltransferase AAC(6')-Ib-cr",
               "AMR", "AMR", "AMINOGLYCOSIDE", "QUINOLONE", "core", "WP_001067855.1", True, "mobile", _ENTERO,
               "aminoglycoside"),
    MarkerSpec("gyrA_S83L", "quinolone-resistant DNA gyrase subunit GyrA", "AMR", "POINT",
               "QUINOLONE", "QUINOLONE", "core", "WP_000130940.1", True, "fixed", _ENTERO),
    MarkerSpec("gyrA_D87N", "quinolone-resistant DNA gyrase subunit GyrA", "AMR", "POINT",
               "QUINOLONE", "QUINOLONE", "core", "WP_000130940.1", True, "fixed", _ENTERO),
    MarkerSpec("parC_S80I", "quinolone-resistant DNA topoisomerase IV subunit ParC", "AMR", "POINT",
               "QUINOLONE", "QUINOLONE", "core", "WP_000195295.1", True, "fixed", _ENTERO),
    MarkerSpec("ompK36_D135DGD", "Klebsiella pneumoniae carbapenem-resistant porin OmpK36", "AMR", "POINT",
               "BETA-LACTAM", "BETA-LACTAM", "core", "WP_002920224.1", True, "fixed", _KP),
    MarkerSpec("iutA", "aerobactin siderophore receptor IutA", "VIRULENCE", "VIRULENCE",
               "NA", "NA", "plus", "WP_004151047.1", True, "mobile", _KP),
    # Hidden blocks: present in the FASTA, never in amrfinder.tsv / pheno_table.txt.
    MarkerSpec(OMPK35_NAME, "outer membrane porin OmpK35 (core; loss is the signal)", "NA", "NA",
               "NA", "NA", "NA", "NA", False, "fixed", _KP),
    MarkerSpec(CRYPTIC_NAME, "unknown ciprofloxacin mechanism (hidden from every annotation tool)", "NA", "NA",
               "NA", "NA", "NA", "NA", False, "mobile", _ENTERO),
)
"""Every planted block, in generation order (never iterate a set of names instead)."""

CARBAPENEMASES: tuple[str, ...] = ("blaKPC-2", "blaKPC-3", "blaNDM-1", "blaOXA-48")
ESBLS: tuple[str, ...] = ("blaCTX-M-15", "blaCTX-M-27")

# --------------------------------------------------------------------------- #
# Presence probabilities per species and lineage tier (high, medium, low)
# --------------------------------------------------------------------------- #

PRESENCE: dict[str, dict[str, tuple[float, float, float]]] = {
    "KPNEU": {
        "blaSHV-11": (1.0, 1.0, 1.0),
        "blaKPC-2": (0.70, 0.10, 0.02),
        "blaKPC-3": (0.12, 0.03, 0.01),
        "blaNDM-1": (0.10, 0.12, 0.03),
        "blaOXA-48": (0.08, 0.10, 0.03),
        "blaCTX-M-15": (0.60, 0.35, 0.10),
        "blaCTX-M-27": (0.05, 0.10, 0.05),
        "blaTEM-1": (0.50, 0.40, 0.25),
        "qnrB1": (0.30, 0.20, 0.08),
        "aac(6')-Ib-cr": (0.50, 0.25, 0.08),
        "gyrA_S83L": (0.85, 0.35, 0.10),
        "gyrA_D87N": (0.0, 0.0, 0.0),  # conditional, see CONDITIONAL_ON
        "parC_S80I": (0.0, 0.0, 0.0),  # conditional
        "ompK36_D135DGD": (0.50, 0.15, 0.05),
        "iutA": (0.35, 0.10, 0.05),
        OMPK35_NAME: (0.40, 0.80, 0.95),  # presence; loss = 1 - presence
        CRYPTIC_NAME: (0.25, 0.15, 0.12),
    },
    "ECOLI": {
        "blaKPC-2": (0.15, 0.05, 0.02),
        "blaKPC-3": (0.03, 0.01, 0.005),
        "blaNDM-1": (0.35, 0.15, 0.05),
        "blaOXA-48": (0.25, 0.10, 0.04),
        "blaCTX-M-15": (0.60, 0.30, 0.12),
        "blaCTX-M-27": (0.15, 0.10, 0.05),
        "blaTEM-1": (0.55, 0.45, 0.35),
        "qnrB1": (0.15, 0.10, 0.05),
        "aac(6')-Ib-cr": (0.45, 0.15, 0.05),
        "gyrA_S83L": (0.90, 0.30, 0.10),
        "gyrA_D87N": (0.0, 0.0, 0.0),
        "parC_S80I": (0.0, 0.0, 0.0),
        CRYPTIC_NAME: (0.22, 0.15, 0.12),
    },
}
"""``PRESENCE[species][marker] = (p_high, p_medium, p_low)``; markers absent from a
species' table are never planted in that species."""

CONDITIONAL_ON: dict[str, tuple[str, float, float]] = {
    "gyrA_D87N": ("gyrA_S83L", 0.65, 0.02),
    "parC_S80I": ("gyrA_S83L", 0.75, 0.03),
}
"""``marker -> (parent, p_if_parent_present, p_if_parent_absent)`` (replaces PRESENCE)."""

EXCLUSIVE_WITH: dict[str, str] = {"blaKPC-3": "blaKPC-2"}
"""``marker -> other``: the marker is skipped when ``other`` is already present."""

# --------------------------------------------------------------------------- #
# MIC model
# --------------------------------------------------------------------------- #

DRUGS: tuple[str, ...] = (
    "meropenem",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "piperacillin-tazobactam",
)
"""Drugs with synthetic lab results, normalized names."""

DRUG_BASE_LOG2: dict[str, float] = {
    "meropenem": -5.0,  # 0.03 mg/L
    "ceftriaxone": -4.0,  # 0.06
    "ciprofloxacin": -5.0,  # 0.03
    "gentamicin": -1.0,  # 0.5
    "piperacillin-tazobactam": 0.0,  # 1
}
"""Wild-type log2 MIC per drug."""

MARKER_EFFECTS: dict[str, dict[str, float]] = {
    "meropenem": {
        "blaKPC-2": 8, "blaKPC-3": 8, "blaNDM-1": 9, "blaOXA-48": 6,
        "ompK36_D135DGD": 1, OMPK35_LOSS: 1,
    },
    "ceftriaxone": {
        "blaCTX-M-15": 9, "blaCTX-M-27": 9, "blaSHV-11": 0, "blaKPC-2": 7, "blaKPC-3": 7,
        "blaNDM-1": 8, "blaOXA-48": 2, "blaTEM-1": 0, "ompK36_D135DGD": 1,
    },
    "ciprofloxacin": {
        "gyrA_S83L": 3, "gyrA_D87N": 2, "parC_S80I": 2, "qnrB1": 2, "aac(6')-Ib-cr": 1,
        CRYPTIC_NAME: 3,
    },
    "gentamicin": {
        # Synthetic simplification: the only planted aminoglycoside gene drives gentamicin.
        "aac(6')-Ib-cr": 4,
    },
    "piperacillin-tazobactam": {
        "blaKPC-2": 6, "blaKPC-3": 6, "blaNDM-1": 7, "blaOXA-48": 5, "blaCTX-M-15": 2,
        "blaCTX-M-27": 2, "blaTEM-1": 1, "blaSHV-11": 0, "ompK36_D135DGD": 1,
    },
}
"""Additive log2 steps per drug per present marker (pseudo-marker :data:`OMPK35_LOSS` included)."""

OMPK36_SYNERGY_EXTRA: float = 2.0
"""Extra meropenem steps for ``ompK36_D135DGD`` when a carbapenemase or ESBL is also present
(total +3 instead of +1)."""

NOISE_SD: float = 0.6
"""Per-row measurement noise on log2 MIC."""

LINEAGE_SD: float = 0.3
"""Per-lineage random effect on log2 MIC, drawn once per lineage x drug."""

PANELS: dict[str, tuple[int, int]] = {
    "meropenem": (-5, 5),  # 0.03 .. 32
    "ceftriaxone": (-4, 5),  # 0.06 .. 32
    "ciprofloxacin": (-6, 3),  # 0.015 .. 8
    "gentamicin": (-2, 4),  # 0.25 .. 16
    "piperacillin-tazobactam": (-1, 7),  # 0.5 .. 128
}
"""``drug -> (lowest, highest)`` log2 well on the dilution panel. Results at or below the
lowest well are reported ``<= low``; above the highest, ``> high``."""

# --------------------------------------------------------------------------- #
# ResFinder baseline
# --------------------------------------------------------------------------- #

RESFINDER_ANTIMICROBIALS: tuple[tuple[str, str], ...] = (
    ("ampicillin", "beta-lactam"),
    ("piperacillin+tazobactam", "beta-lactam"),
    ("cefotaxime", "beta-lactam"),
    ("ceftriaxone", "beta-lactam"),
    ("ceftazidime", "beta-lactam"),
    ("ertapenem", "beta-lactam"),
    ("imipenem", "beta-lactam"),
    ("meropenem", "beta-lactam"),
    ("ciprofloxacin", "quinolone"),
    ("gentamicin", "aminoglycoside"),
    ("tobramycin", "aminoglycoside"),
    ("amikacin", "aminoglycoside"),
)
"""Rows of the fake ``pheno_table.txt`` (ResFinder spelling, e.g. ``piperacillin+tazobactam``)."""

RESFINDER_CALLS: dict[str, tuple[str, ...]] = {
    "blaKPC-2": ("ampicillin", "piperacillin+tazobactam", "cefotaxime", "ceftriaxone", "ceftazidime",
                 "ertapenem", "imipenem", "meropenem"),
    "blaKPC-3": ("ampicillin", "piperacillin+tazobactam", "cefotaxime", "ceftriaxone", "ceftazidime",
                 "ertapenem", "imipenem", "meropenem"),
    "blaNDM-1": ("ampicillin", "piperacillin+tazobactam", "cefotaxime", "ceftriaxone", "ceftazidime",
                 "ertapenem", "imipenem", "meropenem"),
    "blaOXA-48": ("ampicillin", "piperacillin+tazobactam", "ertapenem", "imipenem", "meropenem"),
    "blaCTX-M-15": ("ampicillin", "cefotaxime", "ceftriaxone", "ceftazidime"),
    "blaCTX-M-27": ("ampicillin", "cefotaxime", "ceftriaxone", "ceftazidime"),
    "blaSHV-11": ("ampicillin",),
    "blaTEM-1": ("ampicillin",),
    "qnrB1": ("ciprofloxacin",),
    # Real ResFinder maps aac(6')-Ib-cr to tobramycin/amikacin, not gentamicin, which is
    # exactly why this baseline is plausible but imperfect against the synthetic truth.
    "aac(6')-Ib-cr": ("ciprofloxacin", "tobramycin", "amikacin"),
}
"""Acquired gene -> antimicrobials the fake ResFinder calls ``Resistant`` (acquired genes only)."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def marker_by_name() -> dict[str, MarkerSpec]:
    """``name -> MarkerSpec`` for every planted block (insertion order preserved)."""
    return {m.name: m for m in MARKERS}


def reportable_markers() -> tuple[MarkerSpec, ...]:
    """Markers that the fake AMRFinderPlus lists (hidden blocks excluded)."""
    return tuple(m for m in MARKERS if m.reportable)


def markers_for_species(species: str) -> tuple[MarkerSpec, ...]:
    """Markers that can be planted in ``species`` (those with a PRESENCE entry)."""
    table = PRESENCE.get(species, {})
    return tuple(m for m in MARKERS if species in m.species and m.name in table)


def random_dna(rng: np.random.Generator, length: int) -> str:
    """Uniform random ACGT string of ``length`` bases from ``rng``."""
    if length < 0:
        raise ValueError("length must be >= 0")
    return rng.integers(0, 4, size=int(length)).astype(np.uint8).tobytes().translate(_CODE_TO_BASE).decode("ascii")


_CODE_TO_BASE = bytes.maketrans(bytes([0, 1, 2, 3]), b"ACGT")


def marker_sequences(seed: int = 0) -> dict[str, str]:
    """Deterministic random sequence for every marker in :data:`MARKERS`.

    Reportable markers are 600-1200 bp (multiples of 3, like a coding sequence);
    the cryptic block is ~900 bp. The same ``seed`` always yields the same
    sequences; different seeds yield different ones.
    """
    rng = np.random.default_rng([int(seed), _MARKER_SALT])
    out: dict[str, str] = {}
    for spec in MARKERS:
        if spec.name == CRYPTIC_NAME:
            length = int(rng.integers(870, 931))
        else:
            length = int(rng.integers(600, 1201))
        length -= length % 3
        out[spec.name] = random_dna(rng, length)
    return out


def effect_sum(present: Collection[str], drug: str) -> float:
    """Sum of marker effects on log2 MIC for ``drug`` given the present marker names.

    ``present`` may include the pseudo-marker :data:`OMPK35_LOSS`. The ompK36
    mutation gains :data:`OMPK36_SYNERGY_EXTRA` meropenem steps when a
    carbapenemase or ESBL is also present. Unknown drugs raise ``ValueError``.
    """
    if drug not in MARKER_EFFECTS:
        raise ValueError(f"no synthetic MIC model for drug {drug!r}")
    effects = MARKER_EFFECTS[drug]
    names = list(present)
    total = float(sum(effects.get(name, 0.0) for name in names))
    if drug == "meropenem" and "ompK36_D135DGD" in names:
        if any(n in names for n in CARBAPENEMASES) or any(n in names for n in ESBLS):
            total += OMPK36_SYNERGY_EXTRA
    return total


def present_set(markers: Iterable[str], species: str) -> list[str]:
    """Marker names plus :data:`OMPK35_LOSS` when a KPNEU genome lacks :data:`OMPK35_NAME`."""
    names = list(markers)
    if species == "KPNEU" and OMPK35_NAME not in names:
        names.append(OMPK35_LOSS)
    return names
