"""Call logic, overrides, reasons and ranking for one isolate (``DATA_CONTRACT.md`` stage 12).

Everything here is a pure function of its arguments so it can be unit-tested
without a model or a genome. The pipeline (``predict/pipeline.py``) produces the
numbers; this module turns them into calls.

Call logic (contract)::

    band_high <= S breakpoint   -> likely_active   (margin_steps = log2(S) - log2(band_high))
    band_low  >  R breakpoint   -> likely_inactive
    otherwise                   -> uncertain       (wait for the lab)

The band's **upper** end is compared with the S breakpoint (CLAUDE.md): a drug is
only called active when even the pessimistic end of the band is susceptible.

Overrides, applied by the pipeline after the model, both force ``likely_inactive``:

1. ``natural_resistance`` -- the species is intrinsically resistant
   (``configs/natural_resistance.csv``); the model is skipped and the MIC fields are null.
2. ``strong_marker`` -- the model MIC fields are kept and the call is forced when
   either

   * a known-AMR column listed under ``strong_markers`` in ``drugs.yaml`` is present
     (matched as a column-name prefix so kept variants such as ``gene_blakpc_2`` match
     ``gene_blakpc``), or
   * a detected **acquired gene** (Subtype ``AMR``; never a ``POINT`` mutation such as
     a porin change) carries an AMRFinderPlus ``Subclass`` listed under the drug's
     ``strong_subclasses`` (``CARBAPENEM`` for the carbapenems). This catches
     carbapenemases whose family column is shared with non-carbapenemases
     (``blaOXA-23`` and ``blaOXA-1`` both map to ``gene_blaoxa``; ``blaGES-5`` and
     ``blaGES-1`` to ``gene_blages``) or that are not in any prefix list.

Ranking: likely-active drugs sorted by ``spectrum_tier`` ascending (narrowest first),
then ``margin_steps`` descending, then drug name.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from genome2mic.config import SPECTRUM_TIERS, Breakpoint, Config, DrugConfig
from genome2mic.models import conformal
from genome2mic.predict.amr_detect import SUBTYPE_AMR, SUBTYPE_POINT, TYPE_AMR, Marker

logger = logging.getLogger(__name__)

__all__ = [
    "CALL_LIKELY_ACTIVE",
    "CALL_LIKELY_INACTIVE",
    "CALL_UNCERTAIN",
    "DRUG_CLASSES",
    "NATURAL_RESISTANCE_REASON",
    "OVERRIDE_NATURAL_RESISTANCE",
    "OVERRIDE_STRONG_MARKER",
    "call_from_band",
    "class_tokens",
    "conformal_band",
    "display_name",
    "drug_classes",
    "is_relevant_class",
    "margin_steps",
    "rank_active",
    "reasons_for",
    "strong_marker_hits",
]

CALL_LIKELY_ACTIVE = "likely_active"
CALL_UNCERTAIN = "uncertain"
CALL_LIKELY_INACTIVE = "likely_inactive"

OVERRIDE_NATURAL_RESISTANCE = "natural_resistance"
OVERRIDE_STRONG_MARKER = "strong_marker"

NATURAL_RESISTANCE_REASON = "natural resistance"

# Tolerance on log2 arithmetic so grid-aligned values never lose a step to float drift.
_LOG2_TOL = 1e-9

# --------------------------------------------------------------------------- #
# Drug -> AMRFinderPlus ``Class`` mapping (for the ``reasons`` field)
# --------------------------------------------------------------------------- #

_BETA_LACTAM: tuple[str, ...] = ("BETA-LACTAM",)
_QUINOLONE: tuple[str, ...] = ("QUINOLONE",)
_AMINOGLYCOSIDE: tuple[str, ...] = ("AMINOGLYCOSIDE",)
_TETRACYCLINE: tuple[str, ...] = ("TETRACYCLINE",)

DRUG_CLASSES: dict[str, tuple[str, ...]] = {
    # Penicillins, beta-lactam / inhibitor combinations
    "ampicillin": _BETA_LACTAM,
    "amoxicillin-clavulanate": _BETA_LACTAM,
    "ampicillin-sulbactam": _BETA_LACTAM,
    "piperacillin-tazobactam": _BETA_LACTAM,
    "oxacillin": _BETA_LACTAM,
    # Cephalosporins and monobactam
    "cefazolin": _BETA_LACTAM,
    "cefoxitin": _BETA_LACTAM,
    "cefotaxime": _BETA_LACTAM,
    "ceftriaxone": _BETA_LACTAM,
    "ceftazidime": _BETA_LACTAM,
    "cefepime": _BETA_LACTAM,
    "ceftazidime-avibactam": _BETA_LACTAM,
    "aztreonam": _BETA_LACTAM,
    # Carbapenems
    "ertapenem": _BETA_LACTAM,
    "imipenem": _BETA_LACTAM,
    "meropenem": _BETA_LACTAM,
    # Fluoroquinolones
    "ciprofloxacin": _QUINOLONE,
    "levofloxacin": _QUINOLONE,
    # Aminoglycosides
    "gentamicin": _AMINOGLYCOSIDE,
    "tobramycin": _AMINOGLYCOSIDE,
    "amikacin": _AMINOGLYCOSIDE,
    # Folate pathway
    "trimethoprim-sulfamethoxazole": ("TRIMETHOPRIM", "SULFONAMIDE"),
    # Polymyxins
    "colistin": ("COLISTIN", "POLYMYXIN"),
    # Tetracyclines
    "tetracycline": _TETRACYCLINE,
    "doxycycline": _TETRACYCLINE,
    "minocycline": _TETRACYCLINE,
    "tigecycline": _TETRACYCLINE,
    # Gram-positive agents
    "vancomycin": ("GLYCOPEPTIDE",),
    "linezolid": ("OXAZOLIDINONE",),
    "daptomycin": ("LIPOPEPTIDE", "DAPTOMYCIN"),
    "clindamycin": ("LINCOSAMIDE",),
    "erythromycin": ("MACROLIDE",),
    "rifampicin": ("RIFAMYCIN",),
}
"""AMRFinderPlus ``Class`` tokens whose markers explain a call for each drug.

AMRFinderPlus reports multi-class determinants with ``/`` (``AMINOGLYCOSIDE/QUINOLONE``
for ``aac(6')-Ib-cr``); :func:`class_tokens` splits those so each token is matched on
its own. A drug missing from this table gets no model-based reasons (logged once).
"""

_WARNED_DRUGS: set[str] = set()


def drug_classes(drug: str) -> tuple[str, ...]:
    """AMRFinder class tokens relevant to ``drug`` (empty tuple if unmapped)."""
    classes = DRUG_CLASSES.get(drug)
    if classes is None:
        if drug not in _WARNED_DRUGS:
            _WARNED_DRUGS.add(drug)
            logger.warning("No AMRFinder class mapping for drug %r; reasons will be empty", drug)
        return ()
    return classes


def class_tokens(amr_class: str | None) -> frozenset[str]:
    """Split an AMRFinder ``Class`` string into upper-case tokens (``A/B`` -> ``{A, B}``)."""
    if amr_class is None:
        return frozenset()
    text = str(amr_class).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return frozenset()
    return frozenset(token.strip().upper() for token in text.split("/") if token.strip())


def is_relevant_class(amr_class: str | None, drug: str) -> bool:
    """True if any token of ``amr_class`` is in the drug's class list."""
    wanted = drug_classes(drug)
    if not wanted:
        return False
    return not class_tokens(amr_class).isdisjoint(wanted)


# --------------------------------------------------------------------------- #
# Band and call
# --------------------------------------------------------------------------- #


def conformal_band(pred_mic: float, q: float) -> tuple[float, float]:
    """Split-conformal band ``(pred / 2**q, pred * 2**q)`` snapped to the doubling grid.

    Scalar front for :func:`genome2mic.models.conformal.band` (the same function
    training uses to write ``band_low`` / ``band_high``): the low end is rounded
    **down** and the high end **up**, so the band is only ever widened. ``q`` is in
    doubling steps (``conformal.json``) and must be finite and >= 0; an infinite
    ``q`` would give the uninformative ``(0, inf)`` band, which a report cannot carry.
    """
    if not math.isfinite(q) or q < 0:
        raise ValueError(f"q must be a finite non-negative number of steps, got {q!r}")
    if not math.isfinite(pred_mic) or pred_mic <= 0:
        raise ValueError(f"pred_mic must be a finite positive MIC, got {pred_mic!r}")
    low, high = conformal.band(np.asarray([pred_mic], dtype=np.float64), q)
    return float(low[0]), float(high[0])


def margin_steps(band_high: float, s_breakpoint: float) -> int:
    """Doubling steps between the band's upper end and the S breakpoint, as an int.

    ``log2(s_breakpoint) - log2(band_high)``, floored so an off-grid breakpoint
    (EUCAST's ``0.001`` placeholder) never inflates the margin. Requires
    ``band_high <= s_breakpoint``.
    """
    if band_high <= 0 or s_breakpoint <= 0:
        raise ValueError("band_high and s_breakpoint must be positive")
    steps = math.log2(s_breakpoint) - math.log2(band_high)
    if steps < -_LOG2_TOL:
        raise ValueError(f"band_high {band_high} is above the S breakpoint {s_breakpoint}")
    return int(math.floor(steps + _LOG2_TOL))


def call_from_band(
    band_low: float,
    band_high: float,
    bp: Breakpoint | None,
) -> tuple[str, int | None]:
    """Apply the contract's call table to a band. Returns ``(call, margin_steps)``.

    ``margin_steps`` is only set for ``likely_active``. Without a breakpoint the
    call is ``uncertain`` (there is nothing to compare against).
    """
    if bp is None:
        return CALL_UNCERTAIN, None
    if band_high <= bp.s_breakpoint:
        return CALL_LIKELY_ACTIVE, margin_steps(band_high, bp.s_breakpoint)
    if band_low > bp.r_breakpoint:
        return CALL_LIKELY_INACTIVE, None
    return CALL_UNCERTAIN, None


# --------------------------------------------------------------------------- #
# Overrides and reasons
# --------------------------------------------------------------------------- #


def strong_marker_hits(
    symbols_by_column: Mapping[str, Sequence[str]],
    drug_cfg: DrugConfig | None,
    markers: Iterable[Marker] = (),
) -> list[str]:
    """Symbols of the detected markers that trigger the strong-marker override (override 2).

    A symbol hits when its known-AMR column matches a ``strong_markers`` prefix, or
    when it is an acquired gene (Subtype ``AMR``, never ``POINT``) whose AMRFinderPlus
    ``Subclass`` (``/``-separated tokens) is in ``strong_subclasses``. The subclass
    comes from the detection itself, never from the training-time column metadata: a
    family column such as ``gene_blaoxa`` mixes carbapenemases and narrow-spectrum
    enzymes, so only the detected variant's own subclass can tell them apart.

    Args:
        symbols_by_column: present ``gene_`` / ``point_`` column -> detected symbols.
        drug_cfg: the drug's config (``strong_markers`` are column-name prefixes).
        markers: the detections (:attr:`KnownAmrRow.markers`); needed for the subclass rule.

    Returns:
        Marker symbols, de-duplicated, in detection order when ``markers`` is given.
        Empty when nothing matches.
    """
    if drug_cfg is None or not (drug_cfg.strong_markers or drug_cfg.strong_subclasses):
        return []
    markers = tuple(markers)
    hits: list[str] = []
    for column, symbols in symbols_by_column.items():
        if any(column == prefix or column.startswith(prefix) for prefix in drug_cfg.strong_markers):
            for symbol in symbols:
                if symbol not in hits:
                    hits.append(symbol)
    wanted = frozenset(drug_cfg.strong_subclasses)
    if wanted:
        for marker in markers:
            acquired_gene = marker.subtype == SUBTYPE_AMR and marker.element_type == TYPE_AMR
            if not acquired_gene or marker.symbol in hits:
                continue
            if not class_tokens(marker.subclass).isdisjoint(wanted):
                hits.append(marker.symbol)
    if markers:
        first_seen: dict[str, int] = {}
        for index, marker in enumerate(markers):
            first_seen.setdefault(marker.symbol, index)
        hits.sort(key=lambda symbol: first_seen.get(symbol, len(first_seen)))
    return hits


def display_name(marker: Marker) -> str:
    """Human-readable marker name: ``gyrA_S83L`` -> ``gyrA S83L``; gene symbols unchanged."""
    if marker.subtype == SUBTYPE_POINT:
        return marker.symbol.replace("_", " ")
    return marker.symbol


def reasons_for(
    drug: str,
    markers: Iterable[Marker],
    class_by_column: Mapping[str, Sequence[str | None]] | None = None,
) -> list[str]:
    """Present markers whose AMRFinder class is relevant to ``drug``.

    The class comes from the detection itself; when a detector did not report one
    (a bare ``markers.fasta`` header), ``class_by_column`` from ``features.json``
    (``column -> [CLASS, SUBCLASS]``) is consulted through ``marker.column``.
    """
    out: list[str] = []
    for marker in markers:
        amr_class = marker.amr_class
        if not class_tokens(amr_class) and class_by_column and marker.column in class_by_column:
            amr_class = class_by_column[marker.column][0]
        if is_relevant_class(amr_class, drug):
            name = display_name(marker)
            if name not in out:
                out.append(name)
    return out


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def rank_active(predictions: Sequence[Mapping[str, Any]], config: Config) -> list[str]:
    """Names of ``likely_active`` drugs, best first.

    Sort key: ``spectrum_tier`` ascending (narrowest spectrum first), then
    ``margin_steps`` descending (more room below the S breakpoint first), then
    the drug name. A drug missing from ``drugs.yaml`` is logged and sorted last.
    """
    fallback_tier = max(SPECTRUM_TIERS) + 1
    active = [p for p in predictions if p.get("call") == CALL_LIKELY_ACTIVE]

    def key(prediction: Mapping[str, Any]) -> tuple[int, float, str]:
        drug = str(prediction["drug"])
        drug_cfg = config.drugs.get(drug)
        if drug_cfg is None:
            logger.warning("Drug %r is not in drugs.yaml; ranked last", drug)
            tier = fallback_tier
        else:
            tier = drug_cfg.spectrum_tier
        margin = prediction.get("margin_steps")
        margin_value = float(margin) if margin is not None else -math.inf
        return (tier, -margin_value, drug)

    ranked = [str(p["drug"]) for p in sorted(active, key=key)]
    logger.info("Ranked %d likely-active drug(s): %s", len(ranked), ranked)
    return ranked
