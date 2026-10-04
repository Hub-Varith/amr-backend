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

   Intrinsic chromosomal genes of the species (``configs/intrinsic_markers.csv``,
   e.g. the OXA-51-like genes every *A. baumannii* carries, which AMRFinderPlus
   reports with Subclass ``CARBAPENEM``) never trigger either rule. Training-time
   call metrics (:func:`strong_marker_mask`) apply the same exclusion to the
   per-allele feature columns of those genes.

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
from genome2mic.predict.amr_detect import PREFIX_GENE, SUBTYPE_AMR, SUBTYPE_POINT, TYPE_AMR, Marker, column_name

logger = logging.getLogger(__name__)

__all__ = [
    "CALL_LIKELY_ACTIVE",
    "CALL_LIKELY_INACTIVE",
    "CALL_UNCERTAIN",
    "ACTIVE_GATE_REASON",
    "DRUG_CLASSES",
    "NATURAL_RESISTANCE_REASON",
    "OVERRIDE_NATURAL_RESISTANCE",
    "OVERRIDE_STRONG_MARKER",
    "call_array",
    "call_from_band",
    "class_tokens",
    "conformal_band",
    "conformal_band_asym",
    "display_name",
    "drug_classes",
    "intrinsic_columns",
    "is_relevant_class",
    "margin_steps",
    "rank_active",
    "reasons_for",
    "strong_marker_columns",
    "strong_marker_hits",
    "strong_marker_mask",
    "subclass_is_strong",
]

CALL_LIKELY_ACTIVE = "likely_active"
CALL_UNCERTAIN = "uncertain"
CALL_LIKELY_INACTIVE = "likely_inactive"

OVERRIDE_NATURAL_RESISTANCE = "natural_resistance"
OVERRIDE_STRONG_MARKER = "strong_marker"

NATURAL_RESISTANCE_REASON = "natural resistance"

ACTIVE_GATE_REASON = (
    "likely_active withheld: in cross-validation the model could not show that resistant isolates of this "
    "species x drug are called likely_active in <= 1.5% of cases"
)
"""Reason attached when a bundle's ``active_gate_open`` is False and the band alone would say active."""

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
    "amoxicillin-clavulanic-acid": _BETA_LACTAM,
    "ampicillin-sulbactam": _BETA_LACTAM,
    "piperacillin-tazobactam": _BETA_LACTAM,
    "oxacillin": _BETA_LACTAM,
    "penicillin": _BETA_LACTAM,
    # Cephalosporins and monobactam
    "cefazolin": _BETA_LACTAM,
    "cefoxitin": _BETA_LACTAM,
    "cefotaxime": _BETA_LACTAM,
    "ceftriaxone": _BETA_LACTAM,
    "ceftazidime": _BETA_LACTAM,
    "cefepime": _BETA_LACTAM,
    "ceftazidime-avibactam": _BETA_LACTAM,
    "ceftolozane-tazobactam": _BETA_LACTAM,
    "cefuroxime": _BETA_LACTAM,
    "ceftaroline": _BETA_LACTAM,
    "aztreonam": _BETA_LACTAM,
    # Carbapenems
    "ertapenem": _BETA_LACTAM,
    "imipenem": _BETA_LACTAM,
    "meropenem": _BETA_LACTAM,
    "doripenem": _BETA_LACTAM,
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
    "polymyxin-b": ("COLISTIN", "POLYMYXIN"),
    # Phenicols
    "chloramphenicol": ("PHENICOL",),
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


def conformal_band_asym(pred_mic: float, q_up: float, q_low: float) -> tuple[float, float]:
    """Asymmetric band ``(pred / 2**q_low, pred * 2**q_up)`` snapped outward to the grid.

    Scalar front for :func:`genome2mic.models.conformal.asym_band`, the function training
    uses for the tuned bands (``conformal.json`` ``q_up`` / ``q_low``). Both half-widths
    must be finite and >= 0.
    """
    for name, q in (("q_up", q_up), ("q_low", q_low)):
        if not math.isfinite(q) or q < 0:
            raise ValueError(f"{name} must be a finite non-negative number of steps, got {q!r}")
    if not math.isfinite(pred_mic) or pred_mic <= 0:
        raise ValueError(f"pred_mic must be a finite positive MIC, got {pred_mic!r}")
    low, high = conformal.asym_band(np.asarray([pred_mic], dtype=np.float64), q_up, q_low)
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


def call_array(
    band_low: Sequence[float] | np.ndarray,
    band_high: Sequence[float] | np.ndarray,
    bp: Breakpoint | None,
    *,
    natural_resistance: bool = False,
    strong_marker: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized call for evaluation tables (object array of call strings or ``None``).

    Same rule and overrides as the prediction pipeline (:func:`call_from_band`, then
    override 1 natural resistance and override 2 strong marker, both
    ``likely_inactive``). Differences, all of them conservative for scoring:

    * a row without a band (``NaN``) and without an override gets ``None``;
    * without a call breakpoint a non-overridden row gets ``None`` (the pipeline says
      ``uncertain``; with no breakpoint there is nothing to score against).

    ``strong_marker`` is a per-row boolean (see :func:`strong_marker_mask`).
    """
    low = np.asarray(band_low, dtype=np.float64).ravel()
    high = np.asarray(band_high, dtype=np.float64).ravel()
    if low.shape != high.shape:
        raise ValueError("band_low and band_high must have the same length")
    n = low.size
    out = np.full(n, None, dtype=object)
    if bp is not None:
        has = ~np.isnan(low) & ~np.isnan(high)
        active = has & (high <= float(bp.s_breakpoint) * (1 + _LOG2_TOL))
        inactive = has & ~active & (low > float(bp.r_breakpoint) * (1 + _LOG2_TOL))
        out[has] = CALL_UNCERTAIN
        out[active] = CALL_LIKELY_ACTIVE
        out[inactive] = CALL_LIKELY_INACTIVE
    if strong_marker is not None:
        marker = np.asarray(strong_marker, dtype=bool).ravel()
        if marker.size != n:
            raise ValueError(f"strong_marker has {marker.size} entries for {n} rows")
        out[marker] = CALL_LIKELY_INACTIVE
    if natural_resistance:
        out[:] = CALL_LIKELY_INACTIVE
    return out


def intrinsic_columns(intrinsic_symbols: Iterable[str]) -> frozenset[str]:
    """Per-allele ``gene_`` columns of intrinsic symbols (``blaOXA-66`` -> ``gene_blaoxa_66``).

    Only exact per-allele columns: a family column (``gene_blaoxa``) that may also
    hold an acquired carbapenemase is never excluded.
    """
    return frozenset(column_name(PREFIX_GENE, str(symbol)) for symbol in intrinsic_symbols)


def subclass_is_strong(subclass: str | None, wanted: Iterable[str]) -> bool:
    """True when **every** ``;``-separated member subclass of a column carries a wanted token.

    A column built from several symbols (``known_amr_columns.csv`` joins their subclasses
    with ``;``) triggers the subclass rule only when all of them are, e.g., carbapenemases:
    a family column mixing ``CARBAPENEM`` and ``CEPHALOSPORIN`` members cannot tell which
    one a genome carries, so it never forces the call (training-time calls are then less
    often forced than the pipeline's, never more).
    """
    want = frozenset(str(w).strip().upper() for w in wanted if str(w).strip())
    if subclass is None or not want:
        return False
    members = [m for m in str(subclass).split(";") if m.strip() and m.strip().lower() not in {"nan", "none", "<na>"}]
    return bool(members) and all(not class_tokens(m).isdisjoint(want) for m in members)


def strong_marker_columns(
    columns: Iterable[str],
    drug_cfg: DrugConfig | None,
    exclude_columns: Iterable[str] = (),
    subclass_by_column: Mapping[str, str | None] | None = None,
) -> list[str]:
    """Known-AMR columns that trigger override 2 for this drug.

    * Column rule: the column matches a ``strong_markers`` prefix and no
      ``strong_marker_exceptions`` prefix (:meth:`DrugConfig.is_strong_column`).
    * Subclass rule (when ``subclass_by_column`` is given): an acquired-gene column
      (``gene_`` prefix; never ``point_``) whose AMRFinderPlus Subclass carries one of the
      drug's ``strong_subclasses`` for every member symbol (:func:`subclass_is_strong`),
      unless an exception prefix matches it.

    ``exclude_columns`` (:func:`intrinsic_columns` of the species) are never returned.
    """
    if drug_cfg is None:
        return []
    wanted = tuple(drug_cfg.strong_subclasses)
    if not drug_cfg.strong_markers and not (wanted and subclass_by_column):
        return []
    excluded = frozenset(exclude_columns)
    exceptions = tuple(getattr(drug_cfg, "strong_marker_exceptions", ()) or ())
    out = []
    for column in columns:
        c = str(column)
        if c in excluded:
            continue
        if drug_cfg.is_strong_column(c):
            out.append(c)
            continue
        if (
            wanted
            and subclass_by_column
            and c.startswith(PREFIX_GENE)
            and not any(c == e or c.startswith(e) for e in exceptions)
            and subclass_is_strong(subclass_by_column.get(c), wanted)
        ):
            out.append(c)
    return out


def strong_marker_mask(
    known: Any,
    drug_cfg: DrugConfig | None,
    exclude_columns: Iterable[str] = (),
    subclass_by_column: Mapping[str, str | None] | None = None,
) -> np.ndarray:
    """Per row of a known-AMR table: any override-2 column present (> 0) (:func:`strong_marker_columns`).

    Without ``subclass_by_column`` this is the column-prefix half of override 2 only.
    With it (training passes the AMRFinderPlus Subclass of every ``gene_`` column,
    :mod:`genome2mic.features.subclass_map`) the ``strong_subclasses`` half is applied
    too, on columns whose every member symbol carries the subclass, so carbapenemase
    alleles that no prefix lists (blaOXA-23, blaOXA-58, blaGES-5, ...) force the call as
    at prediction time. ``exclude_columns`` (the species' intrinsic genes,
    :func:`intrinsic_columns`) never count, exactly as :func:`strong_marker_hits` skips
    their symbols at prediction time.
    """
    n = len(known)
    cols = strong_marker_columns(getattr(known, "columns", []), drug_cfg, exclude_columns, subclass_by_column)
    if not cols:
        return np.zeros(n, dtype=bool)
    values = np.asarray(known[cols].to_numpy(dtype=np.float64))
    return np.nan_to_num(values, nan=0.0).max(axis=1) > 0


# --------------------------------------------------------------------------- #
# Overrides and reasons
# --------------------------------------------------------------------------- #


def strong_marker_hits(
    symbols_by_column: Mapping[str, Sequence[str]],
    drug_cfg: DrugConfig | None,
    markers: Iterable[Marker] = (),
    intrinsic_symbols: Iterable[str] = (),
) -> list[str]:
    """Symbols of the detected markers that trigger the strong-marker override (override 2).

    A symbol hits when its known-AMR column matches a ``strong_markers`` prefix (and no
    ``strong_marker_exceptions`` prefix), or when it is an acquired gene (Subtype ``AMR``, never ``POINT``) whose AMRFinderPlus
    ``Subclass`` (``/``-separated tokens) is in ``strong_subclasses``. The subclass
    comes from the detection itself, never from the training-time column metadata: a
    family column such as ``gene_blaoxa`` mixes carbapenemases and narrow-spectrum
    enzymes, so only the detected variant's own subclass can tell them apart.

    Args:
        symbols_by_column: present ``gene_`` / ``point_`` column -> detected symbols.
        drug_cfg: the drug's config (``strong_markers`` are column-name prefixes).
        markers: the detections (:attr:`KnownAmrRow.markers`); needed for the subclass rule.
        intrinsic_symbols: the species' intrinsic chromosomal genes
            (:meth:`Config.intrinsic_symbols`, lower case); never a hit under either rule.

    Returns:
        Marker symbols, de-duplicated, in detection order when ``markers`` is given.
        Empty when nothing matches.
    """
    if drug_cfg is None or not (drug_cfg.strong_markers or drug_cfg.strong_subclasses):
        return []
    markers = tuple(markers)
    intrinsic = frozenset(str(symbol).strip().lower() for symbol in intrinsic_symbols)
    hits: list[str] = []
    exceptions = tuple(getattr(drug_cfg, "strong_marker_exceptions", ()) or ())
    for column, symbols in symbols_by_column.items():
        if drug_cfg.is_strong_column(column):
            for symbol in symbols:
                if symbol not in hits and str(symbol).strip().lower() not in intrinsic:
                    hits.append(symbol)
    wanted = frozenset(drug_cfg.strong_subclasses)
    if wanted:
        for marker in markers:
            acquired_gene = marker.subtype == SUBTYPE_AMR and marker.element_type == TYPE_AMR
            if not acquired_gene or marker.symbol in hits:
                continue
            if str(marker.symbol).strip().lower() in intrinsic:
                continue
            if marker.column and any(marker.column == e or marker.column.startswith(e) for e in exceptions):
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
