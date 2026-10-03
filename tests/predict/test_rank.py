"""Tests for ``genome2mic.predict.rank``: call logic, bands, reasons, overrides, ranking."""

from __future__ import annotations

from pathlib import Path

import pytest

from genome2mic.config import Breakpoint, Config, load_config
from genome2mic.predict import rank
from genome2mic.predict.amr_detect import SUBTYPE_AMR, SUBTYPE_POINT, Marker

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(CONFIGS_DIR)


def _bp(s: float, r: float) -> Breakpoint:
    return Breakpoint("KPNEU", "meropenem", s, r, "EUCAST", "2024")


# --------------------------------------------------------------------------- #
# Band
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pred", "q", "expected"),
    [
        (4.0, 1.0, (2.0, 8.0)),
        (4.0, 0.0, (4.0, 4.0)),
        (4.0, 2.0, (1.0, 16.0)),
        # Fractional q: low rounds down, high rounds up -> the band only widens.
        (4.0, 0.5, (2.0, 8.0)),
        (0.25, 1.0, (0.125, 0.5)),
    ],
)
def test_conformal_band_snaps_outward_to_the_grid(pred: float, q: float, expected: tuple[float, float]) -> None:
    assert rank.conformal_band(pred, q) == expected


@pytest.mark.parametrize("bad_q", [-1.0, float("inf"), float("nan")])
def test_conformal_band_rejects_bad_q(bad_q: float) -> None:
    with pytest.raises(ValueError):
        rank.conformal_band(4.0, bad_q)


# --------------------------------------------------------------------------- #
# Margin and call
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("band_high", "s", "expected"),
    [
        (0.125, 2.0, 4),
        (2.0, 2.0, 0),
        (1.0, 8.0, 3),
        # Off-grid EUCAST placeholder breakpoint: floor, never round up.
        (2.0**-10, 0.001, 0),
    ],
)
def test_margin_steps(band_high: float, s: float, expected: int) -> None:
    assert rank.margin_steps(band_high, s) == expected
    assert isinstance(rank.margin_steps(band_high, s), int)


def test_margin_steps_rejects_band_above_breakpoint() -> None:
    with pytest.raises(ValueError):
        rank.margin_steps(4.0, 2.0)


@pytest.mark.parametrize(
    ("band_low", "band_high", "expected_call", "expected_margin"),
    [
        (0.5, 2.0, rank.CALL_LIKELY_ACTIVE, 0),  # band_high == S -> active
        (0.125, 0.5, rank.CALL_LIKELY_ACTIVE, 2),
        (16.0, 64.0, rank.CALL_LIKELY_INACTIVE, None),  # band_low > R
        (8.0, 32.0, rank.CALL_UNCERTAIN, None),  # band_low == R is not > R
        (2.0, 8.0, rank.CALL_UNCERTAIN, None),  # straddles S
        (1.0, 4.0, rank.CALL_UNCERTAIN, None),
    ],
)
def test_call_from_band_follows_the_contract_table(
    band_low: float, band_high: float, expected_call: str, expected_margin: int | None
) -> None:
    call, margin = rank.call_from_band(band_low, band_high, _bp(2.0, 8.0))
    assert call == expected_call
    assert margin == expected_margin


def test_call_from_band_without_breakpoint_is_uncertain() -> None:
    assert rank.call_from_band(0.25, 0.5, None) == (rank.CALL_UNCERTAIN, None)


# --------------------------------------------------------------------------- #
# Classes and reasons
# --------------------------------------------------------------------------- #


def test_class_tokens_split_combined_classes() -> None:
    assert rank.class_tokens("AMINOGLYCOSIDE/QUINOLONE") == {"AMINOGLYCOSIDE", "QUINOLONE"}
    assert rank.class_tokens(" beta-lactam ") == {"BETA-LACTAM"}
    assert rank.class_tokens(None) == frozenset()
    assert rank.class_tokens("") == frozenset()
    assert rank.class_tokens("nan") == frozenset()


@pytest.mark.parametrize(
    ("amr_class", "drug", "expected"),
    [
        ("BETA-LACTAM", "meropenem", True),
        ("BETA-LACTAM", "ceftriaxone", True),
        ("BETA-LACTAM", "ciprofloxacin", False),
        ("QUINOLONE", "ciprofloxacin", True),
        ("QUINOLONE", "levofloxacin", True),
        ("AMINOGLYCOSIDE/QUINOLONE", "gentamicin", True),
        ("AMINOGLYCOSIDE/QUINOLONE", "ciprofloxacin", True),
        ("AMINOGLYCOSIDE/QUINOLONE", "meropenem", False),
        ("AMINOGLYCOSIDE", "amikacin", True),
        ("TRIMETHOPRIM", "trimethoprim-sulfamethoxazole", True),
        ("COLISTIN", "colistin", True),
        ("GLYCOPEPTIDE", "vancomycin", True),
        (None, "meropenem", False),
        ("BETA-LACTAM", "not-a-drug", False),
    ],
)
def test_is_relevant_class(amr_class: str | None, drug: str, expected: bool) -> None:
    assert rank.is_relevant_class(amr_class, drug) is expected


def test_every_configured_drug_has_a_class_mapping(config: Config) -> None:
    missing = [drug for drug in config.drug_names() if drug not in rank.DRUG_CLASSES]
    assert missing == [], f"drugs.yaml drugs without an AMRFinder class mapping: {missing}"


def _marker(symbol: str, amr_class: str | None, subtype: str = SUBTYPE_AMR, column: str | None = None) -> Marker:
    return Marker(symbol=symbol, subtype=subtype, amr_class=amr_class, subclass=amr_class, column=column)


def test_reasons_for_keeps_only_relevant_markers_and_formats_points() -> None:
    markers = [
        _marker("blaKPC-2", "BETA-LACTAM"),
        _marker("gyrA_S83L", "QUINOLONE", SUBTYPE_POINT),
        _marker("parC_S80I", "QUINOLONE", SUBTYPE_POINT),
        _marker("aac(6')-Ib-cr", "AMINOGLYCOSIDE/QUINOLONE"),
        _marker("gyrA_S83L", "QUINOLONE", SUBTYPE_POINT),  # duplicate detection
    ]
    assert rank.reasons_for("ciprofloxacin", markers) == ["gyrA S83L", "parC S80I", "aac(6')-Ib-cr"]
    assert rank.reasons_for("meropenem", markers) == ["blaKPC-2"]
    assert rank.reasons_for("gentamicin", markers) == ["aac(6')-Ib-cr"]
    assert rank.reasons_for("vancomycin", markers) == []


def test_reasons_for_uses_class_by_column_when_detector_has_no_class() -> None:
    markers = [_marker("blaSHV-11", None, column="gene_blashv")]
    assert rank.reasons_for("ceftriaxone", markers) == []
    class_by_column = {"gene_blashv": ("BETA-LACTAM", "BETA-LACTAM")}
    assert rank.reasons_for("ceftriaxone", markers, class_by_column) == ["blaSHV-11"]
    assert rank.reasons_for("ciprofloxacin", markers, class_by_column) == []


# --------------------------------------------------------------------------- #
# Strong markers
# --------------------------------------------------------------------------- #


def test_strong_marker_hits_match_column_prefixes(config: Config) -> None:
    present = {
        "gene_blakpc_2": ("blaKPC-2",),
        "gene_blactx_m": ("blaCTX-M-15", "blaCTX-M-27"),
        "point_gyra_s83l": ("gyrA_S83L",),
        "n_class_beta_lactam": (),
    }
    assert rank.strong_marker_hits(present, config.drugs["meropenem"]) == ["blaKPC-2"]
    assert rank.strong_marker_hits(present, config.drugs["ceftriaxone"]) == ["blaKPC-2", "blaCTX-M-15", "blaCTX-M-27"]
    assert rank.strong_marker_hits(present, config.drugs["ciprofloxacin"]) == []
    assert rank.strong_marker_hits(present, None) == []


def test_strong_marker_hits_exact_prefix_for_oxa_48(config: Config) -> None:
    assert rank.strong_marker_hits({"gene_blaoxa_48": ("blaOXA-48",)}, config.drugs["meropenem"]) == ["blaOXA-48"]
    # The generic OXA family (OXA-1, OXA-10) is not a carbapenemase marker.
    assert rank.strong_marker_hits({"gene_blaoxa": ("blaOXA-1",)}, config.drugs["meropenem"]) == []


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def _pred(drug: str, call: str, margin: int | None) -> dict[str, object]:
    return {"drug": drug, "call": call, "margin_steps": margin}


def test_rank_active_orders_by_tier_then_margin_then_name(config: Config) -> None:
    predictions = [
        _pred("meropenem", rank.CALL_LIKELY_ACTIVE, 6),  # tier 4
        _pred("piperacillin-tazobactam", rank.CALL_LIKELY_ACTIVE, 1),  # tier 3
        _pred("gentamicin", rank.CALL_LIKELY_ACTIVE, 1),  # tier 2
        _pred("ceftriaxone", rank.CALL_LIKELY_ACTIVE, 1),  # tier 2, same margin -> name
        _pred("ciprofloxacin", rank.CALL_LIKELY_ACTIVE, 3),  # tier 2, larger margin first
        _pred("trimethoprim-sulfamethoxazole", rank.CALL_LIKELY_ACTIVE, 0),  # tier 1
        _pred("amikacin", rank.CALL_UNCERTAIN, None),
        _pred("ampicillin", rank.CALL_LIKELY_INACTIVE, None),
    ]
    assert rank.rank_active(predictions, config) == [
        "trimethoprim-sulfamethoxazole",
        "ciprofloxacin",
        "ceftriaxone",
        "gentamicin",
        "piperacillin-tazobactam",
        "meropenem",
    ]


def test_rank_active_is_empty_without_active_drugs(config: Config) -> None:
    predictions = [_pred("meropenem", rank.CALL_UNCERTAIN, None), _pred("ampicillin", rank.CALL_LIKELY_INACTIVE, None)]
    assert rank.rank_active(predictions, config) == []
    assert rank.rank_active([], config) == []


def test_rank_active_puts_unknown_drugs_last(config: Config) -> None:
    predictions = [_pred("mystery-drug", rank.CALL_LIKELY_ACTIVE, 9), _pred("meropenem", rank.CALL_LIKELY_ACTIVE, 0)]
    assert rank.rank_active(predictions, config) == ["meropenem", "mystery-drug"]
