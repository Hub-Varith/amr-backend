"""Tests for ``genome2mic.predict.rank``: call logic, bands, reasons, overrides, ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genome2mic.config import Breakpoint, Config, load_config
from genome2mic.predict import rank
from genome2mic.predict.amr_detect import SUBTYPE_AMR, SUBTYPE_POINT, Marker, known_amr_row

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
# Strong-marker override end to end: detection table -> known-AMR row -> hits
# --------------------------------------------------------------------------- #

CARBAPENEMS = ("ertapenem", "imipenem", "meropenem")
CEPHALOSPORINS_WITH_MARKERS = ("cefotaxime", "ceftriaxone", "ceftazidime", "cefepime")


def _detections(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """AMRFinder-style detection table from ``(symbol, subtype, class, subclass)`` rows (FAKE hits)."""
    return pd.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "type": ["AMR"] * len(rows),
            "subtype": [r[1] for r in rows],
            "class": [r[2] for r in rows],
            "subclass": [r[3] for r in rows],
            "method": ["EXACTX"] * len(rows),
            "coverage": [100.0] * len(rows),
            "identity": [100.0] * len(rows),
            "backend": ["test"] * len(rows),
        }
    )


def _hits(config: Config, drug: str, *rows: tuple[str, str, str, str]) -> list[str]:
    row = known_amr_row(_detections(*rows), config)
    return rank.strong_marker_hits(row.symbols_by_column, config.drugs[drug], row.markers)


@pytest.mark.parametrize("symbol", ["blaOXA-181", "blaOXA-232"])
@pytest.mark.parametrize("drug", CARBAPENEMS + CEPHALOSPORINS_WITH_MARKERS)
def test_kept_oxa_48_like_variants_are_strong_markers(config: Config, symbol: str, drug: str) -> None:
    """keep_variant.csv keeps OXA-181/OXA-232 as their own columns; drugs.yaml must list them too."""
    row = known_amr_row(_detections((symbol, "AMR", "BETA-LACTAM", "CARBAPENEM")), config)
    assert set(row.symbols_by_column) == {"gene_" + symbol.lower().replace("-", "_")}
    # By the column prefix alone (no markers passed), so the drugs.yaml list itself is checked.
    assert rank.strong_marker_hits(row.symbols_by_column, config.drugs[drug]) == [symbol]


@pytest.mark.parametrize(
    "symbol",
    [
        "blaOXA-181",
        "blaOXA-232",
        "blaOXA-23",  # A. baumannii carbapenemase; collapses to gene_blaoxa with OXA-1
        "blaOXA-58",
        "blaGES-5",  # GES carbapenemase variant; collapses to gene_blages with ESBL GES-1
        "blaIMI-1",
    ],
)
@pytest.mark.parametrize("drug", CARBAPENEMS)
def test_acquired_gene_with_carbapenem_subclass_forces_carbapenems_inactive(config: Config, symbol: str, drug: str) -> None:
    assert _hits(config, drug, (symbol, "AMR", "BETA-LACTAM", "CARBAPENEM")) == [symbol]


def test_subclass_rule_is_limited_to_drugs_that_list_it(config: Config) -> None:
    # Ceftriaxone has no strong_subclasses: a GES-5 hit (family gene_blages) is not an override there.
    assert config.drugs["ceftriaxone"].strong_subclasses == ()
    assert _hits(config, "ceftriaxone", ("blaGES-5", "AMR", "BETA-LACTAM", "CARBAPENEM")) == []
    assert _hits(config, "ciprofloxacin", ("blaKPC-2", "AMR", "BETA-LACTAM", "CARBAPENEM")) == []


def test_point_mutation_with_carbapenem_subclass_is_not_an_override(config: Config) -> None:
    """Porin loss (ompK36) raises carbapenem MICs only modestly; the model decides, not the override."""
    for drug in CARBAPENEMS:
        assert _hits(config, drug, ("ompK36_D135DGD", "POINT", "BETA-LACTAM", "CARBAPENEM")) == []


def test_non_carbapenemase_subclass_is_not_an_override(config: Config) -> None:
    for drug in CARBAPENEMS:
        assert _hits(config, drug, ("blaOXA-1", "AMR", "BETA-LACTAM", "BETA-LACTAM")) == []
        assert _hits(config, drug, ("blaGES-1", "AMR", "BETA-LACTAM", "CEPHALOSPORIN")) == []
        assert _hits(config, drug, ("blaCTX-M-15", "AMR", "BETA-LACTAM", "CEPHALOSPORIN")) == []


def test_multi_token_subclass_matches_on_any_token(config: Config) -> None:
    assert _hits(config, "meropenem", ("blaFAKE-1", "AMR", "BETA-LACTAM", "CEPHALOSPORIN/CARBAPENEM")) == ["blaFAKE-1"]


def test_strong_marker_hits_lists_every_marker_once_in_detection_order(config: Config) -> None:
    hits = _hits(
        config,
        "meropenem",
        ("blaOXA-23", "AMR", "BETA-LACTAM", "CARBAPENEM"),  # subclass rule
        ("blaOXA-1", "AMR", "BETA-LACTAM", "BETA-LACTAM"),  # same column, not a carbapenemase
        ("blaKPC-2", "AMR", "BETA-LACTAM", "CARBAPENEM"),  # prefix and subclass rule
        ("ompK36_D135DGD", "POINT", "BETA-LACTAM", "CARBAPENEM"),  # POINT: never
        ("blaNDM-1", "AMR", "BETA-LACTAM", "CARBAPENEM"),
    )
    assert hits == ["blaOXA-23", "blaKPC-2", "blaNDM-1"]


def test_marker_without_subclass_falls_back_to_prefixes_only(config: Config) -> None:
    present = {"gene_blaoxa": ("blaOXA-23",)}
    markers = (Marker("blaOXA-23", SUBTYPE_AMR, "BETA-LACTAM", None, column="gene_blaoxa"),)
    assert rank.strong_marker_hits(present, config.drugs["meropenem"], markers) == []


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


# --------------------------------------------------------------------------- #
# Intrinsic genes never trigger the strong-marker override (intrinsic_markers.csv)
# --------------------------------------------------------------------------- #

ABAU_CARBAPENEMS = ("imipenem", "meropenem", "doripenem")


def _abau_hits(config: Config, drug: str, *rows: tuple[str, str, str, str]) -> list[str]:
    row = known_amr_row(_detections(*rows), config)
    return rank.strong_marker_hits(
        row.symbols_by_column, config.drugs[drug], row.markers, intrinsic_symbols=config.intrinsic_symbols("ABAU")
    )


def test_oxa_51_family_is_intrinsic_to_abau_only(config: Config) -> None:
    for symbol in ("blaOXA-51", "blaOXA-66", "blaOXA-69", "blaOXA-71", "blaOXA-82"):
        assert config.is_intrinsic_marker("ABAU", symbol)
    for symbol in ("blaOXA-23", "blaOXA-24", "blaOXA-40", "blaOXA-72", "blaOXA-58", "blaOXA-48", "blaNDM-1"):
        assert not config.is_intrinsic_marker("ABAU", symbol)
    assert not config.is_intrinsic_marker("KPNEU", "blaOXA-66")
    assert config.intrinsic_symbols(None) == frozenset()


@pytest.mark.parametrize("drug", ABAU_CARBAPENEMS)
def test_abau_with_only_oxa_66_has_no_override(config: Config, drug: str) -> None:
    # AMRFinderPlus reports the chromosomal OXA-51-like gene with Subclass CARBAPENEM.
    assert _abau_hits(config, drug, ("blaOXA-66", "AMR", "BETA-LACTAM", "CARBAPENEM")) == []
    # Without the species exclusion the subclass rule would fire on every A. baumannii.
    assert _hits(config, drug, ("blaOXA-66", "AMR", "BETA-LACTAM", "CARBAPENEM")) == ["blaOXA-66"]


@pytest.mark.parametrize("symbol", ["blaOXA-23", "blaOXA-24", "blaOXA-40", "blaOXA-58", "blaNDM-1"])
@pytest.mark.parametrize("drug", ABAU_CARBAPENEMS)
def test_abau_acquired_carbapenemase_still_overrides(config: Config, symbol: str, drug: str) -> None:
    assert _abau_hits(config, drug, (symbol, "AMR", "BETA-LACTAM", "CARBAPENEM")) == [symbol]
    # Also next to the intrinsic OXA-66 that every isolate carries.
    hits = _abau_hits(
        config, drug, ("blaOXA-66", "AMR", "BETA-LACTAM", "CARBAPENEM"), (symbol, "AMR", "BETA-LACTAM", "CARBAPENEM")
    )
    assert hits == [symbol]


def test_intrinsic_exclusion_applies_to_the_column_prefix_rule(config: Config) -> None:
    # A per-allele column that a prefix would match (hypothetical config) is still excluded.
    from dataclasses import replace as dc_replace

    drug_cfg = dc_replace(config.drugs["meropenem"], strong_markers=("gene_blaoxa_6",), strong_subclasses=())
    symbols = {"gene_blaoxa_66": ("blaOXA-66",)}
    assert rank.strong_marker_hits(symbols, drug_cfg) == ["blaOXA-66"]
    assert rank.strong_marker_hits(symbols, drug_cfg, intrinsic_symbols=config.intrinsic_symbols("ABAU")) == []


def test_training_time_mask_excludes_intrinsic_columns_identically(config: Config) -> None:
    """Training-time calls (strong_marker_mask) drop the same intrinsic genes as prediction."""
    from dataclasses import replace as dc_replace

    excluded = rank.intrinsic_columns(config.intrinsic_symbols("ABAU"))
    assert "gene_blaoxa_66" in excluded and "gene_blaoxa_51" in excluded
    assert "gene_blaoxa_23" not in excluded and "gene_blaoxa" not in excluded  # family column may hold OXA-23
    known = pd.DataFrame(
        {
            "gene_blaoxa_66": [1, 1, 1, 0],
            "gene_blaoxa_23": [0, 1, 0, 0],
            "gene_blandm_1": [0, 0, 1, 0],
        }
    )
    drug_cfg = dc_replace(config.drugs["meropenem"], strong_markers=("gene_blaoxa_", "gene_blandm"))
    assert rank.strong_marker_mask(known, drug_cfg).tolist() == [True, True, True, False]
    assert rank.strong_marker_mask(known, drug_cfg, exclude_columns=excluded).tolist() == [False, True, True, False]
    # The shipped meropenem config: OXA-66 alone never forces inactive; NDM-1 does.
    assert rank.strong_marker_mask(known, config.drugs["meropenem"], exclude_columns=excluded).tolist() == [
        False, False, True, False,
    ]


@pytest.mark.parametrize(
    ("pred", "q_up", "q_low", "expected"),
    [(4.0, 0.0, 0.0, (4.0, 4.0)), (4.0, 2.0, 1.0, (2.0, 16.0)), (4.0, 0.0, 3.0, (0.5, 4.0)), (1.0, 1.5, 0.5, (0.5, 4.0))],
)
def test_conformal_band_asym_snaps_outward(pred: float, q_up: float, q_low: float, expected: tuple[float, float]) -> None:
    assert rank.conformal_band_asym(pred, q_up, q_low) == expected


@pytest.mark.parametrize(("q_up", "q_low"), [(-1.0, 1.0), (1.0, float("inf")), (float("nan"), 1.0)])
def test_conformal_band_asym_rejects_bad_q(q_up: float, q_low: float) -> None:
    with pytest.raises(ValueError):
        rank.conformal_band_asym(4.0, q_up, q_low)
