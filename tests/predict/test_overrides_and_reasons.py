from pathlib import Path

import pandas as pd

from genome2mic.features.amrfinder_table import AmrFinderHit
from genome2mic.predict.call_overrides import NATURAL_RESISTANCE, STRONG_MARKER, CallOverrides
from genome2mic.predict.drug_reasons import DrugReasons

CONFIGS = Path(__file__).parents[2] / "configs"


def hit(symbol, drug_class="BETA-LACTAM", subclass="CARBAPENEM", subtype="AMR", method="EXACTX", type_="AMR"):
    return AmrFinderHit(symbol, type_, subtype, drug_class, subclass, method)


def overrides():
    natural = pd.DataFrame({"species": ["KPNEU"], "drug": ["ampicillin"], "reason": ["intrinsic"]})
    markers = pd.DataFrame({
        "species": ["KPNEU"], "drug": ["ertapenem"],
        "symbol_regex": [r"^bla(KPC|NDM)(-|$)"], "reason": ["carbapenemase"], "train_r_share": [0.99],
    })
    return CallOverrides(natural, markers)


def test_natural_resistance_needs_no_marker():
    assert overrides().find("KPNEU", "ampicillin", []).kind == NATURAL_RESISTANCE


def test_strong_marker_names_the_gene_and_ignores_broken_copies():
    found = overrides().find("KPNEU", "ertapenem", [hit("blaKPC-2"), hit("blaNDM-1", method="INTERNAL_STOP")])
    assert found.kind == STRONG_MARKER and found.markers == ["blaKPC-2"]
    assert overrides().find("KPNEU", "ertapenem", [hit("blaNDM-1", method="INTERNAL_STOP")]) is None
    assert overrides().find("KPNEU", "ertapenem", [hit("blaKPC")]).markers == ["blaKPC"]   # allele unknown


def test_no_rule_no_override():
    assert overrides().find("KPNEU", "meropenem", [hit("blaKPC-2")]) is None


def test_shipped_configs_load_and_match_real_symbols():
    shipped = CallOverrides.from_configs(CONFIGS)
    assert shipped.find("KPNEU", "ampicillin", []).kind == NATURAL_RESISTANCE
    assert shipped.find("ABAU", "meropenem", [hit("blaOXA-23")]).markers == ["blaOXA-23"]
    assert shipped.find("ABAU", "meropenem", [hit("blaOXA-51")]) is None          # intrinsic OXA-51-like is not a marker
    assert shipped.find("SAUR", "penicillin", [hit("blaZ", subclass="PENICILLIN")]).markers == ["blaZ"]


def test_reasons_follow_class_and_subclass():
    reasons = DrugReasons.from_configs(CONFIGS)
    hits = [
        hit("blaKPC-2"),
        hit("blaSHV-11", subclass="BETA-LACTAM"),
        hit("blaCTX-M-15", subclass="CEPHALOSPORIN"),
        hit("gyrA_S83I", drug_class="QUINOLONE", subclass="QUINOLONE", subtype="POINT", method="POINTX"),
        hit("aac(6')-Ib-cr5", drug_class="AMINOGLYCOSIDE/QUINOLONE", subclass="AMIKACIN/KANAMYCIN/QUINOLONE/TOBRAMYCIN"),
        hit("merA", drug_class="MERCURY", subclass="MERCURY", type_="STRESS"),
    ]
    assert reasons.for_drug("meropenem", hits) == ["blaKPC-2"]
    assert reasons.for_drug("ceftriaxone", hits) == ["blaCTX-M-15", "blaKPC-2"]
    assert reasons.for_drug("ampicillin", hits) == ["blaCTX-M-15", "blaKPC-2", "blaSHV-11"]
    assert reasons.for_drug("ciprofloxacin", hits) == ["aac(6')-Ib-cr5", "gyrA S83I"]
    assert reasons.for_drug("amikacin", hits) == ["aac(6')-Ib-cr5"]
    assert reasons.for_drug("gentamicin", hits) == []
