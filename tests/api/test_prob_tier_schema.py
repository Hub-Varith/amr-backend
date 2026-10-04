"""v0.7: the report's prob_tier is capped to agree with the call (schema-validated)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from genome2mic.api.schemas.drug_prediction import DrugPrediction


def _pred(call: str, p: float, tier: str, override: str | None = None) -> DrugPrediction:
    return DrugPrediction(drug="meropenem", pred_mic=0.25, band_low=0.125, band_high=0.5, s_breakpoint=1.0,
                          r_breakpoint=4.0, call=call, margin_steps=None, override=override, prob_works=p, prob_tier=tier)


def test_works_tier_next_to_likely_inactive_must_show_uncertain() -> None:
    ok = _pred("likely_inactive", 0.92, "uncertain", override="strong_marker")
    assert ok.prob_tier == "uncertain"
    with pytest.raises(ValidationError, match="prob_tier"):
        _pred("likely_inactive", 0.92, "very_likely_works", override="strong_marker")


def test_fails_tier_next_to_likely_active_must_show_uncertain() -> None:
    assert _pred("likely_active", 0.05, "uncertain").prob_tier == "uncertain"
    with pytest.raises(ValidationError, match="prob_tier"):
        _pred("likely_active", 0.05, "very_likely_fails")


def test_uncapped_tiers_still_match_the_probability() -> None:
    assert _pred("uncertain", 0.95, "very_likely_works").prob_tier == "very_likely_works"
    assert _pred("likely_inactive", 0.05, "very_likely_fails").prob_tier == "very_likely_fails"
    with pytest.raises(ValidationError, match="prob_tier"):
        _pred("uncertain", 0.95, "uncertain")
