"""Chance -> confidence level."""

import pytest

from genome2mic.predict.confidence_levels import ConfidenceLevels


@pytest.mark.parametrize(
    ("p_active", "expected"),
    [
        (0.0, "very_likely_inactive"),
        (0.019, "very_likely_inactive"),
        (0.02, "probably_inactive"),
        (0.10, "leans_inactive"),
        (0.30, "unclear"),
        (0.50, "unclear"),
        (0.699, "unclear"),
        (0.70, "leans_active"),
        (0.90, "probably_active"),
        (0.98, "very_likely_active"),
        (1.0, "very_likely_active"),
    ],
)
def test_level_for(p_active: float, expected: str) -> None:
    assert ConfidenceLevels.level_for(p_active) == expected


def test_out_of_range_probability_is_rejected() -> None:
    with pytest.raises(ValueError):
        ConfidenceLevels.level_for(1.2)
