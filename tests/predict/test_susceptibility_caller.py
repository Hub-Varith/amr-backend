"""Band-vs-breakpoint calls (CLAUDE.md call logic). Breakpoint values here are test-only."""

import pandas as pd
import pytest

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.predict.call_thresholds import CallThresholds
from genome2mic.predict.susceptibility_caller import SusceptibilityCaller


@pytest.fixture
def caller() -> SusceptibilityCaller:
    rows = pd.DataFrame(
        [
            ("KPNEU", "test-drug", "CLSI", 2020, 1.0, 2.0, "test"),
            ("KPNEU", "test-drug", "CLSI", 2023, 2.0, 4.0, "test"),
            ("KPNEU", "eu-only-drug", "EUCAST", 2020, 1.0, 2.0, "test"),
        ],
        columns=["species", "drug", "standard", "effective_year", "s_breakpoint", "r_breakpoint", "version"],
    )
    return SusceptibilityCaller(BreakpointTable(rows), standard="CLSI", year=2026)


def test_band_top_at_or_below_s_is_likely_active(caller: SusceptibilityCaller) -> None:
    result = caller.call("KPNEU", "test-drug", band_low=0.25, band_high=0.5)
    assert result.call == "likely_active"
    assert result.margin_steps == 2
    assert (result.s_breakpoint, result.r_breakpoint) == (2.0, 4.0)


def test_band_top_equal_to_s_has_zero_margin(caller: SusceptibilityCaller) -> None:
    result = caller.call("KPNEU", "test-drug", band_low=1.0, band_high=2.0)
    assert result.call == "likely_active"
    assert result.margin_steps == 0


def test_band_bottom_above_r_is_likely_inactive(caller: SusceptibilityCaller) -> None:
    result = caller.call("KPNEU", "test-drug", band_low=8.0, band_high=32.0)
    assert result.call == "likely_inactive"
    assert result.margin_steps is None


def test_band_crossing_a_breakpoint_is_uncertain(caller: SusceptibilityCaller) -> None:
    assert caller.call("KPNEU", "test-drug", band_low=1.0, band_high=4.0).call == "uncertain"
    assert caller.call("KPNEU", "test-drug", band_low=4.0, band_high=8.0).call == "uncertain"


def test_uses_the_breakpoint_version_for_the_chosen_year(caller: SusceptibilityCaller) -> None:
    old_caller = SusceptibilityCaller(caller.breakpoints, standard="CLSI", year=2021)
    assert old_caller.call("KPNEU", "test-drug", band_low=1.0, band_high=2.0).call == "uncertain"


def test_drug_without_breakpoint_in_the_chosen_standard_is_uncertain(caller: SusceptibilityCaller) -> None:
    result = caller.call("KPNEU", "eu-only-drug", band_low=0.25, band_high=0.5)
    assert result.call == "uncertain"
    assert result.s_breakpoint is None
    assert result.reason == "no CLSI breakpoint for this drug"


def test_probability_active_is_the_chance_the_mic_is_at_or_below_s() -> None:
    # mu exactly on the S breakpoint (log2 2 = 1): half the bell curve is at or below it.
    assert SusceptibilityCaller.probability_active(mu_log2=1.0, sigma_log2=2.0, s_breakpoint=2.0) == pytest.approx(0.5)
    assert SusceptibilityCaller.probability_active(mu_log2=-5.0, sigma_log2=1.0, s_breakpoint=2.0) > 0.99
    assert SusceptibilityCaller.probability_active(mu_log2=7.0, sigma_log2=1.0, s_breakpoint=2.0) < 0.01


def test_probability_rule_is_used_when_the_pair_has_thresholds(caller: SusceptibilityCaller) -> None:
    thresholds = CallThresholds(
        [{"species": "KPNEU", "drug": "test-drug", "active_min": 0.9, "inactive_max": 0.2, "n_resistant": 50,
          "n_susceptible": 50}],
        vme_target=0.01,
        me_target=0.03,
    )
    probability_caller = SusceptibilityCaller(caller.breakpoints, standard="CLSI", year=2026, thresholds=thresholds)

    # The band crosses S (4 > 2), so the band rule alone would say uncertain.
    active = probability_caller.call("KPNEU", "test-drug", band_low=0.5, band_high=4.0, p_active=0.95)
    inactive = probability_caller.call("KPNEU", "test-drug", band_low=1.0, band_high=16.0, p_active=0.1)
    unsure = probability_caller.call("KPNEU", "test-drug", band_low=0.5, band_high=4.0, p_active=0.5)

    assert (active.call, active.p_active, active.margin_steps) == ("likely_active", 0.95, 0)
    assert inactive.call == "likely_inactive"
    assert unsure.call == "uncertain"


def test_band_rule_is_the_fallback_without_thresholds_for_the_pair(caller: SusceptibilityCaller) -> None:
    probability_caller = SusceptibilityCaller(
        caller.breakpoints, standard="CLSI", year=2026, thresholds=CallThresholds([], vme_target=0.01, me_target=0.03)
    )

    result = probability_caller.call("KPNEU", "test-drug", band_low=0.5, band_high=4.0, p_active=0.99)

    assert result.call == "uncertain"
    assert result.p_active == 0.99
