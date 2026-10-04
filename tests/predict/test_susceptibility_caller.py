"""Band-vs-breakpoint calls (CLAUDE.md call logic). Breakpoint values here are test-only."""

import pandas as pd
import pytest

from genome2mic.ingest.breakpoint_table import BreakpointTable
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
