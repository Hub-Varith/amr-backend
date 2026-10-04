"""Breakpoint lookup by standard and year, and sanity of the checked-in breakpoint files."""

import math
from pathlib import Path

import pandas as pd
import pytest

from genome2mic.ingest.breakpoint_table import BreakpointTable

CONFIG_BREAKPOINTS = Path(__file__).resolve().parents[2] / "configs" / "breakpoints"


@pytest.fixture
def table() -> BreakpointTable:
    rows = pd.DataFrame(
        [
            ("KPNEU", "ciprofloxacin", "CLSI", 2010, 1.0, 2.0, "M100-S20"),
            ("KPNEU", "ciprofloxacin", "CLSI", 2019, 0.25, 0.5, "M100-29"),
            ("KPNEU", "meropenem", "EUCAST", 2010, 2.0, 8.0, "v1.0"),
            ("KPNEU", "meropenem", "EUCAST", 2019, 2.0, 8.0, "v9.0"),
            ("KPNEU", "ceftazidime", "CLSI", 2020, 4.0, 8.0, "M100-30"),
        ],
        columns=["species", "drug", "standard", "effective_year", "s_breakpoint", "r_breakpoint", "version"],
    )
    return BreakpointTable(rows)


def test_lookup_picks_the_latest_version_in_force(table: BreakpointTable) -> None:
    assert table.lookup("KPNEU", "ciprofloxacin", "CLSI", 2015) == (1.0, 2.0)
    assert table.lookup("KPNEU", "ciprofloxacin", "CLSI", 2019) == (0.25, 0.5)
    assert table.lookup("KPNEU", "ciprofloxacin", "CLSI", 2024) == (0.25, 0.5)


def test_lookup_before_the_first_version_is_none(table: BreakpointTable) -> None:
    assert table.lookup("KPNEU", "ciprofloxacin", "CLSI", 2005) is None


def test_lookup_without_year_works_only_when_all_versions_agree(table: BreakpointTable) -> None:
    assert table.lookup("KPNEU", "meropenem", "EUCAST", None) == (2.0, 8.0)
    assert table.lookup("KPNEU", "ciprofloxacin", "CLSI", None) is None


def test_lookup_without_year_needs_the_pair_in_the_oldest_file(table: BreakpointTable) -> None:
    # Ceftazidime has one CLSI row, from 2020. A result with no year may predate 2020,
    # when the breakpoint could have been different, so it must not "agree" by default.
    assert table.lookup("KPNEU", "ceftazidime", "CLSI", None) is None
    assert table.lookup("KPNEU", "ceftazidime", "CLSI", 2021) == (4.0, 8.0)


def test_lookup_unknown_pair_is_none(table: BreakpointTable) -> None:
    assert table.lookup("SAUR", "ciprofloxacin", "CLSI", 2020) is None
    assert table.lookup("KPNEU", "ciprofloxacin", "EUCAST", 2020) is None


def test_checked_in_breakpoints_are_on_the_doubling_grid_and_ordered() -> None:
    table = BreakpointTable.from_directory(CONFIG_BREAKPOINTS).table
    assert not table.empty
    for value in pd.concat([table["s_breakpoint"], table["r_breakpoint"]]):
        assert math.log2(value) == round(math.log2(value))
    assert (table["s_breakpoint"] <= table["r_breakpoint"]).all()
    assert set(table["standard"]) <= {"CLSI", "EUCAST"}
    assert not table.duplicated(subset=["species", "drug", "standard", "effective_year"]).any()
