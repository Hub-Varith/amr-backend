"""Tests for genome2mic.ingest.harmonize -- written before the implementation.

Covers every row of the contract's interval table, the ``<`` rule, S/I/R with
EUCAST vs CLSI breakpoints, standard-year table selection, every drop reason,
method classification (disk rows never get a numeric MIC), unit handling,
biosample de-duplication and re-keying, the three duplicate rules, the
acceptance checks, the count table and the pair-inclusion thresholds.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic.config import Config
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.ingest import harmonize as hz
from tests.ingest import (
    LABEL_COLUMNS,
    RAW_COLUMNS,
    log_counts,
    make_config,
    make_raw,
    raw_row,
)

INF = math.inf


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    return make_config(tmp_path_factory.mktemp("configs"))


def run(rows: list[dict], config: Config) -> tuple[pd.DataFrame, DropLog]:
    """Harmonize ``rows`` and return (labels, droplog)."""
    log = DropLog("ingest")
    labels = hz.harmonize(make_raw(rows), config, log)
    return labels, log


def only(labels: pd.DataFrame) -> pd.Series:
    assert len(labels) == 1, f"expected exactly one row, got {len(labels)}:\n{labels}"
    return labels.iloc[0]


def interval(row: pd.Series) -> tuple[float, float, str]:
    return (float(row["mic_lower"]), float(row["mic_upper"]), str(row["censor"]))


# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #


def test_schema_constants_match_the_contract() -> None:
    assert tuple(hz.RAW_COLUMNS) == RAW_COLUMNS
    assert tuple(hz.LABEL_COLUMNS) == LABEL_COLUMNS
    forbidden = {"lineage_cluster", "st", "split", "fold"}
    assert forbidden.isdisjoint(hz.LABEL_COLUMNS)


# --------------------------------------------------------------------------- #
# The interval rule (contract table) and the '<' rule, through harmonize()
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("sign", "value", "expected", "raw_result"),
    [
        ("=", "8", (4.0, 8.0, "interval"), "=8"),
        ("<=", "0.25", (0.0, 0.25, "left"), "<=0.25"),
        (">", "32", (32.0, INF, "right"), ">32"),
        (">=", "16", (8.0, INF, "right"), ">=16"),
        ("<", "0.5", (0.0, 0.25, "left"), "<0.5"),
        (None, "8", (4.0, 8.0, "interval"), "=8"),  # blank sign means '='
        ("", "8", (4.0, 8.0, "interval"), "=8"),
        ("=", "6", (4.0, 8.0, "interval"), "=6"),  # off-grid gradient value
        ("=", 8.0, (4.0, 8.0, "interval"), "=8"),  # numeric value, not a string
        ("≤", "0.25", (0.0, 0.25, "left"), "<=0.25"),  # unicode sign
        (None, ">32", (32.0, INF, "right"), ">32"),  # sign embedded in the value
        (None, "<=0.25", (0.0, 0.25, "left"), "<=0.25"),
    ],
)
def test_numeric_interval_rule(
    config: Config, sign: str | None, value: object, expected: tuple, raw_result: str
) -> None:
    labels, _ = run([raw_row(sign=sign, value=value, sir_raw=None)], config)
    row = only(labels)
    assert interval(row) == expected
    assert row["raw_result"] == raw_result
    assert row["method"] == "dilution"
    assert row["drug"] == "meropenem" and row["species"] == "KPNEU"
    assert row["sir"] is None  # nothing reported, nothing derived


@pytest.mark.parametrize(
    ("sir_raw", "expected", "raw_result"),
    [
        ("S", (0.0, 1.0, "left"), "S"),  # breakpoint S <= 1
        ("R", (2.0, INF, "right"), "R"),  # breakpoint R > 2
        ("I", (1.0, 2.0, "interval"), "I"),  # S <= 1 and R > 2
        ("Susceptible", (0.0, 1.0, "left"), "S"),
        ("resistant", (2.0, INF, "right"), "R"),
        ("Intermediate", (1.0, 2.0, "interval"), "I"),
    ],
)
def test_sir_only_interval_rule_eucast_ceftriaxone(
    config: Config, sir_raw: str, expected: tuple, raw_result: str
) -> None:
    """KPNEU x ceftriaxone under EUCAST 2024 has S<=1, R>2: the contract's own example."""
    labels, _ = run(
        [raw_row(antibiotic_raw="ceftriaxone", sir_raw=sir_raw, sign=None, value=None, unit=None)],
        config,
    )
    row = only(labels)
    assert interval(row) == expected
    assert row["raw_result"] == raw_result
    assert row["sir"] == raw_result
    assert row["standard"] == "EUCAST" and row["standard_year"] == 2024


class TestStandards:
    def test_eucast_vs_clsi_meropenem(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", sir_raw="S", sign=None, value=None, standard="EUCAST"),
            raw_row(genome_id="b", sir_raw="S", sign=None, value=None, standard="CLSI"),
            raw_row(genome_id="c", sir_raw="R", sign=None, value=None, standard="EUCAST"),
            raw_row(genome_id="d", sir_raw="R", sign=None, value=None, standard="CLSI"),
            raw_row(genome_id="e", sir_raw="I", sign=None, value=None, standard="CLSI"),
        ]
        labels, _ = run(rows, config)
        by = labels.set_index("genome_id")
        assert interval(by.loc["a"]) == (0.0, 2.0, "left")  # EUCAST S <= 2
        assert interval(by.loc["b"]) == (0.0, 1.0, "left")  # CLSI S <= 1
        assert interval(by.loc["c"]) == (8.0, INF, "right")  # EUCAST R > 8
        assert interval(by.loc["d"]) == (2.0, INF, "right")  # CLSI R >= 4 stored as > 2
        assert interval(by.loc["e"]) == (1.0, 2.0, "interval")  # CLSI I = 2
        assert by.loc["b", "standard"] == "CLSI"

    def test_standard_aliases_and_case(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", sir_raw="S", sign=None, value=None, standard="clsi"),
            raw_row(genome_id="b", sir_raw="S", sign=None, value=None, standard=" NCCLS "),
        ]
        labels, _ = run(rows, config)
        assert labels["standard"].tolist() == ["CLSI", "CLSI"]
        assert labels["mic_upper"].tolist() == [1.0, 1.0]

    @pytest.mark.parametrize(
        ("year", "expected_upper"),
        [
            (2016, 0.5),  # the 2016 table: S <= 0.5
            (2024, 0.25),
            ("2016", 0.5),  # string year from a CSV
            ("2016.0", 0.5),
        ],
    )
    def test_standard_year_selects_the_table(
        self, config: Config, year: object, expected_upper: float
    ) -> None:
        labels, _ = run(
            [raw_row(antibiotic_raw="ciprofloxacin", sir_raw="S", sign=None, value=None, standard_year=year)],
            config,
        )
        row = only(labels)
        assert interval(row) == (0.0, expected_upper, "left")
        assert row["standard_year"] == int(float(str(year)))

    @pytest.mark.parametrize("year", [None, "", "unknown"])
    def test_sir_only_row_with_null_year_is_dropped(self, config: Config, year: object) -> None:
        """No year -> the matching table is unknown; never assume the latest one (decision 2)."""
        labels, log = run(
            [raw_row(antibiotic_raw="ciprofloxacin", sir_raw="S", sign=None, value=None, standard_year=year)],
            config,
        )
        assert labels.empty
        counts = log_counts(log)
        assert counts[hz.Reason.NULL_YEAR] == 1
        assert counts[hz.Reason.NO_TABLE_FOR_YEAR] == 0 and counts[hz.Reason.NO_BREAKPOINT] == 0

    def test_sir_only_row_whose_year_has_no_table_is_dropped(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", antibiotic_raw="ciprofloxacin", sir_raw="S", sign=None, value=None,
                    standard="EUCAST", standard_year=2019),
            raw_row(genome_id="b", antibiotic_raw="ciprofloxacin", sir_raw="R", sign=None, value=None,
                    standard="EUCAST", standard_year=2019),
            raw_row(genome_id="c", antibiotic_raw="meropenem", sir_raw="R", sign=None, value=None,
                    method_raw="Disk diffusion", standard="CLSI", standard_year=2016),
        ]
        labels, log = run(rows, config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.NO_TABLE_FOR_YEAR] == 3
        assert log_counts(log)[hz.Reason.NULL_YEAR] == 0
        detail = [r.detail for r in log.records if r.reason == hz.Reason.NO_TABLE_FOR_YEAR][0]
        assert "EUCAST 2019 (2)" in detail and "CLSI 2016 (1)" in detail

    def test_year_with_a_table_but_no_pair_row_is_no_breakpoint(self, config: Config) -> None:
        # The EUCAST 2016 test table has no ceftriaxone row.
        labels, log = run(
            [raw_row(antibiotic_raw="ceftriaxone", sir_raw="S", sign=None, value=None, standard_year=2016)],
            config,
        )
        assert labels.empty
        assert log_counts(log)[hz.Reason.NO_BREAKPOINT] == 1
        assert log_counts(log)[hz.Reason.NO_TABLE_FOR_YEAR] == 0

    @pytest.mark.parametrize("year", [None, 2019])
    def test_numeric_rows_do_not_need_a_table_for_their_year(self, config: Config, year: int | None) -> None:
        labels, log = run([raw_row(sign="=", value="8", sir_raw="R", standard_year=year)], config)
        row = only(labels)
        assert interval(row) == (4.0, 8.0, "interval")
        assert (pd.isna(row["standard_year"]) if year is None else row["standard_year"] == year)
        assert log_counts(log)[hz.Reason.NULL_YEAR] == 0
        assert log_counts(log)[hz.Reason.NO_TABLE_FOR_YEAR] == 0

    def test_null_standard_sir_only_row_is_dropped(self, config: Config) -> None:
        labels, log = run([raw_row(sir_raw="S", sign=None, value=None, standard=None)], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.NULL_STANDARD] == 1

    def test_unknown_standard_sir_only_row_is_dropped(self, config: Config) -> None:
        labels, log = run([raw_row(sir_raw="R", sign=None, value=None, standard="BSAC")], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.NULL_STANDARD] == 1

    def test_null_standard_numeric_row_is_kept_with_null_standard(self, config: Config) -> None:
        labels, log = run([raw_row(standard=None, standard_year=None)], config)
        row = only(labels)
        assert interval(row) == (4.0, 8.0, "interval")
        assert row["standard"] is None and pd.isna(row["standard_year"])
        assert log_counts(log)[hz.Reason.NULL_STANDARD] == 0

    def test_no_breakpoint_for_pair_drops_sir_only_row(self, config: Config) -> None:
        # ECOLI x ceftriaxone has no row in the test tables.
        labels, log = run(
            [raw_row(species="ECOLI", antibiotic_raw="ceftriaxone", sir_raw="S", sign=None, value=None)],
            config,
        )
        assert labels.empty
        assert log_counts(log)[hz.Reason.NO_BREAKPOINT] == 1

    def test_intermediate_without_an_i_category_is_dropped(self, config: Config) -> None:
        # EUCAST KPNEU ampicillin: S == R == 8 -> there is no I category.
        labels, log = run(
            [raw_row(antibiotic_raw="ampicillin", sir_raw="I", sign=None, value=None)],
            config,
        )
        assert labels.empty
        assert log_counts(log)[hz.Reason.INVALID_I] == 1

    def test_unknown_sir_on_sir_only_row_is_dropped(self, config: Config) -> None:
        labels, log = run([raw_row(sir_raw="SDD", sign=None, value=None)], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.UNKNOWN_SIR] == 1


# --------------------------------------------------------------------------- #
# Evidence, drug, species filters
# --------------------------------------------------------------------------- #


class TestFilters:
    def test_evidence_filter_keeps_laboratory_method_only(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", evidence="Laboratory Method"),
            raw_row(genome_id="b", evidence="Computational Prediction"),
            raw_row(genome_id="c", evidence=None),
            raw_row(genome_id="d", evidence="laboratory method"),  # case-insensitive
        ]
        labels, log = run(rows, config)
        assert labels["genome_id"].tolist() == ["a", "d"]
        assert log_counts(log)[hz.Reason.EVIDENCE] == 2

    def test_ncbi_rows_with_null_evidence_are_lab_by_definition(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="NCBI_SAMN1", source="NCBI", evidence=None),
            raw_row(genome_id="NCBI_SAMN2", source="NCBI", evidence="Computational Prediction"),
        ]
        labels, log = run(rows, config)
        assert labels["genome_id"].tolist() == ["NCBI_SAMN1"]
        assert log_counts(log)[hz.Reason.EVIDENCE] == 1

    def test_drug_normalization_and_unknown_drug(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", antibiotic_raw="MEM"),
            raw_row(genome_id="b", antibiotic_raw=" Meropenem Trihydrate "),
            raw_row(genome_id="c", antibiotic_raw="Ceftriaxone sodium"),
            raw_row(genome_id="d", antibiotic_raw="cefiderocol"),
            raw_row(genome_id="e", antibiotic_raw=None),
        ]
        labels, log = run(rows, config)
        assert labels.set_index("genome_id")["drug"].to_dict() == {
            "a": "meropenem",
            "b": "meropenem",
            "c": "ceftriaxone",
        }
        assert log_counts(log)[hz.Reason.UNKNOWN_DRUG] == 2
        detail = [r.detail for r in log.records if r.reason == hz.Reason.UNKNOWN_DRUG][0]
        assert "cefiderocol" in detail

    def test_unknown_species_is_dropped(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", species="KPNEU"),
            raw_row(genome_id="b", species=None),
            raw_row(genome_id="c", species="SAUR"),  # not in the test config
            raw_row(genome_id="d", species="kpneu"),  # keys are upper-case; be lenient
        ]
        labels, log = run(rows, config)
        assert labels["genome_id"].tolist() == ["a", "d"]
        assert labels["species"].tolist() == ["KPNEU", "KPNEU"]
        assert log_counts(log)[hz.Reason.UNKNOWN_SPECIES] == 2

    def test_missing_genome_id_is_dropped(self, config: Config) -> None:
        labels, log = run([raw_row(genome_id=None), raw_row(genome_id="  ")], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.NO_GENOME_ID] == 2

    def test_every_filter_is_recorded_even_when_nothing_dropped(self, config: Config) -> None:
        _, log = run([raw_row()], config)
        reasons = {r.reason for r in log.records}
        for reason in (
            hz.Reason.EVIDENCE,
            hz.Reason.UNKNOWN_DRUG,
            hz.Reason.UNKNOWN_SPECIES,
            hz.Reason.UNKNOWN_METHOD,
            hz.Reason.UNKNOWN_UNIT,
            hz.Reason.NULL_STANDARD,
            hz.Reason.NULL_YEAR,
            hz.Reason.NO_TABLE_FOR_YEAR,
            hz.Reason.NO_BREAKPOINT,
            hz.Reason.COMBINATION,
            hz.Reason.SNAPPED,
            hz.Reason.DUP_SR,
            hz.Reason.DUP_FAR,
        ):
            assert reason in reasons
        assert all(r.n_dropped == 0 for r in log.records)


# --------------------------------------------------------------------------- #
# Method classification, disk rows, units, value parsing
# --------------------------------------------------------------------------- #


class TestMethod:
    @pytest.mark.parametrize(
        ("method_raw", "expected"),
        [
            ("Broth dilution", "dilution"),
            ("broth microdilution", "dilution"),
            ("Microdilution", "dilution"),
            ("Agar dilution", "dilution"),
            ("BROTH MICRODILUTION (Sensititre)", "dilution"),
            ("Etest", "gradient"),
            ("E-test", "gradient"),
            ("MIC gradient strip", "gradient"),
            ("Gradient diffusion", "gradient"),
            ("Disk diffusion", "disk"),
            ("disc diffusion", "disk"),
            ("Kirby-Bauer", "disk"),
            ("zone diameter", "disk"),
            ("Vitek 2", None),
            ("", None),
            (None, None),
        ],
    )
    def test_classify_method(self, method_raw: str | None, expected: str | None) -> None:
        assert hz.classify_method(method_raw) == expected

    def test_unknown_method_is_dropped(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", method_raw="Vitek 2"),
            raw_row(genome_id="b", method_raw=None),
            raw_row(genome_id="c", method_raw="Etest"),
        ]
        labels, log = run(rows, config)
        assert labels["genome_id"].tolist() == ["c"]
        assert only(labels)["method"] == "gradient"
        assert log_counts(log)[hz.Reason.UNKNOWN_METHOD] == 2

    def test_disk_row_with_a_value_uses_the_sir_path(self, config: Config) -> None:
        """A disk value is a zone diameter in mm, never an MIC."""
        labels, log = run(
            [raw_row(method_raw="Disk diffusion", sir_raw="S", sign="=", value="22", unit="mm")],
            config,
        )
        row = only(labels)
        assert row["method"] == "disk"
        assert interval(row) == (0.0, 2.0, "left")  # EUCAST meropenem S <= 2, not (16, 32]
        assert row["sir"] == "S"
        assert row["raw_result"].startswith("S") and "22" in row["raw_result"]
        # the 'mm' unit must not trigger the MIC unit filter on a disk row
        assert log_counts(log)[hz.Reason.UNKNOWN_UNIT] == 0

    def test_disk_row_without_sir_is_dropped(self, config: Config) -> None:
        labels, log = run(
            [raw_row(method_raw="Disk diffusion", sir_raw=None, sign="=", value="22", unit="mm")],
            config,
        )
        assert labels.empty
        assert log_counts(log)[hz.Reason.NO_RESULT] == 1

    def test_disk_row_needs_a_standard(self, config: Config) -> None:
        labels, log = run(
            [raw_row(method_raw="Disk diffusion", sir_raw="R", sign=None, value=None, standard=None)],
            config,
        )
        assert labels.empty
        assert log_counts(log)[hz.Reason.NULL_STANDARD] == 1


class TestUnitsAndValues:
    @pytest.mark.parametrize("unit", ["mg/L", "mg/l", "µg/mL", "μg/mL", "ug/mL", "ug/ml", "mcg/mL", "", None, "  mg/L "])
    def test_accepted_units(self, config: Config, unit: str | None) -> None:
        labels, log = run([raw_row(unit=unit)], config)
        assert interval(only(labels)) == (4.0, 8.0, "interval")
        assert log_counts(log)[hz.Reason.UNKNOWN_UNIT] == 0

    @pytest.mark.parametrize("unit", ["mm", "g/L", "mg/dL", "IU"])
    def test_rejected_units_drop_the_row(self, config: Config, unit: str) -> None:
        labels, log = run([raw_row(unit=unit)], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.UNKNOWN_UNIT] == 1

    @pytest.mark.parametrize("value", ["abc", "N/A", "0", "-2", "8/abc", "8/", "/4", "8/4/2", "0/4"])
    def test_bad_numeric_values_drop_the_row(self, config: Config, value: str) -> None:
        labels, log = run([raw_row(value=value)], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.BAD_VALUE] == 1

    @pytest.mark.parametrize(
        ("sign", "value", "expected", "raw_result"),
        [
            ("=", "16/4", (8.0, 16.0, "interval"), "=16/4"),
            (None, "16/4", (8.0, 16.0, "interval"), "=16/4"),
            (None, "<=8/4", (0.0, 8.0, "left"), "<=8/4"),
            ("<=", "8/4", (0.0, 8.0, "left"), "<=8/4"),
            (None, ">4/76", (4.0, INF, "right"), ">4/76"),
            ("=", "8 / 4", (4.0, 8.0, "interval"), "=8 / 4"),
        ],
    )
    def test_combination_mic_uses_the_primary_agent(
        self, config: Config, sign: str | None, value: str, expected: tuple, raw_result: str
    ) -> None:
        """Beta-lactam/inhibitor exports print ``piperacillin/tazobactam`` MICs as ``16/4``."""
        labels, log = run([raw_row(antibiotic_raw="TZP", sign=sign, value=value)], config)
        row = only(labels)
        assert row["drug"] == "piperacillin-tazobactam"
        assert interval(row) == expected
        assert row["raw_result"] == raw_result  # the full original text is kept for audit
        assert log_counts(log)[hz.Reason.COMBINATION] == 1
        assert log_counts(log)[hz.Reason.BAD_VALUE] == 0

    def test_combination_count_detail_and_plain_values(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", antibiotic_raw="TZP", value="16/4"),
            raw_row(genome_id="b", antibiotic_raw="TZP", value="16/4"),
            raw_row(genome_id="c", antibiotic_raw="TZP", value="16"),
        ]
        labels, log = run(rows, config)
        assert len(labels) == 3
        assert log_counts(log)[hz.Reason.COMBINATION] == 2
        detail = [r.detail for r in log.records if r.reason == hz.Reason.COMBINATION][0]
        assert "16/4 (2)" in detail

    @pytest.mark.parametrize(
        ("sign", "value", "expected"),
        [
            ("=", "0.016", (2.0**-7, 2.0**-6, "interval")),
            ("=", "0.008", (2.0**-8, 2.0**-7, "interval")),
            ("<=", "0.008", (0.0, 2.0**-7, "left")),
            (">", "0.03", (2.0**-5, INF, "right")),
            (">", "0.032", (2.0**-5, INF, "right")),
            (">=", "0.064", (2.0**-5, INF, "right")),
            (">", "0.12", (0.125, INF, "right")),
            ("=", 0.016, (2.0**-7, 2.0**-6, "interval")),  # numeric value, not a string
        ],
    )
    def test_decimal_renderings_of_powers_of_two_are_snapped(
        self, config: Config, sign: str, value: object, expected: tuple
    ) -> None:
        labels, log = run([raw_row(antibiotic_raw="ciprofloxacin", sign=sign, value=value)], config)
        row = only(labels)
        assert interval(row) == expected
        assert row["raw_result"] == f"{sign}{value}"  # the reported text is kept
        assert log_counts(log)[hz.Reason.SNAPPED] == 1

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("0.19", (0.125, 0.25)), ("0.75", (0.5, 1.0)), ("6", (4.0, 8.0)), ("0.047", (2.0**-5, 2.0**-4)),
         ("0.25", (0.125, 0.25)), ("8", (4.0, 8.0))],
    )
    def test_half_steps_and_exact_powers_are_not_snapped(
        self, config: Config, value: str, expected: tuple
    ) -> None:
        labels, log = run([raw_row(sign="=", value=value)], config)
        assert interval(only(labels)) == (*expected, "interval")
        assert log_counts(log)[hz.Reason.SNAPPED] == 0

    def test_snapped_count_has_a_value_detail(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", antibiotic_raw="ciprofloxacin", sign="<=", value="0.016"),
            raw_row(genome_id="b", antibiotic_raw="ciprofloxacin", sign="<=", value="0.016"),
            raw_row(genome_id="c", antibiotic_raw="ciprofloxacin", sign="=", value="0.12"),
        ]
        _, log = run(rows, config)
        record = [r for r in log.records if r.reason == hz.Reason.SNAPPED][0]
        assert record.n_dropped == 3
        assert "0.016 (2)" in record.detail and "0.12 (1)" in record.detail

    def test_unknown_sign_drops_the_row(self, config: Config) -> None:
        labels, log = run([raw_row(sign="~")], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.BAD_SIGN] == 1

    def test_no_value_and_no_sir_drops_the_row(self, config: Config) -> None:
        labels, log = run([raw_row(sign=None, value=None, sir_raw=None)], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.NO_RESULT] == 1

    def test_dilution_row_without_value_uses_sir_path(self, config: Config) -> None:
        labels, _ = run([raw_row(sign=None, value=None, sir_raw="R")], config)
        row = only(labels)
        assert interval(row) == (8.0, INF, "right")
        assert row["method"] == "dilution" and row["raw_result"] == "R"

    def test_numeric_row_keeps_reported_sir_as_is(self, config: Config) -> None:
        # Reported S/I/R is kept verbatim even when the MIC suggests otherwise
        # (contract: "as reported"; re-deriving is an open question).
        labels, _ = run([raw_row(sir_raw="Resistant", sign="=", value="8")], config)
        row = only(labels)
        assert row["sir"] == "R" and interval(row) == (4.0, 8.0, "interval")

    def test_numeric_row_with_unknown_sir_keeps_the_mic_and_nulls_sir(self, config: Config) -> None:
        labels, log = run([raw_row(sir_raw="SDD", sign="=", value="8")], config)
        row = only(labels)
        assert row["sir"] is None and interval(row) == (4.0, 8.0, "interval")
        assert log_counts(log).get(hz.Reason.UNKNOWN_SIR, 0) == 0


# --------------------------------------------------------------------------- #
# Biosample de-duplication and re-keying
# --------------------------------------------------------------------------- #


class TestBiosampleDedup:
    def test_ncbi_rows_are_rekeyed_to_the_bvbrc_genome_id(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="573.1", biosample="SAMN1", source="BVBRC", antibiotic_raw="meropenem"),
            raw_row(genome_id="NCBI_SAMN1", biosample="SAMN1", source="NCBI", antibiotic_raw="ciprofloxacin",
                    sign="<=", value="0.25"),
            raw_row(genome_id="NCBI_SAMN2", biosample="SAMN2", source="NCBI", antibiotic_raw="ciprofloxacin",
                    sign="<=", value="0.25"),
        ]
        labels, log = run(rows, config)
        assert sorted(labels["genome_id"].unique()) == ["573.1", "NCBI_SAMN2"]
        cipro = labels[(labels["genome_id"] == "573.1") & (labels["drug"] == "ciprofloxacin")]
        assert len(cipro) == 1 and cipro.iloc[0]["source"] == "NCBI"
        assert log_counts(log)[hz.Reason.REKEYED] == 1

    def test_rekeyed_rows_then_go_through_duplicate_resolution(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="573.1", biosample="SAMN1", source="BVBRC", sign="=", value="8"),
            raw_row(genome_id="NCBI_SAMN1", biosample="SAMN1", source="NCBI", sign="<=", value="8"),
        ]
        labels, log = run(rows, config)
        row = only(labels)
        assert row["genome_id"] == "573.1"
        assert interval(row) == (4.0, 8.0, "interval")  # intersection of (4,8] and (0,8]
        assert log_counts(log)[hz.Reason.REKEYED] == 1
        assert log_counts(log)[hz.Reason.DUP_MERGED] == 1

    def test_null_biosample_never_merges_genomes(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="573.1", biosample=None),
            raw_row(genome_id="NCBI_X", biosample=None, source="NCBI"),
        ]
        labels, log = run(rows, config)
        assert len(labels) == 2
        assert log_counts(log)[hz.Reason.REKEYED] == 0

    def test_two_bvbrc_ids_for_one_biosample_collapse_to_one(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="573.2", biosample="SAMN1"),
            raw_row(genome_id="573.1", biosample="SAMN1"),
        ]
        labels, log = run(rows, config)
        assert labels["genome_id"].nunique() == 1
        assert log_counts(log)[hz.Reason.REKEYED] == 1


# --------------------------------------------------------------------------- #
# Duplicate resolution on (genome_id, drug)
# --------------------------------------------------------------------------- #


class TestDuplicates:
    def test_identical_results_collapse_to_one_row(self, config: Config) -> None:
        labels, log = run([raw_row(), raw_row()], config)
        assert interval(only(labels)) == (4.0, 8.0, "interval")
        assert log_counts(log)[hz.Reason.DUP_MERGED] == 1

    def test_overlapping_intervals_take_the_intersection(self, config: Config) -> None:
        labels, _ = run([raw_row(sign="=", value="8"), raw_row(sign="<=", value="8")], config)
        assert interval(only(labels)) == (4.0, 8.0, "interval")

    def test_sir_only_s_and_left_censored_mic_intersect(self, config: Config) -> None:
        # S-only (0, 2] and <=0.25 (0, 0.25] agree; the tighter interval wins.
        labels, log = run(
            [raw_row(sir_raw="S", sign=None, value=None), raw_row(sign="<=", value="0.25")],
            config,
        )
        row = only(labels)
        assert interval(row) == (0.0, 0.25, "left")
        assert row["sir"] == "S"
        assert log_counts(log)[hz.Reason.DUP_SR] == 0

    def test_right_censored_pair_intersects_to_the_higher_edge(self, config: Config) -> None:
        labels, _ = run([raw_row(sign=">", value="16"), raw_row(sign=">", value="32")], config)
        assert interval(only(labels)) == (32.0, INF, "right")

    def test_adjacent_steps_keep_the_higher(self, config: Config) -> None:
        labels, log = run([raw_row(sign="=", value="8"), raw_row(sign="=", value="16")], config)
        row = only(labels)
        assert interval(row) == (8.0, 16.0, "interval")
        assert "=8" in row["raw_result"] and "=16" in row["raw_result"]
        assert log_counts(log)[hz.Reason.DUP_MERGED] == 1

    def test_adjacent_steps_keep_the_higher_sir(self, config: Config) -> None:
        labels, _ = run(
            [raw_row(sign="=", value="2", sir_raw="S"), raw_row(sign="=", value="4", sir_raw="I")],
            config,
        )
        row = only(labels)
        assert interval(row) == (2.0, 4.0, "interval") and row["sir"] == "I"

    def test_more_than_one_step_apart_drops_the_pair(self, config: Config) -> None:
        labels, log = run([raw_row(sign="=", value="8"), raw_row(sign="=", value="32")], config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.DUP_FAR] == 2  # both rows of the pair

    def test_three_readings_spanning_two_steps_drop_the_pair(self, config: Config) -> None:
        rows = [raw_row(sign="=", value="8"), raw_row(sign="=", value="16"), raw_row(sign="=", value="32")]
        labels, log = run(rows, config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.DUP_FAR] == 3

    def test_s_vs_r_drops_the_pair(self, config: Config) -> None:
        rows = [raw_row(sir_raw="S", sign=None, value=None), raw_row(sir_raw="R", sign=None, value=None)]
        labels, log = run(rows, config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.DUP_SR] == 2

    def test_s_vs_r_drops_even_when_intervals_are_adjacent(self, config: Config) -> None:
        # EUCAST ampicillin S == R == 8: S -> (0, 8], R -> (8, inf) are adjacent, but S vs R wins.
        rows = [
            raw_row(antibiotic_raw="ampicillin", sir_raw="S", sign=None, value=None),
            raw_row(antibiotic_raw="ampicillin", sir_raw="R", sign=None, value=None),
        ]
        labels, log = run(rows, config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.DUP_SR] == 2

    def test_reported_s_vs_r_on_numeric_rows_drops_the_pair(self, config: Config) -> None:
        rows = [raw_row(sign="=", value="4", sir_raw="S"), raw_row(sign="=", value="8", sir_raw="R")]
        labels, log = run(rows, config)
        assert labels.empty
        assert log_counts(log)[hz.Reason.DUP_SR] == 2

    def test_duplicates_are_scoped_to_genome_and_drug(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", antibiotic_raw="meropenem", sign="=", value="8"),
            raw_row(genome_id="a", antibiotic_raw="ciprofloxacin", sign="=", value="0.25"),
            raw_row(genome_id="b", antibiotic_raw="meropenem", sign="=", value="32"),
        ]
        labels, log = run(rows, config)
        assert len(labels) == 3
        assert log_counts(log)[hz.Reason.DUP_MERGED] == 0
        assert log_counts(log)[hz.Reason.DUP_FAR] == 0

    def test_a_conflict_in_one_drug_does_not_touch_other_drugs(self, config: Config) -> None:
        rows = [
            raw_row(antibiotic_raw="meropenem", sign="=", value="8"),
            raw_row(antibiotic_raw="meropenem", sign="=", value="32"),
            raw_row(antibiotic_raw="ciprofloxacin", sign="=", value="0.25"),
        ]
        labels, _ = run(rows, config)
        assert labels["drug"].tolist() == ["ciprofloxacin"]


class TestMergedRowProvenance:
    """The merged row's method / standard / standard_year / source describe a row that
    actually supports the merged bounds (finding #3)."""

    @staticmethod
    def disk_s() -> dict:
        # EUCAST 2024 ciprofloxacin S <= 0.25 -> (0, 0.25]
        return raw_row(antibiotic_raw="ciprofloxacin", method_raw="Disk diffusion", sir_raw="S", sign=None,
                       value="25", unit="mm", standard="EUCAST", standard_year=2024, source="BVBRC")

    @staticmethod
    def dilution(sign: str, value: str, **overrides: object) -> dict:
        row = {"standard": "CLSI", "standard_year": None, "source": "NCBI", "genome_id": "573.1"}
        row.update(overrides)
        return raw_row(antibiotic_raw="ciprofloxacin", method_raw="broth microdilution", sign=sign, value=value,
                       sir_raw=None, **row)

    @pytest.mark.parametrize("order", ["disk_first", "dilution_first"])
    def test_exact_mic_merged_with_disk_s_carries_the_dilution_provenance(self, config: Config, order: str) -> None:
        rows = [self.disk_s(), self.dilution("=", "0.125")]
        if order == "dilution_first":
            rows.reverse()
        labels, log = run(rows, config)
        row = only(labels)
        assert interval(row) == (0.0625, 0.125, "interval")
        assert row["method"] == "dilution"
        assert row["source"] == "NCBI" and row["standard"] == "CLSI"
        # provenance is taken as a block: the disk row's year is not borrowed
        assert pd.isna(row["standard_year"])
        assert row["sir"] == "S"  # the only reported category
        assert "S zone=25" in row["raw_result"] and "=0.125" in row["raw_result"]
        assert log_counts(log)[hz.Reason.DUP_MERGED] == 1

    def test_left_censored_mic_tighter_than_disk_s_wins(self, config: Config) -> None:
        labels, _ = run([self.disk_s(), self.dilution("<=", "0.03")], config)
        row = only(labels)
        assert interval(row) == (0.0, 0.03125, "left")
        assert row["method"] == "dilution" and row["source"] == "NCBI"

    def test_equal_intervals_prefer_an_mic_method_over_disk(self, config: Config) -> None:
        # Both rows give (0, 0.25]: the gradient row is preferred over the disk row.
        gradient = raw_row(antibiotic_raw="ciprofloxacin", method_raw="Etest", sign="<=", value="0.25",
                           standard="CLSI", standard_year=2024, source="NCBI")
        labels, _ = run([self.disk_s(), gradient], config)
        row = only(labels)
        assert interval(row) == (0.0, 0.25, "left")
        assert row["method"] == "gradient" and row["source"] == "NCBI" and row["standard"] == "CLSI"

    def test_disk_s_alone_with_a_wider_mic_keeps_the_disk_provenance(self, config: Config) -> None:
        # (0, 0.25] disk S  vs  (0, 1] '<=1': the merged interval is the disk row's own.
        labels, _ = run([self.disk_s(), self.dilution("<=", "1")], config)
        row = only(labels)
        assert interval(row) == (0.0, 0.25, "left")
        assert row["method"] == "disk" and row["standard"] == "EUCAST" and row["standard_year"] == 2024

    def test_partial_overlap_one_step_result_is_never_disk(self, config: Config) -> None:
        # disk S (0, 0.25] and '>0.125' (0.125, inf) intersect to (0.125, 0.25]: one step,
        # equal to neither row -> the MIC-bearing row describes it.
        labels, _ = run([self.disk_s(), self.dilution(">", "0.125")], config)
        row = only(labels)
        assert interval(row) == (0.125, 0.25, "interval")
        assert row["method"] == "dilution"

    def test_one_step_disk_i_and_equal_exact_mic_prefer_dilution(self, config: Config) -> None:
        # CLSI meropenem I is (1, 2]: one step, and '=2' gives the same interval.
        disk_i = raw_row(method_raw="Disk diffusion", sir_raw="I", sign=None, value=None, standard="CLSI",
                         standard_year=2024, source="BVBRC")
        mic_row = raw_row(sign="=", value="2", sir_raw=None, standard="EUCAST", standard_year=2024, source="NCBI")
        labels, _ = run([disk_i, mic_row], config)
        row = only(labels)
        assert interval(row) == (1.0, 2.0, "interval")
        assert row["method"] == "dilution" and row["source"] == "NCBI" and row["sir"] == "I"

    def test_adjacent_steps_take_provenance_from_the_higher_row(self, config: Config) -> None:
        rows = [
            raw_row(sign="=", value="16", standard="CLSI", standard_year=2024, source="BVBRC"),
            raw_row(sign="=", value="8", standard="EUCAST", standard_year=2024, source="NCBI"),
        ]
        labels, _ = run(rows, config)
        row = only(labels)
        assert interval(row) == (8.0, 16.0, "interval")
        assert row["standard"] == "CLSI" and row["source"] == "BVBRC"

    def test_mixed_categories_and_a_silent_representative_keep_the_highest_reported_sir(
        self, config: Config
    ) -> None:
        # S (disk) and I (gradient '<=0.5') disagree; the representative '=0.25' reported nothing.
        gradient_i = raw_row(antibiotic_raw="ciprofloxacin", method_raw="Etest", sign="<=", value="0.5",
                             sir_raw="I", source="BVBRC")
        labels, _ = run([self.disk_s(), gradient_i, self.dilution("=", "0.25")], config)
        row = only(labels)
        assert interval(row) == (0.125, 0.25, "interval")
        assert row["method"] == "dilution" and row["source"] == "NCBI"
        assert row["sir"] == "I"  # same as before the provenance fix: highest reported step

    def test_genome_metadata_is_still_filled_from_other_rows(self, config: Config) -> None:
        rows = [self.disk_s() | {"country": "USA", "year": 2019}, self.dilution("=", "0.125")]
        labels, _ = run(rows, config)
        row = only(labels)
        assert row["method"] == "dilution"
        assert row["country"] == "USA" and row["year"] == 2019

    def test_merged_exact_rows_are_disk_only_when_the_disk_row_alone_supports_them(self, config: Config) -> None:
        """Exhaustive small check over disk S/I/R x MIC results (EUCAST ciprofloxacin S<=0.25, R>0.5).

        A one-step merged interval is labelled ``disk`` only when it is exactly the disk
        ``I`` range (0.25, 0.5] and no MIC row reported that cell (e.g. '=0.25' is the
        adjacent lower step, '>0.125' does not tighten it). Whenever an MIC row reported
        the merged cell or contributed one of its bounds, the row is not ``disk``.
        """
        mic_results = [("=", "0.125"), ("=", "0.25"), ("<=", "0.125"), (">", "0.125"), (">=", "0.25"), ("=", "0.5")]
        rows = []
        for i, (sign, value) in enumerate(mic_results):
            for sir in ("S", "I", "R"):
                gid = f"g{i}_{sir}"
                rows.append(raw_row(genome_id=gid, antibiotic_raw="ciprofloxacin", method_raw="Disk diffusion",
                                    sir_raw=sir, sign=None, value=None, source="BVBRC"))
                rows.append(raw_row(genome_id=gid, antibiotic_raw="ciprofloxacin", sign=sign, value=value,
                                    sir_raw=None, source="NCBI", standard=None, standard_year=None))
        labels, _ = run(rows, config)
        exact = labels[(labels["mic_lower"] > 0) & (labels["mic_upper"] == 2 * labels["mic_lower"])]
        assert len(exact) >= 5
        disk = exact[exact["method"] == "disk"]
        assert set(disk["genome_id"]) == {"g1_I", "g3_I", "g4_I"}
        assert (disk["sir"] == "I").all() and (disk["mic_lower"] == 0.25).all() and (disk["mic_upper"] == 0.5).all()
        by = exact.set_index("genome_id")
        for gid in ("g0_S", "g1_S", "g3_S", "g4_S", "g5_S", "g5_I"):
            assert by.loc[gid, "method"] == "dilution", gid


# --------------------------------------------------------------------------- #
# Output schema, dtypes, ordering
# --------------------------------------------------------------------------- #


class TestOutputSchema:
    def test_columns_dtypes_nulls_and_sort_order(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="573.2", antibiotic_raw="meropenem", biosample="SAMN2", isolation_source="blood",
                    country="USA", year="2016"),
            raw_row(genome_id="573.1", antibiotic_raw="meropenem", biosample=None, isolation_source=None,
                    country=None, year=None, standard=None, standard_year=None),
            raw_row(genome_id="573.1", antibiotic_raw="ciprofloxacin", sign="<=", value="0.25", year=2019.0),
        ]
        labels, _ = run(rows, config)
        assert list(labels.columns) == list(LABEL_COLUMNS)
        assert labels.index.tolist() == [0, 1, 2]
        assert list(zip(labels["genome_id"], labels["drug"])) == [
            ("573.1", "ciprofloxacin"),
            ("573.1", "meropenem"),
            ("573.2", "meropenem"),
        ]
        assert labels["mic_lower"].dtype == np.float64 and labels["mic_upper"].dtype == np.float64
        assert str(labels["standard_year"].dtype) == "Int64" and str(labels["year"].dtype) == "Int64"
        for column in ("genome_id", "biosample", "species", "drug", "censor", "sir", "raw_result",
                       "method", "standard", "source", "isolation_source", "country"):
            assert labels[column].dtype == object, column
        by = labels.set_index(["genome_id", "drug"])
        row = by.loc[("573.1", "meropenem")]
        assert row["biosample"] is None and row["isolation_source"] is None and row["country"] is None
        assert row["standard"] is None and pd.isna(row["standard_year"]) and pd.isna(row["year"])
        row = by.loc[("573.2", "meropenem")]
        assert row["biosample"] == "SAMN2" and row["country"] == "USA" and row["year"] == 2016
        assert by.loc[("573.1", "ciprofloxacin"), "year"] == 2019
        # never an empty-string or sentinel for missing
        for column in LABEL_COLUMNS:
            values = labels[column].dropna().tolist()
            assert "" not in values and "NA" not in values and -1 not in values

    def test_empty_input_gives_empty_frame_with_schema(self, config: Config) -> None:
        labels, log = run([], config)
        assert list(labels.columns) == list(LABEL_COLUMNS) and labels.empty
        assert labels["mic_lower"].dtype == np.float64
        assert str(labels["standard_year"].dtype) == "Int64"
        assert len(log) > 0

    def test_missing_raw_column_raises(self, config: Config) -> None:
        raw = make_raw([raw_row()]).drop(columns=["unit"])
        with pytest.raises(ValueError, match="unit"):
            hz.harmonize(raw, config, DropLog("ingest"))

    def test_result_passes_acceptance_checks(self, config: Config) -> None:
        rows = [
            raw_row(genome_id="a", sign="=", value="8"),
            raw_row(genome_id="b", sign="<=", value="0.25", sir_raw="S"),
            raw_row(genome_id="c", sign=">", value="32", sir_raw="R"),
            raw_row(genome_id="d", sir_raw="I", sign=None, value=None),
        ]
        labels, _ = run(rows, config)
        hz.check_labels(labels, config)  # does not raise


# --------------------------------------------------------------------------- #
# Acceptance checks
# --------------------------------------------------------------------------- #


def good_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genome_id": pd.Series(["573.1", "573.1", "573.2"], dtype=object),
            "biosample": pd.Series([None, None, "SAMN2"], dtype=object),
            "species": pd.Series(["KPNEU"] * 3, dtype=object),
            "drug": pd.Series(["meropenem", "ciprofloxacin", "meropenem"], dtype=object),
            "mic_lower": np.array([4.0, 0.0, 32.0]),
            "mic_upper": np.array([8.0, 0.25, INF]),
            "censor": pd.Series(["interval", "left", "right"], dtype=object),
            "sir": pd.Series(["R", "S", None], dtype=object),
            "raw_result": pd.Series(["=8", "<=0.25", ">32"], dtype=object),
            "method": pd.Series(["dilution", "dilution", "gradient"], dtype=object),
            "standard": pd.Series(["CLSI", None, "EUCAST"], dtype=object),
            "standard_year": pd.array([2016, None, 2024], dtype="Int64"),
            "source": pd.Series(["BVBRC", "BVBRC", "NCBI"], dtype=object),
            "isolation_source": pd.Series(["blood", None, None], dtype=object),
            "country": pd.Series(["USA", None, None], dtype=object),
            "year": pd.array([2016, None, None], dtype="Int64"),
        }
    )


class TestAcceptanceChecks:
    def test_good_frame_passes(self) -> None:
        hz.check_labels(good_labels())

    def test_duplicate_key_raises(self) -> None:
        bad = pd.concat([good_labels(), good_labels().iloc[[0]]], ignore_index=True)
        with pytest.raises(ContractViolation, match="unique"):
            hz.check_labels(bad)

    def test_lower_not_below_upper_raises(self) -> None:
        bad = good_labels()
        bad.loc[0, "mic_lower"] = 8.0
        with pytest.raises(ContractViolation, match="mic_lower < mic_upper"):
            hz.check_labels(bad)

    def test_left_censor_iff_lower_is_zero(self) -> None:
        bad = good_labels()
        bad.loc[1, "mic_lower"] = 0.125  # censor says left but lower != 0
        with pytest.raises(ContractViolation, match="left"):
            hz.check_labels(bad)
        bad = good_labels()
        bad.loc[0, "mic_lower"] = 0.0  # lower == 0 but censor says interval
        with pytest.raises(ContractViolation, match="left"):
            hz.check_labels(bad)

    def test_right_censor_iff_upper_is_inf(self) -> None:
        bad = good_labels()
        bad.loc[2, "mic_upper"] = 64.0
        with pytest.raises(ContractViolation, match="right"):
            hz.check_labels(bad)
        bad = good_labels()
        bad.loc[0, "mic_upper"] = INF
        with pytest.raises(ContractViolation, match="right"):
            hz.check_labels(bad)

    def test_non_lab_evidence_raises_when_column_present(self) -> None:
        bad = good_labels().assign(evidence=["Laboratory Method", "Computational Prediction", "Laboratory Method"])
        with pytest.raises(ContractViolation, match="Laboratory Method"):
            hz.check_labels(bad)

    @pytest.mark.parametrize("column", ["genome_id", "species", "drug", "censor", "raw_result", "method", "source"])
    def test_null_in_required_column_raises(self, column: str) -> None:
        bad = good_labels()
        bad[column] = bad[column].astype(object)
        bad.loc[0, column] = None
        with pytest.raises(ContractViolation, match=column):
            hz.check_labels(bad)

    def test_bad_category_values_raise(self) -> None:
        bad = good_labels()
        bad.loc[0, "censor"] = "exact"
        with pytest.raises(ContractViolation, match="censor"):
            hz.check_labels(bad)
        bad = good_labels()
        bad.loc[0, "sir"] = "NS"
        with pytest.raises(ContractViolation, match="sir"):
            hz.check_labels(bad)
        bad = good_labels()
        bad.loc[0, "method"] = "vitek"
        with pytest.raises(ContractViolation, match="method"):
            hz.check_labels(bad)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ContractViolation, match="country"):
            hz.check_labels(good_labels().drop(columns=["country"]))

    def test_unknown_species_or_drug_against_config_raises(self, config: Config) -> None:
        bad = good_labels()
        bad.loc[0, "species"] = "SAUR"
        with pytest.raises(ContractViolation, match="species"):
            hz.check_labels(bad, config)
        bad = good_labels()
        bad.loc[0, "drug"] = "cefiderocol"
        with pytest.raises(ContractViolation, match="drug"):
            hz.check_labels(bad, config)


# --------------------------------------------------------------------------- #
# Count table and pair inclusion
# --------------------------------------------------------------------------- #


def label(genome_id: str, drug: str, lo: float, hi: float, sir: str | None, species: str = "KPNEU") -> dict:
    censor = "left" if lo == 0 else ("right" if math.isinf(hi) else "interval")
    return {
        "genome_id": genome_id,
        "biosample": None,
        "species": species,
        "drug": drug,
        "mic_lower": lo,
        "mic_upper": hi,
        "censor": censor,
        "sir": sir,
        "raw_result": "x",
        "method": "dilution",
        "standard": "EUCAST",
        "standard_year": 2024,
        "source": "BVBRC",
        "isolation_source": None,
        "country": None,
        "year": None,
    }


def labels_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    frame["standard_year"] = frame["standard_year"].astype("Int64")
    frame["year"] = frame["year"].astype("Int64")
    return frame


class TestCountTable:
    def test_counts_per_species_drug(self) -> None:
        rows = [
            label("g1", "meropenem", 4, 8, "R"),
            label("g2", "meropenem", 8, 16, "R"),
            label("g3", "meropenem", 0, 0.25, "S"),
            label("g4", "meropenem", 0, 2, "S"),  # S-only
            label("g5", "meropenem", 2, 8, "I"),
            label("g6", "meropenem", 32, INF, None),  # numeric, nothing reported
            label("g1", "ciprofloxacin", 0.125, 0.25, "S"),
            label("g1", "ceftriaxone", 0, 1, "S", species="ECOLI"),
        ]
        counts = hz.count_table(labels_frame(rows))
        assert list(counts.columns) == [
            "species", "drug", "n", "n_R", "n_S", "n_I", "n_exact", "n_censored", "n_distinct_mic",
        ]
        assert list(zip(counts["species"], counts["drug"])) == [
            ("ECOLI", "ceftriaxone"),
            ("KPNEU", "ciprofloxacin"),
            ("KPNEU", "meropenem"),
        ]
        mero = counts.set_index(["species", "drug"]).loc[("KPNEU", "meropenem")]
        assert mero["n"] == 6
        assert mero["n_R"] == 2 and mero["n_S"] == 2 and mero["n_I"] == 1
        # exact = one doubling step: (4,8] and (8,16]. The I-only (2,8] spans two steps,
        # so it is counted with the censored rows (n_censored = n - n_exact).
        assert mero["n_exact"] == 2 and mero["n_censored"] == 4
        # reported steps: 8, 16, 0.25, 2, 8 (I-only (2,8] reports 8), 64 (>32) -> 5 distinct
        assert mero["n_distinct_mic"] == 5
        cipro = counts.set_index(["species", "drug"]).loc[("KPNEU", "ciprofloxacin")]
        assert cipro["n"] == 1 and cipro["n_S"] == 1 and cipro["n_distinct_mic"] == 1

    def test_n_exact_uses_the_one_step_definition(self) -> None:
        rows = [
            label("g1", "meropenem", 1, 2, "I"),  # one-step I-only (CLSI): exact
            label("g2", "meropenem", 2, 8, "I"),  # two-step I-only (EUCAST): not exact
            label("g3", "meropenem", 2.0**-10, 16, "I"),  # placeholder S<=0.001 I range: not exact
            label("g4", "meropenem", 0.0625, 0.125, "S"),  # exact
        ]
        counts = hz.count_table(labels_frame(rows))
        row = counts.iloc[0]
        assert row["n_exact"] == 2 and row["n_censored"] == 2 and row["n"] == 4

    def test_disk_one_step_interval_is_not_an_exact_mic(self) -> None:
        """Contract method filter: disk diffusion gives S/I/R, never an MIC -- even a one-step I range."""
        disk_i = {**label("g1", "meropenem", 1, 2, "I"), "method": "disk", "standard": "CLSI"}
        rows = [disk_i, label("g2", "meropenem", 1, 2, "I"), label("g3", "meropenem", 4, 8, "R")]
        row = hz.count_table(labels_frame(rows)).iloc[0]
        assert row["n_exact"] == 2 and row["n_censored"] == 1 and row["n"] == 3

    def test_empty_labels_give_empty_counts_with_schema(self) -> None:
        counts = hz.count_table(labels_frame([]))
        assert counts.empty
        assert "n_distinct_mic" in counts.columns and "n_R" in counts.columns

    def test_count_dtypes_are_integers(self) -> None:
        counts = hz.count_table(labels_frame([label("g1", "meropenem", 4, 8, "R")]))
        for column in ("n", "n_R", "n_S", "n_I", "n_exact", "n_censored", "n_distinct_mic"):
            assert np.issubdtype(counts[column].dtype, np.integer), column


class TestPairsKept:
    @staticmethod
    def counts(n_r: int, n_s: int, n_i: int = 0, levels: int = 6) -> pd.DataFrame:
        n = n_r + n_s + n_i
        return pd.DataFrame(
            {
                "species": ["KPNEU"],
                "drug": ["meropenem"],
                "n": [n],
                "n_R": [n_r],
                "n_S": [n_s],
                "n_I": [n_i],
                "n_exact": [n],
                "n_censored": [0],
                "n_distinct_mic": [levels],
            }
        )

    def test_columns_and_nonsusceptible_definition(self) -> None:
        kept = hz.pairs_kept(self.counts(n_r=30, n_s=60, n_i=25))
        assert list(kept.columns) == ["species", "drug", "n", "n_nonsusceptible", "n_susceptible", "n_distinct_mic"]
        row = kept.iloc[0]
        assert row["n_nonsusceptible"] == 55 and row["n_susceptible"] == 60 and row["n"] == 115

    def test_thresholds_are_inclusive(self) -> None:
        assert len(hz.pairs_kept(self.counts(50, 50, levels=4))) == 1
        assert len(hz.pairs_kept(self.counts(49, 50, levels=4))) == 0
        assert len(hz.pairs_kept(self.counts(50, 49, levels=4))) == 0
        assert len(hz.pairs_kept(self.counts(50, 50, levels=3))) == 0

    def test_thresholds_are_parameters(self) -> None:
        assert len(hz.pairs_kept(self.counts(5, 5, levels=2), min_ns=5, min_s=5, min_levels=2)) == 1

    def test_multiple_pairs_and_ordering(self) -> None:
        counts = pd.concat(
            [
                self.counts(60, 60).assign(drug="meropenem"),
                self.counts(10, 60).assign(drug="ciprofloxacin"),
                self.counts(60, 60).assign(drug="ceftriaxone", species="ECOLI"),
            ],
            ignore_index=True,
        )
        kept = hz.pairs_kept(counts)
        assert list(zip(kept["species"], kept["drug"])) == [("ECOLI", "ceftriaxone"), ("KPNEU", "meropenem")]

    def test_empty_counts(self) -> None:
        kept = hz.pairs_kept(hz.count_table(labels_frame([])))
        assert kept.empty and "n_nonsusceptible" in kept.columns


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


class TestHelpers:
    @pytest.mark.parametrize(
        ("organism", "expected"),
        [
            ("Klebsiella pneumoniae", "KPNEU"),
            ("Klebsiella pneumoniae subsp. pneumoniae KPNIH1", "KPNEU"),
            ("  klebsiella PNEUMONIAE ", "KPNEU"),
            ("K. pneumoniae", "KPNEU"),
            ("Escherichia coli O157:H7 str. Sakai", "ECOLI"),
            ("Staphylococcus aureus", "SAUR"),
            ("Pseudomonas aeruginosa PAO1", "PAER"),
            ("Acinetobacter baumannii", "ABAU"),
            ("Klebsiella oxytoca", None),
            ("Klebsiella", None),
            ("", None),
            (None, None),
        ],
    )
    def test_species_key_from_organism_default_table(self, organism: str | None, expected: str | None) -> None:
        assert hz.species_key_from_organism(organism) == expected

    def test_species_key_from_organism_custom_table(self) -> None:
        names = {"KPNEU": "Klebsiella pneumoniae"}
        assert hz.species_key_from_organism("Escherichia coli", names) is None
        assert hz.species_key_from_organism("Klebsiella pneumoniae", names) == "KPNEU"

    @pytest.mark.parametrize(
        ("unit", "expected"),
        [("mg/L", "mg/L"), ("µg/mL", "mg/L"), ("ug/ml", "mg/L"), (None, "mg/L"), ("", "mg/L"), ("mm", None), ("g/L", None)],
    )
    def test_normalize_unit(self, unit: str | None, expected: str | None) -> None:
        assert hz.normalize_unit(unit) == expected

    @pytest.mark.parametrize(
        ("sign", "value", "expected"),
        [
            ("=", "8", ("=", 8.0)),
            (None, "8", ("=", 8.0)),
            (None, ">32", (">", 32.0)),
            (None, "<= 0.25", ("<=", 0.25)),
            (None, "≥16", (">=", 16.0)),
            ("<=", "0.25", ("<=", 0.25)),
            (">", 32, (">", 32.0)),
            (None, None, ("=", None)),
            (None, "", ("=", None)),
            # combination drugs: the first (primary agent) component
            ("=", "16/4", ("=", 16.0)),
            (None, "<=8/4", ("<=", 8.0)),
            (None, ">4/76", (">", 4.0)),
            ("=", "8 / 4", ("=", 8.0)),
            # the reported number is returned as is; snapping happens in the interval step
            ("=", "0.016", ("=", 0.016)),
        ],
    )
    def test_parse_measurement(self, sign: str | None, value: object, expected: tuple) -> None:
        assert hz.parse_measurement(sign, value) == expected

    @pytest.mark.parametrize("value", ["abc", "8/abc", "abc/4", "8/", "/4", "8/4/2", "0/4", "N/A", "0", "-1",
                                       "inf", "nan", "inf/4"])
    def test_parse_measurement_rejects_bad_values(self, value: str) -> None:
        with pytest.raises(ValueError):
            hz.parse_measurement("=", value)

    def test_parse_measurement_rejects_bad_sign(self) -> None:
        with pytest.raises(ValueError):
            hz.parse_measurement("~", "8")
        with pytest.raises(ValueError):
            hz.parse_measurement("<", ">8")  # conflicting signs


def test_module_has_no_prints(tmp_path: Path) -> None:
    source = Path(hz.__file__).read_text(encoding="utf-8")
    assert "print(" not in source
