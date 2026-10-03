"""Label harmonization: filters, cross-source de-duplication, and duplicate resolution."""

import math

import pandas as pd
import pytest

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.ingest.harmonize import LabelHarmonizer

LABEL_COLUMNS = [
    "genome_id",
    "biosample",
    "species",
    "drug",
    "mic_lower",
    "mic_upper",
    "censor",
    "sir",
    "raw_result",
    "method",
    "standard",
    "standard_year",
    "source",
    "isolation_source",
    "country",
    "year",
]

SPECIES_CONFIG = {"species": {"KPNEU": {"name": "Klebsiella pneumoniae"}, "ECOLI": {"name": "Escherichia coli"}}}
DRUG_CONFIG = {
    "drugs": {
        "meropenem": {"synonyms": ["mem"], "tier": 3},
        "ciprofloxacin": {"synonyms": [], "tier": 2},
        "piperacillin-tazobactam": {"synonyms": ["piperacillin-tazobactam"], "tier": 3},
    }
}


@pytest.fixture
def harmonizer() -> LabelHarmonizer:
    breakpoints = BreakpointTable(
        pd.DataFrame(
            [
                ("KPNEU", "meropenem", "CLSI", 2010, 1.0, 2.0, "M100-S20"),
                ("KPNEU", "ciprofloxacin", "CLSI", 2010, 1.0, 2.0, "M100-S20"),
            ],
            columns=["species", "drug", "standard", "effective_year", "s_breakpoint", "r_breakpoint", "version"],
        )
    )
    return LabelHarmonizer(SPECIES_CONFIG, DRUG_CONFIG, breakpoints)


def bvbrc_row(genome_id: str, antibiotic: str, phenotype: str, sign: str, value: str, method: str, **extra) -> dict:
    row = {
        "genome_id": genome_id,
        "antibiotic": antibiotic,
        "resistant_phenotype": phenotype,
        "measurement_sign": sign,
        "measurement_value": value,
        "measurement_unit": "mg/L",
        "laboratory_typing_method": method,
        "laboratory_typing_platform": "",
        "testing_standard": "CLSI",
        "testing_standard_year": 2016,
        "evidence": "Laboratory Method",
    }
    row.update(extra)
    return row


def ncbi_row(biosample: str, antibiotic: str, phenotype: str, sign: str, value: str, **extra) -> dict:
    row = {
        "biosample": biosample,
        "organism": "Klebsiella pneumoniae subsp. pneumoniae",
        "antibiotic": antibiotic,
        "resistance_phenotype": phenotype,
        "measurement_sign": sign,
        "measurement": value,
        "measurement_units": "mg/L",
        "laboratory_typing_method": "MIC",
        "laboratory_typing_platform": "Sensititre",
        "testing_standard": "CLSI",
    }
    row.update(extra)
    return row


@pytest.fixture
def raw_tables() -> dict[str, pd.DataFrame]:
    bvbrc_ast = pd.DataFrame(
        [
            bvbrc_row("573.1", "meropenem", "Resistant", "=", "4", "Broth dilution"),
            bvbrc_row("573.1", "ciprofloxacin", "Resistant", "", "", "Broth dilution", evidence="Computational Method"),
            bvbrc_row("573.2", "meropenem", "Susceptible", "", "25", "Disk diffusion"),
            bvbrc_row("573.2", "ciprofloxacin", "Susceptible", "", "", "Disk diffusion", testing_standard=""),
            bvbrc_row("573.3", "meropenem", "Susceptible", "", "1.5", "Etest"),
            bvbrc_row("573.3", "ciprofloxacin", "Susceptible", "<=", "0.06", "Broth dilution"),
            bvbrc_row("573.4", "piperacillin/tazobactam", "Resistant", ">", "64/4", "Broth dilution"),
            bvbrc_row("573.5", "meropenem", "Resistant", "=", "16", "Broth dilution"),
            bvbrc_row("573.6", "meropenem", "Resistant", "=", "32", "Broth dilution"),
            bvbrc_row("573.7", "gentamicin", "Resistant", "=", "32", "Broth dilution"),
        ]
    )
    ncbi_ast = pd.DataFrame(
        [
            ncbi_row("SAMN1", "meropenem", "resistant", "==", "8"),
            ncbi_row("SAMN9", "ciprofloxacin", "not defined", "<=", "0.25"),
        ]
    )
    bvbrc_meta = pd.DataFrame(
        [
            ("573.1", "Klebsiella pneumoniae", "SAMN1", 40, "Blood culture", "USA", 2016),
            ("573.2", "Klebsiella pneumoniae", "SAMN2", 50, "urine", "UK", 2021),
            ("573.3", "Klebsiella pneumoniae", "SAMN3", 50, None, None, None),
            ("573.4", "Klebsiella pneumoniae", "SAMN4", 50, "sputum", "India", 2019),
            # 573.5 and 573.6 are two assemblies of one biosample; 573.6 has fewer contigs and wins.
            ("573.5", "Klebsiella pneumoniae", "SAMN5", 300, "blood", "Thailand", 2020),
            ("573.6", "Klebsiella pneumoniae", "SAMN5", 30, "blood", "Thailand", 2020),
            ("573.7", "Klebsiella pneumoniae", "SAMN7", 50, "blood", "USA", 2020),
        ],
        columns=[
            "genome_id",
            "species",
            "biosample_accession",
            "contigs",
            "isolation_source",
            "isolation_country",
            "collection_year",
        ],
    )
    ncbi_meta = pd.DataFrame(
        [
            ("SAMN1", "Klebsiella pneumoniae", "2016-05-01", "USA:CA", "blood"),
            ("SAMN9", "Klebsiella pneumoniae", "2018", "Germany", "missing"),
        ],
        columns=["biosample", "organism", "collection_date", "geo_loc_name", "isolation_source"],
    )
    return {"bvbrc_ast": bvbrc_ast, "ncbi_ast": ncbi_ast, "bvbrc_meta": bvbrc_meta, "ncbi_meta": ncbi_meta}


@pytest.fixture
def result(harmonizer: LabelHarmonizer, raw_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return harmonizer.harmonize(**raw_tables)


def label(labels: pd.DataFrame, genome_id: str, drug: str) -> pd.Series:
    rows = labels[(labels["genome_id"] == genome_id) & (labels["drug"] == drug)]
    assert len(rows) == 1, f"expected one row for {genome_id} x {drug}, got {len(rows)}"
    return rows.iloc[0]


def test_output_matches_the_contract_columns(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, _ = result
    assert list(labels.columns) == LABEL_COLUMNS
    assert not labels.duplicated(subset=["genome_id", "drug"]).any()


def test_non_lab_rows_are_dropped(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, dropped = result
    assert labels[(labels["genome_id"] == "573.1") & (labels["drug"] == "ciprofloxacin")].empty
    assert "not_lab_method" in set(dropped["reason"])


def test_same_biosample_across_sources_merges_and_keeps_the_higher_reading(
    result: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    labels, _ = result
    row = label(labels, "573.1", "meropenem")
    assert (row["mic_lower"], row["mic_upper"], row["censor"]) == (4.0, 8.0, "interval")
    assert row["biosample"] == "SAMN1"
    assert labels[labels["genome_id"].str.startswith("NCBI_SAMN1")].empty


def test_ncbi_only_biosample_gets_an_ncbi_genome_id(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, _ = result
    row = label(labels, "NCBI_SAMN9", "ciprofloxacin")
    assert (row["mic_lower"], row["mic_upper"], row["censor"]) == (0.0, 0.25, "left")
    assert pd.isna(row["sir"])
    assert row["source"] == "NCBI"
    assert row["country"] == "Germany"
    assert row["year"] == 2018
    assert pd.isna(row["isolation_source"])


def test_disk_rows_use_the_sir_path_even_with_a_number(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, _ = result
    row = label(labels, "573.2", "meropenem")
    assert (row["mic_lower"], row["mic_upper"], row["censor"]) == (0.0, 1.0, "left")
    assert row["method"] == "disk"
    assert row["raw_result"] == "S"


def test_sir_only_row_without_standard_is_dropped(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, dropped = result
    assert labels[(labels["genome_id"] == "573.2") & (labels["drug"] == "ciprofloxacin")].empty
    assert "no_standard_for_sir_only" in set(dropped["reason"])


def test_off_grid_values_are_dropped_and_rounded_values_are_snapped(
    result: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    labels, dropped = result
    assert labels[(labels["genome_id"] == "573.3") & (labels["drug"] == "meropenem")].empty
    assert "off_grid_value" in set(dropped["reason"])
    row = label(labels, "573.3", "ciprofloxacin")
    assert (row["mic_lower"], row["mic_upper"], row["censor"]) == (0.0, 0.0625, "left")
    assert row["raw_result"] == "<=0.06"


def test_drug_names_and_combination_values_are_normalized(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, _ = result
    row = label(labels, "573.4", "piperacillin-tazobactam")
    assert (row["mic_lower"], row["mic_upper"], row["censor"]) == (64.0, math.inf, "right")
    assert row["isolation_source"] == "respiratory"


def test_drugs_outside_the_panel_are_dropped(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, dropped = result
    assert "gentamicin" not in set(labels["drug"])
    assert "drug_not_in_panel" in set(dropped["reason"])


def test_two_assemblies_of_one_biosample_collapse_to_the_best_assembly(
    result: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    labels, _ = result
    assert "573.5" not in set(labels["genome_id"])
    row = label(labels, "573.6", "meropenem")
    assert (row["mic_lower"], row["mic_upper"]) == (16.0, 32.0)


def test_metadata_is_normalized(result: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    labels, _ = result
    row = label(labels, "573.1", "meropenem")
    assert (row["isolation_source"], row["country"], row["year"]) == ("blood", "USA", 2016)


def duplicate_rows(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "genome_id": "g1",
                "drug": "meropenem",
                "mic_lower": lower,
                "mic_upper": upper,
                "sir": sir,
                "raw_result": raw,
                "source": "BVBRC",
            }
            for lower, upper, sir, raw in rows
        ]
    )


def test_overlapping_readings_take_the_intersection() -> None:
    rows = duplicate_rows((0.0, 0.5, "S", "<=0.5"), (0.25, 0.5, "S", "=0.5"))
    resolved, dropped = LabelHarmonizer.resolve_duplicates(rows)
    assert dropped.empty
    assert (resolved.iloc[0]["mic_lower"], resolved.iloc[0]["mic_upper"]) == (0.25, 0.5)
    assert resolved.iloc[0]["censor"] == "interval"
    assert resolved.iloc[0]["raw_result"] == "<=0.5|=0.5"


def test_readings_one_step_apart_keep_the_higher() -> None:
    resolved, dropped = LabelHarmonizer.resolve_duplicates(duplicate_rows((2.0, 4.0, "S", "=4"), (4.0, 8.0, "I", "=8")))
    assert dropped.empty
    assert (resolved.iloc[0]["mic_lower"], resolved.iloc[0]["mic_upper"]) == (4.0, 8.0)
    assert resolved.iloc[0]["sir"] == "I"


def test_right_censored_beats_the_step_below() -> None:
    rows = duplicate_rows((16.0, 32.0, "R", "=32"), (32.0, math.inf, "R", ">32"))
    resolved, _ = LabelHarmonizer.resolve_duplicates(rows)
    assert (resolved.iloc[0]["mic_lower"], resolved.iloc[0]["mic_upper"]) == (32.0, math.inf)
    assert resolved.iloc[0]["censor"] == "right"


@pytest.mark.parametrize(
    "rows",
    [
        ((2.0, 4.0, "I", "=4"), (8.0, 16.0, "R", "=16")),
        ((0.0, 0.25, None, "<=0.25"), (0.5, 1.0, None, "=1")),
    ],
)
def test_readings_more_than_one_step_apart_are_dropped(rows: tuple) -> None:
    resolved, dropped = LabelHarmonizer.resolve_duplicates(duplicate_rows(*rows))
    assert resolved.empty
    assert list(dropped["reason"]) == ["conflict_gt_1_step"]


def test_s_versus_r_is_dropped_even_when_mics_agree() -> None:
    rows = duplicate_rows((1.0, 2.0, "S", "=2"), (1.0, 2.0, "R", "=2"))
    resolved, dropped = LabelHarmonizer.resolve_duplicates(rows)
    assert resolved.empty
    assert list(dropped["reason"]) == ["conflict_s_vs_r"]


def test_count_pairs_applies_the_inclusion_rule() -> None:
    rows = []
    for index in range(60):
        rows.append(("g_s_" + str(index), "S", 0.0, 0.25, "<=0.25"))
        rows.append(("g_r_" + str(index), "R", 8.0, 16.0, "=16"))
    for index, upper in enumerate([0.5, 1.0, 2.0, 4.0]):
        rows.append(("g_x_" + str(index), None, upper / 2, upper, "=" + str(upper)))
    labels = pd.DataFrame(rows, columns=["genome_id", "sir", "mic_lower", "mic_upper", "raw_result"])
    labels["species"] = "KPNEU"
    labels["drug"] = "meropenem"
    labels["censor"] = ["left" if lower == 0 else "interval" for lower in labels["mic_lower"]]
    counts = LabelHarmonizer.count_pairs(labels)
    row = counts.iloc[0]
    assert (row["n_S"], row["n_R"], row["n_nonsusceptible"]) == (60, 60, 60)
    assert row["n_exact"] == 64
    assert row["n_distinct_mic"] == 5
    assert bool(row["kept"])
