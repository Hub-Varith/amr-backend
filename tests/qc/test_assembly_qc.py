"""Genome QC rules from DATA_CONTRACT.md stage 3."""

import pandas as pd
import pytest

from genome2mic.qc.assembly_qc import AssemblyQc

QC_COLUMNS = [
    "genome_id",
    "n_contigs",
    "total_length",
    "n50",
    "gc_percent",
    "mash_species",
    "mash_distance",
    "qc_pass",
    "qc_fail_reason",
]
EXPECTED_SIZES = {"KPNEU": 5_600_000, "ECOLI": 5_100_000}


def run_qc(stats_rows: list[tuple], mash_rows: list[tuple]) -> pd.DataFrame:
    genome_ids = sorted({row[0] for row in stats_rows} | {"g_missing_stats"})
    manifest = pd.DataFrame({"genome_id": genome_ids, "species": "KPNEU"})
    stats = pd.DataFrame(stats_rows, columns=["genome_id", "n_contigs", "total_length", "n50", "gc_percent"])
    mash = pd.DataFrame(mash_rows, columns=["genome_id", "reference_species", "distance"])
    return AssemblyQc(EXPECTED_SIZES).evaluate(manifest, stats, mash).set_index("genome_id")


@pytest.fixture
def qc() -> pd.DataFrame:
    return run_qc(
        stats_rows=[
            ("g_good", 120, 5_500_000, 200_000, 57.1),
            ("g_fragmented", 600, 5_500_000, 20_000, 57.0),
            ("g_small", 80, 4_000_000, 150_000, 57.0),
            ("g_ecoli", 100, 5_200_000, 150_000, 50.7),
            ("g_distant", 100, 5_500_000, 150_000, 57.0),
            ("g_two_problems", 900, 7_000_000, 9_000, 57.0),
            ("g_empty", 0, 0, 0, 0.0),
            ("g_no_mash", 100, 5_500_000, 150_000, 57.0),
        ],
        mash_rows=[
            ("g_good", "KPNEU", 0.011),
            ("g_good", "ECOLI", 0.21),
            ("g_fragmented", "KPNEU", 0.012),
            ("g_small", "KPNEU", 0.013),
            ("g_ecoli", "KPNEU", 0.20),
            ("g_ecoli", "ECOLI", 0.008),
            ("g_distant", "KPNEU", 0.07),
            ("g_two_problems", "KPNEU", 0.01),
        ],
    )


def test_output_matches_the_contract_columns(qc: pd.DataFrame) -> None:
    assert list(qc.reset_index().columns) == QC_COLUMNS


def test_good_genome_passes_with_nearest_reference(qc: pd.DataFrame) -> None:
    row = qc.loc["g_good"]
    assert bool(row["qc_pass"])
    assert pd.isna(row["qc_fail_reason"])
    assert (row["mash_species"], row["mash_distance"]) == ("KPNEU", 0.011)


@pytest.mark.parametrize(
    ("genome_id", "reason"),
    [
        ("g_fragmented", "too_fragmented"),
        ("g_small", "wrong_genome_size"),
        ("g_ecoli", "species_mismatch"),
        ("g_distant", "too_distant"),
        ("g_two_problems", "too_fragmented;wrong_genome_size"),
        ("g_empty", "download_failed"),
        ("g_missing_stats", "download_failed"),
        ("g_no_mash", "no_mash_result"),
    ],
)
def test_failures_carry_every_reason(qc: pd.DataFrame, genome_id: str, reason: str) -> None:
    row = qc.loc[genome_id]
    assert not bool(row["qc_pass"])
    assert row["qc_fail_reason"] == reason


def test_failed_download_has_null_measurements(qc: pd.DataFrame) -> None:
    row = qc.loc["g_empty"]
    assert row["n_contigs"] == 0
    assert pd.isna(row["mash_species"])
    assert pd.isna(row["gc_percent"])
