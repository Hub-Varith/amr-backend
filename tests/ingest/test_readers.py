"""Tests for the raw AST readers (bvbrc, ncbi_ast) and the ingest stage ``run``."""

from __future__ import annotations

import logging
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic import ingest
from genome2mic.config import Config
from genome2mic.droplog import DropLog
from genome2mic.ingest import bvbrc, harmonize as hz, ncbi_ast
from genome2mic.paths import Paths
from tests.ingest import LABEL_COLUMNS, RAW_COLUMNS, make_config, write_configs


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    return make_config(tmp_path_factory.mktemp("configs"))


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


BVBRC_CSV = """
    genome_id,genome_name,antibiotic,resistant_phenotype,measurement_sign,measurement_value,measurement_unit,laboratory_typing_method,testing_standard,testing_standard_year,evidence
    573.2002,Klebsiella pneumoniae subsp. pneumoniae KPNIH1,meropenem,Resistant,=,8,mg/L,Broth dilution,CLSI,2016,Laboratory Method
    573.2005,Klebsiella pneumoniae,meropenem,Resistant,>,32,mg/L,Broth dilution,CLSI,2019,Laboratory Method
    573.2011,Klebsiella pneumoniae,meropenem,Susceptible,,,,Disk diffusion,EUCAST,2024,Laboratory Method
    573.2002,Klebsiella pneumoniae subsp. pneumoniae KPNIH1,ciprofloxacin,Resistant,=,1,mg/L,Broth dilution,CLSI,2016,Laboratory Method
    562.1,Escherichia coli O157:H7,meropenem,Susceptible,<=,0.25,µg/mL,Broth microdilution,EUCAST,,Computational Prediction
    999.1,Klebsiella oxytoca,meropenem,Susceptible,<=,0.25,mg/L,Broth microdilution,EUCAST,2020,Laboratory Method
"""

METADATA_CSV = """
    genome_id,biosample,species,source,isolation_source,country,year
    573.2002,SAMN00000001,KPNEU,BVBRC,blood,USA,2016
    573.2005,SAMN00000002,KPNEU,BVBRC,blood,India,2019
    573.2011,SAMN00000004,KPNEU,BVBRC,,UK,2021
    NCBI_SAMN00000003,SAMN00000003,KPNEU,NCBI,urine,Thailand,2020
"""

NCBI_CSV = """
    biosample,organism,antibiotic,resistance_phenotype,measurement_sign,measurement,measurement_units,laboratory_typing_method,testing_standard,isolation_source,geo_loc_name,collection_date
    SAMN00000002,Klebsiella pneumoniae,Meropenem,resistant,>,32,mg/L,Broth dilution,CLSI,blood,India: Delhi,2019-05-01
    SAMN00000003,Klebsiella pneumoniae,meropenem,susceptible,<=,0.25,mg/L,Broth dilution,CLSI,urine,Thailand,2020
    SAMN00000004,Klebsiella pneumoniae,MEM,susceptible,,,,Disk diffusion,EUCAST,blood,"USA: California, Los Angeles",2021-03
    SAMN00000005,Escherichia coli,ciprofloxacin,,,0.5,mg/L,Etest,,,missing,not collected
    SAMN00000006,Enterococcus faecium,vancomycin,resistant,>,32,mg/L,Broth dilution,CLSI,,,
"""


# --------------------------------------------------------------------------- #
# BV-BRC reader
# --------------------------------------------------------------------------- #


class TestBvbrcReader:
    def test_returns_common_raw_schema(self, tmp_path: Path) -> None:
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", BVBRC_CSV))
        assert list(raw.columns) == list(RAW_COLUMNS)
        assert len(raw) == 6
        first = raw.iloc[0]
        assert first["genome_id"] == "573.2002"
        assert first["species"] == "KPNEU"
        assert first["antibiotic_raw"] == "meropenem"  # not normalized by the reader
        assert first["sir_raw"] == "Resistant"
        assert first["sign"] == "=" and first["value"] == "8" and first["unit"] == "mg/L"
        assert first["method_raw"] == "Broth dilution"
        assert first["standard"] == "CLSI" and first["standard_year"] == 2016
        assert first["evidence"] == "Laboratory Method"
        assert first["source"] == "BVBRC"
        assert raw["source"].eq("BVBRC").all()

    def test_blanks_become_none_not_empty_strings(self, tmp_path: Path) -> None:
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", BVBRC_CSV))
        disk = raw[raw["genome_id"] == "573.2011"].iloc[0]
        assert disk["sign"] is None and disk["value"] is None and disk["unit"] is None
        ecoli = raw[raw["genome_id"] == "562.1"].iloc[0]
        assert pd.isna(ecoli["standard_year"])
        assert ecoli["unit"] == "µg/mL"
        for column in ("biosample", "isolation_source", "country"):
            assert raw[column].isna().all(), column
        for column in RAW_COLUMNS:
            assert "" not in raw[column].dropna().tolist(), column

    def test_species_mapping_and_unknown_organism(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="genome2mic.ingest.bvbrc"):
            raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", BVBRC_CSV))
        by = raw.drop_duplicates("genome_id").set_index("genome_id")["species"]
        assert by["573.2002"] == "KPNEU" and by["562.1"] == "ECOLI"
        assert by["999.1"] is None  # Klebsiella oxytoca: not in scope, left for harmonize to drop
        assert any("Klebsiella oxytoca" in r.getMessage() for r in caplog.records)

    def test_evidence_passes_through_unfiltered(self, tmp_path: Path) -> None:
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", BVBRC_CSV))
        assert "Computational Prediction" in raw["evidence"].tolist()  # harmonize filters, not the reader

    def test_column_aliases_and_optional_columns(self, tmp_path: Path) -> None:
        csv = """
            Genome ID,Genome Name,Antibiotic,Resistance Phenotype,Measurement Sign,Measurement,Measurement Units,Lab Typing Method,Testing Standard,Evidence,BioSample Accession,Isolation Source,Isolation Country,Collection Year
            573.2002,Klebsiella pneumoniae,meropenem,Resistant,=,8,mg/L,Broth dilution,CLSI,Laboratory Method,SAMN1,blood,USA,2016
        """
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", csv))
        row = raw.iloc[0]
        assert row["genome_id"] == "573.2002" and row["species"] == "KPNEU"
        assert row["value"] == "8" and row["unit"] == "mg/L" and row["method_raw"] == "Broth dilution"
        assert row["biosample"] == "SAMN1" and row["isolation_source"] == "blood"
        assert row["country"] == "USA" and row["year"] == 2016
        assert pd.isna(row["standard_year"])

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        csv = """
            genome_name,antibiotic,resistant_phenotype
            Klebsiella pneumoniae,meropenem,Resistant
        """
        with pytest.raises(ValueError, match="genome_id"):
            bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", csv))

    def test_metadata_fills_biosample_species_and_evaluation_columns(self, tmp_path: Path) -> None:
        metadata = ingest.read_genome_metadata(write(tmp_path / "genome_metadata.csv", METADATA_CSV))
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", BVBRC_CSV), metadata=metadata)
        by = raw.drop_duplicates("genome_id").set_index("genome_id")
        assert by.loc["573.2002", "biosample"] == "SAMN00000001"
        assert by.loc["573.2002", "country"] == "USA" and by.loc["573.2002", "year"] == 2016
        assert by.loc["573.2002", "isolation_source"] == "blood"
        assert by.loc["573.2011", "isolation_source"] is None  # blank in metadata stays null
        assert by.loc["562.1", "biosample"] is None  # not in metadata
        assert by.loc["999.1", "species"] is None  # metadata has no row; organism unknown

    def test_metadata_species_only_fills_when_organism_is_unknown(self, tmp_path: Path) -> None:
        csv = """
            genome_id,genome_name,antibiotic,resistant_phenotype,measurement_sign,measurement_value,measurement_unit,laboratory_typing_method,testing_standard,testing_standard_year,evidence
            573.2002,,meropenem,Resistant,=,8,mg/L,Broth dilution,CLSI,2016,Laboratory Method
        """
        metadata = ingest.read_genome_metadata(write(tmp_path / "genome_metadata.csv", METADATA_CSV))
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", csv), metadata=metadata)
        assert raw.iloc[0]["species"] == "KPNEU"

    def test_custom_species_names(self, tmp_path: Path, config: Config) -> None:
        names = {key: spec.name for key, spec in config.species.items()}  # KPNEU + ECOLI only
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", BVBRC_CSV), species_names=names)
        assert set(raw["species"].dropna()) == {"KPNEU", "ECOLI"}

    def test_dtypes(self, tmp_path: Path) -> None:
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", BVBRC_CSV))
        assert str(raw["standard_year"].dtype) == "Int64" and str(raw["year"].dtype) == "Int64"
        assert raw["value"].dtype == object and raw["genome_id"].dtype == object

    def test_empty_file_with_header(self, tmp_path: Path) -> None:
        csv = "genome_id,genome_name,antibiotic,resistant_phenotype\n"
        raw = bvbrc.read_raw(write(tmp_path / "ast_bvbrc.csv", csv))
        assert raw.empty and list(raw.columns) == list(RAW_COLUMNS)


# --------------------------------------------------------------------------- #
# NCBI reader
# --------------------------------------------------------------------------- #


class TestNcbiReader:
    def test_returns_common_raw_schema(self, tmp_path: Path) -> None:
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", NCBI_CSV))
        assert list(raw.columns) == list(RAW_COLUMNS)
        assert len(raw) == 5
        first = raw.iloc[0]
        assert first["genome_id"] == "NCBI_SAMN00000002"
        assert first["biosample"] == "SAMN00000002"
        assert first["species"] == "KPNEU"
        assert first["antibiotic_raw"] == "Meropenem"
        assert first["sir_raw"] == "resistant"
        assert first["sign"] == ">" and first["value"] == "32" and first["unit"] == "mg/L"
        assert first["method_raw"] == "Broth dilution" and first["standard"] == "CLSI"
        assert pd.isna(first["standard_year"])  # NCBI has no standard-year column
        assert first["evidence"] == "Laboratory Method"  # lab by definition
        assert first["source"] == "NCBI"
        assert first["isolation_source"] == "blood"

    @pytest.mark.parametrize(
        ("biosample", "country", "year"),
        [
            ("SAMN00000002", "India", 2019),
            ("SAMN00000003", "Thailand", 2020),
            ("SAMN00000004", "USA", 2021),
            ("SAMN00000005", None, None),  # 'missing' / 'not collected'
            ("SAMN00000006", None, None),
        ],
    )
    def test_country_and_year_parsing(self, tmp_path: Path, biosample: str, country: str | None, year: int | None) -> None:
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", NCBI_CSV)).set_index("biosample")
        row = raw.loc[biosample]
        if country is None:
            assert row["country"] is None
        else:
            assert row["country"] == country
        if year is None:
            assert pd.isna(row["year"])
        else:
            assert row["year"] == year

    def test_unknown_organism_gives_null_species(self, tmp_path: Path) -> None:
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", NCBI_CSV)).set_index("biosample")
        assert raw.loc["SAMN00000006", "species"] is None
        assert raw.loc["SAMN00000005", "species"] == "ECOLI"

    def test_blanks_become_none(self, tmp_path: Path) -> None:
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", NCBI_CSV)).set_index("biosample")
        row = raw.loc["SAMN00000005"]
        assert row["sir_raw"] is None and row["sign"] is None and row["standard"] is None
        assert row["isolation_source"] is None
        for column in RAW_COLUMNS:
            if column == "biosample":
                continue
            assert "" not in raw[column].dropna().tolist(), column

    def test_metadata_rekeys_to_bvbrc_genome_id(self, tmp_path: Path) -> None:
        metadata = ingest.read_genome_metadata(write(tmp_path / "genome_metadata.csv", METADATA_CSV))
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", NCBI_CSV), metadata=metadata).set_index("biosample")
        assert raw.loc["SAMN00000002", "genome_id"] == "573.2005"  # contract: prefer the BV-BRC id
        assert raw.loc["SAMN00000004", "genome_id"] == "573.2011"
        assert raw.loc["SAMN00000003", "genome_id"] == "NCBI_SAMN00000003"
        assert raw.loc["SAMN00000005", "genome_id"] == "NCBI_SAMN00000005"  # not in metadata
        # metadata never overrides values the raw file already has
        assert raw.loc["SAMN00000002", "country"] == "India"

    def test_ncbi_browser_style_headers(self, tmp_path: Path) -> None:
        csv = """
            #BioSample,Scientific name,Antibiotic,Resistance phenotype,Measurement sign,Measurement,Measurement units,Laboratory typing method,Testing standard,Isolation source,Location,Collection date
            SAMN1,Klebsiella pneumoniae,meropenem,resistant,=,8,mg/L,Broth dilution,CLSI,blood,USA,2016
        """
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", csv))
        row = raw.iloc[0]
        assert row["genome_id"] == "NCBI_SAMN1" and row["species"] == "KPNEU"
        assert row["country"] == "USA" and row["year"] == 2016
        assert row["value"] == "8" and row["method_raw"] == "Broth dilution"

    def test_optional_standard_year_and_evidence_columns(self, tmp_path: Path) -> None:
        csv = """
            biosample,organism,antibiotic,resistance_phenotype,measurement_sign,measurement,measurement_units,laboratory_typing_method,testing_standard,testing_standard_year,evidence
            SAMN1,Klebsiella pneumoniae,meropenem,resistant,=,8,mg/L,Broth dilution,CLSI,2019,Laboratory Method
        """
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", csv))
        row = raw.iloc[0]
        assert row["standard_year"] == 2019 and row["evidence"] == "Laboratory Method"

    def test_missing_biosample_gives_null_genome_id(self, tmp_path: Path) -> None:
        csv = """
            biosample,organism,antibiotic,resistance_phenotype,measurement_sign,measurement,measurement_units,laboratory_typing_method,testing_standard
            ,Klebsiella pneumoniae,meropenem,resistant,=,8,mg/L,Broth dilution,CLSI
        """
        raw = ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", csv))
        assert raw.iloc[0]["genome_id"] is None and raw.iloc[0]["biosample"] is None

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        csv = "organism,antibiotic\nKlebsiella pneumoniae,meropenem\n"
        with pytest.raises(ValueError, match="biosample"):
            ncbi_ast.read_raw(write(tmp_path / "ast_ncbi.csv", csv))


class TestMetadata:
    def test_read_genome_metadata(self, tmp_path: Path) -> None:
        meta = ingest.read_genome_metadata(write(tmp_path / "genome_metadata.csv", METADATA_CSV))
        assert list(meta.columns) == ["genome_id", "biosample", "species", "source", "isolation_source", "country", "year"]
        assert str(meta["year"].dtype) == "Int64"
        assert meta.set_index("genome_id").loc["573.2011", "isolation_source"] is None

    def test_missing_columns_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="biosample"):
            ingest.read_genome_metadata(write(tmp_path / "genome_metadata.csv", "genome_id,species\n573.1,KPNEU\n"))


# --------------------------------------------------------------------------- #
# Stage run
# --------------------------------------------------------------------------- #


def make_root(tmp_path: Path, *, bvbrc_csv: str | None = BVBRC_CSV, ncbi_csv: str | None = NCBI_CSV,
              metadata_csv: str | None = METADATA_CSV) -> Paths:
    root = tmp_path / "run"
    configs = write_configs(root)
    if bvbrc_csv is not None:
        write(root / "data" / "raw" / "ast_bvbrc.csv", bvbrc_csv)
    if ncbi_csv is not None:
        write(root / "data" / "raw" / "ast_ncbi.csv", ncbi_csv)
    if metadata_csv is not None:
        write(root / "data" / "raw" / "genome_metadata.csv", metadata_csv)
    return Paths(root=root, configs_dir=configs)


class TestRun:
    def test_writes_all_outputs(self, tmp_path: Path, config: Config) -> None:
        paths = make_root(tmp_path)
        labels = ingest.run(paths, config)
        for path in (paths.labels, paths.label_counts, paths.pairs_kept, paths.drop_log("ingest")):
            assert path.exists(), path
        assert list(labels.columns) == list(LABEL_COLUMNS)
        back = pd.read_parquet(paths.labels)
        assert list(back.columns) == list(LABEL_COLUMNS)
        assert len(back) == len(labels)
        assert back["mic_lower"].dtype == np.float64
        assert str(back["standard_year"].dtype) == "Int64"
        hz.check_labels(back, config)

    def test_content_of_the_joined_sources(self, tmp_path: Path, config: Config) -> None:
        paths = make_root(tmp_path)
        labels = ingest.run(paths, config).set_index(["genome_id", "drug"])
        # 573.2002 meropenem: =8 CLSI, lab -> (4, 8]
        assert tuple(labels.loc[("573.2002", "meropenem"), ["mic_lower", "mic_upper"]]) == (4.0, 8.0)
        assert labels.loc[("573.2002", "meropenem"), "biosample"] == "SAMN00000001"  # from metadata
        assert labels.loc[("573.2002", "meropenem"), "country"] == "USA"
        # 573.2005: BV-BRC >32 and NCBI (SAMN00000002, re-keyed) >32 agree -> one row
        assert tuple(labels.loc[("573.2005", "meropenem"), ["mic_lower", "mic_upper"]]) == (32.0, math.inf)
        assert "NCBI_SAMN00000002" not in labels.index.get_level_values("genome_id")
        # 573.2011: disk S under EUCAST 2024 (S <= 2) from BV-BRC. The NCBI copy of the same
        # disk result has no standard_year, so it is dropped (no table can be matched) and
        # only the BV-BRC row survives.
        assert tuple(labels.loc[("573.2011", "meropenem"), ["mic_lower", "mic_upper"]]) == (0.0, 2.0)
        assert labels.loc[("573.2011", "meropenem"), "method"] == "disk"
        assert labels.loc[("573.2011", "meropenem"), "source"] == "BVBRC"
        assert labels.loc[("573.2011", "meropenem"), "standard_year"] == 2024
        # NCBI-only genome keeps its NCBI id
        assert tuple(labels.loc[("NCBI_SAMN00000003", "meropenem"), ["mic_lower", "mic_upper"]]) == (0.0, 0.25)
        # computational-prediction row, unknown species and unknown drug are gone
        assert "562.1" not in labels.index.get_level_values("genome_id")
        assert "999.1" not in labels.index.get_level_values("genome_id")
        assert "NCBI_SAMN00000006" not in labels.index.get_level_values("genome_id")
        # ECOLI ciprofloxacin =0.5 from NCBI, null standard but numeric -> kept with null standard
        assert tuple(labels.loc[("NCBI_SAMN00000005", "ciprofloxacin"), ["mic_lower", "mic_upper"]]) == (0.25, 0.5)
        assert labels.loc[("NCBI_SAMN00000005", "ciprofloxacin"), "standard"] is None or pd.isna(
            labels.loc[("NCBI_SAMN00000005", "ciprofloxacin"), "standard"]
        )

    def test_count_table_and_pairs_kept_files(self, tmp_path: Path, config: Config) -> None:
        paths = make_root(tmp_path)
        ingest.run(paths, config)
        counts = pd.read_csv(paths.label_counts)
        assert list(counts.columns) == ["species", "drug", "n", "n_R", "n_S", "n_I", "n_exact", "n_censored", "n_distinct_mic"]
        assert counts["n"].sum() > 0
        kept = pd.read_csv(paths.pairs_kept)
        assert list(kept.columns) == ["species", "drug", "n", "n_nonsusceptible", "n_susceptible", "n_distinct_mic"]
        assert kept.empty  # the toy data is far below 50/50
        drop_log = pd.read_csv(paths.drop_log("ingest"))
        assert set(drop_log["stage"]) == {"ingest"}
        assert hz.Reason.EVIDENCE in drop_log["reason"].tolist()
        assert int(drop_log.loc[drop_log["reason"] == hz.Reason.EVIDENCE, "n_dropped"].iloc[0]) == 1
        # the NCBI disk row (no standard_year column in NCBI exports) is dropped and counted
        assert int(drop_log.loc[drop_log["reason"] == hz.Reason.NULL_YEAR, "n_dropped"].iloc[0]) == 1
        assert int(drop_log.loc[drop_log["reason"] == hz.Reason.NO_TABLE_FOR_YEAR, "n_dropped"].iloc[0]) == 0

    def test_missing_one_raw_file_is_skipped_with_a_log_line(self, tmp_path: Path, config: Config,
                                                              caplog: pytest.LogCaptureFixture) -> None:
        paths = make_root(tmp_path, ncbi_csv=None)
        with caplog.at_level(logging.WARNING, logger="genome2mic.ingest"):
            labels = ingest.run(paths, config)
        assert any("ast_ncbi.csv" in r.getMessage() for r in caplog.records)
        assert set(labels["source"]) == {"BVBRC"}

    def test_both_raw_files_missing_raises(self, tmp_path: Path, config: Config) -> None:
        paths = make_root(tmp_path, bvbrc_csv=None, ncbi_csv=None)
        with pytest.raises(FileNotFoundError):
            ingest.run(paths, config)

    def test_runs_without_metadata_file(self, tmp_path: Path, config: Config) -> None:
        paths = make_root(tmp_path, metadata_csv=None)
        labels = ingest.run(paths, config)
        assert labels["biosample"].isna().sum() > 0  # BV-BRC rows have no biosample without metadata
        # NCBI rows keep their NCBI ids because nothing maps them to BV-BRC ids
        assert "NCBI_SAMN00000002" in labels["genome_id"].tolist()

    def test_pairs_kept_thresholds_are_passed_through(self, tmp_path: Path, config: Config) -> None:
        paths = make_root(tmp_path)
        ingest.run(paths, config, min_ns=1, min_s=1, min_levels=1)
        kept = pd.read_csv(paths.pairs_kept)
        assert ("KPNEU", "meropenem") in set(zip(kept["species"], kept["drug"]))

    def test_load_raw_concatenates_sources(self, tmp_path: Path, config: Config) -> None:
        paths = make_root(tmp_path)
        raw = ingest.load_raw(paths, config)
        assert list(raw.columns) == list(RAW_COLUMNS)
        assert set(raw["source"]) == {"BVBRC", "NCBI"}
        assert len(raw) == 11
