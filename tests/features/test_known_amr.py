"""Tests for ``genome2mic.features.known_amr`` (stage 5).

AMRFinderPlus tables are written inline into ``tmp_path`` in both header variants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic.config import Config, SpeciesConfig
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation
from genome2mic.features import known_amr as ka
from genome2mic.io import read_parquet, write_parquet
from genome2mic.paths import Paths

KEEP = ("blaKPC", "blaNDM", "blaOXA-48", "blaOXA-181", "blaOXA-232", "blaVIM", "blaIMP")

HEADER_4X = [
    "Protein id", "Contig id", "Start", "Stop", "Strand", "Element symbol", "Element name", "Scope",
    "Type", "Subtype", "Class", "Subclass", "Method", "Target length", "Reference sequence length",
    "% Coverage of reference", "% Identity to reference", "Alignment length",
    "Closest reference accession", "Closest reference name", "HMM accession", "HMM description",
]
HEADER_3X = [
    "Protein identifier", "Contig id", "Start", "Stop", "Strand", "Gene symbol", "Sequence name", "Scope",
    "Element type", "Element subtype", "Class", "Subclass", "Method", "Target length",
    "Reference sequence length", "% Coverage of reference sequence", "% Identity to reference sequence",
    "Alignment length", "Accession of closest sequence", "Name of closest sequence", "HMM id", "HMM description",
]

Row = tuple[str, str, str, str, str]  # symbol, type, subtype, class, subclass


def make_config(keep: tuple[str, ...] = KEEP) -> Config:
    species = {
        key: SpeciesConfig(key=key, name=key, amrfinder_organism=key, expected_genome_size=5_000_000,
                           size_tolerance=0.2, reference_accession="X")
        for key in ("KPNEU", "ECOLI")
    }
    return Config(species=species, drugs={}, call_standard=("EUCAST", "2024"), qc_max_contigs=500,
                  qc_max_mash_distance=0.05, breakpoints={}, natural_resistance=frozenset(),
                  keep_variant=keep, synonym_map={})


def amrfinder_text(rows: list[Row], *, version: str = "4", coverage: str = "100.00") -> str:
    header = HEADER_4X if version == "4" else HEADER_3X
    lines = ["\t".join(header)]
    for symbol, typ, subtype, cls, subclass in rows:
        cells = ["NA", "contig_1", "100", "900", "+", symbol, f"{symbol} protein", "core", typ, subtype, cls, subclass,
                 "EXACTX", "293", "293", coverage, "100.00", "293", "WP_000000000.1", f"{symbol} name", "NA", "NA"]
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def write_amrfinder(paths: Paths, genome_id: str, rows: list[Row], *, version: str = "4") -> Path:
    path = paths.interim_dir(genome_id) / ka.AMRFINDER_TSV_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(amrfinder_text(rows, version=version))
    return path


def write_qc(paths: Paths, rows: list[tuple[str, str | None, bool]], *, extra_metadata: bool = True) -> None:
    frame = pd.DataFrame(
        {
            "genome_id": pd.array([r[0] for r in rows], dtype="str"),
            "species": pd.array([r[1] for r in rows], dtype="str"),
            "n_contigs": [10] * len(rows),
            "qc_pass": np.asarray([r[2] for r in rows], dtype=bool),
            "qc_fail_reason": pd.array([None if r[2] else "too_fragmented" for r in rows], dtype="str"),
        }
    )
    if extra_metadata:  # must never leak into the feature matrix
        frame["country"] = "USA"
        frame["lineage_cluster"] = "KPNEU_ML_001"
    write_parquet(frame, paths.qc)


# --------------------------------------------------------------------- family
class TestFamilyOf:
    @pytest.mark.parametrize(
        ("symbol", "family"),
        [
            ("blaOXA-48", "blaOXA-48"),
            ("blaOXA-1", "blaOXA"),
            ("blaOXA-10", "blaOXA"),
            ("blaOXA-181", "blaOXA-181"),
            ("blaOXA-484", "blaOXA"),  # prefix blaOXA-48 matches at a token boundary only
            ("blaCTX-M-15", "blaCTX-M"),
            ("blaCTX-M-27", "blaCTX-M"),
            ("blaKPC-2", "blaKPC-2"),
            ("blaKPC-3", "blaKPC-3"),
            ("blaNDM-1", "blaNDM-1"),
            ("blaVIM-2", "blaVIM-2"),
            ("blaIMP-4", "blaIMP-4"),
            ("blaTEM-1", "blaTEM"),
            ("blaSHV-11", "blaSHV"),
            ("blaCMY-2", "blaCMY"),
            ("mcr-1.1", "mcr"),
            ("aac(6')-Ib-cr", "aac(6')-Ib-cr"),
            ("aph(3'')-Ib", "aph(3'')-Ib"),
            ("mecA", "mecA"),
            ("tet(A)", "tet(A)"),
            ("qnrB1", "qnrB1"),
            ("sul1", "sul1"),
        ],
    )
    def test_examples(self, symbol: str, family: str) -> None:
        assert ka.family_of(symbol, KEEP) == family

    def test_without_keep_list_variants_collapse(self) -> None:
        assert ka.family_of("blaKPC-2", ()) == "blaKPC"
        assert ka.family_of("blaOXA-48", ()) == "blaOXA"

    def test_exact_prefix_symbol_is_kept(self) -> None:
        assert ka.family_of("blaKPC", KEEP) == "blaKPC"

    def test_strips_whitespace_and_rejects_empty(self) -> None:
        assert ka.family_of("  blaCTX-M-15 ", KEEP) == "blaCTX-M"
        with pytest.raises(ValueError):
            ka.family_of("   ", KEEP)


class TestColumnName:
    @pytest.mark.parametrize(
        ("prefix", "symbol", "expected"),
        [
            ("gene_", "blaCTX-M", "gene_blactx_m"),
            ("gene_", "blaKPC-2", "gene_blakpc_2"),
            ("gene_", "blaOXA-48", "gene_blaoxa_48"),
            ("gene_", "aac(6')-Ib-cr", "gene_aac_6_ib_cr"),
            ("gene_", "tet(A)", "gene_tet_a"),
            ("gene_", "tet((A))", "gene_tet_a"),
            ("gene_", "mecA", "gene_meca"),
            ("point_", "ompK36_D135DGD", "point_ompk36_d135dgd"),
            ("point_", "gyrA_S83L", "point_gyra_s83l"),
            ("n_class_", "BETA-LACTAM", "n_class_beta_lactam"),
            ("n_class_", "AMINOGLYCOSIDE/QUINOLONE", "n_class_aminoglycoside_quinolone"),
            ("n_class_", " QUINOLONE ", "n_class_quinolone"),
        ],
    )
    def test_examples(self, prefix: str, symbol: str, expected: str) -> None:
        assert ka.column_name(prefix, symbol) == expected

    def test_rejects_unknown_prefix_and_empty_body(self) -> None:
        with pytest.raises(ValueError):
            ka.column_name("u_", "x")
        with pytest.raises(ValueError):
            ka.column_name("gene_", "()")


# -------------------------------------------------------------------- parsing
class TestParseAmrfinder:
    ROWS: list[Row] = [
        ("blaKPC-2", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM"),
        ("ompK36_D135DGD", "AMR", "POINT", "BETA-LACTAM", "BETA-LACTAM"),
        ("qacE", "STRESS", "BIOCIDE", "QUATERNARY AMMONIUM", "QUATERNARY AMMONIUM"),
        ("iutA", "VIRULENCE", "VIRULENCE", "NA", "NA"),
        ("blaX", "AMR", "AMR-SUSCEPTIBLE", "BETA-LACTAM", "BETA-LACTAM"),
    ]

    @pytest.mark.parametrize("version", ["4", "3"])
    def test_both_header_variants(self, tmp_path: Path, version: str) -> None:
        path = tmp_path / "amrfinder.tsv"
        path.write_text(amrfinder_text(self.ROWS, version=version))
        frame = ka.parse_amrfinder(path)
        assert list(frame.columns) == list(ka.PARSED_COLUMNS)
        assert list(frame["symbol"]) == ["blaKPC-2", "ompK36_D135DGD"]
        assert list(frame["subtype"]) == ["AMR", "POINT"]
        assert list(frame["type"]) == ["AMR", "AMR"]
        assert list(frame["class"]) == ["BETA-LACTAM", "BETA-LACTAM"]
        assert list(frame["subclass"]) == ["CARBAPENEM", "BETA-LACTAM"]
        assert list(frame["method"]) == ["EXACTX", "EXACTX"]
        assert frame["coverage"].tolist() == [100.0, 100.0]

    def test_na_and_blank_cells_are_null(self, tmp_path: Path) -> None:
        path = tmp_path / "amrfinder.tsv"
        path.write_text(amrfinder_text([("blaTEM-1", "AMR", "AMR", "NA", "")], coverage="NA"))
        frame = ka.parse_amrfinder(path)
        assert len(frame) == 1
        assert pd.isna(frame.loc[0, "class"])
        assert pd.isna(frame.loc[0, "subclass"])
        assert pd.isna(frame.loc[0, "coverage"])

    def test_header_only_is_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "amrfinder.tsv"
        path.write_text("\t".join(HEADER_4X) + "\n")
        frame = ka.parse_amrfinder(path)
        assert len(frame) == 0
        assert list(frame.columns) == list(ka.PARSED_COLUMNS)

    def test_rejects_empty_file_and_foreign_header(self, tmp_path: Path) -> None:
        path = tmp_path / "amrfinder.tsv"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            ka.parse_amrfinder(path)
        path.write_text("# Antimicrobial\tClass\tWGS-predicted phenotype\n")
        with pytest.raises(ValueError, match="not an AMRFinderPlus table"):
            ka.parse_amrfinder(path)


# ---------------------------------------------------------------------- build
G1_ROWS: list[Row] = [
    ("blaKPC-2", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM"),
    ("blaSHV-11", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM"),
    ("blaCTX-M-15", "AMR", "AMR", "BETA-LACTAM", "CEPHALOSPORIN"),
    ("blaCTX-M-15", "AMR", "AMR", "BETA-LACTAM", "CEPHALOSPORIN"),  # second copy
    ("ompK36_D135DGD", "AMR", "POINT", "BETA-LACTAM", "BETA-LACTAM"),
    ("qacE", "STRESS", "BIOCIDE", "QUATERNARY AMMONIUM", "QUATERNARY AMMONIUM"),
]
G2_ROWS: list[Row] = [
    ("blaNDM-1", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM"),
    ("blaCTX-M-27", "AMR", "AMR", "BETA-LACTAM", "CEPHALOSPORIN"),
    ("gyrA_S83L", "AMR", "POINT", "QUINOLONE", "QUINOLONE"),
    ("aac(6')-Ib-cr", "AMR", "AMR", "AMINOGLYCOSIDE/QUINOLONE", "AMIKACIN/KANAMYCIN/QUINOLONE"),
    ("blaOXA-1", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM"),
    ("blaOXA-48", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM"),
]
EXPECTED_COLUMNS = [
    "genome_id", "species",
    "gene_aac_6_ib_cr", "gene_blactx_m", "gene_blakpc_2", "gene_blandm_1", "gene_blaoxa", "gene_blaoxa_48", "gene_blashv",
    "point_gyra_s83l", "point_ompk36_d135dgd",
    "n_class_aminoglycoside_quinolone", "n_class_beta_lactam", "n_class_quinolone",
]


def build_dataset(tmp_path: Path) -> tuple[Paths, Config]:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    write_qc(paths, [("g1", "KPNEU", True), ("g2", "KPNEU", True), ("g3", "ECOLI", True), ("g4", "KPNEU", False)])
    write_amrfinder(paths, "g1", G1_ROWS, version="4")
    write_amrfinder(paths, "g2", G2_ROWS, version="3")
    write_amrfinder(paths, "g4", [("mecA", "AMR", "AMR", "BETA-LACTAM", "METHICILLIN")])  # QC fail: excluded
    # g3 has no amrfinder.tsv -> zero-filled
    return paths, make_config()


class TestBuild:
    def test_end_to_end_values(self, tmp_path: Path) -> None:
        paths, config = build_dataset(tmp_path)
        features, columns = ka.build(paths, config)

        assert list(features.columns) == EXPECTED_COLUMNS
        assert list(features["genome_id"]) == ["g1", "g2", "g3"]
        assert list(features["species"]) == ["KPNEU", "KPNEU", "ECOLI"]
        assert "gene_meca" not in features.columns

        by_id = features.set_index("genome_id")
        g1 = by_id.loc["g1"]
        assert g1["gene_blakpc_2"] == 1 and g1["gene_blashv"] == 1 and g1["gene_blactx_m"] == 1
        assert g1["point_ompk36_d135dgd"] == 1
        assert g1["gene_blandm_1"] == 0 and g1["gene_blaoxa"] == 0 and g1["gene_blaoxa_48"] == 0
        assert g1["n_class_beta_lactam"] == 5  # KPC, SHV, CTX-M x2, ompK36
        assert g1["n_class_quinolone"] == 0

        g2 = by_id.loc["g2"]
        assert g2["gene_blandm_1"] == 1 and g2["gene_blactx_m"] == 1 and g2["gene_aac_6_ib_cr"] == 1
        assert g2["gene_blaoxa"] == 1 and g2["gene_blaoxa_48"] == 1
        assert g2["point_gyra_s83l"] == 1 and g2["point_ompk36_d135dgd"] == 0
        assert g2["n_class_beta_lactam"] == 4
        assert g2["n_class_quinolone"] == 1
        assert g2["n_class_aminoglycoside_quinolone"] == 1

        feature_cols = ka.feature_columns(features)
        assert (by_id.loc["g3", feature_cols] == 0).all()

        assert all(features[c].dtype == np.int8 for c in feature_cols)
        assert not features[feature_cols].isna().any().any()
        assert not set(features.columns) & ka.FORBIDDEN_FEATURES
        assert "country" not in features.columns

        mapping = columns.set_index("column_name")
        assert list(mapping.columns) == [c for c in ka.COLUMNS_FILE_COLUMNS if c != "column_name"]
        assert set(mapping.index) == set(feature_cols)
        ctxm = mapping.loc["gene_blactx_m"]
        assert ctxm["source_symbol"] == "blaCTX-M"
        assert ctxm["member_symbols"] == "blaCTX-M-15;blaCTX-M-27"
        assert ctxm["class"] == "BETA-LACTAM"
        assert ctxm["subclass"] == "CEPHALOSPORIN"
        assert ctxm["n_genomes_present"] == 2
        assert mapping.loc["gene_blaoxa_48", "n_genomes_present"] == 1
        assert mapping.loc["gene_blaoxa_48", "source_symbol"] == "blaOXA-48"
        assert mapping.loc["gene_blaoxa", "member_symbols"] == "blaOXA-1"
        assert mapping.loc["point_gyra_s83l", "source_symbol"] == "gyrA_S83L"
        assert mapping.loc["n_class_beta_lactam", "n_genomes_present"] == 2
        assert mapping.loc["n_class_beta_lactam", "source_symbol"] == "BETA-LACTAM"
        assert pd.isna(mapping.loc["n_class_beta_lactam", "subclass"])

    def test_run_writes_files_and_drop_log(self, tmp_path: Path) -> None:
        paths, config = build_dataset(tmp_path)
        features = ka.run(paths, config)
        assert paths.known_amr.is_file() and paths.known_amr_columns.is_file()
        reread = read_parquet(paths.known_amr)
        assert reread.shape == features.shape
        assert all(reread[c].dtype == np.int8 for c in ka.feature_columns(reread))
        ka.validate_known_amr(reread, pd.read_csv(paths.known_amr_columns), expected_genome_ids=["g1", "g2", "g3"])

        log = pd.read_csv(paths.drop_log("known_amr"))
        assert set(log["stage"]) == {"known_amr"}
        counts = log.set_index("reason")["n_dropped"]
        assert counts["qc_fail"] == 1
        assert counts["amrfinder.tsv missing -> features zero-filled (rows kept)"] == 1
        assert counts["amrfinder rows with Type != AMR (STRESS/VIRULENCE)"] == 1
        assert counts["amrfinder AMR rows with Subtype not in {AMR, POINT}"] == 0
        assert log.loc[log["reason"].str.startswith("amrfinder.tsv missing"), "detail"].iloc[0] == "g3"

    def test_build_requires_qc(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ka.build(Paths(root=tmp_path), make_config())

    def test_species_falls_back_to_labels(self, tmp_path: Path) -> None:
        paths, config = build_dataset(tmp_path)
        qc = read_parquet(paths.qc).drop(columns=["species"])
        write_parquet(qc, paths.qc)
        with pytest.raises(ContractViolation, match="no species"):
            ka.build(paths, config)
        labels = pd.DataFrame({"genome_id": ["g1", "g1", "g2", "g3"], "species": ["KPNEU", "KPNEU", "KPNEU", "ECOLI"],
                               "drug": ["meropenem", "ciprofloxacin", "meropenem", "meropenem"]})
        write_parquet(labels, paths.labels)
        features, _ = ka.build(paths, config)
        assert list(features["species"]) == ["KPNEU", "KPNEU", "ECOLI"]

    def test_no_hits_anywhere_is_a_contract_violation(self, tmp_path: Path) -> None:
        paths = Paths(root=tmp_path)
        write_qc(paths, [("g1", "KPNEU", True)])
        with pytest.raises(ContractViolation, match="no feature columns"):
            ka.build(paths, make_config())

    def test_class_count_over_int8_raises(self, tmp_path: Path) -> None:
        paths = Paths(root=tmp_path)
        write_qc(paths, [("g1", "KPNEU", True)])
        write_amrfinder(paths, "g1", [("blaTEM-1", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM")] * 128)
        with pytest.raises(ContractViolation, match="int8"):
            ka.build(paths, make_config())

    def test_keep_variant_from_config_controls_collapse(self, tmp_path: Path) -> None:
        paths = Paths(root=tmp_path)
        write_qc(paths, [("g1", "KPNEU", True)])
        write_amrfinder(paths, "g1", [("blaKPC-2", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM")])
        kept, _ = ka.build(paths, make_config(keep=("blaKPC",)))
        collapsed, _ = ka.build(paths, make_config(keep=("blaNDM",)))
        assert "gene_blakpc_2" in kept.columns and "gene_blakpc" not in kept.columns
        assert "gene_blakpc" in collapsed.columns and "gene_blakpc_2" not in collapsed.columns

    def test_build_accepts_external_droplog(self, tmp_path: Path) -> None:
        paths, config = build_dataset(tmp_path)
        log = DropLog("known_amr")
        ka.build(paths, config, droplog=log)
        assert any(r.reason == "qc_fail" and r.n_dropped == 1 for r in log.records)


# ------------------------------------------------------------------ validate
def good_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "genome_id": pd.array(["a", "b"], dtype="str"),
            "species": pd.array(["KPNEU", "KPNEU"], dtype="str"),
            "gene_blakpc_2": np.array([1, 0], dtype=np.int8),
            "point_gyra_s83l": np.array([0, 1], dtype=np.int8),
            "n_class_beta_lactam": np.array([3, 0], dtype=np.int8),
        }
    )
    columns = pd.DataFrame(
        {
            "column_name": ["gene_blakpc_2", "point_gyra_s83l", "n_class_beta_lactam"],
            "source_symbol": ["blaKPC-2", "gyrA_S83L", "BETA-LACTAM"],
            "class": ["BETA-LACTAM", "QUINOLONE", "BETA-LACTAM"],
            "subclass": ["CARBAPENEM", "QUINOLONE", None],
            "n_genomes_present": [1, 1, 1],
            "member_symbols": ["blaKPC-2", "gyrA_S83L", "BETA-LACTAM"],
        }
    )
    return features, columns


class TestValidate:
    def test_good_passes(self) -> None:
        features, columns = good_features()
        ka.validate_known_amr(features, columns, expected_genome_ids=["b", "a"])

    def test_row_set_must_match_qc_passing(self) -> None:
        features, columns = good_features()
        with pytest.raises(ContractViolation, match="one row per QC-passing genome"):
            ka.validate_known_amr(features, columns, expected_genome_ids=["a", "b", "c"])

    def test_nulls_rejected(self) -> None:
        features, columns = good_features()
        features["gene_blakpc_2"] = pd.array([1, None], dtype="Int8")
        with pytest.raises(ContractViolation, match="nulls"):
            ka.validate_known_amr(features, columns)

    def test_non_binary_gene_rejected(self) -> None:
        features, columns = good_features()
        features["gene_blakpc_2"] = np.array([2, 0], dtype=np.int8)
        with pytest.raises(ContractViolation, match="0/1"):
            ka.validate_known_amr(features, columns)

    def test_negative_rejected(self) -> None:
        features, columns = good_features()
        features["n_class_beta_lactam"] = np.array([-1, 0], dtype=np.int8)
        with pytest.raises(ContractViolation, match="negative"):
            ka.validate_known_amr(features, columns)

    def test_wrong_dtype_rejected(self) -> None:
        features, columns = good_features()
        features["gene_blakpc_2"] = features["gene_blakpc_2"].astype(np.int64)
        with pytest.raises(ContractViolation, match="int8"):
            ka.validate_known_amr(features, columns)

    @pytest.mark.parametrize("name", sorted(ka.FORBIDDEN_FEATURES))
    def test_forbidden_metadata_rejected(self, name: str) -> None:
        features, columns = good_features()
        features[name] = np.array([0, 1], dtype=np.int8)
        with pytest.raises(ContractViolation):
            ka.validate_known_amr(features, columns)

    def test_unprefixed_column_rejected(self) -> None:
        features, columns = good_features()
        features["Gene_X"] = np.array([0, 1], dtype=np.int8)
        with pytest.raises(ContractViolation, match="prefixed"):
            ka.validate_known_amr(features, columns)

    def test_mapping_must_match(self) -> None:
        features, columns = good_features()
        with pytest.raises(ContractViolation, match="mapping"):
            ka.validate_known_amr(features, columns.iloc[:2])
        with pytest.raises(ContractViolation, match="mapping"):
            ka.validate_known_amr(features, columns.drop(columns=["class"]))

    def test_duplicate_genome_rejected(self) -> None:
        features, columns = good_features()
        features["genome_id"] = pd.array(["a", "a"], dtype="str")
        with pytest.raises(ContractViolation, match="unique"):
            ka.validate_known_amr(features, columns)


# ----------------------------------------------------------------- filter_rare
class TestFilterRare:
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "genome_id": [f"g{i}" for i in range(6)],
                "species": ["KPNEU"] * 6,
                "gene_a": np.array([1, 1, 1, 1, 1, 0], dtype=np.int8),  # 4 of the 4 training rows
                "gene_b": np.array([1, 1, 1, 0, 0, 1], dtype=np.int8),  # 3 training rows, 1 test row
                "point_c": np.array([0, 0, 0, 0, 1, 1], dtype=np.int8),  # test rows only
                "n_class_d": np.array([2, 5, 1, 7, 0, 0], dtype=np.int8),  # counts > 0 in 4 training rows
            }
        )

    def test_counts_training_rows_only(self) -> None:
        X = self.frame()
        mask = np.array([True, True, True, True, False, False])
        assert ka.filter_rare(X, mask, min_count=4) == ["gene_a", "n_class_d"]
        assert ka.filter_rare(X, mask, min_count=3) == ["gene_a", "gene_b", "n_class_d"]
        assert ka.filter_rare(X, mask, min_count=1) == ["gene_a", "gene_b", "n_class_d"]  # point_c never in train
        assert ka.filter_rare(X, np.ones(6, dtype=bool), min_count=1) == ["gene_a", "gene_b", "point_c", "n_class_d"]

    def test_series_mask_aligned_on_index(self) -> None:
        X = self.frame().set_index(pd.Index([10, 11, 12, 13, 14, 15]))
        mask = pd.Series([True, True, True, True, False, False], index=X.index)
        assert ka.filter_rare(X, mask, min_count=4) == ["gene_a", "n_class_d"]

    def test_records_in_droplog(self) -> None:
        log = DropLog("train")
        ka.filter_rare(self.frame(), np.array([True] * 4 + [False] * 2), min_count=4, droplog=log)
        assert log.records[0].reason == "rare known-AMR feature columns"
        assert log.records[0].n_dropped == 2

    def test_input_validation(self) -> None:
        X = self.frame()
        with pytest.raises(ValueError):
            ka.filter_rare(X, np.ones(6, dtype=bool), min_count=0)
        with pytest.raises(ValueError):
            ka.filter_rare(X, np.ones(5, dtype=bool))
        with pytest.raises(ValueError):
            ka.filter_rare(X, np.ones(6, dtype=int))

    def test_forbidden_metadata_raises(self) -> None:
        X = self.frame()
        X["lineage_cluster"] = "KPNEU_ML_001"
        with pytest.raises(ContractViolation, match="forbidden"):
            ka.filter_rare(X, np.ones(6, dtype=bool))

    def test_no_feature_columns(self) -> None:
        X = self.frame()[["genome_id", "species"]]
        assert ka.filter_rare(X, np.ones(6, dtype=bool)) == []
