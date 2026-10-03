"""Tests for genome2mic.paths (project layout) and genome2mic.io (file helpers).

The io tests live here because tests/core/test_io.py is not in this module's file
allocation; both modules are the filesystem layer, so they share a file for now.
"""

from __future__ import annotations

import dataclasses
import gzip
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic import io as g2m_io
from genome2mic.errors import ContractViolation, Genome2MicError, ToolNotAvailable
from genome2mic.paths import RAW_AST_SOURCES, Paths


# ----------------------------------------------------------------------- Paths
class TestPathsDefaults:
    def test_default_matches_design(self) -> None:
        paths = Paths.default()
        assert paths.root == Path(".")
        assert paths.configs_dir == Path("configs")
        assert Paths() == paths

    def test_strings_are_coerced_to_paths(self) -> None:
        paths = Paths(root="runs/synthetic", configs_dir="runs/synthetic/configs")  # type: ignore[arg-type]
        assert isinstance(paths.root, Path) and isinstance(paths.configs_dir, Path)
        assert paths.labels == Path("runs/synthetic/data/processed/labels.parquet")

    def test_frozen(self) -> None:
        paths = Paths.default()
        with pytest.raises(dataclasses.FrozenInstanceError):
            paths.root = Path("/tmp")  # type: ignore[misc]

    def test_hashable_and_equal(self) -> None:
        assert hash(Paths(Path("x"), Path("y"))) == hash(Paths("x", "y"))  # type: ignore[arg-type]


class TestPathsLayout:
    root = Path("/proj")
    paths = Paths(root=root, configs_dir=root / "configs")

    def test_data_layer(self) -> None:
        p, r = self.paths, self.root
        assert p.data_dir == r / "data"
        assert p.raw_dir == r / "data/raw"
        assert p.raw_ast("bvbrc") == r / "data/raw/ast_bvbrc.csv"
        assert p.raw_ast("ncbi") == r / "data/raw/ast_ncbi.csv"
        assert p.genome_metadata == r / "data/raw/genome_metadata.csv"
        assert p.genomes_dir == r / "data/raw/genomes"
        assert p.genome_fasta("573.2002") == r / "data/raw/genomes/573.2002.fasta"
        assert p.genome_fasta("NCBI_SAMN00000001") == r / "data/raw/genomes/NCBI_SAMN00000001.fasta"
        assert p.references_dir == r / "data/raw/references"
        assert p.reference_fasta("KPNEU") == r / "data/raw/references/KPNEU.fasta"
        assert p.interim_root == r / "data/interim"
        assert p.interim_dir("573.2002") == r / "data/interim/573.2002"
        assert p.processed_dir == r / "data/processed"

    def test_processed_contract_files(self) -> None:
        p, proc = self.paths, self.root / "data/processed"
        assert p.labels == proc / "labels.parquet"
        assert p.pairs_kept == proc / "pairs_kept.csv"
        assert p.label_counts == proc / "label_counts.csv"
        assert p.qc == proc / "qc.parquet"
        assert p.known_amr == proc / "known_amr.parquet"
        assert p.known_amr_columns == proc / "known_amr_columns.csv"
        assert p.lineages == proc / "lineages.parquet"
        assert p.sketches("KPNEU") == proc / "sketches_KPNEU.npz"
        assert p.splits == proc / "splits.parquet"
        assert p.unitigs("KPNEU") == proc / "unitigs_KPNEU.npz"
        assert p.unitig_rows("KPNEU") == proc / "unitigs_KPNEU_rows.parquet"
        assert p.unitig_index("KPNEU") == proc / "unitigs_KPNEU_index.parquet"
        assert p.unitig_kmers("KPNEU") == proc / "unitigs_KPNEU_kmers.npz"
        assert p.drop_log("ingest") == proc / "drop_log_ingest.csv"

    def test_results_layer(self) -> None:
        p, res = self.paths, self.root / "results"
        assert p.results_dir == res
        assert p.preds("KPNEU", "meropenem") == res / "preds_KPNEU_meropenem.parquet"
        assert p.preds("ECOLI", "piperacillin-tazobactam") == res / "preds_ECOLI_piperacillin-tazobactam.parquet"
        assert p.metrics == res / "metrics.parquet"
        assert p.metrics_by_distance == res / "metrics_by_distance.parquet"
        assert p.figures_dir == res / "figures"
        assert p.report_md == res / "report.md"

    def test_models_layer(self) -> None:
        p, r = self.paths, self.root
        assert p.models_dir == r / "models"
        assert p.models_manifest == r / "models/manifest.json"
        assert p.model_dir("KPNEU", "ceftriaxone") == r / "models/KPNEU/ceftriaxone"

    def test_species_key_is_upper_cased_in_file_names(self) -> None:
        p = self.paths
        assert p.unitigs("kpneu") == p.unitigs("KPNEU")
        assert p.sketches(" ecoli ") == p.sketches("ECOLI")
        assert p.preds("saur", "oxacillin").name == "preds_SAUR_oxacillin.parquet"
        with pytest.raises(ValueError):
            p.unitigs("")

    def test_drug_name_used_verbatim(self) -> None:
        # Normalization is config's job; paths must not alter a drug name.
        assert self.paths.preds("KPNEU", "ciprofloxacin").name == "preds_KPNEU_ciprofloxacin.parquet"
        assert self.paths.model_dir("KPNEU", "ciprofloxacin").name == "ciprofloxacin"

    def test_raw_ast_accepts_case_insensitive_sources_only(self) -> None:
        assert RAW_AST_SOURCES == {"bvbrc", "ncbi"}
        assert self.paths.raw_ast("BVBRC") == self.paths.raw_ast("bvbrc")
        assert self.paths.raw_ast(" NCBI ") == self.paths.raw_ast("ncbi")
        with pytest.raises(ValueError):
            self.paths.raw_ast("patric")

    def test_every_path_is_under_root(self) -> None:
        p = self.paths
        members = [
            p.raw_dir, p.raw_ast("bvbrc"), p.genomes_dir, p.genome_fasta("g"), p.interim_dir("g"),
            p.processed_dir, p.labels, p.pairs_kept, p.label_counts, p.qc, p.known_amr,
            p.known_amr_columns, p.lineages, p.sketches("KPNEU"), p.splits, p.unitigs("KPNEU"),
            p.unitig_rows("KPNEU"), p.unitig_index("KPNEU"), p.unitig_kmers("KPNEU"),
            p.drop_log("qc"), p.results_dir, p.preds("KPNEU", "meropenem"), p.metrics,
            p.figures_dir, p.report_md, p.models_dir,
        ]
        for member in members:
            assert self.root in member.parents or member == self.root, member

    def test_relative_root_stays_relative(self) -> None:
        paths = Paths(root=Path("runs/synthetic"), configs_dir=Path("runs/synthetic/configs"))
        assert not paths.labels.is_absolute()
        assert paths.labels == Path("runs/synthetic/data/processed/labels.parquet")


# ---------------------------------------------------------------------- errors
class TestErrors:
    def test_tool_not_available_message_and_attributes(self) -> None:
        err = ToolNotAvailable("amrfinder", hint="conda install -c bioconda ncbi-amrfinderplus")
        assert err.tool == "amrfinder"
        assert "amrfinder" in str(err) and "conda install" in str(err)
        assert isinstance(err, Genome2MicError) and isinstance(err, Exception)
        assert str(ToolNotAvailable("mash")).endswith("PATH.")

    def test_contract_violation(self) -> None:
        err = ContractViolation("(genome_id, drug) not unique: 3 duplicates", stage="ingest")
        assert str(err).startswith("[ingest] ")
        assert err.stage == "ingest"
        assert isinstance(err, Genome2MicError)
        assert str(ContractViolation("x")) == "x"


# -------------------------------------------------------------------------- io
class TestTables:
    def test_parquet_round_trip_creates_parent_dirs_and_keeps_nulls(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "genome_id": pd.array(["573.2002", "573.2005"], dtype="str"),
                "mic_lower": [4.0, 32.0],
                "mic_upper": [8.0, math.inf],
                "sir": pd.array(["R", None], dtype="str"),
                "year": pd.array([2016, None], dtype="Int64"),
            }
        )
        target = tmp_path / "nested" / "deeper" / "labels.parquet"
        assert g2m_io.write_parquet(frame, target) == target
        back = g2m_io.read_parquet(target)
        assert list(back.columns) == list(frame.columns)
        assert back["mic_upper"].iloc[1] == math.inf
        assert pd.isna(back["sir"].iloc[1]) and pd.isna(back["year"].iloc[1])
        assert back["year"].iloc[0] == 2016
        assert len(back) == 2
        assert "index" not in back.columns and "__index_level_0__" not in back.columns

    def test_parquet_column_selection(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = g2m_io.write_parquet(frame, tmp_path / "t.parquet")
        assert list(g2m_io.read_parquet(path, columns=["b"]).columns) == ["b"]

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"species": ["KPNEU"], "drug": ["meropenem"], "n": [120]})
        path = g2m_io.write_csv(frame, tmp_path / "out" / "pairs_kept.csv")
        back = g2m_io.read_csv(path)
        assert back.to_dict("records") == [{"species": "KPNEU", "drug": "meropenem", "n": 120}]

    def test_ensure_dir(self, tmp_path: Path) -> None:
        target = g2m_io.ensure_dir(tmp_path / "a" / "b")
        assert target.is_dir()
        assert g2m_io.ensure_dir(target) == target  # idempotent


class TestFasta:
    def test_read_uppercases_and_strips_whitespace(self, tmp_path: Path) -> None:
        text = (
            ">contig_1 length=12 cov=30.1\n"
            "acgt acgt\t\n"
            "AC GT\r\n"
            "\n"
            ">contig_2\n"
            "nnnACGT\n"
            "acg\n"
        )
        path = tmp_path / "g.fasta"
        path.write_text(text)
        records = g2m_io.read_fasta(path)
        assert records == [
            ("contig_1 length=12 cov=30.1", "ACGTACGTACGT"),
            ("contig_2", "NNNACGTACG"),
        ]
        for _, seq in records:
            assert seq == seq.upper() and not any(ch.isspace() for ch in seq)

    def test_header_without_description_and_empty_sequence(self, tmp_path: Path) -> None:
        path = tmp_path / "g.fa"
        path.write_text(">only_header\n>second\nACGT\n")
        assert g2m_io.read_fasta(path) == [("only_header", ""), ("second", "ACGT")]

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.fasta"
        path.write_text("")
        assert g2m_io.read_fasta(path) == []

    def test_sequence_before_header_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.fasta"
        path.write_text("ACGT\n>c1\nACGT\n")
        with pytest.raises(ValueError):
            g2m_io.read_fasta(path)

    def test_gzip_input(self, tmp_path: Path) -> None:
        path = tmp_path / "g.fasta.gz"
        with gzip.open(path, "wt") as handle:
            handle.write(">c1\nacgt\n")
        assert g2m_io.read_fasta(path) == [("c1", "ACGT")]

    def test_iter_fasta_is_lazy_and_matches_read(self, tmp_path: Path) -> None:
        path = tmp_path / "g.fasta"
        path.write_text(">a\nAC\n>b\nGT\n")
        iterator = g2m_io.iter_fasta(path)
        assert next(iterator) == ("a", "AC")
        assert list(iterator) == [("b", "GT")]

    def test_write_then_read_round_trip_with_wrapping(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        seq = "".join(rng.choice(list("ACGT"), size=205))
        records = [("c1 synthetic", seq), ("c2", "ACGT")]
        path = g2m_io.write_fasta(records, tmp_path / "out" / "g.fasta", line_width=60)
        lines = path.read_text().splitlines()
        assert lines[0] == ">c1 synthetic"
        assert max(len(line) for line in lines[1:5]) == 60 and len(lines[4]) == 25
        assert g2m_io.read_fasta(path) == records

    def test_write_unwrapped_and_gzip(self, tmp_path: Path) -> None:
        path = g2m_io.write_fasta([("c1", "ACGT" * 50)], tmp_path / "g.fasta.gz", line_width=0)
        with gzip.open(path, "rt") as handle:
            lines = handle.read().splitlines()
        assert lines == [">c1", "ACGT" * 50]
        assert g2m_io.read_fasta(path) == [("c1", "ACGT" * 50)]
