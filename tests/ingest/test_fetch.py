"""Tests for ``genome2mic.ingest.fetch`` (stage 0: bucket/directory -> data/raw).

No cloud bucket or provider CLI is available here, so the cloud paths are tested at
the command-construction layer (``build_command``) with ``shutil.which`` and
``subprocess.run`` monkeypatched; the local-directory provider runs for real.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from genome2mic import cli
from genome2mic.errors import ToolNotAvailable
from genome2mic.ingest import fetch
from genome2mic.paths import Paths

AST_CSV = (
    "genome_id,genome_name,antibiotic,resistant_phenotype,measurement_sign,measurement_value,"
    "measurement_unit,laboratory_typing_method,testing_standard,testing_standard_year,evidence\n"
    "573.2002,Klebsiella pneumoniae,meropenem,Resistant,=,8,mg/L,Broth dilution,CLSI,2016,Laboratory Method\n"
)
FASTA_A = ">573.2002_1\nACGTACGTACGT\n"
FASTA_B = ">573.2005_1\nTTTTGGGGCCCC\n>573.2005_2\nAAAA\n"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def canonical_source(tmp_path: Path) -> Path:
    """A bucket stand-in laid out exactly as the contract expects."""
    src = tmp_path / "bucket"
    write(src / "ast_bvbrc.csv", AST_CSV)
    write(src / "genome_metadata.csv", "genome_id,biosample,species\n573.2002,SAMN1,KPNEU\n")
    write(src / "genomes" / "573.2002.fasta", FASTA_A)
    write(src / "genomes" / "573.2005.fasta", FASTA_B)
    write(src / "notes.txt", "not part of the pipeline\n")
    return src


def fna_source(tmp_path: Path) -> Path:
    """A bucket stand-in with .fna names, one genome in a sub-folder and one gzip."""
    import gzip

    src = tmp_path / "bucket_fna"
    write(src / "ast_bvbrc.csv", AST_CSV)
    write(src / "genomes" / "573.2002.fna", FASTA_A)
    write(src / "genomes" / "573.2005" / "573.2005_assembly.fna", FASTA_B)
    gz = src / "genomes" / "573.2010.fasta.gz"
    gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write(">573.2010_1\nGGGG\n")
    return src


def make_paths(tmp_path: Path) -> Paths:
    return Paths(root=tmp_path / "root", configs_dir=tmp_path / "configs")


def tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """No provider CLI on PATH."""
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None)


@pytest.fixture
def forbid_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:  # pragma: no cover - only on failure
        raise AssertionError(f"subprocess.run must not be called: {args} {kwargs}")

    monkeypatch.setattr(subprocess, "run", boom)


# --------------------------------------------------------------------------- #
# local directory provider
# --------------------------------------------------------------------------- #


class TestLocalSync:
    def test_syncs_every_file_and_counts(self, tmp_path: Path, forbid_subprocess: None) -> None:
        src = canonical_source(tmp_path)
        paths = make_paths(tmp_path)

        summary = fetch.run(paths, str(src), include=None, dry_run=False)

        assert tree(paths.raw_dir) >= {"ast_bvbrc.csv", "genome_metadata.csv", "genomes/573.2002.fasta",
                                       "genomes/573.2005.fasta", "notes.txt"}
        assert (paths.raw_dir / "genomes" / "573.2002.fasta").read_text() == FASTA_A
        assert summary.provider == "local" and summary.tool is None and not summary.dry_run
        assert summary.n_genomes == 2
        assert summary.n_ast_files == 1 and summary.ast_files == ["ast_bvbrc.csv"]
        # counts are taken before fetch_summary.json is written: synced data only
        synced = [p for p in tree(paths.raw_dir) if p != fetch.SUMMARY_NAME]
        assert summary.n_files == len(synced) == 5
        assert summary.total_bytes == sum((src / rel).stat().st_size for rel in tree(src))
        # canonical layout: no manifest, no layout problems
        assert summary.manifest is None
        assert not (paths.raw_dir / fetch.MANIFEST_NAME).exists()
        assert summary.problems == []
        # the stage writes a redacted summary file
        payload = json.loads((paths.raw_dir / fetch.SUMMARY_NAME).read_text())
        assert payload["n_genomes"] == 2 and payload["provider"] == "local"

    def test_file_uri_is_accepted(self, tmp_path: Path) -> None:
        src = canonical_source(tmp_path)
        paths = make_paths(tmp_path)
        summary = fetch.run(paths, src.as_uri(), include=["ast_*.csv"], dry_run=False)
        assert (paths.raw_dir / "ast_bvbrc.csv").is_file()
        assert summary.source == str(src.resolve())

    def test_second_run_skips_unchanged_files(self, tmp_path: Path) -> None:
        src = canonical_source(tmp_path)
        paths = make_paths(tmp_path)
        fetch.run(paths, str(src), include=None, dry_run=False)
        result = fetch.sync_local(src, paths.raw_dir, None, dry_run=False)
        assert result.n_copied == 0 and result.n_skipped == result.n_selected == 5

    def test_include_patterns_select_subsets(self, tmp_path: Path) -> None:
        src = canonical_source(tmp_path)
        paths = make_paths(tmp_path)

        summary = fetch.run(paths, str(src), include=["ast_*.csv"], dry_run=False)
        copied = tree(paths.raw_dir) - {fetch.SUMMARY_NAME}
        assert copied == {"ast_bvbrc.csv"}
        assert summary.n_genomes == 0
        assert any("no FASTA files" in p for p in summary.problems)

        summary = fetch.run(paths, str(src), include=["genomes/*.fasta"], dry_run=False)
        copied = tree(paths.raw_dir) - {fetch.SUMMARY_NAME}
        assert copied == {"ast_bvbrc.csv", "genomes/573.2002.fasta", "genomes/573.2005.fasta"}
        assert summary.n_genomes == 2 and summary.problems == []

    def test_star_crosses_directory_boundaries(self) -> None:
        # AWS semantics: '*' matches '/' too, so one pattern covers nested genome folders.
        assert fetch.matches_include("genomes/573.2005/573.2005.fna", ["genomes/*.fna"])
        assert fetch.matches_include("genomes/x.fasta", ["*.fasta"])
        assert not fetch.matches_include("notes.txt", ["ast_*.csv", "genomes/*.fasta"])
        assert fetch.matches_include("anything", None) and fetch.matches_include("anything", [])

    def test_dry_run_writes_nothing(self, tmp_path: Path, forbid_subprocess: None) -> None:
        src = canonical_source(tmp_path)
        paths = make_paths(tmp_path)

        summary = fetch.run(paths, str(src), include=None, dry_run=True)

        assert summary.dry_run is True
        assert not paths.raw_dir.exists()
        assert summary.command[0] == "<python copy>" and str(paths.raw_dir) in summary.command
        assert summary.n_files == 0 and summary.problems == []

    def test_missing_local_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            fetch.run(make_paths(tmp_path), str(tmp_path / "nope"), include=None, dry_run=False)


# --------------------------------------------------------------------------- #
# non-canonical genome layouts -> manifest, resolve_genome_paths, symlinks
# --------------------------------------------------------------------------- #


class TestGenomeLayout:
    def test_manifest_written_for_fna_and_subfolders(self, tmp_path: Path) -> None:
        src = fna_source(tmp_path)
        paths = make_paths(tmp_path)

        summary = fetch.run(paths, str(src), include=None, dry_run=False)

        manifest = paths.raw_dir / fetch.MANIFEST_NAME
        assert summary.manifest == str(manifest) and manifest.is_file()
        table = pd.read_csv(manifest, dtype=str)
        assert list(table.columns) == list(fetch.MANIFEST_COLUMNS)
        rows = dict(zip(table["genome_id"], table["path"]))
        assert rows == {
            "573.2002": "data/raw/genomes/573.2002.fna",
            "573.2005": "data/raw/genomes/573.2005/573.2005_assembly.fna",
            "573.2010": "data/raw/genomes/573.2010.fasta.gz",
        }
        assert table["bytes"].astype(int).tolist() == [
            (paths.raw_dir / "genomes" / "573.2002.fna").stat().st_size,
            (paths.raw_dir / "genomes" / "573.2005" / "573.2005_assembly.fna").stat().st_size,
            (paths.raw_dir / "genomes" / "573.2010.fasta.gz").stat().st_size,
        ]
        assert summary.n_genomes == 3
        # files were NOT renamed
        assert (paths.raw_dir / "genomes" / "573.2002.fna").is_file()
        assert not (paths.raw_dir / "genomes" / "573.2002.fasta").exists()
        assert any("not genomes/<genome_id>.fasta" in p for p in summary.problems)
        assert summary.n_links == 0

    def test_resolve_genome_paths_reads_manifest_or_globs(self, tmp_path: Path) -> None:
        paths = make_paths(tmp_path)
        # no manifest, no directory -> empty
        assert fetch.resolve_genome_paths(paths) == {}
        # canonical layout without manifest -> glob *.fasta (same rule as qc.list_genome_fastas)
        write(paths.genomes_dir / "573.2002.fasta", FASTA_A)
        write(paths.genomes_dir / "573.2005.fna", FASTA_B)
        assert fetch.resolve_genome_paths(paths) == {"573.2002": paths.genomes_dir / "573.2002.fasta"}
        # with a manifest -> exactly its rows, relative paths anchored at root
        write(paths.raw_dir / fetch.MANIFEST_NAME,
              "genome_id,path,bytes\n573.2002,data/raw/genomes/573.2002.fasta,13\n573.2005,data/raw/genomes/573.2005.fna,30\n")
        resolved = fetch.resolve_genome_paths(paths)
        assert resolved == {
            "573.2002": paths.root / "data/raw/genomes/573.2002.fasta",
            "573.2005": paths.root / "data/raw/genomes/573.2005.fna",
        }
        assert all(p.is_file() for p in resolved.values())

    def test_link_canonical_adds_symlinks_but_skips_gzip(self, tmp_path: Path) -> None:
        src = fna_source(tmp_path)
        paths = make_paths(tmp_path)

        summary = fetch.run(paths, str(src), include=None, dry_run=False, link_canonical=True)

        link_a = paths.genomes_dir / "573.2002.fasta"
        link_b = paths.genomes_dir / "573.2005.fasta"
        assert link_a.is_symlink() and link_a.read_text() == FASTA_A
        assert link_b.is_symlink() and link_b.read_text() == FASTA_B
        assert not os.path.isabs(os.readlink(link_a))  # relative: survives moving the root
        assert not (paths.genomes_dir / "573.2010.fasta").exists()
        assert summary.n_links == 2
        assert any("gzip" in p for p in summary.problems)
        # the originals are untouched and the manifest still points at them
        rows = pd.read_csv(paths.raw_dir / fetch.MANIFEST_NAME, dtype=str)
        assert rows["path"].tolist()[0] == "data/raw/genomes/573.2002.fna"
        # re-running is idempotent (existing links are recognised, not duplicated)
        again = fetch.run(paths, str(src), include=None, dry_run=False, link_canonical=True)
        assert again.n_links == 0 and again.n_genomes == 3

    def test_scan_handles_duplicates_and_crowded_folders(self, tmp_path: Path) -> None:
        genomes = tmp_path / "genomes"
        write(genomes / "573.2002.fasta", FASTA_A)
        write(genomes / "573.2002.fna", FASTA_A)  # same id, different file
        write(genomes / "573.2005" / "a.fna", FASTA_B)
        write(genomes / "573.2005" / "b.fna", FASTA_B)
        write(genomes / "README.md", "ignored\n")
        problems: list[str] = []
        found = fetch.scan_genomes(genomes, problems)
        assert [g.genome_id for g in found] == ["573.2002", "a", "b"]
        assert found[0].canonical and found[0].path.name == "573.2002.fasta"
        assert any("duplicate genome_id '573.2002'" in p for p in problems)
        assert any("573.2005/ holds 2 FASTA files" in p for p in problems)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("573.2002.fasta", "573.2002"), ("573.2002.fna", "573.2002"), ("x.fa", "x"), ("x.fas.gz", "x"),
         ("x.fasta.gz", "x"), ("ast_bvbrc.csv", None), (".fasta", None), ("x.txt", None)],
    )
    def test_strip_fasta_suffix(self, name: str, expected: str | None) -> None:
        assert fetch.strip_fasta_suffix(name) == expected


# --------------------------------------------------------------------------- #
# layout validation of the AST files
# --------------------------------------------------------------------------- #


class TestAstLayout:
    def test_missing_ast_file_is_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "bucket"
        write(src / "genomes" / "573.2002.fasta", FASTA_A)
        summary = fetch.run(make_paths(tmp_path), str(src), include=None, dry_run=False)
        assert summary.n_ast_files == 0
        assert any("no ast_*.csv" in p for p in summary.problems)

    def test_misnamed_ast_file_is_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "bucket"
        write(src / "ast_export_2026.csv", AST_CSV)
        write(src / "genomes" / "573.2002.fasta", FASTA_A)
        summary = fetch.run(make_paths(tmp_path), str(src), include=None, dry_run=False)
        assert summary.ast_files == ["ast_export_2026.csv"]
        assert any("ingest reads only those names" in p for p in summary.problems)


# --------------------------------------------------------------------------- #
# cloud providers: scheme detection, command construction, tool detection
# --------------------------------------------------------------------------- #


class TestSourceParsing:
    def test_unknown_scheme_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsupported source URI scheme 'ftp'"):
            fetch.run(make_paths(tmp_path), "ftp://host/bucket", include=None, dry_run=True)

    def test_plain_https_that_is_not_azure_blob_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported"):
            fetch.parse_source("https://example.com/data")

    def test_providers_from_scheme(self) -> None:
        assert fetch.parse_source("gs://bucket/prefix/").provider == "gcs"
        assert fetch.parse_source("gs://bucket/prefix/").location == "gs://bucket/prefix"
        assert fetch.parse_source("s3://bucket/prefix").provider == "s3"
        az = fetch.parse_source("az://acct/container/raw/")
        assert az.provider == "azure" and az.location == "https://acct.blob.core.windows.net/container/raw"
        https = fetch.parse_source("https://acct.blob.core.windows.net/container/raw?sv=1&sig=SECRET")
        assert https.provider == "azure" and "SECRET" in https.location  # the CLI needs the SAS token
        assert "SECRET" not in https.display

    def test_az_uri_needs_account_and_container(self) -> None:
        with pytest.raises(ValueError, match="az://<account>/<container>"):
            fetch.parse_source("az://acct")

    def test_redaction(self) -> None:
        assert fetch.redact_uri("/data/raw") == "/data/raw"
        assert fetch.redact_uri("gs://bucket/prefix") == "gs://bucket/prefix"
        redacted = fetch.redact_uri("https://user:pw@acct.blob.core.windows.net/c/p?sv=1&sig=SECRET#frag")
        assert "SECRET" not in redacted and "pw" not in redacted
        assert redacted.startswith("https://<redacted>@acct.blob.core.windows.net/c/p?<redacted>")


class TestCommands:
    DEST = Path("/data/g2m/data/raw")

    def test_gcloud_include_becomes_negative_lookahead_exclude(self) -> None:
        src = fetch.parse_source("gs://bucket/prefix")
        cmd = fetch.build_command(src, self.DEST, ["ast_*.csv", "genomes/*.fasta"], "gcloud")
        assert cmd[:6] == ["gcloud", "storage", "rsync", "-r", "gs://bucket/prefix", str(self.DEST)]
        assert cmd[6].startswith("--exclude=^(?!")
        import re

        regex = re.compile(cmd[6][len("--exclude="):])
        assert not regex.match("ast_bvbrc.csv")  # kept
        assert not regex.match("genomes/573.2002.fasta")  # kept
        assert regex.match("notes.txt")  # excluded
        assert regex.match("genomes/573.2002.fna")  # excluded (different suffix)
        # no filter flags at all when include is empty
        assert fetch.build_command(src, self.DEST, None, "gcloud") == cmd[:6]

    def test_gsutil_fallback_command(self) -> None:
        src = fetch.parse_source("gs://bucket/prefix")
        cmd = fetch.build_command(src, self.DEST, ["ast_*.csv"], "gsutil")
        assert cmd[:4] == ["gsutil", "-m", "rsync", "-r"]
        assert cmd[4] == "-x" and cmd[5].startswith("^(?!")
        assert cmd[-2:] == ["gs://bucket/prefix", str(self.DEST)]

    def test_include_regex_rejects_commas(self) -> None:
        with pytest.raises(ValueError, match="comma"):
            fetch.include_regex(["a,b"])

    def test_aws_sync_command(self) -> None:
        src = fetch.parse_source("s3://bucket/prefix")
        assert fetch.build_command(src, self.DEST, None, "aws") == ["aws", "s3", "sync", "s3://bucket/prefix", str(self.DEST)]
        cmd = fetch.build_command(src, self.DEST, ["ast_*.csv", "genomes/*.fasta"], "aws")
        assert cmd[5:] == ["--exclude", "*", "--include", "ast_*.csv", "--include", "genomes/*.fasta"]

    def test_azcopy_sync_command_uses_basenames(self) -> None:
        src = fetch.parse_source("az://acct/container/raw")
        cmd = fetch.build_command(src, self.DEST, ["ast_*.csv", "genomes/*.fasta", "*.fasta"], "azcopy")
        assert cmd[:5] == ["azcopy", "sync", "https://acct.blob.core.windows.net/container/raw", str(self.DEST), "--recursive=true"]
        assert cmd[5:] == ["--include-pattern", "ast_*.csv;*.fasta"]

    def test_redact_command_hides_sas_tokens(self) -> None:
        src = fetch.parse_source("https://acct.blob.core.windows.net/c/raw?sig=SECRET")
        cmd = fetch.build_command(src, self.DEST, None, "azcopy")
        assert "SECRET" in cmd[2]
        assert "SECRET" not in " ".join(fetch.redact_command(cmd))


class TestToolDetection:
    def test_missing_cli_raises_tool_not_available(self, tmp_path: Path, no_tools: None, forbid_subprocess: None) -> None:
        with pytest.raises(ToolNotAvailable) as info:
            fetch.run(make_paths(tmp_path), "s3://bucket/prefix", include=None, dry_run=False)
        assert info.value.tool == "aws"
        assert "aws configure" in str(info.value)

    def test_missing_gcloud_and_gsutil_raise_with_hint(self, tmp_path: Path, no_tools: None) -> None:
        with pytest.raises(ToolNotAvailable) as info:
            fetch.run(make_paths(tmp_path), "gs://bucket/prefix", include=None, dry_run=False)
        assert info.value.tool == "gcloud" and "gcloud auth login" in str(info.value)

    def test_missing_azcopy_raises(self, tmp_path: Path, no_tools: None) -> None:
        with pytest.raises(ToolNotAvailable) as info:
            fetch.run(make_paths(tmp_path), "az://acct/container", include=None, dry_run=False)
        assert info.value.tool == "azcopy"

    def test_gsutil_is_used_when_gcloud_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/usr/bin/gsutil" if name == "gsutil" else None)
        assert fetch.select_tool("gcs") == "gsutil"
        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
        assert fetch.select_tool("gcs") == "gcloud"
        assert fetch.select_tool("local") is None

    def test_dry_run_still_shows_command_without_the_cli(self, tmp_path: Path, no_tools: None, forbid_subprocess: None) -> None:
        summary = fetch.run(make_paths(tmp_path), "gs://bucket/prefix", include=["ast_*.csv"], dry_run=True)
        assert summary.tool == "gcloud" and summary.command[:3] == ["gcloud", "storage", "rsync"]
        assert not (tmp_path / "root").exists()


class TestExecution:
    def test_cloud_sync_runs_the_command_and_validates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        paths = make_paths(tmp_path)
        calls: list[list[str]] = []

        def fake_run(command: list[str], check: bool = False, **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(list(command))
            # emulate the CLI populating data/raw
            write(paths.raw_dir / "ast_bvbrc.csv", AST_CSV)
            write(paths.genomes_dir / "573.2002.fasta", FASTA_A)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
        monkeypatch.setattr(subprocess, "run", fake_run)

        summary = fetch.run(paths, "s3://bucket/prefix", include=["ast_*.csv", "genomes/*.fasta"], dry_run=False)

        assert calls == [["aws", "s3", "sync", "s3://bucket/prefix", str(paths.raw_dir),
                          "--exclude", "*", "--include", "ast_*.csv", "--include", "genomes/*.fasta"]]
        assert summary.tool == "aws" and summary.n_genomes == 1 and summary.n_ast_files == 1
        assert summary.problems == []

    def test_cli_failure_raises_fetch_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
        monkeypatch.setattr(subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 2))
        with pytest.raises(fetch.FetchError, match="aws exited with status 2"):
            fetch.run(make_paths(tmp_path), "s3://bucket/prefix", include=None, dry_run=False)

    def test_sas_token_never_reaches_logs_or_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        paths = make_paths(tmp_path)
        seen: list[str] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen.append(command[2])
            write(paths.raw_dir / "ast_bvbrc.csv", AST_CSV)
            write(paths.genomes_dir / "573.2002.fasta", FASTA_A)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
        monkeypatch.setattr(subprocess, "run", fake_run)
        uri = "https://acct.blob.core.windows.net/container/raw?sv=2024&sig=SECRETTOKEN"

        with caplog.at_level(logging.DEBUG, logger="genome2mic"):
            summary = fetch.run(paths, uri, include=None, dry_run=False)

        assert seen == [uri]  # the real token went to the CLI ...
        assert "SECRETTOKEN" not in caplog.text  # ... but not to the logs,
        assert "SECRETTOKEN" not in json.dumps(summary.to_dict())  # the summary,
        assert "SECRETTOKEN" not in (paths.raw_dir / fetch.SUMMARY_NAME).read_text()  # or the disk.


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


class TestCli:
    def test_fetch_subcommand_dry_run_prints_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], forbid_subprocess: None
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
        root = tmp_path / "root"
        code = cli.main([
            "fetch", "--root", str(root), "--source-uri", "gs://bucket/prefix",
            "--include", "ast_*.csv", "--include", "genomes/*.fasta", "--dry-run", "-q",
        ])
        out = capsys.readouterr().out
        assert code == 0
        assert "gcloud storage rsync -r gs://bucket/prefix" in out
        assert "--exclude=" in out and "nothing was written" in out
        assert not root.exists()

    def test_fetch_subcommand_local_prints_counts_and_warnings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = fna_source(tmp_path)
        root = tmp_path / "root"
        code = cli.main(["fetch", "--root", str(root), "--source-uri", str(src), "--link-canonical", "-q"])
        out = capsys.readouterr().out
        assert code == 0
        assert "3 genomes, 1 AST file(s) ['ast_bvbrc.csv']" in out
        assert "manifest ->" in out and "LAYOUT WARNING" in out
        assert (root / "data" / "raw" / "genomes" / "573.2002.fasta").is_symlink()

    def test_source_uri_is_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["fetch", "--root", "x"])

    def test_run_all_runs_fetch_first_and_skips_synth(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        order: list[str] = []
        seen_fetch: dict[str, object] = {}

        def record(name: str):
            def fn(ns: cli.argparse.Namespace) -> int:
                order.append(name)
                if name == "fetch":
                    seen_fetch.update(vars(ns))
                return 0
            return fn

        for name, attr in (("synth", "cmd_synth"), ("fetch", "cmd_fetch"), ("ingest", "cmd_ingest"), ("qc", "cmd_qc"),
                           ("known-amr", "cmd_known_amr"), ("lineages", "cmd_lineages"), ("splits", "cmd_splits"),
                           ("unitigs", "cmd_unitigs"), ("train", "cmd_train"), ("evaluate", "cmd_evaluate"),
                           ("report", "cmd_report")):
            monkeypatch.setattr(cli, attr, record(name))

        code = cli.main(["run-all", "--root", str(tmp_path), "--source-uri", "gs://bucket/prefix",
                         "--include", "ast_*.csv", "--link-canonical", "-q"])
        assert code == 0
        assert order == ["fetch", *cli.PIPELINE_STAGES]
        assert seen_fetch["source_uri"] == "gs://bucket/prefix"
        assert seen_fetch["include"] == ["ast_*.csv"]
        assert seen_fetch["dry_run"] is False and seen_fetch["link_canonical"] is True

        order.clear()
        code = cli.main(["run-all", "--root", str(tmp_path), "-q"])
        assert code == 0
        assert order == ["synth", *cli.PIPELINE_STAGES]
