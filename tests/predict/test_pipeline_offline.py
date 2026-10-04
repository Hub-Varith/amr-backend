"""The whole pipeline with Mash and AMRFinderPlus faked and the real trained model (skipped when absent)."""

import shutil
from pathlib import Path

import pytest

from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.predict.pipeline import PredictionPipeline
from genome2mic.predict.tool_runner import ToolRunner

ROOT = Path(__file__).parents[2]
RUN_DIR = ROOT / "models" / "all5_run1"
FIXTURE = ROOT / "tests" / "fixtures" / "amrfinder" / "kpneu_example.tsv"


class FakeTools:
    """Answers the three tool calls the pipeline makes."""

    def __init__(self, database_dir: Path, mash_line: str) -> None:
        self.database_dir = database_dir
        self.mash_line = mash_line
        self.calls: list[list[str]] = []

    def run(self, arguments, cwd=None):
        self.calls.append(arguments)
        if arguments[0] == "mash":
            return self.mash_line
        if "--database_version" in arguments:
            return f"Database directory: '{self.database_dir}'\nDatabase version: 2026-08-07.1\n"
        shutil.copy(FIXTURE, arguments[arguments.index("-o") + 1])
        return ""


@pytest.fixture
def pipeline_parts(tmp_path, monkeypatch):
    if not (RUN_DIR / "spec.json").exists():
        pytest.skip("trained model not downloaded (docs/MODEL_HANDOFF.md section 10)")
    monkeypatch.setattr(ToolRunner, "require", staticmethod(lambda *names: None))
    database = tmp_path / "db"
    database.mkdir()
    (database / "fam.tsv").write_text("#node_id\tclass\nblaKPC\tBETA-LACTAM\nblaSHV\tBETA-LACTAM\nemrD\tEFFLUX\naadA2\tAMINOGLYCOSIDE\n")
    (database / "AMRProt-mutation.tsv").write_text("standard_mutation_symbol\tclass\ngyrA_S83I\tQUINOLONE\n")
    sketch = tmp_path / "references.msh"
    sketch.write_text("fake")
    fasta = tmp_path / "BC-0001.fasta"
    fasta.write_text(">contig_1\n" + "ACGTTGCA" * 700_000 + "\n")      # 5.6 Mb, one contig
    return tmp_path, database, sketch, fasta


def make_pipeline(sketch, tools):
    pipeline = PredictionPipeline(RUN_DIR.parent, ROOT / "configs", references_sketch=sketch, runner=tools)
    pipeline.load()
    return pipeline


def test_kpneu_upload_gives_a_valid_report(pipeline_parts):
    tmp_path, database, sketch, fasta = pipeline_parts
    tools = FakeTools(database, "refs/KPNEU.fna.gz\tq\t0.008\t0\t900/1000\n")
    raw = make_pipeline(sketch, tools).run(fasta, "BC-0001")
    report = PredictionReport.model_validate(raw)

    assert report.species.value == "KPNEU" and report.qc_pass
    assert report.nearest_training_distance is None and report.in_range      # no training sketch shipped
    by_drug = {row.drug: row for row in report.predictions}
    assert set(by_drug) == set(make_pipeline(sketch, tools).available_models()["KPNEU"])
    assert by_drug["ampicillin"].override.value == "natural_resistance" and by_drug["ampicillin"].pred_mic is None
    assert by_drug["ertapenem"].override.value == "strong_marker" and by_drug["ertapenem"].reasons == ["blaKPC-2"]
    assert by_drug["meropenem"].call.value == "likely_inactive"               # KPC-2: the model says so, no override
    assert "gyrA S83I" in by_drug["ciprofloxacin"].reasons
    assert all(by_drug[drug].call.value == "likely_active" for drug in report.ranked_active)
    amrfinder_call = next(call for call in tools.calls if call[0] == "amrfinder" and "-n" in call)
    assert amrfinder_call[amrfinder_call.index("-O") + 1] == "Klebsiella_pneumoniae" and "--plus" in amrfinder_call
    assert not list(tmp_path.glob("BC-0001_*"))                              # work folder cleaned up


def test_species_out_of_scope_stops_early(pipeline_parts):
    _, database, sketch, fasta = pipeline_parts
    tools = FakeTools(database, "refs/KPNEU.fna.gz\tq\t0.21\t0\t3/1000\n")
    report = PredictionReport.model_validate(make_pipeline(sketch, tools).run(fasta, "X"))
    assert report.species is None and not report.in_range and report.predictions == []
    assert not any(call[0] == "amrfinder" and "-n" in call for call in tools.calls)


def test_gzipped_fasta_is_unpacked_before_the_tools(pipeline_parts):
    import gzip

    tmp_path, database, sketch, fasta = pipeline_parts
    zipped = tmp_path / "BC-0002.fasta.gz"
    zipped.write_bytes(gzip.compress(fasta.read_bytes()))
    tools = FakeTools(database, "refs/KPNEU.fna.gz\tq\t0.008\t0\t900/1000\n")
    report = PredictionReport.model_validate(make_pipeline(sketch, tools).run(zipped, "BC-0002"))
    assert report.species.value == "KPNEU" and report.qc_pass
    amrfinder_input = next(call for call in tools.calls if call[0] == "amrfinder" and "-n" in call)
    assert amrfinder_input[amrfinder_input.index("-n") + 1].endswith("genome.fasta")
