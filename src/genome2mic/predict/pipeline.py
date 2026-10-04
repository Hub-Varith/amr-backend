"""Genome-to-report prediction pipeline (INTEGRATION.md section 3.1, DATA_CONTRACT.md stages 3-5 and 12).

    FASTA -> QC stats -> species (Mash) -> novelty (Mash) -> AMRFinderPlus -> known-AMR row
          -> model (MIC, band, p_active, call) -> overrides -> reasons, ranking -> report

Every step writes its output into the job's work folder, so a bad report can be traced to the
step that caused it. The known-AMR row is built by the same NcbiKnownAmrBuilder that built the
training features (features/amrfinder_table.py), so the naming rules exist only once.
"""

import gzip
import json
import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from genome2mic.features.amrfinder_table import AmrFinderHit, KnownAmrRow
from genome2mic.features.ncbi_known_amr import NcbiKnownAmrBuilder
from genome2mic.predict.amrfinder_runner import AmrFinderRunner
from genome2mic.predict.call_overrides import CallOverrides
from genome2mic.predict.drug_reasons import DrugReasons
from genome2mic.predict.genome_qc import AssemblyStats, GenomeQc
from genome2mic.predict.mic_model import MicModel
from genome2mic.predict.novelty_checker import NoveltyChecker
from genome2mic.predict.report_builder import ReportBuilder
from genome2mic.predict.species_identifier import SpeciesIdentifier
from genome2mic.predict.tool_runner import ToolRunner

logger = logging.getLogger(__name__)

class PredictionPipeline:
    """Runs one genome through QC, species ID, AMRFinderPlus, the model and the call logic."""

    def __init__(
        self,
        models_dir: Path,
        configs_dir: Path,
        model_run: str = "all5_run1",
        references_sketch: Path = Path("data/references/references.msh"),
        amrfinder_db: Path | None = None,
        threads: int = 4,
        keep_work_files: bool = False,
        runner: ToolRunner | None = None,
    ) -> None:
        self.models_dir = models_dir
        self.configs_dir = configs_dir
        self.run_dir = models_dir / model_run
        self.references_sketch = references_sketch
        self.keep_work_files = keep_work_files
        self.runner = runner or ToolRunner()
        self.amrfinder = AmrFinderRunner(self.runner, amrfinder_db, threads)
        self.model: MicModel | None = None

    def load(self) -> None:
        """Load configs, the model and the tool databases. Raises FileNotFoundError when something is missing."""
        if not (self.run_dir / "spec.json").exists():
            raise FileNotFoundError(f"no trained model at {self.run_dir} (get it from S3; docs/MODEL_HANDOFF.md section 10)")
        if not self.references_sketch.exists():
            raise FileNotFoundError(f"no Mash reference sketch at {self.references_sketch} (make references)")
        ToolRunner.require("mash", "amrfinder")

        self.species_config = yaml.safe_load((self.configs_dir / "species.yaml").read_text())["species"]
        drugs = yaml.safe_load((self.configs_dir / "drugs.yaml").read_text())["drugs"]
        tiers = {name: entry.get("tier", 9) for name, entry in drugs.items()}
        self.model = MicModel(self.run_dir, self.configs_dir)
        self.qc = GenomeQc({key: value["expected_genome_size"] for key, value in self.species_config.items()})
        self.species_identifier = SpeciesIdentifier(self.references_sketch, list(self.species_config), self.runner)
        self.novelty = NoveltyChecker(self.run_dir, self.runner)
        self.reports = ReportBuilder(CallOverrides.from_configs(self.configs_dir), DrugReasons.from_configs(self.configs_dir), tiers)

        # Same keep-variant list and AMRFinderPlus class tables as the training build.
        self.amrfinder_version, database_dir = self.amrfinder.describe_database()
        keep_variant = pd.read_csv(self.configs_dir / "keep_variant.csv")["family"].tolist()
        self.builders = {
            key: NcbiKnownAmrBuilder(keep_variant, NcbiKnownAmrBuilder.load_class_table(database_dir, value["amrfinder_organism"]))
            for key, value in self.species_config.items()
        }
        logger.info("Pipeline loaded", extra={"run_id": self.model.run_id, "amrfinder_db": self.amrfinder_version})

    def available_models(self) -> dict[str, list[str]]:
        if self.model is None:
            raise FileNotFoundError("pipeline not loaded")
        return self.model.drugs_by_species

    def run(self, fasta_path: Path, sample_id: str) -> dict[str, Any]:
        """Return one report as a dict that matches DATA_CONTRACT.md stage 12."""
        if self.model is None:
            raise RuntimeError("call load() first")
        work_dir = Path(tempfile.mkdtemp(prefix=f"{Path(fasta_path.name.removesuffix('.gz')).stem}_", dir=fasta_path.parent))
        try:
            if fasta_path.suffix.lower() == ".gz":
                # Tools get plain FASTA; the API already unpacks uploads, the CLI may pass .fasta.gz.
                plain = work_dir / "genome.fasta"
                with gzip.open(fasta_path, "rb") as reader, plain.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
                fasta_path = plain
            report = self._run(fasta_path, sample_id, work_dir)
            self._write(work_dir / "report.json", report)
            return report
        finally:
            if not self.keep_work_files:
                shutil.rmtree(work_dir, ignore_errors=True)
            else:
                logger.info("Work files kept", extra={"work_dir": str(work_dir)})

    def _run(self, fasta_path: Path, sample_id: str, work_dir: Path) -> dict[str, Any]:
        stats = AssemblyStats.from_fasta(fasta_path)
        species = self.species_identifier.identify(fasta_path) if stats.n_contigs else None
        species_key = species.species if species else None
        qc = self.qc.evaluate(stats, species_key, species.mash_distance if species else None)
        self._write(work_dir / "qc.json", {**asdict(stats), "qc_pass": qc.qc_pass, "qc_fail_reason": qc.qc_fail_reason,
                                           "mash_species": species_key, "mash_distance": species.mash_distance if species else None})

        if species_key is None or species_key not in self.model.drugs_by_species:
            logger.info("Species not covered; no predictions", extra={"sample_id": sample_id})
            return self._report(sample_id, None, qc.qc_pass, None, False, [], [])

        novelty = self.novelty.check(fasta_path, species_key)
        self._write(work_dir / "novelty.json", asdict(novelty))

        organism = self.species_config[species_key]["amrfinder_organism"]
        hits = self.amrfinder.run(fasta_path, organism, work_dir / "amrfinder.tsv")
        known_row = KnownAmrRow.from_hits(hits, species_key, self.builders[species_key])
        self._write(work_dir / "known_amr.json", self._feature_record(known_row, hits))

        results = self.model.predict(species_key, known_row)
        rows = self.reports.drug_rows(species_key, results, hits)
        return self._report(sample_id, species_key, qc.qc_pass, novelty.nearest_training_distance, novelty.in_range,
                            rows, self.reports.ranked_active(rows))

    def _feature_record(self, known_row: dict[str, int], hits: list[AmrFinderHit]) -> dict[str, Any]:
        model_columns = set(self.model.known_columns())
        return {
            "features": known_row,
            "used_by_model": sorted(column for column in known_row if column in model_columns),
            "not_in_model": sorted(column for column in known_row if column not in model_columns),
            "n_amrfinder_hits": len(hits),
            "amrfinder_db": self.amrfinder_version,
        }

    def _report(self, sample_id, species, qc_pass, distance, in_range, rows, ranked) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "species": species,
            "qc_pass": qc_pass,
            "nearest_training_distance": distance,
            "in_range": in_range,
            "predictions": rows,
            "ranked_active": ranked,
            "model_version": self.model.model_version,
            "run_id": self.model.run_id,
            # disclaimer: filled in by the PredictionReport schema (api/constants.py), never blank.
        }

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str))
