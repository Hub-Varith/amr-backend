"""The trained model run, from a known-AMR row to one called result per drug (MODEL_HANDOFF.md section 10)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.models.mic_predictor import MicPredictor
from genome2mic.models.model_artifact import ModelArtifact
from genome2mic.predict.call_thresholds import CallThresholds
from genome2mic.predict.drug_call import DrugCall
from genome2mic.predict.probability_calibrator import ProbabilityCalibrator
from genome2mic.predict.susceptibility_caller import SusceptibilityCaller

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrugResult:
    drug: str
    pred_mic: float
    band_low: float
    band_high: float
    call: DrugCall


class MicModel:
    """Loads models/<run>/ once: network, conformal bands, calibration curves and call thresholds."""

    def __init__(self, run_dir: Path, configs_dir: Path) -> None:
        self.run_dir = run_dir
        self.artifact = ModelArtifact.load(run_dir)
        self.predictor = MicPredictor(self.artifact)
        self.calibrator = ProbabilityCalibrator.from_dict(json.loads((run_dir / "probability_calibration.json").read_text()))
        thresholds = CallThresholds.from_dict(json.loads((run_dir / "call_thresholds.json").read_text()))
        self.caller = SusceptibilityCaller(BreakpointTable.from_directory(configs_dir / "breakpoints"), thresholds=thresholds)
        self.sigma_by_drug = dict(zip(self.artifact.drugs, np.exp(self.artifact.model.log_sigma.detach().numpy())))
        # The saved spec is the source of truth; the artifact recomputes run_id from its config.
        spec = json.loads((run_dir / "spec.json").read_text())
        self.run_id: str = spec["run_id"]
        self.model_version: str = f"{spec['model_version']}/{run_dir.name}"
        logger.info("Model loaded", extra={"run_dir": str(run_dir), "run_id": self.run_id})

    @property
    def drugs_by_species(self) -> dict[str, list[str]]:
        return self.artifact.drugs_by_species

    def known_columns(self) -> list[str]:
        return list(self.artifact.known_columns)

    def predict(self, species: str, known_row: dict[str, int]) -> list[DrugResult]:
        """Every drug the model reports for this species. Columns not in the model are ignored (and logged)."""
        vector = self.predictor.known_vector(known_row).reshape(1, -1)
        table = self.predictor.predict(species, vector, None)
        results = []
        for row in table.itertuples():
            p_active = None
            found = self.caller.breakpoints.lookup(species, row.drug, self.caller.standard, self.caller.year)
            if found is not None and self.calibrator.has_pair(species, row.drug):
                p_raw = SusceptibilityCaller.probability_active(row.mu_log2, self.sigma_by_drug[row.drug], found[0])
                p_active = float(self.calibrator.calibrate(species, row.drug, np.array([p_raw]))[0])
            call = self.caller.call(species, row.drug, row.band_low, row.band_high, p_active)
            results.append(DrugResult(row.drug, float(row.pred_mic), float(row.band_low), float(row.band_high), call))
        return results
