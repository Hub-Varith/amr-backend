"""FAKE predictor for API tests. Its reports contain no MIC values."""

from pathlib import Path

from genome2mic.api.schemas.call import Call
from genome2mic.api.schemas.drug_prediction import DrugPrediction
from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.api.services.predictor import Predictor

FAKE_REASON = "FAKE FIXTURE - not a real prediction"


class FakePredictor(Predictor):
    """Returns an obviously fake report, or raises when should_fail is set."""

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.seen_paths: list[Path] = []

    async def predict(self, fasta_path: Path, sample_id: str) -> PredictionReport:
        assert fasta_path.exists(), "Upload must still exist while the job runs."
        self.seen_paths.append(fasta_path)
        if self.should_fail:
            raise RuntimeError("FAKE failure for tests")
        fake_prediction = DrugPrediction(
            drug="fake-drug",
            pred_mic=None,
            band_low=None,
            band_high=None,
            call=Call.UNCERTAIN,
            margin_steps=None,
            reasons=[FAKE_REASON],
        )
        return PredictionReport(
            sample_id=sample_id,
            species=None,
            qc_pass=True,
            nearest_training_distance=None,
            in_range=False,
            predictions=[fake_prediction],
            ranked_active=[],
            model_version="fake-0.0",
            run_id="fake-run",
        )
