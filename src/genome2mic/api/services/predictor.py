"""Async wrapper around the prediction pipeline."""

import asyncio
import logging
from pathlib import Path

from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.predict.pipeline import PredictionPipeline

logger = logging.getLogger(__name__)


class Predictor:
    """Runs the pipeline off the event loop and checks its output against the report schema."""

    def __init__(self, pipeline: PredictionPipeline) -> None:
        self.pipeline = pipeline

    async def predict(self, fasta_path: Path, sample_id: str) -> PredictionReport:
        """Predict one genome. Raises if the pipeline fails or returns an invalid report."""
        logger.info("Prediction started", extra={"sample_id": sample_id})
        raw_report = await asyncio.to_thread(self.pipeline.run, fasta_path, sample_id)
        report = PredictionReport.model_validate(raw_report)
        logger.info(
            "Prediction finished",
            extra={"sample_id": sample_id, "n_predictions": len(report.predictions)},
        )
        return report
