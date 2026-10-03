"""Genome-to-report prediction pipeline. Stub until stages 4-10 land."""

from pathlib import Path
from typing import Any


class PredictionPipeline:
    """Runs one genome through QC, AMRFinderPlus, unitig query, the models and the call logic."""

    def __init__(self, models_dir: Path, configs_dir: Path) -> None:
        self.models_dir = models_dir
        self.configs_dir = configs_dir

    def load(self) -> None:
        """Load every trained species x drug model into memory."""
        raise NotImplementedError("Model loading lands with stage 7.")

    def available_models(self) -> dict[str, list[str]]:
        """Return the trained drugs for each species key."""
        raise NotImplementedError("Model loading lands with stage 7.")

    def run(self, fasta_path: Path, sample_id: str) -> dict[str, Any]:
        """Return one report as a dict that matches DATA_CONTRACT.md stage 12."""
        raise NotImplementedError("The prediction pipeline lands with stage 10.")
