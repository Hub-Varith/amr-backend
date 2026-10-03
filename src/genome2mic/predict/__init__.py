"""Prediction for a new genome: QC, known-AMR detection, per-drug MIC models, calls and ranking.

- :mod:`genome2mic.predict.pipeline` -- :class:`~genome2mic.predict.pipeline.PredictionPipeline`
  (``load``, ``available_models``, ``run``), the entry point used by the API.
- :mod:`genome2mic.predict.amr_detect` -- known-AMR detection backends and feature naming.
- :mod:`genome2mic.predict.rank` -- call logic, overrides, reasons and ranking.
"""

from genome2mic.predict.pipeline import PredictionPipeline
from genome2mic.predict.rank import rank_active

__all__ = ["PredictionPipeline", "rank_active"]
