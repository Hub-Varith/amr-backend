"""Conversions between mg/L and log2 doubling steps."""

import numpy as np


class MicSteps:
    """Static helpers for the log2 MIC scale used by the model."""

    @staticmethod
    def to_log2(mic: np.ndarray) -> np.ndarray:
        """mg/L -> log2 steps. 0 becomes -inf (left-censored), inf stays inf."""
        mic = np.asarray(mic, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return np.log2(mic)

    @staticmethod
    def round_up_to_step(log2_mic: np.ndarray) -> np.ndarray:
        """Round a log2 prediction up to the next whole doubling step and return mg/L.

        A slightly high MIC is the safer error (CLAUDE.md rule 9). The small
        epsilon keeps a prediction that is already on a step from jumping up one.
        """
        steps = np.ceil(np.asarray(log2_mic, dtype=np.float64) - 1e-9)
        return np.power(2.0, steps)

    @staticmethod
    def step_difference(pred_mic: np.ndarray, lab_mic: np.ndarray) -> np.ndarray:
        """Signed difference in doubling steps, positive when the prediction is higher."""
        return np.log2(np.asarray(pred_mic, dtype=np.float64)) - np.log2(np.asarray(lab_mic, dtype=np.float64))
