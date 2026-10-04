"""Calibrates the model's raw P(active) so a shown chance matches the observed rate."""

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """One monotone curve (isotonic regression) per species x drug, fitted on out-of-fold predictions.

    The curve never changes the order of two genomes, only the size of the number.
    """

    def __init__(self, curves: dict[tuple[str, str], tuple[list[float], list[float]]]) -> None:
        self.curves = curves

    @classmethod
    def fit(cls, rows: pd.DataFrame, min_per_class: int) -> "ProbabilityCalibrator":
        """rows: species, drug, p_raw, lab_resistant (bool). Pairs with too few R or S get no curve."""
        logger.info("ProbabilityCalibrator fit started: rows=%s", len(rows))
        curves = {}
        for (species, drug), group in rows.groupby(["species", "drug"]):
            lab_active = ~group["lab_resistant"].to_numpy(dtype=bool)
            if lab_active.sum() < min_per_class or (~lab_active).sum() < min_per_class:
                logger.warning("No calibration curve for %s %s: active=%s resistant=%s (need %s each)",
                               species, drug, int(lab_active.sum()), int((~lab_active).sum()), min_per_class)
                continue
            isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            isotonic.fit(group["p_raw"].to_numpy(), lab_active.astype(float))
            curves[(species, drug)] = (isotonic.X_thresholds_.tolist(), isotonic.y_thresholds_.tolist())
        logger.info("ProbabilityCalibrator fit done: pairs=%s", len(curves))
        return cls(curves)

    def has_pair(self, species: str, drug: str) -> bool:
        return (species, drug) in self.curves

    def calibrate(self, species: str, drug: str, p_raw: np.ndarray) -> np.ndarray:
        """Calibrated probabilities. A pair without a curve is returned unchanged."""
        if (species, drug) not in self.curves:
            return np.asarray(p_raw, dtype=float)
        raw_points, calibrated_points = self.curves[(species, drug)]
        return np.interp(p_raw, raw_points, calibrated_points)

    def to_dict(self) -> dict:
        pairs = []
        for (species, drug), (raw_points, calibrated_points) in self.curves.items():
            pairs.append({"species": species, "drug": drug, "raw": raw_points, "calibrated": calibrated_points})
        return {"pairs": pairs}

    @classmethod
    def from_dict(cls, payload: dict) -> "ProbabilityCalibrator":
        curves = {}
        for pair in payload["pairs"]:
            curves[(pair["species"], pair["drug"])] = (pair["raw"], pair["calibrated"])
        return cls(curves)
