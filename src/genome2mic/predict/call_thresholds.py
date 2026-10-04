"""Probability thresholds that turn P(active) into a call, per species x drug."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

THRESHOLD_GRID = np.linspace(0.0, 1.0, 201)
# Sentinels outside [0, 1]: a pair with one of these never gets that call.
NEVER_ACTIVE = 1.01
NEVER_INACTIVE = -0.01


class CallThresholds:
    """Likely active when p_active >= active_min; likely inactive when p_active <= inactive_max.

    Fitted on out-of-fold predictions so that very major errors (called active, lab resistant)
    and major errors (called inactive, lab susceptible) stay at or below their targets.
    """

    def __init__(self, records: list[dict], vme_target: float, me_target: float) -> None:
        self.records = records
        self.vme_target = vme_target
        self.me_target = me_target
        self.by_pair = {
            (record["species"], record["drug"]): (record["active_min"], record["inactive_max"]) for record in records
        }

    @staticmethod
    def fit_pair(
        probabilities: np.ndarray, lab_resistant: np.ndarray, vme_target: float, me_target: float
    ) -> tuple[float, float]:
        """Return (active_min, inactive_max) for one pair. Every row is lab-resistant or lab-susceptible."""
        resistant = probabilities[lab_resistant]
        susceptible = probabilities[~lab_resistant]

        active_min = NEVER_ACTIVE
        for threshold in THRESHOLD_GRID:
            if (resistant >= threshold).sum() <= vme_target * len(resistant):
                active_min = float(threshold)
                break
        inactive_max = NEVER_INACTIVE
        for threshold in THRESHOLD_GRID[::-1]:
            if (susceptible <= threshold).sum() <= me_target * len(susceptible):
                inactive_max = float(threshold)
                break

        # Both targets hold anywhere between the two, so a well-separated pair splits at the middle.
        if inactive_max >= active_min:
            active_min = (active_min + inactive_max) / 2
            inactive_max = active_min - 1e-9
        return active_min, inactive_max

    @classmethod
    def fit(cls, rows: pd.DataFrame, vme_target: float, me_target: float, min_per_class: int) -> "CallThresholds":
        """rows: species, drug, p_active, lab_resistant (bool). Pairs with too few R or S are skipped and logged."""
        logger.info("CallThresholds fit started: rows=%s vme_target=%s me_target=%s", len(rows), vme_target, me_target)
        records = []
        for (species, drug), group in rows.groupby(["species", "drug"]):
            lab_resistant = group["lab_resistant"].to_numpy(dtype=bool)
            n_resistant = int(lab_resistant.sum())
            n_susceptible = int((~lab_resistant).sum())
            if n_resistant < min_per_class or n_susceptible < min_per_class:
                logger.warning("No thresholds for %s %s: resistant=%s susceptible=%s (need %s each); band rule applies",
                               species, drug, n_resistant, n_susceptible, min_per_class)
                continue
            active_min, inactive_max = cls.fit_pair(
                group["p_active"].to_numpy(), lab_resistant, vme_target, me_target
            )
            records.append({"species": species, "drug": drug, "active_min": active_min, "inactive_max": inactive_max,
                            "n_resistant": n_resistant, "n_susceptible": n_susceptible})
        logger.info("CallThresholds fit done: pairs=%s", len(records))
        return cls(records, vme_target, me_target)

    def lookup(self, species: str, drug: str) -> tuple[float, float] | None:
        return self.by_pair.get((species, drug))

    def to_dict(self) -> dict:
        return {"vme_target": self.vme_target, "me_target": self.me_target, "pairs": self.records}

    @classmethod
    def from_dict(cls, payload: dict) -> "CallThresholds":
        return cls(payload["pairs"], payload["vme_target"], payload["me_target"])
