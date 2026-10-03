"""Agreement and error-rate metrics for MIC predictions (DATA_CONTRACT.md stage 11)."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from genome2mic.data.mic_steps import MicSteps

METRIC_ORDER: tuple[str, ...] = (
    "vme_rate",
    "me_rate",
    "mine_rate",
    "essential_agreement",
    "exact_agreement",
    "categorical_agreement",
    "auroc",
    "band_coverage",
    "band_width_steps",
    "n",
    "n_exact",
    "n_lab_r",
    "n_lab_s",
)


class MicMetrics:
    """Static metric functions. VME comes first in every output."""

    @staticmethod
    def sir_from_mic(mic: np.ndarray, s_breakpoint: float, r_breakpoint: float) -> np.ndarray:
        """S when mic <= S breakpoint, R when mic > R breakpoint, else I."""
        mic = np.asarray(mic, dtype=np.float64)
        return np.where(mic <= s_breakpoint, "S", np.where(mic > r_breakpoint, "R", "I"))

    @staticmethod
    def compute(predictions: pd.DataFrame) -> dict[str, float]:
        """One row per genome x drug with pred_mic, band_low, band_high, lab_lower, lab_upper, pred_sir, lab_sir.

        Essential and exact agreement use rows with an exact lab MIC only (finite lower and
        upper), since a censored lab value has no single step to agree with. Band coverage counts
        a row as covered when the band overlaps the lab interval.
        """
        required = {"pred_mic", "band_low", "band_high", "lab_lower", "lab_upper", "pred_sir", "lab_sir"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions is missing columns {sorted(missing)}")
        if len(predictions) == 0:
            return {name: float("nan") for name in METRIC_ORDER} | {"n": 0, "n_exact": 0, "n_lab_r": 0, "n_lab_s": 0}

        pred_mic = predictions["pred_mic"].to_numpy(dtype=np.float64)
        lab_lower = predictions["lab_lower"].to_numpy(dtype=np.float64)
        lab_upper = predictions["lab_upper"].to_numpy(dtype=np.float64)
        pred_sir = predictions["pred_sir"].to_numpy(dtype=str)
        lab_sir = predictions["lab_sir"].to_numpy(dtype=str)

        lab_r = lab_sir == "R"
        lab_s = lab_sir == "S"
        vme_rate = float(np.mean(pred_sir[lab_r] == "S")) if lab_r.any() else float("nan")
        me_rate = float(np.mean(pred_sir[lab_s] == "R")) if lab_s.any() else float("nan")
        one_side_i = ((pred_sir == "I") | (lab_sir == "I")) & (pred_sir != lab_sir)
        mine_rate = float(one_side_i.mean())
        categorical_agreement = float(np.mean(pred_sir == lab_sir))

        exact = (lab_lower > 0) & np.isfinite(lab_upper)
        if exact.any():
            step_gap = np.abs(MicSteps.step_difference(pred_mic[exact], lab_upper[exact]))
            essential_agreement = float(np.mean(step_gap <= 1.0 + 1e-9))
            exact_agreement = float(np.mean(step_gap <= 1e-9))
        else:
            essential_agreement = float("nan")
            exact_agreement = float("nan")

        if lab_r.any() and lab_s.any():
            auroc = float(roc_auc_score(lab_r[lab_r | lab_s], np.log2(pred_mic[lab_r | lab_s])))
        else:
            auroc = float("nan")

        band_low = predictions["band_low"].to_numpy(dtype=np.float64)
        band_high = predictions["band_high"].to_numpy(dtype=np.float64)
        covered = (band_high > lab_lower) & (band_low <= lab_upper)
        band_coverage = float(covered.mean())
        band_width_steps = float(np.mean(np.log2(band_high) - np.log2(band_low)))

        return {
            "vme_rate": vme_rate,
            "me_rate": me_rate,
            "mine_rate": mine_rate,
            "essential_agreement": essential_agreement,
            "exact_agreement": exact_agreement,
            "categorical_agreement": categorical_agreement,
            "auroc": auroc,
            "band_coverage": band_coverage,
            "band_width_steps": band_width_steps,
            "n": int(len(predictions)),
            "n_exact": int(exact.sum()),
            "n_lab_r": int(lab_r.sum()),
            "n_lab_s": int(lab_s.sum()),
        }

    @staticmethod
    def by_pair(predictions: pd.DataFrame) -> pd.DataFrame:
        """compute() for every species x drug x model x split group, VME column first."""
        group_columns = [column for column in ("species", "drug", "model", "split") if column in predictions.columns]
        rows = []
        for keys, group in predictions.groupby(group_columns, sort=True):
            key_dict = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
            rows.append(key_dict | MicMetrics.compute(group))
        return pd.DataFrame(rows, columns=group_columns + list(METRIC_ORDER))
