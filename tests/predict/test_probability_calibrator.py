"""Calibration of p_active. All values here are test-only."""

import numpy as np
import pandas as pd
import pytest

from genome2mic.predict.probability_calibrator import ProbabilityCalibrator


def rows_for(species: str, drug: str) -> pd.DataFrame:
    # Raw 0.5 is too modest (80% are really active); raw 0.1 and 0.9 are about right.
    probabilities = np.concatenate([np.full(100, 0.1), np.full(100, 0.5), np.full(100, 0.9)])
    lab_active = np.concatenate([np.arange(100) < 10, np.arange(100) < 80, np.arange(100) < 90])
    return pd.DataFrame({"species": species, "drug": drug, "p_raw": probabilities, "lab_resistant": ~lab_active})


def test_calibrated_probability_matches_the_observed_rate() -> None:
    calibrator = ProbabilityCalibrator.fit(rows_for("KPNEU", "test-drug"), min_per_class=20)

    assert calibrator.calibrate("KPNEU", "test-drug", np.array([0.5]))[0] == pytest.approx(0.8)
    assert calibrator.calibrate("KPNEU", "test-drug", np.array([0.1]))[0] == pytest.approx(0.1)


def test_calibration_never_reverses_the_order() -> None:
    calibrator = ProbabilityCalibrator.fit(rows_for("KPNEU", "test-drug"), min_per_class=20)

    calibrated = calibrator.calibrate("KPNEU", "test-drug", np.linspace(0.0, 1.0, 21))

    assert (np.diff(calibrated) >= 0).all()
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0


def test_pair_without_a_curve_is_returned_unchanged() -> None:
    calibrator = ProbabilityCalibrator.fit(rows_for("KPNEU", "test-drug"), min_per_class=20)

    assert calibrator.calibrate("KPNEU", "other-drug", np.array([0.5]))[0] == 0.5
    assert not calibrator.has_pair("KPNEU", "other-drug")


def test_pair_with_too_few_rows_gets_no_curve() -> None:
    calibrator = ProbabilityCalibrator.fit(rows_for("KPNEU", "test-drug").head(30), min_per_class=20)

    assert not calibrator.has_pair("KPNEU", "test-drug")


def test_round_trip_through_dict() -> None:
    calibrator = ProbabilityCalibrator.fit(rows_for("KPNEU", "test-drug"), min_per_class=20)

    restored = ProbabilityCalibrator.from_dict(calibrator.to_dict())

    assert restored.calibrate("KPNEU", "test-drug", np.array([0.5]))[0] == pytest.approx(0.8)
