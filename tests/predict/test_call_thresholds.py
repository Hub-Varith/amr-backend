"""Probability thresholds for calls. All values here are test-only."""

import numpy as np
import pytest

from genome2mic.predict.call_thresholds import CallThresholds


def test_fit_pair_keeps_very_major_errors_at_or_below_the_target() -> None:
    # 100 resistant genomes: one has a high "active" probability (0.95), the rest are low.
    probabilities = np.concatenate([[0.95], np.full(99, 0.10), np.full(100, 0.90)])
    lab_resistant = np.concatenate([np.ones(100, dtype=bool), np.zeros(100, dtype=bool)])

    active_min, inactive_max = CallThresholds.fit_pair(probabilities, lab_resistant, vme_target=0.01, me_target=0.03)

    called_active = probabilities >= active_min
    assert (called_active & lab_resistant).sum() / 100 <= 0.01
    # One resistant genome above the threshold is allowed (1%), so the susceptible ones at 0.90 are called.
    assert active_min <= 0.90


def test_fit_pair_with_no_allowed_errors_moves_above_the_highest_resistant_probability() -> None:
    probabilities = np.concatenate([[0.95], np.full(9, 0.10), np.full(10, 0.99)])
    lab_resistant = np.concatenate([np.ones(10, dtype=bool), np.zeros(10, dtype=bool)])

    active_min, _ = CallThresholds.fit_pair(probabilities, lab_resistant, vme_target=0.0, me_target=0.0)

    assert active_min > 0.95


def test_fit_pair_keeps_major_errors_at_or_below_the_target() -> None:
    probabilities = np.concatenate([np.full(100, 0.05), [0.02], np.full(99, 0.80)])
    lab_resistant = np.concatenate([np.ones(100, dtype=bool), np.zeros(100, dtype=bool)])

    _, inactive_max = CallThresholds.fit_pair(probabilities, lab_resistant, vme_target=0.01, me_target=0.01)

    called_inactive = probabilities <= inactive_max
    assert (called_inactive & ~lab_resistant).sum() / 100 <= 0.01
    assert inactive_max >= 0.05


def test_thresholds_never_overlap() -> None:
    # Probabilities carry no signal, so both thresholds would want the middle.
    probabilities = np.tile([0.4, 0.6], 100)
    lab_resistant = np.repeat([True, False], 100)

    active_min, inactive_max = CallThresholds.fit_pair(probabilities, lab_resistant, vme_target=0.5, me_target=0.5)

    assert inactive_max < active_min


def test_lookup_and_round_trip() -> None:
    thresholds = CallThresholds(
        [{"species": "KPNEU", "drug": "test-drug", "active_min": 0.9, "inactive_max": 0.1, "n_resistant": 50,
          "n_susceptible": 60}],
        vme_target=0.01,
        me_target=0.03,
    )

    assert thresholds.lookup("KPNEU", "test-drug") == (0.9, 0.1)
    assert thresholds.lookup("KPNEU", "other-drug") is None
    restored = CallThresholds.from_dict(thresholds.to_dict())
    assert restored.lookup("KPNEU", "test-drug") == (0.9, 0.1)
    assert restored.vme_target == pytest.approx(0.01)
