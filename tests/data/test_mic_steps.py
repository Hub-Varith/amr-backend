import numpy as np

from genome2mic.data.mic_steps import MicSteps


def test_round_up_goes_to_next_doubling_step():
    log2 = np.array([0.1, 0.0, -0.3, 2.9, 3.0])
    assert MicSteps.round_up_to_step(log2).tolist() == [2.0, 1.0, 1.0, 8.0, 8.0]


def test_to_log2_handles_censored_bounds():
    out = MicSteps.to_log2(np.array([0.0, 0.25, np.inf]))
    assert out[0] == -np.inf and out[1] == -2.0 and out[2] == np.inf
