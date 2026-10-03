import numpy as np
import pandas as pd

from genome2mic.eval.metrics import METRIC_ORDER, MicMetrics


def test_sir_from_mic_uses_breakpoints():
    out = MicMetrics.sir_from_mic(np.array([1.0, 2.0, 4.0]), s_breakpoint=1.0, r_breakpoint=2.0)
    assert out.tolist() == ["S", "I", "R"]


def test_hand_computed_rates():
    preds = pd.DataFrame({
        "pred_mic": [1.0, 8.0, 0.5, 16.0, 2.0],
        "band_low": [0.5, 4.0, 0.25, 8.0, 1.0],
        "band_high": [2.0, 16.0, 1.0, 32.0, 4.0],
        "lab_lower": [0.5, 8.0, 0.0, 2.0, 1.0],
        "lab_upper": [1.0, 16.0, 0.25, 4.0, 2.0],
        "pred_sir": ["S", "R", "S", "R", "I"],
        "lab_sir": ["S", "R", "R", "S", "S"],
    })
    out = MicMetrics.compute(preds)
    assert list(out)[:3] == ["vme_rate", "me_rate", "mine_rate"]
    assert out["vme_rate"] == 0.5          # 2 lab R, one predicted S
    assert out["me_rate"] == 1 / 3         # 3 lab S, one predicted R
    assert out["mine_rate"] == 0.2         # one row with I on one side only
    assert out["categorical_agreement"] == 0.4
    # Exact rows: 1, 2, 4, 5. Steps off: 0, 1, 2, 0 -> EA 3/4, exact 2/4.
    assert out["n_exact"] == 4
    assert abs(out["essential_agreement"] - 0.75) < 1e-9
    assert abs(out["exact_agreement"] - 0.5) < 1e-9
    assert out["band_coverage"] == 0.8     # row 4 band (8-32] misses lab (2-4]
    assert out["band_width_steps"] == 2.0
    assert out["n"] == 5


def test_by_pair_keeps_metric_order_and_groups():
    preds = pd.DataFrame({
        "species": ["KPNEU", "KPNEU", "ECOLI"], "drug": ["meropenem"] * 3, "model": ["m"] * 3, "split": ["oof"] * 3,
        "pred_mic": [1.0, 8.0, 1.0], "band_low": [1.0, 4.0, 1.0], "band_high": [2.0, 16.0, 2.0],
        "lab_lower": [0.5, 4.0, 0.5], "lab_upper": [1.0, 8.0, 1.0],
        "pred_sir": ["S", "R", "S"], "lab_sir": ["S", "R", "R"],
    })
    table = MicMetrics.by_pair(preds)
    assert list(table.columns) == ["species", "drug", "model", "split", *METRIC_ORDER]
    assert table.set_index("species").loc["ECOLI", "vme_rate"] == 1.0
    assert table.set_index("species").loc["KPNEU", "vme_rate"] == 0.0
