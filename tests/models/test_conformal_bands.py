import numpy as np

from genome2mic.models.conformal_bands import ConformalBands


def test_quantile_is_conservative_finite_sample():
    residuals = np.arange(1, 11, dtype=float)
    # n=10, level 0.9 -> rank ceil(11*0.9)=10 -> largest residual.
    assert ConformalBands.quantile(residuals, 0.9) == 10.0
    assert ConformalBands.quantile(residuals, 0.5) == 6.0


def test_half_width_falls_back_to_drug_then_global():
    bands = ConformalBands.from_records(
        [{"species": "KPNEU", "drug": "meropenem", "q": 1.0, "n_exact": 50, "source": "pair"}],
        {"meropenem": 1.5, "ceftriaxone": 2.0},
        3.0,
    )
    assert bands.half_width("KPNEU", "meropenem") == 1.0
    assert bands.half_width("ECOLI", "meropenem") == 1.5
    assert bands.half_width("ECOLI", "ciprofloxacin") == 3.0
