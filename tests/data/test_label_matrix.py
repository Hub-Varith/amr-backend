import numpy as np
import pandas as pd
import pytest

from genome2mic.data.label_matrix import LabelMatrix


def make_labels():
    return pd.DataFrame({
        "genome_id": ["g1", "g1", "g2"], "species": ["KPNEU", "KPNEU", "ECOLI"],
        "drug": ["meropenem", "ciprofloxacin", "meropenem"],
        "mic_lower": [4.0, 0.0, 32.0], "mic_upper": [8.0, 0.25, np.inf], "censor": ["interval", "left", "right"],
    })


def test_pivot_keeps_bounds_and_marks_missing():
    matrix = LabelMatrix.from_labels(make_labels())
    assert matrix.drugs == ["ciprofloxacin", "meropenem"]
    g1, g2 = matrix.positions_of(np.array(["g1", "g2"]))
    assert matrix.mask[g1].tolist() == [True, True]
    assert matrix.mask[g2].tolist() == [False, True]
    assert matrix.lower_log2[g1, 1] == 2.0 and matrix.upper_log2[g1, 1] == 3.0
    assert matrix.lower_log2[g1, 0] == -np.inf and matrix.upper_log2[g1, 0] == -2.0
    assert matrix.upper_log2[g2, 1] == np.inf
    assert np.isnan(matrix.lower_log2[g2, 0])


def test_duplicate_pair_is_rejected():
    labels = pd.concat([make_labels(), make_labels().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        LabelMatrix.from_labels(labels)


def test_label_counts_by_species_and_drug():
    counts = LabelMatrix.from_labels(make_labels()).label_counts()
    row = counts[(counts.species == "ECOLI") & (counts.drug == "ciprofloxacin")]
    assert row["n_labels"].item() == 0
