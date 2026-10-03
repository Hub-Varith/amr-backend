import numpy as np
import pandas as pd
import pytest

from genome2mic.data.known_amr_table import KnownAmrTable


def test_forbidden_column_is_rejected():
    table = pd.DataFrame({"genome_id": ["g1"], "species": ["KPNEU"], "gene_blakpc_2": [1], "st": ["258"]})
    with pytest.raises(ValueError, match="forbidden"):
        KnownAmrTable(table)


def test_rare_filter_uses_only_given_genomes():
    table = pd.DataFrame({
        "genome_id": ["g1", "g2", "g3"], "species": ["KPNEU"] * 3,
        "gene_a": [1, 1, 0], "gene_b": [0, 0, 1], "n_class_x": [2, 0, 0],
    })
    known = KnownAmrTable(table)
    kept = known.frequent_columns(np.array(["g1", "g2"]), np.array(["KPNEU", "KPNEU"]), min_count=2)
    assert [known.columns[index] for index in kept] == ["gene_a"]
    assert known.rows(np.array(["g9"])).sum() == 0
