import numpy as np
import pandas as pd
from genome2mic.models.cross_validation import CrossValidation


def test_untrained_species_and_drugs_are_not_advertised():
    class Labels:
        def subset(self, positions):
            return self
        def label_counts(self):
            return pd.DataFrame({"species": ["KPNEU", "KPNEU"],
                                 "drug": ["meropenem", "amikacin"], "n_labels": [12, 0]})
    cv = CrossValidation.__new__(CrossValidation)
    cv.labels = Labels()
    cv.train_positions = np.array([0])
    cv.pairs_kept = pd.DataFrame({"species": ["KPNEU", "ECOLI", "KPNEU"],
                                  "drug": ["meropenem", "meropenem", "amikacin"]})
    assert cv.drugs_by_species() == {"KPNEU": ["meropenem"]}
