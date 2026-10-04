"""Observed-only incremental feature exported from the validated v3 experiment."""
from shared import base
def history_features(newly_observed,previous_guess):
    if previous_guess is None or not newly_observed:return [-1.,0.]
    predicted=previous_guess[:len(newly_observed)]
    a,b=base.sketch([newly_observed]),base.sketch([predicted])
    return [len(a&b)/max(1,len(a|b)),len(predicted)/len(newly_observed)]
