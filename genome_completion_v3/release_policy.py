"""Public output gate; confidence refers to the frozen calibration event."""
import math

THRESHOLD = 0.95
MINIMUM_GROUPS = 10

def rejection_reason(index, probability, groups):
    if index is None:
        return "no_candidate"
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        return "invalid_confidence"
    if probability < THRESHOLD:
        return "confidence_below_95_percent"
    if groups < MINIMUM_GROUPS:
        return "insufficient_calibration_support"
    return None
