"""Fixed choices for turning predicted MICs into calls."""

# Calls use one rule book. CLSI covers all 15 KPNEU headline drugs; EUCAST has no
# breakpoint for cefoxitin or tetracycline. See configs/breakpoints/README.md.
CALL_STANDARD = "CLSI"
CALL_YEAR = 2026

LIKELY_ACTIVE = "likely_active"
UNCERTAIN = "uncertain"
LIKELY_INACTIVE = "likely_inactive"

# Targets for the probability thresholds (CallThresholds). VME is kept below the 1.5% target
# commonly used in AST device evaluation; these are tuning targets, not claims.
VME_TARGET = 0.01
ME_TARGET = 0.03
MIN_PER_CLASS = 20

# Lower edges of the confidence levels on the calibrated P(active).
CONFIDENCE_LEVEL_EDGES = (0.02, 0.10, 0.30, 0.70, 0.90, 0.98)
CONFIDENCE_LEVEL_NAMES = (
    "very_likely_inactive",
    "probably_inactive",
    "leans_inactive",
    "unclear",
    "leans_active",
    "probably_active",
    "very_likely_active",
)
