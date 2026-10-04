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
