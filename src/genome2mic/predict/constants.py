"""Fixed choices for turning predicted MICs into calls."""

# Calls use one rule book. CLSI covers all 15 KPNEU headline drugs; EUCAST has no
# breakpoint for cefoxitin or tetracycline. See configs/breakpoints/README.md.
CALL_STANDARD = "CLSI"
CALL_YEAR = 2026

LIKELY_ACTIVE = "likely_active"
UNCERTAIN = "uncertain"
LIKELY_INACTIVE = "likely_inactive"
