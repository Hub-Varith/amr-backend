"""QC thresholds from DATA_CONTRACT.md stage 3."""

MAX_CONTIGS = 500
GENOME_SIZE_TOLERANCE = 0.20
MAX_MASH_DISTANCE = 0.05

QC_COLUMNS = [
    "genome_id",
    "n_contigs",
    "total_length",
    "n50",
    "gc_percent",
    "mash_species",
    "mash_distance",
    "qc_pass",
    "qc_fail_reason",
]
