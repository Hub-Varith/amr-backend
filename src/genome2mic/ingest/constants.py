"""Fixed values for AST download and label harmonization. See DATA_CONTRACT.md stage 1-2."""

BVBRC_API = "https://www.bv-brc.org/api"
BVBRC_PAGE_SIZE = 25000
BVBRC_ID_BATCH = 500
BVBRC_LAB_EVIDENCE = "Laboratory Method"
# BV-BRC blocks Python's default user agent with HTTP 403.
HTTP_USER_AGENT = "genome2mic/0.1 (+https://github.com/Hub-Varith/amr-backend)"

BVBRC_GENOME_FIELDS = (
    "genome_id",
    "genome_name",
    "species",
    "taxon_id",
    "biosample_accession",
    "assembly_accession",
    "sra_accession",
    "genome_status",
    "contigs",
    "genome_length",
    "isolation_source",
    "isolation_country",
    "collection_year",
    "collection_date",
    "host_name",
)

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_FETCH_BATCH = 200
# NCBI allows 3 requests per second without an API key.
NCBI_REQUEST_PAUSE_SECONDS = 0.4

HTTP_RETRIES = 4
HTTP_TIMEOUT_SECONDS = 180

# Order matters: the first keyword found wins, so disk is checked before dilution.
METHOD_KEYWORDS = (
    ("disk", "disk"),
    ("kirby", "disk"),
    ("zone", "disk"),
    ("etest", "gradient"),
    ("e-test", "gradient"),
    ("gradient", "gradient"),
    ("strip", "gradient"),
    ("broth", "dilution"),
    ("microdilution", "dilution"),
    ("agar dilution", "dilution"),
    ("dilution", "dilution"),
    ("mic", "dilution"),
)

MIC_UNITS = {"mg/l", "ug/ml", "µg/ml", "μg/ml", "mcg/ml"}

PHENOTYPE_TO_SIR = {
    "susceptible": "S",
    "intermediate": "I",
    "susceptible-dose dependent": "I",
    "resistant": "R",
}

SIGN_ALIASES = {
    "": "=",
    "=": "=",
    "==": "=",
    "<=": "<=",
    "≤": "<=",
    "=<": "<=",
    "<": "<",
    ">": ">",
    ">=": ">=",
    "≥": ">=",
    "=>": ">=",
}

KNOWN_STANDARDS = {"clsi": "CLSI", "eucast": "EUCAST"}

# A reported value within 5% of a doubling step is that step (0.06 -> 0.0625).
SNAP_TOLERANCE = 0.05

LABEL_COLUMNS = (
    "genome_id",
    "biosample",
    "species",
    "drug",
    "mic_lower",
    "mic_upper",
    "censor",
    "sir",
    "raw_result",
    "method",
    "standard",
    "standard_year",
    "source",
    "isolation_source",
    "country",
    "year",
)
DROPPED_COLUMNS = ("source", "record_id", "species", "drug", "antibiotic", "reason")
LAB_EVIDENCE = "Laboratory Method"

PAIR_MIN_NONSUSCEPTIBLE = 50
PAIR_MIN_SUSCEPTIBLE = 50
PAIR_MIN_DISTINCT_MIC = 4

MISSING_TEXT = {"", "na", "n/a", "nan", "none", "missing", "unknown", "not collected", "not applicable", "not provided"}

# First keyword found wins.
ISOLATION_SOURCE_KEYWORDS = (
    ("blood", "blood"),
    ("bacteremia", "blood"),
    ("urin", "urine"),
    ("sputum", "respiratory"),
    ("bronch", "respiratory"),
    ("trache", "respiratory"),
    ("lung", "respiratory"),
    ("respir", "respiratory"),
    ("rectal", "gut"),
    ("stool", "gut"),
    ("feces", "gut"),
    ("faeces", "gut"),
    ("fecal", "gut"),
    ("perianal", "gut"),
    ("wound", "wound"),
    ("pus", "wound"),
    ("abscess", "wound"),
)
