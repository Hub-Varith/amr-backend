"""Fixed values used by the API layer."""

import re

from genome2mic.api.schemas.species_key import SpeciesKey

DISCLAIMER = (
    "These are predictions of in-vitro susceptibility, not prescribing advice. "
    "Dose, route, and final drug choice depend on PK/PD, infection site, renal "
    "function, allergies, and other patient factors, and remain with the "
    "clinician. Confirm with standard AST."
)

SPECIES_NAMES = {
    SpeciesKey.ECOLI: "Escherichia coli",
    SpeciesKey.KPNEU: "Klebsiella pneumoniae",
    SpeciesKey.SAUR: "Staphylococcus aureus",
    SpeciesKey.PAER: "Pseudomonas aeruginosa",
    SpeciesKey.ABAU: "Acinetobacter baumannii",
}

CONFIG_FILES = ("species.yaml", "drugs.yaml")

ALLOWED_FASTA_EXTENSIONS = frozenset({".fasta", ".fa", ".fna"})
UPLOAD_CHUNK_BYTES = 1024 * 1024

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

PROBLEM_JSON = "application/problem+json"
