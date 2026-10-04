"""Column names and prefixes fixed by DATA_CONTRACT.md."""

FEATURE_PREFIXES: tuple[str, ...] = ("gene_", "point_", "n_class_")

# Rule 2 of the contract: these may never reach a model.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {"lineage_cluster", "st", "country", "year", "source", "isolation_source"}
)

SPECIES_KEYS: tuple[str, ...] = ("ECOLI", "KPNEU", "SAUR", "PAER", "ABAU")

LABEL_COLUMNS: tuple[str, ...] = ("genome_id", "species", "drug", "mic_lower", "mic_upper", "censor")

SPLIT_COLUMNS: tuple[str, ...] = ("genome_id", "species", "split", "fold")
