"""Known-AMR features from NCBI Pathogen Detection's precomputed AMRFinderPlus calls.

Hackathon shortcut for stage 5: uses the `AMR_genotypes` column instead of running
AMRFinderPlus ourselves. See docs/HACKATHON_DATA.md.
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ALLELE_SUFFIX = re.compile(r"-\d+$")
NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
# A MISTRANSLATION call is a gene with an internal stop codon, so it is likely broken.
SKIPPED_TAGS = {"MISTRANSLATION"}


class NcbiKnownAmrBuilder:
    """Turns comma-separated AMR_genotypes strings into the wide known_amr.parquet table."""

    def __init__(self, keep_variant_families: list[str], class_by_symbol: dict[str, str]) -> None:
        self.keep_variant_families = keep_variant_families
        self.class_by_symbol = class_by_symbol

    @staticmethod
    def load_class_table(amrfinder_db_dir: Path, organism: str) -> dict[str, str]:
        """Map gene families and point mutations to their AMRFinderPlus drug class."""
        families = pd.read_csv(amrfinder_db_dir / "fam.tsv", sep="\t", dtype=str)
        class_by_symbol = dict(zip(families["#node_id"], families["class"].fillna("")))
        for mutation_file in ("AMRProt-mutation.tsv", f"AMR_DNA-{organism}.tsv"):
            mutations = pd.read_csv(amrfinder_db_dir / mutation_file, sep="\t", dtype=str)
            class_by_symbol.update(zip(mutations["standard_mutation_symbol"], mutations["class"].fillna("")))
        return {symbol: amr_class for symbol, amr_class in class_by_symbol.items() if amr_class}

    @staticmethod
    def column_name(prefix: str, symbol: str) -> str:
        """Contract naming: lowercase, every run of non-alphanumeric characters becomes one `_`."""
        return prefix + NON_ALPHANUMERIC.sub("_", symbol.lower()).strip("_")

    def family_for(self, symbol: str) -> str:
        """Collapse beta-lactamase alleles to their family unless the family keeps its variant."""
        if not symbol.startswith("bla"):
            return symbol
        if any(symbol.startswith(kept) for kept in self.keep_variant_families):
            return symbol
        return ALLELE_SUFFIX.sub("", symbol)

    def parse_genotypes(self, genotypes: str) -> list[tuple[str, str]]:
        """Return (kind, symbol) pairs; kind is `gene` or `point`."""
        hits = []
        for item in genotypes.split(","):
            symbol, _, tag = item.strip().partition("=")
            if not symbol or tag in SKIPPED_TAGS:
                continue
            hits.append(("point" if tag == "POINT" else "gene", symbol))
        return hits

    def build(self, genomes: pd.DataFrame, pd_metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (features, column_map). Genomes with no NCBI AMR result are left out and logged."""
        genotypes = (
            pd_metadata.loc[pd_metadata["AMR_genotypes"].fillna("") != "", ["biosample_acc", "AMR_genotypes"]]
            .drop_duplicates(subset="biosample_acc")
            .set_index("biosample_acc")["AMR_genotypes"]
        )
        covered = genomes[genomes["biosample"].isin(genotypes.index)]
        logger.info("Genomes with NCBI AMR results: %s of %s", len(covered), len(genomes))

        hit_rows = []
        for genome_id, biosample in zip(covered["genome_id"], covered["biosample"]):
            for kind, symbol in self.parse_genotypes(genotypes[biosample]):
                if kind == "point":
                    feature_symbol = symbol
                    column = self.column_name("point_", symbol)
                else:
                    feature_symbol = self.family_for(symbol)
                    column = self.column_name("gene_", feature_symbol)
                amr_class = self.class_by_symbol.get(symbol) or self.class_by_symbol.get(ALLELE_SUFFIX.sub("", symbol))
                hit_rows.append((genome_id, column, feature_symbol, amr_class))
        hits = pd.DataFrame(hit_rows, columns=["genome_id", "column_name", "source_symbol", "class"])

        presence = pd.crosstab(hits["genome_id"], hits["column_name"]).clip(upper=1)
        class_hits = hits.dropna(subset=["class"]).assign(amr_class=lambda frame: frame["class"].str.split("/"))
        class_hits = class_hits.explode("amr_class")
        class_hits["column_name"] = class_hits["amr_class"].map(lambda name: self.column_name("n_class_", name))
        class_counts = pd.crosstab(class_hits["genome_id"], class_hits["column_name"])

        features = covered[["genome_id", "species"]].set_index("genome_id")
        features = features.join(presence).join(class_counts).fillna(0)
        feature_columns = [column for column in features.columns if column != "species"]
        features[feature_columns] = features[feature_columns].clip(upper=127).astype("int8")
        features = features.reset_index().sort_values("genome_id").reset_index(drop=True)

        column_map = (
            hits.groupby("column_name")
            .agg(
                source_symbol=("source_symbol", lambda symbols: ",".join(sorted(set(symbols)))),
                amr_class=("class", "first"),
                n_genomes_present=("genome_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"amr_class": "class"})
        )
        logger.info("Known-AMR features: genomes=%s gene/point columns=%s class columns=%s",
                    len(features), presence.shape[1], class_counts.shape[1])
        return features, column_map
