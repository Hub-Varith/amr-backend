"""AMRFinderPlus output for an uploaded genome -> the known-AMR row the model was trained on.

Training features came from NCBI Pathogen Detection's `AMR_genotypes` strings, which are
AMRFinderPlus results written as `symbol[=TAG]` (docs/HACKATHON_DATA.md). To avoid train/serve
skew (INTEGRATION.md 3.2), an upload's AMRFinderPlus table is written back into that same string
and passed through the very same NcbiKnownAmrBuilder that built known_amr.parquet. There is no
second implementation of the naming rules.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from genome2mic.features.ncbi_known_amr import NcbiKnownAmrBuilder

logger = logging.getLogger(__name__)

# AMRFinderPlus 4.x calls the first column "Element symbol"; older versions "Gene symbol".
SYMBOL_COLUMNS = ("Element symbol", "Gene symbol")
UPLOAD_ID = "upload"


@dataclass(frozen=True)
class AmrFinderHit:
    """One row of AMRFinderPlus output (DATA_CONTRACT.md stage 4)."""

    symbol: str
    type: str
    subtype: str
    drug_class: str
    subclass: str
    method: str

    @property
    def is_point(self) -> bool:
        # 4.x adds POINT_DISRUPT (e.g. a porin frameshift); NCBI tags those =POINT as well.
        return self.subtype.startswith("POINT")


class AmrFinderTable:
    """Reads amrfinder.tsv and writes it as an NCBI AMR_genotypes string."""

    @staticmethod
    def parse(path: Path) -> list[AmrFinderHit]:
        table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        symbol_column = next((column for column in SYMBOL_COLUMNS if column in table.columns), None)
        if symbol_column is None:
            raise ValueError(f"{path} has no {' or '.join(SYMBOL_COLUMNS)} column; is it AMRFinderPlus output?")
        hits = [
            AmrFinderHit(
                symbol=row[symbol_column],
                type=row.get("Type", ""),
                subtype=row.get("Subtype", ""),
                drug_class=row.get("Class", ""),
                subclass=row.get("Subclass", ""),
                method=row.get("Method", ""),
            )
            for row in table.to_dict(orient="records")
        ]
        logger.info("AMRFinderPlus hits parsed", extra={"n_hits": len(hits), "n_amr": sum(h.type == "AMR" for h in hits)})
        return hits

    @staticmethod
    def ncbi_tag(hit: AmrFinderHit) -> str:
        """The tag NCBI Pathogen Detection appends for this hit's method ('' = complete gene)."""
        if hit.is_point:
            return "POINT"
        if hit.method == "INTERNAL_STOP":
            return "MISTRANSLATION"
        if hit.method.startswith("PARTIAL_CONTIG_END"):
            return "PARTIAL_END_OF_CONTIG"
        if hit.method.startswith("PARTIAL"):
            return "PARTIAL"
        if hit.method == "HMM":
            return "HMM"
        return ""

    @classmethod
    def genotype_string(cls, hits: list[AmrFinderHit]) -> str:
        """AMR-type hits only: NCBI keeps STRESS and VIRULENCE hits in separate columns."""
        items = []
        for hit in hits:
            if hit.type != "AMR":
                continue
            tag = cls.ncbi_tag(hit)
            items.append(f"{hit.symbol}={tag}" if tag else hit.symbol)
        return ",".join(items)


class KnownAmrRow:
    """{column_name: value} for one genome, built by the training feature builder."""

    @staticmethod
    def from_hits(hits: list[AmrFinderHit], species: str, builder: NcbiKnownAmrBuilder) -> dict[str, int]:
        """Only non-zero columns are returned; every other model column is 0."""
        genotypes = AmrFinderTable.genotype_string(hits)
        if not genotypes:
            return {}
        genomes = pd.DataFrame([(UPLOAD_ID, species, UPLOAD_ID)], columns=["genome_id", "species", "biosample"])
        metadata = pd.DataFrame([(UPLOAD_ID, genotypes)], columns=["biosample_acc", "AMR_genotypes"])
        features, _ = builder.build(genomes, metadata)
        if features.empty:
            return {}
        values = features.drop(columns=["genome_id", "species"]).iloc[0]
        return {column: int(value) for column, value in values.items() if value}
