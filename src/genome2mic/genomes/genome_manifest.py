"""Decides which genomes to download and from where. See DATA_CONTRACT.md stage 3."""

import gzip
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

NCBI_PREFIX = "NCBI_"
MANIFEST_COLUMNS = ["genome_id", "species", "source", "accession"]


class GenomeManifestBuilder:
    """Builds the download list: one row per labelled genome of a species in a kept drug pair."""

    @staticmethod
    def read_assembly_reports(path: Path) -> pd.DataFrame:
        """Read `datasets summary genome --as-json-lines` output into (accession, biosample, n_contigs)."""
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                report = json.loads(line)
                rows.append(
                    {
                        "accession": report["accession"],
                        "biosample": report.get("assembly_info", {}).get("biosample", {}).get("accession"),
                        "n_contigs": report.get("assembly_stats", {}).get("number_of_contigs"),
                    }
                )
        assemblies = pd.DataFrame(rows).dropna(subset=["biosample"])
        logger.info("Read NCBI assembly reports: %s assemblies from %s", len(assemblies), path)
        return assemblies

    @staticmethod
    def choose_assemblies(assemblies: pd.DataFrame) -> pd.DataFrame:
        """One assembly per biosample: RefSeq (GCF) before GenBank (GCA), then the fewest contigs."""
        ranked = assemblies.assign(
            is_refseq=assemblies["accession"].str.startswith("GCF_"),
            n_contigs=pd.to_numeric(assemblies["n_contigs"], errors="coerce"),
        )
        ranked = ranked.sort_values(["is_refseq", "n_contigs", "accession"], ascending=[False, True, True])
        return ranked.drop_duplicates(subset="biosample")[["biosample", "accession"]]

    @staticmethod
    def build(
        labels: pd.DataFrame, pairs_kept: pd.DataFrame, ncbi_assemblies: pd.DataFrame, species: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (manifest, dropped) for one species."""
        kept_labels = labels.merge(pairs_kept[["species", "drug"]], on=["species", "drug"])
        genomes = (
            kept_labels.loc[kept_labels["species"] == species, ["genome_id", "species"]]
            .drop_duplicates()
            .sort_values("genome_id")
        )
        is_ncbi = genomes["genome_id"].str.startswith(NCBI_PREFIX)

        bvbrc = genomes[~is_ncbi].assign(source="bvbrc", accession=genomes.loc[~is_ncbi, "genome_id"])

        chosen = GenomeManifestBuilder.choose_assemblies(ncbi_assemblies).set_index("biosample")["accession"]
        ncbi = genomes[is_ncbi].copy()
        ncbi["source"] = "ncbi"
        ncbi["accession"] = ncbi["genome_id"].str.removeprefix(NCBI_PREFIX).map(chosen)
        has_assembly = ncbi["accession"].notna()
        dropped = ncbi.loc[~has_assembly, ["genome_id", "species"]].assign(reason="no_assembly")

        manifest = pd.concat([bvbrc, ncbi[has_assembly]]).sort_values("genome_id")[MANIFEST_COLUMNS]
        logger.info(
            "Manifest %s: bvbrc=%s ncbi=%s dropped_no_assembly=%s",
            species,
            len(bvbrc),
            int(has_assembly.sum()),
            len(dropped),
        )
        return manifest.reset_index(drop=True), dropped.reset_index(drop=True)

    @staticmethod
    def pilot_sample(manifest: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
        """A fixed random subset for testing the pipeline. Never used for splits or training."""
        if size >= len(manifest):
            return manifest
        return manifest.sample(n=size, random_state=seed).sort_values("genome_id").reset_index(drop=True)
