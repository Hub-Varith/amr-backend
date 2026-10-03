"""CLI: download raw AST and metadata from BV-BRC and NCBI into data/raw/ (stage 1)."""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from genome2mic.ingest.bvbrc import BvbrcDownloader
from genome2mic.ingest.http_client import HttpClient
from genome2mic.ingest.ncbi_ast import NcbiAstDownloader

logger = logging.getLogger(__name__)


def main() -> None:
    """Download every source, write CSVs, and record what was fetched and when."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--species", nargs="*", help="Species keys; default is all in species.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    species_config = yaml.safe_load((args.configs_dir / "species.yaml").read_text())["species"]
    species_keys = args.species or list(species_config)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    client = HttpClient()
    bvbrc = BvbrcDownloader(client)
    ncbi = NcbiAstDownloader(client, args.raw_dir / "ncbi_biosample")

    bvbrc_frames = []
    ncbi_counts = {}
    for species_key in species_keys:
        species_name = species_config[species_key]["name"]
        bvbrc_frames.append(bvbrc.download_ast(species_name))
        ncbi_counts[species_key] = ncbi.download_xml(species_key, species_name)
    bvbrc_ast = pd.concat(bvbrc_frames, ignore_index=True)
    bvbrc_ast.to_csv(args.raw_dir / "ast_bvbrc.csv", index=False)

    ncbi_ast, ncbi_meta = ncbi.parse_xml()
    ncbi_ast.to_csv(args.raw_dir / "ast_ncbi.csv", index=False)
    ncbi_meta.to_csv(args.raw_dir / "meta_ncbi.csv", index=False)

    genome_ids = sorted(bvbrc_ast["genome_id"].astype(str).unique())
    biosamples = sorted(ncbi_meta["biosample"].dropna().unique())
    bvbrc_meta = bvbrc.download_genome_metadata(genome_ids, biosamples)
    bvbrc_meta.to_csv(args.raw_dir / "meta_bvbrc.csv", index=False)

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "species": species_keys,
        "bvbrc_ast_rows": len(bvbrc_ast),
        "bvbrc_genomes_with_metadata": len(bvbrc_meta),
        "ncbi_biosamples_by_species": ncbi_counts,
        "ncbi_ast_rows": len(ncbi_ast),
    }
    (args.raw_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Download finished: %s", manifest)


if __name__ == "__main__":
    main()
