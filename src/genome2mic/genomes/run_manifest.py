"""CLI: write data/interim/genome_manifest.tsv, the list of genomes for Snakemake to download (stage 3)."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from genome2mic.genomes.genome_manifest import GenomeManifestBuilder

logger = logging.getLogger(__name__)

PILOT_SEED = 0


def main() -> None:
    """Build the manifest for the given species, or a fixed pilot subset of it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="+", required=True)
    parser.add_argument("--pilot", type=int, help="Keep only this many random genomes (throwaway test run)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    labels = pd.read_parquet(args.processed_dir / "labels.parquet", columns=["genome_id", "species", "drug"])
    pairs_kept = pd.read_csv(args.processed_dir / "pairs_kept.csv")
    manifests = []
    dropped = []
    for species in args.species:
        assemblies = GenomeManifestBuilder.read_assembly_reports(args.raw_dir / f"ncbi_assemblies_{species}.jsonl.gz")
        species_manifest, species_dropped = GenomeManifestBuilder.build(labels, pairs_kept, assemblies, species)
        manifests.append(species_manifest)
        dropped.append(species_dropped)
    manifest = pd.concat(manifests, ignore_index=True)

    args.interim_dir.mkdir(parents=True, exist_ok=True)
    # The PILOT marker stops push-data from publishing results built on the subset.
    pilot_marker = args.interim_dir / "PILOT"
    if args.pilot:
        manifest = GenomeManifestBuilder.pilot_sample(manifest, args.pilot, PILOT_SEED)
        pilot_marker.write_text(f"pilot of {len(manifest)} genomes, seed {PILOT_SEED}\n")
        logger.warning("PILOT run: %s genomes. Do not build splits or push a release from it.", len(manifest))
    else:
        pilot_marker.unlink(missing_ok=True)

    manifest.to_csv(args.interim_dir / "genome_manifest.tsv", sep="\t", index=False)
    pd.concat(dropped, ignore_index=True).to_parquet(args.processed_dir / "dropped_genomes.parquet", index=False)
    logger.info("Manifest written: genomes=%s by source=%s", len(manifest), manifest["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
