"""CLI: gather per-genome stats and Mash results into data/processed/qc.parquet (stage 3)."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from genome2mic.qc.assembly_qc import AssemblyQc
from genome2mic.validate.qc_checks import QcValidator

logger = logging.getLogger(__name__)

SEQKIT_COLUMNS = {"num_seqs": "n_contigs", "sum_len": "total_length", "N50": "n50", "GC(%)": "gc_percent"}
MASH_COLUMNS = ["reference_path", "query_path", "distance", "p_value", "shared_hashes"]


def main() -> None:
    """Read every genome's stats.tsv and mash.tsv, apply the QC rules, check, and write qc.parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    species_config = yaml.safe_load((args.configs_dir / "species.yaml").read_text())["species"]
    expected_sizes = {key: spec["expected_genome_size"] for key, spec in species_config.items()}
    manifest = pd.read_csv(args.interim_dir / "genome_manifest.tsv", sep="\t", dtype=str)

    stats_frames = []
    mash_frames = []
    for genome_id in manifest["genome_id"]:
        genome_dir = args.interim_dir / genome_id
        stats_path = genome_dir / "stats.tsv"
        if stats_path.exists():
            stats = pd.read_csv(stats_path, sep="\t", usecols=list(SEQKIT_COLUMNS)).rename(columns=SEQKIT_COLUMNS)
            stats_frames.append(stats.assign(genome_id=genome_id))
        mash_path = genome_dir / "mash.tsv"
        if mash_path.exists() and mash_path.stat().st_size > 0:
            mash = pd.read_csv(mash_path, sep="\t", header=None, names=MASH_COLUMNS)
            # Reference files are named <SPECIES>.fna.gz.
            reference_species = mash["reference_path"].str.split("/").str[-1].str.split(".").str[0]
            mash_frames.append(pd.DataFrame(
                {"genome_id": genome_id, "reference_species": reference_species, "distance": mash["distance"]}
            ))
    stats = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame(
        columns=["genome_id", *SEQKIT_COLUMNS.values()]
    )
    mash = pd.concat(mash_frames, ignore_index=True) if mash_frames else pd.DataFrame(
        columns=["genome_id", "reference_species", "distance"]
    )
    logger.info("Read tool output: stats=%s mash_genomes=%s of %s", len(stats), len(mash_frames), len(manifest))

    qc = AssemblyQc(expected_sizes).evaluate(manifest, stats, mash)
    QcValidator.check(qc)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    qc.to_parquet(args.processed_dir / "qc.parquet", index=False)
    logger.info("Stage 3 done: genomes=%s qc_pass=%s", len(qc), int(qc["qc_pass"].sum()))


if __name__ == "__main__":
    main()
