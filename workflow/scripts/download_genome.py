"""Download one genome as gzipped FASTA from BV-BRC or NCBI. Standard library only (runs in genome2mic-tools).

A failed download writes an empty file so the run continues; QC then marks it `download_failed`.
To retry failed genomes, delete their empty files and run `make genomes` again.
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

BVBRC_URL = "https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{accession})&limit(25000)"
# BV-BRC blocks Python's default user agent with HTTP 403.
USER_AGENT = "genome2mic/0.1 (+https://github.com/Hub-Varith/amr-backend)"
ATTEMPTS = 3
TIMEOUT_SECONDS = 300


class GenomeDownloader:
    """Fetches one assembly and writes it atomically, so a half-written file never looks finished."""

    @staticmethod
    def fetch_bvbrc(accession: str, output_path: str) -> None:
        """Write BV-BRC contigs for one genome ID."""
        request = urllib.request.Request(
            BVBRC_URL.format(accession=accession),
            headers={"Accept": "application/dna+fasta", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            fasta = response.read()
        if not fasta.startswith(b">"):
            raise ValueError(f"BV-BRC returned no FASTA records for {accession}")
        with gzip.open(output_path, "wb") as handle:
            handle.write(fasta)

    @staticmethod
    def fetch_ncbi(accession: str, output_path: str) -> None:
        """Write the genomic FASTA of one NCBI assembly accession using the datasets CLI."""
        with tempfile.TemporaryDirectory() as work_dir:
            zip_path = os.path.join(work_dir, "genome.zip")
            subprocess.run(
                ["datasets", "download", "genome", "accession", accession, "--include", "genome",
                 "--filename", zip_path, "--no-progressbar"],
                check=True,
                timeout=TIMEOUT_SECONDS,
            )
            with zipfile.ZipFile(zip_path) as archive:
                fasta_names = [name for name in archive.namelist() if name.endswith(".fna")]
                if not fasta_names:
                    raise ValueError(f"NCBI returned no genomic FASTA for {accession}")
                with archive.open(fasta_names[0]) as source, gzip.open(output_path, "wb") as target:
                    shutil.copyfileobj(source, target)


def main() -> None:
    """Try the download a few times; on final failure write an empty file, or exit 1 with --strict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["bvbrc", "ncbi"], required=True)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true", help="Exit 1 on failure (used for reference genomes)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    partial_path = args.output + ".part"
    fetch = GenomeDownloader.fetch_bvbrc if args.source == "bvbrc" else GenomeDownloader.fetch_ncbi
    for attempt in range(1, ATTEMPTS + 1):
        try:
            fetch(args.accession, partial_path)
            os.replace(partial_path, args.output)
            print(f"downloaded {args.source} {args.accession} on attempt {attempt}")
            return
        except Exception as error:  # noqa: BLE001 - every failure is retried, then recorded
            print(f"WARNING attempt {attempt} failed for {args.source} {args.accession}: {error}")
            time.sleep(2**attempt)

    if args.strict:
        sys.exit(1)
    with gzip.open(partial_path, "wb"):
        pass
    os.replace(partial_path, args.output)
    print(f"FAILED {args.source} {args.accession}: wrote an empty file")


if __name__ == "__main__":
    main()
