"""Download lab AST rows and genome metadata from the BV-BRC API."""

import logging
import urllib.parse

import pandas as pd

from genome2mic.ingest.constants import (
    BVBRC_API,
    BVBRC_GENOME_FIELDS,
    BVBRC_ID_BATCH,
    BVBRC_LAB_EVIDENCE,
    BVBRC_PAGE_SIZE,
)
from genome2mic.ingest.http_client import HttpClient

logger = logging.getLogger(__name__)

RQL_HEADERS = {"Accept": "application/json", "Content-Type": "application/rqlquery+x-www-form-urlencoded"}


class BvbrcDownloader:
    """Fetches BV-BRC `genome_amr` rows and `genome` metadata as DataFrames."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client

    def download_ast(self, species_name: str) -> pd.DataFrame:
        """Return every lab-measured AST row whose genome name starts with the species name.

        Taxon ID alone misses subspecies taxa (e.g. 72407), so the genome name prefix is used.
        The species is confirmed later against genome metadata.
        """
        logger.info("BV-BRC AST download start: %s", species_name)
        evidence = urllib.parse.quote(f'"{BVBRC_LAB_EVIDENCE}"')
        name_prefix = urllib.parse.quote(f'"{species_name}*"', safe="*")
        query = f"and(eq(evidence,{evidence}),eq(genome_name,{name_prefix}))&sort(%2Bid)"
        pages = []
        offset = 0
        while True:
            rows = self.client.post_json(
                f"{BVBRC_API}/genome_amr/", f"{query}&limit({BVBRC_PAGE_SIZE},{offset})", RQL_HEADERS
            )
            pages.append(pd.DataFrame(rows))
            logger.info("BV-BRC AST page: %s offset=%s rows=%s", species_name, offset, len(rows))
            if len(rows) < BVBRC_PAGE_SIZE:
                break
            offset += BVBRC_PAGE_SIZE
        ast = pd.concat(pages, ignore_index=True)
        logger.info("BV-BRC AST download done: %s rows=%s", species_name, len(ast))
        return ast

    def download_genome_metadata(self, genome_ids: list[str], biosamples: list[str]) -> pd.DataFrame:
        """Return metadata for the given genome IDs plus every genome that carries one of the biosamples."""
        logger.info("BV-BRC metadata download start: genome_ids=%s biosamples=%s", len(genome_ids), len(biosamples))
        fields = ",".join(BVBRC_GENOME_FIELDS)
        frames = []
        for field_name, values in (("genome_id", genome_ids), ("biosample_accession", biosamples)):
            for start in range(0, len(values), BVBRC_ID_BATCH):
                batch = values[start : start + BVBRC_ID_BATCH]
                query = f"in({field_name},({','.join(batch)}))&select({fields})&limit({BVBRC_ID_BATCH * 4})"
                rows = self.client.post_json(f"{BVBRC_API}/genome/", query, RQL_HEADERS)
                frames.append(pd.DataFrame(rows))
        metadata = pd.concat(frames, ignore_index=True).drop_duplicates(subset="genome_id")
        logger.info("BV-BRC metadata download done: genomes=%s", len(metadata))
        return metadata
