"""Download NCBI BioSample antibiograms and flatten them into AST and metadata tables."""

import gzip
import logging
import time
import urllib.parse
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pandas as pd

from genome2mic.ingest.constants import NCBI_EUTILS, NCBI_FETCH_BATCH, NCBI_REQUEST_PAUSE_SECONDS
from genome2mic.ingest.http_client import HttpClient

logger = logging.getLogger(__name__)

META_ATTRIBUTES = ("strain", "collection_date", "geo_loc_name", "isolation_source", "host")


class NcbiAstDownloader:
    """Saves raw BioSample XML per species, then parses antibiogram tables and sample attributes."""

    def __init__(self, client: HttpClient, xml_dir: Path) -> None:
        self.client = client
        self.xml_dir = xml_dir

    def download_xml(self, species_key: str, species_name: str) -> int:
        """Save every BioSample with an antibiogram for one species as gzipped XML batches. Returns the count."""
        self.xml_dir.mkdir(parents=True, exist_ok=True)
        term = urllib.parse.quote(f'antibiogram[filter] AND "{species_name}"[orgn]')
        search = self.client.get_json(
            f"{NCBI_EUTILS}/esearch.fcgi?db=biosample&term={term}&usehistory=y&retmax=0&retmode=json"
        )["esearchresult"]
        count = int(search["count"])
        logger.info("NCBI download start: %s biosamples=%s", species_key, count)
        for start in range(0, count, NCBI_FETCH_BATCH):
            batch_path = self.xml_dir / f"{species_key}_{start:06d}.xml.gz"
            if batch_path.exists():
                continue
            url = (
                f"{NCBI_EUTILS}/efetch.fcgi?db=biosample&WebEnv={search['webenv']}"
                f"&query_key={search['querykey']}&retstart={start}&retmax={NCBI_FETCH_BATCH}&retmode=xml"
            )
            xml_text = self.client.get_text(url)
            with gzip.open(batch_path, "wt", encoding="utf-8") as handle:
                handle.write(xml_text)
            time.sleep(NCBI_REQUEST_PAUSE_SECONDS)
        logger.info("NCBI download done: %s", species_key)
        return count

    def parse_xml(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (ast, metadata) tables from every saved XML batch."""
        ast_rows = []
        meta_rows = []
        for batch_path in sorted(self.xml_dir.glob("*.xml.gz")):
            with gzip.open(batch_path, "rt", encoding="utf-8") as handle:
                root = ElementTree.parse(handle).getroot()
            for sample in root.iter("BioSample"):
                biosample = sample.get("accession")
                organism = sample.find("Description/Organism")
                organism_name = organism.get("taxonomy_name") if organism is not None else None
                meta_row = {"biosample": biosample, "organism": organism_name}
                for attribute in sample.iter("Attribute"):
                    name = attribute.get("harmonized_name") or attribute.get("attribute_name")
                    if name in META_ATTRIBUTES:
                        meta_row[name] = attribute.text
                meta_rows.append(meta_row)
                for table in sample.iter("Table"):
                    if not (table.get("class") or "").startswith("Antibiogram"):
                        continue
                    header = [
                        (cell.text or "").strip().lower().replace(" ", "_")
                        for cell in table.findall("Header/Cell")
                    ]
                    for row in table.findall("Body/Row"):
                        values = [(cell.text or "").strip() for cell in row.findall("Cell")]
                        ast_row = {"biosample": biosample, "organism": organism_name}
                        ast_row.update(zip(header, values))
                        ast_rows.append(ast_row)
        ast = pd.DataFrame(ast_rows)
        metadata = pd.DataFrame(meta_rows).drop_duplicates(subset="biosample")
        logger.info("NCBI parse done: ast_rows=%s biosamples=%s", len(ast), len(metadata))
        return ast, metadata
