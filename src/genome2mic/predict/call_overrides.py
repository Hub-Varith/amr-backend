"""Overrides applied after the model (CLAUDE.md call logic, overrides 1 and 2)."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from genome2mic.features.amrfinder_table import AmrFinderHit, AmrFinderTable

logger = logging.getLogger(__name__)

NATURAL_RESISTANCE = "natural_resistance"
STRONG_MARKER = "strong_marker"


@dataclass(frozen=True)
class Override:
    kind: str                 # natural_resistance / strong_marker
    markers: list[str]        # hit symbols behind a strong_marker override


class CallOverrides:
    """configs/natural_resistance.csv and configs/strong_markers.csv. Either forces likely_inactive."""

    def __init__(self, natural: pd.DataFrame, markers: pd.DataFrame) -> None:
        self.natural = set(zip(natural["species"], natural["drug"]))
        self.markers = {
            (row.species, row.drug): re.compile(row.symbol_regex) for row in markers.itertuples(index=False)
        }

    @classmethod
    def from_configs(cls, configs_dir: Path) -> "CallOverrides":
        return cls(
            pd.read_csv(configs_dir / "natural_resistance.csv", comment="#"),
            pd.read_csv(configs_dir / "strong_markers.csv", comment="#"),
        )

    def find(self, species: str, drug: str, hits: list[AmrFinderHit]) -> Override | None:
        if (species, drug) in self.natural:
            return Override(NATURAL_RESISTANCE, [])
        pattern = self.markers.get((species, drug))
        if pattern is None:
            return None
        # A gene with an internal stop codon is likely broken, so it never triggers an override.
        markers = [
            hit.symbol for hit in hits
            if hit.type == "AMR" and AmrFinderTable.ncbi_tag(hit) != "MISTRANSLATION" and pattern.search(hit.symbol)
        ]
        return Override(STRONG_MARKER, markers) if markers else None
