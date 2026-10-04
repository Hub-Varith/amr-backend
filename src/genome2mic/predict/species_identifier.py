"""Species of an upload from Mash distances to the five references (stage 3 QC rule)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from genome2mic.predict.tool_runner import ToolRunner
from genome2mic.qc.constants import MAX_MASH_DISTANCE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesResult:
    species: str | None          # 5-letter key; None when no reference is within MAX_MASH_DISTANCE
    mash_distance: float | None  # to the nearest reference


class SpeciesIdentifier:
    """`mash dist references.msh upload.fasta`. Reference files are named <SPECIES>.fna[.gz]."""

    def __init__(self, references_sketch: Path, species_keys: list[str], runner: ToolRunner) -> None:
        self.references_sketch = references_sketch
        self.species_keys = species_keys
        self.runner = runner

    def identify(self, fasta_path: Path) -> SpeciesResult:
        output = self.runner.run(["mash", "dist", str(self.references_sketch), str(fasta_path)])
        return self.parse(output, self.species_keys)

    @staticmethod
    def parse(mash_output: str, species_keys: list[str]) -> SpeciesResult:
        """Columns: reference, query, distance, p-value, shared hashes."""
        best_species, best_distance = None, None
        for line in mash_output.strip().splitlines():
            reference, _, distance, *_ = line.split("\t")
            key = Path(reference).name.split(".")[0]
            if key not in species_keys:
                continue
            if best_distance is None or float(distance) < best_distance:
                best_species, best_distance = key, float(distance)
        if best_distance is None:
            return SpeciesResult(None, None)
        species = best_species if best_distance <= MAX_MASH_DISTANCE else None
        logger.info("Species identified", extra={"nearest": best_species, "distance": best_distance, "species": species})
        return SpeciesResult(species, best_distance)
