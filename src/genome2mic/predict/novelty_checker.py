"""Distance from an upload to the nearest training genome (CLAUDE.md override 3)."""

import logging
from dataclasses import dataclass
from pathlib import Path

from genome2mic.predict.tool_runner import ToolRunner

logger = logging.getLogger(__name__)

# Mash distance ~ 1 - ANI. 0.02 (about 98% ANI) is within a species but outside a lineage's
# usual spread. A provisional choice: INTEGRATION.md section 8 lists the threshold as open.
MAX_TRAINING_DISTANCE = 0.02


@dataclass(frozen=True)
class NoveltyResult:
    nearest_training_distance: float | None
    in_range: bool
    checked: bool          # False when no training sketch exists (the distance is then unknown)


class NoveltyChecker:
    """Uses <sketch_dir>/training_<SPECIES>.msh when present.

    The hackathon releases ship no genome sequences, so no training sketch exists yet. Without
    one, the distance is reported as null and in_range follows the species check alone; the
    report then cannot warn about a strain unlike anything in training.
    """

    def __init__(self, sketch_dir: Path, runner: ToolRunner, max_distance: float = MAX_TRAINING_DISTANCE) -> None:
        self.sketch_dir = sketch_dir
        self.runner = runner
        self.max_distance = max_distance

    def sketch_for(self, species: str) -> Path:
        return self.sketch_dir / f"training_{species}.msh"

    def check(self, fasta_path: Path, species: str) -> NoveltyResult:
        sketch = self.sketch_for(species)
        if not sketch.exists():
            logger.warning("No training sketch; novelty not checked", extra={"species": species, "sketch": str(sketch)})
            return NoveltyResult(None, True, False)
        output = self.runner.run(["mash", "dist", str(sketch), str(fasta_path)])
        distances = [float(line.split("\t")[2]) for line in output.strip().splitlines() if line]
        nearest = min(distances) if distances else None
        in_range = nearest is not None and nearest <= self.max_distance
        logger.info("Novelty checked", extra={"species": species, "nearest": nearest, "in_range": in_range})
        return NoveltyResult(nearest, in_range, True)
