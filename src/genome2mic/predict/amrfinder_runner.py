"""Runs AMRFinderPlus on one upload (DATA_CONTRACT.md stage 4)."""

import logging
from pathlib import Path

from genome2mic.features.amrfinder_table import AmrFinderHit, AmrFinderTable
from genome2mic.predict.tool_runner import ToolRunner

logger = logging.getLogger(__name__)


class AmrFinderRunner:
    """`amrfinder -n <fasta> -O <organism> --plus`, the way NCBI built the training features."""

    def __init__(self, runner: ToolRunner, database_dir: Path | None = None, threads: int = 4) -> None:
        self.runner = runner
        self.database_dir = database_dir
        self.threads = threads

    def _with_database(self, arguments: list[str]) -> list[str]:
        return arguments + ["-d", str(self.database_dir)] if self.database_dir is not None else arguments

    def run(self, fasta_path: Path, organism: str, output_path: Path) -> list[AmrFinderHit]:
        self.runner.run(self._with_database([
            "amrfinder", "-n", str(fasta_path), "-O", organism, "--plus",
            "--threads", str(self.threads), "-o", str(output_path),
        ]))
        return AmrFinderTable.parse(output_path)

    def describe_database(self) -> tuple[str, Path]:
        """(database version, database directory). The directory holds the class tables training used."""
        output = self.runner.run(self._with_database(["amrfinder", "--database_version"]))
        version, directory = "unknown", self.database_dir
        for line in output.splitlines():
            key, _, value = line.partition(":")
            value = value.strip().strip("'\"")
            if key.strip() == "Database version":
                version = value
            elif key.strip() == "Database directory" and directory is None:
                directory = Path(value)
        if directory is None:
            raise FileNotFoundError("could not find the AMRFinderPlus database directory; set G2M_AMRFINDER_DB")
        return version, directory
