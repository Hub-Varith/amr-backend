"""Runs the external bioinformatics tools (Mash, AMRFinderPlus) for one upload."""

import logging
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolError(RuntimeError):
    """A tool was missing, timed out, or exited non-zero. The message quotes the end of stderr."""


class ToolRunner:
    """Thin subprocess wrapper: one place for timeouts, logging and readable errors."""

    def __init__(self, timeout_seconds: int = 1800) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def require(*executables: str) -> None:
        """Raise FileNotFoundError when a tool is not on PATH, so the API reports 'not ready'."""
        missing = [name for name in executables if shutil.which(name) is None]
        if missing:
            raise FileNotFoundError(f"tools not on PATH: {', '.join(missing)} (they run on Linux; see Dockerfile)")

    def run(self, arguments: list[str], cwd: Path | None = None) -> str:
        """Run and return stdout."""
        started = time.monotonic()
        try:
            completed = subprocess.run(
                arguments, cwd=cwd, capture_output=True, text=True, timeout=self.timeout_seconds, check=False
            )
        except FileNotFoundError as error:
            raise ToolError(f"{arguments[0]} not found") from error
        except subprocess.TimeoutExpired as error:
            raise ToolError(f"{arguments[0]} timed out after {self.timeout_seconds}s") from error
        elapsed = round(time.monotonic() - started, 1)
        if completed.returncode != 0:
            tail = completed.stderr.strip().splitlines()[-5:]
            raise ToolError(f"{arguments[0]} exited {completed.returncode}: {' | '.join(tail)}")
        logger.info("Tool finished", extra={"tool": arguments[0], "seconds": elapsed})
        return completed.stdout
