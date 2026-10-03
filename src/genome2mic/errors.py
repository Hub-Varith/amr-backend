"""Project-wide exception types.

Two failure modes matter enough to get their own class:

* :class:`ToolNotAvailable` -- an external CLI (``amrfinder``, ``mash``, ``mlst``,
  ``unitig-caller``, ``pyseer``, ``poppunk``, ``resfinder``) is not on ``PATH`` and
  no pure-Python fallback exists for the requested operation.
* :class:`ContractViolation` -- a stage output breaks a rule in ``DATA_CONTRACT.md``
  (duplicate primary key, ``mic_lower >= mic_upper``, a lineage cluster in two
  splits, ...). Raised instead of ``assert`` so it survives ``python -O`` and can be
  caught explicitly.
"""

from __future__ import annotations


class Genome2MicError(Exception):
    """Base class for every error raised by genome2mic."""


class ToolNotAvailable(Genome2MicError):
    """An external command-line tool is required but not installed.

    Args:
        tool: Executable name that was looked up on ``PATH``.
        hint: Optional one-line pointer (install command, fallback flag, ...).
    """

    def __init__(self, tool: str, hint: str | None = None) -> None:
        self.tool = tool
        self.hint = hint
        message = f"External tool {tool!r} is not available on PATH."
        if hint:
            message = f"{message} {hint}"
        super().__init__(message)


class ContractViolation(Genome2MicError):
    """A stage output violates ``DATA_CONTRACT.md``.

    Args:
        message: What was violated, including the counts or keys involved.
        stage: Optional stage name (``ingest``, ``splits``, ...) for log filtering.
    """

    def __init__(self, message: str, stage: str | None = None) -> None:
        self.stage = stage
        prefix = f"[{stage}] " if stage else ""
        super().__init__(f"{prefix}{message}")
