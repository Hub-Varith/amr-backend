"""Suite-wide fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Real bioinformatics tools whose presence on PATH changes which backend the code picks.
_REAL_TOOLS = ("amrfinder", "mash", "mlst", "unitig-caller", "resfinder", "poppunk", "pyseer")


@pytest.fixture(autouse=True)
def _hermetic_tool_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide locally installed bioinformatics tools so results do not depend on the machine.

    Tests that need a tool put a stand-in executable on PATH themselves; that still works
    because this fixture runs first.
    """
    entries = os.environ.get("PATH", "").split(os.pathsep)
    kept = [d for d in entries if not any((Path(d) / tool).exists() for tool in _REAL_TOOLS)]
    monkeypatch.setenv("PATH", os.pathsep.join(kept))
