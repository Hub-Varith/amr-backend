"""Seeded synthetic raw-data generator (``python -m genome2mic synth``).

Everything under this package produces **simulated** data so the pipeline can run
end to end without any real genomes or lab results. Every artefact it writes is
labelled synthetic where the file format allows (FASTA headers, ResFinder comment
lines, ``data/raw/SYNTHETIC_DATA.md``). Metrics computed on this data say nothing
about real organisms and must never be presented as real.

Modules:

- :mod:`genome2mic.synthetic.markers` -- the planted resistance markers (fixed DNA
  blocks), their AMRFinderPlus annotations, lineage-tier presence probabilities and
  log2 MIC effects.
- :mod:`genome2mic.synthetic.generate` -- :func:`~genome2mic.synthetic.generate.run`,
  which writes the whole fake raw layer under ``paths.root``.

``run`` is re-exported lazily so ``python -m genome2mic.synthetic.generate`` does not
import the module twice.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run"]


def __getattr__(name: str) -> Any:
    if name == "run":
        from genome2mic.synthetic.generate import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
