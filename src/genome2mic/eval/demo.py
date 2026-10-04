"""Copy a synthetic run's report artefacts into the tracked demo directory (``make demo-copy``).

``reports/synthetic_demo/`` is committed, so each file in it may be read on its own,
away from the report that explains it. :func:`copy_demo` therefore makes every
artefact say that it is synthetic:

* ``report.md``         -- copied as is; it already opens with the SYNTHETIC DATA banner.
* ``figures/*.png``     -- copied as is; the report stage stamps every figure of a
  synthetic run with a ``SYNTHETIC DATA`` watermark, a note line and PNG metadata.
  Stale PNGs from an earlier demo are removed first.
* ``metrics.csv``, ``metrics_by_distance.csv`` -- re-written with a leading
  ``synthetic`` column (``True`` on every row). The other columns are unchanged.
* ``README.md``         -- written here: everything in the directory is simulated.

The copy refuses a root without ``data/raw/SYNTHETIC_DATA.md``: real results must
never land in a directory that labels them synthetic (and committed).

Usage::

    python -m genome2mic.eval.demo --root runs/synthetic --out reports/synthetic_demo
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from genome2mic.api.constants import DISCLAIMER
from genome2mic.eval.report import SYNTHETIC_MARKER
from genome2mic.io import write_csv
from genome2mic.paths import Paths

logger = logging.getLogger(__name__)

SYNTHETIC_COLUMN = "synthetic"
"""Marker column prepended to every metrics CSV in the demo directory."""

METRIC_CSVS: tuple[str, ...] = ("metrics.csv", "metrics_by_distance.csv")

README_TEXT = f"""# Synthetic demo -- SYNTHETIC DATA

**Everything in this directory was produced from simulated data.** The genomes, the
lab MIC results, the trained models and every number, table and figure here come from
the seeded synthetic generator (`python -m genome2mic synth`). They show that the
pipeline runs end to end. They say nothing about how the method performs on real
isolates and must not be quoted as performance figures.

| File | How it is marked as synthetic |
| ---- | ----------------------------- |
| `report.md` | Opens with a SYNTHETIC DATA banner |
| `figures/*.png` | Diagonal `SYNTHETIC DATA` watermark, a note line above the plot, and the same note in the PNG `Description` metadata |
| `metrics.csv`, `metrics_by_distance.csv` | A leading `{SYNTHETIC_COLUMN}` column, `True` on every row; the remaining columns are the evaluate stage's output unchanged (VME first) |

The metric targets quoted in the report (EA >= 90 %, VME <= 1.5 %, ME <= 3 %) are
figures commonly used in AST device evaluation, listed as reference points only.
They are not regulatory thresholds, and nothing here claims they were met.

> {DISCLAIMER}

Regenerate with `make clean-synth demo` (or `make demo-copy` after a synthetic run).
`demo-copy` refuses a run root without `data/raw/SYNTHETIC_DATA.md`.
"""
"""Content of ``README.md`` in the demo directory."""


def _require_synthetic(paths: Paths) -> None:
    marker = paths.raw_dir / SYNTHETIC_MARKER
    if not marker.is_file():
        raise ValueError(
            f"{marker} not found: {paths.root} is not a synthetic run. Refusing to copy its results "
            "into a directory that labels them synthetic."
        )


def _require_inputs(paths: Paths) -> list[Path]:
    sources = [paths.report_md, *(paths.results_dir / name for name in METRIC_CSVS)]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"run the evaluate and report stages first; missing: {missing}")
    return sources


def _labelled_csv(source: Path, target: Path) -> Path:
    """``source`` with a leading ``synthetic`` column (``True`` on every row) written to ``target``."""
    frame = pd.read_csv(source)
    if SYNTHETIC_COLUMN in frame.columns:
        frame = frame.drop(columns=SYNTHETIC_COLUMN)
    frame.insert(0, SYNTHETIC_COLUMN, True)
    return write_csv(frame, target)


def copy_demo(paths: Paths, demo_dir: Path) -> list[Path]:
    """Copy report, figures and metrics CSVs of the synthetic run at ``paths`` into ``demo_dir``.

    Returns the written paths. Raises ``ValueError`` when the run is not synthetic
    (nothing is written) and ``FileNotFoundError`` when the evaluate / report outputs
    are missing.
    """
    _require_synthetic(paths)
    _require_inputs(paths)
    demo_dir = Path(demo_dir)
    figures_out = demo_dir / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)

    stale = sorted(figures_out.glob("*.png"))
    for png in stale:
        png.unlink()
    logger.info("removed %d stale figure(s) from %s", len(stale), figures_out)

    written: list[Path] = []
    written.append(Path(shutil.copyfile(paths.report_md, demo_dir / "report.md")))
    for name in METRIC_CSVS:
        written.append(_labelled_csv(paths.results_dir / name, demo_dir / name))
    pngs = sorted(paths.figures_dir.glob("*.png")) if paths.figures_dir.is_dir() else []
    for png in pngs:
        written.append(Path(shutil.copyfile(png, figures_out / png.name)))
    readme = demo_dir / "README.md"
    readme.write_text(README_TEXT, encoding="utf-8")
    written.append(readme)
    logger.info("copied the SYNTHETIC demo to %s: report, %d metrics CSVs, %d figures, README", demo_dir, len(METRIC_CSVS), len(pngs))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m genome2mic.eval.demo --root <run root> --out <demo dir>``."""
    parser = argparse.ArgumentParser(prog="python -m genome2mic.eval.demo", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="Synthetic run root (contains data/raw/SYNTHETIC_DATA.md)")
    parser.add_argument("--out", type=Path, required=True, help="Demo directory, e.g. reports/synthetic_demo")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    written = copy_demo(Paths(root=args.root), args.out)
    print(f"demo copied to {args.out}: {len(written)} files (SYNTHETIC data; not clinical validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
