# Synthetic demo -- SYNTHETIC DATA

**Everything in this directory was produced from simulated data.** The genomes, the
lab MIC results, the trained models and every number, table and figure here come from
the seeded synthetic generator (`python -m genome2mic synth`). They show that the
pipeline runs end to end. They say nothing about how the method performs on real
isolates and must not be quoted as performance figures.

| File | How it is marked as synthetic |
| ---- | ----------------------------- |
| `report.md` | Opens with a SYNTHETIC DATA banner |
| `figures/*.png` | Diagonal `SYNTHETIC DATA` watermark, a note line above the plot, and the same note in the PNG `Description` metadata |
| `metrics.csv`, `metrics_by_distance.csv` | A leading `synthetic` column, `True` on every row; the remaining columns are the evaluate stage's output unchanged (VME first) |

The metric targets quoted in the report (EA >= 90 %, VME <= 1.5 %, ME <= 3 %) are
figures commonly used in AST device evaluation, listed as reference points only.
They are not regulatory thresholds, and nothing here claims they were met.

> These are predictions of in-vitro susceptibility, not prescribing advice. Dose, route, and final drug choice depend on PK/PD, infection site, renal function, allergies, and other patient factors, and remain with the clinician. Confirm with standard AST.

Regenerate with `make clean-synth demo` (or `make demo-copy` after a synthetic run).
`demo-copy` refuses a run root without `data/raw/SYNTHETIC_DATA.md`.
