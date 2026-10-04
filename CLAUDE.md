# CLAUDE.md

Project context for Claude Code. Read `DATA_CONTRACT.md` before touching any
schema. If this file and `DATA_CONTRACT.md` disagree, the contract wins.

---

## What this project does

Given one assembled bacterial genome (FASTA), predict the **MIC** for each
antibiotic, convert that to likely-active / uncertain / likely-inactive, and rank
the active drugs.

MIC = minimum inhibitory concentration, mg/L: the lowest drug amount that stops the
germ growing **in the lab**. It is a measurement of the organism, not a patient dose.

Clinical context: bloodstream infection. Standard culture-based susceptibility
testing takes 24–48 h after a blood culture flags positive. Genome-based prediction
can answer in hours.

**Scope — 5 species:**

| Key | Species | AMRFinderPlus `-O` |
| --- | --- | --- |
| `ECOLI` | Escherichia coli | `Escherichia` |
| `KPNEU` | Klebsiella pneumoniae | `Klebsiella_pneumoniae` |
| `SAUR` | Staphylococcus aureus | `Staphylococcus_aureus` |
| `PAER` | Pseudomonas aeruginosa | `Pseudomonas_aeruginosa` |
| `ABAU` | Acinetobacter baumannii | `Acinetobacter_baumannii` |

Start with KPNEU and three drugs: ceftriaxone, meropenem, ciprofloxacin.

---

## Architecture in one pass

```
TRAINING
  BV-BRC + NCBI lab AST  ->  MIC intervals (labels.parquet)
  Training genomes       ->  known-AMR features + unitig matrix
                         ->  XGBoost AFT, one model per species x drug
                         ->  conformal bands

PREDICTION
  New genome FASTA -> AMRFinderPlus + unitig query (fixed set)
                   -> predict MIC + band
                   -> compare to breakpoints -> rank
```

Only the per-drug models are learned. Species ID, gene finding, breakpoints,
natural resistance, and ranking are existing tools or fixed rules.

---

## Non-negotiable rules

Violating any of these silently invalidates results. Flag it rather than work around it.

1. **Never impute a label.** A missing lab result is a missing row. Models train on
   the genomes that have a result for that drug and skip the rest.
2. **Never use `lineage_cluster`, `st`, `country`, `year`, `source`, or
   `isolation_source` as a model feature.** They exist for splitting and evaluation.
   Using them destroys the unseen-strain claim.
3. **Unitigs are built on training genomes only.** Every other genome is *queried*
   against that fixed set. Never rebuild the set to include test or new genomes.
4. **Per-fold feature selection.** Frequency filters and pyseer selection run inside
   each training fold, never on the full dataset.
5. **De-duplicate by `biosample`** across BV-BRC and NCBI before splitting.
6. **Lab-measured labels only.** BV-BRC also stores its own ML predictions; keep
   `evidence == 'Laboratory Method'` rows.
7. **Splits are frozen.** `data/processed/splits.parquet` is version-controlled.
   Never regenerate it to fix a problem — fix the problem.
8. **Test set is touched once**, at the very end. Tuning, thresholds, and calibration
   use validation folds.
9. **Round MIC predictions up** to the next doubling step. A high prediction is the
   safer error.
10. **Report VME first** in any results output.

---

## Key concepts

**MIC intervals.** Panels use doubling steps (…0.25, 0.5, 1, 2, 4, 8…). A reading of
`8` means the MIC is above 4 and at most 8. Every lab result — exact, `<=`, `>`, or
S/I/R-only — becomes one interval `(mic_lower, mic_upper]`. `mic_lower == 0` is
left-censored; `mic_upper == inf` is right-censored. This is why the model uses
`survival:aft`: it consumes bounds directly, so nothing is guessed at panel edges.

**Unitig.** A DNA stretch that forms one unbroken path in a de Bruijn graph across
many genomes. One unitig stands in for many overlapping k-mers that always co-occur:
same signal, far fewer columns. Lets the model catch resistance mechanisms that are
not in any curated database.

**The lineage trap.** Dropping the `st` column does NOT stop lineage learning —
thousands of unitigs act as a lineage fingerprint. The real defense is splitting by
lineage cluster, plus pyseer's mixed-model selection and reporting accuracy by
genetic distance to the nearest training genome.

**VME (very major error).** Predicted susceptible, lab says resistant. The error that
harms a patient. Target ≤ 1.5%. ME (major error) is the reverse, target ≤ 3%.

**EA (essential agreement).** Predicted MIC within ±1 doubling step of the lab MIC.
The standard MIC metric, target ≥ 90%.

---

## Repository layout

```
genome2mic/
  configs/
    species.yaml                  # keys, -O names, expected genome sizes, references
    drugs.yaml                    # names, synonyms, spectrum tiers
    breakpoints/eucast_*.csv      # species x drug x version -> S and R breakpoints
    breakpoints/clsi_*.csv
    natural_resistance.csv        # species x drug always inactive
    keep_variant.csv              # gene families where exact variant is kept
  workflow/
    Snakefile                     # per-genome: QC, AMRFinderPlus, ResFinder, mash, mlst
  data/
    raw/        ast_*.csv, genomes/<genome_id>.fasta
    interim/    <genome_id>/{amrfinder.tsv,resfinder/,mlst.tsv,mash.tsv}
    processed/  labels.parquet, qc.parquet, known_amr.parquet,
                lineages.parquet, splits.parquet,
                unitigs_<SPECIES>.{npz,_rows.parquet,_index.parquet}
  src/genome2mic/
    ingest/     bvbrc.py ncbi_ast.py harmonize.py
    features/   known_amr.py unitigs.py select.py
    splits/     lineages.py make_splits.py
    models/     b1_lookup.py b2_xgb_steps.py xgb_aft.py conformal.py multitask_nn.py
    eval/       metrics.py report.py
    predict/    pipeline.py rank.py
  app/streamlit_app.py
  results/      preds_<species>_<drug>.parquet, metrics.parquet
  tests/
```

---

## Conventions

- Python 3.11, conda env `genome2mic`.
- Parquet for all processed data. CSV only for raw downloads and configs.
  Sparse matrices as `.npz` (scipy CSR, int8).
- `genome_id` (str) is the primary key everywhere. BV-BRC format (`573.2002`) when
  available, else `NCBI_<biosample>`.
- Drug names: lowercase, hyphenated, no spaces (`piperacillin-tazobactam`).
- MIC units always mg/L (µg/mL converts 1:1).
- Missing value = null. Never `-1`, `""`, `NA`, or `0`.
- Feature column prefixes: `gene_`, `point_`, `n_class_`, `u_`.
- Every stage writes a file; no stage reaches back past its declared input.
- Pandas/pyarrow for tables; scipy.sparse for unitigs. Do not densify a unitig
  matrix — it will not fit in memory.

---

## Models

| ID | Features | Approach |
| -- | -------- | -------- |
| `b0_resfinder` | genome | ResFinder `pheno_table.txt` S/R calls. External baseline |
| `b1_lookup` | known AMR | Median MIC of training genomes with the same known-AMR profile |
| `b2_xgb_steps` | known AMR | Multi-class over log2 MIC steps, exact MICs only |
| `aft_known` | known AMR | XGBoost `survival:aft` on intervals |
| `aft_known_unitig` | known AMR + unitigs | **Main model** |
| `multitask_nn` | all | Later. Shared body, per-drug heads, masked censored-normal loss |

Main model parameters:

```python
params = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.0,   # tune
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bynode": 0.5,
}
dtrain.set_float_info("label_lower_bound", lo)
dtrain.set_float_info("label_upper_bound", hi)   # np.inf allowed
```

Uncertainty: split conformal on log2 MIC. Take |pred − true| in doubling steps on
validation rows with exact MICs; the 90th percentile `q` gives a ±`q`-step band.
Use the band's **upper** end when comparing to the breakpoint.

---

## Call logic (prediction)

| Condition | Call |
| --------- | ---- |
| `band_high` ≤ S breakpoint | Likely active |
| `band_low` > R breakpoint | Likely inactive |
| Otherwise | Uncertain — wait for lab |

Overrides applied after the model:

1. Natural resistance for the species → inactive.
2. Strong known marker (any carbapenemase for meropenem, etc.) → inactive,
   regardless of model output.
3. Species not covered, or far from all training genomes → all calls flagged low
   confidence.

Rank likely-active drugs by spectrum tier (narrowest first), then by margin in
doubling steps below the S breakpoint.

---

## Safety and claims

Every report, demo, and README must carry this: **these are predictions of in-vitro
susceptibility, not prescribing advice.** Dose, route, and final drug choice depend
on PK/PD, infection site, renal function, allergies, and other patient factors, and
remain with the clinician. Confirm with standard AST.

Do not write code or copy that implies a patient dose, a treatment decision, or
clinical validation we do not have. The system ranks; a clinician decides.

Metric targets (EA ≥ 90%, VME ≤ 1.5%, ME ≤ 3%) are figures commonly used in AST
device evaluation. Do not present them as regulatory thresholds we have met.

---

## Working agreements for Claude Code

- **Read `DATA_CONTRACT.md` before changing any schema.** If a change is needed,
  update the contract in the same commit and say so in the message.
- **Ask before changing**: split logic, the interval rule, feature prefixes, or
  anything on the non-negotiable list.
- **Write the test first** for anything in `ingest/harmonize.py`, `eval/metrics.py`,
  or the interval conversion. These are where silent errors hide.
- **Do not add a dependency** without saying why. Current stack: pandas, pyarrow,
  numpy, scipy, xgboost, scikit-learn, shap, streamlit, snakemake.
- **Per-genome tool calls go in the Snakefile**, not in Python loops.
- **Log dropped rows with reasons.** Every filter emits a count. Silent drops make
  dataset bugs invisible.
- **Prefer small, checkable steps.** Emit the counts table after stage 2 before
  anyone trains anything.
- When unsure whether something is leakage, assume it is and raise it.

---

## Current state

- [ ] Stage 1 — ingest + harmonize → `labels.parquet`, `pairs_kept.csv`
- [ ] Stage 2 — genomes + QC
- [ ] Stage 3 — lineages + splits (freeze)
- [ ] Stage 4 — AMRFinderPlus → `known_amr.parquet`
- [ ] Stage 5 — metrics module + unit tests
- [ ] Stage 6 — baselines B0–B2
- [ ] Stage 7 — AFT on known AMR only
- [ ] Stage 8 — unitigs: build, query, filter, collapse
- [ ] Stage 9 — AFT on known AMR + unitigs; ablations
- [ ] Stage 10 — conformal bands + ranking + report
- [ ] Stage 11 — external validation + distance plots

Update this list as stages land.

---

## Open questions

- PopPUNK or Mash single-linkage for lineage clusters? Test both on KPNEU.
- Which breakpoint version to standardize on for the final call?
- Re-derive all S/I/R from MIC under one standard, or use labels as reported?
- Is 50 resistant / 50 susceptible the right inclusion bar after seeing real counts?
