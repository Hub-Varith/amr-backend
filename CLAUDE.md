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
    intrinsic_markers.csv         # species x intrinsic gene, never a strong marker
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
    predict/    pipeline.py rank.py release_features.py
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
- Per-genome Python work that must scale goes through `genome2mic.parallel.ordered_map`
  (results in input order, so outputs never depend on the core count). External tools
  still go in the Snakefile.

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
dtrain.set_weight(w)   # exact-MIC rows 2x, others 1, mean 1 (train --exact-weight 2.0)
```

Uncertainty: cross-conformal asymmetric band on log2 MIC (`train --band asym_tuned`,
the default; `--band symmetric` is the old ±`q` band at 90 %). The lower half-width is
the 95 % quantile of over-prediction residuals on exact MICs, widened when the
leave-one-fold-out values disagree by more than one step. The upper level (92 to 99.5 %)
is tuned per species × drug inside the training folds: the narrowest level whose nested
cross-validated call VME, counted only over folds that issue likely-active calls, passes
(n_VME + 1) / (n_R + 1) ≤ 1.5 %. The shipped bundle may only use a level that also
passed in every CV fold that issues calls, and closes its active-call gate if there is
none or if its own out-of-fold CV calls fail the same rule. Gate closed → likely-active
calls are withheld (shown as uncertain) for that drug. Use the band's **upper** end when
comparing to the breakpoint.

---

## Call logic (prediction)

| Condition | Call |
| --------- | ---- |
| `band_high` ≤ S breakpoint | Likely active |
| `band_low` > R breakpoint | Likely inactive |
| Bundle's `active_gate_open` false and the band says active | Uncertain (reason `likely_active withheld ...`) |
| Otherwise | Uncertain — wait for lab |

Overrides applied after the model:

1. Natural resistance for the species → inactive.
2. Strong known marker → inactive, regardless of model output: a `strong_markers`
   column (any carbapenemase family/variant column for the carbapenems, etc.) or, for
   drugs with `strong_subclasses`, any acquired gene whose AMRFinderPlus Subclass is
   listed (CARBAPENEM for the carbapenems). Point mutations never trigger it, and
   neither do the species' intrinsic genes in `configs/intrinsic_markers.csv` (the
   OXA-51 family for ABAU).
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

Whenever a count of pairs passing call VME is quoted, also give how many of those make
any likely-active calls and how many pass only because the gate is closed or natural
resistance / no breakpoint applies. CV numbers are selection-biased (about 6
point-model candidates and 29 band variants were compared on the same OOF rows) and, on
the current release, measure generalisation to new NCBI SNP clusters, not new lineages.

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

All stages below run end to end on the seeded synthetic data
(`python -m genome2mic run-all --root runs/synthetic`, ~2.5 min; `make demo` copies the
report to `reports/synthetic_demo/`). Synthetic numbers say nothing about real isolates.
The only real-data numbers are the cv-only results on the provisional 5-species release
(see "Real data" below; DATA_CONTRACT v0.5). `runs/synthetic` and
`reports/synthetic_demo/` were regenerated on 2026-10-03 after the review fixes
(DATA_CONTRACT v0.3); regenerate with `make clean-synth demo` after later changes.

- [x] Stage 1 — ingest + harmonize → `labels.parquet`, `label_counts.csv`, `pairs_kept.csv`. S/I/R-only rows need a breakpoint table for exactly their `(standard, standard_year)`. Only `eucast_2024.csv` and `clsi_2024.csv` ship, so real BV-BRC rows from other years and NCBI S/I/R-only rows without a year (the usual NCBI export has no year column) are dropped and counted. **Add per-year tables before ingesting real data.** `fetch --aws-profile NAME` selects an AWS CLI profile for `s3://` sources
- [x] Stage 2 — genomes + QC (`qc.parquet`; Mash species ID via `mash.tsv` or the pure-Python sketch fallback)
- [x] Stage 3 — lineages (Mash single-linkage, 0.005, computed as connected components of the sparse `d <= 0.005` pair graph; no `n x n` matrix) + splits (frozen; PopPUNK backend is a stub). The `n_lolo` (2) largest clusters per species are reserved for LOLO before test selection and always stay train; the R/S repair is NA-safe for `string` columns. A `splits.parquet` built before this change has LOLO lineages on test clusters and must be rebuilt (`--force-splits` or a fresh root), and every result from it is invalid. That rebuild follows from the owner's LOLO decision (a split-logic change); it is not a regeneration to fix a downstream problem, and rule 7 still holds
- [x] Stage 4 — AMRFinderPlus TSV → `known_amr.parquet` (the Snakefile `amrfinder` rule is checked only with stand-ins; the prediction path runs the real AMRFinderPlus 4.2.7, see "Real data")
- [x] Stage 5 — metrics module + unit tests
- [x] Stage 6 — baselines B0 (ResFinder pheno tables) – B2
- [x] Stage 7 — AFT on known AMR only
- [x] Stage 8 — unitigs: implemented with the pure-Python **k-mer backend** (canonical 31-mers, frequency window, pattern collapse streamed genome by genome, fixed-set query); it refuses > 1,000 training genomes per species (`--unitig-max-kmer-genomes`). The `unitig-caller` wrapper exists but is untested (tool not installed); pyseer selection is replaced by in-fold |correlation| ranking
- [x] Stage 9 — AFT on known AMR + unitigs (`aft_known_unitig`, main model); `aft_unitig_only` ablation behind `--ablation`
- [x] Stage 10 — conformal bands + ranking + report (`results/report.md`, 8 figure types, leakage checklist) + prediction CLI/API bundle
- [x] Stage 11 — external sets (country/time hold-out), LOLO runs and accuracy-by-distance tables/plots on the synthetic data; **not yet run on any real external data**

Since the review fixes (DATA_CONTRACT v0.3):

- `train`: nearest-training distances are computed exactly, block-wise, once per
  species (no `n x n` matrix); species are trained one at a time and released.
  `train --species K... --drugs D...` (also on `run-all`) trains a shard and merges
  `models/manifest.json` / `drop_log_train.csv`. Every test scoring is appended to
  `results/test_ledger.csv`. Preds carry `lab_sir_rederived`.
- Report: EA, exact agreement and band coverage are computed on exact (one-step) lab
  MICs only and printed with that denominator (`n_exact` / `n_band`). Categorical
  metrics are shown as reported and re-derived under the call breakpoint. The leakage
  checklist checks rule 8 against `results/test_ledger.csv` and validates LOLO preds
  rows against `splits.lolo_lineage`. Figures from a synthetic run carry a SYNTHETIC
  DATA watermark and PNG metadata. `make demo-copy` writes
  `reports/synthetic_demo/README.md` and a `synthetic` column in the demo CSVs, and
  refuses non-synthetic roots.
- Workers: `qc`, `lineages` and `unitigs` run per-genome work in worker processes
  (`genome2mic.parallel`; output identical for any core count; `unitigs
  --unitig-threads N`). Scripts calling them from Python need the
  `if __name__ == "__main__":` guard (otherwise they warn and run on one core).
- CLI seeds are independent: `--seed` (synth), `--split-seed`, `--train-seed`
  (argparse dests `synth_seed`, `split_seed`, `train_seed`). `fetch`/`run-all` accept
  `--aws-profile NAME`.
- Prediction: the `markers.fasta` MarkerScan fallback runs only for bundles whose
  `manifest.json` says `synthetic: true`. Real bundles need `amrfinder` on PATH or a
  precomputed AMRFinderPlus TSV per genome; otherwise prediction raises
  `ToolNotAvailable`. The API stays up and reports `/ready` 503 on any bundle load error.
- The carbapenem strong-marker override also fires on any acquired gene with
  AMRFinderPlus Subclass CARBAPENEM (`strong_subclasses` in drugs.yaml). A. baumannii's
  intrinsic OXA-51-like genes are excluded via `configs/intrinsic_markers.csv` (v0.5,
  below); ISAba1 upstream of them is not modelled.
- Test ledger rows carry `inputs_sha1` (data, unitig set, configs and model code), so a
  re-train on the same root after any code or data change makes the leakage check fail
  by design; iterate on a fresh root (`make clean-synth demo`). Preds carry `lab_exact`:
  disk-diffusion results are never exact MICs. Drug bundles record
  `unitig_kmer_set_sha1`; a subset train that would swap a species' k-mer set under
  other drugs refuses to start. CV bands are cross-conformal since v0.4 (fold f's band
  is calibrated on the other folds), so CV coverage is out-of-fold.
- Breakpoints: the project follows the US standard (CLSI M100) for calls and scoring.
  Every table was entered from memory; `docs/BREAKPOINT_VERIFICATION.md` and
  `docs/breakpoint_verification_checklist.csv` list what to verify, in priority order.

Real data (DATA_CONTRACT v0.4 and v0.5, 2026-10-03):

- Root `runs/hackathon5` (gitignored): provisional local release
  `2026-10-04-hackathon+pd5-local` (not an S3 release; its splits are provisional, used as
  given), brought in with `import-release` (SHA256-verified byte copy,
  `IMPORTED_RELEASE.json`). 97 species × drug pairs: KPNEU 29, ECOLI 25, ABAU 17, PAER
  14, SAUR 12. Folds are NCBI SNP clusters. No assemblies, Mash sketches or unitig
  matrix ship, so QC is not assessed, the main model is `aft_known`, and predictions
  have a null `nearest_training_distance` and every call flagged low confidence.
- `train --cv-only`: out-of-fold CV preds plus the final bundle on all train rows; test
  rows are never loaded or scored, no ledger row. The test split stays untouched until
  the user asks for the single test run. Current bundle run `07e65644a16d`.
- Panel caps: each raw log2 prediction is clipped to the fit rows' finite-bound range
  ±1 step before rounding up (stored in `conformal.json`, applied at prediction).
- Call-level metrics (`call_vme_rate` right after `vme_rate`, then `call_me_rate`,
  `active_call_rate_s`, `uncertain_rate`), as reported and re-derived under CLSI 2024.
  Only the 15 CLSI Enterobacterales drugs Hub checked (KPNEU, ECOLI) are verified; every
  other S/I/R and call metric is provisional.
- Band: tuned asymmetric cross-conformal band with the active-call gate (default; see
  "Uncertainty"). Gate open on 69 of 97 bundles. Out of fold, 96 of 97 pairs pass call
  VME ≤ 1.5 % pooled and 90 over calling folds only. Of the 96, 59 make likely-active
  calls, 16 pass only because the gate is closed, 21 by natural resistance or no
  breakpoint (`runs/hackathon5/results/report.md`, call-safety summary). The gate needs
  ≥ 66 lab-R rows in the calling folds, so rare-resistance drugs (SAUR vancomycin,
  daptomycin) never get likely-active calls, by design.
- Exact-MIC rows weigh 2x in the AFT fit (`--exact-weight 2.0`, default); every row
  still trains, nothing is imputed. Not shipped (no gain beyond fold noise): monotone
  constraints and nested xgboost tuning.
- Release feature converter: bundles trained on an imported release are
  `feature_naming: ncbi_release`. Prediction maps AMRFinderPlus output to the release's
  NCBI columns with `predict/release_features.py` and needs
  `models/<SPECIES>/feature_spec.json` (`genome2mic release-feature-spec --root ROOT`;
  `train` writes it when `amrfinder` is on PATH). Parity tables:
  `reports/hackathon_demo/parity_*.csv`.
- AMRFinderPlus 4.2.7 (DB 2026-08-07.1) runs on macOS from a micromamba env:
  `export PATH=$HOME/micromamba/envs/amrfinder/bin:$PATH`. Demo:
  `scripts/hackathon_demo.py` and `reports/hackathon_demo/README.md` (15 genomes absent
  from the splits; it checks that the pipeline runs and does not validate the model; its
  tables were made with the earlier bundle run `1e4dea5e413f`).
- `configs/intrinsic_markers.csv`: 403 ABAU OXA-51-family symbols that never trigger the
  strong-marker override, at prediction or in training-time calls. They stay features.
- Caveats: CV numbers are selection-biased (see "Safety and claims"). Cross-conformal
  tuning is not fully nested (a nested re-check gave the same verdict). Training-time
  calls apply only the column-prefix half of override 2; the pipeline also applies
  `strong_subclasses`.

Tool wrappers untested because the tools are not on PATH: `resfinder`, `mlst`, `mash`,
`unitig-caller`, `pyseer`, `poppunk` (the Snakefile skips them on synthetic data).
`amrfinder` runs from the micromamba env above. Update this list as real data lands. The
Snakefile's `mash`/`amrfinder` command lines are checked with stand-in executables
(`tests/qc/test_snakefile.py`), not the real tools.

## Open questions

- PopPUNK or Mash single-linkage for lineage clusters? Test both on KPNEU.
- Which breakpoint version to standardize on for the final call? Decided 2026-10-03:
  CLSI (US standard), current M100 edition once verified. (Label conversion uses the
  row's own year only; per-year tables are needed.)
- Re-derive all S/I/R from MIC under one standard, or use labels as reported?
  (Interim: labels stay as reported; metrics show both, as-reported first.)
- Is 50 resistant / 50 susceptible the right inclusion bar after seeing real counts?
- Should the cephalosporins also list `strong_subclasses: [CARBAPENEM]`? Today a GES-5
  or OXA-23 isolate is overridden for meropenem but not for ceftriaxone.
- Should `PredictionReport` say which known-AMR backend ran? It would change the
  stage-12 report schema (`extra='forbid'`); contract owner's call.
