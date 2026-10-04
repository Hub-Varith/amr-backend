# Model implementation and training

This guide describes the implementation committed on **`Axion747/gk5s3`**.
It is a training handoff for the current genome-to-MIC model. Streaming sequencing,
partial-genome prediction, and gene autocomplete have not been implemented or
validated in this branch.

**These are predictions of in-vitro susceptibility, not prescribing advice.**
MIC is a laboratory concentration in mg/L, not a patient dose. Dose, route, and
final drug choice depend on PK/PD, infection site, renal function, allergies, and
other patient factors, and remain with the clinician. Confirm with standard AST.

[DATA_CONTRACT.md](DATA_CONTRACT.md) is authoritative for schemas and label rules.

## 1. What is ready to train

The Python training CLI, interval-aware XGBoost models, model selection,
uncertainty bands, probability calibration, bundle writer, prediction pipeline,
and tests are implemented. Install the package and supply a compatible processed
dataset to train a new bundle.

The repository does not contain the local `runs/hackathon5` dataset or its fitted
weights. `runs/`, root-level `models/`, and generated `results/` are gitignored.
The provisional local release has 97 species–drug pairs; a different dataset's
eligible pairs are determined by its `pairs_kept.csv`.

| Key | Species |
| --- | --- |
| `ECOLI` | *Escherichia coli* |
| `KPNEU` | *Klebsiella pneumoniae* |
| `SAUR` | *Staphylococcus aureus* |
| `PAER` | *Pseudomonas aeruginosa* |
| `ABAU` | *Acinetobacter baumannii* |

A new isolate does not need prior lab results or an exact public-genome match.
Prediction uses its measured resistance features and the model learned from
lab-tested isolates. An unsupported species, or a novel resistance mechanism,
does not acquire validated predictions merely by matching a related reference.

## 2. Environment

Use Python 3.11. From this branch's repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m genome2mic train --help
.venv/bin/python -m pytest tests -q
```

Dependencies are declared in `pyproject.toml`. The model uses pandas/pyarrow,
NumPy, SciPy CSR matrices, XGBoost, and scikit-learn. Add the existing `pipeline`
extra (`pip install -e '.[pipeline,test]'`) when running Snakemake or Streamlit.

AMRFinderPlus is an external executable. Real FASTA prediction needs it on `PATH`
or a precomputed AMRFinderPlus TSV for that isolate. For imported NCBI features,
provide the appropriate AMRFinderPlus database during training so that release
feature specifications and subclass mappings are written. The local demo used
AMRFinderPlus 4.2.7 with database 2026-08-07.1; record the versions used in a new run.

## 3. Training inputs

Each run root contains `data/`, `models/`, and `results/`. The processed training
inputs under `<root>/data/processed/` are:

| File | Purpose |
| --- | --- |
| `labels.parquet` | Lab-measured MIC intervals, species, drug, and label provenance |
| `known_amr.parquet` | One row per genome with known-AMR features |
| `known_amr_columns.csv` | Feature-to-marker mapping, when available |
| `qc.parquet` | QC eligibility and assembly statistics |
| `lineages.parquet` | Cluster membership for grouping and evaluation |
| `splits.parquet` | Frozen train/test membership and CV folds |
| `pairs_kept.csv` | Species–drug pairs eligible for training |
| `unitigs_<SPECIES>.*` | Optional sparse unitig matrix, row/index tables, and fixed query set |

`genome_id` is the join key. Missing drug measurements remain missing rows.
Lineage, sequence type, country, year, and source metadata are never predictors.
Feature prefixes are `gene_`, `point_`, `n_class_`, and optionally `u_`.

Unitigs are built only from eligible training genomes. Frequency filtering and
selection run inside each fitting fold; evaluation genomes query the fixed set.
Matrices remain sparse throughout modelling.

### Import an existing processed release

Use a fresh run root and a local release directory supplied by the data owner:

```bash
export G2M_RUN_ROOT=runs/training_release
export G2M_RELEASE_DIR=/absolute/path/to/processed_release
export G2M_AMRFINDER_DB=/absolute/path/to/amrfinder_database

.venv/bin/python -m genome2mic import-release \
  --root "$G2M_RUN_ROOT" --configs-dir configs \
  --release-dir "$G2M_RELEASE_DIR"
```

The release must contain `SHA256SUMS` covering at least `labels.parquet`,
`known_amr.parquet`, `lineages.parquet`, `splits.parquet`, and `pairs_kept.csv`.
The importer verifies checksums, copies the files without changing their contents,
and refuses to replace a different frozen split table.

The current imported-release path does not assess assembly QC and supplies no
unitigs or training-genome sketches. Its generated QC table records null QC
measurements. Unitig models are skipped, and nearest-training distances remain
unavailable. This limitation must accompany results from that path.

For FASTA prediction, also provide species reference FASTAs under
`<root>/data/raw/references/<SPECIES>.fasta` **before training**. Training writes
their reference sketches into the bundle. Accessions and acquisition instructions
are in [the VM runbook](docs/VM_RUNBOOK.md#3-reference-genomes-qc-species-check).

### Build inputs from raw data

Follow [the VM runbook](docs/VM_RUNBOOK.md) for ingest, per-genome external tools,
QC, feature extraction, clustering, frozen splits, and unitigs. Inspect the label
counts, eligible-pair counts, and drop reasons before training.

Do not rerun raw-data stages over an imported release. Do not regenerate an
existing split to repair a downstream failure. The pure-Python k-mer backend
refuses more than 1,000 training genomes per species by default. `unitig-caller`
and several other external wrappers still need validation with real tools;
in-fold correlation selection is currently implemented instead of pyseer's
mixed-model selection.

## 4. Train without touching the test split

Using the variables from the release-import example:

```bash
.venv/bin/python -m genome2mic train \
  --root "$G2M_RUN_ROOT" --configs-dir configs \
  --amrfinder-db "$G2M_AMRFINDER_DB" \
  --cv-only --workers 1 --nthread 4

.venv/bin/python -m genome2mic evaluate --root "$G2M_RUN_ROOT" --configs-dir configs
.venv/bin/python -m genome2mic report --root "$G2M_RUN_ROOT" --configs-dir configs
```

`--cv-only` generates out-of-fold predictions and fits the final bundle on all
train rows. It does not load test-split labels, score test or LOLO rows, or append
to the test ledger. `evaluate` and `report` summarize the generated predictions;
they do not fit or score additional isolates.

To train only the initial KPNEU drugs, add
`--species KPNEU --drugs ceftriaxone meropenem ciprofloxacin` to the train command.
Sharded runs merge into the same root's manifest and preserve other pairs.
Choose `workers × nthread` to fit the available CPU and RAM budget.

Important defaults from `TrainConfig`:

| Setting | Default |
| --- | --- |
| Known-AMR minimum training count | 5 |
| Unitig features retained per fold | 2,000 |
| Training seed | 7 |
| Maximum boosting rounds | 400 |
| Early-stopping patience | 20 rounds |
| Exact-MIC weight relative to other intervals | 2.0 |
| Band | `asym_tuned` |
| AFT/B2/average selection | Enabled |

Use `--no-model-select` to ship the AFT candidate directly. The derived
`aft_b2_select` wrapper is assembled by the trainer; do not list it in `--models`.

**A train command without `--cv-only` also scores the frozen test split.** That
is a final evaluation action after all choices are frozen. Do not use the default
`train`, `run-all`, or `make pipeline` as development shortcuts on real data.
The local hackathon test evaluation has already been performed; its results must
not be used to tune and then re-evaluate that same holdout. New development needs
an independent final evaluation set. Preserve frozen splits and test ledgers.

## 5. How the model works

### MIC labels and AFT loss

Laboratory panels use doubling concentrations. An exact reading of 8 mg/L becomes
`(4, 8]`; `<= 0.25` becomes `(0, 0.25]`; `> 32` becomes `(32, infinity)`.
S/I/R-only labels need the matching standard/year breakpoint table. Missing
measurements are never imputed.

`XgbAft` uses XGBoost `survival:aft` with `aft-nloglik` and normal log-MIC errors.
It minimizes the negative log probability assigned to the observed interval.
Bounds enter in mg/L; XGBoost handles the log transformation internally.
Exact measured MIC rows receive 2× weight, normalized for fitting. Disk-diffusion
categories do not become exact MIC measurements.

Within each fitting fold, a cluster-grouped holdout selects the distribution
scale from `{0.5, 1.0, 1.5}` and the boosting-round count. The booster is then
refitted on all fitting rows. Feature selection sees only those fitting rows.

### Candidate models

| ID | Role |
| --- | --- |
| `b0_resfinder` | External S/R baseline, when phenotype tables exist |
| `b1_lookup` | Median training MIC for a matching known-AMR profile |
| `b2_xgb_steps` | XGBoost classification of exact measured MIC steps |
| `aft_known` | Interval AFT model using known-AMR features |
| `aft_known_unitig` | Interval AFT model using known-AMR plus unitigs |
| `aft_b2_select` | Shipped choice of the main AFT model, B2, or their log2 average |

The main AFT candidate is `aft_known_unitig` when unitigs are available and
`aft_known` otherwise. B2 uses known-AMR features. Selection prioritizes the
call-safety screen, useful likely-active calls, and exact-MIC essential agreement.
A CV fold's candidate choice uses the other folds; the final bundle uses all
training-fold evidence. The average has no separate loss function.

### Bands, calls, and probability

Point predictions are capped to the fitting rows' finite-bound panel range
expanded by one doubling step, then rounded **up** to the doubling grid.
Cross-conformal residuals on exact MICs provide asymmetric uncertainty bands.
The upper level is tuned using training-fold call VME, with pooled calling-fold
and additional per-fold checks. A failed active-call gate withholds likely-active
calls. These are empirical screening rules, not guarantees for future isolates.

| Condition | Call |
| --- | --- |
| Band upper end at or below the susceptible breakpoint | Likely active, if the gate is open |
| Band lower end above the resistant boundary | Likely inactive |
| Otherwise, including a closed gate on an active result | Uncertain |

Natural resistance and configured acquired strong markers force inactivity.
Intrinsic-marker exclusions and token-aware marker matching prevent inappropriate
overrides. Relevant acquired AMRFinderPlus subclasses can also trigger overrides.

Isotonic calibration estimates `P(lab susceptible under the configured breakpoint)`.
The main map is selected from pair-specific, species-pooled, or blended maps using
training-fold evidence. Strong-marker cases use a smoothed empirical rate.
Probability is displayed beside the call and never changes it.

Likely-active drugs are ranked by narrower spectrum, then by their margin below
the susceptible breakpoint. Calls with unavailable nearest-training distance are
flagged low confidence; that flag does not itself remove active calls.

## 6. Outputs and prediction

Training writes:

```text
<root>/models/manifest.json
<root>/models/reference_sketches.npz                 # when references are supplied
<root>/models/<SPECIES>/feature_spec.json            # imported NCBI features + DB
<root>/models/<SPECIES>/train_sketches.npz            # when training sketches exist
<root>/models/<SPECIES>/<drug>/features.json
<root>/models/<SPECIES>/<drug>/params.json
<root>/models/<SPECIES>/<drug>/aft/ and/or b2/         # selected components
<root>/models/<SPECIES>/<drug>/conformal.json
<root>/models/<SPECIES>/<drug>/calibration.json
<root>/models/<SPECIES>/<drug>/meta.json
<root>/results/preds_<SPECIES>_<drug>.parquet
<root>/data/processed/drop_log_train.csv
```

Unitig bundles also contain the species' fixed query set and index. AFT-only
bundles save their booster directly instead of component subdirectories.

With reference sketches and a compatible feature specification available:

```bash
.venv/bin/python -m genome2mic predict \
  --root "$G2M_RUN_ROOT" --configs-dir configs \
  --fasta /absolute/path/to/isolate.fasta --sample-id BC-0142
```

Add `--amrfinder-tsv /absolute/path/to/isolate.amrfinder.tsv` to use a precomputed
report. Imported-release bundles require the stored NCBI naming rules in
`feature_spec.json`. To write a missing specification using the appropriate
database:

```bash
.venv/bin/python -m genome2mic release-feature-spec \
  --root "$G2M_RUN_ROOT" --amrfinder-db "$G2M_AMRFINDER_DB"
```

Resolve a missing specification before prediction.

The FastAPI service reads `G2M_MODELS_DIR` and `G2M_CONFIGS_DIR`. Its readiness
endpoint returns 503 if a usable bundle cannot be loaded.

## 7. Validation and limitations

Report VME first, then call VME, major errors, essential agreement, band coverage,
active-call coverage, and uncertainty. Exact-MIC agreement and band coverage must
include their exact-MIC denominators. When counting pairs that pass call VME,
separate pairs that make active calls from pairs passing only because their gates
are closed, natural resistance applies, or no breakpoint exists.

The current release's groups are NCBI SNP clusters; its evaluation does not prove
generalization to entirely new lineages. Candidate and band comparisons make CV
results selection-biased. Conformal tuning is not fully nested. Breakpoint tables
are provisional outside the verified entries: consult
[the verification checklist](docs/BREAKPOINT_VERIFICATION.md). Common AST metric
targets are evaluation references, not regulatory thresholds demonstrated here.

Current input is an assembled single-isolate FASTA. Undetected trained markers
become zero-valued features. There is no per-locus unobserved state, streaming
read interface, or calibrated partial-genome mode. Missing resistance sequence
can therefore look like absent resistance. Genome autocomplete needs a separate
design and validation; reference similarity alone cannot establish susceptibility.

### Loss-curve diagnostics

`scripts/plot_loss_curves.py` reconstructs the known-AMR AFT component's diagnostic
loss curves using the frozen train rows and saved fitting settings:

```bash
.venv/bin/python scripts/plot_loss_curves.py \
  --root "$G2M_RUN_ROOT" --pair KPNEU:meropenem
.venv/bin/python scripts/plot_loss_curves.py --plot-only
```

It writes PNG, SVG, PDF, loss CSV, metadata, and drop counts under
`.context/charts/loss_curves_clean/` by default. These are reconstructed AFT
diagnostics, not histories recorded during the original training run. The script
does not load test labels or rewrite models or prediction tables. B2 has a
different loss; a selected average has no single training-loss curve.

## 8. Source map

| File/module | Responsibility |
| --- | --- |
| `src/genome2mic/cli.py` | Command entry points and training options |
| `src/genome2mic/ingest/` | Lab-only ingest, harmonization, release import |
| `src/genome2mic/features/` | Known-AMR features, sparse unitigs, in-fold selection |
| `src/genome2mic/models/train.py` | CV orchestration, gates, bundle writing, test ledger |
| `src/genome2mic/models/xgb_aft.py` | Interval likelihood, scale selection, early stopping |
| `src/genome2mic/models/b2_xgb_steps.py` | Exact-MIC classifier |
| `src/genome2mic/models/aft_b2_select.py` | Selected candidate bundle |
| `src/genome2mic/models/conformal.py` | Bands and call-safety checks |
| `src/genome2mic/models/calibration.py` | Calibrated lab-susceptibility probability |
| `src/genome2mic/predict/` | FASTA processing, feature parity, calls, ranking |
| `src/genome2mic/eval/` | Metrics, leakage checks, figures, reports |
| `tests/` | Unit, API, and pipeline tests |
