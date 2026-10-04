# Developer handoff — training data

**For:** anyone (person or AI agent) training or serving models on this repo.
**Branch:** `develop` · **Release:** `2026-10-04-hackathon-all5` (provisional) · **Species with features:** all 5 (KPNEU, ECOLI, SAUR, PAER, ABAU)
**Data owner:** Hub · **Last updated:** 2026-10-04

Read this whole file before writing training code. `DATA_CONTRACT.md` defines every
column; this file says what is actually in the current release, how to get it, and
what to watch for.

These are predictions of in-vitro susceptibility, not prescribing advice.

---

## 0. Quick start (about 15 minutes)

Ask Hub for your own AWS access key first (one per person). Never put it in the repo —
the repo is public.

```bash
# 1. Tools and code
brew install awscli
git clone -b develop https://github.com/Hub-Varith/amr-backend.git
cd amr-backend

# 2. Python env (3.11) and the package
conda create -n genome2mic python=3.11 -y
conda activate genome2mic
pip install -e ".[model,data,test]"

# 3. Your AWS key, saved under the profile "g2m"
aws configure --profile g2m        # key id + secret from Hub, region us-east-1, output json
aws sts get-caller-identity --profile g2m   # should show .../user/<your-name>

# 4. Data
make pull-data                     # -> OK: release 2026-10-04-hackathon-all5 verified in data/processed

# 5. Check everything works (2 epochs, about 1 minute)
python -m genome2mic.models.train_cli --processed-dir data/processed --out-dir /tmp/g2m_smoke \
  --species KPNEU --drugs meropenem --no-unitigs --max-epochs 2
```

## 1. Where the data is

| What | Where |
| ---- | ----- |
| S3 bucket | `s3://g2m-data-v1` (region `us-east-1`) |
| Current release | `s3://g2m-data-v1/releases/2026-10-04-hackathon-all5/` |
| Previous release (KPNEU only) | `s3://g2m-data-v1/releases/2026-10-04-hackathon/` |
| Name of the newest release | `s3://g2m-data-v1/releases/LATEST` (a text file) |
| After `make pull-data` | `data/processed/` in your repo folder |

- Your key can **read** every release and every model, and **write** only to
  `s3://g2m-data-v1/models/<your-aws-user-name>/`.
- A release never changes after upload. New data = new dated release.
- Genomes and raw tool output stay on the data VM; you never need them for training.

## 2. Pulling data

```bash
make pull-data                                 # newest release
make pull-data RELEASE=2026-10-04-hackathon-all5    # a pinned release (use this when comparing models)
aws s3 cp s3://g2m-data-v1/releases/LATEST - --profile g2m   # which release is newest?
```

What `make pull-data` does (`scripts/pull_data.sh`): reads `LATEST`, runs `aws s3 sync`
into `data/processed/`, checks every file against `SHA256SUMS`, and writes the release
name to `data/processed/RELEASE`. Running it again downloads only changed files.

`data/processed/` is gitignored (except `pairs_kept.csv`). **Never commit data.** The
current `splits.parquet` is provisional and is deliberately not in git.

| Error | Fix |
| ----- | --- |
| `cannot read .../LATEST` | Profile missing or wrong key: `aws configure --profile g2m` |
| `AccessDenied` | Your user is not in the IAM group `g2m-engineers` — ask Hub |
| `checksum mismatch` | `rm -rf data/processed`, then `make pull-data` |
| `No rule to make target 'pull-data'` | Old code: `git checkout develop && git pull` |

## 3. Files in the release

| File | Rows | One row per | Use |
| ---- | ---: | ----------- | --- |
| `labels.parquet` | 383,150 | genome × drug | **Targets** (MIC intervals). All 5 species, 48 drugs |
| `known_amr.parquet` | 28,170 | genome | **Features**, all 5 species |
| `known_amr_columns.csv` | 2,999 | feature column | Column → source gene symbol, AMR class, genome count |
| `splits.parquet` | 28,170 | genome | `train` (folds 0–4) / `test`, per species |
| `lineages.parquet` | 28,170 | genome | Cluster IDs. **Splitting/evaluation only, never a feature** |
| `pairs_kept.csv` | 97 | species × drug | Pairs with enough data (KPNEU 29, ECOLI 25, ABAU 17, PAER 14, SAUR 12) |
| `label_counts.csv` | 219 | species × drug | Counts behind `pairs_kept.csv` |
| `RELEASE`, `SHA256SUMS`, `download_manifest.json`, `tool_versions.json` | — | — | Provenance |

`genome_id` (str) is the key in every file. `known_amr`, `splits`, and `lineages` hold
exactly the same 28,170 genome IDs.

Trainable lab results (labels × `pairs_kept` × genomes with features):

| Species | Genomes | Drugs | Train rows | Test rows |
| ------- | ------: | ----: | ---------: | --------: |
| ECOLI | 16,082 | 25 | 158,315 | 28,270 |
| KPNEU | 7,229 | 29 | 72,203 | 13,298 |
| ABAU | 1,206 | 17 | 11,382 | 1,969 |
| SAUR | 1,899 | 12 | 9,847 | 1,614 |
| PAER | 1,754 | 14 | 8,482 | 1,513 |
| **Total** | **28,170** | 37 distinct | **260,229** | **46,664** |

## 4. What each file looks like

### `labels.parquet` — targets

| Column | Type | Notes |
| ------ | ---- | ----- |
| `genome_id` | str | BV-BRC ID (`573.14046`) or `NCBI_<biosample>` |
| `biosample` | str, nullable | |
| `species` | str | `KPNEU`, `ECOLI`, `SAUR`, `PAER`, `ABAU` |
| `drug` | str | lowercase-hyphenated (`piperacillin-tazobactam`) |
| `mic_lower` | float64 | exclusive bound, mg/L. `0.0` = left-censored |
| `mic_upper` | float64 | inclusive bound, mg/L. `inf` = right-censored |
| `censor` | str | `interval` / `left` / `right` |
| `sir` | str, nullable | S/I/R **as reported** by the lab; null when only an MIC was given |
| `raw_result` | str | original reading, e.g. `<=0.125`, `>8`, `=4`, `R`; merged duplicates joined with `|` |
| `method` | str | `dilution` / `gradient` / `disk` |
| `standard`, `standard_year` | str / Int64, nullable | CLSI / EUCAST |
| `source` | str | `BVBRC` / `NCBI` — **not a feature** |
| `isolation_source`, `country`, `year` | nullable | **not features** (evaluation only) |

Real rows (KPNEU × meropenem):

```
 genome_id     drug       mic_lower mic_upper censor sir raw_result  method
 573.14046     meropenem  0.0       1.000     left   S   <=1         dilution
 573.67259     meropenem  8.0       inf       right  R   R           dilution   <- S/I/R-only, via breakpoint
 573.12986     meropenem  0.0       0.125     left   S   <=0.125     dilution
 72407.2374    meropenem  0.0       0.500     left   NaN <=0.5       dilution   <- MIC with no reported S/I/R
```

How to read it: the true MIC is in `(mic_lower, mic_upper]`. `=8` → `(4, 8]`;
`<=1` → `(0, 1]`; `>8` → `(8, inf]`. Every finite bound is an exact doubling step 2^k
(0.015625 … 512), because rounded panel values were snapped (`0.06` → 0.0625).

KPNEU rows that join to `known_amr`: 89,440 over 7,229 genomes. Censoring: 44,631 right,
22,784 left, 22,025 interval. `sir`: 37,152 R, 20,176 S, 3,212 I, 28,900 null.

### `known_amr.parquet` — features

- Columns: `genome_id`, `species`, then 3,028 int8 feature columns.
  - 708 `gene_<family>`: acquired gene present (0/1).
  - 2,291 `point_<gene>_<mutation>`: resistance mutation present (0/1). Mutations are species-specific.
  - 29 `n_class_<class>`: count of hits in that drug class (small non-negative int).
- No nulls. A genome with no hit for a column has 0, including columns only another species has.
- Only **832** columns are present in ≥ 5 genomes of some species. Most `point_` columns are rare.
- KPNEU alone: 952 columns, median 17 gene/point hits per genome.

```
 genome_id species gene_blakpc_2 gene_blactx_m gene_blashv point_gyra_s83i point_parc_s80i n_class_beta_lactam n_class_quinolone
 1284787.3 KPNEU   0             0             1           1               1               3                   5
 1284788.3 KPNEU   1             0             1           1               1               3                   4
```

Naming rules (the prediction side must apply the same ones; see section 9 — code:
`src/genome2mic/features/ncbi_known_amr.py`):

- lowercase; every run of non-alphanumeric characters → one `_` (`aac(6')-Ib` → `gene_aac_6_ib`).
- beta-lactamase alleles collapse to the family (`blaSHV-12` → `gene_blashv`), except the
  families in `configs/keep_variant.csv`: blaKPC, blaNDM, blaVIM, blaIMP, blaGES, and all
  blaOXA (`blaKPC-2` → `gene_blakpc_2`, `blaOXA-48` → `gene_blaoxa_48`).
- non-bla genes keep their exact symbol; `=MISTRANSLATION` calls are skipped; partial
  calls count as present.
- `n_class_*` comes from the AMRFinderPlus database class; a multi-class entry
  (`AMINOGLYCOSIDE/QUINOLONE`) counts once in each class. Some genes have no class (e.g.
  `fosA`), so `n_class_*` can undercount; the `gene_` column is still there.

Sanity check: meropenem R share is 0.89 with any `gene_blakpc_*` vs 0.27 without;
ceftriaxone R share is 0.87 with `gene_blactx_m` vs 0.64 without.

### `splits.parquet`

| Column | Type | Values |
| ------ | ---- | ------ |
| `genome_id`, `species` | str | |
| `split` | str | `train` (6,141) / `test` (1,088 = 15.1%) |
| `fold` | Int64 | 0–4 for train (1,228–1,229 each); null for test |
| `external_set`, `lolo_lineage` | null | not used in this release |

### `lineages.parquet`

`genome_id`, `species`, `lineage_cluster` (e.g. `KPNEU_PDS000045272`, or
`KPNEU_SOLO_<biosample>` for genomes in no cluster), `st` (null), `cluster_method`
(`ncbi_snp_cluster`). 19,162 clusters over 5 species; 13,310 genomes are SOLO. Every
cluster is species-prefixed. No cluster spans train/test, two folds, or two species
(checked when built).

## 5. Training the existing shared model

The shared multi-drug model (`multitask_aft`, see `MODEL_DESIGN.md`) reads
`data/processed/` directly.

```bash
python -m genome2mic.models.train_cli --processed-dir data/processed --out-dir models/multitask \
  --no-unitigs                       # all 5 species; add --species KPNEU for one
```

- `--no-unitigs` is **required**: this release has no unitig matrix.
- `--drugs a b c` restricts the drugs; `--max-epochs N` (default 200) shortens a run.
- Output in `--out-dir`: `model.pt`, `spec.json`, `conformal.json`, `preds_oof.parquet`
  (out-of-fold predictions), `history.parquet`, `label_counts.csv`.
- `models/` is not a data release. Share runs through S3 (section 8), not git.

## 6. Building your own training table

```python
import numpy as np, pandas as pd

labels = pd.read_parquet("data/processed/labels.parquet")
features = pd.read_parquet("data/processed/known_amr.parquet")
splits = pd.read_parquet("data/processed/splits.parquet")
pairs = pd.read_csv("data/processed/pairs_kept.csv")

labels = labels.merge(pairs[["species", "drug"]], on=["species", "drug"])   # 97 species x drug pairs
data = labels.merge(features, on=["genome_id", "species"]).merge(splits[["genome_id", "split", "fold"]], on="genome_id")
feature_columns = [c for c in features.columns if c not in ("genome_id", "species")]

lower = np.log2(data["mic_lower"])          # 0 -> -inf (left-censored): fine for AFT / censored losses
upper = np.log2(data["mic_upper"])          # inf stays inf (right-censored)
train = data[data["split"] == "train"]       # tune and calibrate with `fold`
test = data[data["split"] == "test"]         # score ONCE, at the very end
```

Inner joins are correct: labels for genomes without NCBI AMR results (3–17% per species)
drop out; nothing is imputed.

## 7. Rules (break one and the results are invalid)

1. **Never impute a label.** No row = no result.
2. **Never use as a feature:** `lineage_cluster`, `st`, `country`, `year`, `source`,
   `isolation_source`, `biosample`, `method`, `standard`, `standard_year`, `raw_result`, `sir`.
3. **Feature selection inside each fold.** The rare-column filter (≥ 5 training genomes)
   and any correlation screen run on the training folds only, never on all rows.
4. **Test set is touched once**, at the end. Tuning, thresholds, and conformal
   calibration use the train folds.
5. **Round MIC predictions up** to the next doubling step (a high prediction is the safer error).
6. **Report VME first** (predicted S, lab R). Targets: VME ≤ 1.5%, ME ≤ 3%, EA ≥ 90% — these
   are common device-evaluation figures, not thresholds we have met.
7. **Write the release name next to every result** (`release=2026-10-04-hackathon-all5`).
8. **Never edit files in `data/processed/` by hand.** If the data looks wrong, tell Hub.

## 8. Sharing trained models

```bash
aws s3 cp --recursive --profile g2m models/multitask/ s3://g2m-data-v1/models/<your-aws-user-name>/<run-name>/
aws s3 ls --profile g2m s3://g2m-data-v1/models/            # everyone's runs
```

Put a `README.txt` in each run folder: release name, git commit, command line, and the
headline metrics (VME first).

## 9. Known limits of this release (say them in any demo)

1. **Test scores are optimistic.** Splits come from NCBI SNP clusters: near-identical
   isolates stay together, but a lineage such as ST258 can sit in train and test.
2. **Features come from NCBI's AMRFinderPlus runs** (mixed versions), not our own run.
3. **Some labelled genomes are missing** (no NCBI AMR result): KPNEU 9%, ECOLI 10%, SAUR 17%,
   PAER 3%, ABAU 13%.
4. **Breakpoints are checked for 15 KPNEU drugs only** (`configs/breakpoints/README.md`).
   Calls for other species and drugs are provisional or `uncertain` (no breakpoint).
5. **Thin drugs:** tigecycline, colistin, minocycline, chloramphenicol, polymyxin-b,
   ceftazidime-avibactam, ceftolozane-tazobactam have few R or S in test — noisy scores.
6. **Ampicillin:** KPNEU is naturally resistant; exclude it from ranking.
7. **Each species has its own test set** (15–20% of its genomes). Report results per species;
   ECOLI has twice as many rows as all others together and will dominate pooled numbers.

Background on the shortcut: `docs/HACKATHON_DATA.md`.

## 10. Prediction side (API / demo)

The trained model is `models/all5_run1/` (run `4d85f96f288e`). Get it from S3:

```bash
aws s3 cp --recursive --profile g2m s3://g2m-data-v1/models/hub/all5_run1/ models/all5_run1/
```

`models/` is not in git. `models/all5_run1/README.txt` lists the files and the scores.

### What the pipeline must give the model

1. **Species key** (`ECOLI`, `KPNEU`, `SAUR`, `PAER`, `ABAU`). Identify it first (Mash against the 5
   references, as in stage 3 QC). Any other species: do not predict; set `species` null and
   `in_range` false.
2. **Known-AMR values** as `{column_name: value}`: `gene_*` and `point_*` are 0/1, `n_class_*` are
   counts. Build them from AMRFinderPlus output (`amrfinder -n <fasta> -O <amrfinder_organism> --plus`,
   organism names in `configs/species.yaml`) with the naming rules in section 4
   (`NcbiKnownAmrBuilder.column_name` and `family_for` are the reference). Columns the model does not
   know are ignored; columns you do not send count as 0. `spec.json` → `known_columns` is the full list.
3. AMRFinderPlus runs on Linux, not on a Mac.

### From inputs to one report row per drug

```python
import json
from pathlib import Path
import numpy as np
from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.models.mic_predictor import MicPredictor
from genome2mic.models.model_artifact import ModelArtifact
from genome2mic.predict.call_thresholds import CallThresholds
from genome2mic.predict.probability_calibrator import ProbabilityCalibrator
from genome2mic.predict.susceptibility_caller import SusceptibilityCaller

run_dir = Path("models/all5_run1")
artifact = ModelArtifact.load(run_dir)                       # load once at startup
predictor = MicPredictor(artifact)
calibrator = ProbabilityCalibrator.from_dict(json.loads((run_dir / "probability_calibration.json").read_text()))
thresholds = CallThresholds.from_dict(json.loads((run_dir / "call_thresholds.json").read_text()))
caller = SusceptibilityCaller(BreakpointTable.from_directory(Path("configs/breakpoints")), thresholds=thresholds)
sigma_by_drug = dict(zip(artifact.drugs, np.exp(artifact.model.log_sigma.detach().numpy())))

# species: str, known_row: dict[str, int] from steps 1-2
predictions = predictor.predict(species, predictor.known_vector(known_row).reshape(1, -1), None)
for row in predictions.itertuples():                         # one row per drug of that species
    p_active = None
    found = caller.breakpoints.lookup(species, row.drug, caller.standard, caller.year)
    if found is not None and calibrator.has_pair(species, row.drug):
        p_raw = SusceptibilityCaller.probability_active(row.mu_log2, sigma_by_drug[row.drug], found[0])
        p_active = float(calibrator.calibrate(species, row.drug, np.array([p_raw]))[0])
    call = caller.call(species, row.drug, row.band_low, row.band_high, p_active)
    # -> DrugPrediction(drug=row.drug, pred_mic=row.pred_mic, band_low=row.band_low, band_high=row.band_high,
    #      s_breakpoint=call.s_breakpoint, r_breakpoint=call.r_breakpoint, p_active=call.p_active,
    #      confidence_level=call.confidence_level, call=call.call, margin_steps=call.margin_steps, ...)
```

This code was run against `models/all5_run1` on 2026-10-03. `p_active` and `confidence_level` are null
when the pair has no breakpoint or no calibration curve (e.g. SAUR vancomycin: almost no resistant
genomes); the call then uses the band rule.

### Prediction pipeline (built on branch `genome-data-pipeline`)

FASTA -> QC -> species -> AMRFinderPlus -> known-AMR row -> this model -> overrides -> report is in
`src/genome2mic/predict/pipeline.py`; see `docs/PREDICTION_PIPELINE.md`. Still open: the novelty
check needs a Mash sketch of the training genomes, which the releases do not ship.

- **Demo without AMRFinderPlus:** take the rows of a few `test` genomes straight from
  `known_amr.parquet` (drop `genome_id` and `species`, `.to_dict()`) as `known_row`.

## 11. Where to read more

| Doc | What |
| --- | ---- |
| `DATA_CONTRACT.md` | Every column, the interval rule, the leakage checklist |
| `MODEL_DESIGN.md` | The shared multi-drug model |
| `docs/HACKATHON_DATA.md` | Why this release is provisional |
| `docs/ENGINEER_SETUP.md` | AWS key setup in more detail |
| `docs/DATA_PIPELINE_PLAN.md` | How the data is built, and what comes next |

## 12. Next releases

- New releases are announced by name in the team chat; check with
  `aws s3 cp s3://g2m-data-v1/releases/LATEST - --profile g2m`, then `make pull-data`.
- Planned: own AMRFinderPlus run, Mash/PopPUNK lineages, frozen splits committed to git,
  unitig matrix, more species.
