# Handoff to the model agent — training data

**From:** data pipeline (workspace `naypyidaw`, branch `Hub-Varith/training-data-pipeline-plan`)
**Date:** 2026-10-04 · **Release:** `2026-10-04-hackathon` (provisional) · **Species with features:** KPNEU only

Read this whole file before writing training code. `DATA_CONTRACT.md` defines the
columns; this file says what is actually in the release and what to watch for.

These are predictions of in-vitro susceptibility, not prescribing advice.

---

## 1. Where the data is

| What | Where |
| ---- | ----- |
| S3 bucket | `s3://g2m-data-v1` (region `us-east-1`) |
| This release | `s3://g2m-data-v1/releases/2026-10-04-hackathon/` |
| Newest release name | `s3://g2m-data-v1/releases/LATEST` (text file; now `2026-10-04-hackathon`) |
| After pulling | `data/processed/` in your working folder (gitignored) |

Access is read-only through the AWS CLI profile `g2m` (already set up on Hub's Mac in
`~/.aws/credentials`). You may write only to `s3://g2m-data-v1/models/<aws-user-name>/`.

## 2. How to pull it

Your branch (`Hub-Varith/multi-drug-mic-model-design`, workspace `niamey`) does not have
the pull script yet. Run the data workspace's script from inside your folder:

```bash
cd /Users/hubvarith/conductor/workspaces/amr_backend/niamey
AWS_PROFILE=g2m bash ../naypyidaw/scripts/pull_data.sh
```

Success looks like:

```
OK: release 2026-10-04-hackathon verified in data/processed
release=2026-10-04-hackathon
```

The script downloads with `aws s3 sync`, checks every file against `SHA256SUMS`, and
writes `data/processed/RELEASE`. Pin a release with `RELEASE=<name>`. Do not merge the
data branch into the model branch tonight; only the files are needed.

| Error | Fix |
| ----- | --- |
| `cannot read .../LATEST` | AWS profile missing: `aws configure --profile g2m` |
| `checksum mismatch` | `rm -rf data/processed` and pull again |

## 3. Files in the release

| File | Rows | One row per | Use |
| ---- | ---: | ----------- | --- |
| `labels.parquet` | 383,150 | genome × drug | **Targets** (MIC intervals). All 5 species, 48 drugs |
| `known_amr.parquet` | 7,229 | genome | **Features** (KPNEU only) |
| `known_amr_columns.csv` | 952 | feature column | Column → source gene symbol, AMR class, genome count |
| `splits.parquet` | 7,229 | genome | `train` (folds 0–4) / `test` |
| `lineages.parquet` | 7,229 | genome | Cluster IDs. **Evaluation/splitting only, never a feature** |
| `pairs_kept.csv` | 97 | species × drug | Pairs with enough data (29 for KPNEU) |
| `label_counts.csv` | 219 | species × drug | Counts behind `pairs_kept.csv` |
| `RELEASE`, `SHA256SUMS`, `download_manifest.json`, `tool_versions.json` | — | — | Provenance |

`genome_id` (str) is the key in every file. `known_amr`, `splits`, and `lineages` hold
exactly the same 7,229 genome IDs.

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

Meaning: the true MIC is in `(mic_lower, mic_upper]`. `=8` → `(4, 8]`; `<=1` → `(0, 1]`;
`>8` → `(8, inf]`. Every finite bound is an exact doubling step 2^k (0.015625 … 512),
because rounded panel values were snapped (`0.06` → 0.0625).

KPNEU rows that join to `known_amr`: 89,440 over 7,229 genomes. Censoring: 44,631 right,
22,784 left, 22,025 interval. `sir`: 37,152 R, 20,176 S, 3,212 I, 28,900 null.

### `known_amr.parquet` — features

- Columns: `genome_id`, `species`, then 952 int8 feature columns.
  - 319 `gene_<family>`: acquired gene present (0/1).
  - 615 `point_<gene>_<mutation>`: resistance mutation present (0/1).
  - 18 `n_class_<class>`: count of hits in that drug class (small non-negative int).
- No nulls. A genome with no hit for a column has 0.
- Only **307** columns are present in ≥ 5 genomes. Most `point_` columns are rare.
- Median 17 gene/point hits per genome.

```
 genome_id species gene_blakpc_2 gene_blactx_m gene_blashv point_gyra_s83i point_parc_s80i n_class_beta_lactam n_class_quinolone
 1284787.3 KPNEU   0             0             1           1               1               3                   5
 1284788.3 KPNEU   1             0             1           1               1               3                   4
```

Naming rules (the prediction side must apply the same ones; see section 7):

- lowercase; every run of non-alphanumeric characters → one `_` (`aac(6')-Ib` → `gene_aac_6_ib`).
- beta-lactamase alleles collapse to the family (`blaSHV-12` → `gene_blashv`), except the
  families in `configs/keep_variant.csv`: blaKPC, blaNDM, blaVIM, blaIMP, blaGES, and all
  blaOXA (`blaKPC-2` → `gene_blakpc_2`, `blaOXA-48` → `gene_blaoxa_48`).
- non-bla genes keep their exact symbol; `=MISTRANSLATION` calls are skipped; partial
  calls count as present.
- `n_class_*` comes from the AMRFinderPlus database class; a multi-class entry
  (`AMINOGLYCOSIDE/QUINOLONE`) counts once in each class. Some genes have no class (e.g.
  `fosA`), so `n_class_*` can undercount; the `gene_` column is still there.

Sanity check done: meropenem R share is 0.89 with any `gene_blakpc_*` vs 0.27 without;
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
(`ncbi_snp_cluster`). 3,573 clusters; 4,950 genomes are in a multi-genome cluster.
No cluster spans train/test or two folds (checked when built).

## 5. How to build the training table

```python
import numpy as np, pandas as pd

labels = pd.read_parquet("data/processed/labels.parquet")
features = pd.read_parquet("data/processed/known_amr.parquet")
splits = pd.read_parquet("data/processed/splits.parquet")
pairs = pd.read_csv("data/processed/pairs_kept.csv")

labels = labels.merge(pairs[["species", "drug"]], on=["species", "drug"])   # 29 KPNEU drugs
data = labels.merge(features, on=["genome_id", "species"]).merge(splits[["genome_id", "split", "fold"]], on="genome_id")
feature_columns = [c for c in features.columns if c not in ("genome_id", "species")]

lower = np.log2(data["mic_lower"])          # 0 -> -inf (left-censored): fine for AFT / censored losses
upper = np.log2(data["mic_upper"])          # inf stays inf (right-censored)
train = data[data["split"] == "train"]       # tune and calibrate with `fold`
test = data[data["split"] == "test"]         # score ONCE, at the very end
```

Inner joins are correct: labels without features (other species, and 10% of KPNEU) drop
out; nothing is imputed.

## 6. Rules (break one and the results are invalid)

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

## 7. Known limits of this release (say them in the demo)

1. **Test scores are optimistic.** Splits come from NCBI SNP clusters: near-identical
   isolates stay together, but a lineage such as ST258 can sit in train and test.
2. **Features come from NCBI's AMRFinderPlus runs** (mixed versions), not our own run.
3. **10% of labelled KPNEU genomes are missing** (no NCBI AMR result).
4. **Breakpoints are unverified** (`configs/breakpoints/README.md`; only ceftriaxone,
   meropenem, ciprofloxacin exist for KPNEU/ECOLI). S/I/R-only labels for those three drugs
   used them; any S/I/R call made from a predicted MIC is provisional.
5. **Thin drugs:** tigecycline, colistin, minocycline, chloramphenicol, polymyxin-b,
   ceftazidime-avibactam, ceftolozane-tazobactam have few R or S in test — noisy scores.
6. **Ampicillin:** KPNEU is naturally resistant; exclude it from ranking.
7. Only KPNEU has features. The other species' labels are in `labels.parquet` but cannot
   train until their features exist.

## 8. Prediction side (demo)

- A new genome needs AMRFinderPlus (`amrfinder -n <fasta> -O Klebsiella_pneumoniae --plus`),
  then conversion of its output to **exactly** these 952 columns with the naming rules in
  section 4: unknown genes ignored, missing columns 0, same order. That converter does not
  exist yet; use `known_amr_columns.csv` as the column list.
- AMRFinderPlus is installed only on the data VM (Linux), not on the Mac.
- **For tonight's demo:** take rows for a few `test` genomes straight from
  `known_amr.parquet` and feed them to the model — no AMRFinderPlus needed.

## 9. Contacts and next releases

- Data questions: Hub. Do not edit files in `data/processed/` by hand.
- New releases are announced by name; check with
  `aws s3 cp s3://g2m-data-v1/releases/LATEST - --profile g2m`.
- Planned (after the hackathon): own AMRFinderPlus run, Mash/PopPUNK lineages, frozen
  splits committed to git, unitig matrix, more species.
