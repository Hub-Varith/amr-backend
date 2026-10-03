# Genome-to-MIC — Data Contract

**Status:** draft v0.1 · **Owner:** Hub · **Last updated:** 2026-10-03

This is the agreement between the data team and the model team. If you change a
column name, a type, or a meaning, update this file first and tell the team.

---

## 0. What we are building

In: one bacterial genome (FASTA).
Out: for each antibiotic, a predicted MIC with an uncertainty band, turned into
"likely active / uncertain / likely inactive", then ranked.

**MIC** = minimum inhibitory concentration: the lowest drug amount that stops the
germ growing in the lab, in mg/L. It is a lab measurement of the germ, **not** a
patient dose.

**Scope:** 5 species, bloodstream infection drugs.

| Species key   | Full name                  | AMRFinderPlus `-O` |
| ------------- | -------------------------- | ------------------ |
| `ECOLI`       | Escherichia coli           | `Escherichia`      |
| `KPNEU`       | Klebsiella pneumoniae      | `Klebsiella_pneumoniae` |
| `SAUR`        | Staphylococcus aureus      | `Staphylococcus_aureus` |
| `PAER`        | Pseudomonas aeruginosa     | `Pseudomonas_aeruginosa` |
| `ABAU`        | Acinetobacter baumannii    | `Acinetobacter_baumannii` |

Run `amrfinder -l` to confirm the exact `-O` strings for the installed version.

---

## 1. Ownership and handoff

| Stage | Output file | Owner |
| ----- | ----------- | ----- |
| 1. Raw AST download | `data/raw/ast_<source>.csv` | Data team |
| 2. Clean labels | `data/processed/labels.parquet` | Data team |
| 3. Genomes + QC | `data/raw/genomes/`, `data/processed/qc.parquet` | Data team |
| 4. Per-genome tool output | `data/interim/<genome_id>/` | Data team |
| 5. Known-AMR features | `data/processed/known_amr.parquet` | Data team |
| 6. Lineage clusters | `data/processed/lineages.parquet` | Data team |
| 7. Splits | `data/processed/splits.parquet` | Data team |
| 8. Unitig matrix | `data/processed/unitigs_<SPECIES>.npz` + index | Data team |
| 9. Training join | in-memory | Model team |
| 10. Predictions | `results/preds_<species>_<drug>.parquet` | Model team |
| 11. Metrics | `results/metrics.parquet` | Model team |
| 12. Report | app output | Model team |

**The handoff is stages 2, 5, 6, 7, 8.** Those five files are the contract. The
model team should never need to open a raw download or a tool output.

### Rules that cannot be broken

1. **No label imputation.** A missing lab result is a missing row. Never fill it in.
2. **No lineage, country, year, or source as a feature.** They exist for splitting
   and evaluation only. Keeping them out is what makes our "unseen strain" claim
   real.
3. **Unitigs are built on training genomes only.** Other genomes are *queried*
   against that fixed set. Building on everything leaks test information.
4. **De-duplicate by `biosample` across sources** before splitting. BV-BRC and NCBI
   share many isolates; the same genome in train and test invalidates results.
5. **Keep lab-measured results only.** BV-BRC also stores its own ML predictions.
   Training on those teaches us to copy another model.

### Global conventions

- All files are **Parquet** except raw downloads (CSV) and the unitig matrix (NPZ).
- `genome_id` is our primary key everywhere. String. Use the BV-BRC genome ID
  format (`573.2002`) when available, else `NCBI_<biosample>`.
- Drug names are lowercase, hyphenated, no spaces: `piperacillin-tazobactam`.
- MIC units are always **mg/L**. Convert µg/mL 1:1 (they are the same).
- `species` is always the 5-letter key from the table above.
- Missing value = null, never `-1`, `""`, `NA`, or `0`.

---

## 2. Stage schemas

### Stage 1 — Raw AST download

`data/raw/ast_bvbrc.csv`, `data/raw/ast_ncbi.csv`

Keep these as downloaded. Do not edit. They exist so we can rebuild stage 2.

Also in `data/raw/`: `meta_bvbrc.csv` (BV-BRC genome metadata), `meta_ncbi.csv`
(BioSample attributes), `ncbi_biosample/*.xml.gz` (the NCBI BioSample XML that
`ast_ncbi.csv` is flattened from), and `download_manifest.json` (date and counts).
Built by `make download-ast`.

Expected BV-BRC columns (names vary slightly by export):

| Column | Type | Note |
| ------ | ---- | ---- |
| `genome_id` | str | |
| `genome_name` | str | |
| `antibiotic` | str | Raw name, un-normalized |
| `resistant_phenotype` | str | `Resistant` / `Susceptible` / `Intermediate` |
| `measurement_sign` | str | `=`, `<=`, `>`, `>=`, `<` — may be empty |
| `measurement_value` | str | May be empty |
| `measurement_unit` | str | Usually `mg/L` |
| `laboratory_typing_method` | str | Broth dilution, Disk diffusion, etc. |
| `testing_standard` | str | `CLSI` or `EUCAST` |
| `testing_standard_year` | int | May be empty |
| `evidence` | str | Keep `Laboratory Method` rows only |

Sample:

```csv
genome_id,antibiotic,resistant_phenotype,measurement_sign,measurement_value,measurement_unit,laboratory_typing_method,testing_standard,testing_standard_year,evidence
573.2002,meropenem,Resistant,=,8,mg/L,Broth dilution,CLSI,2016,Laboratory Method
573.2005,meropenem,Resistant,>,32,mg/L,Broth dilution,CLSI,2019,Laboratory Method
573.2011,meropenem,Susceptible,,,,Disk diffusion,EUCAST,2021,Laboratory Method
573.2002,ciprofloxacin,Resistant,=,1,mg/L,Broth dilution,CLSI,2016,Laboratory Method
```

---

### Stage 2 — Clean labels ★ CONTRACT

`data/processed/labels.parquet` — **long format, one row per genome × drug.**

| Column | Type | Null? | Meaning |
| ------ | ---- | ----- | ------- |
| `genome_id` | str | no | Primary key part 1 |
| `biosample` | str | yes | For cross-source de-duplication |
| `species` | str | no | 5-letter key |
| `drug` | str | no | Primary key part 2, normalized |
| `mic_lower` | float | no | Exclusive lower bound, mg/L. `0.0` = left-censored |
| `mic_upper` | float | no | Inclusive upper bound, mg/L. `inf` = right-censored |
| `censor` | str | no | `interval` / `left` / `right` |
| `sir` | str | yes | `S` / `I` / `R` as reported |
| `raw_result` | str | no | Original string, for audit |
| `method` | str | no | `dilution` / `gradient` / `disk` |
| `standard` | str | yes | `CLSI` / `EUCAST` |
| `standard_year` | int | yes | |
| `source` | str | no | `BVBRC` / `NCBI` |
| `isolation_source` | str | yes | `blood`, `urine`, … — for subgroup reporting |
| `country` | str | yes | Evaluation only |
| `year` | int | yes | Evaluation only |

**Primary key:** (`genome_id`, `drug`). Must be unique.

#### The interval rule

MIC panels use doubling steps (…0.25, 0.5, 1, 2, 4, 8…). A reading of `8` means
the true MIC is above 4 and at most 8. Every result becomes one interval
`(mic_lower, mic_upper]`:

| Raw result | `mic_lower` | `mic_upper` | `censor` |
| ---------- | ----------- | ----------- | -------- |
| `= 8` | 4 | 8 | `interval` |
| `<= 0.25` | 0 | 0.25 | `left` |
| `< 0.5` | 0 | 0.5 | `left` — read as `<=`: wider, never wrong |
| `> 32` | 32 | inf | `right` |
| `>= 16` | 8 | inf | `right` |
| `S` only, breakpoint S ≤ 1 | 0 | 1 | `left` |
| `R` only, breakpoint R > 2 | 2 | inf | `right` |
| `I` only, S ≤ 1 and R > 2 | 1 | 2 | `interval` |

For `S`/`I`/`R`-only rows you need the breakpoint table matching that row's
`standard` and `standard_year`. **If the standard is unknown, drop the row.** Do
not guess. If the year is unknown, use the breakpoint only when every version of
that standard agrees for the species × drug; otherwise drop the row. Breakpoint
files and their one shared form are described in `configs/breakpoints/README.md`.

**Snapping to the doubling grid** (agreed 2026-10-03). Panels print rounded values
(`0.06` for 0.0625, `0.12` for 0.125). A reported value within 5% of a doubling step
2^k becomes that step. A value between steps (Etest half-steps such as `0.75`, `1.5`,
`3`) is dropped with reason `off_grid_value`. `raw_result` keeps the printed value.

**Combination drugs** report `8/4`. The first number is the MIC.

**Phenotype mapping:** `Susceptible` → `S`, `Intermediate` and
`Susceptible-dose dependent` → `I`, `Resistant` → `R`. `Nonsusceptible` and
`not defined` → null (an MIC row keeps its interval; an S/I/R-only row is dropped).

Why intervals: it lets exact MICs, "≤", ">", and S/R-only labels all train the
same model with no made-up numbers at the panel edges. The model loss
(`survival:aft`) consumes exactly these two columns.

#### Method filter

| `laboratory_typing_method` contains | `method` | MIC usable? |
| ----------------------------------- | -------- | ----------- |
| broth, microdilution, agar dilution, `MIC` | `dilution` | yes |
| Etest, gradient, MIC strip | `gradient` | yes |
| disk, Kirby-Bauer, zone | `disk` | **no — S/I/R only** |
| empty or `missing` | — | row dropped (`unknown_method`) |

Disk diffusion measures a zone diameter, not an MIC. Those rows still enter as
censored intervals via the S/I/R path. A disk row with no S/I/R is dropped, even
when it carries a number in mg/L (about 33k BV-BRC rows do; the method label and the
unit disagree, so neither is trusted).

An MIC row whose unit is not mg/L (or µg/mL) is dropped (`bad_unit`).

#### Duplicate resolution

- Same (`genome_id`, `drug`), intervals within 1 doubling step → take the
  intersection; if empty, keep the higher (safer) one.
- More than 1 step apart, or S vs R → **drop that genome × drug pair** and log it.
- "Steps apart" for intervals that do not overlap = log2(higher `mic_lower` / lower
  `mic_upper`) + 1. So `=4` vs `=8` is 1 step (keep `=8`); `<=0.25` vs `=1` is 2 steps
  (drop).
- A merged row keeps the most resistant `sir` (R > I > S) and joins every original
  reading in `raw_result` with `|` (e.g. `=32|>32`).

#### Genome IDs across sources

- One genome per `biosample`. When BV-BRC holds several assemblies of one biosample,
  the one with the fewest contigs (then the lowest ID) becomes the `genome_id`, and
  every label from that biosample moves to it.
- An NCBI result whose biosample has a BV-BRC genome takes that BV-BRC `genome_id`.
  Otherwise its ID is `NCBI_<biosample>`.
- BV-BRC rows are selected by genome name prefix (taxon ID alone misses subspecies
  taxa such as 72407), then the species is confirmed from BV-BRC genome metadata.

#### Other stage 2 outputs

- `data/processed/dropped_labels.parquet`: one row per dropped record, with
  `source, record_id, species, drug, antibiotic, reason`.
- `data/processed/label_counts.csv`: the counts table below, plus `n_I` and `kept`.
- `isolation_source` is grouped into `blood`, `urine`, `respiratory`, `wound`, `gut`,
  `other`; null when not reported.

Sample:

```csv
genome_id,biosample,species,drug,mic_lower,mic_upper,censor,sir,raw_result,method,standard,standard_year,source,isolation_source,country,year
573.2002,SAMN00000001,KPNEU,meropenem,4.0,8.0,interval,R,=8,dilution,CLSI,2016,BVBRC,blood,USA,2016
573.2005,SAMN00000002,KPNEU,meropenem,32.0,inf,right,R,>32,dilution,CLSI,2019,NCBI,blood,India,2019
573.2010,SAMN00000003,KPNEU,meropenem,0.0,0.25,left,S,<=0.25,dilution,CLSI,2020,BVBRC,urine,Thailand,2020
573.2011,SAMN00000004,KPNEU,meropenem,0.0,2.0,left,S,S,disk,EUCAST,2021,NCBI,blood,UK,2021
573.2002,SAMN00000001,KPNEU,ciprofloxacin,0.5,1.0,interval,R,=1,dilution,CLSI,2016,BVBRC,blood,USA,2016
```

#### Acceptance checks

- [ ] (`genome_id`, `drug`) unique
- [ ] `mic_lower < mic_upper` on every row
- [ ] `censor == 'left'` ⟺ `mic_lower == 0`
- [ ] `censor == 'right'` ⟺ `mic_upper == inf`
- [ ] No row where `evidence != 'Laboratory Method'`
- [ ] Count table emitted: species × drug × (n_R, n_S, n_exact, n_censored, n_distinct_mic)

Count definitions: `n_R`, `n_S` use `sir` as reported (rows with null `sir` count for
neither). A row is exact when every reading in `raw_result` starts with `=`.
`n_distinct_mic` = distinct `mic_upper` among exact rows.

**Pair inclusion rule:** keep a species × drug pair only if it has ≥ 50 non-susceptible,
≥ 50 susceptible, and ≥ 4 distinct MIC levels. Below that the model cannot learn
the MIC scale. Emit the list of kept pairs as `data/processed/pairs_kept.csv`.

---

### Stage 3 — Genomes and QC

`data/raw/genomes/<genome_id>.fasta` — assembled contigs, one file per genome.

`data/processed/qc.parquet`:

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `genome_id` | str | |
| `n_contigs` | int | |
| `total_length` | int | bp |
| `n50` | int | |
| `gc_percent` | float | |
| `mash_species` | str | Species key from Mash distance to references |
| `mash_distance` | float | Distance to nearest species reference |
| `qc_pass` | bool | See rules |
| `qc_fail_reason` | str | Null if pass |

**QC fail rules:**

| Rule | Threshold |
| ---- | --------- |
| Too fragmented | `n_contigs > 500` |
| Wrong genome size | Outside ±20% of the species' expected size |
| Species mismatch | `mash_species != species` from the label table |
| Too distant | `mash_distance > 0.05` from every reference |

Genomes failing QC are excluded from training **and** from the unitig build.

---

### Stage 4 — Per-genome tool output

`data/interim/<genome_id>/` — raw tool output, kept for debugging and re-parsing.

```
data/interim/573.2002/
  amrfinder.tsv
  resfinder/pheno_table.txt
  mlst.tsv
  mash.tsv
```

Commands (Snakemake rules):

```bash
amrfinder -n data/raw/genomes/573.2002.fasta \
  -O Klebsiella_pneumoniae --plus \
  -o data/interim/573.2002/amrfinder.tsv

python -m resfinder -ifa data/raw/genomes/573.2002.fasta \
  -o data/interim/573.2002/resfinder \
  -s "Klebsiella pneumoniae" --acquired --point

mlst data/raw/genomes/573.2002.fasta > data/interim/573.2002/mlst.tsv
```

AMRFinderPlus 4.x output (columns we use):

```tsv
Element symbol	Type	Subtype	Class	Subclass	Method	% Coverage of reference	% Identity to reference
blaKPC-2	AMR	AMR	BETA-LACTAM	CARBAPENEM	EXACTX	100.00	100.00
blaSHV-11	AMR	AMR	BETA-LACTAM	BETA-LACTAM	EXACTX	100.00	100.00
ompK36_D135DGD	AMR	POINT	BETA-LACTAM	BETA-LACTAM	POINTX	100.00	99.70
```

Older versions call the first column `Gene symbol`. Handle both.

**ResFinder is a baseline, not a feature source.** Parse `pheno_table.txt` into
stage 11 comparisons only. Do not put ResFinder calls into `known_amr.parquet`.

---

### Stage 5 — Known-AMR features ★ CONTRACT

`data/processed/known_amr.parquet` — **wide, one row per genome.**

| Column pattern | Type | Meaning |
| -------------- | ---- | ------- |
| `genome_id` | str | Primary key |
| `species` | str | 5-letter key |
| `gene_<FAMILY>` | int8 | 0/1, acquired gene family present |
| `point_<GENE>_<MUT>` | int8 | 0/1, resistance mutation present |
| `n_class_<CLASS>` | int8 | Count of AMRFinderPlus hits in that drug class |

**Gene family grouping.** Collapse variants to the family:
`blaCTX-M-15`, `blaCTX-M-27` → `gene_blaCTX_M`.
**Exception — keep the exact variant** where it changes the drug answer. Maintain
this list in `configs/keep_variant.csv`. At minimum: carbapenemases
(`blaKPC`, `blaNDM`, `blaOXA-48`, `blaVIM`, `blaIMP`) keep their variant number.

**Column naming:** lowercase after the prefix, non-alphanumeric → `_`.
`blaCTX-M` → `gene_blactx_m`. Emit the mapping as
`data/processed/known_amr_columns.csv` (`column_name`, `source_symbol`,
`class`, `subclass`, `n_genomes_present`).

**Rare feature filter:** drop columns present in < 5 **training** genomes. Apply
per species. This filter is part of the fold, so emit the unfiltered table and let
the model team filter — or expose a `min_count` parameter.

Sample:

```csv
genome_id,species,gene_blakpc_2,gene_blandm,gene_blaoxa_48,gene_blactx_m,gene_blashv,point_ompk36_d135dgd,point_gyra_s83l,n_class_beta_lactam,n_class_quinolone
573.2002,KPNEU,1,0,0,0,1,1,0,3,0
573.2005,KPNEU,0,1,0,1,1,0,1,4,1
573.2010,KPNEU,0,0,0,0,1,0,0,1,0
573.2011,KPNEU,0,0,0,0,1,0,0,1,0
```

#### Acceptance checks

- [ ] One row per QC-passing genome
- [ ] All feature columns are 0/1 or small non-negative ints
- [ ] No nulls in feature columns
- [ ] Column mapping file emitted and non-empty

---

### Stage 6 — Lineage clusters ★ CONTRACT

`data/processed/lineages.parquet`

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `genome_id` | str | |
| `species` | str | |
| `lineage_cluster` | str | e.g. `KPNEU_PP_12` — **must be species-prefixed** |
| `st` | str | MLST sequence type, or `NA`. Evaluation only |
| `cluster_method` | str | `poppunk` or `mash_single_linkage` |

**Preferred method:** PopPUNK, per species. Fallback: Mash distance + single-linkage
clustering at a threshold that keeps known lineages (ST131, ST258) intact.

**Hard requirement:** near-identical genomes (outbreak isolates, re-submissions of
the same strain) must land in the same cluster. Spot-check by confirming that all
genomes within Mash distance 0.0001 of each other share a cluster.

---

### Stage 7 — Splits ★ CONTRACT

`data/processed/splits.parquet` — frozen once and version-controlled. Changing
splits mid-project invalidates every result produced before the change.

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `genome_id` | str | |
| `species` | str | |
| `split` | str | `train` / `test` |
| `fold` | int | 0–4 for train rows, null for test |
| `external_set` | str | Null, or `country_holdout` / `time_holdout` / `source_holdout` |
| `lolo_lineage` | str | Null, or the lineage name for leave-one-lineage-out runs |

**Construction:**

1. Assign whole `lineage_cluster` values to `test` until ~15–20% of genomes per
   species are held out. Check the test set retains both R and S for each kept drug.
2. `GroupKFold(n_splits=5)` on the remaining clusters → `fold`.
3. Mark external sets separately (a genome can be both `test` and in an external set
   only if the external set is defined on test; prefer disjoint).
4. For the biggest clinical lineages (ECOLI ST131, KPNEU ST258), emit LOLO rows.

**Invariant:** no `lineage_cluster` appears in two different values of `split`, and
no cluster spans two folds.

---

### Stage 8 — Unitig matrix ★ CONTRACT

Per species. Three files:

```
data/processed/unitigs_KPNEU.npz           # scipy sparse CSR, int8, genomes x patterns
data/processed/unitigs_KPNEU_rows.parquet  # row index -> genome_id
data/processed/unitigs_KPNEU_index.parquet # col index -> pattern_id -> unitig sequences
```

A **unitig** is a DNA stretch that appears as one unbroken path in a de Bruijn graph
built across many genomes. One unitig stands in for many overlapping k-mers that
always travel together — same signal, far fewer columns.

**Build procedure (order matters):**

1. Build the unitig set with `unitig-caller --build` on **training genomes of this
   species only** (`split == 'train'` and `qc_pass`).
2. Query every other genome (test, external, and later new isolates) with
   `unitig-caller --query` against that fixed set. Never rebuild.
3. **Frequency filter:** drop unitigs present in < 1% or > 99% of training genomes.
4. **Collapse identical patterns:** many unitigs have the exact same presence/absence
   vector. Keep one column per distinct pattern; record every member sequence in the
   index.

Expected scale after steps 3–4: tens of thousands to a few hundred thousand pattern
columns per species.

`unitigs_<SPECIES>_index.parquet`:

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `col_index` | int | Column number in the NPZ matrix |
| `pattern_id` | str | e.g. `u_000017` |
| `n_unitigs` | int | How many unitigs share this pattern |
| `unitig_sequences` | list[str] | The DNA sequences |
| `train_frequency` | float | Share of training genomes carrying it |

**Not the data team's job:** per-drug feature selection (pyseer). That happens inside
each training fold, model-side, because selecting on all data leaks test information.

---

### Stage 9 — Training join (model side)

Built in memory per species × drug:

```python
y = labels[(labels.species == sp) & (labels.drug == drug)]
X = known_amr.merge(y[["genome_id"]], on="genome_id")
X = hstack([X_known.values, unitigs[rows, selected_cols]])   # sparse
lo = y.mic_lower.values
hi = y.mic_upper.values   # inf allowed
```

Shape example — KPNEU × meropenem:

```
genome_id  gene_blakpc_2  gene_blandm  point_ompk36_d135dgd  u_000017  u_004211  ...  mic_lower  mic_upper
573.2002   1              0            1                     1         0         ...  4.0        8.0
573.2005   0              1            0                     0         1         ...  32.0       inf
573.2010   0              0            0                     0         0         ...  0.0        0.25
573.2011   0              0            0                     1         0         ...  0.0        2.0
```

`species` drops out inside a per-species model. It returns only in a later
multi-species model.

---

### Stage 10 — Predictions

`results/preds_<species>_<drug>.parquet`

| Column | Type | Meaning |
| ------ | ---- | ------- |
| `genome_id` | str | |
| `species`, `drug` | str | |
| `split` | str | Which evaluation set |
| `pred_mic` | float | Point prediction, mg/L, rounded **up** to the next doubling step |
| `band_low`, `band_high` | float | 90% conformal band, mg/L |
| `lab_lower`, `lab_upper` | float | Ground truth interval, copied from stage 2 |
| `pred_sir` | str | `S` / `I` / `R` after applying breakpoints |
| `lab_sir` | str | |
| `model` | str | `b1_lookup` / `b2_xgb_steps` / `aft_known` / `aft_known_unitig` |
| `run_id` | str | Config hash, for reproducibility |

Round **up**: a slightly high MIC prediction is the safer error.

---

### Stage 11 — Metrics

`results/metrics.parquet` — one row per species × drug × model × evaluation set.

| Column | Meaning |
| ------ | ------- |
| `essential_agreement` | Within ±1 doubling step of the lab MIC. Target ≥ 90% |
| `exact_agreement` | Same doubling step |
| `categorical_agreement` | Same S/I/R. Target ≥ 90% |
| `vme_rate` | **Predicted S, lab R.** The dangerous error. Target ≤ 1.5% |
| `me_rate` | Predicted R, lab S. Target ≤ 3% |
| `mine_rate` | One side is I |
| `auroc` | Predicted MIC as a score for R vs S |
| `band_coverage` | Share of lab MICs inside the 90% band. Should be ≈ 0.90 |
| `band_width_steps` | Mean band width in doubling steps |
| `n` | Rows evaluated |

Targets are figures commonly used for AST device evaluation — confirm against current
FDA/ISO guidance before quoting them externally.

**Always report VME first.** It is the error that harms a patient.

---

### Stage 12 — Report (new isolate)

```
Sample: BC-0142        Species: Klebsiella pneumoniae      QC: pass
Nearest training genome: Mash distance 0.004 (in range)

Drug                      Pred. MIC (90% band)   Breakpoint S   Call
Piperacillin-tazobactam   4  (2-8)  mg/L         <= 8           Likely active    margin 0 steps
Meropenem                 0.06 (0.03-0.125)      <= 2           Likely active    margin 4 steps
Gentamicin                2  (1-4)               <= 2           Uncertain - wait for lab
Ceftriaxone               >32                    <= 1           Likely inactive  (blaCTX-M)
Ciprofloxacin             8  (4-16)              <= 0.25        Likely inactive  (gyrA S83L, parC S80I)
Ampicillin                -                      -              Natural resistance

Ranked likely-active: 1. Piperacillin-tazobactam   2. Meropenem
```

#### Report fields (API: `GET /v1/jobs/{job_id}/result`)

`PredictionReport` — one per isolate:

| Field | Type | Null? | Meaning |
| ----- | ---- | ----- | ------- |
| `sample_id` | str | no | |
| `species` | str | yes | 5-letter key. Null = species not covered |
| `qc_pass` | bool | no | |
| `nearest_training_distance` | float | yes | Mash distance to the nearest training genome |
| `in_range` | bool | no | False → all calls low confidence (override 3) |
| `predictions` | list[`DrugPrediction`] | no | |
| `ranked_active` | list[str] | no | Drug names, best first. Each must be a `likely_active` prediction |
| `model_version` | str | no | |
| `run_id` | str | no | Config hash |
| `disclaimer` | str | no | Scope note below. Must not be blank |

`DrugPrediction` — one per drug. MICs in mg/L:

| Field | Type | Null? | Meaning |
| ----- | ---- | ----- | ------- |
| `drug` | str | no | Normalized drug name |
| `pred_mic` | float | yes | Null when an override skips the model |
| `band_low`, `band_high` | float | yes | 90% conformal band. All three MIC fields are null together |
| `s_breakpoint`, `r_breakpoint` | float | yes | Breakpoints used for the call |
| `call` | str | no | `likely_active` / `uncertain` / `likely_inactive` |
| `margin_steps` | int | yes | Doubling steps below the S breakpoint |
| `reasons` | list[str] | no | Markers behind the call, e.g. `gyrA S83L` |
| `override` | str | yes | `natural_resistance` / `strong_marker`. Set → call is `likely_inactive` |

**Call logic:**

| Condition | Call |
| --------- | ---- |
| `band_high` ≤ S breakpoint | Likely active |
| `band_low` > R breakpoint | Likely inactive |
| Otherwise | Uncertain — wait for lab |

**Overrides, applied after the model:**

1. Natural resistance for that species (`configs/natural_resistance.csv`) → inactive.
2. A strong known marker (e.g. any carbapenemase for meropenem) → inactive,
   regardless of the model.
3. Out of range (species not covered, or far from all training genomes) → all calls
   flagged low confidence.

**Scope note that must stay on every report:** these are predictions of in-vitro
susceptibility. Dose, route, and final drug choice depend on PK/PD, infection site,
renal function, allergies, and other patient factors, and remain with the clinician.
Confirm with standard AST.

---

## 3. Config files

Checked into the repo, reviewed by the whole team.

| File | Contents |
| ---- | -------- |
| `configs/species.yaml` | Species keys, AMRFinderPlus `-O` names, expected genome size, reference accessions |
| `configs/drugs.yaml` | Drug names, synonyms for normalization, spectrum tiers for ranking |
| `configs/breakpoints/eucast_<version>.csv` | `species`, `drug`, `s_breakpoint`, `r_breakpoint`, `version` |
| `configs/breakpoints/clsi_<version>.csv` | Same shape, for rows labelled CLSI |
| `configs/natural_resistance.csv` | `species`, `drug` — always inactive |
| `configs/keep_variant.csv` | Gene families where the exact variant is kept |

**Breakpoints vary by infection site** for some drugs (meropenem is stricter for
meningitis; some drugs are urinary-only). We use bloodstream breakpoints. Record the
site assumption in the breakpoint file.

---

## 4. Leakage checklist

Run before any result is reported.

- [ ] Unitig set built on training genomes only; all others queried
- [ ] Frequency filter and pyseer selection redone inside every fold
- [ ] Genomes de-duplicated by `biosample` across BV-BRC and NCBI
- [ ] No `country`, `year`, `source`, `lineage_cluster`, or `st` among features
- [ ] Thresholds and conformal calibration fitted on validation folds only
- [ ] Test set touched once, at the end
- [ ] No lineage cluster spans two splits or two folds

---

## 5. Build order

Data team can work stages 1–8 in parallel with the model team on 5–7 and 9–11.
Model team is unblocked as soon as stages 2, 5, 6, 7 exist — unitigs (8) come later.

| # | Task | Blocks |
| - | ---- | ------ |
| 1 | Ingest + harmonize → `labels.parquet`, `pairs_kept.csv` | Everything |
| 2 | Genomes + QC | 3, 4 |
| 3 | Lineages + splits (freeze) | All modeling |
| 4 | AMRFinderPlus → `known_amr.parquet` | Baselines |
| 5 | Metrics module + unit tests on toy data | All evaluation |
| 6 | Baselines B0 (ResFinder), B1 (lookup), B2 (XGB steps) | — |
| 7 | AFT model on known AMR only | — |
| 8 | Unitigs: build, query, filter, collapse | 9 |
| 9 | AFT on known AMR + unitigs; ablations | — |
| 10 | Conformal bands + ranking + report | App |
| 11 | External validation + distance plots | — |

**Start small:** one species (KPNEU), three drugs (ceftriaxone, meropenem,
ciprofloxacin). Get 1–7 working end to end before adding unitigs or more pairs.

---

## 6. Open questions

- [ ] PopPUNK or Mash clustering? Data team to test both on KPNEU and report.
- [ ] Which breakpoint version do we standardize on for the final call?
- [ ] Do we re-derive all S/I/R labels from MIC under one standard, or use labels
      as reported? (Doc says as-reported for now; revisit once MIC coverage is known.)
- [ ] Minimum `n` per species × drug — is 50/50 the right bar after seeing real counts?
