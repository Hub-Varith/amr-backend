# Genome-to-MIC — Data Contract

**Status:** draft v0.3 · **Owner:** Hub · **Last updated:** 2026-10-03

**Changelog:** v0.3 (2026-10-03): review fixes. Stage text was updated in place for
stages 2, 4, 6, 7, 8, 11 and 12, section 3 (`drugs.yaml`), section 4 (how the report
checks it) and section 6. Changes: strict breakpoint-year lookup; combination and
decimal MIC parsing; merged-row provenance; one-step "exact" MIC as the EA,
exact-agreement and band-coverage denominator; LOLO lineages reserved as train
clusters; re-derived lab S/I/R; test ledger; carbapenem `strong_subclasses`; shared Mash
reference sketch; sparse lineage clustering; disk-diffusion rows are never exact MICs
(`lab_exact`); the test ledger records an input fingerprint (`inputs_sha1`); drug
bundles record their unitig k-mer set (`unitig_kmer_set_sha1`) and subset runs cannot
swap it; CV band coverage is labelled in-sample. Section 7 gained rows, and replaced rows
say what they replace. v0.2: section 7 additions from the first end-to-end build.

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
| `> 32` | 32 | inf | `right` |
| `>= 16` | 8 | inf | `right` |
| `S` only, breakpoint S ≤ 1 | 0 | 1 | `left` |
| `R` only, breakpoint R > 2 | 2 | inf | `right` |
| `I` only, S ≤ 1 and R > 2 | 1 | 2 | `interval` |

For `S`/`I`/`R`-only rows you need the breakpoint table matching that row's
`standard` and `standard_year` **exactly** (`configs/breakpoints/<std>_<year>.csv`).
**If the standard is unknown, drop the row.** If `standard_year` is null, or no table
exists for that year, drop the row too: there is no fallback to the latest table.
Each case has its own drop-log reason. Do not guess.

Numeric results: a combination-drug MIC `x/y` (`16/4` for piperacillin/tazobactam)
uses the primary agent's `x`, and `raw_result` keeps the full text. Panels print small
powers of two as rounded decimals (`0.016` = 2^-6, `0.008` = 2^-7, `0.12` = 2^-3). A
reported value within 0.1 log2 units of `2^k` is read as `2^k` before the rule
above, so `= 0.016` is `(0.0078125, 0.015625]`. Gradient half steps (`0.19`, `0.75`,
`1.5`, `3`, `6`, ...) are at least 0.39 log2 units away and keep their own grid cell.
Both are counted in the drop log; they are counts, not drops.

Why intervals: it lets exact MICs, "≤", ">", and S/R-only labels all train the
same model with no made-up numbers at the panel edges. The model loss
(`survival:aft`) consumes exactly these two columns.

#### Method filter

| `laboratory_typing_method` contains | `method` | MIC usable? |
| ----------------------------------- | -------- | ----------- |
| broth, microdilution, agar dilution | `dilution` | yes |
| Etest, gradient, MIC strip | `gradient` | yes |
| disk, Kirby-Bauer, zone | `disk` | **no — S/I/R only** |

Disk diffusion measures a zone diameter, not an MIC. Those rows still enter as
censored intervals via the S/I/R path. Even when a disk row's I range is one doubling
step (CLSI meropenem I = (1, 2]), it is not an exact MIC: `mic.lab_exact_mask`
(interval rule and `method != 'disk'`) is the definition of an exact MIC used by B2,
the conformal residuals, EA / exact agreement / band coverage and the `n_exact` counts.

#### Duplicate resolution

- Same (`genome_id`, `drug`), intervals within 1 doubling step → take the
  intersection; if empty, keep the higher (safer) one.
- More than 1 step apart, or S vs R → **drop that genome × drug pair** and log it.
- The merged row's `method`, `standard`, `standard_year` and `source` come as one
  block from a row that supports the merged bounds. Rows whose own interval equals
  the merged interval are preferred, then `dilution`/`gradient` over `disk`, then the
  highest reported step. When a dilution or gradient row reported the merged one-step
  cell, or supplied one of its bounds, the merged row is never `disk`. Other columns
  (biosample, isolation_source, country, year) may be filled from the other rows.
  One case stays `disk` on purpose: a disk `I`-only row whose own I range is one step
  (CLSI meropenem `(1, 2]`) that no MIC row tightened, because no dilution or gradient
  row measured that interval.

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
- [ ] Count table emitted: species × drug × (n_R, n_S, n_exact, n_censored, n_distinct_mic).
  `n_exact` = rows with an exact measured MIC: pinned to one doubling step
  (`mic_upper == 2 * mic_lower`) and `method != 'disk'` (the project's single "exact
  MIC" definition, `genome2mic.mic.lab_exact_mask`). `n_censored = n - n_exact`, so it
  also counts multi-step `interval` rows such as an `I`-only `(2, 8]` and one-step disk
  rows.

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

`data/interim/_references/references.msh` is the one shared file under `interim/`
(not a genome id). `mash.tsv` has one row per species reference
(`reference, query, distance, p_value, shared_hashes`; the reference column is the
reference FASTA path, whose file name `<SPECIES>.fasta` names the species). An empty
`mash.tsv` means mash was not run (tool or references missing); QC then identifies
the species with its pure-Python sketch.

Commands (Snakemake rules):

```bash
amrfinder -n data/raw/genomes/573.2002.fasta \
  -O Klebsiella_pneumoniae --plus --threads 4 \
  -o data/interim/573.2002/amrfinder.tsv

python -m resfinder -ifa data/raw/genomes/573.2002.fasta \
  -o data/interim/573.2002/resfinder \
  -s "Klebsiella pneumoniae" --acquired --point

mlst data/raw/genomes/573.2002.fasta > data/interim/573.2002/mlst.tsv

# once per run: every species reference into one multi-sketch file
mash sketch -o data/interim/_references/references data/raw/references/*.fasta
# per genome: the .msh is the reference, so every species reference is compared
mash dist data/interim/_references/references.msh data/raw/genomes/573.2002.fasta \
  > data/interim/573.2002/mash.tsv
```

The amrfinder rule declares 4 threads and 4000 MB (`--config amrfinder_threads=N
amrfinder_mem_mb=M` to change them).

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
clustering at a threshold that keeps known lineages (ST131, ST258) intact. Implemented
without a dense matrix: all-pairs distances are computed block-wise and only pairs with
`d <= max(threshold, 0.0001)` are kept; clusters are the connected components of the
`d <= threshold` graph, which is exactly single linkage cut at the threshold (same
cluster numbering). The 0.0001 spot-check runs on the kept pairs.

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
| `lolo_lineage` | str | Null, or the lineage name for leave-one-lineage-out runs (train rows only; the cluster is reserved before test selection) |

**Construction:**

1. Reserve the `n_lolo` (default 2) largest `lineage_cluster` values per species for
   leave-one-lineage-out. Reserved clusters are never test candidates, so they always
   stay `train` (a LOLO fit = train minus the lineage, a genuine refit).
2. Assign whole non-reserved `lineage_cluster` values to `test` until ~15–20% of all
   genomes of the species (reserved clusters included in the denominator) are held out.
   Check the test set retains both R and S for each kept drug; repair swaps never bring
   a reserved cluster into test (a class that exists only in test and reserved clusters
   is logged as unfixable).
3. `GroupKFold(n_splits=5)` on the remaining clusters → `fold`.
4. Mark external sets separately (a genome can be both `test` and in an external set
   only if the external set is defined on test; prefer disjoint).
5. `lolo_lineage` = the reserved cluster's id on every row of that cluster (train rows
   only). For the biggest clinical lineages (ECOLI ST131, KPNEU ST258) this is where
   LOLO rows come from.

**Invariant:** no `lineage_cluster` appears in two different values of `split`, no
cluster spans two folds, and `lolo_lineage` is null on every `test` row.

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

**CV and LOLO rows are part of the build set.** "Training genomes" in steps 1-3 means
every `split == 'train'` genome, so the validation genomes of each CV fold and the
held-out lineage of each LOLO run helped define which k-mers/unitigs exist, the
1 %/99 % build-time filter and the pattern collapse; their rows are `built` (exact
encoding). Only `split == 'test'` rows, and new isolates, are `queried` (>= 50 % of
member k-mers rule). This is allowed because (a) no label is read when building (the
build reads `splits.parquet` and `qc.parquet` only), (b) every per-fold choice that
*uses* the features is recomputed on the fold's fit rows inside the fold: the frequency
window is re-applied on fit rows (a pattern private to the validation genomes or to the
held-out lineage has fit-row frequency 0 and is dropped), and the correlation / pyseer
ranking sees fit-row labels only, and (c) the test split, which carries the reported
claims, is untouched. Consequence to state in reports: CV and LOLO estimates carry mild
transductive optimism (the feature universe was fitted on held-out *sequence*, not
labels) and do not exercise the query rule that test rows and new isolates go through;
test-set metrics have neither caveat.

**k-mer backend (no unitig-caller):** the pure-Python fallback builds canonical
31-mer patterns in three passes over the training genomes and never materialises
the `genomes x k-mers` matrix. It refuses species with more than
`--unitig-max-kmer-genomes` (default 1,000) training genomes; real data uses
`--unitig-backend unitig-caller`.

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

Rows are in the file's column order (VME first).

| Column | Meaning |
| ------ | ------- |
| `vme_rate` | **Predicted S, lab R.** The dangerous error. Target ≤ 1.5% |
| `me_rate` | Predicted R, lab S. Target ≤ 3% |
| `mine_rate` | One side is I |
| `categorical_agreement` | Same S/I/R. Target ≥ 90% |
| `essential_agreement` | Within ±1 doubling step of the lab MIC, over rows with an **exact** lab MIC (interval of one doubling step, `genome2mic.mic.exact_interval_mask`, and preds `lab_exact` true, so never disk diffusion; section 7); censored (`<=`, `>`) and multi-step S/I/R-only intervals are excluded. Denominator `n_exact`. Target ≥ 90% |
| `exact_agreement` | Same doubling step, over the same exact rows (denominator `n_exact`) |
| `auroc` | Predicted MIC as a score for R vs S |
| `band_coverage` | Share of exact lab MICs inside the 90% band (the rows the conformal `q` is calibrated on). Denominator `n_band`. Should be ≈ 0.90. On `cv` rows this is the conformal calibration set (in-sample by construction); only test, external and LOLO coverage is an evaluation |
| `band_width_steps` | Mean band width in doubling steps, over every row with a band |
| `n` | Rows in the group (species × drug × model × evaluation set) |

The denominators (`n_exact`, `n_band`, `n_cat`, `n_lab_r`, `n_lab_s`) and the
re-derived categorical block (`vme_rate_rederived` first) follow `n`; see section 7.

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
   regardless of the model. Two rules, both from `configs/drugs.yaml`:
   - `strong_markers`: known-AMR feature columns, matched as column-name prefixes
     (kept variants such as `gene_blakpc_2` match `gene_blakpc`). Every carbapenemase
     prefix in `keep_variant.csv` (blaOXA-48, blaOXA-181, blaOXA-232, ...) must be
     listed for the carbapenems and the cephalosporins that carry the list.
   - `strong_subclasses` (optional): AMRFinderPlus `Subclass` values. A detected
     acquired gene (Type `AMR`, Subtype `AMR`; never a `POINT` mutation such as an
     ompK36 porin change) whose own Subclass contains one of them also triggers the
     override. The carbapenems (ertapenem, imipenem, meropenem) list `CARBAPENEM`, so
     carbapenemases that share a family column with non-carbapenemases
     (blaOXA-23/-58 with blaOXA-1 in `gene_blaoxa`, blaGES-5 with blaGES-1 in
     `gene_blages`) or that no prefix covers still force the call.

   `reasons` lists the triggering marker symbols. The model's MIC fields are kept.
3. Out of range (species not covered, or far from all training genomes) → all calls
   flagged low confidence.

Prediction-time known-AMR rows follow stage 5 exactly: an AMRFinderPlus table is read
by the stage-5 parser (Type == AMR exactly; Subtype in {AMR, POINT}; non-null symbol;
`NA` cells null), and `markers.fasta` records honour `type=` / `subtype=` the same
way, so a VIRULENCE or AMR-SUSCEPTIBLE hit never becomes a `gene_` feature at
prediction time. Drop counts go to the prediction DropLog
(`amrfinder_type_not_amr`, `amrfinder_subtype_not_amr_or_point`, `amrfinder_no_symbol`,
`marker_records_not_amr`).

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
| `configs/drugs.yaml` | Drug names, synonyms for normalization, spectrum tiers for ranking, `strong_markers` (feature-column prefixes) and optional `strong_subclasses` (AMRFinderPlus Subclass values) for override 2, `call_standard` |
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

**How the report checks these** (`genome2mic.eval.leakage`, printed in
`results/report.md`):

- The checklist preamble states that unitig patterns are built from every train genome,
  including genomes held out within CV folds and LOLO runs; only test genomes are
  queried. No labels enter the build and per-fold selection recomputes frequency and
  ranking on fit rows, so it is not label leakage, but CV/LOLO rows use the build-time
  encoding and may read slightly optimistic (stage 8).
- Check "prediction rows match their split" also requires every `lolo_<X>` preds row
  to be a `train` genome with `splits.lolo_lineage == X`. LOLO rows without a
  `lolo_lineage` column in `splits.parquet` fail.
- Check "test set touched once" reads `results/test_ledger.csv`. It fails when a
  species × drug has more than one distinct `(run_id, inputs_sha1)`; the pair and each
  `run_id/inputs_sha1` are named, so re-scoring after any change to labels, features,
  QC, lineages, the unitig set, drugs.yaml, breakpoint tables or model/feature/MIC code
  fails even under the same `run_id`. It also fails when the ledger is missing although
  preds exist ("no test ledger: cannot verify"), when it lacks `run_id`/`species`/`drug`,
  or when a preds file's test rows carry a `run_id` the ledger never recorded for that
  pair. A ledger without the `inputs_sha1` column still parses (it then passes with a
  note that input changes under the same `run_id` cannot be detected). An empty
  `inputs_sha1` (a row written before fingerprints existed) is unknown and counts as
  distinct from any recorded fingerprint. Re-running the identical configuration on
  identical inputs passes. It is NOT RUN only when there is neither a ledger nor any
  preds file. What it proves: the test rows of each pair were only ever scored by one
  configuration on one set of inputs. What it cannot prove: that nobody looked at test
  metrics before settling on that configuration. The report says this.

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
      (Interim, v0.3: label conversion uses the row's own `(standard, standard_year)`
      table only, so historical years need their own tables.)
- [ ] Do we re-derive all S/I/R labels from MIC under one standard, or use labels
      as reported? (Doc says as-reported for now; revisit once MIC coverage is known.
      v0.3: labels stay as reported; metrics show both the as-reported and the
      re-derived categorical metrics, as-reported first.)
- [ ] Minimum `n` per species × drug — is 50/50 the right bar after seeing real counts?
- [ ] Should the cephalosporins also list `strong_subclasses: [CARBAPENEM]`? Today a
      GES-5 or OXA-23 isolate is overridden for meropenem but not for ceftriaxone.
- [ ] Should `PredictionReport` name the known-AMR backend that ran (AMRFinderPlus,
      precomputed TSV, synthetic MarkerScan)? Adding a field changes the stage-12
      report schema (`extra='forbid'`). MarkerScan is now synthetic-only, so real
      bundles are not at risk meanwhile.

---

## 7. Additions (v0.2, v0.3)

Columns and files added on top of v0.1, sorted by file. Each row names the stage that
writes it. v0.2 rows came from the first end-to-end build and are strictly additive.
Rows marked v0.3 came with the review fixes, which also changed stage text above (see
the changelog). A v0.3 row that replaces a v0.2 row says so and says what changed.

| Where | Addition | Written by | Meaning |
| ----- | -------- | ---------- | ------- |
| `configs/drugs.yaml` | `strong_subclasses` (optional list) (v0.3) | config | See stage 12 override 2 |
| `data/interim/_references/references.msh` | new file (v0.3) | Snakefile (`mash_references`) | Mash sketch of every `data/raw/references/<SPECIES>.fasta`, the reference side of every per-genome `mash dist` |
| `data/processed/drop_log_<stage>.csv` | new files | every stage | `stage, reason, n_dropped, detail` for every filter, including zero counts |
| `data/processed/drop_log_evaluate.csv` | reason `lab interval censored or wider than one doubling step: excluded from EA, exact agreement and band coverage` (v0.3) | evaluate | Count of prediction rows left out of the exact-MIC metrics (not dropped from the preds) |
| `data/processed/drop_log_evaluate.csv` | reason `lab_exact false (one-step interval but not a measured MIC, e.g. disk diffusion): excluded from EA, exact agreement and band coverage`; plus `lab_exact null: treated as not an exact MIC` (only when nulls occur) (v0.3) | evaluate | Prediction rows with a one-step lab interval that `lab_exact` marks as not an MIC. Preds without the column log a zero count with detail `column lab_exact absent: interval rule only`. The existing reason `lab interval censored or wider than one doubling step: ...` is unchanged |
| `data/processed/drop_log_evaluate.csv` | reason `pred_sir or lab_sir_rederived null: excluded from re-derived categorical metrics` (v0.3) | evaluate | Count of rows without a re-derived lab category or a predicted category |
| `data/processed/drop_log_ingest.csv` | reason `S/I/R-only row with null standard_year` (v0.3) | ingest | S/I/R-only row (disk, or no value) with a standard but no `standard_year`: no table can be matched. NCBI AST exports usually have no year column, so their S/I/R-only rows land here (the reader uses a `testing_standard_year` column when one is present) |
| `data/processed/drop_log_ingest.csv` | reason `no breakpoint table for standard_year` (v0.3) | ingest | S/I/R-only row whose `(standard, standard_year)` has no `configs/breakpoints/<std>_<year>.csv`; `detail` lists the most common missing `<STANDARD> <year>` pairs |
| `data/processed/drop_log_ingest.csv` | reason `combination MIC 'x/y': primary-agent value x used, full text kept in raw_result (count, not a drop)` (v0.3) | ingest | Kept numeric rows whose value was `x/y`; `x` was used. `detail` lists the most common texts |
| `data/processed/drop_log_ingest.csv` | reason `decimal rendering of a power of two snapped to the doubling grid, e.g. 0.016 -> 2^-6 (count, not a drop)` (v0.3) | ingest | Kept numeric rows whose reported value was read as the nearby power of two (`0.016` -> 2^-6). `detail` lists the most common texts |
| `data/processed/drop_log_report.csv` | reason `mic_confusion: lab result not an exact MIC (interval wider than one doubling step, or lab_exact false), or null prediction` (v0.3; renamed from `mic_confusion: censored lab interval or null prediction`) | report | Rows left out of the MIC confusion heatmap; it uses the same exact-row mask as EA (`eval.metrics.exact_lab_mask`) |
| `data/processed/drop_log_train.csv` | `detail` always starts with `"<SPECIES> x <drug>"` (v0.3) | train | Lets a subset run replace only its own pairs' rows (rows of other pairs are kept) |
| `data/processed/drop_log_train.csv` | reasons `one-step lab interval but not a measured MIC (e.g. disk diffusion): B2 trains on exact MICs only` and `one-step lab interval but not a measured MIC (e.g. disk diffusion): conformal residuals use exact MICs only` (v0.3) | train | Counts of one-step disk rows left out of B2 fitting and of the conformal residuals |
| `data/processed/known_amr_columns.csv` | `member_symbols` column | known-amr | `;`-joined original symbols collapsed into the column |
| `data/processed/label_counts.csv` | new file | ingest | The stage-2 count table (`species, drug, n, n_R, n_S, n_I, n_exact, n_censored, n_distinct_mic`) persisted next to `pairs_kept.csv` |
| `data/processed/label_counts.csv` | `n_exact` definition (v0.3; `censor == 'interval'` in v0.2) | ingest | Rows with an exact measured MIC (`mic.lab_exact_mask`: one doubling step, `mic_upper == 2 * mic_lower`, and method not `disk`); `n_censored = n - n_exact` also counts one-step disk rows |
| `data/processed/qc.parquet` | `species` column | qc | Label species (or Mash species for unlabelled genomes) used by the size rule |
| `data/processed/sketches_<SPECIES>.npz` | new file | lineages | MinHash sketches of every QC-passing genome (ids, sketches, k, s) |
| `data/processed/unitigs_<SPECIES>_index.parquet` | written in row groups; readers select columns (v0.3) | unitigs | Same columns/types as stage 8 (`unitig_sequences` stays `list<string>`). `load_unitigs` reads `col_index, pattern_id, n_unitigs, train_frequency` by default; read `unitig_sequences` only for the patterns you display |
| `data/processed/unitigs_<SPECIES>_kmers.npz` | new file | unitigs | Frozen k-mer set (+ pattern column per k-mer) used to query new genomes without rebuilding |
| `data/processed/unitigs_<SPECIES>_rows.parquet` | `role` column (`built` / `queried`) | unitigs | Which rows built the k-mer set; the leakage check asserts `built` implies `split == 'train'` |
| `data/raw/fetch_summary.json` | new file; `aws_profile` key (v0.3) | fetch | Record of the last real (non-dry) `fetch` run: `source`, `provider`, `tool`, `command`, `dest`, `dry_run`, the counts (`n_files`, `total_bytes`, `n_genomes`, `n_ast_files`, `ast_files`), `manifest`, `n_links`, layout `problems`, `elapsed_s`, `aws_profile` (the profile name set for `aws s3 sync`, or null) and `written_at`. URI query strings and `user:password@` parts are redacted; credentials are never written |
| `data/raw/genomes_manifest.csv` | new file | fetch | `genome_id, path, bytes` (paths relative to the run root). Written only when some genome file under `data/raw/genomes/` is not `<genome_id>.fasta` (other suffix, gzip, sub-folder); `fetch.resolve_genome_paths` reads it when present |
| `models/` | bundle layout | train | `manifest.json`, `reference_sketches.npz`, `markers.fasta` (synthetic runs only; see its own row), `<SPECIES>/train_sketches.npz`, `<SPECIES>/unitig_kmers.npz`, `<SPECIES>/unitig_index.parquet`, `<SPECIES>/<drug>/{model.ubj, params.json, features.json, conformal.json, meta.json, importance.json}` |
| `models/<SPECIES>/<drug>/features.json` | `feature_names` (list[str]) (v0.3) | train | Exactly `known_columns + unitig_cols` in model-input order (unitig ids `u_000017`). Prediction refuses to load (BundleError, service not ready) when it differs from that concatenation or from the saved model's own feature names, order included |
| `models/<SPECIES>/<drug>/features.json` | `unitig_kmer_set_sha1` (str, 40 hex, or null) (v0.3) | train | `KmerSet.sha1()` of the k-mer set whose pattern columns the model's `unitig_cols` index. It equals the `sha1` stored in `unitigs_<SPECIES>_kmers.npz` and in the shipped `models/<SPECIES>/unitig_kmers.npz`. Null for a model without unitig features. The prediction pipeline refuses to load (BundleError, service not ready) when a drug's value differs from the species' `unitig_kmers.npz`. Bundles without the key load with a warning, because the pairing cannot be verified |
| `models/<SPECIES>/<drug>/meta.json` | `inputs_sha1`, `unitig_kmer_set_sha1` (v0.3) | train | Training record: the species' input fingerprint (as in `results/test_ledger.csv`) and the k-mer set the model used |
| `models/manifest.json` | `run_ids` key (`{SPECIES: {drug: run_id}}`) (v0.3) | train | Which training run produced each pair's bundle. `run_id` = the latest run. A run over every kept pair overwrites the manifest; a subset run (`train --species/--drugs`) merges its pairs into the existing manifest (pairs whose bundle directory is gone are dropped and logged). `models/<SPECIES>/unitig_kmers.npz` is shared by every drug of the species: a subset run refuses to start (ContractViolation, nothing written) when the k-mer set it would ship differs from the set recorded by any drug of that species that it does not retrain but that stays in `models/manifest.json`. Older bundles that have unitig columns but no recorded sha1 are compared with the currently shipped set. Fix: retrain all drugs of the species in one run, or run a full train. Training also stops if `unitigs_<SPECIES>_kmers.npz` changes between loading and shipping |
| `models/manifest.json` | `synthetic` (bool) (v0.3) | train | `true` only for bundles trained on SYNTHETIC data. The prediction pipeline uses the `markers.fasta` exact-match fallback (MarkerScan) only when it is `true`; a real bundle without `amrfinder` on PATH and without a precomputed AMRFinderPlus TSV (`--amrfinder-tsv` or `<fasta>.amrfinder.tsv`) fails with `ToolNotAvailable`. A non-boolean value is a bundle error |
| `models/markers.fasta` | synthetic runs only (v0.3; replaces "markers.fasta (synthetic runs)" in the `models/` row) | train | Written only when `data/raw/SYNTHETIC_DATA.md` exists; any pre-existing file is deleted on a non-synthetic run so a real bundle cannot fall back to the synthetic MarkerScan |
| `reports/synthetic_demo/` | `README.md`; leading `synthetic` column (`True`) in `metrics.csv` and `metrics_by_distance.csv` (v0.3) | `make demo-copy` (`python -m genome2mic.eval.demo`) | Every artefact copied for the tracked demo says it is synthetic; the copier refuses a run root without `data/raw/SYNTHETIC_DATA.md` |
| `results/figures/*.png` | `SYNTHETIC DATA` watermark + note line; PNG `Description` metadata (v0.3) | report | Only on a synthetic run (`data/raw/SYNTHETIC_DATA.md` present) |
| `results/metrics.csv`, `results/metrics_by_distance.{parquet,csv}` | new files | evaluate | CSV copies for the demo; metrics stratified by `nearest_training_distance` bin (`distance_bin, bin_low, bin_high` + the metrics, VME first) |
| `results/metrics.parquet` | `n_exact`, `n_band`, `n_cat`, `n_lab_r`, `n_lab_s` columns (v0.3; replaces the v0.2 row `n_exact, n_cat, n_lab_r, n_lab_s`: adds `n_band`, and "exact" is now one doubling step) | evaluate | Denominators: `n_exact` = rows with a prediction and an exact lab MIC (one step and `lab_exact`; see the exact-lab-MIC row below), behind EA and exact agreement (0 for `b0_resfinder`, which has no MIC); `n_band` = rows with a band and an exact lab MIC, behind band coverage; `n_cat`, `n_lab_r`, `n_lab_s` behind CA / minor error, VME and ME |
| `results/metrics.parquet` | `vme_rate_rederived, me_rate_rederived, mine_rate_rederived, categorical_agreement_rederived, n_cat_rederived, n_lab_r_rederived, n_lab_s_rederived` (a block after the counts, VME first) (v0.3) | evaluate | The categorical metrics recomputed against `lab_sir_rederived` (the lab interval re-classified under the call breakpoint). The as-reported columns (`vme_rate` ...) are unchanged and stay first. NaN rates and 0 counts when the preds have no `lab_sir_rederived` |
| `results/metrics.parquet`, `results/metrics_by_distance.parquet` | definition of "exact lab MIC" behind `essential_agreement`, `exact_agreement`, `band_coverage`, `n_exact`, `n_band` (v0.3) | evaluate | One-step lab interval AND `lab_exact` true when the preds carry the column (older preds without it: interval rule only; a null `lab_exact` counts as not exact). The column can only narrow the interval rule (`eval.metrics.exact_lab_mask`) |
| `results/metrics_by_distance.{parquet,csv}` | same columns as `metrics.parquet` (v0.3) | evaluate | `distance_bin, bin_low, bin_high` + every metric and count column above (VME first, re-derived block last) |
| `results/preds_<SPECIES>_<drug>.parquet` | `external_set` column | train | Copied from `splits.parquet` on `split == 'test'` rows; null elsewhere |
| `results/preds_<SPECIES>_<drug>.parquet` | `nearest_training_distance` column | train | Min Mash distance from the genome to the genomes the model was fitted on (fold training genomes for `cv`, all train rows for `test`, the LOLO training set for `lolo_*`) |
| `results/preds_<SPECIES>_<drug>.parquet` | `split` values `cv`, `lolo_<lineage>` (v0.3; replaces the v0.2 row, which said only "`lolo_<cluster>` = leave-one-lineage-out") | train | `cv` = out-of-fold predictions of train genomes; `lolo_<cluster>` = the reserved cluster's *train* genomes predicted by a model fitted on the train rows outside it. A LOLO lineage with no train rows for the pair, or with < 10 remaining fit rows, is skipped and counted in `drop_log_train.csv` (`lolo_lineage_without_train_rows`, `lolo_too_few_fit_rows`) |
| `results/preds_<SPECIES>_<drug>.parquet` | `model` value `b0_resfinder` | train | ResFinder S/R baseline; `pred_mic`, `band_low`, `band_high` null |
| `results/preds_<SPECIES>_<drug>.parquet` | `lab_sir_rederived` column (str, nullable) (v0.3) | train | The lab interval classified under the call breakpoint (`config.call_breakpoint`): `hi <= S` → `S`, `lo >= R` → `R`, `lo >= S and hi <= R` → `I`, otherwise null (also null when there is no call breakpoint). Present on every model, including `b0_resfinder`. `lab_sir` stays as reported (the lab's own standard/year) |
| `results/preds_<SPECIES>_<drug>.parquet` | `lab_exact` column (bool, never null) (v0.3) | train | True when the lab result is an exact measured MIC: the lab interval is one doubling step (`mic.exact_interval_mask`: `lo > 0`, `hi < inf`, `hi == 2*lo`) AND `labels.method != 'disk'` (`mic.lab_exact_mask`). A disk-diffusion S/I/R-only row whose I range happens to be one step (e.g. CLSI meropenem I = (1, 2]) is not an MIC (stage 2 method filter: disk -> "MIC not usable"). The same mask selects B2's training rows and the conformal residuals. Present on every model, including `b0_resfinder` |
| `results/report.md` | CV band-coverage footnote (v0.3) | report | In CV metrics tables the band-coverage header reads `Band coverage % (90% band; exact lab MICs; in-sample, see note)`, followed by the note "CV coverage is the conformal calibration set (in-sample by construction); only test/external/LOLO coverage is an evaluation." |
| `results/test_ledger.csv` | new file, append-only (v0.3) | train | Header `run_id,created_utc,species,drug,n_test_rows,inputs_sha1`. One row each time `train` writes test predictions for a pair (`n_test_rows` = rows with `split == 'test'` in that preds file, all models; `created_utc` ISO 8601). `run_id` is unchanged (sha1 of TrainConfig + `splits.parquet`). `inputs_sha1` = `models.train.compute_inputs_sha1(paths, species)`: the first 12 hex characters of sha1 over one `name TAB content-sha1` line per input, in this order (names, not paths: the data files live in `data/processed/`): `data/labels.parquet`, `data/known_amr.parquet`, `data/qc.parquet`, `data/lineages.parquet`, the species' `unitigs_<SPECIES>.npz`, `unitigs_<SPECIES>_rows.parquet`, `unitigs_<SPECIES>_index.parquet` and `unitigs_<SPECIES>_kmers.npz` (npz archives hashed member by member, so zip metadata never matters), `configs/drugs.yaml`, every `configs/breakpoints/*.csv`, every `.py` file under `src/genome2mic/models/` and `src/genome2mic/features/`, and `src/genome2mic/mic.py`. A missing file contributes `absent`. Names are root-relative, so the value does not depend on the checkout or run root. It is per species: other species' unitig files are not part of it. Not covered: `splits.parquet` and the training parameters (they are in `run_id`) and per-genome tool output under `data/interim/` (the ResFinder tables behind `b0_resfinder`). Never truncated; a ledger with the legacy 5-column header is upgraded once, in place and atomically, on the next append: every row is kept, with an empty `inputs_sha1`. Any other header makes `train` stop. The leakage check fails a pair with more than one distinct `(run_id, inputs_sha1)` (section 4) |

Open: stage 6 names `NA` as the `st` sentinel while the global conventions forbid
`NA`; the lineages stage follows the explicit column rule until the owner decides.
