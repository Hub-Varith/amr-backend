# Data Pipeline Plan — Stages 1–8

**Status:** steps A–D done (2026-10-03) · **Date:** 2026-10-03 · **Scope:** KPNEU × {ceftriaxone, meropenem, ciprofloxacin} first

This plan builds the five handoff files in `DATA_CONTRACT.md`:
`labels.parquet`, `known_amr.parquet`, `lineages.parquet`, `splits.parquet`,
`unitigs_<SPECIES>.*`. If this plan and the contract disagree, the contract wins.

Each step lists: what it reads, what it writes, how we check it.
No step reads anything older than its declared input.

---

## 0. Ground rules for this work

- Write the test first for the interval rule, harmonize, and duplicate resolution.
- Every filter logs a count and a reason. Dropped rows go to
  `data/processed/dropped_<stage>.parquet` (`genome_id`, `drug`, `reason`).
- Every stage ends with a validator that runs the contract's acceptance checks
  and fails loudly.
- Per-genome tool calls live in `workflow/Snakefile`. Python stages are
  `python -m genome2mic.<module>` CLIs, wired into `make` targets.
- Seeds are fixed and written into the output (`splits.parquet` metadata).

---

## 1. Environment and machine setup (Stage 0)

### What we have today

| Item | State |
| ---- | ----- |
| Machine | Apple Silicon (arm64), 10 cores, 16 GB RAM |
| Free disk | **~23 GB** — see risk R1 |
| Python | 3.9 system; no `genome2mic` env yet (contract needs 3.11) |
| Bio tools | None installed (amrfinder, mash, mlst, unitig-caller, poppunk) |

### Steps

1. Create the Python env:
   `conda create -n genome2mic python=3.11` then `pip install -e ".[data,test]"`.
2. `data` extra in `pyproject.toml`: pandas, pyarrow, numpy (done). Add scipy,
   scikit-learn, snakemake when stages 3–8 need them. Downloads use the standard
   library `urllib`, so no new dependency.
3. Create a second env for bio tools, `genome2mic-tools`, from
   `workflow/envs/tools.yaml` (bioconda: `ncbi-amrfinderplus`, `mash`, `mlst`,
   `unitig-caller`, `poppunk`, `resfinder`). Many bioconda packages have no
   arm64 build, so create it with `CONDA_SUBDIR=osx-64` (Rosetta). If a tool
   still fails, run that Snakemake rule in a `linux/amd64` Docker image.
4. Pin tool versions. Record `amrfinder --version` and the database version in
   `data/interim/tool_versions.json`. Run `amrfinder -l` and confirm the `-O`
   names in `configs/species.yaml`.
5. Update `.gitignore`: ignore `data/raw/`, `data/interim/`, and the large
   processed files. **Track** `data/processed/splits.parquet` and
   `data/processed/pairs_kept.csv` (the contract says splits are version-controlled).

### Config files to write first

| File | Content for the KPNEU start |
| ---- | --------------------------- |
| `configs/species.yaml` | 5 keys, `-O` names, NCBI taxon IDs (ECOLI 562, KPNEU 573, SAUR 1280, PAER 287, ABAU 470), expected genome size, Mash reference accessions |
| `configs/drugs.yaml` | Canonical names + synonyms (`cefTRIAXone`, `ceftriaxone sodium`, `CRO` → `ceftriaxone`), spectrum tier |
| `configs/breakpoints/clsi_<year>.csv`, `eucast_<version>.csv` | `species, drug, s_breakpoint, r_breakpoint, version, site` — bloodstream |
| `configs/natural_resistance.csv` | e.g. KPNEU × ampicillin |
| `configs/keep_variant.csv` | `blaKPC`, `blaNDM`, `blaOXA-48`, `blaVIM`, `blaIMP` at minimum |

**Breakpoint form.** Store every breakpoint in one form: S if MIC ≤ `s_breakpoint`,
R if MIC > `r_breakpoint`. CLSI writes R as "≥ x", so CLSI `R ≥ 4` is stored as
`r_breakpoint = 2`. A unit test checks this conversion.

---

## 2. Stage 1 — Raw AST download

**Writes:** `data/raw/ast_bvbrc.csv`, `data/raw/ast_ncbi.csv` (never edited after download).

| Source | How |
| ------ | --- |
| BV-BRC | `genome_amr` via the BV-BRC API, `evidence = Laboratory Method`, genome name starting with the species name (taxon ID alone misses subspecies taxa). |
| NCBI | E-utilities: BioSamples with `antibiogram[filter]` per species. Raw XML kept in `data/raw/ncbi_biosample/`, flattened to `ast_ncbi.csv`. |

Also download genome metadata (BV-BRC `genome` table, NCBI isolate metadata) to
`data/raw/meta_<source>.csv`. That is where `biosample`, assembly accession,
`isolation_source`, `country`, `year` come from.

**Code:** `ingest/bvbrc.py` (`BvbrcDownloader`), `ingest/ncbi_ast.py` (`NcbiAstDownloader`).
Log row counts and download date. Save the request URL in a sidecar `.json`.

---

## 3. Stage 2 — Clean labels ★ contract

**Reads:** `data/raw/ast_*.csv`, `data/raw/meta_*.csv`, `configs/`.
**Writes:** `data/processed/labels.parquet`, `data/processed/pairs_kept.csv`,
`data/processed/label_counts.csv`, `data/processed/dropped_labels.parquet`.

### Steps, in order

| # | Step | Drop reason logged |
| - | ---- | ------------------ |
| 1 | Keep `evidence == 'Laboratory Method'` (BV-BRC). NCBI AST is lab-only by design. | `not_lab_method` |
| 2 | Map species from taxon ID / name to the 5-letter key. | `species_out_of_scope` |
| 3 | Normalize drug names via `drugs.yaml` synonyms. Unknown names are dropped, not guessed. | `drug_not_in_panel` |
| 4 | Map `laboratory_typing_method` → `dilution` / `gradient` / `disk`. Unknown method → drop. | `unknown_method` |
| 5 | Convert units to mg/L. µg/mL is 1:1. Any other unit → drop. | `bad_unit` |
| 6 | Build the interval (rule below). | `no_standard_for_sir_only`, `no_breakpoint`, `unparseable_value` |
| 7 | Assign `genome_id`: BV-BRC ID when present, else `NCBI_<biosample>`. | — |
| 8 | **De-duplicate by `biosample`** across sources. Same biosample in both → one `genome_id` (the BV-BRC one). | — |
| 9 | Resolve duplicate (`genome_id`, `drug`) rows (rule below). | `conflict_gt_1_step`, `conflict_s_vs_r` |
| 10 | Join metadata: `isolation_source`, `country`, `year`. | — |
| 11 | Run acceptance checks, write files, print the counts table. | — |

### The interval rule (`ingest/mic_interval.py`, `MicIntervalConverter`)

Exactly as the contract table:

| Raw | Interval | `censor` |
| --- | -------- | -------- |
| `= x` | (x/2, x] | `interval` |
| `<= x` | (0, x] | `left` |
| `> x` | (x, inf] | `right` |
| `>= x` | (x/2, inf] | `right` |
| `S` only | (0, s_bp] | `left` |
| `R` only | (r_bp, inf] | `right` |
| `I` only | (s_bp, r_bp] | `interval` |

Disk rows always take the S/I/R path, even if a number is present (it is a zone
diameter, not an MIC). S/I/R-only rows with unknown `standard` are dropped.

**Gaps in the contract — agreed 2026-10-03 (D2 = a), now in `DATA_CONTRACT.md`:**

- `< x` is not in the table. Proposal: treat as `<= x` → (0, x]. This is a wider,
  never-wrong interval.
- Panels report rounded values (`0.06` for 0.0625, `0.12` for 0.125, `0.03`).
  Proposal: snap any value within 5% of a doubling step (2^k) to that step; values
  that do not snap → drop with `off_grid_value`.

### Duplicate resolution (`ingest/harmonize.py`, `LabelHarmonizer`)

- Intervals within 1 doubling step → take the intersection. If the intersection is
  empty, keep the higher interval (safer).
- More than 1 step apart, or S vs R → drop the pair and log it.
- Test cases: exact vs exact, exact vs censored, S-only vs MIC, three-way duplicates.

### Acceptance checks (`validate/labels_checks.py`)

- (`genome_id`, `drug`) unique
- `mic_lower < mic_upper` on every row
- `censor == 'left'` ⟺ `mic_lower == 0`; `censor == 'right'` ⟺ `mic_upper == inf`
- No non-lab row survived
- No `-1`, `""`, `"NA"` anywhere; nulls are real nulls
- Schema and dtypes match the contract exactly

### Counts table and pair gate

`label_counts.csv`: species × drug × `n_R`, `n_S`, `n_exact`, `n_censored`,
`n_distinct_mic`. Non-susceptible = I + R.

`pairs_kept.csv`: pairs with ≥ 50 non-susceptible, ≥ 50 susceptible, ≥ 4 distinct
MIC levels.

**STOP POINT 1.** Review `label_counts.csv` together before any genome download.
If one of the 3 start drugs fails the gate, decide then (open question on 50/50).

---

## 4. Stage 3 — Genomes + QC

**Reads:** `labels.parquet`, `data/raw/meta_*.csv`.
**Writes:** `data/raw/genomes/<genome_id>.fasta.gz`, `data/interim/<genome_id>/mash.tsv`,
`data/processed/qc.parquet`.

1. Genome list = distinct `genome_id` in `labels.parquet` for kept pairs only.
   This keeps the download small.
2. Snakemake rule `download_genome`: BV-BRC API `/api/genome_sequence/` for BV-BRC IDs
   (the BV-BRC FTP server did not answer from the Mac); NCBI
   `datasets download genome accession` for `NCBI_` IDs. Store **gzipped** (risk R1).
   A genome that fails to download is logged, not retried forever.
   The 12,987 `NCBI_` genomes first need a BioSample → assembly lookup. Some
   BioSamples have reads only and no assembly; drop those with reason `no_assembly`.
3. Snakemake rule `assembly_stats`: `n_contigs`, `total_length`, `n50`,
   `gc_percent` (Python, streaming over the gz file).
4. Snakemake rule `mash_species`: `mash dist` against a sketch of the 5 species
   references (`configs/species.yaml`) → `mash_species`, `mash_distance`.
5. `qc/assembly_qc.py` (`AssemblyQc`) gathers the per-genome files and applies the
   4 fail rules. When several rules fail, `qc_fail_reason` lists all of them
   (`;`-joined).

**Check:** one row per downloaded genome; pass rate per species printed; the
species-mismatch list saved for review (it often reveals mislabelled uploads).

---

## 5. Stage 4 — Per-genome tool output

**Reads:** QC-passing FASTA. **Writes:** `data/interim/<genome_id>/{amrfinder.tsv,resfinder/,mlst.tsv}`.

Snakemake rules exactly as the contract (`amrfinder --plus -O <species>`,
`resfinder --acquired --point`, `mlst`). Species `-O` value comes from
`species.yaml`, never hard-coded. Run with `snakemake --cores 8`.

ResFinder output feeds only the B0 baseline. It never enters `known_amr.parquet`.

---

## 6. Stage 5 — Known-AMR features ★ contract

**Reads:** `data/interim/*/amrfinder.tsv`, `qc.parquet`, `configs/keep_variant.csv`.
**Writes:** `data/processed/known_amr.parquet`, `data/processed/known_amr_columns.csv`.

`features/known_amr.py` (`KnownAmrBuilder`):

1. Read each `amrfinder.tsv`. Accept `Element symbol` or `Gene symbol` as the name column.
2. Keep `Type == 'AMR'`. (STRESS/VIRULENCE rows are not drug resistance — ask if
   we want them later.)
3. `Subtype == 'POINT'` → `point_<gene>_<mut>`. Everything else → `gene_<family>`.
4. Family collapse: strip the variant suffix (`blaCTX-M-15` → `blaCTX-M`), unless
   the family is in `keep_variant.csv` (`blaKPC-2` stays `blaKPC-2`).
5. Name: lowercase, non-alphanumeric → `_` (`blaCTX-M` → `gene_blactx_m`).
6. `n_class_<class>` = count of AMR hits per AMRFinderPlus `Class`.
7. Pivot to wide, int8, fill absent with 0 (absent tool hit is a real 0, not a
   missing label — this is not imputation).
8. **No rare-feature filter here.** Emit the full table. Add
   `KnownAmrBuilder.load(min_count, train_ids)` so the model side filters inside
   each fold.

**Checks:** one row per QC-passing genome; no nulls; all values ≥ 0; mapping file
non-empty; a genome with no hits still has a row of zeros.

---

## 7. Stage 6 — Lineage clusters ★ contract

**Reads:** QC-passing genomes, `mlst.tsv`. **Writes:** `data/processed/lineages.parquet`.

`splits/lineages.py` (`LineageClusterer`):

1. Mash sketch all QC-passing genomes of the species; all-vs-all `mash dist`
   (Snakemake rule; ~3k genomes is ~4.5M pairs, fine on this machine).
2. Single-linkage clustering at threshold `t`. Pick `t` so ST258 and its CG258
   neighbours (KPNEU), ST131 (ECOLI) each stay inside one cluster. Save the chosen
   `t` and a short sweep table (t → n_clusters, largest cluster share).
3. PopPUNK run on KPNEU in parallel for the open question. Compare: adjusted Rand
   index vs Mash clusters, cluster size spread, ST purity.
4. `lineage_cluster` = `KPNEU_MS_<n>` (Mash) or `KPNEU_PP_<n>` (PopPUNK).
   `st` from `mlst.tsv`, `NA` when unknown.

**Hard check (automated):** every pair with Mash distance ≤ 0.0001 shares a cluster.
Also report the largest cluster's share of genomes — if one cluster holds > 40%,
the 15–20% test target cannot be met cleanly and we discuss before splitting.

---

## 8. Stage 7 — Splits ★ contract (freeze)

**Reads:** `lineages.parquet`, `labels.parquet`, `pairs_kept.csv`, `qc.parquet`.
**Writes:** `data/processed/splits.parquet` (committed, then never regenerated).

`splits/make_splits.py` (`SplitMaker`), per species, fixed seed:

1. Universe = QC-passing genomes with ≥ 1 label in a kept pair.
2. Shuffle clusters (seeded). Add whole clusters to `test` until 15–20% of genomes.
3. Check the test set has ≥ 10 R and ≥ 10 S for every kept drug (number to agree).
   If not, re-draw with the next seed and log every seed tried.
4. `GroupKFold(n_splits=5)` on remaining clusters → `fold` 0–4.
5. External sets: `time_holdout` (latest years), `country_holdout`,
   `source_holdout` — defined on test rows only, so train stays clean.
6. LOLO: `lolo_lineage = 'ST258'` (KPNEU), `'ST131'` (ECOLI) for those genomes.

**Invariant checks:** no cluster in two splits; no cluster in two folds; every
train row has a fold; every test row has null fold; per-fold R/S counts printed.

**STOP POINT 2.** Review splits together, then commit `splits.parquet` in its own
commit. From here, rule 7 applies: fix problems, never regenerate.

---

## 9. Stage 8 — Unitig matrix ★ contract

**Reads:** `splits.parquet`, QC-passing genomes. **Writes:** `unitigs_KPNEU.npz`,
`unitigs_KPNEU_rows.parquet`, `unitigs_KPNEU_index.parquet`.

`features/unitigs.py` (`UnitigMatrixBuilder`):

1. `unitig-caller --build` on `split == 'train'` and `qc_pass` genomes only.
2. `unitig-caller --query` for every other genome against that fixed set.
3. Stream the pyseer-format output straight into a sparse CSR int8 matrix. Never
   densify.
4. Frequency filter: keep unitigs present in 1%–99% of **training** genomes.
5. Collapse identical columns: hash each column's row-index set; one column per
   distinct pattern; all member sequences go into the index.
6. Write the 3 files. `pattern_id` = `u_000000…`.

**Checks:** rows file matches matrix rows; no test genome changed the unitig set
(build input list saved and diffed against train IDs); column count in the
expected range (tens of thousands to a few hundred thousand).

Per-drug pyseer selection is **not** done here. It is model-side, inside each fold.

---

## 10. Code layout to add

```
configs/                     species.yaml, drugs.yaml, breakpoints/, natural_resistance.csv, keep_variant.csv
workflow/Snakefile           download_genome, assembly_stats, mash_species, amrfinder, resfinder, mlst, mash_all_vs_all, unitig_build, unitig_query
workflow/envs/tools.yaml
src/genome2mic/
  ingest/   constants.py bvbrc.py ncbi_ast.py mic_interval.py harmonize.py
  qc/       assembly_qc.py
  features/ known_amr.py unitigs.py
  splits/   lineages.py make_splits.py
  validate/ labels_checks.py known_amr_checks.py splits_checks.py unitig_checks.py
tests/ingest/ test_mic_interval.py test_harmonize.py test_breakpoint_table.py   (done)
tests/<stage>/ test_known_amr_naming.py test_splits_invariants.py test_unitig_collapse.py
```

Make targets: `make labels`, `make genomes`, `make tools`, `make known-amr`,
`make lineages`, `make splits`, `make unitigs`, `make validate-data`, `make test-data`.

---

## 11. Order of work

| Step | Work | Gate |
| ---- | ---- | ---- |
| A ✅ | Env, configs, `.gitignore`, deps | `make test-data` runs |
| B ✅ | Tests then code: interval rule, breakpoint form, harmonize | 68 tests green |
| C ✅ | Download raw AST + metadata (`make download-ast`) | Raw row counts logged |
| D ✅ | Build `labels.parquet` + counts (`make labels`) | **Stop point 1** — waiting for review |
| E | Download KPNEU genomes, QC | Pass rate reviewed |
| F | AMRFinderPlus, mlst, ResFinder | All QC-passing genomes have output |
| G | `known_amr.parquet` | Checks green → **model team unblocked for B1/B2/AFT-known** once H done |
| H | Lineages (Mash + PopPUNK compare), splits | **Stop point 2**, commit splits |
| I | Unitigs | Checks green |
| J | Leakage checklist (contract §4), tick each line | Ready for stage 9 |

---

## 12. Risks

| ID | Risk | Plan |
| -- | ---- | ---- |
| R1 | Only ~23 GB free disk. ~3k KPNEU genomes ≈ 17 GB raw FASTA; unitig-caller temp files are larger. | Store FASTA gzipped (~5 GB). Put `data/` on an external drive or a cloud VM before step E. |
| R2 | 16 GB RAM may not fit `unitig-caller --build` on ~2.5k genomes. | Run stage 8 on a Linux VM if it runs out of memory. |
| R3 | Bio tools lack arm64 builds. | `CONDA_SUBDIR=osx-64` env, Docker `linux/amd64` fallback. |
| R4 | S/I/R-only rows need the breakpoint table for each `standard_year`. | Build tables only for years that cover most rows; drop and log the rest. |
| R5 | NCBI MIC strings are messy (`>=32`, `<=0.25/4` for combos). | For combos, the MIC is the first number (piperacillin part); test it. |

---

## 13. Decisions needed

**D1 — Disk.** Decided: AWS EC2 `r6i.2xlarge`, Ubuntu 24.04, 200 GB gp3 data disk,
S3 for backup. Labels (A–D) run on the Mac.

**D2 — Interval gaps.** Decided (a): `<` read as `<=`; values within 5% of a doubling
step snap to it; values between steps are dropped. Written into `DATA_CONTRACT.md`.

## 14. Stage 2 results (2026-10-03)

- 383,150 labels, 35,653 genomes, 5 species, 48 drugs. 97 species × drug pairs pass
  the inclusion rule (`data/processed/pairs_kept.csv`).
- KPNEU start drugs: ceftriaxone 4,131 rows, ciprofloxacin 5,309, meropenem 6,841 —
  all kept.
- 3,591 NCBI BioSamples matched a BV-BRC genome; 140 duplicate BV-BRC assemblies
  collapsed; 889 genome × drug pairs dropped for conflicting readings.
- Open items before training:
  - Verify `configs/breakpoints/*.csv` against CLSI M100 / EUCAST (entered from memory).
  - About 33k BV-BRC rows say "Disk diffusion" but carry mg/L values and no S/I/R.
    They are dropped. If the source paper shows they are MICs, they could be recovered.
  - `natural_resistance.csv` and `keep_variant.csv` are not written yet (needed at
    stages 5 and 10).
