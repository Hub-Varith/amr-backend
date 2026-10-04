# genome2mic hackathon demo: one genome in, ranked drugs out (5 species)

> **These are predictions of in-vitro susceptibility, not prescribing advice.** Dose, route,
> and final drug choice depend on PK/PD, infection site, renal function, allergies, and other
> patient factors, and remain with the clinician. Confirm with standard AST.
>
> This is a small check on 15 genomes the model never saw. It shows that the pipeline runs end to
> end. It does not validate the model. All accuracy numbers for the model come from
> cross-validation (out-of-fold predictions on the release's training folds). The test split was
> not used here.

## What runs

```
genome FASTA
  -> QC + species ID (MinHash distance to 5 species references; > 0.05 from all of them = no species, no predictions)
  -> AMRFinderPlus 4.2.7 (-n, --plus, -O <organism>, DB 2026-08-07.1)
  -> known-AMR features with the NCBI-release rules (predict/release_features.py; parity checks below)
  -> one model per species x drug: aft_b2_select (the AFT model, B2 or their average, chosen inside the
     training folds; bundle runs/hackathon5/models, run 72a57bc3e199, v0.7 code)
  -> panel-edge cap, round up to the doubling grid, asymmetric cross-conformal band
  -> CLSI 2024 call: likely_active (band_high <= S) / likely_inactive (band_low > R) / uncertain
     overrides: natural resistance -> inactive; strong marker (e.g. carbapenemase for carbapenems) -> inactive
  -> prob_works: calibrated P(lab MIC <= S breakpoint), shown next to the call (never changes it);
     strong-marker hits get the pair's smoothed rate (lab S + 1) / (n + 2); the tier never says
     'works' next to a likely_inactive call
  -> likely-active drugs ranked by spectrum tier, then by margin below the S breakpoint
```

The bundle was trained on `runs/hackathon5`. That is the provisional local 5-species release
`2026-10-04-hackathon+pd5-local`, imported with `import-release` and trained with
`train --cv-only`, so the test rows were never loaded. It covers 97 species x drug pairs:
KPNEU 29, ECOLI 25, ABAU 17, PAER 14, SAUR 12.

## How to run

From the repo root:

```bash
export PATH=$HOME/micromamba/envs/amrfinder/bin:$PATH        # AMRFinderPlus 4.2.7 + its DB

# once per bundle trained on an imported NCBI release (train writes it too when amrfinder is on PATH):
.venv/bin/python -m genome2mic release-feature-spec --root runs/hackathon5

# one genome -> JSON report
.venv/bin/python -m genome2mic predict --root runs/hackathon5 \
    --fasta runs/hackathon5/demo/genomes/573.24243.fasta --sample-id 573.24243 > runs/hackathon5/demo/573.24243.json

# the whole demo (selection, BV-BRC download, 15 predictions, comparison with the lab)
.venv/bin/python scripts/hackathon_demo.py select  --root runs/hackathon5      # first pick (4 failed species QC, see below)
.venv/bin/python scripts/hackathon_demo.py select  --root runs/hackathon5 --exclude 573.17927,562.143964,287.5689,287.5751
.venv/bin/python scripts/hackathon_demo.py fetch   --root runs/hackathon5
.venv/bin/python scripts/hackathon_demo.py predict --root runs/hackathon5 --jobs 3
.venv/bin/python scripts/hackathon_demo.py compare --root runs/hackathon5
```

The JSON reports and logs are in `runs/hackathon5/demo/<genome_id>.{json,log}`, which is
gitignored. Copies of the tables are in this folder:

- `demo_table.csv`: one row per genome x drug lab result
- `demo_per_genome.csv`
- `demo_qc_rejected.csv`
- `genomes.csv`
- `parity_*.csv`

## Demo genomes

How genomes were picked: for each species, BV-BRC genome ids that have lab results in
`labels.parquet` (for a kept species x drug pair) but are **absent from `splits.parquet`**. So
the models never saw them in training, and they are not test-split genomes.

Each species gets 3 genomes, taken in order of most drugs with a lab result:

1. the first genome with at least 2 lab-R drugs,
2. the first with no lab-R drug,
3. the next with at least 1 lab-R drug.

Lab S/I/R is re-derived from the MIC under CLSI 2024 where the MIC interval allows it. Otherwise
the lab's own S/I/R is used.

### Species QC rejected 4 of the first picks

The pipeline gave no prediction for these 4. They were replaced with the `--exclude` re-pick.

| genome | BV-BRC species | assembly | nearest reference distance | result |
| --- | --- | --- | --- | --- |
| 573.17927 | KPNEU | 2 contigs, 5.60 Mb | 0.0563 | too_distant: no species, no predictions |
| 562.143964 | ECOLI | 454 contigs, **8.86 Mb** (likely mixed/contaminated) | 0.0517 | too_distant |
| 287.5689 | PAER | 4 contigs, 7.25 Mb | 0.0559 | too_distant |
| 287.5751 | PAER | 4 contigs, 7.25 Mb | 0.0561 | too_distant |

A distance of about 0.055 to the species reference is near the species boundary (roughly 95% ANI). Likely
explanations are *K. quasipneumoniae* / *K. variicola* labelled as *K. pneumoniae*, and the
PA7-like *P. aeruginosa* clade (*P. paraeruginosa*). Refusing to predict is the intended behaviour.

### Results (15 genomes, 233 lab results; final bundle run 72a57bc3e199)

All 19 reports (15 genomes plus the 4 QC rejections) were regenerated on the final bundle and
carry `run_id: 72a57bc3e199`.

Columns:

- `VME`: lab R, called likely_active.
- `ME`: lab S, called likely_inactive.
- `agree`: lab R called likely_inactive, or lab S called likely_active.
- `EA`: predicted MIC within ±1 doubling step, counted only on exact (one-step) lab MICs.
- `lab R told works`: lab R with `prob_works` >= 0.70.
- `ranked active`: the top 5 likely-active drugs, narrowest spectrum first.

| species | genome | lab results | VME | ME | agree | uncertain | EA | lab R told works | ranked likely-active (top 5) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KPNEU | 573.24243 | 26 | 0 | 0 | 19 | 4 | 6/6 | 0 | none |
| KPNEU | 573.12886 | 19 | 0 | 0 | 7 | 11 | 5/5 | 0 | tobramycin, amoxicillin-clavulanic-acid, ampicillin-sulbactam, ceftriaxone, cefepime |
| KPNEU | 574.20 | 24 | 0 | 0 | 16 | 6 | 7/7 | 0 | none |
| ECOLI | 562.42835 | 22 | 0 | 0 | 3 | 17 | 1/4 | 1 | ciprofloxacin, levofloxacin, cefepime |
| ECOLI | 562.145336 | 18 | 0 | 0 | 5 | 5 | 1/2 | 0 | tetracycline, ciprofloxacin, ampicillin-sulbactam, ceftriaxone, levofloxacin |
| ECOLI | 562.42842 | 22 | **1** | 0 | 3 | 17 | 1/1 | 2 | ciprofloxacin, levofloxacin, cefepime |
| SAUR | 1280.16760 | 11 | 0 | 0 | 3 | 8 | 3/4 | 1 | none |
| SAUR | 1280.51740 | 3 | 0 | 0 | 1 | 2 | 0/1 | 0 | cefoxitin, oxacillin, levofloxacin |
| SAUR | 1280.25977 | 8 | 0 | 0 | 0 | 8 | 4/4 | 0 | levofloxacin |
| PAER | 287.5972 | 13 | 0 | 0 | 0 | 13 | 4/6 | 1 | none |
| PAER | 287.47321 | 7 | 0 | 0 | 0 | 6 | 3/5 | 0 | none |
| PAER | 287.6329 | 13 | 0 | 0 | 3 | 10 | 5/5 | 0 | none |
| ABAU | 470.7392 | 17 | 0 | 0 | 6 | 11 | 7/7 | 0 | trimethoprim-sulfamethoxazole |
| ABAU | 470.7513 | 13 | 0 | 0 | 9 | 3 | 3/5 | 0 | trimethoprim-sulfamethoxazole, tetracycline, levofloxacin, ciprofloxacin, gentamicin |
| ABAU | 470.7393 | 17 | 0 | 0 | 6 | 11 | 6/6 | 0 | trimethoprim-sulfamethoxazole |

Lab S/I/R against the call, for all 233 lab results:

| lab | likely_active | likely_inactive | uncertain | total |
| --- | ---: | ---: | ---: | ---: |
| R | **1 (VME)** | 53 | 52 | 106 |
| S | 28 | 0 (ME) | 66 | 94 |
| I | 1 | 2 | 14 | 17 |
| no S/I/R (no CLSI breakpoint, or interval straddles one) | 2 | 0 | 14 | 16 |

Lab S/I/R against the `prob_works` tier (`none` = no calibration: no CLSI breakpoint):

| lab | very likely works (>= 0.90) | probably works (0.70-0.90) | uncertain | probably fails (0.10-0.30) | very likely fails (<= 0.10) | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 1 | 4 | 13 | 19 | 62 | 7 |
| S | 58 | 12 | 8 | 0 | 0 | 16 |
| I | 3 | 1 | 3 | 4 | 6 | 0 |
| no S/I/R | 6 | 1 | 0 | 0 | 0 | 9 |

EA was 56/68 on exact lab MICs. Every lab result with its predicted MIC, band, call,
`prob_works` and `prob_tier` is in `demo_table.csv` / `demo_table.md`.

These counts come from 15 genomes and say nothing reliable about error rates.

**The errors, explained:**

- **VME, ECOLI 562.42842 x cefepime.** The lab MIC is >32 (R). The prediction was 0.25 mg/L
  (band 0.03-2), so the call was likely_active, with `prob_works` 0.95. AMRFinderPlus finds only
  blaCMY-2, blaTEM-1, blaEC and efflux genes, and CMY-2 alone does not usually give cefepime
  resistance. The likely cause is a mechanism outside the known-gene features, such as porin
  loss. Genome 562.42835 (lab cefepime 8, SDD/I) has the same pattern. Known-gene features cannot
  see this.
- **Lab R told works (P >= 0.70) but not called active (4):** ECOLI 562.42835 and 562.42842 x
  piperacillin-tazobactam (lab >128, predicted 4, P 0.85, call uncertain); SAUR 1280.16760 x
  daptomycin (lab 2, predicted 1, P 0.86, uncertain); PAER 287.5972 x ceftazidime (lab 128,
  predicted 4, P 0.71, uncertain). ECOLI piperacillin-tazobactam and SAUR daptomycin are also
  among the worst pairs out of fold (`results/report.md`, 'Lab R told works').
- The earlier `aft_known` bundle's ME, SAUR 1280.16760 x clindamycin (lab <=0.25, predicted 64),
  is gone: the `aft_b2_select` bundle predicts 0.25 mg/L, call uncertain, P 0.83.

All 15 reports have `in_range: false` and `nearest_training_distance: null`. The imported release
ships no assemblies, so there are no training Mash sketches. Per the call logic, every call should
be treated as low confidence.

## Feature parity: NCBI release rules at prediction time

The release's `known_amr.parquet` was built with develop's rules from NCBI's `AMR_genotypes`. It
was not built with our `features/known_amr.py`. Those rules are:

- `MISTRANSLATION` hits are skipped.
- `=POINT` becomes `point_`.
- Only `bla` alleles collapse to their family, except blaKPC/NDM/VIM/IMP/GES/OXA.
- `n_class_` counts every hit per class, using the AMRFinderPlus DB tables. A multi-class `A/B`
  hit counts once in each class.

`predict/release_features.py` turns an AMRFinderPlus TSV into exactly the bundle's columns with
these rules. A bundle trained on an imported release records `feature_naming: ncbi_release` in
`manifest.json` and carries `models/<SP>/feature_spec.json`, which holds the rules plus the
symbol-to-class table from AMRFinderPlus DB 2026-08-07.1. `predict/pipeline.py` uses that spec
and refuses to load such a bundle without it.

- **(a) NCBI strings to release rows.** 200 release genomes, 40 per species, were rebuilt from
  their NCBI `AMR_genotypes`. **200/200 matched exactly** on every column. A test also matches
  develop's own `NcbiKnownAmrBuilder` on randomized genotype strings.
  - Files: `parity_a_ncbi_strings.csv`
  - Test: `tests/predict/test_release_features.py`
- **(b) Our AMRFinderPlus run vs the release row**, on 5 training genomes. Their BV-BRC assemblies
  were downloaded and AMRFinderPlus 4.2.7 was run on them. This only checks features; no
  prediction or metric was made on these genomes. File: `parity_b_amrfinder_vs_release.csv`.

| species | genome | gene/point columns agree | all non-zero columns agree | release only | ours only | count differences |
| --- | --- | --- | --- | --- | --- | --- |
| KPNEU | 1284787.3 | 17/17 | 26/26 | - | - | - |
| ECOLI | 1045010.64 | 10/11 | 17/19 | - | gene_emrd | n_class_efflux 1->2 |
| SAUR | 1280.15878 | 16/16 | 23/23 | - | - | - |
| PAER | 287.1000 | 16/19 | 21/26 | gene_aph_3, gene_cml (both NCBI `=HMM` hits) | point_nalc_g71e | n_class_aminoglycoside 4->3, n_class_beta_lactam 4->5 |
| ABAU | 1409926.3 | 18/18 | 27/27 | - | - | - |

The remaining differences come from the AMRFinderPlus software, database and assembly, not from
the converter. NCBI ran an older AMRFinderPlus/DB on the NCBI assembly; we ran 4.2.7 / 2026-08-07.1
on the BV-BRC assembly. The converter itself reproduces NCBI strings exactly (check a).

## Context: cross-validation (out-of-fold) for the same bundle, `aft_b2_select`

Out-of-fold CV rows of the main model (`aft_b2_select`; each fold's candidate chosen on the
other folds, so these rows score the selection procedure, not the single shipped bundle).
Lab S/I/R re-derived under CLSI 2024. Provisional: breakpoints are unverified outside the 15
checked Enterobacterales drugs, and picking among a few band and model settings by out-of-fold
score adds a little optimism. Columns use drugs with >= 30 lab S and >= 30 lab I/R. Danger first.

| species | drugs | danger: lab R called likely_active (of all lab R) | confident (P >= 0.9 or <= 0.1) | right when confident | call gives an answer | call right | straight accuracy (forced at 0.5) | forced danger (lab R with P >= 0.5) | median EA | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KPNEU | 21 | 0.61% (227/37,021) | 64.9% | 95.8% | 49.3% | 95.4% | 88.0% | 7.0% | 69.3% | 1.2% |
| ECOLI | 21 | 0.91% (310/34,187) | 82.3% | 98.0% | 44.2% | 95.8% | 94.1% | 6.5% | 77.8% | 0.2% |
| SAUR | 9 | 0.20% (3/1,464) | 73.9% | 95.8% | 18.2% | 95.5% | 90.7% | 26.5% | 93.7% | 1.8% |
| PAER | 10 | 0.03% (1/3,735) | 32.3% | 96.4% | 26.6% | 97.9% | 79.2% | 8.9% | 59.9% | 1.8% |
| ABAU | 15 | 0.33% (24/7,191) | 67.8% | 96.0% | 40.6% | 95.0% | 88.1% | 5.8% | 76.1% | 3.6% |

Shipped-bundle view (likely-active calls withheld where the shipped gate is closed; pooled over
all lab-R CV rows of all pairs): call VME 492 / 86,176 (0.6 %); 29.0 % of lab S called likely
active; 50 of 97 shipped gates open. Per pair and per fold (calling-fold call VME) see
`runs/hackathon5/results/report.md`.

## Limitations

- **Provisional local data for 4 species.** ECOLI, SAUR, PAER and ABAU come from a local build
  (`runs/pd5_build/release`, status PROVISIONAL). It was made with develop's unmodified
  `run_hackathon_data` from NCBI Pathogen Detection metadata and is not an S3 release. Only the
  KPNEU rows are exactly Hub's release.
- **Provisional splits.** The local build's splits are not frozen. They are used as built, never
  edited.
- **Breakpoints verified for 15 Enterobacterales drugs only.** Those CLSI breakpoints were checked
  by Hub (develop ddd76ef) and match ours. Every other breakpoint, including all SAUR, PAER and
  ABAU ones, was entered from memory and is unverified. All S/I/R, call and VME figures outside
  those 15 are provisional.
- **Known-gene features only.** There are no unitigs and no k-mers in this bundle. Resistance
  outside AMRFinderPlus's catalogue is invisible to the model, for example porin loss, efflux
  up-regulation, or inducible resistance that depends on a lab test. See the VME above.
- **No training sketches.** `nearest_training_distance` is null and `in_range` is false for every
  genome. The species reference sketches still drive species ID and QC.
- **Tiny check.** 15 genomes from BV-BRC, mostly from the same few submissions, with lab results
  as recorded in BV-BRC. This is not a validation and no error rate should be read from it.
- **Feature-version drift.** At prediction time we run AMRFinderPlus 4.2.7 with DB 2026-08-07.1,
  while the training features came from NCBI's runs (see parity b). Small differences in
  HMM-only hits and new point mutations are expected.
