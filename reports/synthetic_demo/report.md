# genome2mic results report

> **SYNTHETIC DATA.** `data/raw/SYNTHETIC_DATA.md` is present: every genome, lab result, model and number in this report was produced from simulated data. These figures demonstrate that the pipeline runs end to end. They say nothing about real-world performance and must not be quoted as such.

> **Scope note.** These are predictions of in-vitro susceptibility, not prescribing advice. Dose, route, and final drug choice depend on PK/PD, infection site, renal function, allergies, and other patient factors, and remain with the clinician. Confirm with standard AST.

Generated 2026-10-03 22:15 UTC from `/Users/bensonzhang/conductor/workspaces/amr-backend/taipei/runs/synthetic`. Predictions of in-vitro MIC (mg/L) per species x drug; the system ranks, a clinician decides.

## How to read this report

- **VME** (very major error: predicted S, lab R) is listed first in every table and figure; it is the error that would harm a patient. Its denominator is the number of lab-R rows.
- **ME** (major error: predicted R, lab S) uses the lab-S rows as denominator; **minor error** has exactly one side I; **CA** is same S/I/R; **EA** is within +/-1 doubling step of an exact lab MIC; **exact agreement** is the same doubling step.
- Rates are shown as percentages with their denominator `n`. Band coverage is the share of lab MICs inside the 90 % conformal band; band width is in doubling steps.
- Targets quoted on the figures and below -- EA >= 90 %, CA >= 90 %, VME <= 1.5 %, ME <= 3 % -- are figures commonly used in AST device evaluation. They are listed as reference points only; this report does not claim they were met, and they are not regulatory thresholds.
- `test set` = lineage-held-out genomes scored once at the end; `CV` = out-of-fold predictions on the train split; external and leave-one-lineage-out (LOLO) sets are listed separately.

_All numbers below come from synthetic data (see the banner above)._

## Headline: main model on the test set (VME first)

| Species | Drug | Model | VME % (of lab R) | ME % (of lab S) | CA % | EA % (exact MICs) | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ECOLI | ceftriaxone | aft_known_unitig | 0.0% (n=20) | 0.0% (n=20) | 100.0% (n=40) | 77.5% (n=19) | 40 |
| ECOLI | ciprofloxacin | aft_known_unitig | 0.0% (n=5) | 0.0% (n=34) | 97.5% (n=40) | 85.0% (n=32) | 40 |
| ECOLI | meropenem | aft_known_unitig | 0.0% (n=7) | 0.0% (n=30) | 92.7% (n=41) | 97.6% (n=16) | 41 |
| ECOLI | piperacillin-tazobactam | aft_known_unitig | 0.0% (n=12) | 0.0% (n=23) | 97.2% (n=36) | 83.3% (n=24) | 36 |
| KPNEU | ceftriaxone | aft_known_unitig | 0.0% (n=75) | 0.0% (n=16) | 100.0% (n=91) | 96.7% (n=22) | 91 |
| KPNEU | ciprofloxacin | aft_known_unitig | 0.0% (n=65) | 0.0% (n=24) | 96.7% (n=91) | 96.7% (n=52) | 91 |
| KPNEU | gentamicin | aft_known_unitig | 0.0% (n=17) | 0.0% (n=36) | 84.1% (n=63) | 96.8% (n=46) | 63 |
| KPNEU | meropenem | aft_known_unitig | 0.0% (n=60) | 0.0% (n=30) | 75.5% (n=94) | 93.6% (n=44) | 94 |
| KPNEU | piperacillin-tazobactam | aft_known_unitig | 0.0% (n=45) | 50.0% (n=12) | 89.5% (n=57) | 91.2% (n=16) | 57 |

## Data

### Label counts (species x drug, all lab-measured labels)

| species | drug | n | n_R | n_S | n_I | n_exact | n_censored | n_distinct_mic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ECOLI | ceftriaxone | 239 | 129 | 110 | 0 | 82 | 157 | 10 |
| ECOLI | ciprofloxacin | 239 | 100 | 130 | 9 | 181 | 58 | 11 |
| ECOLI | gentamicin | 57 | 17 | 39 | 1 | 36 | 21 | 7 |
| ECOLI | meropenem | 248 | 60 | 167 | 21 | 93 | 155 | 10 |
| ECOLI | piperacillin-tazobactam | 213 | 85 | 125 | 3 | 136 | 77 | 10 |
| KPNEU | ceftriaxone | 572 | 315 | 257 | 0 | 179 | 393 | 10 |
| KPNEU | ciprofloxacin | 591 | 231 | 331 | 29 | 402 | 189 | 11 |
| KPNEU | gentamicin | 357 | 64 | 272 | 21 | 229 | 128 | 8 |
| KPNEU | meropenem | 606 | 192 | 384 | 30 | 249 | 357 | 12 |
| KPNEU | piperacillin-tazobactam | 365 | 145 | 212 | 8 | 209 | 156 | 10 |

### Pairs kept (>= 50 non-susceptible, >= 50 susceptible, >= 4 distinct MIC levels)

| species | drug | n | n_nonsusceptible | n_susceptible | n_distinct_mic |
| --- | --- | --- | --- | --- | --- |
| ECOLI | ceftriaxone | 239 | 129 | 110 | 10 |
| ECOLI | ciprofloxacin | 239 | 109 | 130 | 11 |
| ECOLI | meropenem | 248 | 81 | 167 | 10 |
| ECOLI | piperacillin-tazobactam | 213 | 88 | 125 | 10 |
| KPNEU | ceftriaxone | 572 | 315 | 257 | 10 |
| KPNEU | ciprofloxacin | 591 | 260 | 331 | 11 |
| KPNEU | gentamicin | 357 | 85 | 272 | 8 |
| KPNEU | meropenem | 606 | 222 | 384 | 12 |
| KPNEU | piperacillin-tazobactam | 365 | 153 | 212 | 10 |

### Drop logs (every filter, with its count)

#### Stage `evaluate` — 11224 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| evaluate | pred_mic null: categorical metrics only | 3288 | models: b0_resfinder |
| evaluate | lab_lower/lab_upper null: excluded from MIC metrics | 0 | — |
| evaluate | lab interval (0, inf): excluded from MIC metrics | 0 | — |
| evaluate | pred_sir or lab_sir null: excluded from categorical metrics | 0 | — |
| evaluate | lab_sir I: excluded from AUROC | 680 | — |
| evaluate | band_low/band_high null: excluded from band metrics | 3288 | models: b0_resfinder |
| evaluate | nearest_distance null: excluded from distance bins | 3288 | — |
| evaluate | nearest_distance outside bins: excluded from distance bins | 0 | — |
| evaluate | pred_mic null: categorical metrics only | 0 | — |
| evaluate | lab_lower/lab_upper null: excluded from MIC metrics | 0 | — |
| evaluate | lab interval (0, inf): excluded from MIC metrics | 0 | — |
| evaluate | pred_sir or lab_sir null: excluded from categorical metrics | 0 | — |
| evaluate | lab_sir I: excluded from AUROC | 680 | — |
| evaluate | band_low/band_high null: excluded from band metrics | 0 | — |

#### Stage `ingest` — 1038 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| ingest | missing genome_id | 0 | — |
| ingest | evidence != Laboratory Method | 183 | Computational Prediction (183) |
| ingest | unknown drug | 20 | Nitrofurantoin (7), Cefiderocol (5), Ceftolozane/tazobactam (5), Fosfomycin (3) |
| ingest | unknown species | 0 | — |
| ingest | unknown laboratory typing method | 240 | Vitek 2 (142), automated system (98) |
| ingest | unknown measurement sign | 0 | — |
| ingest | unparseable or non-positive MIC value | 0 | — |
| ingest | unknown MIC unit | 0 | — |
| ingest | no MIC value and no S/I/R | 0 | — |
| ingest | unknown S/I/R value on S/I/R-only row | 0 | — |
| ingest | null or unknown standard on S/I/R-only row | 111 | — |
| ingest | no breakpoint for species x drug x standard on S/I/R-only row | 0 | — |
| ingest | 'I' reported but the standard has no I category (S == R) | 0 | — |
| ingest | biosample de-dup: rows re-keyed to the BV-BRC genome_id (count, not a drop) | 0 | — |
| ingest | duplicate (genome_id, drug): extra rows merged into one interval | 418 | 411 pairs |
| ingest | duplicate (genome_id, drug): S vs R conflict, pair dropped | 66 | 33 pairs: 573.1000/ciprofloxacin, 573.1001/piperacillin-tazobactam, 573.1028/gentamicin, 573.1035/meropenem, 573.1041/ciprofloxacin, ... 28 more |
| ingest | duplicate (genome_id, drug): intervals more than 1 step apart, pair dropped | 0 | — |

#### Stage `known_amr` — 152 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| known_amr | qc_fail | 40 | qc_pass == False; excluded from known_amr |
| known_amr | amrfinder.tsv missing -> features zero-filled (rows kept) | 0 | — |
| known_amr | amrfinder rows with Type != AMR (STRESS/VIRULENCE) | 112 | — |
| known_amr | amrfinder AMR rows with Subtype not in {AMR, POINT} | 0 | — |
| known_amr | amrfinder AMR rows without a symbol | 0 | — |
| known_amr | amrfinder AMR hits without a Class (kept as gene_/point_, not counted in n_class_) | 0 | — |

#### Stage `lineages` — 40 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| lineages | labelled_genome_without_qc_record | 0 | — |
| lineages | qc_fail | 40 | qc_pass is not True |
| lineages | species_unknown | 0 | — |
| lineages | species_not_in_config | 0 | — |
| lineages | fasta_missing | 0 | ECOLI:  |
| lineages | sketch_too_small | 0 | ECOLI: fewer than s=1000 distinct 21-mers |
| lineages | non_acgt_window | 0 | ECOLI: k-mer windows skipped for non-ACGT bases while sketching (window count, not genomes) |
| lineages | fasta_missing | 0 | KPNEU:  |
| lineages | sketch_too_small | 0 | KPNEU: fewer than s=1000 distinct 21-mers |
| lineages | non_acgt_window | 0 | KPNEU: k-mer windows skipped for non-ACGT bases while sketching (window count, not genomes) |

#### Stage `qc` — 80 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| qc | missing_fasta | 0 | in labels.parquet but no FASTA under data/raw/genomes |
| qc | too_fragmented | 10 | n_contigs > 500 |
| qc | wrong_size | 10 | total_length outside expected_genome_size +/- size_tolerance |
| qc | species_mismatch | 10 | mash_species != label species, or no Mash result for a labelled genome |
| qc | too_distant | 10 | mash_distance > 0.05 from every reference, or no Mash result |
| qc | qc_fail_any | 40 | distinct genomes failing >= 1 rule (per-rule counts above overlap); excluded from training and the unitig build |

#### Stage `splits` — 145 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| splits | lineage_rows_species_not_in_config | 0 | — |
| splits | label_rows_for_genomes_without_lineage | 145 | genome failed QC or has no lineage row; excluded from the test-set R/S check |
| splits | kept_pairs_for_species_without_lineages | 0 | — |

#### Stage `synth` — 808 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| synth | planted expected_drop: evidence == Computational Prediction | 183 | — |
| synth | planted expected_drop: S/I/R-only row with blank standard | 89 | — |
| synth | planted expected_drop: unknown antibiotic name | 20 | — |
| synth | planted expected_drop: unknown typing method | 220 | — |
| synth | planted expected_drop: conflicting cross-source duplicate (genome x drug pairs) | 37 | — |
| synth | planted: within-source duplicate rows (consistent) (not a drop; ingest should merge) | 68 | — |
| synth | planted: biosamples present in both sources (not a drop; ingest should merge) | 150 | — |
| synth | planted expected_drop: QC-fail genome (too_many_contigs) | 10 | — |
| synth | planted expected_drop: QC-fail genome (wrong_size) | 10 | — |
| synth | planted expected_drop: QC-fail genome (wrong_species) | 10 | — |
| synth | planted expected_drop: QC-fail genome (too_distant) | 10 | — |
| synth | planted expected_drop: species x drug pairs failing the 50/50 inclusion rule | 1 | ECOLI x gentamicin |

#### Stage `train` — 1697160 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| train | label_without_split_row | 11 | ECOLI x ceftriaxone |
| train | label_without_qc_row | 0 | ECOLI x ceftriaxone |
| train | qc_fail | 0 | ECOLI x ceftriaxone |
| train | label_without_lineage_row | 0 | ECOLI x ceftriaxone |
| train | label_without_known_amr_row | 0 | ECOLI x ceftriaxone |
| train | label_without_unitig_row | 0 | ECOLI x ceftriaxone |
| train | known_rare_feature | 3 | present in < 5 of 156 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 156 training genomes |
| train | unitig_below_min_freq | 114 | present in < 1% of 156 training rows |
| train | unitig_above_max_freq | 567 | present in > 99% of 156 training rows |
| train | unitig_beyond_top_k | 14268 | ranked below top_k=2000 by \|corr\| with y_point |
| train | censored row (B2 trains on exact MICs only) | 105 | 51 exact rows kept of 156 |
| train | known_rare_feature | 3 | present in < 5 of 164 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 164 training genomes |
| train | unitig_below_min_freq | 78 | present in < 1% of 164 training rows |
| train | unitig_above_max_freq | 468 | present in > 99% of 164 training rows |
| train | unitig_beyond_top_k | 14403 | ranked below top_k=2000 by \|corr\| with y_point |
| train | censored row (B2 trains on exact MICs only) | 114 | 50 exact rows kept of 164 |
| train | known_rare_feature | 3 | present in < 5 of 112 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 112 training genomes |
| train | unitig_below_min_freq | 510 | present in < 1% of 112 training rows |
| train | unitig_above_max_freq | 1045 | present in > 99% of 112 training rows |
| train | unitig_beyond_top_k | 13394 | ranked below top_k=2000 by \|corr\| with y_point |
| train | censored row (B2 trains on exact MICs only) | 76 | 36 exact rows kept of 112 |
| train | known_rare_feature | 3 | present in < 5 of 168 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 168 training genomes |
| train | unitig_below_min_freq | 81 | present in < 1% of 168 training rows |
| train | unitig_above_max_freq | 367 | present in > 99% of 168 training rows |
| train | unitig_beyond_top_k | 14501 | ranked below top_k=2000 by \|corr\| with y_point |
| train | censored row (B2 trains on exact MICs only) | 112 | 56 exact rows kept of 168 |
| train | known_rare_feature | 3 | present in < 5 of 152 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 152 training genomes |
| train | unitig_below_min_freq | 167 | present in < 1% of 152 training rows |
| train | unitig_above_max_freq | 424 | present in > 99% of 152 training rows |
| train | unitig_beyond_top_k | 14358 | ranked below top_k=2000 by \|corr\| with y_point |
| train | censored row (B2 trains on exact MICs only) | 109 | 43 exact rows kept of 152 |
| train | censored row (conformal residuals use exact MICs only) | 129 | — |
| train | missing prediction (conformal residuals) | 0 | — |
| train | censored row (conformal residuals use exact MICs only) | 129 | — |
| train | missing prediction (conformal residuals) | 0 | — |
| train | censored row (conformal residuals use exact MICs only) | 129 | — |
| train | missing prediction (conformal residuals) | 0 | — |
| train | censored row (conformal residuals use exact MICs only) | 129 | — |
| train | missing prediction (conformal residuals) | 0 | — |
| train | known_rare_feature | 3 | present in < 5 of 188 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 188 training genomes |
| train | unitig_below_min_freq | 19 | present in < 1% of 188 training rows |
| train | unitig_above_max_freq | 66 | present in > 99% of 188 training rows |
| train | unitig_beyond_top_k | 14864 | ranked below top_k=2000 by \|corr\| with y_point |
| train | censored row (B2 trains on exact MICs only) | 129 | 59 exact rows kept of 188 |
| train | known_rare_feature | 3 | present in < 5 of 150 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 150 training genomes |
| train | unitig_below_min_freq | 307 | present in < 1% of 150 training rows |
| train | unitig_above_max_freq | 340 | present in > 99% of 150 training rows |
| train | unitig_beyond_top_k | 14302 | ranked below top_k=2000 by \|corr\| with y_point |
| train | censored row (B2 trains on exact MICs only) | 105 | 45 exact rows kept of 150 |
| train | known_rare_feature | 3 | present in < 5 of 158 training genomes |
| train | known_rare_feature | 3 | present in < 5 of 158 training genomes |
| train | unitig_below_min_freq | 167 | present in < 1% of 158 training rows |
| train | unitig_above_max_freq | 268 | present in > 99% of 158 training rows |

_507 more rows not shown._

#### Stage `unitigs` — 365431 rows dropped

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| unitigs | no_qc_row | 0 | species=ECOLI |
| unitigs | qc_fail | 0 | species=ECOLI |
| unitigs | fasta_missing | 0 | species=ECOLI |
| unitigs | kmer_below_min_freq | 108308 | present in < 1% of 238 training genomes |
| unitigs | kmer_above_max_freq | 2977 | present in > 99% of 238 training genomes |
| unitigs | no_qc_row | 0 | species=KPNEU |
| unitigs | qc_fail | 0 | species=KPNEU |
| unitigs | fasta_missing | 0 | species=KPNEU |
| unitigs | kmer_below_min_freq | 252337 | present in < 1% of 561 training genomes |
| unitigs | kmer_above_max_freq | 1809 | present in > 99% of 561 training genomes |

## Results by species and drug

### ECOLI x ceftriaxone

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 1 mg/L, R if MIC > 2 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 40 | 0.0% (n=20) | 0.0% (n=20) | 0.0% (n=40) | 100.0% (n=40) | — | — | — | — | — |
| b1_lookup | 40 | 0.0% (n=20) | 5.0% (n=20) | 0.0% (n=40) | 97.5% (n=40) | 77.5% (n=19) | 40.0% (n=19) | 0.989 | 100.0% (n=40) | 14.00 |
| b2_xgb_steps | 40 | 0.0% (n=20) | 0.0% (n=20) | 0.0% (n=40) | 100.0% (n=40) | 95.0% (n=19) | 32.5% (n=19) | 1.000 | 100.0% (n=40) | 12.00 |
| aft_known | 40 | 0.0% (n=20) | 0.0% (n=20) | 0.0% (n=40) | 100.0% (n=40) | 100.0% (n=19) | 60.0% (n=19) | 1.000 | 100.0% (n=40) | 6.00 |
| aft_known_unitig | 40 | 0.0% (n=20) | 0.0% (n=20) | 0.0% (n=40) | 100.0% (n=40) | 77.5% (n=19) | 57.5% (n=19) | 1.000 | 100.0% (n=40) | 10.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 188 | 0.0% (n=104) | 0.0% (n=84) | 0.0% (n=188) | 100.0% (n=188) | — | — | — | — | — |
| b1_lookup | 188 | 50.0% (n=104) | 8.3% (n=84) | 0.0% (n=188) | 68.6% (n=188) | 54.8% (n=59) | 41.0% (n=59) | 0.875 | 84.6% (n=188) | 14.00 |
| b2_xgb_steps | 188 | 5.8% (n=104) | 14.3% (n=84) | 0.0% (n=188) | 90.4% (n=188) | 74.5% (n=59) | 32.4% (n=59) | 0.934 | 94.7% (n=188) | 12.00 |
| aft_known | 188 | 0.0% (n=104) | 0.0% (n=84) | 1.6% (n=188) | 98.4% (n=188) | 88.3% (n=59) | 67.6% (n=59) | 1.000 | 98.9% (n=188) | 6.00 |
| aft_known_unitig | 188 | 9.6% (n=104) | 0.0% (n=84) | 1.6% (n=188) | 93.1% (n=188) | 83.5% (n=59) | 57.4% (n=59) | 0.990 | 97.3% (n=188) | 10.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (ECOLI_ML_001)** (`lolo_ECOLI_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 38 | 80.0% (n=35) | 0.0% (n=3) | 0.0% (n=38) | 26.3% (n=38) | 21.1% (n=14) | 13.2% (n=14) | 0.600 | 100.0% (n=38) | 14.00 |
| b2_xgb_steps | 38 | 8.6% (n=35) | 0.0% (n=3) | 0.0% (n=38) | 92.1% (n=38) | 52.6% (n=14) | 26.3% (n=14) | 0.971 | 97.4% (n=38) | 12.00 |
| aft_known | 38 | 0.0% (n=35) | 0.0% (n=3) | 0.0% (n=38) | 100.0% (n=38) | 84.2% (n=14) | 71.1% (n=14) | 1.000 | 100.0% (n=38) | 6.00 |
| aft_known_unitig | 38 | 11.4% (n=35) | 0.0% (n=3) | 0.0% (n=38) | 89.5% (n=38) | 65.8% (n=14) | 57.9% (n=14) | 0.995 | 100.0% (n=38) | 10.00 |

**leave-one-lineage-out (ECOLI_ML_002)** (`lolo_ECOLI_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 30 | 0.0% (n=24) | 83.3% (n=6) | 0.0% (n=30) | 83.3% (n=30) | 30.0% (n=8) | 23.3% (n=8) | 0.670 | 100.0% (n=30) | 14.00 |
| b2_xgb_steps | 30 | 0.0% (n=24) | 16.7% (n=6) | 0.0% (n=30) | 96.7% (n=30) | 80.0% (n=8) | 36.7% (n=8) | 0.969 | 100.0% (n=30) | 12.00 |
| aft_known | 30 | 0.0% (n=24) | 0.0% (n=6) | 0.0% (n=30) | 100.0% (n=30) | 93.3% (n=8) | 80.0% (n=8) | 1.000 | 100.0% (n=30) | 6.00 |
| aft_known_unitig | 30 | 4.2% (n=24) | 0.0% (n=6) | 6.7% (n=30) | 90.0% (n=30) | 83.3% (n=8) | 76.7% (n=8) | 1.000 | 100.0% (n=30) | 10.00 |

![VME and ME by model, test set](figures/vme_me_by_model_ECOLI_ceftriaxone.png)

![EA and CA by model, test set](figures/ea_ca_by_model_ECOLI_ceftriaxone.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_ECOLI_ceftriaxone.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_ECOLI_ceftriaxone.png)

### ECOLI x ciprofloxacin

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 0.25 mg/L, R if MIC > 0.5 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 40 | 80.0% (n=5) | 11.8% (n=34) | 2.5% (n=40) | 77.5% (n=40) | — | — | — | — | — |
| b1_lookup | 40 | 0.0% (n=5) | 0.0% (n=34) | 22.5% (n=40) | 77.5% (n=40) | 70.0% (n=32) | 37.5% (n=32) | 0.976 | 100.0% (n=40) | 12.00 |
| b2_xgb_steps | 40 | 0.0% (n=5) | 0.0% (n=34) | 2.5% (n=40) | 97.5% (n=40) | 85.0% (n=32) | 55.0% (n=32) | 1.000 | 90.0% (n=40) | 4.00 |
| aft_known | 40 | 0.0% (n=5) | 0.0% (n=34) | 5.0% (n=40) | 95.0% (n=40) | 82.5% (n=32) | 42.5% (n=32) | 1.000 | 90.0% (n=40) | 4.00 |
| aft_known_unitig | 40 | 0.0% (n=5) | 0.0% (n=34) | 2.5% (n=40) | 97.5% (n=40) | 85.0% (n=32) | 55.0% (n=32) | 1.000 | 100.0% (n=40) | 4.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 188 | 53.8% (n=91) | 12.4% (n=89) | 4.3% (n=188) | 63.8% (n=188) | — | — | — | — | — |
| b1_lookup | 188 | 65.9% (n=91) | 19.1% (n=89) | 6.4% (n=188) | 52.7% (n=188) | 40.4% (n=142) | 22.9% (n=142) | 0.652 | 88.3% (n=188) | 12.00 |
| b2_xgb_steps | 188 | 3.3% (n=91) | 4.5% (n=89) | 4.8% (n=188) | 91.5% (n=188) | 79.3% (n=142) | 35.6% (n=142) | 0.984 | 92.0% (n=188) | 4.00 |
| aft_known | 188 | 3.3% (n=91) | 0.0% (n=89) | 4.3% (n=188) | 94.1% (n=188) | 81.4% (n=142) | 41.5% (n=142) | 0.993 | 95.2% (n=188) | 4.00 |
| aft_known_unitig | 188 | 3.3% (n=91) | 0.0% (n=89) | 4.8% (n=188) | 93.6% (n=188) | 79.8% (n=142) | 34.6% (n=142) | 0.999 | 96.3% (n=188) | 4.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (ECOLI_ML_001)** (`lolo_ECOLI_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 40 | 79.4% (n=34) | 0.0% (n=2) | 10.0% (n=40) | 22.5% (n=40) | 27.5% (n=30) | 7.5% (n=30) | 0.801 | 100.0% (n=40) | 12.00 |
| b2_xgb_steps | 40 | 2.9% (n=34) | 0.0% (n=2) | 10.0% (n=40) | 87.5% (n=40) | 82.5% (n=30) | 30.0% (n=30) | 1.000 | 95.0% (n=40) | 4.00 |
| aft_known | 40 | 0.0% (n=34) | 0.0% (n=2) | 10.0% (n=40) | 90.0% (n=40) | 87.5% (n=30) | 40.0% (n=30) | 1.000 | 100.0% (n=40) | 4.00 |
| aft_known_unitig | 40 | 2.9% (n=34) | 0.0% (n=2) | 10.0% (n=40) | 87.5% (n=40) | 95.0% (n=30) | 40.0% (n=30) | 1.000 | 100.0% (n=40) | 4.00 |

**leave-one-lineage-out (ECOLI_ML_002)** (`lolo_ECOLI_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 32 | 82.1% (n=28) | 0.0% (n=3) | 3.1% (n=32) | 25.0% (n=32) | 21.9% (n=20) | 3.1% (n=20) | 1.000 | 100.0% (n=32) | 12.00 |
| b2_xgb_steps | 32 | 0.0% (n=28) | 0.0% (n=3) | 6.2% (n=32) | 93.8% (n=32) | 62.5% (n=20) | 28.1% (n=20) | 1.000 | 87.5% (n=32) | 4.00 |
| aft_known | 32 | 0.0% (n=28) | 0.0% (n=3) | 3.1% (n=32) | 96.9% (n=32) | 75.0% (n=20) | 37.5% (n=20) | 1.000 | 93.8% (n=32) | 4.00 |
| aft_known_unitig | 32 | 0.0% (n=28) | 0.0% (n=3) | 3.1% (n=32) | 96.9% (n=32) | 90.6% (n=20) | 40.6% (n=20) | 1.000 | 100.0% (n=32) | 4.00 |

![VME and ME by model, test set](figures/vme_me_by_model_ECOLI_ciprofloxacin.png)

![EA and CA by model, test set](figures/ea_ca_by_model_ECOLI_ciprofloxacin.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_ECOLI_ciprofloxacin.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_ECOLI_ciprofloxacin.png)

### ECOLI x meropenem

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 2 mg/L, R if MIC > 8 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 41 | 0.0% (n=7) | 6.7% (n=30) | 9.8% (n=41) | 85.4% (n=41) | — | — | — | — | — |
| b1_lookup | 41 | 57.1% (n=7) | 0.0% (n=30) | 12.2% (n=41) | 78.0% (n=41) | 85.4% (n=16) | 65.9% (n=16) | 0.924 | 100.0% (n=41) | 18.00 |
| b2_xgb_steps | 41 | 0.0% (n=7) | 0.0% (n=30) | 7.3% (n=41) | 92.7% (n=41) | 100.0% (n=16) | 24.4% (n=16) | 1.000 | 100.0% (n=41) | 14.00 |
| aft_known | 41 | 0.0% (n=7) | 0.0% (n=30) | 14.6% (n=41) | 85.4% (n=41) | 97.6% (n=16) | 70.7% (n=16) | 1.000 | 100.0% (n=41) | 6.00 |
| aft_known_unitig | 41 | 0.0% (n=7) | 0.0% (n=30) | 7.3% (n=41) | 92.7% (n=41) | 97.6% (n=16) | 73.2% (n=16) | 1.000 | 100.0% (n=41) | 6.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 196 | 2.0% (n=51) | 4.7% (n=128) | 8.7% (n=196) | 87.8% (n=196) | — | — | — | — | — |
| b1_lookup | 196 | 92.2% (n=51) | 0.0% (n=128) | 9.7% (n=196) | 66.3% (n=196) | 63.8% (n=75) | 39.8% (n=75) | 0.545 | 89.3% (n=196) | 18.00 |
| b2_xgb_steps | 196 | 7.8% (n=51) | 0.0% (n=128) | 18.9% (n=196) | 79.1% (n=196) | 80.6% (n=75) | 27.6% (n=75) | 0.947 | 98.0% (n=196) | 14.00 |
| aft_known | 196 | 13.7% (n=51) | 0.0% (n=128) | 20.9% (n=196) | 75.5% (n=196) | 77.6% (n=75) | 55.6% (n=75) | 0.981 | 95.4% (n=196) | 6.00 |
| aft_known_unitig | 196 | 11.8% (n=51) | 0.0% (n=128) | 24.5% (n=196) | 72.4% (n=196) | 66.3% (n=75) | 41.8% (n=75) | 0.982 | 96.4% (n=196) | 6.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (ECOLI_ML_001)** (`lolo_ECOLI_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 42 | 95.2% (n=21) | 0.0% (n=16) | 11.9% (n=42) | 40.5% (n=42) | 42.9% (n=16) | 23.8% (n=16) | 0.583 | 73.8% (n=42) | 18.00 |
| b2_xgb_steps | 42 | 38.1% (n=21) | 0.0% (n=16) | 16.7% (n=42) | 64.3% (n=42) | 61.9% (n=16) | 23.8% (n=16) | 0.930 | 97.6% (n=42) | 14.00 |
| aft_known | 42 | 0.0% (n=21) | 0.0% (n=16) | 11.9% (n=42) | 88.1% (n=42) | 83.3% (n=16) | 57.1% (n=16) | 1.000 | 100.0% (n=42) | 6.00 |
| aft_known_unitig | 42 | 0.0% (n=21) | 0.0% (n=16) | 23.8% (n=42) | 76.2% (n=42) | 66.7% (n=16) | 35.7% (n=16) | 1.000 | 100.0% (n=42) | 6.00 |

**leave-one-lineage-out (ECOLI_ML_002)** (`lolo_ECOLI_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 31 | 91.7% (n=12) | 0.0% (n=13) | 19.4% (n=31) | 45.2% (n=31) | 45.2% (n=19) | 25.8% (n=19) | 0.577 | 96.8% (n=31) | 18.00 |
| b2_xgb_steps | 31 | 16.7% (n=12) | 0.0% (n=13) | 32.3% (n=31) | 61.3% (n=31) | 87.1% (n=19) | 45.2% (n=19) | 0.942 | 100.0% (n=31) | 14.00 |
| aft_known | 31 | 8.3% (n=12) | 0.0% (n=13) | 9.7% (n=31) | 87.1% (n=31) | 96.8% (n=19) | 71.0% (n=19) | 0.984 | 100.0% (n=31) | 6.00 |
| aft_known_unitig | 31 | 8.3% (n=12) | 0.0% (n=13) | 19.4% (n=31) | 77.4% (n=31) | 93.5% (n=19) | 58.1% (n=19) | 0.978 | 100.0% (n=31) | 6.00 |

![VME and ME by model, test set](figures/vme_me_by_model_ECOLI_meropenem.png)

![EA and CA by model, test set](figures/ea_ca_by_model_ECOLI_meropenem.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_ECOLI_meropenem.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_ECOLI_meropenem.png)

### ECOLI x piperacillin-tazobactam

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 8 mg/L, R if MIC > 8 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 36 | 0.0% (n=12) | 0.0% (n=23) | 2.8% (n=36) | 97.2% (n=36) | — | — | — | — | — |
| b1_lookup | 36 | 83.3% (n=12) | 0.0% (n=23) | 2.8% (n=36) | 69.4% (n=36) | 61.1% (n=24) | 27.8% (n=24) | 0.946 | 77.8% (n=36) | 6.00 |
| b2_xgb_steps | 36 | 58.3% (n=12) | 0.0% (n=23) | 2.8% (n=36) | 77.8% (n=36) | 69.4% (n=24) | 30.6% (n=24) | 0.947 | 80.6% (n=36) | 4.00 |
| aft_known | 36 | 0.0% (n=12) | 0.0% (n=23) | 2.8% (n=36) | 97.2% (n=36) | 88.9% (n=24) | 41.7% (n=24) | 1.000 | 100.0% (n=36) | 4.00 |
| aft_known_unitig | 36 | 0.0% (n=12) | 0.0% (n=23) | 2.8% (n=36) | 97.2% (n=36) | 83.3% (n=24) | 44.4% (n=24) | 1.000 | 100.0% (n=36) | 6.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 167 | 12.9% (n=70) | 0.0% (n=95) | 1.2% (n=167) | 93.4% (n=167) | — | — | — | — | — |
| b1_lookup | 167 | 90.0% (n=70) | 4.2% (n=95) | 1.2% (n=167) | 58.7% (n=167) | 51.5% (n=106) | 28.7% (n=106) | 0.777 | 76.0% (n=167) | 6.00 |
| b2_xgb_steps | 167 | 30.0% (n=70) | 6.3% (n=95) | 1.2% (n=167) | 82.6% (n=167) | 66.5% (n=106) | 31.7% (n=106) | 0.916 | 82.6% (n=167) | 4.00 |
| aft_known | 167 | 8.6% (n=70) | 5.3% (n=95) | 1.2% (n=167) | 92.2% (n=167) | 83.2% (n=106) | 42.5% (n=106) | 0.990 | 97.0% (n=167) | 4.00 |
| aft_known_unitig | 167 | 10.0% (n=70) | 7.4% (n=95) | 1.2% (n=167) | 90.4% (n=167) | 72.5% (n=106) | 40.7% (n=106) | 0.980 | 97.6% (n=167) | 6.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (ECOLI_ML_001)** (`lolo_ECOLI_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 36 | 88.0% (n=25) | 36.4% (n=11) | 0.0% (n=36) | 27.8% (n=36) | 30.6% (n=13) | 16.7% (n=13) | 0.400 | 52.8% (n=36) | 6.00 |
| b2_xgb_steps | 36 | 100.0% (n=25) | 9.1% (n=11) | 0.0% (n=36) | 27.8% (n=36) | 22.2% (n=13) | 8.3% (n=13) | 0.409 | 30.6% (n=36) | 4.00 |
| aft_known | 36 | 0.0% (n=25) | 54.5% (n=11) | 0.0% (n=36) | 83.3% (n=36) | 83.3% (n=13) | 66.7% (n=13) | 0.989 | 100.0% (n=36) | 4.00 |
| aft_known_unitig | 36 | 0.0% (n=25) | 45.5% (n=11) | 0.0% (n=36) | 86.1% (n=36) | 77.8% (n=13) | 50.0% (n=13) | 0.991 | 100.0% (n=36) | 6.00 |

**leave-one-lineage-out (ECOLI_ML_002)** (`lolo_ECOLI_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 30 | 90.5% (n=21) | 0.0% (n=9) | 0.0% (n=30) | 36.7% (n=30) | 23.3% (n=14) | 20.0% (n=14) | 0.548 | 56.7% (n=30) | 6.00 |
| b2_xgb_steps | 30 | 100.0% (n=21) | 0.0% (n=9) | 0.0% (n=30) | 30.0% (n=30) | 30.0% (n=14) | 13.3% (n=14) | 0.820 | 46.7% (n=30) | 4.00 |
| aft_known | 30 | 4.8% (n=21) | 0.0% (n=9) | 0.0% (n=30) | 96.7% (n=30) | 83.3% (n=14) | 60.0% (n=14) | 0.989 | 100.0% (n=30) | 4.00 |
| aft_known_unitig | 30 | 4.8% (n=21) | 0.0% (n=9) | 0.0% (n=30) | 96.7% (n=30) | 93.3% (n=14) | 70.0% (n=14) | 0.997 | 100.0% (n=30) | 6.00 |

![VME and ME by model, test set](figures/vme_me_by_model_ECOLI_piperacillin-tazobactam.png)

![EA and CA by model, test set](figures/ea_ca_by_model_ECOLI_piperacillin-tazobactam.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_ECOLI_piperacillin-tazobactam.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_ECOLI_piperacillin-tazobactam.png)

### KPNEU x ceftriaxone

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 1 mg/L, R if MIC > 2 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 91 | 0.0% (n=75) | 0.0% (n=16) | 0.0% (n=91) | 100.0% (n=91) | — | — | — | — | — |
| b1_lookup | 91 | 70.7% (n=75) | 0.0% (n=16) | 0.0% (n=91) | 41.8% (n=91) | 31.9% (n=22) | 24.2% (n=22) | 0.868 | 70.3% (n=91) | 10.00 |
| b2_xgb_steps | 91 | 0.0% (n=75) | 0.0% (n=16) | 0.0% (n=91) | 100.0% (n=91) | 71.4% (n=22) | 29.7% (n=22) | 1.000 | 71.4% (n=91) | 2.00 |
| aft_known | 91 | 0.0% (n=75) | 0.0% (n=16) | 0.0% (n=91) | 100.0% (n=91) | 94.5% (n=22) | 89.0% (n=22) | 1.000 | 98.9% (n=91) | 4.00 |
| aft_known_unitig | 91 | 0.0% (n=75) | 0.0% (n=16) | 0.0% (n=91) | 100.0% (n=91) | 96.7% (n=22) | 89.0% (n=22) | 1.000 | 100.0% (n=91) | 6.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 457 | 0.0% (n=228) | 0.0% (n=229) | 0.0% (n=457) | 100.0% (n=457) | — | — | — | — | — |
| b1_lookup | 457 | 38.2% (n=228) | 6.6% (n=229) | 0.0% (n=457) | 77.7% (n=457) | 66.7% (n=153) | 43.1% (n=153) | 0.945 | 84.2% (n=457) | 10.00 |
| b2_xgb_steps | 457 | 3.5% (n=228) | 0.0% (n=229) | 0.0% (n=457) | 98.2% (n=457) | 87.1% (n=153) | 36.1% (n=153) | 0.989 | 87.1% (n=457) | 2.00 |
| aft_known | 457 | 0.0% (n=228) | 0.0% (n=229) | 0.2% (n=457) | 99.8% (n=457) | 90.4% (n=153) | 69.1% (n=153) | 1.000 | 99.3% (n=457) | 4.00 |
| aft_known_unitig | 457 | 1.3% (n=228) | 0.0% (n=229) | 0.0% (n=457) | 99.3% (n=457) | 85.3% (n=153) | 68.3% (n=153) | 1.000 | 99.1% (n=457) | 6.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (KPNEU_ML_001)** (`lolo_KPNEU_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 88 | 90.1% (n=81) | 0.0% (n=7) | 0.0% (n=88) | 17.0% (n=88) | 12.5% (n=28) | 3.4% (n=28) | 0.549 | 37.5% (n=88) | 10.00 |
| b2_xgb_steps | 88 | 1.2% (n=81) | 0.0% (n=7) | 0.0% (n=88) | 98.9% (n=88) | 68.2% (n=28) | 31.8% (n=28) | 0.993 | 68.2% (n=88) | 2.00 |
| aft_known | 88 | 0.0% (n=81) | 0.0% (n=7) | 0.0% (n=88) | 100.0% (n=88) | 89.8% (n=28) | 75.0% (n=28) | 1.000 | 98.9% (n=88) | 4.00 |
| aft_known_unitig | 88 | 1.2% (n=81) | 0.0% (n=7) | 0.0% (n=88) | 98.9% (n=88) | 92.0% (n=28) | 73.9% (n=28) | 1.000 | 98.9% (n=88) | 6.00 |

**leave-one-lineage-out (KPNEU_ML_002)** (`lolo_KPNEU_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 81 | 72.6% (n=73) | 0.0% (n=8) | 0.0% (n=81) | 34.6% (n=81) | 24.7% (n=17) | 19.8% (n=17) | 0.773 | 66.7% (n=81) | 10.00 |
| b2_xgb_steps | 81 | 0.0% (n=73) | 0.0% (n=8) | 0.0% (n=81) | 100.0% (n=81) | 67.9% (n=17) | 29.6% (n=17) | 1.000 | 67.9% (n=81) | 2.00 |
| aft_known | 81 | 0.0% (n=73) | 0.0% (n=8) | 0.0% (n=81) | 100.0% (n=81) | 93.8% (n=17) | 90.1% (n=17) | 1.000 | 98.8% (n=81) | 4.00 |
| aft_known_unitig | 81 | 0.0% (n=73) | 0.0% (n=8) | 0.0% (n=81) | 100.0% (n=81) | 97.5% (n=17) | 91.4% (n=17) | 1.000 | 100.0% (n=81) | 6.00 |

![VME and ME by model, test set](figures/vme_me_by_model_KPNEU_ceftriaxone.png)

![EA and CA by model, test set](figures/ea_ca_by_model_KPNEU_ceftriaxone.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_KPNEU_ceftriaxone.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_KPNEU_ceftriaxone.png)

### KPNEU x ciprofloxacin

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 0.25 mg/L, R if MIC > 0.5 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 91 | 30.8% (n=65) | 37.5% (n=24) | 2.2% (n=91) | 65.9% (n=91) | — | — | — | — | — |
| b1_lookup | 91 | 67.7% (n=65) | 0.0% (n=24) | 2.2% (n=91) | 49.5% (n=91) | 38.5% (n=52) | 22.0% (n=52) | 0.803 | 74.7% (n=91) | 8.00 |
| b2_xgb_steps | 91 | 1.5% (n=65) | 0.0% (n=24) | 5.5% (n=91) | 93.4% (n=91) | 69.2% (n=52) | 40.7% (n=52) | 0.995 | 94.5% (n=91) | 6.00 |
| aft_known | 91 | 0.0% (n=65) | 0.0% (n=24) | 4.4% (n=91) | 95.6% (n=91) | 80.2% (n=52) | 51.6% (n=52) | 1.000 | 94.5% (n=91) | 4.00 |
| aft_known_unitig | 91 | 0.0% (n=65) | 0.0% (n=24) | 3.3% (n=91) | 96.7% (n=91) | 96.7% (n=52) | 56.0% (n=52) | 1.000 | 96.7% (n=91) | 2.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 476 | 45.3% (n=159) | 23.6% (n=292) | 5.3% (n=476) | 65.1% (n=476) | — | — | — | — | — |
| b1_lookup | 476 | 67.3% (n=159) | 0.7% (n=292) | 6.1% (n=476) | 71.0% (n=476) | 57.4% (n=333) | 31.7% (n=333) | 0.847 | 86.3% (n=476) | 8.00 |
| b2_xgb_steps | 476 | 8.2% (n=159) | 2.1% (n=292) | 6.9% (n=476) | 89.1% (n=476) | 76.9% (n=333) | 40.3% (n=333) | 0.988 | 96.0% (n=476) | 6.00 |
| aft_known | 476 | 4.4% (n=159) | 0.7% (n=292) | 6.5% (n=476) | 91.6% (n=476) | 77.7% (n=333) | 41.0% (n=333) | 0.985 | 94.3% (n=476) | 4.00 |
| aft_known_unitig | 476 | 1.9% (n=159) | 0.7% (n=292) | 5.3% (n=476) | 93.7% (n=476) | 90.8% (n=333) | 51.9% (n=333) | 0.997 | 90.8% (n=476) | 2.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (KPNEU_ML_001)** (`lolo_KPNEU_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 97 | 87.5% (n=80) | 0.0% (n=8) | 9.3% (n=97) | 18.6% (n=97) | 15.5% (n=58) | 7.2% (n=58) | 0.617 | 49.5% (n=97) | 8.00 |
| b2_xgb_steps | 97 | 3.8% (n=80) | 12.5% (n=8) | 11.3% (n=97) | 84.5% (n=97) | 61.9% (n=58) | 26.8% (n=58) | 0.980 | 96.9% (n=97) | 6.00 |
| aft_known | 97 | 1.2% (n=80) | 0.0% (n=8) | 12.4% (n=97) | 86.6% (n=97) | 79.4% (n=58) | 48.5% (n=58) | 0.989 | 92.8% (n=97) | 4.00 |
| aft_known_unitig | 97 | 0.0% (n=80) | 0.0% (n=8) | 9.3% (n=97) | 90.7% (n=97) | 93.8% (n=58) | 52.6% (n=58) | 0.998 | 93.8% (n=97) | 2.00 |

**leave-one-lineage-out (KPNEU_ML_002)** (`lolo_KPNEU_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 82 | 67.7% (n=65) | 0.0% (n=16) | 1.2% (n=82) | 45.1% (n=82) | 35.4% (n=44) | 20.7% (n=44) | 0.725 | 72.0% (n=82) | 8.00 |
| b2_xgb_steps | 82 | 1.5% (n=65) | 0.0% (n=16) | 4.9% (n=82) | 93.9% (n=82) | 69.5% (n=44) | 41.5% (n=44) | 0.993 | 95.1% (n=82) | 6.00 |
| aft_known | 82 | 0.0% (n=65) | 0.0% (n=16) | 3.7% (n=82) | 96.3% (n=82) | 81.7% (n=44) | 53.7% (n=44) | 1.000 | 95.1% (n=82) | 4.00 |
| aft_known_unitig | 82 | 0.0% (n=65) | 0.0% (n=16) | 2.4% (n=82) | 97.6% (n=82) | 97.6% (n=44) | 58.5% (n=44) | 1.000 | 97.6% (n=82) | 2.00 |

![VME and ME by model, test set](figures/vme_me_by_model_KPNEU_ciprofloxacin.png)

![EA and CA by model, test set](figures/ea_ca_by_model_KPNEU_ciprofloxacin.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_KPNEU_ciprofloxacin.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_KPNEU_ciprofloxacin.png)

### KPNEU x gentamicin

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 2 mg/L, R if MIC > 2 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 63 | 100.0% (n=17) | 0.0% (n=36) | 15.9% (n=63) | 57.1% (n=63) | — | — | — | — | — |
| b1_lookup | 63 | 88.2% (n=17) | 0.0% (n=36) | 15.9% (n=63) | 60.3% (n=63) | 60.3% (n=46) | 42.9% (n=46) | 0.583 | 77.8% (n=63) | 6.00 |
| b2_xgb_steps | 63 | 0.0% (n=17) | 0.0% (n=36) | 15.9% (n=63) | 84.1% (n=63) | 93.7% (n=46) | 52.4% (n=46) | 1.000 | 93.7% (n=63) | 2.00 |
| aft_known | 63 | 0.0% (n=17) | 0.0% (n=36) | 15.9% (n=63) | 84.1% (n=63) | 90.5% (n=46) | 41.3% (n=46) | 1.000 | 90.5% (n=63) | 2.00 |
| aft_known_unitig | 63 | 0.0% (n=17) | 0.0% (n=36) | 15.9% (n=63) | 84.1% (n=63) | 96.8% (n=46) | 55.6% (n=46) | 1.000 | 100.0% (n=63) | 4.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 280 | 100.0% (n=45) | 0.0% (n=224) | 3.9% (n=280) | 80.0% (n=280) | — | — | — | — | — |
| b1_lookup | 280 | 68.9% (n=45) | 0.4% (n=224) | 3.9% (n=280) | 84.6% (n=280) | 77.9% (n=174) | 46.4% (n=174) | 0.625 | 92.1% (n=280) | 6.00 |
| b2_xgb_steps | 280 | 2.2% (n=45) | 1.3% (n=224) | 3.9% (n=280) | 94.6% (n=280) | 96.8% (n=174) | 57.9% (n=174) | 0.986 | 96.8% (n=280) | 2.00 |
| aft_known | 280 | 2.2% (n=45) | 1.3% (n=224) | 3.9% (n=280) | 94.6% (n=280) | 96.1% (n=174) | 58.9% (n=174) | 0.972 | 96.1% (n=280) | 2.00 |
| aft_known_unitig | 280 | 2.2% (n=45) | 1.3% (n=224) | 3.9% (n=280) | 94.6% (n=280) | 86.4% (n=174) | 51.4% (n=174) | 0.976 | 98.2% (n=280) | 4.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (KPNEU_ML_001)** (`lolo_KPNEU_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 53 | 100.0% (n=13) | 0.0% (n=36) | 7.5% (n=53) | 67.9% (n=53) | 60.4% (n=30) | 30.2% (n=30) | 0.486 | 90.6% (n=53) | 6.00 |
| b2_xgb_steps | 53 | 0.0% (n=13) | 8.3% (n=36) | 7.5% (n=53) | 86.8% (n=53) | 98.1% (n=30) | 54.7% (n=30) | 0.968 | 98.1% (n=53) | 2.00 |
| aft_known | 53 | 0.0% (n=13) | 8.3% (n=36) | 7.5% (n=53) | 86.8% (n=53) | 94.3% (n=30) | 60.4% (n=30) | 0.940 | 94.3% (n=53) | 2.00 |
| aft_known_unitig | 53 | 0.0% (n=13) | 8.3% (n=36) | 7.5% (n=53) | 86.8% (n=53) | 92.5% (n=30) | 43.4% (n=30) | 0.937 | 96.2% (n=53) | 4.00 |

**leave-one-lineage-out (KPNEU_ML_002)** (`lolo_KPNEU_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 56 | 93.8% (n=16) | 0.0% (n=30) | 17.9% (n=56) | 55.4% (n=56) | 57.1% (n=41) | 41.1% (n=41) | 0.578 | 75.0% (n=56) | 6.00 |
| b2_xgb_steps | 56 | 0.0% (n=16) | 0.0% (n=30) | 17.9% (n=56) | 82.1% (n=56) | 94.6% (n=41) | 53.6% (n=41) | 1.000 | 94.6% (n=56) | 2.00 |
| aft_known | 56 | 0.0% (n=16) | 0.0% (n=30) | 17.9% (n=56) | 82.1% (n=56) | 92.9% (n=41) | 41.1% (n=41) | 1.000 | 92.9% (n=56) | 2.00 |
| aft_known_unitig | 56 | 0.0% (n=16) | 0.0% (n=30) | 17.9% (n=56) | 82.1% (n=56) | 98.2% (n=41) | 57.1% (n=41) | 1.000 | 100.0% (n=56) | 4.00 |

![VME and ME by model, test set](figures/vme_me_by_model_KPNEU_gentamicin.png)

![EA and CA by model, test set](figures/ea_ca_by_model_KPNEU_gentamicin.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_KPNEU_gentamicin.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_KPNEU_gentamicin.png)

### KPNEU x meropenem

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 2 mg/L, R if MIC > 8 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 94 | 0.0% (n=60) | 6.7% (n=30) | 4.3% (n=94) | 93.6% (n=94) | — | — | — | — | — |
| b1_lookup | 94 | 63.3% (n=60) | 0.0% (n=30) | 9.6% (n=94) | 50.0% (n=94) | 51.1% (n=44) | 34.0% (n=44) | 0.726 | 77.7% (n=94) | 16.00 |
| b2_xgb_steps | 94 | 6.7% (n=60) | 0.0% (n=30) | 27.7% (n=94) | 68.1% (n=94) | 68.1% (n=44) | 35.1% (n=44) | 0.987 | 89.4% (n=94) | 4.00 |
| aft_known | 94 | 0.0% (n=60) | 0.0% (n=30) | 11.7% (n=94) | 88.3% (n=94) | 83.0% (n=44) | 58.5% (n=44) | 1.000 | 97.9% (n=94) | 4.00 |
| aft_known_unitig | 94 | 0.0% (n=60) | 0.0% (n=30) | 24.5% (n=94) | 75.5% (n=94) | 93.6% (n=44) | 60.6% (n=44) | 1.000 | 97.9% (n=94) | 4.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 485 | 0.0% (n=123) | 1.5% (n=337) | 5.2% (n=485) | 93.8% (n=485) | — | — | — | — | — |
| b1_lookup | 485 | 82.9% (n=123) | 0.0% (n=337) | 5.8% (n=485) | 73.2% (n=485) | 70.7% (n=195) | 45.4% (n=195) | 0.808 | 88.7% (n=485) | 16.00 |
| b2_xgb_steps | 485 | 14.6% (n=123) | 0.3% (n=337) | 16.7% (n=485) | 79.4% (n=485) | 83.3% (n=195) | 35.9% (n=195) | 0.949 | 90.5% (n=485) | 4.00 |
| aft_known | 485 | 4.1% (n=123) | 0.0% (n=337) | 16.1% (n=485) | 82.9% (n=485) | 81.9% (n=195) | 54.4% (n=195) | 1.000 | 94.8% (n=485) | 4.00 |
| aft_known_unitig | 485 | 8.9% (n=123) | 0.0% (n=337) | 17.7% (n=485) | 80.0% (n=485) | 77.1% (n=195) | 54.2% (n=195) | 0.999 | 90.3% (n=485) | 4.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (KPNEU_ML_001)** (`lolo_KPNEU_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 93 | 95.5% (n=67) | 0.0% (n=20) | 8.6% (n=93) | 22.6% (n=93) | 15.1% (n=39) | 9.7% (n=39) | 0.618 | 51.6% (n=93) | 16.00 |
| b2_xgb_steps | 93 | 1.5% (n=67) | 0.0% (n=20) | 55.9% (n=93) | 43.0% (n=93) | 47.3% (n=39) | 21.5% (n=39) | 0.998 | 75.3% (n=93) | 4.00 |
| aft_known | 93 | 3.0% (n=67) | 0.0% (n=20) | 49.5% (n=93) | 48.4% (n=93) | 52.7% (n=39) | 24.7% (n=39) | 1.000 | 67.7% (n=93) | 4.00 |
| aft_known_unitig | 93 | 0.0% (n=67) | 0.0% (n=20) | 57.0% (n=93) | 43.0% (n=93) | 46.2% (n=39) | 21.5% (n=39) | 1.000 | 63.4% (n=93) | 4.00 |

**leave-one-lineage-out (KPNEU_ML_002)** (`lolo_KPNEU_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 82 | 64.4% (n=59) | 0.0% (n=19) | 9.8% (n=82) | 43.9% (n=82) | 45.1% (n=40) | 30.5% (n=40) | 0.661 | 74.4% (n=82) | 16.00 |
| b2_xgb_steps | 82 | 6.8% (n=59) | 0.0% (n=19) | 29.3% (n=82) | 65.9% (n=82) | 64.6% (n=40) | 37.8% (n=40) | 0.983 | 87.8% (n=82) | 4.00 |
| aft_known | 82 | 0.0% (n=59) | 0.0% (n=19) | 12.2% (n=82) | 87.8% (n=82) | 80.5% (n=40) | 56.1% (n=40) | 1.000 | 97.6% (n=82) | 4.00 |
| aft_known_unitig | 82 | 0.0% (n=59) | 0.0% (n=19) | 26.8% (n=82) | 73.2% (n=82) | 92.7% (n=40) | 57.3% (n=40) | 1.000 | 97.6% (n=82) | 4.00 |

![VME and ME by model, test set](figures/vme_me_by_model_KPNEU_meropenem.png)

![EA and CA by model, test set](figures/ea_ca_by_model_KPNEU_meropenem.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_KPNEU_meropenem.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_KPNEU_meropenem.png)

### KPNEU x piperacillin-tazobactam

Call breakpoint (EUCAST 2024, bloodstream): S if MIC <= 8 mg/L, R if MIC > 8 mg/L.

#### Test set (lineage-held-out genomes, scored once)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 57 | 6.7% (n=45) | 0.0% (n=12) | 0.0% (n=57) | 94.7% (n=57) | — | — | — | — | — |
| b1_lookup | 57 | 84.4% (n=45) | 8.3% (n=12) | 0.0% (n=57) | 31.6% (n=57) | 26.3% (n=16) | 19.3% (n=16) | 0.611 | 33.3% (n=57) | 4.00 |
| b2_xgb_steps | 57 | 17.8% (n=45) | 41.7% (n=12) | 0.0% (n=57) | 77.2% (n=57) | 63.2% (n=16) | 28.1% (n=16) | 0.818 | 64.9% (n=57) | 4.00 |
| aft_known | 57 | 2.2% (n=45) | 25.0% (n=12) | 0.0% (n=57) | 93.0% (n=57) | 93.0% (n=16) | 64.9% (n=16) | 0.987 | 93.0% (n=57) | 2.00 |
| aft_known_unitig | 57 | 0.0% (n=45) | 50.0% (n=12) | 0.0% (n=57) | 89.5% (n=57) | 91.2% (n=16) | 49.1% (n=16) | 0.989 | 100.0% (n=57) | 4.00 |

#### Cross-validation (out-of-fold, train split)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b0_resfinder | 298 | 6.2% (n=96) | 0.0% (n=194) | 2.7% (n=298) | 95.3% (n=298) | — | — | — | — | — |
| b1_lookup | 298 | 81.2% (n=96) | 2.1% (n=194) | 2.7% (n=298) | 69.8% (n=298) | 65.4% (n=188) | 33.6% (n=188) | 0.775 | 74.5% (n=298) | 4.00 |
| b2_xgb_steps | 298 | 31.2% (n=96) | 4.1% (n=194) | 2.7% (n=298) | 84.6% (n=298) | 70.5% (n=188) | 38.9% (n=188) | 0.920 | 78.2% (n=298) | 4.00 |
| aft_known | 298 | 3.1% (n=96) | 1.5% (n=194) | 2.7% (n=298) | 95.3% (n=298) | 92.3% (n=188) | 56.4% (n=188) | 0.998 | 92.3% (n=298) | 2.00 |
| aft_known_unitig | 298 | 6.2% (n=96) | 2.1% (n=194) | 2.7% (n=298) | 94.0% (n=298) | 83.2% (n=188) | 42.6% (n=188) | 0.995 | 97.7% (n=298) | 4.00 |

#### External and leave-one-lineage-out sets

**leave-one-lineage-out (KPNEU_ML_001)** (`lolo_KPNEU_ML_001`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 58 | 97.7% (n=44) | 0.0% (n=12) | 3.4% (n=58) | 22.4% (n=58) | 20.7% (n=17) | 10.3% (n=17) | 0.511 | 27.6% (n=58) | 4.00 |
| b2_xgb_steps | 58 | 2.3% (n=44) | 8.3% (n=12) | 3.4% (n=58) | 93.1% (n=58) | 44.8% (n=17) | 17.2% (n=17) | 0.966 | 50.0% (n=58) | 4.00 |
| aft_known | 58 | 0.0% (n=44) | 8.3% (n=12) | 3.4% (n=58) | 94.8% (n=58) | 93.1% (n=17) | 56.9% (n=17) | 0.999 | 93.1% (n=58) | 2.00 |
| aft_known_unitig | 58 | 0.0% (n=44) | 0.0% (n=12) | 3.4% (n=58) | 96.6% (n=58) | 65.5% (n=17) | 34.5% (n=17) | 1.000 | 94.8% (n=58) | 4.00 |

**leave-one-lineage-out (KPNEU_ML_002)** (`lolo_KPNEU_ML_002`)

| Model | n | VME % (predicted S, lab R; of lab R) | ME % (predicted R, lab S; of lab S) | Minor error % (of categorised) | CA % (same S/I/R) | EA % (within +/-1 step; exact MICs) | Exact agreement % | AUROC (R vs S) | Band coverage % (90% band) | Band width (doubling steps) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b1_lookup | 52 | 84.4% (n=45) | 14.3% (n=7) | 0.0% (n=52) | 25.0% (n=52) | 21.2% (n=11) | 15.4% (n=11) | 0.575 | 26.9% (n=52) | 4.00 |
| b2_xgb_steps | 52 | 17.8% (n=45) | 57.1% (n=7) | 0.0% (n=52) | 76.9% (n=52) | 59.6% (n=11) | 25.0% (n=11) | 0.735 | 61.5% (n=52) | 4.00 |
| aft_known | 52 | 2.2% (n=45) | 42.9% (n=7) | 0.0% (n=52) | 92.3% (n=52) | 92.3% (n=11) | 65.4% (n=11) | 0.979 | 92.3% (n=52) | 2.00 |
| aft_known_unitig | 52 | 0.0% (n=45) | 71.4% (n=7) | 0.0% (n=52) | 90.4% (n=52) | 90.4% (n=11) | 50.0% (n=11) | 0.984 | 100.0% (n=52) | 4.00 |

![VME and ME by model, test set](figures/vme_me_by_model_KPNEU_piperacillin-tazobactam.png)

![EA and CA by model, test set](figures/ea_ca_by_model_KPNEU_piperacillin-tazobactam.png)

![MIC confusion heatmap, test set, exact lab MICs](figures/mic_confusion_KPNEU_piperacillin-tazobactam.png)

![Top-20 gain importances, final fit on train](figures/feature_importance_KPNEU_piperacillin-tazobactam.png)

## Species-level figures

### ECOLI

![Conformal band coverage and width per drug, test set](figures/band_coverage_ECOLI.png)

![EA and VME by distance to the nearest training genome, test set](figures/accuracy_vs_distance_ECOLI.png)

![Lab label counts per drug](figures/label_counts_ECOLI.png)

![Lineage cluster sizes, train vs test](figures/lineage_clusters_ECOLI.png)

### KPNEU

![Conformal band coverage and width per drug, test set](figures/band_coverage_KPNEU.png)

![EA and VME by distance to the nearest training genome, test set](figures/accuracy_vs_distance_KPNEU.png)

![Lab label counts per drug](figures/label_counts_KPNEU.png)

![Lineage cluster sizes, train vs test](figures/lineage_clusters_KPNEU.png)

## Leakage checklist (automated)

Checks from `DATA_CONTRACT.md` section 4 that can be verified from the files on disk. `NOT RUN` means the input needed for that check is missing. Fold-internal feature selection and calibration-on-validation-only are enforced in the training code and are not re-verifiable from outputs.

6 passed, 0 failed, 0 not run.

| Check | Result | Detail |
| --- | --- | --- |
| no forbidden columns among features | PASS | 23575 feature names across 20 source(s); none forbidden, all prefixed |
| no lineage cluster in two splits or two folds | PASS | 50 clusters over 960 genomes; every cluster is in one split and one fold |
| unitig patterns built on train genomes only | PASS | unitigs_ECOLI_rows.parquet: 238 built, 50 queried; unitigs_KPNEU_rows.parquet: 561 built, 111 queried |
| labels de-duplicated by biosample | PASS | 943 biosamples over 943 genomes (0 rows without biosample) |
| prediction rows match their split | PASS | 20524 rows across 9 preds file(s); cv rows are train genomes, test rows are test genomes |
| test set touched once (informational: one run_id per preds file) | PASS | preds_ECOLI_ceftriaxone.parquet: f53e58239efc; preds_ECOLI_ciprofloxacin.parquet: f53e58239efc; preds_ECOLI_meropenem.parquet: f53e58239efc; preds_ECOLI_piperacillin-tazobactam.parquet: f53e58239efc; preds_KPNEU_ceftriaxone.parquet: f53e58239efc; preds_KPNEU_ciprofloxacin.parquet: f53e58239efc; preds_KPNEU_gentamicin.parquet: f53e58239efc; preds_KPNEU_meropenem.parquet: f53e58239efc; preds_KPNEU_piperacillin-tazobactam.parquet: f53e58239efc |

## Inputs and notes

| Input | Status |
| --- | --- |
| drop_log_evaluate.csv | found |
| drop_log_ingest.csv | found |
| drop_log_known_amr.csv | found |
| drop_log_lineages.csv | found |
| drop_log_qc.csv | found |
| drop_log_splits.csv | found |
| drop_log_synth.csv | found |
| drop_log_train.csv | found |
| drop_log_unitigs.csv | found |
| known_amr_columns.csv | found |
| label_counts.csv | found |
| lineages.parquet | found |
| metrics.parquet | found |
| metrics_by_distance.parquet | found |
| models/ECOLI/ceftriaxone/importance.json | found |
| models/ECOLI/ciprofloxacin/importance.json | found |
| models/ECOLI/meropenem/importance.json | found |
| models/ECOLI/piperacillin-tazobactam/importance.json | found |
| models/KPNEU/ceftriaxone/importance.json | found |
| models/KPNEU/ciprofloxacin/importance.json | found |
| models/KPNEU/gentamicin/importance.json | found |
| models/KPNEU/meropenem/importance.json | found |
| models/KPNEU/piperacillin-tazobactam/importance.json | found |
| pairs_kept.csv | found |
| preds_*.parquet | found |
| preds_ECOLI_ceftriaxone.parquet | found |
| preds_ECOLI_ciprofloxacin.parquet | found |
| preds_ECOLI_meropenem.parquet | found |
| preds_ECOLI_piperacillin-tazobactam.parquet | found |
| preds_KPNEU_ceftriaxone.parquet | found |
| preds_KPNEU_ciprofloxacin.parquet | found |
| preds_KPNEU_gentamicin.parquet | found |
| preds_KPNEU_meropenem.parquet | found |
| preds_KPNEU_piperacillin-tazobactam.parquet | found |
| splits.parquet | found |
| unitigs_ECOLI_index.parquet | found |
| unitigs_KPNEU_index.parquet | found |

### Rows excluded while drawing figures

| stage | reason | n_dropped | detail |
| --- | --- | --- | --- |
| report | mic_confusion: censored lab interval or null prediction | 21 | ECOLI ceftriaxone aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 8 | ECOLI ciprofloxacin aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 25 | ECOLI meropenem aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 12 | ECOLI piperacillin-tazobactam aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 69 | KPNEU ceftriaxone aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 39 | KPNEU ciprofloxacin aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 17 | KPNEU gentamicin aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 50 | KPNEU meropenem aft_known_unitig test |
| report | mic_confusion: censored lab interval or null prediction | 41 | KPNEU piperacillin-tazobactam aft_known_unitig test |

> These are predictions of in-vitro susceptibility, not prescribing advice. Dose, route, and final drug choice depend on PK/PD, infection site, renal function, allergies, and other patient factors, and remain with the clinician. Confirm with standard AST.
