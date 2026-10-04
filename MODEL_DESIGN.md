# One shared model for every species and every drug

**Status:** v0.1, built on fake data · **Code:** `src/genome2mic/models/`, `src/genome2mic/data/`, `src/genome2mic/eval/`

This document explains why the main model is one network with many outputs instead of
one XGBoost per species x drug, what the research says, and how to run it once the
contract files exist. Plain-language first, details after.

---

## 1. The idea in one picture

```
                 species key ─────────► embedding (16)  ──┐
genome ─► AMRFinderPlus ─► gene_/point_/n_class_ ─► MLP  ──┼─► shared trunk ─► one head per drug ─► log2 MIC per drug
                 unitig query ─► sparse adapter (per species) ┘      ▲                 │
                                                                     │          per-species shift per drug
                                                     one forward pass gives every drug
```

- **Multiple inputs:** species, curated known-AMR features, and the unitig presence
  vector. Each goes through its own small encoder, then they are joined.
- **Multiple outputs:** one output per drug, from one forward pass. A genome tested for
  three drugs teaches the shared trunk about all three at once.
- **Not classic:** the loss is an interval-censored normal likelihood on log2 MIC. It
  reads `(mic_lower, mic_upper]` straight from `labels.parquet`, so `<=`, `>`, and
  S/R-only rows train the same model with no invented numbers (same maths as XGBoost
  `survival:aft`, which is why the name is `multitask_aft`).
- **Missing labels are masked, never imputed.** If a genome has no lab result for a
  drug, that output cell contributes nothing to the loss. Contract rule 1 holds.

## 2. What the literature says (and how we used it)

Full citations are in the survey the research assistant produced (summarised here).

| Finding | Source | What we did |
| --- | --- | --- |
| Multi-label nets across drugs and species work; gains concentrate in drugs with few labels | Aytan-Aktug 2020 (mSystems); Chen 2019 (EBioMedicine); Green 2022 (Nat Commun) | Shared trunk, per-drug heads, `drug_balanced_loss=True` so rare drugs are not drowned |
| Encoding "untested" as `-1` leaks test-panel information | Aytan-Aktug 2020 | Masked loss; untested cells never enter the forward signal or the loss |
| Pooling species blindly can hurt vs species-specific training | Hicks 2019 (PLoS Comput Biol) | Species embedding into the trunk **plus** a per-species bias per drug (`species_drug_bias`) |
| Shared multi-drug models shift errors toward sensitivity (fewer missed resistant) | Green 2022 | Right direction for a VME-first objective; keep as a stated reason |
| Halving `<=` and doubling `>` values biases panel edges | PGSE 2026 (npj AMR) | Censored likelihood instead; `inf` and `0` bounds handled exactly |
| Lineage leakage is the dominant failure mode (random split 91.7% -> external recall 22.3%) | 2025 KPNEU meropenem pipeline; PLoS Biol 2025 | Trainer reads `splits.parquet` only; column selection runs per fold; test rows never loaded |
| Global pooling of k-mers destroys plasmid-cassette signal | 2026 cross-species foundation-model study | Unitig columns stay as individual sparse inputs; no global collapse |
| No published model is multi-drug, multi-species **and** interval-censored | gap | This is that model |

## 3. Components

| File | Class | Job |
| --- | --- | --- |
| `data/label_matrix.py` | `LabelMatrix` | Long `labels.parquet` -> genomes x drugs bounds in log2 + mask |
| `data/known_amr_table.py` | `KnownAmrTable` | Wide known-AMR table; rejects forbidden columns; per-fold rare filter |
| `data/unitig_store.py` | `UnitigStore` | One sparse CSR per species; never densified; per-fold column selection |
| `models/feature_bundle.py` | `FeatureBundle` | Column choices fitted on one training fold, saved with the model |
| `models/genome_batcher.py` | `GenomeBatcher` | Species-homogeneous mini-batches (one unitig adapter per batch) |
| `models/censored_normal_loss.py` | `CensoredNormalLoss` | Masked interval-censored normal NLL |
| `models/multitask_mic_net.py` | `MultitaskMicNet` | The network |
| `models/multitask_trainer.py` | `MultitaskTrainer` | One fit with early stopping on validation NLL |
| `models/conformal_bands.py` | `ConformalBands` | 90% half-width per species x drug from out-of-fold residuals |
| `models/cross_validation.py` | `CrossValidation` | Runs frozen folds -> OOF preds -> conformal -> final fit |
| `models/model_artifact.py` | `ModelArtifact` | `model.pt` + `spec.json` (drugs, columns, pattern ids, conformal, run_id) |
| `models/mic_predictor.py` | `MicPredictor` | New genome's features -> every drug's MIC and band |
| `eval/metrics.py` | `MicMetrics` | VME first, then ME, mE, EA, CA, AUROC, coverage, width |
| `models/train_cli.py` | runner | `python -m genome2mic.models.train_cli` |

### Network details

- Species: `nn.Embedding(5, 16)`.
- Known AMR: `Linear(n_known, 256) -> GELU -> Dropout`. Counts (`n_class_`) are `log1p`-compressed.
- Unitigs: one `Linear(n_selected_columns, 128)` per species, applied with `torch.sparse.mm`
  so the matrix is never densified. Species without a unitig matrix get a zero vector.
- Trunk: two `Linear(256) + GELU + Dropout`, then `LayerNorm`.
- Heads: `Linear(256, n_drugs)` for `mu`, plus `Embedding(5, n_drugs)` species shift,
  plus one learned `log_sigma` per drug.
- Loss: `-log[Phi((hi-mu)/sigma) - Phi((lo-mu)/sigma)]` with `log_ndtr` and interval
  mirroring for stability. Bounds are clamped to ±1000 **before** dividing by sigma;
  `inf / sigma` has a NaN gradient (found and fixed by a unit test).

### Per-fold feature selection (contract rule 4)

Inside each fold, using training-fold genomes only:
1. Known-AMR columns present in >= 5 genomes of at least one species.
2. Unitig columns with training frequency in [1%, 99%], then the top `unitig_max_columns`
   by absolute correlation with any drug's reported log2 bound. This is a cheap
   stand-in for pyseer's mixed-model selection. Swap in pyseer output here when ready;
   the interface is `UnitigStore.select_columns`.

### Uncertainty and rounding

- Conformal residuals use only out-of-fold rows with an exact lab MIC.
  `q = ceil((n+1)·0.9)/n` quantile of `|mu - log2(lab)|`, per species x drug, with
  drug-level then global fallback when a pair has < 20 exact rows (logged).
- `pred_mic = 2^ceil(mu)`, `band = [2^ceil(mu-q), 2^ceil(mu+q)]`. Everything rounds up.

## 4. Results on fake data

`tests/fixtures/synthetic_contract_data.py` plants gene effects (blaKPC -> meropenem
+7 steps, gyrA S83L -> ciprofloxacin +4, ...), lineage-structured gene frequencies,
panel floors/ceilings, 10% missing cells, and whole-cluster test holdout.
Out-of-fold, 400 genomes per species, known-AMR only:

| species | drug | VME | ME | EA | band coverage | band width (steps) |
| --- | --- | --- | --- | --- | --- | --- |
| ECOLI | ceftriaxone | 0.000 | 0.000 | 0.985 | 0.993 | 2.5 |
| ECOLI | ciprofloxacin | 0.000 | 0.000 | 0.935 | 0.990 | 3.6 |
| ECOLI | meropenem | 0.000 | 0.000 | 0.788 | 0.997 | 4.5 |
| KPNEU | ceftriaxone | 0.000 | 0.000 | 0.962 | 0.993 | 2.7 |
| KPNEU | ciprofloxacin | 0.000 | 0.053 | 0.894 | 0.987 | 3.5 |
| KPNEU | meropenem | 0.000 | 0.000 | 0.857 | 1.000 | 4.3 |

These numbers prove the plumbing works. They say nothing about real genomes.
Adding 100 random unitig columns to this tiny fake set lowered EA by 3-8 points, as
expected for noise features on 400 rows. The `--no-unitigs` flag is the ablation.

## 5. How to run

```bash
conda activate genome2mic          # python 3.11, torch CPU, pandas, scipy, sklearn
make install                       # pip install -e ".[model,test]"
make test                          # 48 tests, ~7 s
make train                         # needs data/processed/{labels,known_amr,splits}.parquet (+ unitigs_*.npz)
```

Outputs in `models/multitask/`: `model.pt`, `spec.json`, `preds_oof.parquet`,
`history.parquet`, `conformal.json`, `label_counts.csv`.

`spec.json` lists the unitig `pattern_id`s the model uses. At prediction time, run
`unitig-caller --query` against the fixed training set, collect the pattern ids
present, and call `MicPredictor.unitig_vector(species, present_ids)`.

## 6. Known limits and next steps

1. **Bands are too wide / over-cover** (99% vs 90% target). Rounding both edges up and
   counting interval overlap as coverage both inflate it. Decide whether to report
   coverage on exact rows only; tune `conformal_level`.
2. **Test set is untouched.** There is no `evaluate_test` entry point yet, on purpose.
   Add one that requires an explicit flag, run it once, at the end.
3. **pyseer selection** should replace the correlation filter once the data team
   has kinship matrices. Same interface.
4. **Breakpoints** are not wired: `preds_oof.parquet` lacks `pred_sir`/`lab_sir` until
   `configs/breakpoints/*.csv` exist. `MicMetrics.sir_from_mic` does the conversion.
5. **Distance-to-training** reporting (override 3) belongs in the prediction pipeline,
   not here.
6. **Optional ablations** to run on real data: per-drug XGBoost AFT vs this model;
   drug-class-grouped heads; heteroscedastic (per-sample) sigma.
7. **Dependency note:** torch was added under the `model` extra. xgboost is not
   installed; the per-drug baselines (B1, B2, AFT) are not built.

All outputs are predictions of in-vitro susceptibility, not prescribing advice.
