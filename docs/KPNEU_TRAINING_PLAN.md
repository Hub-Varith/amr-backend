# KPNEU model preparation — no training started

## Model and available data

Use the shared PyTorch `multitask_aft` model described in MODEL_DESIGN.md, copied from develop commit fca451353e81951e619335cc4dabec8e35fcec4c. It is a small shared neural network with a separate MIC output for each drug. It learns log2 MIC with a masked, interval-censored normal likelihood. Missing laboratory results contribute no loss; limits such as >8 remain limits.

The verified release is `2026-10-04-hackathon`. The preparation check finds 6,141 training genomes, 72,203 eligible training labels, 29 drug heads, and 1,088 held-out genome IDs. All feature rows are KPNEU. There are 952 candidate AMR features; the five fitting folds retain 256–266 features each using the existing minimum-count rule. No unitigs, genome FASTA, or sequencing reads are present in this release. Other species cannot be trained until their feature tables arrive.

The copied release is ignored by Git and its input hashes are recorded in KPNEU_MODEL_PREFLIGHT.json. No release file or split assignment was edited. The source develop worktree is unchanged. Missing feature rows now fail instead of becoming all-zero rows; reportable species/drug pairs require actual training labels.

## Planned training protocol

1. Keep the supplied test set sealed for scoring until model choices are fixed. Use the existing five training folds as-is. Feature selection is repeated using only each fitting fold.
2. After authorization, run a short pilot to measure seconds per epoch and check finite losses, memory, and per-drug label coverage. A pilot is a runtime/implementation check, not an accuracy claim.
3. Train the shared model with AdamW (learning rate 0.001, weight decay 0.0001), dropout 0.2, batch size 128, drug-balanced loss, gradient clipping, maximum 200 epochs and early stopping patience 15. Start with seed 7 and two CPU threads. These are the existing design defaults, not tuned results.
4. Use the five validation folds to select the duration; the existing trainer refits on the full training set for the median best epoch count. Inspect per-drug losses and exact-MIC agreement, not only a pooled score.
5. Before calling this the best model, compare with per-drug XGBoost AFT on identical training folds and feature rules. That comparison is future work, not an implemented or completed experiment. Repeat the selected configuration with seeds 17 and 27 if time permits.
6. Evaluate the frozen test set once after selection. Report performance per antibiotic and label type, and coverage of usable labels. Missing or censored lab answers must not be converted into invented exact answers.

## Reliability limits and calibration work

The supplied split uses provisional NCBI SNP clusters; do not describe it as a validated external-hospital/general-lineage test. Breakpoints are unverified, so categorical susceptibility and serious-error claims remain provisional. Audit breakpoint-derived labels and include a numeric-MIC-only sensitivity analysis before reliable reporting.

The inherited band calculation treats finite intervals as exact MICs, although some may come from S/I/R labels or merged measurements. It also uses validation predictions whose epochs were selected on those validation folds. Before presenting coverage claims, distinguish actual exact numeric measurements using raw-result provenance and use a disjoint calibration procedure within training data. Do not call the existing bands guaranteed 90% intervals. The current model scaffold retains this inherited calibration implementation; it is not calibration-ready.

This prepares full-genome-derived-feature MIC prediction. Reduced-read validation additionally requires raw reads matched to samples and a feature converter with exactly the trained column meanings. Assembly accuracy alone does not validate MIC accuracy.

## Runtime estimate — not measured

On the 16 GB Apple Silicon Mac, budget roughly 20–90 minutes for one five-fold run plus final refit, and 2–6 hours for a small repeated-seed/model comparison. A short pilot should take roughly 1–5 minutes. These are planning ranges, not benchmarks: actual early-stopping epochs, CPU throughput and other workloads can move them substantially. This is CPU tabular training, not a DNA foundation-model fine-tune. Do not run all species in parallel on this Mac. Refine the estimate using the first authorized pilot before launching a long run.

## Preparation commands (do not train)

```sh
.venv-model/bin/python -m genome2mic.models.prepare_cli --processed-dir data/processed --report docs/KPNEU_MODEL_PREFLIGHT.json
```

After calibration/data review and explicit authorization, the initial fitting command is:

```sh
.venv-model/bin/python -m genome2mic.models.train_cli --processed-dir data/processed --out-dir models/kpneu_multitask_seed7 --species KPNEU --no-unitigs --threads 2 --seed 7
```

This command has NOT been executed. No trained weights have been produced. Preparation validation: 19 data/loss/metrics/pair-support checks passed; the synthetic end-to-end training test was deliberately excluded.
