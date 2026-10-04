# Training plan — shared MIC model on KPNEU

**Status:** draft · **Date:** 2026-10-03 · **Data:** release `2026-10-04-hackathon` · **Branch:** `training-plan`

These are predictions of in-vitro susceptibility, not prescribing advice. Dose, route,
and final drug choice remain with the clinician. Confirm with standard AST.

---

## 1. What we are building

We give the model the resistance genes found in one germ's DNA. For each drug, it guesses
the germ's **MIC**: the smallest amount of drug (mg/L) that stops the germ growing in the
lab. It also gives a **range** around that guess. Then we compare the range with the drug's
**cut-off numbers** (breakpoints) and say *likely works*, *likely fails*, or *not sure —
wait for the lab*.

One model handles all 29 KPNEU drugs at once (`multitask_aft`, see `MODEL_DESIGN.md`).
It gives the "works / fails" call for the 15 drugs that have checked cut-offs
(`configs/breakpoints/`, US CLSI rules).

## 2. Words used in this plan

| Word | Plain meaning |
| ---- | ------------- |
| MIC | Lowest drug amount (mg/L) that stops the germ growing in the lab. Small = drug works well. |
| Doubling step | Labs test 0.25, 0.5, 1, 2, 4, 8… Each step doubles. "1 step off" = guess is half or double the lab value. |
| Lab result types | `=8` exact. `<=0.25` "0.25 or less". `>8` "more than 8". The model learns from all three as ranges. |
| Cut-off (breakpoint) | Official number per drug: at or below "works", above another number "fails". |
| Practice groups (folds) | Training germs are split into 5 groups by family tree. Learn on 4, quiz on the 5th, 5 times. |
| Final exam set (test) | 1,088 germs locked away. Used **once**, at the very end. |
| Danger mistake (VME) | Model says "works", lab says "fails". Can hurt a patient. Most important number. |
| False alarm (ME) | Model says "fails", lab says "works". Wastes a good drug. |
| Close guess (EA) | Guess within 1 doubling step of an exact lab value. |
| Not sure share | How often the model says "wait for the lab". Lower is more useful, if danger mistakes stay low. |

## 3. Sample input

Real training genome `1284817.3` (practice group 1). The model sees three things:

**a. Species:** `KPNEU`.

**b. Genes and mutations** (289 columns after the rare-column filter; 0 = absent, 1 = present;
`n_class_*` = count of hits in that drug class). Only the non-zero ones are shown:

```
gene_blakpc_2 1     gene_blactx_m 1     gene_blashv 1       gene_blaoxa_1 1    gene_blaoxa_9 1
gene_aac_3_iie 1    gene_aac_6_ib_cr5 1 gene_aph_3_ib 1     gene_aph_6_id 1    gene_catb3 1
gene_dfra14 1       gene_sul2 1         gene_tet_a 1        gene_fosa 1        gene_oqxa 1
gene_oqxb 1         gene_emrd 1
point_gyra_s83f 1   point_gyra_d87a 1   point_parc_s80i 1   point_ompk36_q313ter 1   point_ramr_a19v 1
n_class_beta_lactam 6   n_class_quinolone 5   n_class_tetracycline 4   n_class_aminoglycoside 3   ...
```

In words: a carbapenem-breaking gene (KPC-2), an ESBL gene (CTX-M), and the classic
ciprofloxacin mutations (gyrA S83F, D87A, parC S80I).

**c. DNA pieces (unitigs):** not in this release yet. The model runs with `--no-unitigs`.

**Never inputs:** lineage cluster, sequence type, country, year, source, isolation source,
lab method, S/I/R text. These are for splitting and scoring only.

## 4. What the model learns from (labels)

The same genome has 18 lab results. The other 11 drugs were never tested on it. Those
cells are **skipped**, never filled in.

```
drug                           lab result   meaning as a range (mg/L)
amikacin                       =8           above 4, up to 8
ciprofloxacin                  >2           above 2 (no upper end)
colistin                       <=0.25       up to 0.25
levofloxacin                   =4           above 2, up to 4
meropenem                      >8           above 8
piperacillin-tazobactam        >64/4        above 64
tigecycline                    =0.5         above 0.25, up to 0.5
... (18 in total)
ceftriaxone, cefoxitin, tetracycline, ...   not tested → skipped
```

The training loss reads these ranges directly, so "more than 8" teaches the model
without anyone guessing the true number.

## 5. Sample output

### 5a. Run 1 (2026-10-03, before any fixes)

Out-of-practice-group guesses for `1284817.3` (made by the model that did **not** see
group 1). US cut-offs:

```
drug                       guess   range           lab      call
meropenem                  256     4 – 16384       >8       likely fails
ciprofloxacin              512     32 – 4096       >2       likely fails
levofloxacin               256     16 – 4096       =4       likely fails
gentamicin                 64      8 – 1024        >8       likely fails
amikacin                   16      2 – 256         =8       not sure
tigecycline                1       0.25 – 4        =0.5     not sure (no US cut-off loaded)
piperacillin-tazobactam    4096    256 – 65536     >64/4    likely fails
```

The calls point the right way. **The numbers do not.** No lab tests meropenem up to
16,384 mg/L. Section 7 explains why, and step 1 fixes it.

### 5b. What the API returns (shape, from `DATA_CONTRACT.md` stage 12)

```json
{
  "sample_id": "1284817.3",
  "species": "KPNEU",
  "qc_pass": true,
  "nearest_training_distance": null,
  "in_range": true,
  "predictions": [
    {
      "drug": "meropenem",
      "pred_mic": 256.0, "band_low": 4.0, "band_high": 16384.0,
      "s_breakpoint": 1.0, "r_breakpoint": 2.0,
      "call": "likely_inactive", "margin_steps": null,
      "reasons": ["blaKPC-2"], "override": "strong_marker"
    }
  ],
  "ranked_active": [],
  "model_version": "multitask_aft@317cc8167a11",
  "run_id": "317cc8167a11",
  "disclaimer": "These are predictions of in-vitro susceptibility, not prescribing advice. ..."
}
```

Numbers above are from run 1 and are shown only to make the shape concrete.

## 6. Training strategy

1. **One shared model, 29 outputs.** Genes go in once; every drug gets its own output.
   A germ tested for 3 drugs still teaches the shared part about all of them.
2. **Learn from ranges.** Exact, "or less", and "more than" results all train the model.
   Untested drugs are skipped (masked).
3. **5 practice groups, split by family tree.** Close relatives always sit in the same
   group, so the model is quizzed on families it has not seen.
4. **Rare columns are dropped inside each practice round**, using that round's training
   germs only.
5. **Stop early.** Each round stops when quiz loss stops improving (patience 15 epochs).
6. **Ranges (conformal bands).** From the quiz-round errors on exact lab values we take
   the 90% error size per drug. That becomes the ± range.
7. **Round guesses up** to the next doubling step. A high guess is the safer mistake.
8. **Final model** trains on all 5 groups for the median stopping epoch.
9. **Calls and overrides.** Range vs US cut-off → call. Then: natural resistance →
   "fails" (ampicillin); carbapenemase gene → meropenem "fails".
10. **Final exam once.** Only after every choice is frozen.

Run time: one full run is about **2 minutes** on a Mac (16 GB). Tuning is cheap.

## 7. What run 1 showed

Run 1: KPNEU, 29 drugs, 289 gene columns, defaults, `--no-unitigs`. Out-of-practice-group
scores for the 15 headline drugs, lab call worked out from the lab range with US cut-offs:

| | Best drug | Worst drug | Target (device studies) |
| --- | --- | --- | --- |
| Danger mistakes (VME) | 0.0% | 1.6% (gentamicin) | ≤ 1.5% |
| False alarms (ME) | 0.1% | 6.7% (aztreonam) | ≤ 3% |
| Close guess (EA) | 78% (tetracycline) | 22% (ertapenem) | ≥ 90% |
| Not sure share | 24% (ceftriaxone) | 73% (tetracycline) | as low as safely possible |
| Range width | 5 steps | 16 steps | ~2–3 steps |

**Why the numbers are poor:** about 75% of labels are "or less" / "more than". For those
rows, the loss gets better the further the guess moves past the panel edge, and nothing
stops it. Meropenem guesses go from 0.001 to 3,600 mg/L while lab panels only cover
0.008–128. Because guesses for exact rows are then far off, the ±range becomes huge
(about ±6 steps for meropenem), and most calls become "not sure". Danger mistakes are low
mostly because the model rarely commits.

## 8. Plan of work

| # | Step | Why | Done when | Time |
| - | ---- | --- | --------- | ---- |
| 1 | **Keep guesses inside the lab range.** Limit each drug's output to its panel range seen in training (lowest step − 1 to highest step + 1). Try also: a floor on the noise term, and more weight on exact rows. | Fixes section 7. | Range width and "not sure" share drop; danger mistakes do not rise. | ½ day |
| 2 | **Fix the stopping cheat.** Pick the stopping epoch on a family-grouped slice of the 4 training groups, not on the quiz group. | Quiz scores are a bit too good today. | Quiz group touched only for scoring. Test added. | 2 h |
| 3 | **Scoring script** `eval/score_oof.py`. Adds `pred_sir` / `lab_sir` with US cut-offs, writes a table with danger mistakes first and the release name on every row. | We need one trusted scoreboard. | Matches the preview in this plan; unit tests. | 2 h |
| 4 | **Simple baselines.** B1: median MIC of training germs with the same gene set. XGBoost AFT per drug (in the listed stack). | Shows whether the shared model adds value. | Same scoreboard for all three. | ½ day |
| 5 | **Tuning, about 10 runs.** Width, dropout, weight decay, drug-balanced loss on/off, range level. | Pick the best settings honestly. | Pick by danger mistakes, then close guesses, then not-sure share, on the 15 drugs, quiz rounds only. | ½ day |
| 6 | **Overrides.** `configs/natural_resistance.csv` (KPNEU × ampicillin). Carbapenemase genes → carbapenems "fails". | Contract call rules. | Unit tests. | 2 h |
| 7 | **Final exam, once.** Script refuses to run without `--i-understand-this-uses-test`. Writes `results/metrics.parquet`. | Honest final number. | One run, results saved with release + run id. | 1 h |
| 8 | **Connect to the API.** `PredictionPipeline.load` reads `model.pt` + `spec.json`; demo uses test genomes' gene rows. | Demo. | `/ready` returns 200; one job ends with a real report. | ½ day |

Steps 1–3 come first. Without them, tuning would tune the wrong thing.

## 9. How we pick the winner

On the 15 headline drugs, practice-group scores only:

1. Danger mistakes (VME) as low as possible. Any drug above 1.5% is flagged.
2. Then false alarms (ME) ≤ 3%.
3. Then close guesses (EA) as high as possible.
4. Then "not sure" share as low as possible.

The other 14 drugs are reported as exploratory. Ampicillin is never ranked.

## 10. Known risks

1. **Optimistic test scores.** Family groups come from NCBI SNP clusters. One clone (e.g.
   ST258) can still be in train and test.
2. **Gene source mismatch.** Training genes come from NCBI's AMRFinderPlus runs; a new
   genome will use our own AMRFinderPlus 4.2.7 run.
3. **Lineage markers.** `gene_erm_42` looks "protective" but marks one clone. Do not show it
   as biology.
4. **Thin drugs.** Colistin, tigecycline, minocycline and others have few results.
5. **Mixed labs.** BV-BRC and NCBI panels differ slightly (e.g. ceftriaxone guesses run about
   1 step high on BV-BRC rows).
6. **Targets.** EA ≥ 90%, VME ≤ 1.5%, ME ≤ 3% are common device-study figures, not
   thresholds we claim to meet.

## 11. Commands

```bash
make pull-data                                  # release 2026-10-04-hackathon
python -m genome2mic.models.train_cli --processed-dir data/processed \
  --out-dir models/<run-name> --species KPNEU --no-unitigs
pytest                                          # all tests
```

Write `release=2026-10-04-hackathon` and the run id next to every result.
