# Training plan — one shared MIC model for 5 germ types

**Status:** draft v2 · **Date:** 2026-10-03 · **Data:** release `2026-10-04-hackathon-all5` · **Branch:** `training-plan`

These are predictions of in-vitro susceptibility, not prescribing advice. Dose, route,
and final drug choice remain with the clinician. Confirm with standard AST.

---

## 1. What we are building

We give the model the resistance genes found in one germ's DNA, plus which germ type it
is. For **each of 37 drugs** it guesses the germ's **MIC**: the smallest amount of drug
(mg/L) that stops the germ growing in the lab. It also gives a **range** around the guess.
We compare the range with the drug's **cut-off numbers** and say *likely works*,
*likely fails*, or *not sure — wait for the lab*.

One model handles all 5 germ types and all 37 drugs at once (`multitask_aft`, see
`MODEL_DESIGN.md`).

## 2. Words used in this plan

| Word | Plain meaning |
| ---- | ------------- |
| MIC | Lowest drug amount (mg/L) that stops the germ growing in the lab. Small = drug works well. |
| Doubling step | Labs test 0.25, 0.5, 1, 2, 4, 8… Each step doubles. The model works in steps: `log2(MIC)`. 0 = 1 mg/L, 3 = 8 mg/L, −2 = 0.25 mg/L. |
| Lab result types | `=8` exact. `<=0.25` "0.25 or less". `>8` "more than 8". All three train the model as ranges. |
| Cut-off (breakpoint) | Official number per drug: at or below "works", above another number "fails". |
| Epoch | One full pass over all training germs. |
| Batch | 128 germs of the same germ type, processed together. |
| Practice rounds (folds) | Training germs are split into 5 groups by family tree. Learn on 4, quiz on the 5th, 5 times. |
| Final exam set (test) | 4,253 germs locked away. Used **once**, at the very end. |
| Danger mistake (VME) | Model says "works", lab says "fails". Can hurt a patient. Most important number. |
| False alarm (ME) | Model says "fails", lab says "works". |
| Close guess (EA) | Guess within 1 doubling step of an exact lab value. Target ≥ 90%. |
| Not sure share | How often the model says "wait for the lab". |

## 3. The data at a glance

| Germ type | Germs | Drugs | Training lab results | Final exam results |
| --------- | ----: | ----: | -------------------: | -----------------: |
| *E. coli* (ECOLI) | 16,082 | 25 | 158,315 | 28,270 |
| *K. pneumoniae* (KPNEU) | 7,229 | 29 | 72,203 | 13,298 |
| *Acinetobacter* (ABAU) | 1,206 | 17 | 11,382 | 1,969 |
| *S. aureus* (SAUR) | 1,899 | 12 | 9,847 | 1,614 |
| *P. aeruginosa* (PAER) | 1,754 | 14 | 8,482 | 1,513 |
| **Total** | **28,170** | **37** | **260,229** | **46,664** |

- A training germ has **10 lab results on average** (1 to 26). The other drugs are untested.
- Lab result types in training: **43% "or less"**, **29% "more than"**, **28% exact**.
  *E. coli* is mostly "or less" (54%); *Klebsiella* and *Acinetobacter* mostly "more than".
- Gene data: 3,028 columns in the file. After dropping very rare ones inside each practice
  round, the model sees **694–771 columns**.

## 4. One germ, start to finish

Real training germ `1284817.3` (*Klebsiella*, practice group 1). All numbers below come
from the real training code (`GenomeBatcher.tensors`).

### 4a. Raw inputs

```
species: KPNEU
genes:   blaKPC-2, blaCTX-M, blaSHV, blaOXA-1, blaOXA-9, aac(3)-IIe, aac(6')-Ib-cr5, aph(3'')-Ib,
         aph(6)-Id, catB3, dfrA14, sul2, tet(A), fosA, oqxA, oqxB, emrD
mutations: gyrA S83F, gyrA D87A, parC S80I, ompK36 Q313*, ramR A19V
```

In words: a carbapenem-breaking gene (KPC-2), an ESBL gene (CTX-M), and the classic
ciprofloxacin mutations.

### 4b. What the model actually receives (tensors)

```
species   = 1                       # index into [ECOLI, KPNEU, SAUR, PAER, ABAU]
known     = float32[694]            # one number per gene column, log(1 + count)
            gene_blakpc_2     0.693  # present (log 2)
            gene_blactx_m     0.693
            point_parc_s80i   0.693
            n_class_beta_lactam 1.946  # 6 beta-lactam hits -> log(7)
            n_class_quinolone   1.792  # 5 hits -> log(6)
            ... 31 non-zero of 694; all others 0
unitigs   = None                    # DNA pieces: not in this release
```

### 4c. What the model learns from (targets)

Each drug gets a **lower** and **upper** bound in doubling steps, plus a **mask**
(1 = tested, 0 = not tested). 18 drugs are tested, 19 are not.

```
drug                      lab     lower   upper   mask   meaning
amikacin                  =8       2.0     3.0     1     4 to 8 mg/L
levofloxacin              =4       1.0     2.0     1     2 to 4 mg/L
meropenem                 >8       3.0     inf     1     more than 8 mg/L
piperacillin-tazobactam   >64/4    6.0     inf     1     more than 64 mg/L
colistin                  <=0.25  -inf    -2.0     1     0.25 mg/L or less
ceftriaxone               (none)   nan     nan     0     not tested -> ignored by the loss
... 37 drugs in total
```

Nothing is filled in for untested drugs. The loss simply skips them.

### 4d. What the model gives back

For every germ and every drug, the model outputs a **centre guess `mu`** (in doubling
steps) and, per drug, a **spread `sigma`**. Then:

1. `pred_mic = 2^ceil(mu)`: the guess, rounded **up** to the next doubling step.
2. `band = 2^ceil(mu ± q)`: `q` is the 90% error size measured on the quiz rounds.
3. Call: band top ≤ "works" cut-off → *likely works*; band bottom > "fails" cut-off →
   *likely fails*; else *not sure*.

Real output for the same germ, from the 5-germ run (the model that did **not** see group 1):

```
drug                      mu      guess    range            lab      call (US cut-offs)
meropenem                 6.88    128      2 – 8,192        >8       not sure
levofloxacin              5.41    64       4 – 1,024        =4       likely fails
gentamicin                6.36    128      16 – 1,024       >8       likely fails
amikacin                  2.13    8        1 – 64           =8       not sure
piperacillin-tazobactam   14.13   32,768   2,048 – 524,288  >64/4    likely fails
tigecycline              -0.78    1        0.25 – 4         =0.5     not sure (no cut-off yet)
```

A second germ, *E. coli* `1416672.3` (practice group 2), mostly exact lab values:

```
drug            mu      guess    range              lab       call
ceftriaxone     7.89    256      8 – 16,384         =32       likely fails      (right)
meropenem      -6.57    0.016    0.0005 – 0.5       <=0.06    likely works      (right)
gentamicin     -1.35    0.5      0.125 – 4          =0.5      not sure          (guess exact)
ciprofloxacin  -8.11    0.004    0.0001 – 0.25      =1        likely works      (WRONG: lab 1 = fails under US rules)
```

The ciprofloxacin row is a **danger mistake**. Section 8 explains why it happens.

### 4e. What the website backend returns

The API wraps the same numbers in the `DATA_CONTRACT.md` stage 12 report (`PredictionReport`):

```json
{
  "sample_id": "1284817.3", "species": "KPNEU", "qc_pass": true,
  "nearest_training_distance": null, "in_range": true,
  "predictions": [
    {"drug": "meropenem", "pred_mic": 128.0, "band_low": 2.0, "band_high": 8192.0,
     "s_breakpoint": 1.0, "r_breakpoint": 2.0, "call": "likely_inactive",
     "margin_steps": null, "reasons": ["blaKPC-2"], "override": "strong_marker"}
  ],
  "ranked_active": [], "model_version": "multitask_aft@4d85f96f288e", "run_id": "4d85f96f288e",
  "disclaimer": "These are predictions of in-vitro susceptibility, not prescribing advice. ..."
}
```

Here the carbapenemase rule (KPC-2 → meropenem fails) overrides the model's "not sure".

## 5. One training batch

Every batch holds 128 germs of **one** germ type (so a future per-germ DNA-piece layer fits).

```
species  int64   [128]         same value for the whole batch
known    float32 [128, 694]    gene columns, log(1 + count)
unitigs  None                  (later: sparse [128, n_pieces])
lower    float32 [128, 37]     lower bound per drug, log2 mg/L (-inf = "or less")
upper    float32 [128, 37]     upper bound per drug, log2 mg/L (+inf = "more than")
mask     float32 [128, 37]     1 = tested; a typical E. coli batch: 1,448 of 4,736 cells
```

One practice round has **153 batches per epoch**: 86 *E. coli*, 39 *Klebsiella*,
11 *S. aureus*, 10 *Pseudomonas*, 7 *Acinetobacter*. Batches are shuffled across germ types.

## 6. The model

```
species ──► embedding (5 → 16) ───────────────┐
genes   ──► Linear(771 → 256) + GELU + Dropout ┼─► concat (400) ─► Linear(400 → 256) + GELU + Dropout
DNA pieces ► per-germ Linear(n → 128) (zeros now) ┘                ─► Linear(256 → 256) + GELU + Dropout ─► LayerNorm
                                                                   ─► 37 drug heads: Linear(256 → 37)
                                                                      + per-germ shift per drug (5 × 37)
                                                                   ─► mu [37]   and one sigma per drug [37]
```

- **376,403 parameters** (small; trains on a laptop CPU).
- The per-germ shift lets the same gene mean a different MIC in a different germ type.

## 7. How it learns

### The loss in plain words

For each tested drug, the model's guess is a bell curve (centre `mu`, width `sigma`).
The loss asks: *how likely is the lab's range under that bell curve?*

- `=8` (4 to 8): the curve should put weight between 4 and 8.
- `>8`: the curve should put weight anywhere above 8.
- `<=0.25`: the curve should put weight anywhere below 0.25.

Untested drugs add nothing. Every drug counts equally (`drug_balanced_loss`), so rare
drugs are not drowned by common ones.

### Settings

| Setting | Value | Why |
| ------- | ----- | --- |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 | Standard, stable |
| Batch size | 128 germs | Fits easily in memory |
| Max epochs | 200 | Upper limit only |
| Early stopping | stop after **15 epochs** with no quiz improvement; keep the best epoch | Avoids memorising |
| Dropout | 0.2 | Avoids memorising |
| Gradient clipping | 5.0 | Stops rare big jumps |
| Rare gene filter | column must appear in ≥ 5 training germs of one germ type, **inside each round** | Contract rule 4 |
| Practice rounds | 5, split by family tree (`splits.parquet`) | Honest quiz scores |
| Range level | 90% (`conformal_level`) | Range should hold the lab value 9 times in 10 |
| Final model | trained on all 5 groups for the **median best epoch** | Uses all training data |
| Seed | 7 | Same result every run |

### What actually happened (5-germ run, 2026-10-03)

| Practice round (quiz group) | Best epoch | Stopped at | Train loss | Quiz loss |
| --- | ---: | ---: | ---: | ---: |
| 0 | 31 | 46 | 0.742 | 0.945 |
| 1 | 22 | 37 | 0.755 | 0.999 |
| 2 | 34 | 49 | 0.736 | 0.978 |
| 3 | 20 | 35 | 0.759 | 1.013 |
| 4 | 29 | 44 | 0.741 | 0.963 |
| **Final model** | **29 epochs** (median) | — | — | — |

Whole run (5 rounds + final): **about 6 minutes** on a Mac. The *Klebsiella*-only run
stopped later (best epochs 47–74), because it had less data per epoch.

Outputs in the run folder: `model.pt`, `spec.json` (drugs, gene columns, ranges, run id),
`conformal.json`, `preds_oof.parquet` (one guess per training germ × tested drug, from the
round that did not see it), `history.parquet` (loss per epoch), `label_counts.csv`.

## 8. Results so far, and the main problem

Scores on *Klebsiella* × the 15 headline drugs, quiz rounds only, US cut-offs. Medians
across the 15 drugs:

| | *Klebsiella*-only run | 5-germ run | Target |
| --- | ---: | ---: | ---: |
| Danger mistakes (VME) | 0.2% | 0.2% (worst: gentamicin 1.6%) | ≤ 1.5% |
| False alarms (ME) | 1.1% | 0.9% | ≤ 3% |
| Close guesses (EA) | 44% | 45% | ≥ 90% |
| "Not sure" share | 64% | 65% | low |
| Range width | 8.2 steps | 8.0 steps | ~2–3 steps |

Close guesses by germ type (5-germ run, exact lab values): *S. aureus* 79%, *E. coli* 72%,
*Klebsiella* 53%, *Pseudomonas* 52%, *Acinetobacter* 50%.

**The main problem: guesses run far past what any lab measures.** 98% of lab bounds sit
between 0.016 and 128 mg/L (−6 to +7 steps). The model's `mu` ranges from **−15 to +20
steps** (0.00003 to 1,000,000 mg/L). Why:

- 72% of results are "or less" or "more than".
- For those, the loss keeps improving as the guess moves further past the edge, and nothing
  stops it.
- Germs that only have edge results get extreme guesses. Germs with exact results then get
  pulled around too, so the error on exact results is large.
- A large error makes `q` large (about ±4–6 steps), so ranges are very wide and most calls
  are "not sure".
- Sometimes the extreme guess crosses a cut-off the wrong way: the *E. coli* ciprofloxacin
  example above (guess 0.004, lab 1) is a danger mistake.

Danger mistakes are low today mostly because the model rarely commits.

## 9. Plan of work

| # | Step | Done when | Time |
| - | ---- | --------- | ---- |
| 1 | **Keep guesses inside the lab range.** Per drug, limit `mu` to the panel range seen in training (lowest step − 1 to highest step + 1). Also try: a floor on `sigma`, and extra weight on exact results. Compare on quiz rounds. | `mu` stays within about −8 to +10; range width and "not sure" share drop; danger mistakes do not rise. | ½ day |
| 2 | **Fix the stopping shortcut.** Pick the stopping epoch on a family-grouped slice of the 4 training groups, not on the quiz group that is scored. | Quiz group used only for scoring; test added. | 2 h |
| 3 | **Scoring script** `eval/score_oof.py`: per germ type and drug, uses that germ type's cut-offs, danger mistakes first, release name on every row. | One trusted scoreboard. | 2 h |
| 4 | **Cut-offs for the other germ types.** Check CLSI numbers for the main drugs of *E. coli*, *S. aureus*, *Pseudomonas*, *Acinetobacter* (same method as the 15 KPNEU drugs). | Calls and danger-mistake counts for all 5. | ½ day |
| 5 | **Simple baselines.** B1: median MIC of training germs with the same gene set. Per-drug XGBoost. | Same scoreboard for all three. | ½ day |
| 6 | **Tuning, about 10 runs** (6 min each): width, dropout, weight decay, drug-balanced loss on/off, range level. | Pick by danger mistakes, then close guesses, then "not sure" share, quiz rounds only. | ½ day |
| 7 | **Overrides.** `configs/natural_resistance.csv` (e.g. KPNEU × ampicillin); carbapenemase genes → carbapenems "fails". | Unit tests. | 2 h |
| 8 | **Final exam, once.** Script refuses to run without `--i-understand-this-uses-test`. | Results saved with release + run id. | 1 h |
| 9 | **Connect to the API.** `PredictionPipeline.load` reads `model.pt` + `spec.json`. | `/ready` returns 200; one job gives a real report. | ½ day |

Steps 1–3 first. Tuning before step 1 would tune the wrong thing.

## 10. How we pick the winner

Per germ type, on drugs with checked cut-offs, quiz rounds only:

1. Danger mistakes (VME) as low as possible. Any drug above 1.5% is flagged.
2. False alarms (ME) ≤ 3%.
3. Close guesses (EA) as high as possible.
4. "Not sure" share as low as possible.

Report each germ type separately. *E. coli* has more rows than all others together and
would dominate pooled numbers.

## 11. Known risks

1. **Optimistic test scores.** Family groups come from NCBI SNP clusters; one clone can
   still be in train and test.
2. **Gene source mismatch.** Training genes come from NCBI's AMRFinderPlus runs; a new
   genome will use our own AMRFinderPlus 4.2.7 run.
3. **Lineage markers.** Some genes (e.g. `gene_erm_42` in *Klebsiella*) mark one clone, not a
   mechanism. Do not show them as biology.
4. **Thin drugs.** Colistin, tigecycline, minocycline and others have few results.
5. **Missing germs.** 3–17% of labelled germs per type have no NCBI gene result.
6. **Odd lab values.** 251 of 383,150 lab results sit outside 0.004–1,024 mg/L, e.g.
   BV-BRC ampicillin `=1000000` (mostly ECOLI ertapenem, ciprofloxacin, cefotaxime). They
   look like entry errors. Too few to cause the problem in section 8, but the data owner
   should decide whether stage 2 drops them.
7. **Targets** (EA ≥ 90%, VME ≤ 1.5%, ME ≤ 3%) are common device-study figures, not
   thresholds we claim to meet.

## 12. Commands

```bash
make pull-data                                   # release 2026-10-04-hackathon-all5
python -m genome2mic.models.train_cli --processed-dir data/processed \
  --out-dir models/<run-name> --no-unitigs       # all 5 germ types, about 6 minutes
python -m genome2mic.models.train_cli --processed-dir data/processed \
  --out-dir models/<run-name> --no-unitigs --species KPNEU   # one germ type
pytest
```

Write `release=2026-10-04-hackathon-all5` and the run id next to every result.
