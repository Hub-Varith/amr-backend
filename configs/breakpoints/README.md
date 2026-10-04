# Breakpoints

Used only to turn S/I/R-only lab results into MIC intervals (and later, for calls).

## File layout

- One file per standard and first year in force: `<clsi|eucast>_<year>.csv`.
- A result tested in year Y uses the latest file with year <= Y.
- A result with no year uses a value only when every file agrees for that species
  and drug. Otherwise the row is dropped.
- Columns: `species, drug, s_breakpoint, r_breakpoint, version, site`.

## One form for both standards

- S if MIC <= `s_breakpoint`.
- R if MIC > `r_breakpoint`.
- CLSI writes R as "MIC >= x". Store it as `r_breakpoint = x / 2`.
  Example: CLSI meropenem R >= 4 is stored as `r_breakpoint = 2`.
- Site is bloodstream. Meningitis and urinary-only breakpoints are not used.

## Status

**KPNEU, 15 drugs: checked 2026-10-03.** Enterobacterales rows (bloodstream,
non-meningitis) for amikacin, aztreonam, cefoxitin, ceftazidime, ceftriaxone,
ciprofloxacin, ertapenem, gentamicin, imipenem, levofloxacin, meropenem,
piperacillin-tazobactam, tetracycline, tobramycin, trimethoprim-sulfamethoxazole.

| Files | Checked against |
| ----- | --------------- |
| `eucast_2019.csv` to `eucast_2021.csv` | EUCAST Clinical Breakpoint Tables v15.0 PDF, and the year-by-year table in the AMR R package (`msberends/AMR`, `data-raw/datasets/clinical_breakpoints`, built from WHONET). Both agree. |
| `clsi_2020.csv` to `clsi_2023.csv` | The AMR package table (CLSI 2020-2026). Cross-checked where a public source exists: FDA NARMS criteria from M100-Ed30 (meropenem, cefoxitin, ceftriaxone, tetracycline, trimethoprim-sulfamethoxazole) and the CLSI M100-Ed33 aminoglycoside release. Not checked against the paid M100 book. |

Year files mark when a value changed:

- CLSI: piperacillin-tazobactam in 2022 (Ed32); amikacin, gentamicin, tobramycin in 2023 (Ed33).
- EUCAST: ertapenem and imipenem R lowered in 2019 (v9.0); systemic aminoglycosides
  moved to bracketed values in 2020 (v10.0); piperacillin-tazobactam R lowered in 2021 (v11.0).
- EUCAST lowered ciprofloxacin and levofloxacin in 2017 (v7.0). `eucast_2010.csv`
  ciprofloxacin (0.5 / 1) matches 2013-2016.

EUCAST gives no breakpoint for cefoxitin (screening only) or tetracycline in
Enterobacterales, so those two have CLSI rows only. EUCAST systemic aminoglycoside
breakpoints are in brackets: use only in combination with other therapy.

**Still not checked:** `clsi_2000.csv` and `clsi_2010.csv` (older than the sources above),
and every ECOLI row.

A result with no year uses a value only when every file agrees **and** the drug is in
the oldest file of that standard (`clsi_2000`, `eucast_2010`). Otherwise an older,
unlisted version could have had a different value, so the row is dropped. With this
rule, adding the 2019-2023 files changes no row of the 2026-10-04-hackathon labels.

Predicted-MIC calls use CLSI, latest version (`genome2mic.predict.constants`).

Drugs with no row here keep their MIC results. Only their S/I/R-only rows are dropped
(reason `no_breakpoint`).
