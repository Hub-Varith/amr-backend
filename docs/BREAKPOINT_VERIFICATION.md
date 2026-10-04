# Breakpoint verification checklist

**Why:** every breakpoint in `configs/breakpoints/` (ours) and in develop's
`configs/breakpoints/` was typed from memory and has never been checked against the
published tables. Breakpoints decide (1) how S/I/R-only lab results become training
labels, (2) every "likely active / likely inactive" call the app makes, and (3) VME, ME
and categorical agreement. EA (MIC accuracy) does not use them.

**Standard:** the project follows the **US standard, CLSI M100**, for calls and scoring
(decision 2026-10-03). In the US, the FDA recognizes breakpoints through its STIC page,
mostly by recognizing CLSI M100. Note any drug where FDA recognition differs.

These are predictions of in-vitro susceptibility, not prescribing advice.

**Status 2026-10-03:** Hub checked the CLSI Enterobacterales breakpoints for 15
K. pneumoniae drugs (develop `ddd76ef`: AMR package CLSI 2020-2026 table, FDA NARMS
M100-Ed30, CLSI Ed33 aminoglycosides; not the paid M100 book). Our `clsi_2024.csv`
matches all 15, for K. pneumoniae and E. coli (30 rows, including every priority-1
row); they are marked in the checklist. Everything else is still unchecked.

## How to verify (one person, ideally a clinical microbiologist)

1. Open `docs/breakpoint_verification_checklist.csv`. Work in `priority` order.
2. For each row, open the `source_document` and go to `where_to_look`:
   - CLSI: *M100 Performance Standards for Antimicrobial Susceptibility Testing*, the
     edition in force for that year. Use the **current edition** for the 2024 call table.
     Read-only access is free at clsi.org. Tables: 2A Enterobacterales (E. coli,
     K. pneumoniae), 2B-1 P. aeruginosa, 2B-2 Acinetobacter spp., 2C Staphylococcus spp.
   - EUCAST: *Clinical Breakpoint Tables* for that year's version (free PDF/Excel at
     eucast.org; older versions in the breakpoint-table archive).
3. Write the **published** values in `published_S` and `published_R` exactly as printed
   (CLSI: "S ≤ x", "R ≥ y"; EUCAST: "S ≤ x", "R > y"), set `fda_recognized` for CLSI rows,
   then `matches_entered` = yes/no against `published_form_entered`, plus your name and date.
4. Use bloodstream / systemic breakpoints only (no urinary-only or meningitis values). For
   combinations, note the fixed partner concentration (e.g. tazobactam 4 mg/L).
5. Hand the sheet back. The code side then corrects the CSVs: CLSI "R ≥ y" is stored as
   `r_breakpoint = y / 2`; EUCAST "R > y" is stored as `r_breakpoint = y`.

## Priorities

| Priority | Rows | What | Why first |
| --- | --- | --- | --- |
| 1 | 6 | CLSI 2024, E. coli + K. pneumoniae × meropenem, ciprofloxacin, ceftriaxone | The start drugs; every app call and VME for them uses these |
| 2 | 70 | CLSI 2024, other species × drug pairs that have enough data in the release | Calls and scoring for every trainable pair |
| 3 | 16 | CLSI 2024, pairs with no data yet | Needed once more data lands |
| 4 | 80 | EUCAST 2024 (ours) | Only converts EUCAST-2024 S/I/R-only lab rows; not used for calls |
| 5 | 30 | develop's tables (CLSI 2000/2010/2019, EUCAST 2010/2019) | Built the 4,721 S/I/R-only labels in release `2026-10-04-hackathon`; report errors to Hub |

## Historical tables the data needs

S/I/R-only lab results are dropped unless a table exists for their own standard and
year. BV-BRC lab-method AST rows by year (all 5 species, rows with and without MICs;
queried 2026-10-03). No row is from 2024.

| Standard | Years with the most rows |
| --- | --- |
| CLSI | 2021 (79k), 2017 (33k), 2018 (17k), 2015 (16k), 2016 (2.8k), 2013 (2.5k), 2012 (1.7k), 2011 (1.4k), 2019 (1.2k) |
| EUCAST | 2016 (15k), 2017 (10k), 2019 (9.8k), 2021 (4.7k), 2018 (4.0k), 2020 (2.0k) |

Adding CLSI 2021, 2017, 2018, 2015 and EUCAST 2016, 2017, 2019 tables (from the
published documents, not from memory) would recover most S/I/R-only rows.
