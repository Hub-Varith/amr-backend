# Breakpoint tables

One CSV per standard and version: `eucast_<year>.csv`, `clsi_<year>.csv`. The
filename is the lookup key: `genome2mic.config.Config.breakpoint()` matches
`(standard, version)` **exactly**. There is no fallback. An S/I/R-only label row whose
`standard_year` is missing or has no table here is dropped by the ingest stage and
counted in `drop_log_ingest.csv` (`S/I/R-only row with null standard_year`,
`no breakpoint table for standard_year`). A null standard never matches
(DATA_CONTRACT.md: "If the standard is unknown, drop the row"). The prediction call
uses `call_standard` from `drugs.yaml`, whose table must exist.

## Columns

| Column | Meaning |
| ------ | ------- |
| `species` | 5-letter key from `species.yaml` |
| `drug` | Normalized drug name from `drugs.yaml` |
| `s_breakpoint` | MIC (mg/L). **S if MIC <= s_breakpoint** |
| `r_breakpoint` | MIC (mg/L). **R if MIC > r_breakpoint**; I otherwise |
| `version` | Table year; must equal the year in the filename |
| `site` | Always `bloodstream` |
| `note` | Provenance and caveats. Every row carries a reminder to re-verify |

`s_breakpoint <= r_breakpoint` on every row. When they are equal there is no I
category.

## Conventions

- **Site.** Bloodstream / systemic (non-meningitis) breakpoints only. UTI-only
  and meningitis breakpoints are not stored. Some drugs are stricter for
  meningitis (meropenem) or only defined for urine (cefazolin surrogate, EUCAST
  cefalexin); those variants are deliberately absent.
- **CLSI "R >= x" is stored as `r_breakpoint = x / 2`**, the previous doubling
  step, so that the shared rule "R if MIC > r_breakpoint" reproduces the CLSI
  category on the doubling grid. CLSI SDD (susceptible-dose-dependent) ranges
  are folded into I.
- **EUCAST "S <= 0.001"** (susceptible-increased-exposure only, e.g. PAER
  ciprofloxacin) is stored exactly as published. Any realistic MIC up to the R
  breakpoint therefore maps to I, which is the EUCAST intent.
- **Omitted pairs.** A species x drug with no published MIC breakpoint in that
  standard has no row. Examples: EUCAST gentamicin for PAER; EUCAST
  tetracyclines for Enterobacterales and Acinetobacter; CLSI colistin (no S
  category, only I/R); CLSI tigecycline (FDA breakpoints only); CLSI gentamicin
  and amikacin for PAER after the 2023 aminoglycoside revision (no systemic
  breakpoint). S/I/R-only label rows for omitted pairs are dropped and logged by
  the ingest stage.
- **Natural resistance.** A breakpoint row may exist for a species x drug that is
  also listed in `../natural_resistance.csv` (KPNEU ampicillin, ABAU ceftriaxone
  under CLSI). The row is needed to convert reported S/I/R labels to intervals;
  the natural-resistance override still forces the final call to inactive.
- **Special rows.** SAUR cefoxitin is a mecA/mecC screening MIC, not a treatment
  breakpoint. CLSI daptomycin publishes only S and "nonsusceptible"; nonsusceptible
  is stored as R. Methicillin results are normalized to oxacillin in `drugs.yaml`.

## Provenance and verification

Values were transcribed from memory of EUCAST Clinical Breakpoint Tables v14.0
(2024) and CLSI M100 34th edition (2024). They have **not** been checked against
the published PDFs. Before any external use, a clinical microbiologist must
re-verify every row against the current tables and record the check here.

Breakpoints change between versions (CLSI cephalosporins and carbapenems in
2010, fluoroquinolones in 2019, aminoglycosides in 2023; EUCAST
susceptible-increased-exposure redefinition in 2019). Historical S/I/R-only rows
are dropped (and counted) until a table for their year is added here. Check the
`no breakpoint table for standard_year` detail in `drop_log_ingest.csv` to see which
`<standard> <year>` tables the data needs, then add `clsi_<year>.csv` /
`eucast_<year>.csv` files for them.

These tables support predictions of in-vitro susceptibility. They are not
prescribing advice.
