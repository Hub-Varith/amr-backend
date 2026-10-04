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

**Not verified.** The values were entered from memory for the KPNEU start drugs
(ceftriaxone, meropenem, ciprofloxacin) and ECOLI, which shares Enterobacterales
breakpoints. Check each row against the CLSI M100 edition and the EUCAST breakpoint
table for that year before training. The rows most at risk:

- `eucast_2010.csv` ciprofloxacin (0.5 / 1): confirm which EUCAST version lowered it.
- `clsi_2000.csv`: values before the 2010 CLSI revision.

Drugs with no row here keep their MIC results. Only their S/I/R-only rows are dropped
(reason `no_breakpoint`).
