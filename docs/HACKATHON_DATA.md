# Hackathon data release (provisional)

Built 2026-10-04 to train within the hackathon deadline. **Not the final data.**
Replace it with the full pipeline (own AMRFinderPlus run, Mash/PopPUNK lineages) after.

These are predictions of in-vitro susceptibility, not prescribing advice.

## What is different from the contract

| File | Contract way | Hackathon way |
| ---- | ------------ | ------------- |
| `known_amr.parquet` | Our own AMRFinderPlus 4.2.7 run on QC-passing genomes | NCBI Pathogen Detection `AMR_genotypes` (NCBI's AMRFinderPlus runs, mixed versions), no own QC |
| `lineages.parquet` | Mash single-linkage or PopPUNK | NCBI SNP clusters (`PDS…`); genomes outside a cluster are their own cluster; `st` is null |
| `splits.parquet` | Frozen, committed to git | Provisional, **not** committed; only in the S3 release |

## Known weaknesses (say them in the demo)

1. **Test scores are optimistic.** SNP clusters keep near-identical isolates together, but
   one lineage (e.g. ST258) can sit in both train and test.
2. **Feature mismatch.** Training features come from NCBI's runs; the demo runs
   AMRFinderPlus 4.2.7 on a new genome. Most calls match, a few may not.
3. **10% of labelled KPNEU genomes are missing**: NCBI has no AMR result for them.
4. **Breakpoints are unverified** (`configs/breakpoints/README.md`), so S/I/R calls are provisional.

## Feature rules

- `MISTRANSLATION` calls (internal stop codon) are skipped. Partial calls count as present.
- `=POINT` → `point_<gene>_<mutation>`; everything else → `gene_<symbol>`.
- Beta-lactamase alleles collapse to the family (`blaSHV-12` → `gene_blashv`) except the
  families in `configs/keep_variant.csv` (carbapenemases and all `blaOXA`).
- `n_class_<class>` counts hits per AMRFinderPlus class (from the database `fam.tsv` and
  mutation tables). A hit in a multi-class (`A/B`) entry counts once in each class.

## Rebuild

```bash
make ncbi-pd          # NCBI Klebsiella metadata + SNP clusters
make hackathon-data   # known_amr, known_amr_columns, lineages, splits
make push-data RELEASE=2026-10-04-hackathon
```
