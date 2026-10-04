# Hackathon data release (provisional)

Built 2026-10-04 to train within the hackathon deadline. **Not the final data.**
Replace it with the full pipeline (own AMRFinderPlus run, Mash/PopPUNK lineages) after.

| Release | Species with features | Genomes |
| ------- | --------------------- | ------- |
| `2026-10-04-hackathon` | KPNEU | 7,229 |
| `2026-10-04-hackathon-all5` (LATEST) | KPNEU, ECOLI, SAUR, PAER, ABAU | 28,170 |

The KPNEU rows, splits, and lineages are identical in both releases; `all5` only adds
species (and gene columns that are 0 for KPNEU).

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
3. **Some labelled genomes are missing**: NCBI has no AMR result for them. KPNEU 718 of
   7,947 (9%), ECOLI 1,717 of 17,799 (10%), SAUR 402 of 2,301 (17%), PAER 54 of 1,808 (3%),
   ABAU 180 of 1,386 (13%).
4. **Breakpoints:** checked for 15 KPNEU drugs only (`configs/breakpoints/README.md`). Other
   species' S/I/R calls are provisional.
5. **No DNA mutation table for PAER** in the AMRFinderPlus database, so PAER `n_class_*`
   counts use protein mutations only. Its `point_` columns are still present.

## Feature rules

- `MISTRANSLATION` calls (internal stop codon) are skipped. Partial calls count as present.
- `=POINT` → `point_<gene>_<mutation>`; everything else → `gene_<symbol>`.
- Beta-lactamase alleles collapse to the family (`blaSHV-12` → `gene_blashv`) except the
  families in `configs/keep_variant.csv` (carbapenemases and all `blaOXA`).
- `n_class_<class>` counts hits per AMRFinderPlus class (from the database `fam.tsv` and
  mutation tables). A hit in a multi-class (`A/B`) entry counts once in each class.

## Split drugs per species

Each species' test set (15-20% of its genomes, whole SNP clusters) must hold R and S for
three drugs, set as `hackathon_split_drugs` in `configs/species.yaml`:

| Species | Split drugs |
| ------- | ----------- |
| KPNEU | ceftriaxone, meropenem, ciprofloxacin |
| ECOLI | ceftriaxone, ciprofloxacin, gentamicin |
| SAUR | oxacillin, erythromycin, clindamycin |
| PAER | meropenem, amikacin, levofloxacin |
| ABAU | meropenem, amikacin, ciprofloxacin |

## Rebuild

```bash
make ncbi-pd          # NCBI metadata + SNP clusters for every species in configs/species.yaml
make hackathon-data   # known_amr, known_amr_columns, lineages, splits for all 5 species
make push-data RELEASE=<new-name>
```

NCBI files are `data/raw/ncbi_pd/<SPECIES>.{metadata,clusters}.tsv` plus `<SPECIES>.VERSION`.
`make ncbi-pd` skips a species whose files exist, so a rerun keeps the same NCBI version.
Versions in `2026-10-04-hackathon-all5`: KPNEU PDG000000012.2542, ECOLI PDG000000004.6341,
SAUR PDG000000073.1299, PAER PDG000000036.1804, ABAU PDG000000010.1967.
