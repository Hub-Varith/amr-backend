# Flye reduced-read consistency tests

Six comparisons completed using minimap2 asm5, primary alignments, one thread. These are comparisons between assemblies from nested read sets, not independent truth/reference accuracy. Shared errors can agree. Alignment breadth includes internal gaps; identity applies only to aligned sequence.

| Comparison | Larger assembly covered | Aligned identity |
|---|---:|---:|
| ABAU_25_vs_100 | 99.9259% | 99.9506% |
| ABAU_50_vs_100 | 99.9988% | 99.9798% |
| ECOLI_25_vs_100 | 99.9752% | 99.0823% |
| KPNEU_25_vs_50 | 100.0000% | 99.9818% |
| PAER_25_vs_100 | 95.4106% | 99.9541% |
| PAER_50_vs_100 | 99.9789% | 99.9942% |

KPNEU has no completed 100% assembly; its comparison is 25% versus 50%. SAUR has only the fragmented 25% assembly and cannot be compared. Failed assemblies were not retried.

None of the five sample identifiers in samples.json matches a BioSample in the current labels.parquet. No lab-MIC accuracy or resistance-gene recovery was measured. AMRFinderPlus is unavailable on the local command path.

The previous E. coli manufacturer-reference comparison is in ../side_validation/REPORT.md and has a reference-provenance caveat.

Next: extract AMR genes and mutations consistently from each assembly, compare feature recovery, and use matched read/reference/lab samples for MIC accuracy. High whole-genome alignment agreement alone does not establish AMR preservation.
