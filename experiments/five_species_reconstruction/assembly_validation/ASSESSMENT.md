# Five-species assembly assessment

## New checks

All 11 completed assembly FASTAs were inspected: no invalid sequence characters or N bases. This is a format check, not base accuracy. Four of 15 assembly jobs exceeded the configured memory guard; no retries were run.

## KPNEU matched-reference comparison

Downloaded GCA_026256385.1 (ASM2625638v1), confirmed metadata BioSample SAMEA3357075 matches read run ERR10367347, and verified archive MD5 checksums. Reference sequencing technology: Illumina + Oxford Nanopore MinION. This is a same-sample published assembly, not independently proven ground truth; source reads may overlap.

| Reads used | Reference span covered | Aligned identity | Contigs |
|---|---:|---:|---:|
| 25% | 100.0000% | 99.97292% | 2 |
| 50% | 100.0000% | 99.98443% | 2 |

Reference breadth includes internal alignment gaps and is not 100% correctly recovered bases. Both reference records are spanned. Gene/mutation recovery and structural accuracy are untested.

## Species conclusions

- KPNEU: promising same-sample reference agreement at 25% and 50%; 100% run failed memory guard.
- PAER: 50% closely agrees with 100%; 25% spans only 95.41% of the 100% assembly. No matched reference validated.
- ABAU: 50% closely agrees with 100%; no matched reference validated.
- ECOLI: 25% versus 100% identity is 99.08%. Existing manufacturer-reference comparisons show substantial residual differences and have a reference-provenance caveat.
- SAUR: 25% output contains 356 contigs, N50 12,109 bp, total 3,910,977 bp; 50% and 100% runs failed memory guard. Investigate input/sample composition, coverage and assembly parameters before accepting it. Size alone does not diagnose contamination.

## Decision

Do not claim that assembly works well across all five species yet. These are one sample per species and one subsampling seed. Next validation needs matched references for the remaining species, AMR-gene and mutation concordance, and matched lab-MIC samples. No MIC validation was performed.

Sources: https://www.ncbi.nlm.nih.gov/biosample/SAMEA3357075 ; https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_026256385.1/

Artifacts: fasta_audit.json; kpneu_reference/provenance.json; kpneu_reference/00/metrics.json; kpneu_reference/01/metrics.json; prior comparisons in REPORT.md.
