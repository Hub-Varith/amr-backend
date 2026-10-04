# E. coli reference comparison

Completed alongside the main batch, using one alignment thread. No training or MIC prediction.

| Reads used | Contigs | Reference alignment breadth | Aligned identity | Substitution differences | Inserted bases | Deleted bases |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 4 | 99.97527% | 98.83958% | 2,380 | 2,224 | 52,560 |
| 100% | 2 | 99.99990% | 99.09121% | 360 | 16,061 | 28,484 |

Reference alignment breadth counts alignment spans, including internal gaps; it is NOT the percentage of reference bases correctly reconstructed. Difference counts sum primary alignment records and are not deduplicated per genomic position. They are relative to the reference, not independently confirmed sequencing errors. Structural rearrangements and AMR features have not been evaluated.

Reference: manufacturer ZymoBIOMICS v3 E. coli chromosome and plasmid. This is not verified byte-identical to the historical reference for the ONT tutorial reads. Reference differences may contribute to these counts. See ../ecoli_reconstruction/README.md for provenance.

Interpretation: using all reads improves aligned identity over 25% in this run, but both retain substantial differences. Neither result establishes suitability for MIC prediction. The current baseline uses a 30x initial-assembly cap and two threads; earlier pilot numbers used different settings and must not be pooled.

Reproduce from the experiment directory:
```sh
/tmp/amr-benchmark-env/bin/snakemake --snakefile SideValidation.smk --directory side_validation --cores 1
python3 compare_ecoli.py
```

These are research inputs for future predictions of in-vitro susceptibility, not prescribing advice.
