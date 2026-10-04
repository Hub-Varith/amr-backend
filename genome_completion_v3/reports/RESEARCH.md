# Research notes and decisions

Reviewed 2026-10-04. This project withholds a continuous chromosome segment. Its task differs from assembling overlapping reads that still cover most genomic positions.

## Alignment and reference construction

[Minimap2 methods](https://arxiv.org/abs/1708.01492) and its [official manual](https://lh3.github.io/minimap2/minimap2.html) describe seed chaining, alignment and assembly-divergence presets. A rough mapping endpoint should not automatically be treated as an exact nucleotide boundary. This motivated precise terminal/full-fragment alignment, circular reference handling and strand/origin tests.

Local pilot evidence, rather than the paper alone, selected the implementation. Approximate whole-fragment context was quicker but lost useful candidates. Full alignment to six retrieved groups with explicit length-anchored alternatives preserved more candidate quality at acceptable runtime. See `PILOT_COMPARISON.json`. These results do not prove the chosen recipe is globally optimal.

[Pandora's primary paper](https://doi.org/10.1186/s13059-021-02473-1) demonstrates the limitations of choosing a single bacterial reference and the benefit of a pan-genome representation for variation recovered from sequencing data. Its evaluation uses evidence from the sequenced sample. My inference: richer reference diversity is valuable here, but a graph alone does not reveal which entirely unobserved accessory genes or alleles this particular sample contains. Blindly mixing donor regions could create a plausible-looking chromosome with the wrong resistance alleles. V3 therefore retains alternative coherent paths and explicitly evaluates marker errors.

## Why not replace this run with a large DNA language model?

[Evo 2's primary paper](https://doi.org/10.1038/s41586-026-10176-5) reports long-context genomic modeling and generation, including million-token context models. Generating biologically plausible sequence and recovering the exact withheld segment from a particular isolate are different evaluation targets. The paper's generation results do not validate our requested 20%-input/95%-confidence stopping rule.

The [official Evo 2 implementation](https://github.com/arcinstitute/evo2) specifies Linux/WSL2 and NVIDIA CUDA; the larger variants require specialized hardware, while supported 7B variants can use bfloat16. The user's Mac is not its documented local runtime. My decision for this overnight experiment: improve and directly test the reference-assisted system rather than spend the window porting a much larger inference stack with an unestablished benefit for this target.

This is a practical decision, not evidence that genomic foundation models can never help. A later controlled experiment could compare their candidate scores or short-gap completions on a new validation cohort, with hardware/runtime costs included and without reusing this run's final test for model selection.

## Data and biological evaluation

[NCBI Datasets documentation](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/how-tos/genomes/download-genome/) provides assembly metadata and genome-download routes. The experiment saves selected accessions, download URLs, source checksums, preparation checksums, duplicate edges and split membership. Additional assemblies were selected before observing completion outcomes.

[AMRFinderPlus interpretation guidance](https://github.com/ncbi/amr/wiki/Interpreting-results) distinguishes identified genes and curated mutations from a phenotype prediction. Consequently, marker concordance is reported separately from DNA alignment scores. An empty point-mutation result is not proof that a genome has no mutations; gene-name agreement is not full gene-sequence correctness. MIC/antibiotic performance needs paired laboratory measurements and evaluation of the downstream model on the actual inferred outputs.

## Statistical decisions

The independent unit is a genome group, not a base or a repeated cut. Fresh validation, calibration and test groups are disjoint from all previously inspected genomes. Two training origins improve variation in cut positions; the final second-origin check reports robustness on the same 50 genomes rather than claiming 100 independent samples. Final candidate selection is saved before truth evaluation. Confidence concerns both >=95% missing-region recall and >=95% prediction precision, not an exactly correct genome or a clinically reliable antibiotic.
