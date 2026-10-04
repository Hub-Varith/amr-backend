# Partial-DNA MIC experiments

Status: experimental preparation only. No trained models, biological benchmark,
or validated stopping rule yet. Start with E. coli as requested. This document
extends the research plan; it does not replace DATA_CONTRACT.md or change its
schemas, splits, label intervals, feature prefixes, or prediction API.

Goal: predict lab MIC from less observed sequencing data, without reconstructing
unobserved DNA. These are predictions of in-vitro susceptibility, not prescribing
advice. Confirm with standard AST; treatment decisions remain with clinicians.

## What is available now

A dependency-free FASTQ subsampler produces nested subsets and a JSON run manifest.
It does not extract AMR features or predict MIC. Synthetic tests check software
behavior only, not biological performance.

From the repository root:

```bash
PYTHONPATH=src python3 -m genome2mic.partial.subsample reads.fastq.gz results/read_subsets/sample_1 --fractions 0.01 0.05 0.1 0.25 0.5 1 --seed 42
PYTHONPATH=src python3 -m unittest discover -s tests/partial -v
```

The output directory must not already exist. Input is single-end, four-line FASTQ,
optionally gzipped. Paired-end/interleaved data are NOT supported: do not run mates
independently. A pair-aware extension is required before using paired-end datasets.
Inputs are validated before output creation. Failed writes can leave partial output;
only use a run after successful completion with its manifest present.

Random mode assigns one seeded random draw to each read, keeping smaller subsets
inside larger ones. Fractions are approximate read fractions, NOT genome coverage
or fractions of genome positions observed. Manifest counts record actual reads and
bases. Repeat experiments with several seeds; these are repeated observations of
the same isolate, not independent samples.

`--mode prefix` keeps the first ceil(fraction * total_reads) records. It approximates
read arrival only if the input preserves acquisition order. Neither mode estimates
elapsed sequencing time without timestamps. Neither makes an existing dataset's
sequencing errors or contamination representative of a different platform.

## Data needed

- FASTQ reads linked to an isolate/genome ID, with platform and acquisition metadata.
- Corresponding high-quality assembly for the full-genome comparison.
- Lab MIC labels cleaned under the existing contract (including bounded results).
- Existing frozen isolate/lineage splits, or their creation by the data pipeline.

Keep every subset of an isolate in that isolate's original split. Synthetic reads
from assemblies can later support debugging, but cannot alone validate real early
sequencing performance. The current CSVs supply labels, not sequencing reads.

## Planned comparisons (not implemented models)

1. Full-assembly reference: the contract's known-AMR XGBoost AFT model, followed by
   its known-AMR plus unitig variant when available.
2. Reduced-data baseline: assemble each subset, apply assembly QC, then the same
   frozen feature extraction/model. Record failed assemblies and abstentions rather
   than silently excluding difficult samples. Never feed FASTQ directly to an API
   that expects an assembly FASTA.
3. Partial-data-trained model: train AFT on features from training-isolate subsets,
   with explicit observation/coverage information. Choose how to represent an
   undetected versus unassessable feature before implementing a new feature schema.
   Weight/subsample repeated versions so isolates with more subsets do not dominate.
4. Later comparison: read-based marker/unitig evidence without assembly, after
   selecting and verifying a suitable read-level method for the sequencing platform.

Do not interpret an undetected marker as proven absent at low coverage. A partial
model requires validation/calibration under partial-data conditions; full-genome
uncertainty bands cannot simply be assumed to retain their coverage.

## Evaluation to implement once predictions and labels exist

Use validation folds for model choice, subset levels, calibration, and stopping-rule
selection. Freeze these before the final test. Follow the existing contract's
training-only vocabulary and within-fold selection rules.

Compare predictions to LAB answers, not just to full-genome predictions. Report VME
first when categorical interpretation is available (denominator: lab-resistant
isolates), then MIC agreement on suitable measured MICs, interval-aware evaluation
for bounded labels, uncertainty coverage/width, and the fraction of cases answered.
Do not replace <= or > labels with exact endpoint measurements. Report sample sizes
and confidence intervals, and retain failures/abstentions in reporting.

Plot performance versus actual read bases (and estimated coverage where justified)
per drug. A model answering only easy cases must be compared at matched answer rates.
Report latency separately. A stable prediction as reads accumulate is not proof
that it is correct. Repeatedly checking uncertainty and stopping early needs its own
held-out validation; a per-check 90% band is not a validated sequential guarantee.

## Next implementation checkpoint

Confirm available read type/platform and choose a small E. coli cohort with paired
lab labels. Build or obtain the full-genome baseline and feature workflow. Then wire
these subsets into that workflow and add real prediction evaluation. Until then,
there is no evidence to rank models or claim a minimum required amount of DNA.
