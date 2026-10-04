# E. coli reduced-read reconstruction pilot

This is a Flye assembly benchmark, not neural-model training and not a validated
MIC predictor. No pretrained DNA language model is assumed to recover unobserved
resistance genes. The output is an assembly based on observed read overlaps.
These are research inputs for future predictions of in-vitro susceptibility,
not prescribing advice.

## Baseline and data

- Flye 2.9.6, commit 886b8c17412cdf3a2868a28237bca6c5ad1da156.
- Use `--nano-raw`, following this older R10.3 tutorial's command; do not assume
  these are current R10 SUP reads suitable for the latest high-quality presets.
- Four CPU threads. No GPU or learned weights required.
- Reads: https://ont-exd-int-s3-euwst1-epi2me-labs.s3-eu-west-1.amazonaws.com/assembly_tutorial/ecoli.60x.r103.fastq.gz
- Tutorial: https://epi2me.nanoporetech.com/notebooks/Assembly_Tutorial.html
- Reference: E. coli FASTA extracted from the manufacturer's bundle at
  https://zymo-files.s3.amazonaws.com/BioPool/ZymoBIOMICS.STD.refseq.v3.zip
- The tutorial's historical reference URL returns 404. The manufacturer's v3
  reference is a replacement for the same mock-community organism, not a verified
  byte-identical copy of the historical reference. Reference discrepancies can
  contribute to measured differences.

The reference contains a 4,804,267 bp chromosome and a 120,874 bp plasmid. The
reference is used only for evaluation, not to guide assembly. Tutorial reads were
already selected by reference alignment from a mock community: this introduces
selection bias and does not evaluate mixed blood-culture samples, contamination,
modern chemistry, or a general collection of clinical isolates.

## Experiment

Seed 42, random nested read fractions 0.25, 0.5 and 1.0. These nominally represent
15x, 30x and 60x based on the source description. Actual read bases divided by
reference length are reported separately and are not uniform per-position coverage.
This is one isolate and one subsampling seed, not model selection or external
validation. There is no training/test split because no parameters are learned here.
Do not use these results as independent biological replicates.

## Reproduce

Install/build Flye as described at https://github.com/mikolmogorov/Flye/blob/flye/docs/INSTALL.md
and install Snakemake in an isolated environment. The pilot used Snakemake 9.27.0.
Compile Flye with its top-level `make` (not parallel top-level make: its bundled
samtools configuration can race against minimap2 compilation).

Place `reads.fastq.gz` and `reference.fasta` in this directory. From the repo root:

```bash
PYTHONPATH=src python3 -m genome2mic.partial.subsample experiments/ecoli_reconstruction/reads.fastq.gz experiments/ecoli_reconstruction/subsets --fractions 0.25 0.5 1 --seed 42
```

Then, from this experiment directory:

```bash
snakemake --cores 4 --config flye=/absolute/path/Flye/bin/flye minimap=/absolute/path/Flye/bin/flye-minimap2 python=python3
python3 -m unittest test_summary -v
```

Large raw data, tool builds, intermediate assemblies, and Snakemake state are
ignored by Git. The metrics and provenance snapshot can be committed separately.

## Interpreting metrics

- Reference covered: union of primary alignment spans divided by reference length.
  This measures aligned breadth, not the fraction of all bases proven correct.
- Aligned identity: matches / alignment block length from minimap2 `-c` primary PAF
  records. This is alignment-weighted and not an independent complete-genome score.
- Per-reference breadth: report the chromosome and plasmid separately.
- Contigs: number of assembled sequence pieces; fewer alone does not prove accuracy.

No structural-error benchmark, AMR gene/variant audit, learned calibration, MIC
prediction, or clinical evaluation is included. Later comparisons need multiple
isolates/seeds, AMR feature preservation, and ultimately lab MIC labels. Gaps must
remain unknown; reference sequence must not be copied into missing regions.

## Colab continuation

[Open the notebook](https://colab.research.google.com/drive/1Cr8UxLKwcoGO5MrEIot9vA_R2v0-r5Ho). A standalone copy is saved as `Ecoli_reconstruction_pilot.ipynb`.

The local 25% run completed. The local 100% run was stopped during indexing due to memory pressure; the 50% run had not started. The Colab notebook reruns all levels for a consistent cloud comparison. Local partial results are in RESULTS.md and benchmark_results.json.

Cloud setup, data checksum verification, metric tests, and subsampling succeeded. The assembly workflow was launched, but Colab subsequently reported `Runtime died` and could not reconnect. Its cause was not established. No completed cloud results were retrieved; do not treat the cloud comparison as completed. The local notebook copy now streams subprocess progress instead of buffering it.
