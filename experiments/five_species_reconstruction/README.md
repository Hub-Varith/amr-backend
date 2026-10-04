# Five-species reduced-read assembly batch

Flye 2.9.6 baseline, not model training. Five species × three nested read fractions
(25%, 50%, 100%), seed 42, one public read run per species. No MIC labels are used.
These are research inputs for future predictions of in-vitro susceptibility, not prescribing advice.

## Execution

The batch uses two CPU threads and one assembly at a time. Each assembly is
stopped if its observed process-tree RSS exceeds 4 GiB or it runs longer than
90 minutes. RSS is sampled every two seconds; this is not a hard OS memory cap.
Snakemake continues independent jobs after a failure. Re-run `run_batch.py` to
resume, after checking failures. `STATUS.md` updates every ten seconds while active;
`batch_status.json`, `status/*.json`, and `logs/` retain details. Do not launch a
second copy. A PID lock prevents ordinary accidental duplicate starts.

Flye uses --nano-raw conservatively because exact basecaller quality profiles for
the four new runs have not been established. --asm-coverage 30 limits reads used
in the initial disjointig stage to the longest approximate 30x, to reduce memory.
Later stages use all reads in each subset. This differs from the original E. coli
pilot: do not pool its results with this run. Genome sizes here are approximate
species-level constants, not sample-specific reference sequences.

## Data and evaluation

`samples.json` preserves ENA run/sample/study accessions, HTTPS URLs, expected
MD5s and read/base counts. Downloads must pass checksums and counts before
assembly. E. coli reuses the existing verified ONT tutorial reads. Four additional
compressed downloads total approximately 1.01 GB. Subsets/intermediates take more.
At least 12 GiB free disk space is required at launch.

The new samples were chosen as manageable public WGS Nanopore runs of the exact
species. They are convenience samples, not a representative validation cohort.
The four new species have no validated comparison reference wired into this batch.
Only assembly size, N50, and contig count are reported for them for now. These
cannot establish correctness. E. coli additionally uses the existing manufacturer
reference for alignment breadth/identity, with the limitations in the first pilot.

Source metadata: https://www.ebi.ac.uk/ena/portal/api/search
Flye: https://github.com/mikolmogorov/Flye/tree/2.9.6
Original E. coli: ../ecoli_reconstruction/README.md

The Mac must remain powered and awake with its lid open. The launch command uses
`caffeinate -i -w <batch_pid>` to prevent idle sleep only while the batch runs;
it does not change permanent power settings or prevent lid-close sleep.

Flye rejects spaces in read paths. The local run uses `/tmp/amr-five-species`, a symlink to this experiment directory, for its absolute input/output paths. Recreate this alias when moving or resuming the experiment after temporary files are cleared.
