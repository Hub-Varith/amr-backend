# Genome completion v3 — frozen trained model

This directory contains the **actual trained weights** and a standalone inference runtime for ECOLI, KPNEU, SAUR, PAER and ABAU. It exports the validation-selected v3 system from `develop-bhavya` commit `3d3223a`; it does not retrain or choose new models using the final test results. The `bhavyak` package adds the strict public-output confidence gate described below.

Given one continuous chromosome fragment and its observed percentage, it preserves those bases and appends an inferred candidate suffix. This is reference-assisted completion, not a guarantee that unseen strain-specific DNA is correct. Plasmids are excluded.

The existing MIC API and data pipeline are unchanged. This module is a separate research tool; its inferred DNA has not been validated as input to the antibiotic model. These are predictions of in-vitro susceptibility, not prescribing advice, where the separate MIC pipeline is used.

## Files included in Git

- `models/rankers.joblib`: frozen species-specific candidate selectors.
- `models/confidence.joblib`: calibrated stateless success classifier.
- `models/adaptive_confidence.joblib`: calibrated incremental classifier using newly revealed DNA.
- `core.py`, `pairwise.py`, `step_state.py`: unchanged inference code from the tested version.
- `predict.py`, `release_policy.py`: inference entrypoint with a strict 95% public-output gate.
- `experiment.py`, `adaptive_confidence.py`: inference-only helper functions exported verbatim; no training commands.
- `references.json`: pinned IDs, URLs, chromosome hashes, sketch hashes and groups for the 756 training references.
- `tools/minimap2-macos`: the exact tested Apple Silicon aligner, with its MIT license. This is the minimap2 executable only.
- `reports/`: results, limitations, provenance and an explanation for the team.

Only the approximately 3.9 GB reference cache and generated predictions are Git-ignored. **The three small trained model files are committed.** The cache is populated in the local `amr-backend-bhavyak` checkout; a fresh clone needs the setup step below. No files in another worktree are required by inference.

## Setup on another machine

Use a separate Python 3.12 environment so the serialized scikit-learn objects use the same versions as training. The dependency pins are needed for model compatibility, rather than for the existing FastAPI service.

```sh
cd genome_completion_v3
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download only one species, or omit --species to restore all five.
python setup_references.py --species ECOLI
python setup_references.py --verify-only --species ECOLI
```

Setup downloads from the pinned NCBI sources, keeps one non-plasmid chromosome, reproduces its 19-mer sketch, and verifies both checksums. It accepts NCBI gzip FASTA and Datasets ZIP responses, with an accession-based fallback. A changed or corrupted cache fails verification instead of silently becoming new reference data. Completed files are reused.

Apple Silicon macOS uses the bundled, checksum-pinned minimap2 binary. On Linux or another architecture, install a compatible minimap2 2.24 executable and set `COMPLETION_MINIMAP2=/path/to/minimap2`. The transfer replay was verified on Apple Silicon; predictions on a different aligner build/platform must be checked before making equivalent performance claims. `COMPLETION_REFERENCE_DIR` can point to a different reference-cache directory.

## Predict from a continuous fragment

```sh
python predict.py --input observed.fasta --species ECOLI --percent 65 --output run/example
```

The caller supplies the input fraction; the model does not know the full chromosome length from the fragment. The benchmark used the exact known fraction. Robustness to errors in that supplied fraction is untested. Do not concatenate unrelated draft contigs or sequencing reads into a fake chromosome.

A successful invocation writes `prediction.json`. For partial inputs, `completed.fasta`
is released only if confidence is **at least 95%** and at least **ten independent
calibration groups** support that probability locally. Otherwise the response is:

```json
{"status":"no_result","decision":"no_result","message":"no result",
 "result":null,"sequence_file":null,"predicted_bases":0,
 "next_action":"request_more_sequence"}
```

The response also includes confidence, rejection reason and the next requested
percentage. Stale public FASTA output is removed before inference. Accepted
partial outputs have `status=accepted` and `sequence_file=completed.fasta`.
Consumers must check status and use only the declared sequence file.

A 100% input is passed through unchanged with `status=full_sequence_observed` and
null confidence. This is not successful reconstruction. Stateful runs keep an
internal guess under `.completion_internal/` beside the state file to compare
with the next newly observed segment. Internal guesses must never be sent to the
MIC model as accepted DNA.

Confidence means probability of missing-region recall AND precision >=95%; it
does not mean probability of a perfectly correct chromosome. Model weights,
calibration and the existing ten-group support requirement are unchanged.

For the incremental 20%, 30%, …, 90%, 100% schedule:

```sh
python predict.py --input observed20.fasta --species ECOLI --percent 20 --output run/session/p20 --state run/session/state.json
python predict.py --input observed30.fasta --species ECOLI --percent 30 --output run/session/p30 --state run/session/state.json
```

Each new input must extend the exact earlier prefix. Use a distinct output directory per step. `--fine-final-steps` adds 95% and 98% before 100%; keep this option consistent within a session. The caller supplies real additional observations; the system does not manufacture them.

## Measured performance and limits

On 50 fresh genomes (ten per species), with 65% observed and 35% hidden, mean hidden-region F1 improved from **73.33% to 89.54%** over v2. The mean improved for all five species. A second cut position scored 88.72%; the stricter order-sensitive diagnostic scored 85.11%.

**No withheld sequence was reconstructed exactly. The 95% confidence stopping rule stopped early on 0/50 genomes**, even with the smaller final increments. Hidden AMR-gene recall was 61.2%; curated-mutation recall was 61.1%. These results do not establish reliable antibiotic selection from inferred DNA.

The reference library and candidate construction account for most of the gain. The same-library support rule scored 89.66%, versus 89.54% for the frozen validation-selected system. The selected models were retained; the export does not make a new post-test choice.

Read [the full results](reports/OVERNIGHT_RESULTS.md) and [the explanation](reports/EXPLAINER.md). `TRANSFER_VERIFICATION.json` records the original pre-gate transfer replay (matching original predictions, not database truth); `BHAVYAK_VERIFICATION.json` records the gated transfer checks; `EXPORT_PROVENANCE.json` pins model and unchanged-source hashes. Model serialization is trusted project output; do not replace these files with untrusted pickle/joblib downloads.

```sh
python -m unittest test_core test_export test_release_policy -v
```

The requested two perfectly reconstructed, confidence-accepted database demo inputs have not been verified. Do not present the transfer replay as proof of exact genome reconstruction.
