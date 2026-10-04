# Partial chromosome completion

The frozen five-species v3 completion model, including its trained weights, is in [genome_completion_v3](../genome_completion_v3/README.md). See that guide for reference setup, single-fragment prediction and incremental confidence.

This is a separate research module. The current genome-to-MIC API is unchanged; inferred DNA is not automatically passed into it. The model improved missing-region recovery, but did not meet its 95% confidence stopping goal and does not establish reliable antibiotic prediction from inferred DNA.

These are predictions of in-vitro susceptibility, not prescribing advice, when the separate MIC pipeline is used.

On `bhavyak`, partial predictions below 95% confidence return `no_result` with no completed FASTA. Acceptance also retains the ten-group calibration support requirement. Read the package output contract before integrating.
