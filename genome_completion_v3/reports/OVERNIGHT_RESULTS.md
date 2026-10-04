# Overnight completion-model results

All comparisons use genuinely fresh chromosome groups. The original observed prefix is preserved. Missing-region scores do not include those copied bases.

|Species|V1 F1|V2 F1|V3 F1|V3−V2|
|---|---:|---:|---:|---:|
|ALL|60.74%|73.33%|89.54%|+16.20 pp|
|ECOLI|30.84%|51.81%|84.61%|+32.80 pp|
|KPNEU|82.85%|82.21%|89.61%|+7.41 pp|
|SAUR|44.47%|70.67%|91.96%|+21.29 pp|
|PAER|74.61%|85.00%|92.69%|+7.69 pp|
|ABAU|70.93%|76.97%|88.80%|+11.83 pp|

This table is 65% observed / 35% missing. The comparison includes both reference-library and algorithm changes. See PAIRED_COMPARISON.md for genome-level uncertainty intervals.

The requested 20%, 30%, …, 90% confidence rule stopped early on 0/50 genomes; the rest required all observed sequence. Confidence measures the >=95% recall-and-precision event, not an exactly correct genome.
Fresh-test confidence Brier score: 0.1417. See FINAL_CALIBRATION.json for reliability bins.

Second-origin robustness check (same 50 genomes, a genome-specific randomized circular cut; no refitting):

|Species|Missing F1 at 65% observed|
|---|---:|
|ECOLI|84.38%|
|KPNEU|93.03%|
|SAUR|85.22%|
|PAER|91.43%|
|ABAU|89.52%|

AMR markers are evaluated separately in AMR_RESULTS.md. These are gene-symbol and curated mutation checks; no MIC or antibiotic-efficacy result is claimed. Plasmids are excluded.
Artifacts: models/rankers.joblib and models/confidence.joblib (local, Git-ignored), plus the prepared reference files listed in split.json. Use predict.py as documented in README.md.

## Additional verified checks

The confidence model selected before test used **history** features. Fresh-test Brier error was 0.1390, compared with 0.1417 without the newly revealed DNA check. Maximum test confidence was 57.3%. Both the requested schedule and the additional 95%/98% increments produced **0/50 early stops**. The 95% stopping goal was not established.

|Observed input|Mean missing-region F1|Genomes meeting both 95% recall and precision|Exact missing strings|
|---|---:|---:|---:|
|20%|89.70%|6/50|0/50|
|50%|88.43%|2/50|0/50|
|65%|89.54%|6/50|0/50|
|90%|90.82%|15/50|0/50|
|95%|90.54%|24/50|0/50|
|98%|92.93%|34/50|0/50|

Different input fractions leave different hidden regions; their F1 scores need not increase monotonically. These scores do not establish reliable full-genome recovery.
The order-sensitive forward-chain diagnostic scored 85.11% F1 at 65% input. It is conservative and may discard correct overlapping blocks; it is not a complete structural or copy-number validation.
At a second genome-specific randomized cut, mean missing-region F1 was 88.72%. These are repeated measurements on the same 50 genomes, not 50 additional independent genomes.

### Resistance-marker recovery in the hidden 35%

|Marker type|Version|True calls|Recovered|Extra predicted calls|Recall|Precision|
|---|---|---:|---:|---:|---:|---:|
|genes|v2|85|41|62|48.2%|39.8%|
|point_mutations|v2|36|16|9|44.4%|64.0%|
|disruptions|v2|2|0|2|0.0%|0.0%|
|genes|v3|85|52|43|61.2%|54.7%|
|point_mutations|v3|36|22|4|61.1%|84.6%|
|disruptions|v3|2|0|4|0.0%|0.0%|

These are AMRFinderPlus symbol-set and curated-mutation comparisons on 25 preselected genomes. They are not full allele-sequence validation. Boundary calls are reported separately; plasmids are excluded. No MIC or antibiotic-efficacy validation was performed.

At 65% input, the same-reference support rule scored 89.66% F1 versus 89.54% for the frozen validation-selected system. The learned selectors did not improve the aggregate final score over that rule. The whole-system gain over v2 therefore should not be attributed to training alone. This descriptive ablation did not change the selected models; see SELECTOR_ABLATION.json.

### Saved implementation and validation

- Seven focused regression tests pass, including circular/reversed sequence placement, cache reuse/input-change rejection, truth-independent selection, full-input behavior, and incremental state validation.
- Five standalone species CLI replays match the benchmark DNA; ten incremental CLI steps match the frozen confidence results.
- Frozen artifact checksums, reference requirements and sizes are in MODEL_ARTIFACTS.json. Trained binaries and reference sequences remain local and Git-ignored; code, manifests and reports are committed.
- NCBI additions: 500 complete assemblies, 100 per species. The total complete pool is 900 chromosomes; the inference reference bank contains 756. See extra_sources.json and split.json.
- Gains reflect the whole revised system: added references, improved candidate placement/construction, and species-specific selection. They are not an isolated claim about a new learner.

Use v3 for further completion experiments. Preserve the distinction between observed and inferred sequence when integrating with downstream models. Reliable recovery of the entire unseen strain-specific genome, and reliable antibiotic prediction from it, remain unproven.
