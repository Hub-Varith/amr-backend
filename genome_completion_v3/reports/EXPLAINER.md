# What this model does

## Input and output

Give the system one continuous piece of a bacterial chromosome, its species, and the fraction supplied. For example, 65% input means the model sees the first 65% of a chromosome rotated to a chosen starting point and predicts the remaining 35%. It preserves the input exactly and appends a candidate missing sequence. The benchmark knows the true fraction; robustness to an incorrectly supplied fraction has not been tested.

It handles E. coli, K. pneumoniae, S. aureus, P. aeruginosa and A. baumannii. This experiment excludes plasmids and does not turn unrelated draft contigs into a chromosome.

## How it makes the prediction

1. Compare DNA fingerprints to a library of complete reference chromosomes from NCBI RefSeq.
2. Retrieve six related reference groups. Align the observed DNA and its ends to each reference, including circular and reversed orientations.
3. Construct candidate missing sections from references with compatible boundaries. Explicit alternatives handle a missing usable boundary, with lower evidence features.
4. Select a candidate using the species-specific method that performed best on separate validation genomes. E. coli uses learned pairwise ranking; S. aureus and A. baumannii use boosted-tree quality models; K. pneumoniae and P. aeruginosa use the better-performing observed-support rule.
5. Estimate the chance that the selected missing section meets a defined recovery-and-precision target, using a separately calibrated classifier.

The learned part selects and assesses candidate DNA. It does not invent the exact strain-specific genome from nothing. An unseen insertion, deletion, plasmid, or mutation may have no evidence in the observed fragment or reference library.

## Why the data split matters

The collection includes 900 complete reference chromosomes after adding 500 assemblies, plus 125 earlier draft genomes used for duplicate screening. The reference bank has 756 chromosomes. Learning examples come from 40 training-query groups across five species, two starting positions, and seven input fractions; candidate choices create additional training rows but not additional independent genomes.

There are 25 fresh validation, 50 fresh calibration, and 50 fresh final-test genome groups. Final groups are disjoint from all previously inspected genomes under the recorded duplicate screen. Repeated input fractions from the same genome are not independent test samples.

## What a score means

Scores in the main comparison measure only the withheld 35%. They do not count copied input bases as a prediction success.

- Recall: how many withheld bases are recovered as exact matching bases in alignments.
- Precision: how much of the predicted sequence matches the withheld sequence, penalizing extra or wrong bases.
- F1: the balance between recall and precision.
- Exact recovery: the entire missing string is exactly correct. This is a stricter, separate outcome.

A 90% F1 score is not a 90% probability of an exactly correct genome. Alignment-derived scores also need separate checks for order, rearrangements and repeated sequence. The experiment reports an order-sensitive diagnostic and repeats the 65% condition at another starting position.

## Confidence and asking for more DNA

The target event is both withheld-base recall and prediction precision at least 95%. A reported probability estimates that event; it does not mean every nucleotide is correct. The primary schedule is 20%, 30%, …, 90%, then 100% observed if necessary. The system stops early only at a probability of at least 95% with nearby calibration support from at least ten genome groups.

An additional confidence experiment checks whether the preceding predicted section agrees with newly supplied DNA. The caller must actually supply more observed DNA; the system cannot create new observations itself. The final report states how often early stopping worked on unseen genomes. Reaching 100% input is fallback, not successful completion.

## Connection to antibiotic prediction

AMRFinderPlus separately compares known resistance-gene symbols and curated point-mutation calls in the true and inferred missing regions. This can reveal biologically important errors hidden by an average DNA score. These checks do not establish MIC prediction accuracy or the effectiveness of an antibiotic. Any downstream model needs its own paired evaluation using real susceptibility labels, including how inference errors change its output.

Use OVERNIGHT_RESULTS.md, PAIRED_COMPARISON.md and AMR_RESULTS.md for the measured results. Do not substitute validation or pilot scores for final-test results.
