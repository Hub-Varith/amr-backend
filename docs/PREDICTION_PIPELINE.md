# Prediction pipeline: one genome in, one report out

**Code:** `src/genome2mic/predict/` · **Entry points:** the API (`POST /v1/predict`), `make predict FASTA=...`,
`python -m genome2mic.predict.run_predict` · **Contract:** `DATA_CONTRACT.md` stages 3, 4, 5, 12;
`INTEGRATION.md` section 3.

These are predictions of in-vitro susceptibility, not prescribing advice.

## What happens to a genome

The model never reads DNA. It reads one row of known-AMR columns (`gene_*`, `point_*`, `n_class_*`),
the same columns as `known_amr.parquet`. The pipeline turns a FASTA into that row, runs the model,
and applies the call rules.

| # | Step | Code | Tool | Writes (work folder) |
| - | ---- | ---- | ---- | -------------------- |
| 1 | Assembly stats + QC rules | `genome_qc.py` | Python | `qc.json` |
| 2 | Species: nearest of the 5 references, distance <= 0.05 | `species_identifier.py` | Mash | (in `qc.json`) |
| 3 | Novelty: distance to the nearest training genome | `novelty_checker.py` | Mash | `novelty.json` |
| 4 | Known resistance genes and mutations | `amrfinder_runner.py` | AMRFinderPlus 4.2.7 `-O <organism> --plus` | `amrfinder.tsv` |
| 5 | Hits -> known-AMR row | `features/amrfinder_table.py` | Python | `known_amr.json` |
| 6 | MIC, 90% band, `p_active`, call | `mic_model.py` (MODEL_HANDOFF.md section 10) | model `models/<run>` | |
| 7 | Overrides, reasons, ranking | `call_overrides.py`, `drug_reasons.py`, `report_builder.py` | configs | `report.json` |

Uploads may be plain FASTA or gzipped FASTA (`.fasta.gz`, `.fa.gz`, `.fna.gz`). The API unpacks a
gzipped upload on arrival (`api/services/upload_validator.py`), with the 50 MB limit applied to the
upload and again to the unpacked genome; the CLI unpacks `.gz` into the work folder. The tools always
see plain FASTA.

A species outside the five stops after step 2 (`species: null`, `in_range: false`, no predictions).
A QC failure is reported (`qc_pass: false`) but does not stop the run (INTEGRATION.md 3.1).

## No train/serve skew in step 5

Training features came from NCBI Pathogen Detection's `AMR_genotypes` strings, which are AMRFinderPlus
results written as `symbol[=TAG]`. Step 5 writes the upload's AMRFinderPlus table back into that exact
string and passes it through the same `NcbiKnownAmrBuilder` that built `known_amr.parquet`, with the
same `configs/keep_variant.csv` and the same AMRFinderPlus database class tables (2026-08-07.1).

| AMRFinderPlus output | NCBI string | Training builder |
| -------------------- | ----------- | ---------------- |
| `Type` STRESS / VIRULENCE | not in `AMR_genotypes` | dropped |
| `Subtype` POINT, POINT_DISRUPT | `=POINT` | `point_<gene>_<mutation>` |
| `Method` INTERNAL_STOP | `=MISTRANSLATION` | skipped (broken gene) |
| `Method` PARTIAL_CONTIG_END* / PARTIAL* / HMM | `=PARTIAL_END_OF_CONTIG` / `=PARTIAL` / `=HMM` | gene present |
| anything else | bare symbol | gene present |

`tests/features/test_amrfinder_table.py` checks that the row equals what the training builder makes
from the same string.

## Overrides (CLAUDE.md overrides 1 and 2)

- `configs/natural_resistance.csv`: KPNEU ampicillin. The model is skipped; no MIC is shown.
- `configs/strong_markers.csv`: a matching AMRFinderPlus symbol forces likely_inactive. A rule is kept
  only if marker carriers were lab-R at least 95% of the time in the training labels (CLSI). That
  left out carbapenemase -> meropenem for KPNEU (93%) and ECOLI (61%), and mecA -> oxacillin (88%);
  the model still sees those genes. The measured share is in the file.
- `configs/drug_reasons.csv`: which hits are shown as `reasons` per drug (display only).

## Running it

Linux only (AMRFinderPlus and Mash have no Windows builds). Once:

```bash
# tools: conda/micromamba env with ncbi-amrfinderplus=4.2.7 mash=2.3 ncbi-datasets-cli, then `amrfinder -u`
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[model]"
aws s3 cp --recursive --profile g2m s3://g2m-data-v1/models/hub/all5_run1/ models/all5_run1/
make references        # 5 Mash species references -> data/references/references.msh
```

One genome: `make predict FASTA=sample.fasta` (prints the report, keeps the work folder).
API: `make api`, then upload on `POST /v1/predict`. Docker: `make docker-build && make docker-run`.

Settings (`G2M_` environment variables, `api/config.py`): `MODEL_RUN`, `REFERENCES_SKETCH`,
`AMRFINDER_DB`, `TOOL_THREADS`, `KEEP_WORK_FILES`.

## Checked on real genomes (2026-10-03, WSL Ubuntu, AMRFinderPlus 4.2.7, database 2026-08-07.1)

Six test-split genomes from release `2026-10-04-hackathon-all5`, downloaded from NCBI by assembly
accession, run through the whole pipeline. The pipeline's known-AMR row was compared with the row in
`known_amr.parquet` (built from NCBI's own AMRFinderPlus run):

| Genome | Species | Gene/point columns | Differences | Same calls | Seconds |
| ------ | ------- | -----------------: | ----------- | ---------: | ------: |
| 1284812.3 | KPNEU (blaKPC-2) | 13 | none | 29/29 | 11 |
| 1328363.3 | KPNEU | 5 | none | 29/29 | 44 |
| 562.104197 | ECOLI | 11 | +`gene_emrd` (newer AMRFinderPlus), one MIC moves 1 step | 25/25 | 12 |
| 1280.15939 | SAUR | 4 | none | 12/12 | 7 |
| 287.1002 | PAER | 10 | none | 14/14 | 18 |
| 470.1304 | ABAU | 21 | none | 17/17 | 6 |

The API path (`/ready` -> `POST /v1/predict` -> poll -> `/result`) returned a valid report in 9 s.
An empty FASTA returns `qc_pass: false` (`empty_assembly`) and no predictions instead of failing.

### Demo set: 125 genomes uploaded through the API as `.fasta.gz` (2026-10-03)

Hub's `demo_genomes.zip` (25 per species, all from the test split, lab results in `labels.csv`),
each uploaded to `POST /v1/predict` one at a time. Not a new accuracy claim: this split was already
scored (docs/TRAINING_PLAN.md).

- 125/125 jobs finished, 8 s per genome on average (12 tool threads). 123/125 species confirmed.
- Two genomes labelled KPNEU and PAER are 0.0517 and 0.0531 Mash distance from their single species
  reference (cutoff 0.05), so they are reported as not covered. They are likely close relatives
  (K. quasipneumoniae / variicola; the PA7-like P. paraeruginosa clade).
- Calls vs lab (CLSI, lab S/R from the measured MIC): lab-R called likely active 2/636; lab-S called
  likely inactive 24/610; 58% of drug results got a committed call, 96.4% of those correct.
  Both lab-R misses are one ECOLI with the chromosomal `ampC T-32A` promoter mutation
  (ceftriaxone, cefotaxime): the model underweights AmpC hyperproduction.
- Per-genome reports: `data/interim/demo_run/*.json`; table: `predicted_vs_lab.csv` (not in git).

## Known limits

1. **Novelty is not checked.** It needs a Mash sketch of the training genomes
   (`models/<run>/training_<SPECIES>.msh`), and the hackathon releases ship no genome sequences.
   Until then `nearest_training_distance` is null and `in_range` follows the species check only.
2. **AMRFinderPlus version.** Training features come from NCBI's runs (mixed versions); the pipeline
   runs 4.2.7. A few hits can differ (one extra efflux gene in 1 of 6 genomes above).
4. **The Docker image is not yet built and tested** (the checks above ran in a WSL micromamba env
   with the same tool versions).
3. **Unitigs** are not used: the current model was trained with `--no-unitigs`.
