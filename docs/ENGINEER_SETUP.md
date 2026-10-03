# Engineer setup — train on your Mac with S3 data

The VM builds the data and uploads it to S3 as a **release**: a dated, frozen set of
files. You download a release and train on it. You never upload data.

```
VM ──make push-data──► s3://g2m-data-v1/releases/<date>/ ──make pull-data──► your Mac ──► train
```

These are predictions of in-vitro susceptibility, not prescribing advice.

---

## One-time setup (about 15 minutes)

1. Get your AWS access key from Hub, in private. Never put it in the repo — the repo
   is public.
2. Install the AWS CLI and save the key under the profile `g2m`:

   ```bash
   brew install awscli
   aws configure --profile g2m
   #   AWS Access Key ID:     <your key id>
   #   AWS Secret Access Key: <your secret>
   #   Default region name:   us-east-1
   #   Default output format: json
   ```

3. Get the code and the Python env:

   ```bash
   git clone https://github.com/Hub-Varith/amr-backend.git
   cd amr-backend
   conda create -n genome2mic python=3.11 -y
   conda activate genome2mic
   make install-data
   ```

4. Get the data:

   ```bash
   make pull-data
   ```

   You should see `OK: release <date> verified in data/processed`.

## Every day

```bash
git pull          # newest code
make pull-data    # newest data release; only changed files download
```

Pin a release when you compare models: `make pull-data RELEASE=2026-10-05`.
The release you have is written in `data/processed/RELEASE`.

## What is in a release

| File | One row per | Use |
| ---- | ----------- | --- |
| `labels.parquet` | genome × drug | The answers: MIC interval `(mic_lower, mic_upper]` |
| `label_counts.csv`, `pairs_kept.csv` | species × drug | Which pairs have enough data |
| `known_amr.parquet` (+ `known_amr_columns.csv`) | genome | Known resistance genes and mutations (inputs) |
| `lineages.parquet` | genome | Lineage clusters — for splitting and evaluation only, **never** an input |
| `splits.parquet` | genome | `train` with `fold` 0–4, or `test` |
| `unitigs_<SPECIES>.npz` (+ `_rows`, `_index`) | genome | Sparse DNA-piece matrix (inputs); never densify it |
| `RELEASE`, `SHA256SUMS`, `*_manifest.json`, `tool_versions.json` | — | Where the data came from |

Later releases add files as the pipeline stages land. Column meanings are in
`DATA_CONTRACT.md`.

## Rules

1. Never edit files in `data/processed/` by hand. If the data is wrong, tell Hub.
2. Write the release name next to every result (`release=2026-10-05`).
3. Use `splits.parquet` exactly as given. Tune on the train folds. Touch `test` once,
   at the very end.
4. Never use `lineage_cluster`, `st`, `country`, `year`, `source`, or
   `isolation_source` as a model input.

## Sharing trained models

You may write only to your own folder, `models/<your-aws-user-name>/`. You can read
everyone's models.

```bash
aws s3 cp --recursive --profile g2m models/my_run/ s3://g2m-data-v1/models/<your-aws-user-name>/my_run/
aws s3 ls --profile g2m s3://g2m-data-v1/models/
```

Put a `README.txt` in each run folder with the release name, the git commit, and the
model settings.

## Memory

A Mac has 16 GB of RAM. KPNEU with unitigs should fit. If a model runs out of memory
(likely ECOLI with unitigs), ask Hub to run that one on the VM.

## When something fails

| Message | Fix |
| ------- | --- |
| `cannot read .../releases/LATEST` | Check `aws configure --profile g2m`, or no release exists yet |
| `checksum mismatch` | `rm -rf data/processed` and run `make pull-data` again |
| `splits.parquet ... differs from the copy in git` | Stop and tell Hub. Do not train |
