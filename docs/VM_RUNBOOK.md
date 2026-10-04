# VM runbook: real data from a cloud bucket to a trained model

How to run genome2mic end to end on one Linux VM against the real dataset (lab-only
BV-BRC AST results plus thousands of assembled genomes in a cloud bucket). Companion
to `CLAUDE.md` and `DATA_CONTRACT.md`; if anything here disagrees with the contract,
the contract wins.

These are predictions of in-vitro susceptibility, not prescribing advice. Nothing in
this runbook produces a clinical claim.

## 0. What was and was not tested

Written on a laptop with no bucket, no cloud credentials, no Docker daemon and none
of the bioinformatics tools installed. Treat every item in the second column as
"run `--dry-run` / `-n` first and read the output".

| Step | Status |
| ---- | ------ |
| `fetch` from a local directory / `file://` (copy, `--include`, manifest, symlinks, dry run) | Tested (`tests/ingest/test_fetch.py`) |
| `fetch` command construction for `gcloud`, `gsutil`, `aws` (incl. `--aws-profile`), `azcopy`, SAS-token redaction | Tested at the argv level only; **never run against a real bucket** |
| `gcloud storage rsync --exclude` regex semantics (relative-path match) | **Unverified assumption** - check with `--dry-run` on a tiny prefix |
| `azcopy sync --include-pattern` (matches file names, not paths) | **Unverified** |
| `Dockerfile.pipeline` (base tag, bioconda package names, `amrfinder -u` layer) | **Never built** |
| `workflow/Snakefile` command lines (`mash sketch`/`mash dist` against one reference `.msh`, `amrfinder -O ... --threads`, `synthetic=false`) | Checked with stand-in `mash`/`amrfinder` scripts (`tests/qc/test_snakefile.py`); **never run with the real tools** |
| `ingest`, `qc`, `known-amr`, `lineages`, `splits`, `unitigs`, `train`, `evaluate`, `report` | Run end to end on synthetic data only (`make demo`) |
| Worker-process pools in `qc`, `lineages`, `unitigs` (output identical for any thread count) | Tested on macOS (`spawn`); the Linux default (`fork`) path has not run here |
| `unitigs --unitig-backend unitig-caller` | **Never run** (tool not installed); output parsing tested on small hand-written tables only |
| Wall times and RAM figures below | Estimates from the code's complexity, not measurements |

## 1. Sizing the VM

Let `N` be the number of genomes you will pull and `n_train` the number of training
genomes of the largest species (roughly 0.8 x the genomes of that species).

### Disk

```
disk_GB ~= 50 + N x 10 MB x 1.5
           ^    ^         ^
           |    |         headroom for scratch, unitig matrices, model bundles
           |    5 MB raw FASTA + 5 MB interim/processed per genome (rule from the task brief;
           |    real AMRFinder/MLST/Mash output is KBs, ResFinder a few MB)
           OS, container image (~3-5 GB), AMRFinder DB, conda env
```

| N genomes | raw | data total (x2) | recommended disk |
| --------- | --- | --------------- | ---------------- |
| 1,000 | 5 GB | 10 GB | 100 GB |
| 10,000 | 50 GB | 100 GB | 250 GB |
| 50,000 | 250 GB | 500 GB | 1 TB |

Use a separate data disk mounted at `/data` (SSD; the per-genome stage is I/O heavy).
`data/raw` is never modified after fetch, so it can be remounted read-only later.

### RAM

The unitig stage (stage 8) is the only memory-heavy step; everything else fits in
16 GB.

| Stage | Memory model | Rule of thumb |
| ----- | ------------ | ------------- |
| `unitigs --unitig-backend kmer` (default, pure numpy) | distinct 31-mers of the species x 12 B (count table) + kept 31-mers x ~40-80 B (pattern keys and their sort) + the `genomes x patterns` int8 CSR. No `genomes x k-mers` matrix is built. | **Refuses more than `--unitig-max-kmer-genomes` (default 1,000) training genomes per species**: it re-encodes every FASTA up to three times. Raise the limit only knowingly. |
| `unitigs --unitig-backend unitig-caller` | Bifrost graph build, then one unitig vector per genome; the `.rtab` it writes is a dense text table (unitigs x genomes), so budget disk and parse time for it | `8 + 5 x n_train/1000` GB (unmeasured); 64 GB covers ~5,000 training genomes |
| `unitigs` query of non-training genomes (both backends) | the frozen k-mer set (8 B per kept k-mer + 4 B pattern id) memory-mapped once and shared by all worker processes | k-mer set size + ~0.2 GB per worker |
| `lineages` | `(n, 1000)` sketch table (8 B per hash, plus int32 ranks) + only the pairs with `d <= 0.005`; the pair stream is folded into a spanning forest every 20M pairs. No `n x n` matrix. | 50k genomes: ~2-3 GB peak (sketches + the one-off hash ranking) + up to ~0.5 GB of pairs; measured on 8k random synthetic sketches (laptop, 18 cores): 13 s, +0.4 GB |
| `train` | sparse `genomes x selected unitigs` (top-k 2000 per fold) + known-AMR | a few GB |

Worker processes receive large read-only arrays (k-mer sets, sketch ranks) as
memory-mapped files under `$TMPDIR`; on a VM with a small root disk point it at the
data disk first: `export TMPDIR=/data/tmp && mkdir -p "$TMPDIR"`.

Run the unitig stage one species at a time on big sets (`--species KPNEU`) with
`--unitig-backend unitig-caller`; the default `kmer` backend stops with a clear
error once `n_train` passes `--unitig-max-kmer-genomes`.

### CPU

The per-genome tools dominate: AMRFinderPlus (`--plus`, run with `--threads 4`) and
ResFinder take 1-4 min each per genome; `mlst` 5-20 s; `mash dist` ~1 s (the species
references are sketched once, not per genome). Budget about **5 CPU-minutes per
genome**:

```
snakemake_wall_h ~= N x 5 min / (60 x cores)        10k genomes on 32 cores ~= 26 h
                                                    10k genomes on 64 cores ~= 13 h
```

Every Snakefile rule declares `threads:` (amrfinder 4, the rest 1) and amrfinder
declares `mem_mb` (4000), so give Snakemake the whole machine and let it schedule:
`--cores $(nproc) --resources mem_mb=<RAM in MB minus ~8000>`. Override with
`--config amrfinder_threads=N amrfinder_mem_mb=M`.

The Python stages `qc`, `lineages` and `unitigs` use every core through worker
processes (results are identical for any core count); `unitigs --unitig-threads N`
caps them for that stage. `train` uses `--nthread` xgboost threads.

Suggested shapes (any provider): 1k genomes -> 8 vCPU / 32 GB; 10k -> 32-64 vCPU /
128 GB; 50k -> 64+ vCPU / 256 GB, or shard the Snakemake run across several VMs
sharing the data disk / bucket.

## 2. Install the toolchain

Two options. Either way, clone the repo on the VM (`git clone ... /opt/genome2mic`)
and work from that directory so `./configs` and `workflow/Snakefile` resolve.

### Option A - Docker image (recommended)

```bash
sudo apt-get update && sudo apt-get install -y docker.io     # Debian/Ubuntu
sudo usermod -aG docker "$USER" && newgrp docker

cd /opt/genome2mic
make docker-build-pipeline                                     # tools + AMRFinder DB, ~3-5 GB
# with the provider CLI inside the image (otherwise run `fetch` on the host, section 5):
make docker-build-pipeline PIPELINE_BUILD_ARGS="--build-arg INSTALL_GCLOUD=true"      # or INSTALL_AWSCLI / INSTALL_AZCOPY
```

The image (`Dockerfile.pipeline`) is `mambaorg/micromamba` + bioconda
`ncbi-amrfinderplus mash mlst unitig-caller snakemake-minimal` (+ `pyseer` with
`--build-arg INSTALL_PYSEER=true`), `pip install .[pipeline]`, `amrfinder -u` baked in,
non-root user `mambauser`, entrypoint `python -m genome2mic`. The AMRFinder DB
layer is the slow one (~0.5 GB download); pass `--build-arg AMRFINDER_DB_UPDATE=false`
to skip it and run `amrfinder -u` into a mounted volume instead.

Make the data root writable by the container user (uid 57439 in the micromamba
image), or run with `--user "$(id -u):$(id -g)"`:

```bash
sudo mkdir -p /data/g2m && sudo chown -R "$(id -u):$(id -g)" /data/g2m
alias g2m='docker run --rm -it --user "$(id -u):$(id -g)" -v /data/g2m:/data/g2m -v "$HOME/.config/gcloud:/tmp/gcloud:ro" -e CLOUDSDK_CONFIG=/tmp/gcloud genome2mic-pipeline'
alias g2m-snakemake='docker run --rm -it --user "$(id -u):$(id -g)" -v /data/g2m:/data/g2m --entrypoint snakemake genome2mic-pipeline'
```

(Drop the gcloud mount on GCE: `gcloud` inside the container finds the VM service
account through the metadata server with no files at all. For AWS mount `~/.aws`
read-only (e.g. `-v "$HOME/.aws:/tmp/aws:ro" -e AWS_CONFIG_FILE=/tmp/aws/config
-e AWS_SHARED_CREDENTIALS_FILE=/tmp/aws/credentials`) and pass `--aws-profile NAME`
to `fetch`, or pass `-e AWS_*`; for Azure pass `-e AZCOPY_AUTO_LOGIN_TYPE=MSI`.)

### Option B - micromamba on the host

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
sudo mv bin/micromamba /usr/local/bin/
micromamba create -y -n genome2mic -c conda-forge -c bioconda \
    python=3.11 pip ncbi-amrfinderplus mash mlst unitig-caller snakemake-minimal   # + pyseer
micromamba activate genome2mic
amrfinder -u                                   # database, ~0.5 GB
cd /opt/genome2mic && pip install -e ".[pipeline,test]"
python -m pytest tests -q                      # sanity: everything should pass without tools
```

Then `alias g2m='python -m genome2mic'` and `alias g2m-snakemake='snakemake'` and
read the rest of this runbook with those aliases.

## 3. Reference genomes (QC species check)

`qc` and the Snakefile `mash` rule need one reference FASTA per species under
`data/raw/references/<SPECIES>.fasta`. Accessions are in `configs/species.yaml`.
Fetch them once (no extra tool needed):

```bash
mkdir -p /data/g2m/data/raw/references
for pair in ECOLI:NC_000913.3 KPNEU:NC_016845.1 SAUR:NC_007795.1 PAER:NC_002516.2 ABAU:CP000521.1; do
  sp=${pair%%:*}; acc=${pair##*:}
  curl -fsSL "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=${acc}&rettype=fasta&retmode=text" \
    > "/data/g2m/data/raw/references/${sp}.fasta"
done
grep -c '>' /data/g2m/data/raw/references/*.fasta     # one or more records each
```

If you keep the references in the bucket instead, put them under
`<prefix>/references/` and `fetch` will pull them with everything else.

## 4. Authenticate to the bucket (no keys in the repo)

The `fetch` stage never reads, writes or logs credentials; it only runs the
provider CLI, which uses its own login state. Pick one row and do it **outside** the
repository checkout. Never commit, `scp` into the repo, or paste a key into a
config file.

| Provider | Preferred (no secret on disk) | Fallback | Env-var only |
| -------- | ------------------------------ | -------- | ------------ |
| GCS | VM service account with `roles/storage.objectViewer` on the bucket; `gcloud` picks it up automatically | `gcloud auth login` (device flow, state in `~/.config/gcloud`) | `GOOGLE_APPLICATION_CREDENTIALS=/etc/g2m/sa.json` (mode 600, outside the repo) |
| S3 | EC2 instance role with `s3:GetObject`, `s3:ListBucket` | a named profile: `aws configure --profile g2m` (or `aws configure sso --profile g2m`), then `fetch --aws-profile g2m` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION` |
| Azure Blob | Managed identity: `export AZCOPY_AUTO_LOGIN_TYPE=MSI` | `azcopy login` (device flow) | SAS token appended to the URL (see below) |

Verify before fetching:

```bash
gcloud storage ls gs://BUCKET/PREFIX/ | head          # GCS
aws s3 ls s3://BUCKET/PREFIX/ --profile g2m | head    # S3 (drop --profile with an instance role)
azcopy list "https://ACCOUNT.blob.core.windows.net/CONTAINER/PREFIX" | head   # Azure
```

S3 profiles: `--aws-profile NAME` sets `AWS_PROFILE=NAME` for the `aws s3 sync`
subprocess only (your shell is untouched); `export AWS_PROFILE=NAME` before running
`fetch` does the same for the whole shell. Only the profile *name* appears in logs,
the dry run and `fetch_summary.json`; key material never does. The AWS CLI prefers
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` from the environment
over any profile; `fetch --aws-profile` warns when they are set (naming the variables
only), so `unset` them if a profile is meant to win.

SAS tokens: `fetch` redacts the query string from its logs, from
`data/raw/fetch_summary.json` and from `--dry-run` output, but your **shell
history** is not redacted. Put the token in a variable first:

```bash
read -rs SAS                                           # paste, press Enter; not echoed
g2m fetch --root /data/g2m --source-uri "https://ACCOUNT.blob.core.windows.net/CONTAINER/PREFIX?${SAS}"
unset SAS
```

## 5. Fetch the raw layer

Expected bucket layout under the prefix you pass (confirm this with whoever
exported the data; see section 11):

```
<prefix>/
  ast_bvbrc.csv                 BV-BRC AST export (lab rows; `evidence` column present)
  ast_ncbi.csv                  optional
  genome_metadata.csv           optional but strongly recommended: genome_id,biosample,species,source,isolation_source,country,year
  genomes/<genome_id>.fasta     one assembly per genome (or .fna/.fa/.gz or genomes/<genome_id>/<file> - see below)
  references/<SPECIES>.fasta    optional (section 3)
```

Always dry-run first; it prints the exact provider command and touches nothing:

```bash
g2m fetch --root /data/g2m --source-uri gs://BUCKET/PREFIX --dry-run
g2m fetch --root /data/g2m --source-uri gs://BUCKET/PREFIX --include 'ast_*.csv' --dry-run   # AST only
g2m fetch --root /data/g2m --source-uri s3://BUCKET/PREFIX --aws-profile g2m --dry-run       # S3, named profile
```

Then the real sync. On a 10k-genome / 50 GB prefix expect 10-60 min depending on
bandwidth; the provider CLIs are incremental, so re-running after an interruption
only transfers what is missing:

```bash
g2m fetch --root /data/g2m --source-uri gs://BUCKET/PREFIX        # or az://ACCOUNT/CONTAINER/PREFIX
g2m fetch --root /data/g2m --source-uri s3://BUCKET/PREFIX --aws-profile g2m   # S3 (omit the profile with an instance role)
# equivalent: make fetch SOURCE_URI=gs://BUCKET/PREFIX ROOT=/data/g2m
```

What `fetch` does after the sync:

1. Counts files and bytes under `data/raw`, the genomes, and the `ast_*.csv` files;
   writes `data/raw/fetch_summary.json` (redacted command + counts).
2. Warns if there is no `ast_bvbrc.csv` / `ast_ncbi.csv` (other `ast_*.csv` names are
   **not** read by `ingest`; rename or symlink them).
3. Checks the genome layout. If files are not exactly
   `data/raw/genomes/<genome_id>.fasta` (`.fna`, `.fa`, gzip, or one sub-folder per
   genome) it does **not** rename anything; it writes
   `data/raw/genomes_manifest.csv` (`genome_id, path, bytes`) and reports it.
   `genome2mic.ingest.fetch.resolve_genome_paths(paths)` returns
   `genome_id -> Path` from that manifest (falling back to `genomes/*.fasta`), and is
   the helper the per-genome stages should adopt. **Today `qc`, `lineages`,
   `unitigs` and the Snakefile still call `paths.genome_fasta()`**, so for a
   non-canonical bucket run with `--link-canonical`, which adds
   `genomes/<genome_id>.fasta` symlinks next to the real files (gzip files are
   skipped - decompress them first):

   ```bash
   g2m fetch --root /data/g2m --source-uri gs://BUCKET/PREFIX --link-canonical
   ```

Include-pattern semantics differ per CLI (documented in
`src/genome2mic/ingest/fetch.py`): patterns are relative to the prefix and `*`
also matches `/` for `aws`/`gcloud`/local; `azcopy` matches file **names** only.
`gcloud`/`gsutil` have no include flag, so `fetch` passes one negative-lookahead
`--exclude` regex - inspect it in the dry run.

Where to run it: on the host VM if the provider CLI is already there (GCE and
Amazon Linux images ship `gcloud`/`aws`), or inside the container built with the
matching `INSTALL_*` build arg. The pure-Python local provider needs no CLI at all,
which also covers a bucket mounted with gcsfuse / s3fs / blobfuse:

```bash
g2m fetch --root /data/g2m --source-uri /mnt/bucket/PREFIX
```

## 6. Labels first: `ingest`

```bash
g2m ingest --root /data/g2m                 # minutes; writes data/processed/labels.parquet,
                                            # label_counts.csv, pairs_kept.csv, drop_log_ingest.csv
column -ts, /data/g2m/data/processed/label_counts.csv | less
column -ts, /data/g2m/data/processed/pairs_kept.csv
column -ts, /data/g2m/data/processed/drop_log_ingest.csv
```

Read the drop log before going on: every filter (non-lab evidence, unknown drug,
unknown method, missing standard, conflicting duplicates) reports its count. The
`pairs_kept` table decides which species x drug models can be trained (>= 50 R,
>= 50 S, >= 4 MIC levels).

Running `ingest` **before** the per-genome tools matters: the Snakefile reads
`labels.parquet` and only processes genomes that have a label, which can cut the
most expensive step substantially.

## 7. Per-genome tools: Snakemake

```bash
cd /opt/genome2mic
g2m-snakemake -s workflow/Snakefile -n --config root=/data/g2m synthetic=false | tail -20   # dry run: job counts
g2m-snakemake -s workflow/Snakefile --cores $(nproc) --resources mem_mb=$(( $(free -m | awk '/^Mem:/{print $2}') - 8000 )) \
    --keep-going --rerun-incomplete \
    --config root=/data/g2m synthetic=false configs_dir=/opt/genome2mic/configs
```

Inside the container use `-s /app/workflow/Snakefile` and `configs_dir=/app/configs`.
`synthetic=false` is required (the Snakefile accepts `false/true/0/1/no/yes`): it
otherwise looks for `data/raw/SYNTHETIC_DATA.md`, and a tool missing from `PATH`
turns its rule into a no-op that fails loudly only when no precomputed output
exists. Confirm all four tools resolve first (`which amrfinder mlst mash resfinder`).
Targets on the command line go **before** `--config` (everything after `--config` is
read as `key=value`).

Species ID: one job sketches every `data/raw/references/<SPECIES>.fasta` into
`data/interim/_references/references.msh` (`mash sketch`); each genome then runs
`mash dist references.msh <genome>`, so `mash.tsv` has one row per species reference
and `qc` keeps the nearest. Without `mash` (or references) the rule writes an empty
`mash.tsv` and `qc` falls back to its pure-Python sketch, quietly.

`genome_metadata.csv` is what gives AMRFinderPlus its `-O <organism>` (point
mutations are only called with it). Without that file `amrfinder` still runs but
`-O` is omitted; the `known_amr` features then lack `point_*` columns.

Output: `data/interim/<genome_id>/{amrfinder.tsv, resfinder/pheno_table.txt, mlst.tsv, mash.tsv}`.
Wall time: section 1 (about 5 CPU-min per genome).

Resume: Snakemake skips genomes whose outputs exist. After a crash or a VM
preemption run the same command again with `--rerun-incomplete`. To shard across
VMs, give each a subset via `--batch all=1/4` ... `--batch all=4/4`.

## 8. Python stages

Each stage writes its files and a `data/processed/drop_log_<stage>.csv`; run them in
order and read each drop log.

```bash
export TMPDIR=/data/tmp && mkdir -p "$TMPDIR"                    # worker processes share memory-mapped arrays here
g2m qc        --root /data/g2m                                   # ~1-3 CPU-s per genome, all cores (assembly stats + sketch when mash.tsv is empty)
g2m known-amr --root /data/g2m                                   # minutes
g2m lineages  --root /data/g2m                                   # sketches + all-pairs distances on all cores; sparse, no n x n matrix
g2m splits    --root /data/g2m                                   # seconds; FREEZES data/processed/splits.parquet
g2m unitigs   --root /data/g2m --species KPNEU --unitig-backend unitig-caller --unitig-threads $(nproc)   # one species at a time
g2m train     --root /data/g2m --nthread $(nproc)                # 5-fold CV x models x pairs: tens of minutes to hours per pair
g2m evaluate  --root /data/g2m                                   # minutes; VME first
g2m report    --root /data/g2m                                   # minutes; results/report.md + figures
```

Or, once you trust the layout, everything after fetch in one go (synth is skipped
automatically when `--source-uri` is given):

```bash
g2m run-all --root /data/g2m --source-uri s3://BUCKET/PREFIX --aws-profile g2m --link-canonical \
    --unitig-backend unitig-caller --nthread $(nproc)
```

Notes:

- `splits` refuses to overwrite `splits.parquet`. That is the point (contract rule 7).
  Do **not** pass `--force` to "fix" a downstream problem; fix the problem. Commit
  `splits.parquet` once it exists.
- `unitigs`: the default `kmer` backend is pure Python and stops with an error
  naming `--unitig-backend unitig-caller` / `--unitig-max-kmer-genomes` when a
  species has more than 1,000 training genomes (section 1). Use
  `--unitig-backend unitig-caller` on real data (`auto` picks it when installed).
  Querying the test genomes runs on all cores for either backend.
- `train --models b1_lookup aft_known` is a cheap first pass before the unitig
  model; `--no-lolo` skips leave-one-lineage-out runs; `--species` / `--drugs`
  train a shard of the kept pairs (separate jobs merge into `models/manifest.json`).
- Cores: `qc`, `lineages` and `unitigs` use worker processes on every core (output
  does not depend on the count); `train` uses `--nthread` xgboost threads. A script
  that calls these stages from Python (rather than `python -m genome2mic`) must
  guard its entry point with `if __name__ == "__main__":`; otherwise the stage warns
  and runs on one core.

### Resuming

Every stage is a pure function of the files before it. If stage *k* fails, fix the
input and re-run stage *k* only; later stages are re-run from there. `fetch` is
incremental; Snakemake is incremental; `train` rewrites `results/preds_*.parquet`
for the pairs it trains (a `--species`/`--drugs` shard leaves the other pairs and
merges `models/manifest.json`). Every test-set scoring is appended to
`results/test_ledger.csv`; re-running the identical configuration is fine, but a
pair scored on the test set by two different configurations fails the
"test set touched once" leakage check -- tune on CV folds, not by re-running.

## 9. Copy results back to the bucket

Copy `results/`, `models/` and the contract files under `data/processed/`; never
copy `data/raw` back (the bucket already has it) and never copy anything into the
repository:

```bash
RUN=$(date +%Y%m%d)_$(python -c "import json;print(json.load(open('/data/g2m/models/manifest.json'))['run_id'])")
gcloud storage rsync -r /data/g2m/results        gs://BUCKET/runs/$RUN/results
gcloud storage rsync -r /data/g2m/models         gs://BUCKET/runs/$RUN/models
gcloud storage rsync -r /data/g2m/data/processed gs://BUCKET/runs/$RUN/processed
# aws s3 sync <dir> s3://BUCKET/runs/$RUN/<name>   |   azcopy sync <dir> "https://...CONTAINER/runs/$RUN/<name>" --recursive
```

`results/report.md` carries the disclaimer and reports VME first; keep it that way
when sharing.

## 10. Leakage checklist - run before reporting anything

From `DATA_CONTRACT.md` section 4, with the file to look at for each:

- [ ] **Unitig set built on training genomes only.**
      `data/processed/unitigs_<SP>_rows.parquet`: every row with `role == 'built'` has
      `split == 'train'`; `report` repeats this check and prints pass/fail. The build
      set is every `split == 'train'` genome, so CV validation folds and LOLO
      held-out lineages helped define the k-mer universe and patterns (no labels are
      read); only test rows are `queried`. CV/LOLO numbers therefore carry slight
      transductive optimism; the test split does not.
- [ ] **Frequency filter and pyseer/correlation selection redone inside every fold.**
      `train` does this per fold (`features/select.py`); no selection step runs on
      the full table.
- [ ] **Genomes de-duplicated by `biosample` across BV-BRC and NCBI.**
      `drop_log_ingest.csv` shows the biosample re-key count; `labels.parquet` has
      no biosample under two `genome_id`s.
- [ ] **No `country`, `year`, `source`, `isolation_source`, `lineage_cluster`, `st`,
      `biosample`, `split`, `fold` among features.**
      `models/<SP>/<drug>/features.json` lists only `gene_*`, `point_*`, `n_class_*`,
      `u_*` columns.
- [ ] **Thresholds and conformal calibration fitted on validation folds only.**
      `models/<SP>/<drug>/conformal.json` is computed from `split == 'cv'` rows.
- [ ] **Test set touched once, at the end.** One `train` run produces the test
      predictions; do not iterate on hyper-parameters after looking at
      `split == 'test'` metrics. `results/test_ledger.csv` must show one `run_id`
      per species x drug; `report` checks it and prints pass/fail.
- [ ] **No lineage cluster spans two splits or two folds.** `splits` raises
      `ContractViolation` otherwise; `splits.parquet` is committed and unchanged
      (`git status data/processed/splits.parquet`).
- [ ] **No label imputation.** `labels.parquet` rows exist only where a lab result
      exists; `drop_log_ingest.csv` accounts for everything removed.
- [ ] **Lab-measured only.** `drop_log_ingest.csv` has a non-zero (or explicitly
      zero) count for `evidence != Laboratory Method`.
- [ ] **Data was not synthetic.** `data/raw/SYNTHETIC_DATA.md` does not exist under
      the root; the report has no synthetic banner.

## 11. Assumptions about the bucket to confirm with the data owner

1. The prefix holds `ast_bvbrc.csv` (lab rows, with an `evidence` column) at its top
   level and genomes under `genomes/`. Other layouts need `--include` patterns or a
   different `--source-uri`.
2. Genome file names are `<genome_id>.<fasta|fna|fa>[.gz]` or
   `genomes/<genome_id>/<one file>`, where `<genome_id>` matches the `genome_id`
   column of the AST export (BV-BRC `573.2002` style). If the bucket uses assembly
   accessions instead, a `genome_metadata.csv` mapping is required before `qc`.
3. `genome_metadata.csv` (`genome_id, biosample, species, source, isolation_source,
   country, year`) exists or can be exported; without it AMRFinderPlus runs without
   `-O`, biosample de-duplication cannot run, and QC species checks rely on Mash alone.
4. `az://` is interpreted as `az://<account>/<container>/<prefix>`; pass the
   `https://<account>.blob.core.windows.net/<container>/<prefix>` form if that is
   not how the Azure storage is organized.
5. The VM identity (service account / instance role / managed identity) can read the
   bucket, so no key files are needed anywhere.
