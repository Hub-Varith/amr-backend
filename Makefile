.PHONY: install install-data api test-api test-data download-ast labels genome-manifest genomes qc push-data pull-data docker-build docker-run

install:
	pip install -e ".[test]"

install-data:
	pip install -e ".[data,test]"

# Request logging comes from RequestIdMiddleware, so uvicorn's access log is off.
api:
	uvicorn genome2mic.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000 --no-access-log

test-api:
	pytest tests/api

test-data:
	PYTHONPATH=src pytest tests/ingest tests/genomes tests/qc

# Stage 1: raw AST + metadata from BV-BRC and NCBI into data/raw/ (about 3 minutes).
download-ast:
	PYTHONPATH=src python -m genome2mic.ingest.run_download

# Stage 2: labels.parquet, label_counts.csv, pairs_kept.csv, dropped_labels.parquet.
labels:
	PYTHONPATH=src python -m genome2mic.ingest.run_harmonize

# Stage 3, VM only. Run inside the genome2mic env; tool steps use genome2mic-tools.
# SPECIES=KPNEU (space-separated for several). PILOT=500 makes a throwaway test subset.
SPECIES ?= KPNEU
SNAKEMAKE = $${CONDA_EXE:-conda} run --no-capture-output -n genome2mic-tools snakemake -s workflow/Snakefile \
	--cores 16 --resources downloads=16 --keep-going --rerun-incomplete --quiet rules

genome-manifest:
	$(SNAKEMAKE) $(foreach species,$(SPECIES),data/raw/ncbi_assemblies_$(species).jsonl.gz)
	PYTHONPATH=src python -m genome2mic.genomes.run_manifest --species $(SPECIES) $(if $(PILOT),--pilot $(PILOT))

genomes:
	$(SNAKEMAKE) data/interim/references/references.msh qc_inputs

qc:
	PYTHONPATH=src python -m genome2mic.qc.run_qc

# VM only: upload the handoff files as a new frozen S3 release. Optional: RELEASE=<name>.
push-data:
	bash scripts/push_data.sh

# Engineers: download and verify a release. Uses the AWS profile g2m unless AWS_PROFILE is set.
pull-data:
	AWS_PROFILE=$${AWS_PROFILE:-g2m} bash scripts/pull_data.sh

docker-build:
	docker build -t genome2mic-api .

docker-run:
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		-v "$(CURDIR)/models:/app/models:ro" \
		genome2mic-api
