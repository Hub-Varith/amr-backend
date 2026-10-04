.PHONY: install install-api install-data api test test-api test-data train download-ast labels genome-manifest genomes qc ncbi-pd hackathon-data push-data pull-data docker-build docker-run

install:
	pip install -e ".[model,test]"

install-api:
	pip install -e ".[test]"

install-data:
	pip install -e ".[data,test]"

# Request logging comes from RequestIdMiddleware, so uvicorn's access log is off.
api:
	uvicorn genome2mic.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000 --no-access-log

test:
	pytest

test-api:
	pytest tests/api

test-data:
	PYTHONPATH=src pytest tests/ingest tests/genomes tests/qc tests/features tests/splits

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
	--cores 16 --resources downloads=16 --keep-going --rerun-incomplete

genome-manifest:
	$(SNAKEMAKE) $(foreach species,$(SPECIES),data/raw/ncbi_assemblies_$(species).jsonl.gz)
	PYTHONPATH=src python -m genome2mic.genomes.run_manifest --species $(SPECIES) $(if $(PILOT),--pilot $(PILOT))

genomes:
	$(SNAKEMAKE) data/interim/references/references.msh qc_inputs

qc:
	PYTHONPATH=src python -m genome2mic.qc.run_qc

# Hackathon shortcut (KPNEU): NCBI Pathogen Detection AMR calls + SNP clusters. See docs/HACKATHON_DATA.md.
NCBI_PD_URL = https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Klebsiella/latest_snps
AMRFINDER_DB ?= $(lastword $(wildcard $(HOME)/miniforge3/envs/genome2mic-tools/share/amrfinderplus/data/2*))

ncbi-pd:
	mkdir -p data/raw/ncbi_pd
	version=$$(curl -s $(NCBI_PD_URL)/Metadata/ | grep -oE 'PDG[0-9]+\.[0-9]+' | head -1) && \
	curl -sf -o data/raw/ncbi_pd/kleb.metadata.tsv $(NCBI_PD_URL)/Metadata/$$version.metadata.tsv && \
	curl -sf -o data/raw/ncbi_pd/kleb.clusters.tsv $(NCBI_PD_URL)/Clusters/$$version.reference_target.cluster_list.tsv && \
	echo $$version > data/raw/ncbi_pd/VERSION && echo "NCBI Pathogen Detection $$version"

hackathon-data:
	PYTHONPATH=src python -m genome2mic.features.run_hackathon_data --species KPNEU --amrfinder-db $(AMRFINDER_DB)

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

# One shared model for every species x drug. Needs the contract files in data/processed.
train:
	python -m genome2mic.models.train_cli --processed-dir data/processed --out-dir models/multitask
