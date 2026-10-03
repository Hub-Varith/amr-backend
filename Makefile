.PHONY: install api test-api test synth fetch pipeline demo demo-copy evaluate report clean-synth docker-build docker-run docker-build-pipeline

PYTHON ?= .venv/bin/python
ROOT ?= runs/synthetic
DEMO_DIR ?= reports/synthetic_demo
G2M = $(PYTHON) -m genome2mic
# make fetch SOURCE_URI=gs://bucket/prefix ROOT=/data/g2m [FETCH_ARGS="--include 'ast_*.csv' --dry-run"]
SOURCE_URI ?=
FETCH_ARGS ?=

install:
	pip install -e ".[test]"

# Request logging comes from RequestIdMiddleware, so uvicorn's access log is off.
api:
	uvicorn genome2mic.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000 --no-access-log

test-api:
	$(PYTHON) -m pytest tests/api -q

test:
	$(PYTHON) -m pytest tests -q

# --- synthetic end-to-end run (everything under $(ROOT) is SIMULATED data) ---------

synth:
	$(G2M) synth --root $(ROOT) --configs-dir configs

# --- real data: sync ast_*.csv + genomes/ from a bucket (or directory) into $(ROOT)/data/raw
# Credentials come from the provider CLI's own login (gcloud auth / aws configure / azcopy login),
# never from this repo. See docs/VM_RUNBOOK.md.
fetch:
	@test -n "$(SOURCE_URI)" || { echo "SOURCE_URI is required, e.g. make fetch SOURCE_URI=gs://bucket/prefix ROOT=/data/g2m"; exit 2; }
	$(G2M) fetch --root $(ROOT) --source-uri "$(SOURCE_URI)" $(FETCH_ARGS)

# ingest -> qc -> known-amr -> lineages -> splits -> unitigs -> train -> evaluate -> report
pipeline:
	$(G2M) ingest    --root $(ROOT)
	$(G2M) qc        --root $(ROOT)
	$(G2M) known-amr --root $(ROOT)
	$(G2M) lineages  --root $(ROOT)
	$(G2M) splits    --root $(ROOT)
	$(G2M) unitigs   --root $(ROOT)
	$(G2M) train     --root $(ROOT)
	$(G2M) evaluate  --root $(ROOT)
	$(G2M) report    --root $(ROOT)

evaluate:
	$(G2M) evaluate --root $(ROOT)

report:
	$(G2M) report --root $(ROOT)

# synth + pipeline on a fresh $(ROOT) (splits are frozen: run `make clean-synth` first to
# redo an existing root), then copy the report, metrics CSVs and figures into the tracked demo dir.
demo: synth pipeline demo-copy

demo-copy:
	mkdir -p $(DEMO_DIR)/figures
	cp $(ROOT)/results/report.md $(DEMO_DIR)/report.md
	cp $(ROOT)/results/metrics.csv $(DEMO_DIR)/metrics.csv
	cp $(ROOT)/results/metrics_by_distance.csv $(DEMO_DIR)/metrics_by_distance.csv
	cp $(ROOT)/results/figures/*.png $(DEMO_DIR)/figures/
	@echo "demo copied to $(DEMO_DIR) (SYNTHETIC data; not clinical validation)"

clean-synth:
	rm -rf $(ROOT)

docker-build:
	docker build -t genome2mic-api .

docker-run:
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		-v "$(CURDIR)/models:/app/models:ro" \
		genome2mic-api

# VM image with the bioconda tools (Dockerfile.pipeline). Optional CLIs via build args, e.g.
#   make docker-build-pipeline PIPELINE_BUILD_ARGS="--build-arg INSTALL_GCLOUD=true"
PIPELINE_BUILD_ARGS ?=
docker-build-pipeline:
	docker build -f Dockerfile.pipeline -t genome2mic-pipeline $(PIPELINE_BUILD_ARGS) .
