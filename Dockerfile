# syntax=docker/dockerfile:1
# API + prediction pipeline. AMRFinderPlus and Mash come from bioconda (Linux only), pinned to the
# versions in the data release's tool_versions.json. The trained model run is baked into the image:
# from models/ when present (gitignored; fetch it from S3, docs/MODEL_HANDOFF.md section 10), otherwise
# from a private Hugging Face model repo. docs/DEPLOY.md has the steps.
FROM mambaorg/micromamba:1.5.10

USER root
RUN micromamba install -y -n base -c conda-forge -c bioconda \
        python=3.11 ncbi-amrfinderplus=4.2.7 mash=2.3 ncbi-datasets-cli make unzip \
    && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The AMRFinderPlus database. Training class tables used 2026-08-07.1; the build warns if it differs.
ARG AMRFINDER_DB_VERSION=2026-08-07.1
RUN amrfinder -u \
    && (amrfinder --database_version | grep -q "$AMRFINDER_DB_VERSION" \
        || echo "WARNING: AMRFinderPlus database is not $AMRFINDER_DB_VERSION; n_class_* counts may differ from training")

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install ".[model,db]"

# Mash species references (configs/species.yaml), downloaded and sketched once at build time.
COPY configs ./configs
COPY workflow/scripts ./workflow/scripts
COPY Makefile ./
RUN make references

# Only the run the API serves (G2M_MODEL_RUN). Override MODEL_RUN to bake a different one.
# Local builds copy models/<run>. The Hugging Face Space ships no models/ content and instead downloads
# the run from the private repo MODEL_REPO, reading the HF_TOKEN build secret (Space variable + secret).
ARG MODEL_RUN=all5_run1
ARG MODEL_REPO=
COPY models/ ./models/
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    if [ ! -f "models/$MODEL_RUN/spec.json" ]; then \
        if [ -z "$MODEL_REPO" ] || [ ! -s /run/secrets/HF_TOKEN ]; then \
            echo "ERROR: no models/$MODEL_RUN, and no MODEL_REPO build arg + HF_TOKEN secret to download it" >&2; \
            exit 1; \
        fi; \
        pip install -q huggingface_hub \
        && HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['MODEL_REPO'], local_dir='models/' + os.environ['MODEL_RUN'], token=os.environ['HF_TOKEN'])" \
        && rm -rf "models/$MODEL_RUN/.cache"; \
    fi \
    && test -f "models/$MODEL_RUN/spec.json"

USER mambauser
ENV G2M_CONFIGS_DIR=/app/configs \
    G2M_MODELS_DIR=/app/models \
    G2M_UPLOAD_DIR=/tmp/genome2mic_uploads \
    G2M_REFERENCES_SKETCH=/app/data/references/references.msh \
    G2M_MODEL_RUN=${MODEL_RUN} \
    PORT=8000

EXPOSE 8000
# Hosts like Railway, Render and Fly set PORT; shell form so it expands. exec keeps uvicorn as PID 1 for clean shutdown.
# Proxy headers: hosts terminate TLS in front of the container, so trust X-Forwarded-Proto for https URLs.
CMD exec uvicorn genome2mic.api.main:create_app --factory --host 0.0.0.0 --port "$PORT" --no-access-log \
    --proxy-headers --forwarded-allow-ips '*'
