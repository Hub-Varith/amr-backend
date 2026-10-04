# API + prediction pipeline. AMRFinderPlus and Mash come from bioconda (Linux only), pinned to the
# versions in the data release's tool_versions.json. The trained model is mounted at /app/models.
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

USER mambauser
ENV G2M_CONFIGS_DIR=/app/configs \
    G2M_MODELS_DIR=/app/models \
    G2M_UPLOAD_DIR=/tmp/genome2mic_uploads \
    G2M_REFERENCES_SKETCH=/app/data/references/references.msh

EXPOSE 8000
CMD ["uvicorn", "genome2mic.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
