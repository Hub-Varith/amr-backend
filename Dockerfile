# API image only. The bioinformatics tools (AMRFinderPlus, unitig-caller, mash)
# are not installed yet; add them when the prediction pipeline lands.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 10001 app
USER app

ENV G2M_CONFIGS_DIR=/app/configs \
    G2M_MODELS_DIR=/app/models \
    G2M_UPLOAD_DIR=/tmp/genome2mic_uploads

EXPOSE 8000
CMD ["uvicorn", "genome2mic.api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
