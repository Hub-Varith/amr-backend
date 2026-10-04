#!/usr/bin/env bash
# Deploy the API to a Hugging Face Docker Space (CPU basic, 16 GB RAM; Docker Spaces need HF PRO). docs/DEPLOY.md has the steps.
#
#   hf auth login                                  # once, with a write token
#   scripts/deploy_hf_space.sh <user>/dnagen-api   # first run creates the repos; later runs update them
#
# The Space is public, so the model is kept out of it: it goes to the private model repo
# <user>/dnagen-model, and the Space's Docker build downloads it with the HF_TOKEN secret (a read token
# added once in the Space settings). Hugging Face builds the image itself (~15 min the first time).
# Needs models/<MODEL_RUN> locally (gitignored; MODEL_HANDOFF.md section 10).
set -euo pipefail

SPACE="${1:-${HF_SPACE:-}}"
MODEL_RUN="${MODEL_RUN:-all5_run1}"
HF="${HF_CLI:-hf}"
FRONTEND_ORIGINS='["https://dnagen-app.vercel.app"]'
FRONTEND_ORIGIN_REGEX='https://dnagen-[a-z0-9-]+-mh-acks3\.vercel\.app'

if [ -z "$SPACE" ]; then
    echo "usage: $0 <user-or-org>/<space-name>" >&2
    exit 2
fi
MODEL_REPO="${MODEL_REPO:-${SPACE%%/*}/dnagen-model}"
cd "$(dirname "$0")/.."
if [ ! -f "models/$MODEL_RUN/spec.json" ]; then
    echo "no trained model at models/$MODEL_RUN (get it from S3; docs/MODEL_HANDOFF.md section 10)" >&2
    exit 1
fi

ROOT="$(pwd)"
# hf on Windows glob-expands '*' against the working directory; from an empty one it reaches the Hub as is.
EMPTY="$(mktemp -d)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$EMPTY" "$STAGE"' EXIT
hf_upload() { (cd "$EMPTY" && "$HF" upload "$@"); }

# 1. The model, in a private repo.
"$HF" repos create "$MODEL_REPO" --type model --private --exist-ok
hf_upload "$MODEL_REPO" "$ROOT/models/$MODEL_RUN" . --delete '*' --commit-message "Model run $MODEL_RUN"

# 2. The Space. Variables reach the Docker build as build args and the app as env vars.
#    Ignored if the Space already exists; change them under Settings -> Variables and secrets.
"$HF" repos create "$SPACE" --type space --sdk docker --no-private --exist-ok \
    -e "MODEL_REPO=$MODEL_REPO" \
    -e "MODEL_RUN=$MODEL_RUN" \
    -e "G2M_CORS_ORIGINS=$FRONTEND_ORIGINS" \
    -e "G2M_CORS_ORIGIN_REGEX=$FRONTEND_ORIGIN_REGEX" \
    -e "G2M_TOOL_THREADS=2"

if [ "${SKIP_UPLOAD:-}" = 1 ]; then
    echo "Repos ready. Add the HF_TOKEN secret: https://huggingface.co/spaces/$SPACE/settings"
    exit 0
fi

# 3. Code only: what the Dockerfile needs, an empty models/ (the build fills it), and the Space card.
cp Dockerfile .dockerignore pyproject.toml Makefile "$STAGE/"
cp -r src configs "$STAGE/"
mkdir -p "$STAGE/workflow" "$STAGE/models"
cp -r workflow/scripts "$STAGE/workflow/"
touch "$STAGE/models/.gitkeep"
cp deploy/hf-space/README.md "$STAGE/README.md"
find "$STAGE" \( -name __pycache__ -o -name '*.egg-info' \) -prune -exec rm -rf {} +

# --delete removes files from earlier deploys that are no longer staged.
hf_upload "$SPACE" "$STAGE" . --repo-type space --delete '*' \
    --commit-message "Deploy $(git rev-parse --short HEAD 2>/dev/null || echo local)"

SUBDOMAIN="$(echo "$SPACE" | tr '/_.' '---' | tr '[:upper:]' '[:lower:]')"
echo
echo "Space:  https://huggingface.co/spaces/$SPACE   (build logs under Logs)"
echo "API:    https://$SUBDOMAIN.hf.space            (set as VITE_API_BASE_URL on Vercel)"
