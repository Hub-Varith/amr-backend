#!/usr/bin/env bash
# Download a data release from S3 into data/processed/ and verify it. Run on an engineer's Mac.
#   make pull-data                      # the latest release
#   make pull-data RELEASE=2026-10-05   # a pinned release
# See docs/ENGINEER_SETUP.md.
set -euo pipefail

BUCKET="${G2M_BUCKET:-g2m-data-v1}"
TARGET_DIR="data/processed"
if [ -z "${RELEASE:-}" ]; then
  if ! RELEASE="$(aws s3 cp "s3://$BUCKET/releases/LATEST" -)"; then
    echo "ERROR: cannot read s3://$BUCKET/releases/LATEST. Check your AWS key, or no release exists yet."
    exit 1
  fi
fi

echo "Pulling release $RELEASE from s3://$BUCKET"
mkdir -p "$TARGET_DIR"
aws s3 sync --only-show-errors "s3://$BUCKET/releases/$RELEASE/" "$TARGET_DIR/"

if ! (cd "$TARGET_DIR" && shasum -a 256 --check --quiet SHA256SUMS); then
  echo "ERROR: checksum mismatch. A file is damaged or was edited by hand. Delete $TARGET_DIR and pull again."
  exit 1
fi

# The frozen splits live in git as well. The two copies must match (CLAUDE.md rule 7).
if git ls-files --error-unmatch "$TARGET_DIR/splits.parquet" >/dev/null 2>&1; then
  if ! git diff --quiet -- "$TARGET_DIR/splits.parquet"; then
    echo "ERROR: splits.parquet in release $RELEASE differs from the copy in git. Stop and tell the data owner."
    exit 1
  fi
fi

echo "OK: release $RELEASE verified in $TARGET_DIR"
cat "$TARGET_DIR/RELEASE"
