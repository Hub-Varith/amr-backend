#!/usr/bin/env bash
# Upload the handoff files in data/processed/ to S3 as a new frozen release. Run on the VM.
#   make push-data                      # release name = today's UTC date
#   make push-data RELEASE=2026-10-05b  # a second release on the same day
# See docs/ENGINEER_SETUP.md.
set -euo pipefail

BUCKET="${G2M_BUCKET:-g2m-data-v1}"
RELEASE="${RELEASE:-$(date -u +%Y-%m-%d)}"
SOURCE_DIR="data/processed"
RELEASE_URI="s3://$BUCKET/releases/$RELEASE"
# Only the DATA_CONTRACT.md handoff files. Genomes and tool output never leave the VM.
HANDOFF_PATTERNS=(
  labels.parquet label_counts.csv pairs_kept.csv
  known_amr.parquet known_amr_columns.csv
  lineages.parquet splits.parquet
  "unitigs_*.npz" "unitigs_*_rows.parquet" "unitigs_*_index.parquet"
)

# aws s3 ls exits non-zero when the prefix is empty, which is what we want here.
if aws s3 ls "$RELEASE_URI/" >/dev/null 2>&1; then
  echo "ERROR: release $RELEASE already exists. Releases never change; choose a new RELEASE name."
  exit 1
fi

staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT
# Later stages may not exist yet; copy whichever handoff files are present.
for pattern in "${HANDOFF_PATTERNS[@]}"; do
  for path in "$SOURCE_DIR"/$pattern; do
    if [ -f "$path" ]; then
      cp "$path" "$staging_dir/"
    fi
  done
done
for provenance in data/interim/tool_versions.json data/raw/download_manifest.json; do
  if [ -f "$provenance" ]; then
    cp "$provenance" "$staging_dir/"
  fi
done
if [ ! -f "$staging_dir/labels.parquet" ]; then
  echo "ERROR: $SOURCE_DIR/labels.parquet is missing. Run make labels first."
  exit 1
fi

git_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: uncommitted code changes. The release records commit $git_commit, which may not match."
fi
cat > "$staging_dir/RELEASE" <<EOF
release=$RELEASE
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=$git_commit
EOF
(cd "$staging_dir" && shasum -a 256 -- * > SHA256SUMS)

echo "Uploading release $RELEASE:"
cat "$staging_dir/SHA256SUMS"
aws s3 cp --recursive --only-show-errors "$staging_dir" "$RELEASE_URI/"
echo "$RELEASE" | aws s3 cp - "s3://$BUCKET/releases/LATEST"
echo "Done. Engineers get it with: make pull-data"
