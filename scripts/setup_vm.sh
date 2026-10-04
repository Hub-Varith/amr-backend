#!/usr/bin/env bash
# One-time setup of the data VM (Ubuntu, x86_64). Safe to run again. See docs/DATA_PIPELINE_PLAN.md.
#   bash scripts/setup_vm.sh
set -euo pipefail

DATA_DEVICE="${DATA_DEVICE:-/dev/nvme1n1}"
DATA_MOUNT="/data"
CONDA_DIR="$HOME/miniforge3"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="$CONDA_DIR/bin/conda"

echo "== 1/5 Data disk $DATA_DEVICE -> $DATA_MOUNT"
if ! mountpoint -q "$DATA_MOUNT"; then
  if [ -z "$(sudo blkid -s TYPE -o value "$DATA_DEVICE" || true)" ]; then
    echo "Disk has no filesystem. Formatting it."
    sudo mkfs.ext4 -q -L g2m-data "$DATA_DEVICE"
  fi
  sudo mkdir -p "$DATA_MOUNT"
  disk_uuid="$(sudo blkid -s UUID -o value "$DATA_DEVICE")"
  # nofail: the VM still boots if the disk is ever detached.
  grep -q "$disk_uuid" /etc/fstab || echo "UUID=$disk_uuid $DATA_MOUNT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
  sudo mount "$DATA_MOUNT"
fi
sudo chown "$USER" "$DATA_MOUNT"
# Genomes and tool output go on the big disk. data/processed/ stays in the repo because
# git tracks pairs_kept.csv and splits.parquet there.
mkdir -p "$REPO_DIR/data/processed"
for big_dir in raw interim; do
  mkdir -p "$DATA_MOUNT/g2m/$big_dir"
  link_path="$REPO_DIR/data/$big_dir"
  if [ ! -L "$link_path" ]; then
    if [ -e "$link_path" ]; then
      echo "ERROR: $link_path exists and is not a link. Move it to $DATA_MOUNT/g2m/$big_dir, then re-run."
      exit 1
    fi
    ln -s "$DATA_MOUNT/g2m/$big_dir" "$link_path"
  fi
done

# Ubuntu cloud images ship without make or the AWS CLI.
if ! command -v make >/dev/null; then
  sudo apt-get update -q && sudo apt-get install -y -q make
fi
if ! command -v aws >/dev/null; then
  sudo snap install aws-cli --classic
fi

echo "== 2/5 Miniforge"
if [ ! -x "$CONDA" ]; then
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  "$CONDA" init bash
fi

echo "== 3/5 Python env genome2mic"
if ! "$CONDA" env list | grep -q "^genome2mic "; then
  "$CONDA" create -y -q -n genome2mic python=3.11
fi
"$CONDA" run -n genome2mic pip install -q -e "$REPO_DIR[data,test]"

echo "== 4/5 Bio tool envs genome2mic-tools, genome2mic-mlst"
for env_file in tools mlst; do
  env_name="$(grep '^name:' "$REPO_DIR/workflow/envs/$env_file.yaml" | cut -d' ' -f2)"
  if ! "$CONDA" env list | grep -q "^$env_name "; then
    "$CONDA" env create -q -f "$REPO_DIR/workflow/envs/$env_file.yaml"
  fi
done
"$CONDA" run -n genome2mic-tools amrfinder -u

echo "== 5/5 Record tool versions"
tools_run="$CONDA run -n genome2mic-tools"
amrfinder_version="$($tools_run amrfinder --version)"
amrfinder_db="$($tools_run amrfinder --database_version 2>&1 | grep -i 'database version' | cut -d: -f2 | xargs)"
mash_version="$($tools_run mash --version)"
datasets_version="$($tools_run datasets --version | cut -d: -f2 | xargs)"
snakemake_version="$($tools_run snakemake --version)"
mlst_version="$($CONDA run -n genome2mic-mlst mlst --version | cut -d' ' -f2)"
mkdir -p "$REPO_DIR/data/interim"
cat > "$REPO_DIR/data/interim/tool_versions.json" <<JSON
{
  "amrfinder": "$amrfinder_version",
  "amrfinder_db": "$amrfinder_db",
  "mash": "$mash_version",
  "datasets": "$datasets_version",
  "snakemake": "$snakemake_version",
  "mlst": "$mlst_version"
}
JSON
cat "$REPO_DIR/data/interim/tool_versions.json"

echo "Setup done. Open a new shell, then: conda activate genome2mic"
