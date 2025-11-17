#!/bin/bash
set -e

OUT_DIR="/iopsstor/scratch/cscs/$USER/mera-runs/processed_datasets"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets ARC-Easy \
    --overwrite

echo "[DONE] ARC-Easy exported to $OUT_DIR"
