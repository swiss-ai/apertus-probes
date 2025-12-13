#!/bin/bash
set -e

OUT_DIR="/iopsstor/scratch/cscs/$USER/mera-runs"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets ARC-Challenge \
    --overwrite

echo "[DONE] ARC-Challenge exported to $OUT_DIR"
