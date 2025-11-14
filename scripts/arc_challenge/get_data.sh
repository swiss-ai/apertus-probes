#!/bin/bash
set -e

OUT_DIR="/iopsstor/scratch/cscs/$USER/apertus/processed_datasets"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets arc_challenge \
    --overwrite

echo "[DONE] ARC-Challenge exported to $OUT_DIR"
