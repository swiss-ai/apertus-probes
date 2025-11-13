#!/bin/bash
set -e

OUT_DIR="/iopsstor/scratch/cscs/tunguyen1/apertus/processed_datasets"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets arc_easy \
    --overwrite

echo "[DONE] ARC-Easy exported to $OUT_DIR"
