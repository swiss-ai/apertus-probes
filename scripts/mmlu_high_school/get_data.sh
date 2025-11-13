#!/bin/bash
set -e

# === CONFIG ===
OUT_DIR="/iopsstor/scratch/cscs/tunguyen1/apertus/processed_datasets"

# === RUN ===
python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets mmlu_high_school \
    --overwrite

echo "[DONE] mmlu_high_school exported to $OUT_DIR"