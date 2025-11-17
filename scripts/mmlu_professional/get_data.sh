#!/bin/bash
set -e

OUT_DIR="/iopsstor/scratch/cscs/$USER/mera-runs/processed_datasets"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets mmlu_professional \
    --overwrite

echo "[DONE] mmlu_professional exported to $OUT_DIR"
