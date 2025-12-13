#!/bin/bash
set -e

OUT_DIR="/iopsstor/scratch/cscs/$USER/mera-runs"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets yes_no_question \
    --overwrite

echo "[DONE] yes_no_question exported to $OUT_DIR"