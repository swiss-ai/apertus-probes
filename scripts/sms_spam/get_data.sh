#!/bin/bash
set -e

OUT_DIR="/iopsstor/scratch/cscs/tunguyen1/apertus/processed_datasets"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets sms_spam \
    --overwrite

echo "[DONE] sms_spam exported to $OUT_DIR"
