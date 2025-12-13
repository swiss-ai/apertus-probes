#!/bin/bash
set -e

# === CONFIG ===
OUT_DIR="/iopsstor/scratch/cscs/$USER/mera-runs"
# OUT_DIR="/capstor/store/cscs/swissai/infra01/apertus_probes"

python3 "$(dirname "$0")/../download_datasets.py" \
    --out_dir "$OUT_DIR" \
    --datasets mmlu_high_school \
    --overwrite

echo "[DONE] mmlu_high_school exported to $OUT_DIR"