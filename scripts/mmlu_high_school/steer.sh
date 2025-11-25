#!/usr/bin/env bash
# Script to run steering evaluation
# Usage: bash steer.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

python3 "$ROOT/src/steering/steering_run.py" \
    --fname "mmlu_hs_experiment" \
    --cache_dir "$SCRATCH/mera-runs/processed_datasets/" \
    --save_dir "$SCRATCH/mera-runs/" \
    --model_names "swiss-ai/Apertus-8B-Instruct-2509" \
    --dataset_names "mmlu_high_school" \
    --steering_methods "optimal_probe" "optimal_contrastive" "no_steering" \
    --top_k_sets 50 100 200 \
    --probe_token_pos "exact" \
    --error_type "sm" \
    --objective_key "Accuracy" \
    --probe_file_name "df_probes_trans" \
    --nr_test_samples 250 \
    --nr_ref_samples 250 \
    --device "cuda:0" \
    --wandb_key "private"

