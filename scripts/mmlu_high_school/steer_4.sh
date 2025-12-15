#!/usr/bin/env bash
# Script to run steering evaluation
# Usage: bash steer.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

export WANDB_MODE=offline

python3 "$ROOT/src/steering/steering_run.py" \
    --fname "mmlu_hs_4" \
    --cache_dir "/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/" \
    --save_dir "/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/steering_outputs/" \
    --model_names "meta-llama/Meta-Llama-3-8B" \
    --dataset_names "mmlu_high_school" \
    --steering_methods "optimal_contrastive" \
    --top_k_sets 50 100 200 \
    --probe_token_pos "exact" \
    --error_type "sm" \
    --objective_key "Accuracy" \
    --probe_file_name "df_probes_hs" \
    --nr_test_samples 250 \
    --nr_ref_samples 250 \
    --device "cuda:0" \
    --wandb_key "private"