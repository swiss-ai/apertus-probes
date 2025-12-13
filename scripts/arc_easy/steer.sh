#!/usr/bin/env bash
# Script to run steering evaluation
# Usage: bash steer.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"

export WANDB_MODE=offline

python3 "$ROOT/src/steering/steering_run.py" \
    --fname "mmlu_arc_easy_experiment" \
    --cache_dir "$SCRATCH/mera-runs/" \
    --save_dir "$SCRATCH/mera-runs/steering_outputs/" \
    --model_names "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_names "ARC-Easy" \
    --steering_methods "no_steering" "optimal_probe" "optimal_logistic_probe" "optimal_contrastive" "sub_optimal_probe" "median_optimal_probe" "additive_probe" "additive_sub_probe" "additive_median_probe" "additive_logistic_probe" "vanilla_contrastive" "prompt_steering" \
    --top_k_sets 50 100 200 \
    --probe_token_pos "exact" \
    --error_type "sm" \
    --objective_key "Accuracy" \
    --probe_file_name "df_probes_arc_easy" \
    --nr_test_samples 250 \
    --nr_ref_samples 250 \
    --device "cuda:0" \
    --wandb_key "private"