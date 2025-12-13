#!/usr/bin/env bash
set -euo pipefail

# Adjust these paths if needed
CACHE_DIR="/iopsstor/scratch/cscs/$USER/mera-runs/"
SAVE_DIR="/iopsstor/scratch/cscs/$USER/mera-runs/steering_outputs/"

MODEL_NAME="swiss-ai/Apertus-8B-2509"

# Evaluation dataset (where we steer)
DATASET_NAME="ARC-Challenge"
# Dataset where probes were trained
PROBE_DATASET_NAME="mmlu_high_school"

FNAME="arc_challenge_steer_from_mmlu_hs"

# Turn off wandb logging in this job
export WANDB_MODE=disabled

# List of steering methods (same as you used for your 4-group split)
STEERING_METHODS=(
  "no_steering"
  "optimal_probe"
  "optimal_logistic_probe"
  "additive_probe"
  "prompt_steering"
)

cd "$HOME/projects/apertus-probes"

python -m steering.steering_run \
  --fname "$FNAME" \
  --cache_dir "$CACHE_DIR" \
  --save_dir "$SAVE_DIR" \
  --device "cuda:0" \
  --steering_methods "${STEERING_METHODS[@]}" \
  --dataset_names "$DATASET_NAME" \
  --model_names "$MODEL_NAME" \
  --probe_dataset_name "$PROBE_DATASET_NAME" \
  --top_k_sets 50 100 200 \
  --probe_token_pos "exact" \
  --error_type "sm" \
  --probe_file_name "df_probes_hs" \
  --nr_test_samples 250 \
  --nr_ref_samples 250