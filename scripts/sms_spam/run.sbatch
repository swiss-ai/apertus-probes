#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 10:00:00
#SBATCH -J sms_spam
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --environment=probes

# Usage (short):
#   sbatch --export=S=cache scripts/sms_spam/run.sbatch
#   sbatch --export=S=postprocess scripts/sms_spam/run.sbatch
#   sbatch --export=S=train_probes scripts/sms_spam/run.sbatch

export HF_TOKEN="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"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache"

set -euo pipefail
cd "$HOME/projects/apertus-probes"

STEPS=("get_data" "cache" "postprocess" "train_probes" "steer_1" "steer_2" "steer_3" "steer_4" "steer_5" "steer_arc_easy" "steer_arc_challenge" "steer_mmlu_high_school" "steer_mmlu_pro" "all")

STEP="${S:-}"

if [ -z "$STEP" ]; then
    echo "Error: S parameter is required"
    echo "Usage: sbatch --export=S=<step> scripts/sms_spam/run.sbatch"
    echo "Available steps: ${STEPS[*]}"
    exit 1
fi

if [[ ! " ${STEPS[*]} " =~ " ${STEP} " ]]; then
    echo "Error: Invalid step '$STEP'"
    echo "Available steps: ${STEPS[*]}"
    exit 1
fi

run_step() {
    local name="$1"
    local path="scripts/sms_spam/${name}.sh"
    echo "-----------------------------------------------------------"
    echo "Running: $name"
    echo "Script: $path"
    echo "-----------------------------------------------------------"
    bash "$path"
}

if [ "$STEP" = "all" ]; then
    run_step "get_data"
    run_step "cache"
    run_step "postprocess"
    run_step "train_probes"
    run_step "steer_1"
    run_step "steer_2"
    run_step "steer_3"
    run_step "steer_4"
    run_step "steer_5"
else
    run_step "$STEP"
fi

