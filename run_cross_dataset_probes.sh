#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 03:00:00
#SBATCH -J cross_dataset_probes
#SBATCH -o logs/cross_dataset_probes-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --environment=probes

# Simple script to run cross-dataset probes for all combinations
# Usage: 
#   Direct execution: ./run_cross_dataset_probes.sh [model] [alphas] [token_pos]
#   With sbatch: sbatch run_cross_dataset_probes.sh [model] [alphas] [token_pos]

set -e  # Exit on first error

# Change to project directory if running under sbatch
if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "$HOME/projects/apertus-probes" || cd "$(dirname "$0")" || exit 1
fi

MODEL="${1:-llama-instruct}"
ALPHAS="${2:-0.02 0.05}"
TOKEN_POS="${3:-both}"

DATASETS=("sms_spam" "mmlu_high_school" "mmlu_professional" "ARC-Easy" "ARC-Challenge" "sujet_finance_yesno_5k")

counter=1
total=$(( ${#DATASETS[@]} * ${#DATASETS[@]} ))

# Run all combinations sequentially
# When run with sbatch, all run in one job on one node
# When run directly, all run locally
for train_ds in "${DATASETS[@]}"; do
    for test_ds in "${DATASETS[@]}"; do
        echo "========================================"
        echo "($counter / $total) [$train_ds → $test_ds]"
        echo "========================================"
        ./run_task.sh cross_dataset_probes \
            --model "$MODEL" \
            --train-dataset "$train_ds" \
            --test-dataset "$test_ds" \
            --alphas $ALPHAS \
            --token-pos "$TOKEN_POS" \
            --max-workers 60 \
            --save-name "linear"
        ((counter++))
    done
done

