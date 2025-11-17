#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -J arc_easy
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --environment=probes

# Usage (short):
#   sbatch --export=S=cache scripts/arc_easy/run.sbatch
#   sbatch --export=S=postprocess scripts/arc_easy/run.sbatch
#   sbatch --export=S=train_probes scripts/arc_easy/run.sbatch

set -euo pipefail
cd "$HOME/projects/apertus-probes"

# Available steps
STEPS=("cache" "postprocess" "train_probes")

# Get step from environment variable (short name: S)
STEP="${S:-}"

if [ -z "$STEP" ]; then
    echo "Error: S parameter is required"
    echo "Usage: sbatch --export=S=<step> scripts/arc_easy/run.sbatch"
    echo "Available steps: ${STEPS[*]}"
    exit 1
fi

# Validate step
if [[ ! " ${STEPS[*]} " =~ " ${STEP} " ]]; then
    echo "Error: Invalid step '$STEP'"
    echo "Available steps: ${STEPS[*]}"
    exit 1
fi

# Construct script path
SCRIPT_PATH="scripts/arc_easy/${STEP}.sh"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

echo "Running $STEP for arc_easy"
echo "Script: $SCRIPT_PATH"

# (optional) environment setup
# module load <stuff>    # or:
# source ~/.bashrc
# conda activate <env>

bash "$SCRIPT_PATH"

