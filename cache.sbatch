#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 03:00:00
#SBATCH -J mmlu_cache
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --environment=probes

set -euo pipefail
cd "$HOME/projects/apertus-probes"

# (optional) environment setup
# module load <stuff>    # or:
# source ~/.bashrc
# conda activate <env>

bash ./scripts/mmlu_high_school/cache.sh