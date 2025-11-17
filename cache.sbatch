#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -J mmlu_cache
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --environment=probes

set -euo pipefail
cd "$HOME/projects/apertus-probes"


bash ./scripts/mmlu_high_school/cache.sh