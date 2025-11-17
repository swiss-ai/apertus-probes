#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -J arc_easy_cache
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --environment=probes

set -euo pipefail
cd "$HOME/projects/apertus-probes"

bash ./scripts/arc_easy/cache.sh

