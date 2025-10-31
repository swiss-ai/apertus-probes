ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
python3 "$ROOT/src/cache/cache_run.py" \
    --cache_dir "$SCRATCH/mera-cache/datasets/" \
    --save_dir "$SCRATCH/mera-runs/" \
    --model_name "swiss-ai/Apertus-8B-Instruct-2509" \
    --dataset_names mmlu_professional \
    --nr_samples 2000 \
    --batch_size 1 \
    --n_devices 1 \
    --flexible_match \
    --no-overwrite \
    --device "cuda:0"

