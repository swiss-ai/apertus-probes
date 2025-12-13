ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
python3 "$ROOT/src/cache/cache_run.py" \
    --cache_dir "$SCRATCH/mera-runs/processed_datasets/" \
    --save_dir "$SCRATCH/mera-runs/" \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --dataset_names mmlu_high_school \
    --nr_samples 3000 \
    --batch_size 1 \
    --n_devices 1 \
    --flexible_match \
    --overwrite \
    --device "cuda:0"

