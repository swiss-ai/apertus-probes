ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
python3 "$ROOT/src/cache/cache_postprocess.py" \
    --save_dir "$SCRATCH/mera-runs" \
    --dataset_name "ARC-Easy" \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --nr_layers "32" 

