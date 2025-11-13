ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
python3 "$ROOT/src/cache/cache_postprocess.py" \
    --save_dir "$SCRATCH/mera-runs" \
    --save_cache_key "3000" \
    --dataset_name "mmlu_high_school" \
    --model_name "swiss-ai/Apertus-8B-Instruct-2509" \
    --nr_layers "32" \


