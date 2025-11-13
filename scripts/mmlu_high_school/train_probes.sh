ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
python3 "$ROOT/src/probes/probes_train.py" \
    --save_dir "$SCRATCH/mera-runs" \
    --save_cache_key "3000" \
    --dataset_names "mmlu_high_school" \
    --model_names "swiss-ai/Apertus-8B-Instruct-2509" \
    --nr_layers "32" \
    --token_pos "" "_exact" \
    --process_saes "False" \
    --transform_targets "True" \
    --save_name "hs"

