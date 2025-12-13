ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
python3 "$ROOT/src/probes/probes_train.py" \
    --save_dir "$SCRATCH/mera-runs/" \
    --dataset_names "mmlu_high_school" \
    --model_names "meta-llama/Meta-Llama-3-8B" \
    --nr_layers "32" \
    --token_pos "" "_exact" \
    --process_saes "False" \
    --transform_targets "True" \
    --save_name "hs"

