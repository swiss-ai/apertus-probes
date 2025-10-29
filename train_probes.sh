python3 MERA-steering/src/probes/probes_train.py \
    --save_dir "$SCRATCH/mera-runs/" \
    --save_cache_key "1000" \
    --dataset_names "sms_spam" \
    --model_names "swiss-ai/Apertus-8B-Instruct-2509" \
    --nr_layers "32" \
    --token_pos "" "_exact" \
    --process_saes "False" \
    --transform_targets "True" \
    --save_name "demo"