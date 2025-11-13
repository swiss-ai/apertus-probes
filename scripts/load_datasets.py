import os
from datasets import load_dataset

cache_dir = os.path.join(os.environ["SCRATCH"], "mera-cache", "datasets") + "/"
os.makedirs(cache_dir, exist_ok=True)

# Map HF dataset IDs -> on-disk folder names your code expects
datasets_to_cache = {
    # name used by your project : HF hub id, second param for load_dataset
    "sms_spam":         ("ucirvine/sms_spam", None),
    "mmlu": ("cais/mmlu", "all"),
    "arc-easy": ("allenai/ai2_arc", "ARC-Easy")
}

for name, (hf_id, ds_arg) in datasets_to_cache.items():
    target = f"{cache_dir}{name}.hf"
    print(f"→ Caching {hf_id} ({ds_arg}) to {target}")
    if ds_arg is not None:
        ds = load_dataset(hf_id, ds_arg)
    else:
        ds = load_dataset(hf_id)
    ds.save_to_disk(target)
print("Done.")