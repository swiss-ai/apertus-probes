import os
from datasets import load_dataset

cache_dir = os.path.join(os.environ["SCRATCH"], "mera-cache", "datasets") + "/"
os.makedirs(cache_dir, exist_ok=True)

# Map HF dataset IDs -> on-disk folder names your code expects
datasets_to_cache = {
    # name used by your project : HF hub id
    "sms_spam":         "ucirvine/sms_spam",
    # "mmlu": "cais/mmlu"     
}

for name, hf_id in datasets_to_cache.items():
    target = f"{cache_dir}{name}.hf"
    print(f"→ Caching {hf_id} to {target}")
    ds = load_dataset(hf_id)
    ds.save_to_disk(target)
print("Done.")