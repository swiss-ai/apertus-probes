#!/usr/bin/env python3
import argparse, os, sys, json, csv, io, re, tempfile, shutil
from typing import Iterable, Optional, Dict, List

# Core HF libs
from datasets import load_dataset, get_dataset_split_names
from huggingface_hub import list_repo_files, hf_hub_download

# ----------------------------
# Plan (no normalization; just download & convert to jsonl)
# ----------------------------
PLAN = {
    "rvv-karma/Math-QA": [None],
    "cais/mmlu": ["all"],
    "openai/gsm8k": ["main"],
    "allenai/ai2_arc": ["ARC-Challenge", "ARC-Easy"],
    "brucewlee1/truthfulqa-mc2": [None],
    "openai/openai_humaneval": [None],
    "ibm-research/acp_bench": [
        "acp_app_bool", "acp_app_mcq",
        "acp_areach_bool", "acp_areach_mcq",
        "acp_just_bool", "acp_just_mcq",
        "acp_land_bool", "acp_land_mcq",
        "acp_prog_bool", "acp_prog_mcq",
        "acp_reach_bool", "acp_reach_mcq",
        "acp_val_bool", "acp_val_mcq",
        "acp_app_gen", "acp_areach_gen",
        "acp_just_gen", "acp_land_gen",
        "acp_prog_gen", "acp_nexta_gen",
        "acp_reach_gen", "acp_val_gen",
    ],
    "EleutherAI/drop": [None],
}

PREFERRED_SPLITS = ["train", "validation", "dev", "auxiliary_train", "test"]

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def cfg_name(config: Optional[str]) -> str:
    return "default" if (config is None or str(config).strip().lower() in {"", "none"}) else str(config)

def out_path(out_dir: str, repo_id: str, config: Optional[str], split: str) -> str:
    ds_name = repo_id.split("/")[-1]
    return os.path.join(out_dir, ds_name, cfg_name(config), f"{split}.jsonl")

def write_jsonl(iterable: Iterable[Dict], fp: str) -> int:
    n = 0
    with open(fp, "w", encoding="utf-8") as f:
        for ex in iterable:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n

# ----------------------------
# datasets streaming path
# ----------------------------
def list_splits_via_datasets(repo_id: str, config: Optional[str]) -> List[str]:
    try:
        return get_dataset_split_names(repo_id, config_name=config)
    except Exception as ex:
        eprint(f"[INFO] Split listing via datasets failed for {repo_id}/{cfg_name(config)}: {ex}")
        return []

def try_export_streaming(repo_id: str, config: Optional[str], split: str, fp_out: str) -> bool:
    # Skip if exists
    if os.path.exists(fp_out):
        eprint(f"[SKIP] {split}: already exists -> {fp_out}")
        return True
    try:
        it = load_dataset(repo_id, name=config, split=split, streaming=True)
        ensure_dir(os.path.dirname(fp_out))
        eprint(f"[..] Writing {fp_out}")
        n = write_jsonl(it, fp_out)
        eprint(f"[OK] {split}: {n} rows")
        return True
    except Exception as ex:
        # This will catch "dataset scripts are no longer supported" / remote code required / etc.
        eprint(f"[INFO] Streaming load failed for {repo_id}/{cfg_name(config)}:{split}: {ex}")
        return False

# ----------------------------
# hub-files fallback (no trust_remote_code)
# ----------------------------
SPLIT_PATTS = {
    "train": re.compile(r"(?:^|/)(?:train|training)\.(?:jsonl?|csv|tsv)$", re.I),
    "validation": re.compile(r"(?:^|/)(?:validation|valid|val)\.(?:jsonl?|csv|tsv)$", re.I),
    "dev": re.compile(r"(?:^|/)(?:dev|development)\.(?:jsonl?|csv|tsv)$", re.I),
    "auxiliary_train": re.compile(r"(?:^|/)(?:auxiliary[_\-]?train)\.(?:jsonl?|csv|tsv)$", re.I),
    "test": re.compile(r"(?:^|/)(?:test|testing)\.(?:jsonl?|csv|tsv)$", re.I),
}

# looser patterns if strict not found (catch things like data/train.json etc.)
ALT_PATTS = {
    "train": re.compile(r"(?:^|/)train[^/]*\.(?:jsonl?|csv|tsv)$", re.I),
    "validation": re.compile(r"(?:^|/)(?:validation|valid|val)[^/]*\.(?:jsonl?|csv|tsv)$", re.I),
    "dev": re.compile(r"(?:^|/)(?:dev|development)[^/]*\.(?:jsonl?|csv|tsv)$", re.I),
    "auxiliary_train": re.compile(r"(?:^|/)auxiliary[_\-]?train[^/]*\.(?:jsonl?|csv|tsv)$", re.I),
    "test": re.compile(r"(?:^|/)(?:test|testing)[^/]*\.(?:jsonl?|csv|tsv)$", re.I),
}

def find_repo_files_for_split(files: List[str], split: str) -> Optional[str]:
    # Prefer strict, then alternate
    patt = SPLIT_PATTS[split]
    for f in files:
        if patt.search(f):
            return f
    patt2 = ALT_PATTS[split]
    for f in files:
        if patt2.search(f):
            return f
    return None

def iter_json_file(path: str) -> Iterable[Dict]:
    # Accept either JSONL (one per line) or a JSON array
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
            for row in data:
                yield row
        else:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

def iter_csv_tsv_file(path: str) -> Iterable[Dict]:
    ext = os.path.splitext(path)[1].lower()
    delim = "," if ext == ".csv" else "\t"
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        for row in reader:
            yield row

def export_via_hub_files(repo_id: str, config: Optional[str], split: str, fp_out: str) -> bool:
    """
    Download raw files from the repo (no remote code), parse minimally, and write JSONL.
    Content is preserved as-is (row dicts unchanged).
    """
    if os.path.exists(fp_out):
        eprint(f"[SKIP] {split}: already exists -> {fp_out}")
        return True

    # If the dataset has configs, they’re often subfolders; we’ll try to match by prefix.
    # list_repo_files returns all repo paths; we’ll search for split-specific filenames.
    try:
        files = list_repo_files(repo_id)
    except Exception as ex:
        eprint(f"[ERROR] list_repo_files failed for {repo_id}: {ex}")
        return False

    # Narrow by config folder if present
    cfg = cfg_name(config)
    # Many repos use e.g. data/{split}.ext or {config}/{split}.ext.
    if cfg != "default":
        files_in_scope = [f for f in files if re.search(rf"(?:^|/){re.escape(cfg)}(?:/|$)", f)]
        # If nothing under config, fall back to all files
        if not files_in_scope:
            files_in_scope = files
    else:
        files_in_scope = files

    target = find_repo_files_for_split(files_in_scope, split)
    if target is None:
        # As a last resort, search globally
        target = find_repo_files_for_split(files, split)
    if target is None:
        eprint(f"[SKIP] hub fallback: no file matched split '{split}' in {repo_id}")
        return False

    # Download file and stream-convert
    try:
        local_fp = hf_hub_download(repo_id, filename=target)
    except Exception as ex:
        eprint(f"[ERROR] hf_hub_download failed for {repo_id}:{target}: {ex}")
        return False

    ext = os.path.splitext(local_fp)[1].lower()
    if ext in {".jsonl", ".json"}:
        iterator = iter_json_file(local_fp)
    elif ext in {".csv", ".tsv"}:
        iterator = iter_csv_tsv_file(local_fp)
    else:
        eprint(f"[SKIP] unsupported file type for {repo_id}:{target}")
        return False

    ensure_dir(os.path.dirname(fp_out))
    eprint(f"[..] Writing {fp_out} (hub file: {target})")
    n = write_jsonl(iterator, fp_out)
    eprint(f"[OK] {split}: {n} rows (hub)")

    return True

# ----------------------------
# Orchestrator
# ----------------------------
def export_repo(repo_id: str, configs: List[Optional[str]], out_dir: str):
    for config in (configs or [None]):
        eprint(f"\n=== {repo_id} | config={cfg_name(config)} ===")

        # Choose candidate splits
        splits = list_splits_via_datasets(repo_id, config)
        if splits:
            # Keep a stable preferred order
            candidates = [s for s in PREFERRED_SPLITS if s in splits] + [s for s in splits if s not in PREFERRED_SPLITS]
        else:
            candidates = PREFERRED_SPLITS  # we'll probe common ones

        any_ok = False
        for split in candidates:
            fp_out = out_path(out_dir, repo_id, config, split)

            # 1) Try streaming via datasets (no Arrow, no cache)
            ok = try_export_streaming(repo_id, config, split, fp_out)
            if ok:
                any_ok = True
                continue

            # 2) Fallback via hub files (handles math_qa etc., no trust_remote_code)
            ok2 = export_via_hub_files(repo_id, config, split, fp_out)
            any_ok = any_ok or ok2

        if not any_ok:
            eprint(f"[WARN] No splits exported for {repo_id}/{cfg_name(config)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to write JSONL files")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    print(f"[OUT] {out_dir}")

    for repo_id, configs in PLAN.items():
        export_repo(repo_id, configs, out_dir)

    print(f"\n[DONE] All datasets exported to {out_dir}")

if __name__ == "__main__":
    main()
