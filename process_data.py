import os
import json
import argparse
from glob import glob
from typing import Dict, List, Any

RAW_DIR = "/iopsstor/scratch/cscs/tunguyen1/apertus/huggingface_datasets"
OUT_DIR = "/iopsstor/scratch/cscs/tunguyen1/apertus/normalized_datasets"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# Helper to read / write JSONL
# ============================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(rows: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================
# Dataset-specific converters
# ============================

def convert_mmlu(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "mmlu",
            "topic": ex.get("subject", None),
            "context": None,
            "question": ex.get("question", ""),
            "choices": ex.get("choices", None),
            "answer": ex.get("answer", ""),
            "answer_type": "token",
        }
        out.append(item)
    return out


def convert_gsm8k(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "gsm8k",
            "topic": None,
            "context": None,
            "question": ex.get("question", ""),
            "choices": None,
            "answer": ex.get("answer", ""),
            "answer_type": "long",
        }
        out.append(item)
    return out


def convert_arc(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "arc",
            "topic": "Abstraction and Reasoning",
            "context": None,
            "question": ex.get("question", ""),
            "choices": (ex.get("choices") or {}).get("text", []),
            "answer": ex.get("answerKey", ""),
            "answer_type": "token",
        }
        out.append(item)
    return out


def convert_mathqa(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "mathqa",
            "topic": ex.get("topic", None),
            "context": None,
            "question": ex.get("question", ""),
            "choices": None,
            "answer": ex.get("answer", ""),
            "answer_type": "long",
        }
        out.append(item)
    return out


def convert_drop(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "drop",
            "topic": None,
            "context": ex.get("passage", ""),
            "question": ex.get("question", ""),
            "choices": None,
            "answer": ex.get("answers_spans", ""),
            "answer_type": "span",
        }
        out.append(item)
    return out


def convert_truthfulqa(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "truthfulqa",
            "topic": None,
            "context": None,
            "question": ex.get("centerpiece", ""),
            "choices": ex.get("options", None),
            "answer": ex.get("correct_options", []),
            "answer_type": "multiple answers MCQ",
        }
        out.append(item)
    return out


def convert_humaneval(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "humaneval",
            "topic": None,
            "context": ex.get("prompt", ""),
            "question": "What is the implementation of the function?",
            "choices": None,
            "answer": ex.get("canonical_solution", ""),
            "answer_type": "long",
        }
        out.append(item)
    return out


# --- ACP bench: three variants ---
def convert_acp_bench_bool(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "acp_bench",
            "topic": ex.get("group", None),               # or "domain"
            "context": ex.get("context", None),
            "question": ex.get("question", ""),
            "choices": ex.get("choices", None),           # often ["Yes","No"] or similar
            "answer": ex.get("answer", ""),
            "answer_type": "token",
        }
        out.append(item)
    return out


def convert_acp_bench_gen(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "acp_bench",
            "topic": ex.get("group", None),
            "context": ex.get("context", None),
            "question": ex.get("question", ""),
            "choices": ex.get("choices", None),           # usually None for gen
            "answer": ex.get("answer", ""),
            "answer_type": "long",
        }
        out.append(item)
    return out


def convert_acp_bench_mcq(raw: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for ex in raw:
        item = {
            "name": "acp_bench",
            "topic": ex.get("group", None),
            "context": ex.get("context", None),
            "question": ex.get("question", ""),
            "choices": ex.get("choices", None),           # list[str]
            "answer": ex.get("answer", ""),               # label or text
            "answer_type": "token",
        }
        out.append(item)
    return out


# ============================
# Routing helpers
# ============================

def pick_acp_converter(config_dir: str):
    """
    Decide converter based on the acp_bench config folder name.
    Examples:
        acp_app_bool       -> bool
        acp_app_mcq        -> mcq
        acp_app_gen        -> gen
        acp_reach_mcq      -> mcq
    Fallback: detect from a sample row if name doesn't contain a known suffix.
    """
    cfg = os.path.basename(config_dir).lower()
    if "gen" in cfg:
        return convert_acp_bench_gen
    if "mcq" in cfg:
        return convert_acp_bench_mcq
    if "bool" in cfg:
        return convert_acp_bench_bool
    # unknown config name: return None and let caller inspect a sample row
    return None


def infer_acp_converter_from_sample(sample: Dict[str, Any]):
    """
    Fallback heuristic if the config name doesn't contain bool/mcq/gen.
    """
    choices = sample.get("choices", None)
    if choices is None or choices == []:
        return convert_acp_bench_gen
    # if there are exactly 2 or they look like yes/no -> treat as bool
    lo = [str(c).strip().lower() for c in (choices or [])]
    if len(choices) == 2 or set(lo) & {"yes", "no"}:
        return convert_acp_bench_bool
    # otherwise assume general MCQ
    return convert_acp_bench_mcq


# ============================
# Main orchestration
# ============================

CONVERTERS = {
    "Math-QA": convert_mathqa,
    "mmlu": convert_mmlu,
    "gsm8k": convert_gsm8k,
    "ai2_arc": convert_arc,
    "drop": convert_drop,
    "truthfulqa-mc2": convert_truthfulqa,
    "openai_humaneval": convert_humaneval,
    # acp_bench handled separately due to multiple config families
}

def normalize_all():
    # 1) Regular datasets
    for repo_name, fn in CONVERTERS.items():
        pattern = os.path.join(RAW_DIR, repo_name, "**", "*.jsonl")
        files = sorted(glob(pattern, recursive=True))
        if not files:
            print(f"[SKIP] {repo_name} — no files found.")
            continue

        for fp in files:
            split = os.path.splitext(os.path.basename(fp))[0]
            print(f"[PROCESS] {repo_name} | split={split}")
            raw = read_jsonl(fp)
            out_fp = fp.replace(RAW_DIR, OUT_DIR)
            if os.path.exists(out_fp):
                print(f"[SKIP] {out_fp} already exists.")
                continue

            norm = fn(raw, split)
            write_jsonl(norm, out_fp)
            print(f"[OK] wrote {len(norm)} rows -> {out_fp}")

    # 2) Special-case: acp_bench with many configs
    acp_root = os.path.join(RAW_DIR, "acp_bench")
    if not os.path.isdir(acp_root):
        print("[SKIP] acp_bench — no files found.")
        return

    # iterate per-config directory
    for config_dir in sorted(glob(os.path.join(acp_root, "*"))):
        if not os.path.isdir(config_dir):
            continue

        files = sorted(glob(os.path.join(config_dir, "*.jsonl")))
        if not files:
            continue

        # choose converter by config name suffix; fallback to sample inspection
        converter = pick_acp_converter(config_dir)
        for fp in files:
            split = os.path.splitext(os.path.basename(fp))[0]
            raw = read_jsonl(fp)

            chosen = converter
            if chosen is None:
                chosen = infer_acp_converter_from_sample(raw[0] if raw else {})

            converter_name = (
                "acp_bench_bool" if chosen is convert_acp_bench_bool else
                "acp_bench_mcq" if chosen is convert_acp_bench_mcq else
                "acp_bench_gen"
            )
            print(f"[PROCESS] acp_bench | config={os.path.basename(config_dir)} | split={split} -> {converter_name}")

            norm = chosen(raw, split)
            out_fp = fp.replace(RAW_DIR, OUT_DIR)
            if os.path.exists(out_fp):
                print(f"[SKIP] {out_fp} already exists.")
                continue

            norm = chosen(raw, split)
            write_jsonl(norm, out_fp)
            print(f"[OK] wrote {len(norm)} rows -> {out_fp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=RAW_DIR, help="Path to the raw JSONL datasets")
    ap.add_argument("--out_dir", default=OUT_DIR, help="Path to write normalized datasets")
    args = ap.parse_args()

    RAW_DIR = args.raw_dir
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    normalize_all()
