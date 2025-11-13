#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any, Iterable, Optional
from datasets import load_dataset, DatasetDict

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_jsonl(rows: Iterable[Dict[str, Any]], out_path: str, overwrite: bool):
    if (not overwrite) and os.path.exists(out_path):
        print(f"[SKIP] Exists: {out_path}")
        return
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {sum(1 for _ in rows) if isinstance(rows, list) else 'N'} rows -> {out_path}")


def prompt_mmlu(question: str, labels: List[str], choices: List[str]) -> str:
    assert len(labels) == len(choices), "Labels and choices length mismatch"
    formatted_options = "\n".join(
        [
            f"{labels[i]}. {choice}"
            for i, choice in enumerate(choices)
        ]
    )
    answer_options = ", ".join(labels)
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer.\n"
        f"Only return one letter {answer_options}.\n"
        f"Answer:\n "
    )
    return prompt

def prompt_arc(question: str, labels: List[str], choices: List[str]) -> str:
    assert len(labels) == len(choices), "Labels and choices length mismatch"
    formatted_options = "\n".join(
        [
            f"{labels[i]}. {choice}"
            for i, choice in enumerate(choices)
        ]
    )
    answer_options = ", ".join(labels)
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n"
        f"Please select the correct answer.\n"
        f"Only return one letter {answer_options}.\n"
        f"Answer:\n "
    )
    return prompt

def prompt_sms_spam(sms: str) -> str:
    # Copying the prompt from task_handler.py
    prompt = (
        f'\n'
        f'This SMS (text message): "{sms.strip()}" is classified as either spam or ham.\n'
        f'Please evaluate the content of the SMS and select the correct classification.\n'
        f'Only return one word: "ham" or "spam".\n'
        f'Answer:\n '
    )
    return prompt

def _letter_from_index(idx: int) -> str:
    if idx is None or idx < 0:
        return ""
    return LETTERS[idx]

def convert_mmlu_split(split_ds, category_prefix: str) -> List[Dict[str, Any]]:
    """
    category_prefix: "high_school_" or "professional_"
    MMLU (cais/mmlu, config='all') provides fields:
      - 'subject' (e.g., 'high_school_biology')
      - 'question'
      - 'choices' (list[str])
      - 'answer' (int index of correct choice)
    """
    rows: List[Dict[str, Any]] = []
    labels = ["A", "B", "C", "D"]
    for ex in split_ds:
        subject = ex.get("subject", "") or ""
        if not subject.startswith(category_prefix):
            continue
        question = ex.get("question", "")
        choices = ex.get("choices", []) or []
        ans_idx = ex.get("answer", None)
        answer_letter = _letter_from_index(ans_idx)
        row = {
            "question": question,
            "question_with_prompt": prompt_mmlu(question, labels, choices),
            "answer": answer_letter,          # store the letter (A/B/C/D)
            "choices": choices,               # list[str]
        }
        rows.append(row)
    return rows

def convert_arc_split(split_ds) -> List[Dict[str, Any]]:
    """
    ARC fields:
      - 'question' (str)
      - 'choices': {'label': [...], 'text': [...]}
      - 'answerKey': 'A' | 'B' | ...
    """
    rows: List[Dict[str, Any]] = []
    labels = ["A", "B", "C", "D"]
    for ex in split_ds:
        question = ex.get("question", "")
        c = ex.get("choices", {}) or {}
        choices = c.get("text", []) or []
        # Some variants also carry labels; we normalize to A.. as presentation only
        answer_letter = ex.get("answerKey", "") or ""
        row = {
            "question": question,
            "question_with_prompt": prompt_arc(question, labels, choices),
            "answer": answer_letter,  # the letter as provided by ARC
            "choices": choices,
        }
        rows.append(row)
    return rows

def convert_sms_spam_split(split_ds) -> List[Dict[str, Any]]:
    """
    Aim for the typical HF sms_spam datasets with fields like:
      - 'sms' (text), 'label' or 'label_num' / 'target' mapping to ham/spam.
    We try common key variants robustly.
    """
    rows: List[Dict[str, Any]] = []
    CHOICES = ["ham", "spam"]

    for ex in split_ds:
        sms = ex.get("sms", None)
        label = ex.get("label", None)
        row = {
            "question": sms,
            "question_with_prompt": prompt_sms_spam(sms),
            "answer": CHOICES[label] if label is not None else "",
            "choices": CHOICES,
        }
        rows.append(row)
    return rows

# -----------------------------
# Dataset runners
# -----------------------------
def run_mmlu_high_school(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("cais/mmlu", "all")
    for split in ds.keys():
        rows = convert_mmlu_split(ds[split], category_prefix="high_school_")
        outp = os.path.join(out_dir, "mmlu_high_school", "all", f"{split}.jsonl")
        write_jsonl(rows, outp, overwrite)

def run_mmlu_professional(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("cais/mmlu", "all")
    for split in ds.keys():
        rows = convert_mmlu_split(ds[split], category_prefix="professional_")
        outp = os.path.join(out_dir, "mmlu_professional", "all", f"{split}.jsonl")
        write_jsonl(rows, outp, overwrite)

def run_arc_easy(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("allenai/ai2_arc", "ARC-Easy")
    for split in ds.keys():
        rows = convert_arc_split(ds[split])
        outp = os.path.join(out_dir, "arc_easy", "ARC-Easy", f"{split}.jsonl")
        write_jsonl(rows, outp, overwrite)

def run_arc_challenge(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    for split in ds.keys():
        rows = convert_arc_split(ds[split])
        outp = os.path.join(out_dir, "arc_challenge", "ARC-Challenge", f"{split}.jsonl")
        write_jsonl(rows, outp, overwrite)

def run_sms_spam(out_dir: str, overwrite: bool):
    """
    Try common IDs. If the first fails on your cluster mirror, try the next.
    You can swap these in your environment if you already know the exact one you use.
    """
    ds: DatasetDict = load_dataset("ucirvine/sms-spam")
    for split in ds.keys():
        rows = convert_sms_spam_split(ds[split])
        outp = os.path.join(out_dir, "sms_spam", "default", f"{split}.jsonl")
        write_jsonl(rows, outp, overwrite)


# -----------------------------
# Main
# -----------------------------
ALL_DATASETS = {
    "mmlu_high_school": run_mmlu_high_school,
    "mmlu_professional": run_mmlu_professional,
    "sms_spam": run_sms_spam,
    "arc_easy": run_arc_easy,
    "arc_challenge": run_arc_challenge,
}

def parse_args():
    ap = argparse.ArgumentParser(description="Download and convert datasets to jsonl with {question, question_with_prompt, answer, choices}.")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to store the jsonl outputs.")
    ap.add_argument("--datasets", type=str, default=",".join(ALL_DATASETS.keys()),
                    help="Comma-separated subset of: " + ",".join(ALL_DATASETS.keys()))
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing jsonl files if present.")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in selected:
        if d not in ALL_DATASETS:
            print(f"[SKIP] Unknown dataset key '{d}'. Allowed: {list(ALL_DATASETS.keys())}")
            continue
        print(f"=== {d} ===")
        ALL_DATASETS[d](out_dir, args.overwrite)
    print(f"[DONE] All requested datasets exported to {out_dir}")

if __name__ == "__main__":
    main()
