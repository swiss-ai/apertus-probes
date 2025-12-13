#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Dict, Any, Iterable
from datasets import load_dataset, DatasetDict
import re

LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
    "Y", "Z"
]

# -----------------------------
# Utilities
# -----------------------------

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


def _letter_from_index(idx: int) -> str:
    if idx is None or idx < 0:
        return ""
    return LETTERS[idx]

# -----------------------------
# Prompt builders
# -----------------------------

def prompt_mmlu(question: str, labels: List[str], choices: List[str]) -> str:
    assert len(labels) == len(choices), "Labels and choices length mismatch"
    formatted_options = "\n".join(
        [f"{labels[i]}. {choice}" for i, choice in enumerate(choices)]
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
        [f"{labels[i]}. {choice}" for i, choice in enumerate(choices)]
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






def convert_sujet_finance_yesno(split_ds) -> List[Dict[str, Any]]:
    """
    From sujet-ai/Sujet-Finance-Instruct-177k keep ONLY:
      - dataset == 'AdaptLLM/finance-tasks_Headline'
      - task_type == 'yes_no_question'
      - answer that starts with yes/no (case-insensitive)

    Map to schema:
      - question: user_prompt if available, else inputs (just for logging/analysis)
      - question_with_prompt: the exact HF `inputs` string + " "
      - answer: 'yes' / 'no'
      - choices: ['yes', 'no']
    """
    rows: List[Dict[str, Any]] = []

    total_candidates = 0
    kept = 0

    for ex in split_ds:
        # Restrict to AdaptLLM Headline yes/no questions
        if ex.get("dataset") != "AdaptLLM/finance-tasks_Headline":
            continue
        if ex.get("task_type") != "yes_no_question":
            continue

        total_candidates += 1

        inputs = (ex.get("inputs") or "").strip()
        if not inputs:
            continue

        raw_ans = str(ex.get("answer", "") or "").strip()
        lower = raw_ans.lower()

        if lower.startswith("yes"):
            ans = "yes"
        elif lower.startswith("no"):
            ans = "no"
        else:
            # not a yes/no style answer → skip
            continue

        # For `question`, we just keep something human-readable:
        question_text = (ex.get("user_prompt") or inputs).strip()
        question_text = " ".join(question_text.split())

        rows.append(
            {
                "question": question_text,
                # IMPORTANT PART: use inputs as prompt, just like in training
                "question_with_prompt": inputs + " ",
                "answer": ans,
                "choices": ["yes", "no"],
            }
        )
        kept += 1

    print(f"[DEBUG] finance Headline yes/no candidates: {total_candidates}")
    print(f"[DEBUG] Kept rows (parsed answers + non-empty inputs): {kept}")

    return rows

# -----------------------------
# Other converters
# -----------------------------

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
    anomaly = 0
    for ex in split_ds:
        question = ex.get("question", "")
        c = ex.get("choices", {}) or {}
        choices = c.get("text", []) or []
        if len(choices) < 4:
            anomaly += 1
            continue
        if len(choices) > 4:
            choices = choices[:4]
            anomaly += 1
        answer_letter = ex.get("answerKey", "") or ""
        if answer_letter not in labels:
            anomaly += 1
            continue
        row = {
            "question": question,
            "question_with_prompt": prompt_arc(question, labels, choices),
            "answer": answer_letter,
            "choices": choices,
        }
        rows.append(row)
    print(f"anomaly count: {anomaly} over {len(split_ds)} examples")
    return rows


def convert_sms_spam_split(split_ds) -> List[Dict[str, Any]]:
    """
    Typical HF sms_spam datasets with fields like:
      - 'sms' (text), 'label' or 'label_num' / 'target' mapping to ham/spam.
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

def run_sujet_finance_yesno_5k(out_dir: str, overwrite: bool):
    """
    Take sujet-ai/Sujet-Finance-Instruct-177k (train split),
    restrict to AdaptLLM/finance-tasks_Headline yes/no questions,
    then sample up to 5000 rows.
    """
    ds: DatasetDict = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")
    train_ds = ds["train"]

    # Convert & filter to yes/no headline questions
    all_rows = convert_sujet_finance_yesno(train_ds)
    if not all_rows:
        print("[WARN] No matching yes/no rows found in Sujet-Finance dataset.")
        return

    import random
    random.seed(52)
    random.shuffle(all_rows)
    rows = all_rows[:5000]

    outp = os.path.join(out_dir, "sujet_finance_yesno_5k", "dataset.jsonl")
    write_jsonl(rows, outp, overwrite)


def run_mmlu_high_school(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("cais/mmlu", "all")
    rows = []
    for split in ds.keys():
        rows.extend(convert_mmlu_split(ds[split], category_prefix="high_school_"))
    outp = os.path.join(out_dir, "mmlu_high_school", "dataset.jsonl")
    write_jsonl(rows, outp, overwrite)


def run_mmlu_professional(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("cais/mmlu", "all")
    rows = []
    for split in ds.keys():
        rows.extend(convert_mmlu_split(ds[split], category_prefix="professional_"))
    outp = os.path.join(out_dir, "mmlu_professional", "dataset.jsonl")
    write_jsonl(rows, outp, overwrite)


def run_arc_easy(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("allenai/ai2_arc", "ARC-Easy")
    rows = []
    for split in ds.keys():
        rows.extend(convert_arc_split(ds[split]))
    outp = os.path.join(out_dir, "ARC-Easy", "dataset.jsonl")
    write_jsonl(rows, outp, overwrite)


def run_arc_challenge(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    rows = []
    for split in ds.keys():
        rows.extend(convert_arc_split(ds[split]))
    outp = os.path.join(out_dir, "ARC-Challenge", "dataset.jsonl")
    write_jsonl(rows, outp, overwrite)


def run_sms_spam(out_dir: str, overwrite: bool):
    ds: DatasetDict = load_dataset("ucirvine/sms_spam", None)
    rows = []
    for split in ds.keys():
        rows.extend(convert_sms_spam_split(ds[split]))
    outp = os.path.join(out_dir, "sms_spam", "dataset.jsonl")
    write_jsonl(rows, outp, overwrite)

# -----------------------------
# Main
# -----------------------------

ALL_DATASETS = {
    "finance-yesno": run_sujet_finance_yesno_5k,
    "mmlu_high_school": run_mmlu_high_school,
    "mmlu_professional": run_mmlu_professional,
    "sms_spam": run_sms_spam,
    "ARC-Easy": run_arc_easy,
    "ARC-Challenge": run_arc_challenge,
}


def parse_args():
    # Default to the standard dataset directory used by the project
    default_out_dir = os.path.join(
        os.environ.get("SCRATCH", "/tmp"),
        "mera-runs",
        "processed_datasets"
    )

    ap = argparse.ArgumentParser(
        description="Download and convert datasets to jsonl with "
                    "{question, question_with_prompt, answer, choices}."
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=default_out_dir,
        help=f"Directory to store the jsonl outputs. Default: {default_out_dir}",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default=",".join(ALL_DATASETS.keys()),
        help="Comma-separated subset of: " + ",".join(ALL_DATASETS.keys()),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing jsonl files if present.",
    )
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
