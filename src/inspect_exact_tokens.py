#!/usr/bin/env python3
import argparse
import os
import sys
import pickle
from typing import List, Any, Dict

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def flatten_batched_arrays(arr_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    For a list of [B, T] arrays, return a flat list of [T] arrays.
    """
    flat = []
    for arr in arr_list:
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D [B, T] arrays, got {arr.shape}")
        for i in range(arr.shape[0]):
            flat.append(arr[i])
    return flat


def flatten_completions_str(obj):
    """
    completions_str.pkl can be:
      - a flat list[str]
      - or list[list[str]] per batch.
    Return flat list[str] aligned with flattened examples.
    """
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], list):
            # list of batches
            return [s for batch in obj for s in batch]
        else:
            # already flat
            return obj
    # fallback
    return list(obj)


def extract_question(full_text: str) -> str:
    """
    Heuristic: return everything up to 'Answer:' / 'Answer :'
    so you see the question (+options) context.
    """
    if full_text is None:
        return ""
    s = str(full_text)
    # try typical markers
    markers = ["Answer:", "Answer :"]
    cut = len(s)
    for m in markers:
        idx = s.find(m)
        if idx != -1:
            cut = min(cut, idx)
    q = s[:cut]
    # compact whitespace for display
    q = q.replace("\n", "\\n").replace("\t", "\\t").strip()
    return q


def main():
    ap = argparse.ArgumentParser(
        description="Inspect exact-match tokens as a pandas DataFrame with question and local context."
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Dir with pickles, e.g. $SCRATCH/mera-runs/mmlu_high_school/Apertus-8B-Instruct-2509/",
    )
    ap.add_argument(
        "--save-key",
        required=True,
        help="Prefix used in filenames, e.g. 3000_",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="Tokenizer id/path used when running cache_run.py",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Show first N examples (after --start)",
    )
    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in flattened examples",
    )
    ap.add_argument(
        "--focus",
        type=int,
        default=None,
        help="If set, show only this global example index",
    )
    ap.add_argument(
        "--window-radius",
        type=int,
        default=4,
        help="How many tokens left/right to show in the context_window",
    )
    args = ap.parse_args()

    comp_path = os.path.join(args.run_dir, f"{args.save_key}completions.pkl")
    comp_str_path = os.path.join(args.run_dir, f"{args.save_key}completions_str.pkl")
    targ_path = os.path.join(args.run_dir, f"{args.save_key}targets.pkl")

    if not os.path.exists(comp_path):
        print(f"Missing {comp_path}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(targ_path):
        print(f"Missing {targ_path}", file=sys.stderr)
        sys.exit(2)

    comp = load_pickle(comp_path)
    targ = load_pickle(targ_path)

    comp_str = None
    if os.path.exists(comp_str_path):
        comp_str_data = load_pickle(comp_str_path)
        # try common key names
        if isinstance(comp_str_data, dict):
            if "completions_str" in comp_str_data:
                comp_str = flatten_completions_str(comp_str_data["completions_str"])
            else:
                # fallback: assume it's already a flat list
                comp_str = flatten_completions_str(list(comp_str_data.values())[0])
        else:
            comp_str = flatten_completions_str(comp_str_data)

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    to_piece = tok.convert_ids_to_tokens

    def piece_str(tid: int) -> str:
        try:
            s = to_piece([int(tid)])[0]
        except Exception:
            return str(tid)
        return s.replace("\n", "\\n").replace("\t", "\\t")

    def decoded_str(tid: int) -> str:
        try:
            s = tok.decode([int(tid)], clean_up_tokenization_spaces=False)
        except Exception:
            return str(tid)
        return s.replace("\n", "\\n").replace("\t", "\\t")

    def decoded_window(ids, center_pos: int, radius: int) -> str:
        if center_pos is None or center_pos < 0 or center_pos >= len(ids):
            return ""
        L = max(0, center_pos - radius)
        R = min(len(ids), center_pos + radius + 1)
        window_ids = [int(x) for x in ids[L:R]]
        s = tok.decode(window_ids, clean_up_tokenization_spaces=False)
        return s.replace("\n", "\\n").replace("\t", "\\t")

    # --- sequences & match info ---
    seqs_list: List[np.ndarray] = comp["completions"]  # list of [B, T]
    seqs = flatten_batched_arrays(seqs_list)           # per-example [T]

    match_indices: List[int] = comp.get("match_indices", [])
    match_tokens: List[int] = comp.get("match_tokens", [])

    N = len(seqs)
    if len(match_indices) != N or len(match_tokens) != N:
        print(
            f"[warn] Length mismatch: seqs={N}, match_indices={len(match_indices)}, "
            f"match_tokens={len(match_tokens)}; truncating to min.",
            file=sys.stderr,
        )
        N = min(N, len(match_indices), len(match_tokens))

    if comp_str is not None and len(comp_str) < N:
        # keep them aligned
        N = len(comp_str)

    # --- targets ---
    print("targets keys", targ.keys())
    y_true = targ.get("y_true", [None] * N)
    y_pred_exact = targ.get("y_pred_exact", [None] * N)
    y_correct_exact = targ.get("y_correct_exact", [None] * N)
    class_idx_to_label = targ.get("CLASS_INDEX_TO_LABEL") or targ.get("class_idx_to_label")
    softmax_values = targ.get("y_softmax_exact", None)
    def idx_to_label(idx: Any):
        if idx is None:
            return None
        try:
            i = int(idx)
        except Exception:
            return idx
        if class_idx_to_label is None:
            return i
        if isinstance(class_idx_to_label, dict):
            return class_idx_to_label.get(i, i)
        try:
            return class_idx_to_label[i]
        except Exception:
            return i

    # --- index range ---
    if args.focus is not None:
        indices = [args.focus]
    else:
        stop = min(N, args.start + args.limit)
        indices = list(range(args.start, stop))

    rows: List[Dict[str, Any]] = []

    for i in indices:
        if i < 0 or i >= N:
            continue

        seq = seqs[i]
        pos = match_indices[i]
        mtok = match_tokens[i]

        full_text = comp_str[i] if (comp_str is not None and i < len(comp_str)) else None
        question = extract_question(full_text)

        row: Dict[str, Any] = {
            "idx": i,
            "question": question,
            "match_pos": int(pos) if pos is not None else None,
            "match_tid": int(mtok) if mtok is not None else None,
            "match_piece": piece_str(mtok) if mtok is not None else None,
            "match_decoded": decoded_str(mtok) if mtok is not None else None,
            "y_true": y_true[i],
            "y_true_label": idx_to_label(y_true[i]),
            "y_pred_ex": y_pred_exact[i],
            "ok_ex": bool(y_correct_exact[i]) if y_correct_exact[i] is not None else None,
        }

        if pos is not None and 0 <= pos < len(seq):
            seq_tid = int(seq[pos])
            row.update(
                {
                    "seq_tid": seq_tid,
                    "seq_piece": piece_str(seq_tid),
                    "seq_decoded": decoded_str(seq_tid),
                    "seq_equals_match": (seq_tid == int(mtok)) if mtok is not None else None,
                    "context_window": decoded_window(seq, pos, args.window_radius),
                }
            )
        else:
            row.update(
                {
                    "seq_tid": None,
                    "seq_piece": None,
                    "seq_decoded": None,
                    "seq_equals_match": None,
                    "context_window": "",
                }
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    # nicer in terminal
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 160)

    print(f"Run dir: {args.run_dir}")
    print(f"Save key: {args.save_key}")
    print(f"Tokenizer: {args.model}")
    print("-" * 220)
    print(df)


if __name__ == "__main__":
    main()
