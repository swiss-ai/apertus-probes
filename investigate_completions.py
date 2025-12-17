#!/usr/bin/env python3
"""
Investigate completions.pkl to understand why exact matches aren't being found.
"""

import pickle
import numpy as np
from transformers import AutoTokenizer

# File paths
completions_path = "/iopsstor/scratch/cscs/astepancic/mera-runs/sujet_finance_yesno_5k/Apertus-8B-2509/completions.pkl"
# model_name = "swiss-ai/Apertus-8B-Instruct-2509"
model_name = 

print("Loading completions.pkl...")
with open(completions_path, "rb") as f:
    completions_data = pickle.load(f)

print("\n=== COMPLETIONS.PKL STRUCTURE ===")
print(f"Keys in completions.pkl: {list(completions_data.keys())}")

# Load tokenizer
print(f"\nLoading tokenizer: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Examine match data
match_indices = completions_data.get("match_indices", [])
match_flags = completions_data.get("match_flags", [])
match_tokens = completions_data.get("match_tokens", [])
completions = completions_data.get("completions", [])
completions_str = completions_data.get("completions_str", [])

print(f"\n=== MATCH STATISTICS ===")
print(f"Total samples: {len(match_indices)}")
print(f"Match indices range: {min(match_indices)} to {max(match_indices)}")
print(f"Number of -1 (no match): {sum(1 for mi in match_indices if mi == -1)}")
print(f"Match flags: {sum(match_flags)} True, {sum(1 for f in match_flags if not f)} False")

# Check data structure
print(f"\n=== DATA STRUCTURE ===")
print(f"completions type: {type(completions)}")
print(f"completions length: {len(completions)}")
if len(completions) > 0:
    print(f"completions[0] type: {type(completions[0])}")
    print(f"completions[0] shape: {completions[0].shape if hasattr(completions[0], 'shape') else 'no shape'}")
    if hasattr(completions[0], 'shape') and len(completions[0].shape) > 0:
        print(f"completions[0][0] shape: {completions[0][0].shape if len(completions[0]) > 0 else 'empty'}")
print(f"completions_str length: {len(completions_str)}")

# Check what tokens should be matched for yes/no
print(f"\n=== EXPECTED MATCH TOKENS ===")
# Generate variants for yes/no
def generate_text_variants(word: str, remove_lower: bool = False) -> list:
    transformations = [word.lower(), word.upper(), word.capitalize()]
    if remove_lower:
        del transformations[0]
    spaces = [" ", ""]
    return [
        f"{prefix}{variant}" for variant in transformations for prefix in spaces
    ]

yes_variants = generate_text_variants("yes")
no_variants = generate_text_variants("no")

# Get token IDs for each variant
yes_token_ids = {}
for variant in yes_variants:
    tokenized = tokenizer([variant], return_tensors="pt").input_ids[0]
    token_id = tokenized[-1].item()
    yes_token_ids[variant] = token_id

no_token_ids = {}
for variant in no_variants:
    tokenized = tokenizer([variant], return_tensors="pt").input_ids[0]
    token_id = tokenized[-1].item()
    no_token_ids[variant] = token_id

print(f"YES token IDs: {yes_token_ids}")
print(f"NO token IDs: {no_token_ids}")
expected_token_ids = set(list(yes_token_ids.values()) + list(no_token_ids.values()))
print(f"Expected token IDs to match: {expected_token_ids}")

# Examine some samples that failed to match
print(f"\n=== SAMPLES WITH NO MATCH (first 10) ===")
failed_indices = [i for i, mi in enumerate(match_indices) if mi == -1][:10]

for sample_idx in failed_indices:
    print(f"\nSample {sample_idx}:")
    match_idx = match_indices[sample_idx]
    match_token = match_tokens[sample_idx]
    completion_str = completions_str[sample_idx] if sample_idx < len(completions_str) else "N/A"
    
    print(f"  Match index: {match_idx}")
    print(f"  Match token ID: {match_token}")
    print(f"  Completion: {completion_str[:300]}")
    
    # Check what tokens are actually in the completion
    # completions is a list where each element is a batch (numpy array)
    # Since batch_size=1, each element has shape (1, seq_len), so we use [0] to get the sequence
    if sample_idx < len(completions):
        batch_data = completions[sample_idx]
        if isinstance(batch_data, np.ndarray):
            if len(batch_data.shape) == 2:  # Shape (1, seq_len)
                completion_token_ids = batch_data[0]
            else:  # Shape (seq_len,)
                completion_token_ids = batch_data
        else:
            completion_token_ids = None
        
        if completion_token_ids is not None:
            completion_token_ids = list(completion_token_ids) if isinstance(completion_token_ids, np.ndarray) else completion_token_ids
            
            # Find prompt length
            prompt_seq_lens = completions_data.get("prompt_sequence_lengths", [])
            prompt_seq_len = prompt_seq_lens[sample_idx] if sample_idx < len(prompt_seq_lens) else 0
            
            # Check completion tokens (after prompt)
            if prompt_seq_len < len(completion_token_ids):
                completion_only = completion_token_ids[prompt_seq_len:]
                # Show first 50 tokens
                print(f"  Prompt length: {prompt_seq_len}, Total length: {len(completion_token_ids)}")
                print(f"  First 30 completion token IDs: {completion_only[:30]}")
                decoded_completion = tokenizer.decode(completion_only[:30])
                print(f"  Decoded (first 30 tokens): '{decoded_completion}'")
                
                # Check if any expected tokens appear
                found_tokens = [tid for tid in completion_only if tid in expected_token_ids]
                print(f"  Expected tokens found in completion: {found_tokens}")
                if found_tokens:
                    positions = [i+prompt_seq_len for i, tid in enumerate(completion_only) if tid in expected_token_ids]
                    print(f"  Positions (absolute): {positions[:10]}")  # First 10 positions
                    print(f"  Positions (relative to prompt end): {[pos-prompt_seq_len for pos in positions[:10]]}")
                else:
                    print(f"  WARNING: No expected tokens found in completion!")
                    # Check what tokens ARE present
                    unique_tokens = set(completion_only[:50])
                    print(f"  Sample of unique tokens in first 50: {list(unique_tokens)[:20]}")

# Examine some samples that did match
print(f"\n=== SAMPLES WITH MATCHES (first 5) ===")
matched_indices = [i for i, mi in enumerate(match_indices) if mi != -1][:5]

for sample_idx in matched_indices:
    print(f"\nSample {sample_idx}:")
    match_idx = match_indices[sample_idx]
    match_token = match_tokens[sample_idx]
    completion_str = completions_str[sample_idx] if sample_idx < len(completions_str) else "N/A"
    
    print(f"  Match index: {match_idx}")
    print(f"  Match token ID: {match_token}")
    print(f"  Match token text: '{tokenizer.decode([match_token])}'")
    print(f"  Completion: {completion_str[:300]}")

# Check tokenization of "yes" and "no" variants
print(f"\n=== TOKENIZATION DETAILS ===")
print("Testing 'yes' variants:")
for variant in ["yes", "Yes", "YES", " yes", "Yes ", " YES"]:
    tokenized = tokenizer([variant], return_tensors="pt").input_ids[0]
    token_ids = tokenized.tolist()
    print(f"  '{variant}': {token_ids} (last token: {token_ids[-1]})")

print("\nTesting 'no' variants:")
for variant in ["no", "No", "NO", " no", "No ", " NO"]:
    tokenized = tokenizer([variant], return_tensors="pt").input_ids[0]
    token_ids = tokenized.tolist()
    print(f"  '{variant}': {token_ids} (last token: {token_ids[-1]})")

