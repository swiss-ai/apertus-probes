import argparse
import pickle
import glob
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np

from tasks.task_handler import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Postprocess cache to use for probe training."
    )
    parser.add_argument("--nr_layers", required=True, type=int, help="Number of layers.")

    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="Save directory to retrieve the cache.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Specify dataset or task name. (e.g., sms_spam), which must be predifined in task_handler.py",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Specify full model name (e.g., Qwen/Qwen2.5-3B-Instruct).",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")

    nr_layers, dataset_name, save_dir, model_name = (
        args.nr_layers,
        args.dataset_name,
        args.save_dir,
        args.model_name,
    )

    results = {}

    print(f"Processing {model_name}...")
    print(f"Processing {dataset_name}...")


    act_layer_data = {}
    act_layer_data_exact = {}
    results[dataset_name] = {}

    file_path_cache = f"{save_dir}/{dataset_name}/{model_name.split('/')[1]}"
    print("file_path_cache", file_path_cache)
    
    print(f"[DEBUG] 'nr_layers' set to {nr_layers}.")

    with open(f"{file_path_cache}/targets.pkl", "rb") as f:
        targets = pickle.load(f)  # torch.load(f, map_location="cuda:0")
    with open(f"{file_path_cache}/completions.pkl", "rb") as f:
        completions = pickle.load(f)  # torch.load(f, map_location="cuda:0")

    last_matches = completions["prompt_sequence_lengths"]
    exact_matches = completions["match_indices"]
    match_flags = completions.get("match_flags", [True] * len(exact_matches))  # Default to True if not present

    # LAST TOKEN targets: Keep ALL samples (no filtering)
    # Last token activations use prompt end, which is always valid
    y_error_sm = 1 - np.array(targets["y_softmax"])
    y_correct = targets["y_correct"]
    y_error_ce = targets["y_error"]
    
    # EXACT TOKEN targets: Filter to only include samples where exact match was found
    # This ensures targets align with activations that have valid exact matches
    y_error_sm_exact = 1 - np.array(targets["y_softmax_exact"])
    y_correct_exact = targets["y_correct_exact"]
    y_error_ce_exact = targets["y_error_exact"]
    
    # Filter exact-match targets to only include samples where exact match was found
    valid_exact_mask = np.array(match_flags, dtype=bool)
    num_valid_exact = np.sum(valid_exact_mask)
    num_invalid_exact = len(valid_exact_mask) - num_valid_exact
    
    if num_invalid_exact > 0:
        print(f"\n=== FILTERING SAMPLES ===")
        print(f"Total samples: {len(valid_exact_mask)}")
        print(f"Samples with valid exact matches: {num_valid_exact}")
        print(f"Samples without exact matches (will be filtered): {num_invalid_exact}")
        print(f"Filtering targets to match valid samples...")
        
        # Filter exact-match targets
        y_error_sm_exact = y_error_sm_exact[valid_exact_mask]
        y_correct_exact = np.array(y_correct_exact)[valid_exact_mask].tolist()
        y_error_ce_exact = np.array(y_error_ce_exact)[valid_exact_mask].tolist()
        print(f"Filtered exact targets to {len(y_correct_exact)} samples")
    else:
        print(f"\nAll {len(valid_exact_mask)} samples have valid exact matches - no filtering needed.")

    for layer in tqdm(
        range(nr_layers), desc=f"Processing Layers for {dataset_name}"
    ):
        # Find all shard files for this layer
        pattern = f"{file_path_cache}/activations/activations_{layer}_part*.pkl"
        shard_files = sorted(glob.glob(pattern))
        assert len(shard_files) > 0, f"No activation files found for layer {layer} with pattern: {pattern}"
        print(f"Found {len(shard_files)} shards for layer {layer}")
        # Process each shard file-by-file to save memory
        act_layer_last_list = []
        act_layer_exact_list = []
        sample_offset = 0
        
        for shard_file in shard_files:
            # Load one shard at a time
            # shard data shape: (num_samples_in_shard, seq_len, hidden_size)
            print(f"Loading shard file: {shard_file}")
            with open(shard_file, "rb") as f:
                shard_data = pickle.load(f)
            
            # Process this shard: extract activations for samples in this shard
            shard_samples = len(shard_data)
            shard_last_matches = last_matches[sample_offset:sample_offset + shard_samples]
            shard_exact_matches = exact_matches[sample_offset:sample_offset + shard_samples]
            
            # Diagnostic: Check for invalid indices
            invalid_exact_count = sum(1 for em in shard_exact_matches if em < 0)
            if invalid_exact_count > 0:
                print(f"  Layer {layer}, shard {shard_file}: {invalid_exact_count}/{shard_samples} samples have invalid exact_match (likely -1, meaning no match found)")
            
            # Extract activations for this shard
            # For exact matches: only extract activations for samples where exact_match >= 0
            # This ensures activations align with filtered targets
            shard_match_flags = match_flags[sample_offset:sample_offset + shard_samples]
            
            try:
                # LAST TOKEN: Extract activations for ALL samples (no filtering needed)
                # Last token uses prompt_sequence_lengths which is always valid
                shard_last_activations = np.vstack(
                    [
                        sample[last_match]
                        for sample, last_match in zip(shard_data, shard_last_matches)
                    ]
                )
                # EXACT TOKEN: Only extract activations for samples with valid exact matches
                # This filters out samples where the model didn't generate "yes"/"no" tokens
                shard_exact_activations = np.vstack(
                    [
                        sample[exact_match]
                        for sample, exact_match, is_valid in zip(shard_data, shard_exact_matches, shard_match_flags)
                        if is_valid and exact_match >= 0
                    ]
                )
            except (IndexError, ValueError) as e:
                print(f"ERROR extracting activations from {shard_file}: {e}")
                print(f"  shard_samples={shard_samples}, sample_offset={sample_offset}")
                print(f"  Sample lengths: {[len(s) for s in shard_data[:5]]}...")
                print(f"  Last matches range: {min(shard_last_matches)} to {max(shard_last_matches)}")
                print(f"  Exact matches range: {min(em for em in shard_exact_matches if em >= 0)} to {max(em for em in shard_exact_matches if em >= 0)}")
                raise
            
            # Accumulate results
            act_layer_last_list.append(shard_last_activations)
            if shard_exact_activations.shape[0] > 0:  # Only append if we have valid samples
                act_layer_exact_list.append(shard_exact_activations)
            
            # Update offset for next shard
            sample_offset += shard_samples
            
            # Free memory immediately
            del shard_data
            del shard_last_activations
            del shard_exact_activations
        
        # Concatenate all processed shards
        act_layer_data[layer] = np.vstack(act_layer_last_list) if len(act_layer_last_list) > 1 else act_layer_last_list[0]
        if len(act_layer_exact_list) > 0:
            act_layer_data_exact[layer] = np.vstack(act_layer_exact_list) if len(act_layer_exact_list) > 1 else act_layer_exact_list[0]
        else:
            # This shouldn't happen if filtering is correct, but handle it gracefully
            print(f"  WARNING: Layer {layer} has no valid exact activations!")
            act_layer_data_exact[layer] = np.array([]).reshape(0, act_layer_data[layer].shape[1])
        
        n_samples_last = act_layer_data[layer].shape[0]
        n_samples_exact = act_layer_data_exact[layer].shape[0]
        print(f"  Layer {layer}: {n_samples_last} samples (last), {n_samples_exact} samples (exact)")
        
        del act_layer_last_list
        del act_layer_exact_list

    # Diagnostic: Report sample count variations
    print("\n=== SAMPLE COUNT SUMMARY ===")
    print(f"Targets (y_correct_exact): {len(y_correct_exact)} samples")
    print(f"Targets (y_correct): {len(y_correct)} samples")
    
    sample_counts_last = {layer: data.shape[0] for layer, data in act_layer_data.items()}
    sample_counts_exact = {layer: data.shape[0] for layer, data in act_layer_data_exact.items()}
    
    print(f"\nLast-token activations sample counts:")
    min_last, max_last = min(sample_counts_last.values()), max(sample_counts_last.values())
    print(f"  Range: {min_last} to {max_last} samples")
    if min_last != max_last:
        print(f"  WARNING: Inconsistent sample counts across layers!")
        inconsistent_layers = [layer for layer, count in sample_counts_last.items() if count != max_last]
        print(f"  Layers with fewer samples: {sorted(inconsistent_layers)}")
    
    print(f"\nExact-token activations sample counts:")
    min_exact, max_exact = min(sample_counts_exact.values()), max(sample_counts_exact.values())
    print(f"  Range: {min_exact} to {max_exact} samples")
    if min_exact != max_exact:
        print(f"  WARNING: Inconsistent sample counts across layers!")
        inconsistent_layers = [layer for layer, count in sample_counts_exact.items() if count != max_exact]
        print(f"  Layers with fewer samples: {sorted(inconsistent_layers)}")
    
    if min_exact < len(y_correct_exact):
        print(f"\nWARNING: Activations have fewer samples ({min_exact}) than targets ({len(y_correct_exact)})!")
        print(f"  This suggests some samples were skipped during activation extraction.")
    print("=" * 30 + "\n")

    # LAST MATCHES
    results[dataset_name]["activations_cache"] = act_layer_data
    results[dataset_name]["y_correct"] = y_correct
    results[dataset_name]["y_error_sm"] = y_error_sm
    results[dataset_name]["y_error_ce"] = y_error_ce

    # EXACT MATCHES
    results[dataset_name]["activations_cache_exact"] = act_layer_data_exact
    results[dataset_name]["y_correct_exact"] = y_correct_exact
    results[dataset_name]["y_error_sm_exact"] = y_error_sm_exact
    results[dataset_name]["y_error_ce_exact"] = y_error_ce_exact

    file_path_save = f"{save_dir}/{dataset_name}/{model_name.split('/')[1]}/acts.pkl"

    with open(file_path_save, "wb") as f:
        pickle.dump(results[dataset_name], f)
