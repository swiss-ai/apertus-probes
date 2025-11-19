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

    y_error_sm = 1 - np.array(targets["y_softmax"])
    y_correct = targets["y_correct"]
    y_error_ce = targets["y_error"]
    y_error_sm_exact = 1 - np.array(targets["y_softmax_exact"])
    y_correct_exact = targets["y_correct_exact"]
    y_error_ce_exact = targets["y_error_exact"]

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
            # Extract activations for this shard
            shard_last_activations = np.vstack(
                [
                    sample[last_match]
                    for sample, last_match in zip(shard_data, shard_last_matches)
                ]
            )
            shard_exact_activations = np.vstack(
                [
                    sample[exact_match]
                    for sample, exact_match in zip(shard_data, shard_exact_matches)
                ]
            )
            
            # Accumulate results
            act_layer_last_list.append(shard_last_activations)
            act_layer_exact_list.append(shard_exact_activations)
            
            # Update offset for next shard
            sample_offset += shard_samples
            
            # Free memory immediately
            del shard_data
            del shard_last_activations
            del shard_exact_activations
        
        # Concatenate all processed shards
        act_layer_data[layer] = np.vstack(act_layer_last_list) if len(act_layer_last_list) > 1 else act_layer_last_list[0]
        act_layer_data_exact[layer] = np.vstack(act_layer_exact_list) if len(act_layer_exact_list) > 1 else act_layer_exact_list[0]
        
        del act_layer_last_list
        del act_layer_exact_list

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
