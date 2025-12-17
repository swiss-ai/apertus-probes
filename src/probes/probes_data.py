# probe_data.py

import os
import pickle
from typing import Tuple, Dict, List

import numpy as np


def merge_activations(activations_list: List[Dict[int, np.ndarray]]) -> Dict[int, np.ndarray]:
    """
    Take a list of activation dicts (layer -> [N_i, d]) and concatenate along
    the sample axis for each layer.
    """
    if not activations_list:
        raise ValueError("activations_list is empty.")

    merged: Dict[int, np.ndarray] = {}
    all_layers = list(activations_list[0].keys())
    for layer in all_layers:
        arrays = [acts[layer] for acts in activations_list]
        merged[layer] = np.concatenate(arrays, axis=0)
    return merged


def load_single_dataset(
    dataset_name: str,
    model_name: str,
    save_dir: str,
    error_type: str,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, dict]:
    """
    Load activations and targets for a single dataset.

    Args:
        dataset_name: Name of the dataset
        model_name: Model name used in cache paths
        save_dir: Base directory where caches are stored
        error_type: "SM" or "CE"

    Returns:
        y_error_exact, y_correct_exact, activations_exact,
        y_error_last,  y_correct_last,  activations_last
    """
    activations_path = f"{save_dir}/{dataset_name}/{model_name}/acts.pkl"
    print("[LOAD]", activations_path)
    if not os.path.exists(activations_path):
        raise FileNotFoundError(f"Activations file not found: {activations_path}")

    with open(activations_path, "rb") as f:
        activations = pickle.load(f)

    # Extract data
    activations_last = activations["activations_cache"]
    y_error_last = (
        activations["y_error_sm"] if error_type == "SM" else activations["y_error_ce"]
    )
    y_correct_last = np.array(activations["y_correct"], dtype=float)

    activations_exact = activations["activations_cache_exact"]
    y_error_exact = (
        activations["y_error_sm_exact"] if error_type == "SM" else activations["y_error_ce_exact"]
    )
    y_correct_exact = np.array(activations["y_correct_exact"], dtype=float)

    return (
        y_error_exact,
        y_correct_exact,
        activations_exact,
        y_error_last,
        y_correct_last,
        activations_last,
    )


def load_datasets(
    config,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray, dict]:
    """
    Load & merge activations and targets for a mixture of datasets.

    Expects `config` to have at least:
      - selected_datasets: list[str]
      - save_dir: str
      - model_name: str
      - error_type: "SM" or "CE"

    Returns:
        y_error_exact, y_correct_exact, activations_exact,
        y_error_last,  y_correct_last,  activations_last
    """
    y_error_last_list = []
    y_correct_last = []
    y_error_exact_list = []
    y_correct_exact = []
    activations_exact_list = []
    activations_last_list = []

    for dataset_name in config.selected_datasets:
        activations_path = f"{config.save_dir}/{dataset_name}/{config.model_name}/acts.pkl"
        print("[LOAD]", activations_path)
        if not os.path.exists(activations_path):
            raise FileNotFoundError(f"Activations file not found: {activations_path}")

        with open(activations_path, "rb") as f:
            activations = pickle.load(f)

        # last-token cache
        activations_last_list.append(activations["activations_cache"])
        y_error_last_list.append(
            activations["y_error_sm"] if config.error_type == "SM" else activations["y_error_ce"]
        )
        y_correct_last.extend(activations["y_correct"])

        # exact-token cache
        activations_exact_list.append(activations["activations_cache_exact"])
        y_error_exact_list.append(
            activations["y_error_sm_exact"] if config.error_type == "SM" else activations["y_error_ce_exact"]
        )
        y_correct_exact.extend(activations["y_correct_exact"])

    # stack across datasets
    y_error_exact = np.concatenate(y_error_exact_list, axis=0)
    y_error_last = np.concatenate(y_error_last_list, axis=0)
    activations_exact = merge_activations(activations_exact_list)
    activations_last = merge_activations(activations_last_list)
    y_correct_exact = np.array(y_correct_exact, dtype=float)
    y_correct_last = np.array(y_correct_last, dtype=float)

    # Subsample if total examples exceed max_samples
    n_samples = len(y_error_exact)
    if n_samples > config.max_samples:
        print(f"[SUBSAMPLE] Total samples: {n_samples} > {config.max_samples}, subsampling to {config.max_samples}")
        rng = np.random.RandomState(config.seed)
        indices = rng.choice(n_samples, size=config.max_samples, replace=False)
        indices = np.sort(indices)  # Keep original order for reproducibility
        
        # Subsample targets
        y_error_exact = y_error_exact[indices]
        y_error_last = y_error_last[indices]
        y_correct_exact = y_correct_exact[indices]
        y_correct_last = y_correct_last[indices]
        
        # Subsample activations for each layer
        activations_exact = {layer: acts[indices] for layer, acts in activations_exact.items()}
        activations_last = {layer: acts[indices] for layer, acts in activations_last.items()}
        
        print(f"[SUBSAMPLE] Completed. New sample count: {len(y_error_exact)}")

    return (
        y_error_exact,
        y_correct_exact,
        activations_exact,
        y_error_last,
        y_correct_last,
        activations_last,
    )
