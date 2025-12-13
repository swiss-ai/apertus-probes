# probe_core.py

import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU/RAM monitoring disabled.")

from probes.probes_data import load_datasets

# load_datasets imported lazily in run_probe_experiment to avoid circular import

# -------------------------
# System monitoring
# -------------------------

def print_system_resources():
    """Print CPU and RAM usage information."""
    if not PSUTIL_AVAILABLE:
        return
    
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        active_cpus = sum(1 for p in cpu_per_core if p > 5.0)  # CPUs with >5% usage
        
        # Memory information
        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_available_gb = mem.available / (1024**3)
        mem_used_gb = mem.used / (1024**3)
        mem_percent = mem.percent
        
        print("\n" + "="*60)
        print("SYSTEM RESOURCES")
        print("="*60)
        print(f"CPU: {cpu_percent:.1f}% usage | {active_cpus}/{cpu_count} active cores | "
              f"{cpu_count_physical} physical cores")
        print(f"RAM: {mem_used_gb:.2f} GB used / {mem_total_gb:.2f} GB total "
              f"({mem_percent:.1f}%) | {mem_available_gb:.2f} GB available")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Warning: Could not get system resources: {e}")

# -------------------------
# Config
# -------------------------

@dataclass
class ProbeConfig:
    selected_datasets: list[str]
    model_name: str                    # "Apertus-8B-Instruct-2509"
    save_dir: str                      # base dir for caches and results
    save_name: str = ""
    seed: int = 52
    error_type: str = "SM"             # "SM" or "CE"
    transform_targets: bool = True
    normalize_features: bool = True
    nr_attempts: int = 5
    max_trials: int = 5
    eps: float = 1e-10                 # for non-zero coeffs
    max_workers: int = 25
    alphas: tuple = (0.5, 0.25, 0.1, 0.05)
    token_pos: str = "both"           # "exact, last or both"
    use_logit_regression: bool = False  # If True, use LogitRegression; if False, use standard Lasso (default: Lasso)

# -------------------------
# Metrics & models
# -------------------------

METRICS = {
    "classification": {
        "AUCROC": roc_auc_score,
        "Accuracy": accuracy_score,
        "Accuracy (Balanced)": balanced_accuracy_score,
    },
    "regression": {
        "MSE": mean_squared_error,
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    },
}

class LogitRegression(Lasso):
    def fit(self, x, p):
        p = np.asarray(p)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        p = np.log(p / (1 - p))
        return super().fit(x, p)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)

def initialise_regression_models(seed: int, alphas, use_logit_regression: bool = False) -> dict:
    """
    Initialize regression models (Lasso or LogitRegression).
    
    Args:
        seed: Random seed
        alphas: Tuple of alpha values for L1 regularization
        use_logit_regression: If True, use LogitRegression (applies logit transform);
                             if False, use standard Lasso
    """
    if use_logit_regression:
        models = {
            f"L-{alpha}": LogitRegression(
                alpha=alpha, fit_intercept=True, max_iter=2000, random_state=seed
            ) for alpha in alphas
        }
    else:
        models = {
            f"L-{alpha}": Lasso(
                alpha=alpha, fit_intercept=True, max_iter=2000, random_state=seed
            ) for alpha in alphas
        }
    # models["L-0"] = LinearRegression(fit_intercept=False, n_jobs=5)
    return models


def initalise_classification_models(seed: int) -> dict:
    models = {
        "LogReg-l1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            fit_intercept=True,
            random_state=seed,
        )
    }
    return models


def make_models(config: ProbeConfig) -> dict:
    return {
        "classification": initalise_classification_models(config.seed),
        "regression": initialise_regression_models(
            config.seed, 
            config.alphas, 
            use_logit_regression=config.use_logit_regression
        ),
    }

# -------------------------
# Core training for one layer/model
# -------------------------

def train_model_on_layer(
    task_type: str,
    probe_name: str,
    base_model,
    layer_idx: int,
    layer_data: np.ndarray,
    y_error_exact: np.ndarray,
    y_correct_exact: np.ndarray,
    token_pos: str,
    config: ProbeConfig,
    dataset_name_for_logging: str,
    X_train_separate: np.ndarray = None,
    X_test_separate: np.ndarray = None,
    y_train_separate: np.ndarray = None,
    y_test_separate: np.ndarray = None,
) -> list[dict]:
    """
    Train a single probe model on one layer; returns list of result dicts
    (one per attempt) to be appended into the main results DF.
    
    If X_train_separate, X_test_separate, y_train_separate, y_test_separate are provided,
    uses those directly (cross-dataset mode). Otherwise, does train/test split on layer_data.
    """
    results_for_layer = []
    
    # Cross-dataset mode: use provided train/test data
    if X_train_separate is not None and X_test_separate is not None:
        if y_train_separate is None or y_test_separate is None:
            raise ValueError("If X_train_separate/X_test_separate provided, y_train_separate/y_test_separate must also be provided")
        
        # Use all data for training (no multiple attempts in cross-dataset mode)
        attempts = [0]  # Single attempt
        X_train = X_train_separate
        X_test = X_test_separate
        y_train = np.array(y_train_separate, dtype=float)
        y_test = np.array(y_test_separate, dtype=float)
        
        # Apply target transformation if needed (for regression)
        if task_type == "regression" and config.transform_targets:
            y_train = np.clip(y_train, 1e-8, 1 - 1e-8)
            y_train = np.log(y_train / (1 - y_train))
            y_test = np.clip(y_test, 1e-8, 1 - 1e-8)
            y_test = np.log(y_test / (1 - y_test))
    else:
        # Standard mode: train/test split on same data
        X = layer_data
        if task_type == "classification":
            y_true = np.array(y_correct_exact, dtype=float)
        elif task_type == "regression":
            y_true = np.array(y_error_exact, dtype=float)
            if config.transform_targets:
                y_true = np.clip(y_true, 1e-8, 1 - 1e-8)
                y_true = np.log(y_true / (1 - y_true))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Validate that X and y have matching sample counts
        n_samples_X = X.shape[0] if len(X.shape) > 1 else len(X)
        n_samples_y = len(y_true)
        if n_samples_X != n_samples_y:
            # Align by truncating targets to match activations (assumes samples are in order)
            if n_samples_X < n_samples_y:
                print(f"WARNING: Layer {layer_idx} has fewer activations ({n_samples_X}) than targets ({n_samples_y}). Truncating targets to match.")
                y_true = y_true[:n_samples_X]
            else:
                print(f"WARNING: Layer {layer_idx} has more activations ({n_samples_X}) than targets ({n_samples_y}). This shouldn't happen. Skipping this layer.")
                return []  # Return empty results list, skipping this layer
        
        attempts = range(config.nr_attempts)

    for m in attempts:
        if X_train_separate is None:
            # Standard mode: do train/test split
            print(f"[{task_type} | {probe_name}] Layer {layer_idx}, attempt {m}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_true, test_size=0.3, random_state=config.seed + m
            )
        else:
            # Cross-dataset mode: already have train/test split
            print(f"[{task_type} | {probe_name}] Layer {layer_idx} (cross-dataset)")

        if config.normalize_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        dummy_model = (
            DummyClassifier(strategy="most_frequent")
            if task_type == "classification"
            else DummyRegressor(strategy="mean")
        )
        dummy_model.fit(X_train, y_train)
        dummy_y_pred = dummy_model.predict(X_test)
        dummy_metrics = {
            f"Dummy-{metric_name}": metric(y_test, dummy_y_pred)
            for metric_name, metric in METRICS[task_type].items()
        }

        no_coeffs = True
        trials = 0
        coef_norm = None
        y_pred = None
        non_zero_coeffs = np.array([], dtype=int)

        while no_coeffs and trials < config.max_trials:
            trials += 1

            model = clone(base_model)
            model.fit(X_train, y_train)


            assert hasattr(model, "coef_")
            coef = model.coef_
            coef_norm = np.linalg.norm(coef)
            y_pred = model.predict(X_test)

            non_zero_coeffs = np.where(np.abs(coef) > config.eps)[0]
            no_coeffs = len(non_zero_coeffs) == 0

        if no_coeffs:
            print(
                f"Warning: no non-zero coeffs after {config.max_trials} trials "
                f"(layer {layer_idx}, attempt {m})."
            )

        metrics = {
            **{name: fn(y_test, y_pred) for name, fn in METRICS[task_type].items()},
            **dummy_metrics,
        }
        print("Token-Pos:", token_pos, "Probe-Name:", probe_name, "Layer:", layer_idx, "Results:", metrics)

        if not no_coeffs:
            row = {
                "Dataset": dataset_name_for_logging,
                "LLM_Model": config.model_name,
                "Task": task_type,
                "Model": probe_name,
                "Layer": layer_idx,
                "Coefficients": coef.flatten(),
                "Coef-norm": coef_norm,
                "# of non-zero coefficients": len(non_zero_coeffs),
                "Attempt": m,
                "Token-Pos": token_pos,
                "y_pred": y_pred,
                "y_test": y_test,
                "Error-Type": config.error_type,
                **metrics,
            }
            results_for_layer.append(row)

    return results_for_layer

# -------------------------
# High-level runner
# -------------------------

def run_probe_experiment(config: ProbeConfig) -> pd.DataFrame:
    """
    Run the full probe training for one (mixture of) dataset(s) and one model.
    Returns the DataFrame of results, and also saves them to disk.
    """
    dataset_mixture_name = "+".join(config.selected_datasets)
    models = make_models(config)

    save_root = f"{config.save_dir}/mix/{dataset_mixture_name}/{config.model_name}"
    os.makedirs(save_root, exist_ok=True)
    save_prefix = f"{save_root}/df_probes_{config.save_name}"
    print(f"[SAVE] Results prefix: {save_prefix}")

    # Print system resources at start
    print_system_resources()

    (
        y_error_exact,
        y_correct_exact,
        activations_exact,
        y_error_last,
        y_correct_last,
        activations_last,
    ) = load_datasets(config)
    
    # Print system resources after loading data
    print_system_resources()

  
    if config.token_pos == "both":
        token_positions = ["exact", "last"]
    else:
        token_positions = [config.token_pos]

    probe_results: list[dict] = []

    for token_pos in token_positions:
        # Choose which activations to use
        if token_pos == "exact":
            activations = activations_exact
        elif token_pos == "last":
            activations = activations_last
        else:
            raise ValueError(f"Unknown token_pos: {token_pos}")
        y_error = y_error_exact if token_pos == "exact" else y_error_last
        y_correct = y_correct_exact if token_pos == "exact" else y_correct_last


        print(f"y_error_exact shape: {y_error.shape}")
        print(f"y_correct_exact length: {len(y_correct)}")
        print(f"activations layers: {list(activations.keys())[:5]} ...")
        
        # Check sample counts for all layers
        layer_sample_counts = {layer_idx: layer_data.shape[0] if len(layer_data.shape) > 1 else len(layer_data) 
                              for layer_idx, layer_data in activations.items()}
        n_samples_y_error = len(y_error)
        n_samples_y_correct = len(y_correct)
        
        # Find the minimum and maximum sample counts across all layers
        min_samples = min(layer_sample_counts.values())
        max_samples = max(layer_sample_counts.values())
        
        # Report on sample count variation
        if min_samples != max_samples or min_samples != n_samples_y_error:
            print(f"INFO: Sample count variation detected:")
            print(f"  Targets have: {n_samples_y_error} samples")
            print(f"  Activations range: {min_samples} to {max_samples} samples across layers")
            print(f"  Will align targets per-layer to match each layer's activation count")
            print(f"  (Assumes samples are in order - first N targets correspond to first N activations)")

        futures = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            for task_type, model_dict in models.items():
                print("Task type:", task_type)
                for probe_name, base_model in model_dict.items():
                    print("Probe model:", probe_name)
                    for layer_idx, layer_data in tqdm(
                        activations.items(), desc=f"Layers ({task_type}, {probe_name})"
                    ):
                        futures.append(
                            executor.submit(
                                train_model_on_layer,
                                task_type,
                                probe_name,
                                base_model,
                                layer_idx,
                                layer_data,
                                y_error,
                                y_correct,
                                token_pos,
                                config,
                                dataset_mixture_name,
                            )
                        )

            # Print system resources before training starts
            print_system_resources()
            
            completed_count = 0
            total_futures = len(futures)
            checkpoints = {int(total_futures * 0.25), int(total_futures * 0.50), int(total_futures * 0.75)}
            
            for fut in tqdm(as_completed(futures), total=total_futures, desc="Training probes"):
                rows = fut.result()
                probe_results.extend(rows)
                completed_count += 1
                
                # Print system resources at 25%, 50%, 75% completion
                if completed_count in checkpoints:
                    print_system_resources()

    df_results = pd.DataFrame(probe_results)
    df_results.to_pickle(save_prefix + ".pkl")

    metric_cols = sorted({name for task in METRICS.values() for name in task.keys()})
    dummy_metric_cols = [f"Dummy-{name}" for name in metric_cols]
    essential_cols = [
        "Dataset",
        "LLM_Model",
        "Task",
        "Model",
        "Error-Type",
        "Layer",
        "# of non-zero coefficients",
        "Attempt",
        "Token-Pos",
    ]
    essential_results = df_results[essential_cols + metric_cols + dummy_metric_cols]
    essential_results.to_csv(save_prefix + ".csv", index=False)

    # Print system resources at end
    print_system_resources()
    
    print("[DONE] Saved:", save_prefix + ".pkl", "and", save_prefix + ".csv")
    return df_results


def run_cross_dataset_probe_experiment(
    config: ProbeConfig,
    train_dataset: str,
    test_dataset: str,
) -> pd.DataFrame:
    """
    Train probes on one dataset and test on another dataset.
    
    Args:
        config: ProbeConfig with model_name, save_dir, error_type, etc.
        train_dataset: Name of dataset to train on
        test_dataset: Name of dataset to test on
    
    Returns:
        DataFrame of results, also saved to disk
    """
    models = make_models(config)
    
    save_root = f"{config.save_dir}/cross_dataset/{train_dataset}_to_{test_dataset}/{config.model_name}"
    os.makedirs(save_root, exist_ok=True)
    save_prefix = f"{save_root}/df_probes_{config.save_name}"
    print(f"[SAVE] Results prefix: {save_prefix}")
    print(f"[CROSS-DATASET] Train on: {train_dataset}, Test on: {test_dataset}")
    
    # Print system resources at start
    print_system_resources()
    
    # Lazy import to avoid circular import
    import sys
    probes_dir = os.path.dirname(os.path.abspath(__file__))
    if probes_dir not in sys.path:
        sys.path.insert(0, probes_dir)
    import probes_data
    load_single_dataset = probes_data.load_single_dataset
    
    # Load train dataset
    print("\n[LOADING TRAIN DATASET]")
    (
        y_error_exact_train,
        y_correct_exact_train,
        activations_exact_train,
        y_error_last_train,
        y_correct_last_train,
        activations_last_train,
    ) = load_single_dataset(train_dataset, config.model_name, config.save_dir, config.error_type)
    
    # Load test dataset
    print("\n[LOADING TEST DATASET]")
    (
        y_error_exact_test,
        y_correct_exact_test,
        activations_exact_test,
        y_error_last_test,
        y_correct_last_test,
        activations_last_test,
    ) = load_single_dataset(test_dataset, config.model_name, config.save_dir, config.error_type)
    
    # Print system resources after loading data
    print_system_resources()
    
    if config.token_pos == "both":
        token_positions = ["exact", "last"]
    else:
        token_positions = [config.token_pos]
    
    probe_results: list[dict] = []
    
    for token_pos in token_positions:
        # Choose which activations to use
        if token_pos == "exact":
            activations_train = activations_exact_train
            activations_test = activations_exact_test
            y_error_train = y_error_exact_train
            y_error_test = y_error_exact_test
            y_correct_train = y_correct_exact_train
            y_correct_test = y_correct_exact_test
        elif token_pos == "last":
            activations_train = activations_last_train
            activations_test = activations_last_test
            y_error_train = y_error_last_train
            y_error_test = y_error_last_test
            y_correct_train = y_correct_last_train
            y_correct_test = y_correct_last_test
        else:
            raise ValueError(f"Unknown token_pos: {token_pos}")
        
        # Verify layers match
        train_layers = set(activations_train.keys())
        test_layers = set(activations_test.keys())
        if train_layers != test_layers:
            print(f"Warning: Layer mismatch. Train layers: {sorted(train_layers)}, Test layers: {sorted(test_layers)}")
            common_layers = sorted(train_layers & test_layers)
            print(f"Using common layers: {common_layers[:5]} ... ({len(common_layers)} total)")
        else:
            common_layers = sorted(train_layers)
        
        print(f"Train dataset shape: {y_error_train.shape}, {len(y_correct_train)} samples")
        print(f"Test dataset shape: {y_error_test.shape}, {len(y_correct_test)} samples")
        print(f"Layers: {common_layers[:5]} ... ({len(common_layers)} total)")
        
        futures = []
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            for task_type, model_dict in models.items():
                print("Task type:", task_type)
                for probe_name, base_model in model_dict.items():
                    print("Probe model:", probe_name)
                    for layer_idx in tqdm(common_layers, desc=f"Layers ({task_type}, {probe_name})"):
                        X_train = activations_train[layer_idx]
                        X_test = activations_test[layer_idx]
                        
                        # Prepare targets
                        if task_type == "classification":
                            y_train = y_correct_train
                            y_test = y_correct_test
                        elif task_type == "regression":
                            y_train = y_error_train
                            y_test = y_error_test
                        else:
                            raise ValueError(f"Unknown task type: {task_type}")
                        
                        futures.append(
                            executor.submit(
                                train_model_on_layer,
                                task_type,
                                probe_name,
                                base_model,
                                layer_idx,
                                None,  # layer_data not used in cross-dataset mode
                                None,  # y_error_exact not used
                                None,  # y_correct_exact not used
                                token_pos,
                                config,
                                f"{train_dataset}_to_{test_dataset}",
                                X_train_separate=X_train,
                                X_test_separate=X_test,
                                y_train_separate=y_train,
                                y_test_separate=y_test,
                            )
                        )
            
            # Print system resources before training starts
            print_system_resources()
            
            completed_count = 0
            total_futures = len(futures)
            checkpoints = {int(total_futures * 0.25), int(total_futures * 0.50), int(total_futures * 0.75)}
            
            for fut in tqdm(as_completed(futures), total=total_futures, desc="Training probes"):
                rows = fut.result()
                probe_results.extend(rows)
                completed_count += 1
                
                # Print system resources at 25%, 50%, 75% completion
                if completed_count in checkpoints:
                    print_system_resources()
    
    df_results = pd.DataFrame(probe_results)
    df_results.to_pickle(save_prefix + ".pkl")
    
    metric_cols = sorted({name for task in METRICS.values() for name in task.keys()})
    dummy_metric_cols = [f"Dummy-{name}" for name in metric_cols]
    essential_cols = [
        "Dataset",
        "LLM_Model",
        "Task",
        "Model",
        "Error-Type",
        "Layer",
        "# of non-zero coefficients",
        "Attempt",
        "Token-Pos",
    ]
    essential_results = df_results[essential_cols + metric_cols + dummy_metric_cols]
    essential_results.to_csv(save_prefix + ".csv", index=False)
    
    # Print system resources at end
    print_system_resources()
    
    print("[DONE] Saved:", save_prefix + ".pkl", "and", save_prefix + ".csv")
    return df_results
