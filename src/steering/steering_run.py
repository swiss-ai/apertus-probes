######################################
####### Evaluate with steering #######
######################################

wandb_key = "207968e7bf072b1e77e9bf0429fb873659230b9d"
import uuid
import argparse
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
import wandb
import torch
import numpy as np
import pandas as pd
from itertools import product
import os
import sys
import json
import pickle
import gc

from tasks.task_handler import *
from cache.cache_utils import *
from probes.probes_utils import *
from steering.steering_utils import *
from steering.base import *
from steering.constants import *

##############################
####### Hyperparameters ######
##############################

parser = argparse.ArgumentParser(
    description="Filter steering methods, tasks, and models and hyperparameters."
)
parser.add_argument(
    "--fname",
    type=str,
    required=True,
    help="Filename identifier (required).",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    required=True,
    help=(
        "Base cache directory under 'mera-runs', expected to contain:\n"
        "  - '<cache_dir>/mix/...': probe results\n"
        "  - '<cache_dir>/processed_datasets/<dataset>/dataset.jsonl': processed datasets\n"
        "  - '<cache_dir>/<dataset>/<model>/{{targets,acts}}.pkl': postprocessed cache"
    ),
)
parser.add_argument(
    "--save_dir",
    type=str,
    required=True,
    help="Save directory for results.",
)

parser.add_argument(
    "--device",
    type=str,
    default="",
    help="Device for single GPU use (default: cuda:0).",
)
parser.add_argument(
    "--wandb_key", type=str, default="private", help="Weights & Biases API key."
)
parser.add_argument(
    "--steering_methods",
    nargs="+",
    required=True,
    help="Steering steering_methods to include (e.g., optimal_probe vanilla_contrastive).",
)
parser.add_argument(
    "--dataset_names",
    nargs="+",
    required=True,
    help="Tasks to include (e.g., sms_spam sentiment_analysis).",
)
parser.add_argument(
    "--model_names",
    nargs="+",
    required=True,
    help="Models to include (e.g., Qwen/Qwen2.5-3B-Instruct).",
)
parser.add_argument(
    "--probe_dataset_name",
    type=str,
    default=None,
    help=(
        "Logical dataset/mixture name from which probes were trained. "
        "If omitted, it defaults to the '+'-joined list of --dataset_names "
        "(e.g. 'mmlu_high_school+ARC-Easy'). Probes are always loaded from "
        "<cache_dir>/mix/<probe_dataset_name>/<model_name>/<probe_file_name>.pkl."
    ),
)
parser.add_argument(
    "--top_k_sets",
    nargs="+",
    type=int,
    default=[50, 100, 200],
    help="Top k constrastive pairs sets (e.g., 50 100, 200).",
)
parser.add_argument(
    "--probe_token_pos",
    type=str,
    default="exact",
    help="What token position to use for probe weights (default: exact).",
)
parser.add_argument(
    "--error_type",
    type=str,
    default="sm",
    help="What error type to use for parsing constrastive sets (default: sm).",
)
parser.add_argument(
    "--objective_key",
    type=str,
    default="Accuracy",
    help="What objective to use for calibration (default: Accuracy).",
)
parser.add_argument(
    "--probe_file_name",
    type=str,
    required=True,
    help="What probe_file_name to use for coefficients (eg.: df_probes_transform).",
)
parser.add_argument(
    "--regression-model-type",
    type=str,
    choices=["linear", "logit"],
    default="linear",
    help=(
        "Type of regression model to use for steering. "
        "'linear' uses L-{alpha} models (Lasso), "
        "'logit' uses Logit-L-{alpha} models (LogitRegression). "
        "Default: linear"
    ),
)

args = parser.parse_args()

# Get the args.
fname = args.fname
# wandb_key = args.wandb_key
cache_dir = args.cache_dir
save_dir = args.save_dir
if not cache_dir.endswith("/"):
    cache_dir = cache_dir + "/"

# Datasets (dataset.jsonl) live under '<cache_dir>/processed_datasets/<dataset>/dataset.jsonl'
dataset_cache_root = cache_dir + "processed_datasets/"
top_k_sets = args.top_k_sets
probe_token_pos = args.probe_token_pos
error_type = args.error_type
objective_key = args.objective_key
probe_file_name = args.probe_file_name
regression_model_type = args.regression_model_type
device = args.device

# Apply validation (if no args, use all)!
dataset_names = args.dataset_names
model_names = args.model_names

if len(dataset_names) != 1:
    raise ValueError(
        "This script evaluates ONE dataset at a time (compute_logits/compute_targets need one dataset_info). "
        "Pass exactly one --dataset_names (the eval dataset). "
        "Use --probe_dataset_name for the mixture probe source."
    )

# Derive the logical probe dataset / mixture name:
# - If user passed --probe_dataset_name, use it verbatim.
# - Otherwise, default to a '+'-joined mixture of all dataset_names.
if args.probe_dataset_name is not None:
    probe_dataset_name = args.probe_dataset_name
else:
    probe_dataset_name = "+".join(dataset_names)
valid_methods = filter_valid(list(SUPPORTED_METHODS.values()), args.steering_methods)
print(f"[INFO] Tasks: {dataset_names} | Models: {model_names}")
print(f"[DEBUG] Valid methods: {valid_methods}")
print(f"[DEBUG] Filtered datasets: {dataset_names}")
print(f"[DEBUG] Filtered models: {model_names}")

for model_name in model_names:

    # Treat all dataset_names as one mixture
    mixture_name = probe_dataset_name
    all_results_list = []
    
    # Save LLM model name before it might get overwritten
    llm_model_name = model_name

    #################################
    ####### Load model once ########
    #################################

    primary_dataset = dataset_names[0]
    base_task_config = TaskConfig(
        cache_dir=dataset_cache_root,
        dataset_name=primary_dataset,
        model_name=model_name,
        device=device,  # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=1,
        flexible_match=True,
    )
    model_handler = ModelHandler(base_task_config)
    nr_layers = model_handler.nr_layers

    ############################################
    ####### Load and mix cached data ###########
    ############################################

    print("[INFO] Building mixed test/ref sets and activations.")
    mixed_test_prompts: list[str] = []
    mixed_test_labels: list[int] = []
    mixed_ref_prompts: list[str] = []
    mixed_ref_labels: list[int] = []
    mixed_y_correct: list[int] = []
    mixed_y_error: list[float] = []
    mixed_activations_per_layer: dict[int, list[np.ndarray]] = {}

    model_id = llm_model_name.split("/")[-1]

    for dataset_name in dataset_names:
        print(f"[INFO] Loading dataset {dataset_name} for mixture {mixture_name}")

        task_config = TaskConfig(
            cache_dir=dataset_cache_root,
            dataset_name=dataset_name,
            model_name=model_name,
            device=device,
            batch_size=1,
            flexible_match=True,
        )
        dataset_handler = DatasetHandler(task_config, tokenizer=model_handler.tokenizer)

        # Acts and targets live under '<cache_dir>/<dataset>/<model>/'
        file_path_acts = f"{cache_dir}{dataset_name}/{llm_model_name.split('/')[-1]}/acts.pkl"
        file_path_targets = f"{cache_dir}{dataset_name}/{llm_model_name.split('/')[-1]}/targets.pkl"
        print("[INFO] Loading targets from", file_path_targets)
        with open(file_path_targets, "rb") as f:
            y_targets = pickle.load(f)

        y_correct_ds = [
            (pred == true).astype(int)
            for pred, true in zip(y_targets["y_pred"], y_targets["y_true"])
        ]
        y_error_ds = (
            1 - np.array(y_targets["y_softmax"])
            if error_type == "sm"
            else y_targets["y_error"]
        )

        with open(file_path_acts, "rb") as f:
            processed_data = pickle.load(f)
        activations_cache_ds = processed_data["activations_cache"]  # dict[layer] -> [N,d]

        # Get test and reference sets for this dataset
        test_prompts = dataset_handler.prompts_test
        test_labels = dataset_handler.y_true_test
        ref_prompts = dataset_handler.prompts_ref
        ref_labels = dataset_handler.y_true_ref

        print(f"[DEBUG] Total dataset size: {len(dataset_handler.prompts)}")
        print(f"[DEBUG] len(prompts_test) = {len(test_prompts)}")
        print(f"[DEBUG] len(y_true_test)  = {len(test_labels)}")
        print(f"[DEBUG] len(prompts_ref)  = {len(ref_prompts)}")
        print(f"[DEBUG] len(y_true_ref)   = {len(ref_labels)}")

        # Fallback: if test/ref are empty, create them manually from the full dataset
        if len(test_prompts) == 0 or len(ref_prompts) == 0:
            print("[WARN] Empty test/ref split from DatasetHandler – falling back to simple slicing.")
            total = len(dataset_handler.prompts)
            num_test = min(task_config.nr_test_samples, total)
            num_ref = min(task_config.nr_ref_samples, max(0, total - num_test))

            test_idxs = list(range(num_test))
            ref_idxs = list(range(num_test, num_test + num_ref))

            test_prompts = [dataset_handler.prompts[i] for i in test_idxs]
            test_labels = [dataset_handler.y_true[i] for i in test_idxs]
            ref_prompts = [dataset_handler.prompts[i] for i in ref_idxs]
            ref_labels = [dataset_handler.y_true[i] for i in ref_idxs]

            print(f"[DEBUG] Fallback len(prompts_test) = {len(test_prompts)}")
            print(f"[DEBUG] Fallback len(prompts_ref)  = {len(ref_prompts)}")

        # Append to mixed containers
        mixed_test_prompts.extend(test_prompts)
        mixed_test_labels.extend(test_labels)
        mixed_ref_prompts.extend(ref_prompts)
        mixed_ref_labels.extend(ref_labels)
        mixed_y_correct.extend(y_correct_ds)
        mixed_y_error.extend(list(y_error_ds))

        # Append activations per layer
        for layer_idx, layer_data in activations_cache_ds.items():
            mixed_activations_per_layer.setdefault(layer_idx, []).append(layer_data)

    # Final mixed arrays
    test_prompts = mixed_test_prompts
    test_labels = mixed_test_labels
    ref_prompts = mixed_ref_prompts
    ref_labels = mixed_ref_labels
    y_correct = np.array(mixed_y_correct, dtype=int)
    y_error = np.array(mixed_y_error, dtype=float)
    activations_cache = {
        layer_idx: np.concatenate(layer_list, axis=0)
        for layer_idx, layer_list in mixed_activations_per_layer.items()
    }

    print(
        f"[INFO] Mixed test size: {len(test_prompts)}, "
        f"mixed ref size: {len(ref_prompts)}"
    )

    ############################################################
    ####### Load mixed probes and best coefficients ############
    ############################################################

    file_path_probes = (
        f"{cache_dir}mix/{mixture_name}/{model_id}/{probe_file_name}.pkl"
    )
    print(f"[INFO] Using probes from mixture: {mixture_name}")
    print(f"[INFO] Probes file: {file_path_probes}")

    df_all_probes = postprocess_df_probes(
        pd.read_pickle(file_path_probes),
        filter_error_type=error_type,
        filter_probe_token_pos=probe_token_pos,
        filter_inputs="activations",
    )

    # Filter by regression model type if specified
    if regression_model_type == "linear":
        # Filter to only L-{alpha} models (exclude Logit-L-{alpha})
        # Keep all classification models, and regression models that start with "L-" but not "Logit-L-"
        mask = (df_all_probes["Task"] != "regression") | (
            df_all_probes["Model"].str.startswith("L-")
            & ~df_all_probes["Model"].str.startswith("Logit-L-")
        )
        df_all_probes = df_all_probes[mask]
        print(f"[INFO] Filtered to linear regression models (L-{{alpha}})")
    elif regression_model_type == "logit":
        # Filter to only Logit-L-{alpha} models
        # Keep all classification models, and regression models that start with "Logit-L-"
        mask = (df_all_probes["Task"] != "regression") | df_all_probes[
            "Model"
        ].str.startswith("Logit-L-")
        df_all_probes = df_all_probes[mask]
        print(f"[INFO] Filtered to logit regression models (Logit-L-{{alpha}})")

    # Define task and metric mappings.
    tasks_metrics = {"regression": "RMSE", "classification": "AUCROC"}
    steering_options = ["best", "worst", "median"]

    steering_dataset_name = dataset_names[0]  # The dataset being steered on
    print(f"[INFO] Steering {llm_model_name}")
    print(f"[INFO]   Probe source (mixture): {mixture_name}")
    print(f"[INFO]   Steering target (dataset): {steering_dataset_name}")
    probe_weights = {}
    probe_intercepts = {}
    probe_models = {}  # Store model names (L-{alpha} or Logit-L-{alpha}) for each layer
    probe_layers = {}

    for task, metric in tasks_metrics.items():
        for steer_flag in steering_options:
            # The mixed probe DataFrame already contains only a single mixture
            # (mixture_name), so we don't need to filter by Dataset here.
            coefficients, intercepts, models = get_best_coefficients(
                df_all_probes,
                task=task,
                metric=metric,
                mode=steer_flag,
            )
            
            # Validate: must have coefficients/intercepts for all layers
            if len(coefficients) != nr_layers or len(intercepts) != nr_layers:
                raise ValueError(
                    f"Expected {nr_layers} coefficients/intercepts for {(task, steer_flag, metric)}, "
                    f"but got {len(coefficients)} coefficients and {len(intercepts)} intercepts. "
                    f"This may indicate missing layers in the probe dataframe."
                )
            
            probe_weights[(task, steer_flag)] = {
                int(i): weights
                for i, weights in zip(range(nr_layers), coefficients)
            }

            probe_intercepts[(task, steer_flag)] = {
                int(i): intercept
                for i, intercept in zip(range(nr_layers), intercepts)
            }
            
            # Store model names for each layer (for printing)
            probe_models[(task, steer_flag)] = {
                int(i): model_name
                for i, model_name in zip(range(nr_layers), models)
            }
            
            # Assert: intercepts must match probe_weights keys
            assert set(probe_intercepts[(task, steer_flag)].keys()) == set(probe_weights[(task, steer_flag)].keys()), (
                f"Intercept keys do not match probe_weights keys for {(task, steer_flag)}"
            )
            
            # Assert: verify that coefficients and intercepts are paired correctly
            # (they should come from the same rows in the dataframe)
            assert len(coefficients) == len(intercepts), (
                f"Number of coefficients ({len(coefficients)}) does not match "
                f"number of intercepts ({len(intercepts)}) for {(task, steer_flag)}"
            )

            probe_layers[(task, steer_flag)] = get_best_layer(
                df_all_probes,
                task=task,
                metric=metric,
                mode=steer_flag,
            )

    # Log all best/worst/median layers for steering (useful for debugging/analysis)
    print("\n" + "="*60)
    print("[INFO] Best layers for steering:")
    print("="*60)
    for task, metric in tasks_metrics.items():
        print(f"\n[{task.upper()}] (metric: {metric}):")
        for steer_flag in steering_options:
            layer = probe_layers.get((task, steer_flag), None)
            # Get the probe model name used for this layer (e.g., L-0.5, Logit-L-0.25)
            probe_model_name = probe_models.get((task, steer_flag), {}).get(layer, "Unknown")
            print(f"  {steer_flag.capitalize()}: Layer {layer} (model: {probe_model_name})")
    print("="*60 + "\n")

    ############################################################
    ####### Build contrastive sets for the mixture ############
    ############################################################

    sets = {
        f"{k}_sets": apply_activation_filtering(
            activations_cache=activations_cache,
            y_correct=y_correct,
            y_error=y_error,
            filter_type="top_k",
            k=k,
        )
        for k in top_k_sets
    }

    nr_best_coefficients_to_steer = [
        (int(k), int(np.count_nonzero(v)))
        for k, v in probe_weights[("regression", "best")].items()
    ]
    nr_worst_coefficients_to_steer = [
        (int(k), int(np.count_nonzero(v)))
        for k, v in probe_weights[("regression", "worst")].items()
    ]
    nr_median_coefficients_to_steer = [
        (int(k), int(np.count_nonzero(v)))
        for k, v in probe_weights[("regression", "median")].items()
    ]
    print(
        f"[INFO] Number of best coefficients to steer per layer {nr_best_coefficients_to_steer}"
    )
    print(
        f"[INFO] Number of worst coefficients to steer per layer {nr_worst_coefficients_to_steer}"
    )
    print(
        f"[INFO] Number of median coefficients to steer per layer {nr_median_coefficients_to_steer}"
    )

    ######################################################
    ####### Prepare steering method hyperparameters ######
    ######################################################

    print("[INFO] Preparing steering method hyperparameters.")

    # Hyperparameters save files: combine mixture_name (probe source) and dataset_name (steering target)
    save_dir_steering = f"{save_dir}{mixture_name}_steered_on_{steering_dataset_name}/{llm_model_name.split('/')[-1]}/steering/"
    os.makedirs(save_dir_steering, exist_ok=True)
    print(f"[INFO] Results will be saved to: {save_dir_steering}")
    save_key = f"{fname}_{len(test_prompts)}"
    file_path_single_run = f"{save_dir_steering}{save_key}_method.pkl"
    file_path_all_runs = f"{save_dir_steering}{save_key}_steering_all_results.pkl"

    base_kwargs = {
        "model": model_handler.model,
        "tokenizer": model_handler.tokenizer,
        "dataset_info": base_task_config.dataset_info,
        "tokenizer_kwargs": base_task_config.tokenizer_kwargs,
        "save_dir": save_dir_steering,
    }
    layers_settings = {
        "all_layers": list(range(nr_layers)),
        # "best_layer": [],
        # "last_layer": [nr_layers],
    }
    token_pos_settings = {
        "all_token_pos": "all",
        # "generation_token_pos": "generation",  # FIXME.
        # , "specific", "probe_position"] "probe_token_pos": probe_token_pos.replace("_", "") if "exact" in probe_token_pos else "last",
    }
    # Derive settings are now determined per method based on task type and regression model type:
    # - For classification: derive_with_logit=True, derive_with_sigmoid=False
    # - For regression with linear models: derive_with_logit=False, derive_with_sigmoid=False
    # - For regression with logit models: derive_with_logit=True, derive_with_sigmoid=False
    # derive_with_sigmoid is always False
    eta = 1.0  # etas = [1.0] #, -1.0] for eta in etas:

    # Add steering methods.
    benchmark_list = {}
    benchmark_list["no_steering"] = {"no_steering": True}
    benchmark_list["prompt_steering"] = {
        "no_steering": True,
        "prompt_addition": "Think before you answer.",
    }

    for token_pos_key, token_pos_to_steer in token_pos_settings.items():

        for layer_key, layers_to_steer in layers_settings.items():

            for k, sets_k in sets.items():

                        ######################################
                        ######## MERA steering methods #######
                        ######################################

                        mera_methods = [
                            (
                                f"optimal_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("regression", "best"),
                                "optimal_probe",
                            ),
                            (
                                f"optimal_logistic_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("classification", "best"),
                                "optimal_probe",
                            ),
                            (
                                f"optimal_contrastive_{eta}_{layer_key}_{token_pos_key}",
                                ("regression", "best"),
                                "optimal_contrastive",
                            ),
                            (
                                f"sub_optimal_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("regression", "worst"),
                                "optimal_probe",
                            ),
                            (
                                f"median_optimal_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("regression", "median"),
                                "optimal_probe",
                            ),
                        ]

                        for method_name, setting, mode in mera_methods:
                            # Set derive settings based on task type and regression model type
                            task_type = setting[0]  # "classification" or "regression"
                            if task_type == "classification":
                                # Classification: use logit derivation
                                derive_with_logit = True
                                derive_with_sigmoid = False
                            else:
                                # Regression: use logit derivation if using logit regression models
                                if regression_model_type == "logit":
                                    derive_with_logit = True
                                else:
                                    derive_with_logit = False
                                derive_with_sigmoid = False
                            
                            # best_alpha_last, best_alpha_exact, best_metric_last, best_metric_exact, _ = get_best_alpha_from_searches(model_name.split("/")[1], dataset_name, threshold=threshold, method_name=method_name_ours)
                            kwargs_mera = {
                                "eta": eta,
                                "alpha_range": list(np.linspace(1e-3, 0.99, 10)),
                                # "refine_best_alpha": False,  # FIXME
                                "ref_prompts": ref_prompts,
                                "ref_labels": ref_labels,
                                "derive_with_sigmoid": derive_with_sigmoid,
                                "derive_with_logit": derive_with_logit,
                                "derive_with_all": True,
                                "apply_token_pos_to_steer": token_pos_to_steer,
                                "objective_key": objective_key,
                                # "nr_samples": 210 if "mmlu" in dataset_name else 250,
                                "best_alpha_last": None,  # FIXME
                                "best_alpha_exact": None,  # FIXME.
                            }

                            kwargs_mera["probe_weights"] = probe_weights[
                                (setting[0], setting[1])
                            ]
                            kwargs_mera["probe_intercepts"] = probe_intercepts[
                                (setting[0], setting[1])
                            ]
                            kwargs_mera["probe_models"] = probe_models[
                                (setting[0], setting[1])
                            ]
                            pw = kwargs_mera["probe_weights"]              # dict: layer -> w
                            avail_layers = set(pw.keys())

                            if layer_key == "best_layer":
                                bestL = probe_layers.get((setting[0], setting[1]), None)
                                desired_layers = [] if bestL is None else [bestL]
                            elif layer_key == "all_layers":
                                desired_layers = list(range(nr_layers))
                            else:
                                desired_layers = layers_to_steer

                            # skip missing layers
                            kwargs_mera["apply_layers_to_steer"] = sorted(set(desired_layers) & avail_layers)

                            kwargs_mera["mode"] = mode
                            kwargs_mera["logging_calibration_table_key"] = method_name
                            kwargs_mera["logging_theta_table_key"] = method_name
                            if "contrastive" in method_name:
                                extras = {
                                    "a": sets_k[0],
                                    "b": sets_k[1],
                                    "k": k,
                                }
                            else:
                                extras = {}
                            benchmark_list[method_name] = {**kwargs_mera, **extras}

                        ####################################
                        ####### Base additive methods ######
                        ####################################

                        base_additive_methods = [
                            (
                                f"additive_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("regression", "best"),
                                "additive_probe",
                            ),
                            (
                                f"additive_sub_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("regression", "worst"),
                                "additive_probe",
                            ),
                            (
                                f"additive_median_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("regression", "median"),
                                "additive_probe",
                            ),
                            (
                                f"additive_logistic_probe_{eta}_{layer_key}_{token_pos_key}",
                                ("classification", "best"),
                                "additive_probe",
                            ),
                        ]
                        kwargs_additive = {
                            "eta": eta,
                            "apply_token_pos_to_steer": token_pos_to_steer,
                        }
                        for method_name, setting, mode in base_additive_methods:
                            pw = probe_weights[(setting[0], setting[1])]
                            pi = probe_intercepts[(setting[0], setting[1])]
                            pm = probe_models[(setting[0], setting[1])]

                            if layer_key == "best_layer":
                                bestL = probe_layers.get((setting[0], setting[1]), None)
                                desired_layers = [] if bestL is None else [bestL]
                            else:
                                desired_layers = layers_to_steer

                            effective_layers = sorted(set(desired_layers) & avail_layers)
                            
                            benchmark_list[method_name] = {
                                **kwargs_additive, 
                                "apply_layers_to_steer": effective_layers,
                                "probe_weights": pw,
                                "probe_intercepts": pi,
                                "probe_models": pm,
                                "mode": mode,
                            } 
                        for key, val in benchmark_list.items():
                            print(
                                f"BENCHMARK {key}:",
                                "\n  probe_weights:", val.get("probe_weights"),
                                "\n  probe_intercepts:", val.get("probe_intercepts"),
                                "\n  mode:", val.get("mode")
                            )

                        ###########################################
                        ####### Baseline contrastive methods ######
                        ###########################################

                        method_name = f"vanilla_contrastive_{eta}_k_{k}_{layer_key}_{token_pos_key}"
                        benchmark_list[method_name] = {
                            "eta": eta,
                            "a": sets_k[0],
                            "b": sets_k[1],
                            "k": k,
                            "apply_token_pos_to_steer": token_pos_to_steer,
                            "apply_layers_to_steer": layers_to_steer,
                        }

        # Filter benchmark_list based on valid methods!
        benchmark_list_filtered = OrderedDict(
            [("no_steering", benchmark_list.get("no_steering", {}))]
            + [
                (k, v)
                for k, v in benchmark_list.items()
                if any(
                    k.startswith(method + "_") or k == method
                    for method in valid_methods
                )
            ]
        )
        print(
            f"[INFO] All benchmark keys: {len(benchmark_list)} steering method(s): {benchmark_list_filtered.keys()}"
        )
        print(
            f"[DEBUG] Testing subset of {len(benchmark_list_filtered)} steering method(s): {benchmark_list_filtered.keys()}"
        )

        ###########################################
        ####### Benchmark steering methods! #######
        ###########################################

        all_keys = {
            "steering_key",
            "dataset_name",
            "alpha_calibration_token_pos_target",
            "best_alpha",
        }
        for steering_key, steering_kwargs in benchmark_list_filtered.items():
            all_keys.update(steering_kwargs.keys())
        all_keys.update([k.replace("overall_evaluation/", "") for k in APPEND_COLS])

        print("\n[INFO] Beginning benchmarking!")
        overall_baseline = None
        first_result = True

        ####################################
        ####### Init logging per task #####
        ####################################

        # Helper function to truncate tags to wandb's 64 character limit
        def truncate_tag(tag, max_len=64):
            """Truncate tag to max_len characters if needed."""
            if len(tag) <= max_len:
                return tag
            return tag[:max_len-3] + "..."

        # Prepare tags, truncating if necessary
        tags_list = [
            fname,
            llm_model_name.split("/")[-1],
            truncate_tag(mixture_name),
            steering_dataset_name,
            "multi_gpu"
        ]

        wandb.init(
            project="MERA",
            name=f"{mixture_name}_on_{steering_dataset_name}-{llm_model_name.split('/')[-1]}-{fname}",
            tags=tags_list,
            group=f"multi_gpu_{fname}",
            config={
                "probe_dataset_name": mixture_name,  # Dataset(s) used to train probes
                "steering_dataset_name": steering_dataset_name,  # Dataset being steered on
                "dataset_name": mixture_name,  # Keep for backward compatibility
                "model_name": llm_model_name.split("/")[-1],
                "fname": fname,
                "nr_test_samples": len(test_prompts),
                "nr_ref_samples": len(ref_prompts),
                "nr_layers": nr_layers,
                "file_path_probes": file_path_probes,
                # Mixed activations aggregated from all datasets in the mixture
                "top_k_sets": top_k_sets,
                "probe_token_pos": probe_token_pos,
                "error_type": error_type,
                "regression_model_type": regression_model_type,
                "steering_methods": valid_methods,
                # "threshold": threshold,
            },
        )

        for ix, (steering_key, steering_kwargs) in enumerate(benchmark_list_filtered.items()):

            # if steering_key.startswith(("no_steering", "prompt_steering")): # FIXME
            #    continue

            # Update table logging keys.
            steering_kwargs = dict(steering_kwargs)
            steering_kwargs["logging_calibration_table_key"] = f"{steering_key}"
            # Note: save_dir is now in base_kwargs, but keep it in steering_kwargs for backward compatibility
            steering_kwargs["save_dir"] = save_dir_steering
            # Initialise and evaluate method on test set.
            print(f"\n[INFO] Processing: {steering_key}")
            steering_init = init_steering(steering_kwargs)(
                **base_kwargs, steering_kwargs=steering_kwargs
            )

            # Set evaluation targets.
            requires_dual_alpha = hasattr(steering_init, "best_alpha_last") and hasattr(
                steering_init, "best_alpha_exact"
            )
            calibration_targets = (
                ["last", "exact"] if requires_dual_alpha else ["single"]
            )

            # Looping over different evaluation targets (last or exact mode).
            for alpha_calibration_token_pos_target in calibration_targets:

                # if steering_kwargs.get("apply_token_pos_to_steer", "") == "generation" and alpha_calibration_token_pos_target == "last":
                #    continue

                target_suffix = (
                    f"_calibrated_for_{alpha_calibration_token_pos_target}"
                    if requires_dual_alpha
                    else ""
                )
                steering_key_with_target = f"{steering_key}{target_suffix}"

                print(
                    f"[INFO] Evaluating with steering_key: {steering_key_with_target}"
                )
                steering_kwargs["logging_theta_table_key"] = (
                    f"{steering_key_with_target}"
                )
                (
                    setattr(
                        steering_init,
                        "logging_theta_table_key",
                        steering_key_with_target,
                    )
                    if hasattr(steering_init, "logging_theta_table_key")
                    else None
                )
                
                model_generation_kwargs = {
                    "pad_token_id": model_handler.tokenizer.eos_token_id,
                    "temperature": 0,
                    "top_p": None
                }
            

                best_alpha_found = None
                if requires_dual_alpha:
                    best_alpha_found = getattr(
                        steering_init,
                        f"best_alpha_{alpha_calibration_token_pos_target}",
                        None,
                    )

                if best_alpha_found is None or best_alpha_found != 1.0:
                    evaluation_metrics = steering_init.evaluate(
                        prompts=test_prompts,
                        labels=test_labels,
                        # errors_baselines=errors_baselines if errors_baselines is not None else None,
                        alpha_calibration_token_pos_target=(
                            alpha_calibration_token_pos_target
                            if requires_dual_alpha
                            else None
                        ),
                        enable_theta_tracking=False,
                        prefix="overall_evaluation/",
                        model_generation_kwargs=model_generation_kwargs,
                    )
                else:
                    # Copy from overall_baseline metrics.
                    evaluation_metrics = deepcopy(evaluation_metrics_baseline)

                print(f"DEBUG evaluation_metrics keys: {evaluation_metrics.keys()}")

                if steering_key == "no_steering":
                    prefix = "overall_evaluation/"
                    overall_baseline = {
                        f"{prefix}{k}": evaluation_metrics[f"{prefix}{k}"]
                        for k in [
                            "Accuracy Last",
                            "F1 Score Last",
                            "Recall Last",
                            "Precision Last",
                            "Error Last",
                            "Correct Predictions Last",
                            "Accuracy Exact",
                            "F1 Score Exact",
                            "Recall Exact",
                            "Precision Exact",
                            "Error Exact",
                            "Correct Predictions Exact",
                            #"Transitions (0->1) Last",
                            #"Transitions (0->1) Exact",
                            #"Transitions (0->0) Last",
                            #"Transitions (0->0) Exact",
                            #"Transitions (1->1) Last",
                            #"Transitions (1->1) Exact",
                            #"Transitions (1->0) Last",
                            #"Transitions (1->0) Exact",
                        ]
                    }
                    overall_baseline.update({k: 0.0 for k in APPEND_COLS})
                    print(f"[INFO] Baseline metrics set: {overall_baseline}")
                    evaluation_metrics_baseline = deepcopy(evaluation_metrics)
                else:
                    evaluation_metrics = append_metrics(
                        evaluation_metrics,
                        overall_baseline,
                        steering_key_with_target,
                        alpha_calibration_token_pos_target,
                        prefix="overall_evaluation/",
                    )

                # Post-processing and logging.
                single_results = {
                    **{
                        "steering_key":          steering_key_with_target,
                        "dataset_name":          mixture_name,
                        "alpha_calibration_token_pos_target": (
                            alpha_calibration_token_pos_target
                            if requires_dual_alpha
                            else ""
                        ),
                        f"best_alpha_{alpha_calibration_token_pos_target}": (
                            getattr(
                                steering_init,
                                f"best_alpha_{alpha_calibration_token_pos_target}",
                                None,
                            )
                            if requires_dual_alpha
                            else None
                        ),
                    },
                    **steering_kwargs,
                    **evaluation_metrics,
                }
                # Some processing.
                excludes = ["ref_", "probe_weights", "inner_evaluation"] # "Correct", 
                single_results = {k: v for k, v in single_results.items() if not any(ex in k for ex in excludes)}

                # Print results.
                print_single_results = {
                    k: v for k, v in single_results.items() if "overall_evaluation" in k
                }
                print(
                    f"[INFO] Single results {steering_key_with_target}\n",
                    " ".join(
                        (
                            f"{k} {v:.3f}"
                            if isinstance(v, (float, int))
                            else f"{k} {np.mean(v):.3f}"
                        )
                        for k, v in print_single_results.items()
                        if "Correct" not in k
                    ),
                )

                single_results = {k.replace('overall_evaluation/', ''): v for k, v in single_results.items()}
                all_results_list.append(single_results)

                # Logging!
                if first_result:
                    all_keys.update(single_results.keys())
                    results_table = wandb.Table(columns=list(all_keys))
                    first_result = False

                results_table.add_data(
                    *[safe_serialize(single_results.get(k, None)) for k in all_keys]
                )
                wandb.log(single_results)

                file_path_single_run_method = file_path_single_run.replace(
                    "method", steering_key_with_target
                )
                with open(file_path_single_run_method, "wb") as f:
                    pickle.dump(single_results, f)

                print(
                    f"[INFO] Steering results saved at: {file_path_single_run_method}.\n"
                )

                del single_results
                torch.cuda.empty_cache()
                gc.collect()

        # Logging!
        wandb.log({"overall_evaluation_results_table": results_table})
    
        # Finish wandb.
        wandb.finish()

    with open(file_path_all_runs, "wb") as f:
        pickle.dump(all_results_list, f)

    print(f"[INFO] Steering results saved at: {file_path_all_runs}")

    df_all_results = pd.DataFrame(all_results_list)
    df_all_results.to_csv(file_path_all_runs.replace(".pkl", ".csv"))
    # df_all_results.sort_values("overall_evaluation/Accuracy", ascending=False).head(20)

    del all_results_list, df_all_results
