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
    help="Cache directory to retrieve the data.",
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
    help="Dataset from which probes were trained (for cross-dataset steering). "
         "If None, use the same dataset as the steering target.",
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

args = parser.parse_args()

# Get the args.
fname = args.fname
# wandb_key = args.wandb_key
cache_dir = args.cache_dir
save_dir = args.save_dir
top_k_sets = args.top_k_sets
probe_token_pos = args.probe_token_pos
error_type = args.error_type
objective_key = args.objective_key
probe_file_name = args.probe_file_name
device = args.device

# Apply validation (if no args, use all)!
dataset_names = args.dataset_names
model_names = args.model_names
valid_methods = filter_valid(list(SUPPORTED_METHODS.values()), args.steering_methods)
print(f"[INFO] Tasks: {dataset_names} | Models: {model_names}")
print(f"[DEBUG] Valid methods: {valid_methods}")
print(f"[DEBUG] Filtered datasets: {dataset_names}")
print(f"[DEBUG] Filtered models: {model_names}")

for model_name in model_names:

   
    all_results_list = []

    for dataset_name in dataset_names:

        #################################
        ####### Load dataset_names ######
        #################################

        task_config = TaskConfig(
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            device=device, #torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=1,
            flexible_match=True,
        )
        model_handler = ModelHandler(task_config)
        dataset_handler = DatasetHandler(task_config, tokenizer=model_handler.tokenizer)
        nr_layers = model_handler.nr_layers
        probe_dataset_name = args.probe_dataset_name or dataset_name
        file_path_acts = f"{save_dir}{dataset_name}/{model_name.split('/')[1]}/acts.pkl"
        file_path_probes = (
            f"{task_config.cache_dir}/{probe_dataset_name}/{model_name.split('/')[-1]}/{probe_file_name}.pkl"
        )

        print(f"[INFO] Using probes from dataset: {probe_dataset_name}")
        print(f"[INFO] Probes file: {file_path_probes}")

        # Hyperparameters save files.
        save_dir_steering = f"{save_dir}{dataset_name}/{model_name.split('/')[1]}/steering/"
        os.makedirs(save_dir_steering, exist_ok=True)
        save_key = f"{fname}_{task_config.nr_test_samples}"
        file_path_single_run = f"{save_dir_steering}{save_key}_method.pkl"
        file_path_all_runs = f"{save_dir_steering}{save_key}_steering_all_results.pkl"

        print(f"[INFO] Steering {model_name} | {dataset_name}")
        print(
            f"[INFO] nr samples (test, cal) ({task_config.nr_test_samples}, {task_config.nr_ref_samples})"
        )

        ############################################
        ####### Load cached errors and lables ######
        ############################################

        print("[INFO] Loading specific post-processed data for vanilla steering.")

        # Load task-specific post-processed data for vanilla steering.

        # y_targets = load_saved_data(
        #     save_dir=f"{save_dir}{dataset_name}/{model_name.split('/')[1]}/",
        #     data_type="targets",
        # )

        file_path_targets = (
            f"{task_config.cache_dir}/{dataset_name}/{model_name.split('/')[-1]}/targets.pkl"
        )
        print("[INFO] Loading targets from", file_path_targets)

        with open(file_path_targets, "rb") as f:
            y_targets = pickle.load(f)

        y_correct = [
            (pred == true).astype(int)
            for pred, true in zip(y_targets["y_pred"], y_targets[f"y_true"])
        ]  # {probe_token_pos}
        y_error = (
            1 - np.array(y_targets[f"y_softmax"])
            if error_type == "sm"
            else y_targets[f"y_error"]
        )  # {probe_token_pos}

        with open(file_path_acts, "rb") as f:
            processed_data = pickle.load(f)
        activations_cache = processed_data["activations_cache"]  # post-processed

        ############################################################
        ####### Get contrastive, test, val sets, coefficients ######
        ############################################################

        print(
            "[INFO] Loading contrastive, test and validation sets and probe coefficients."
        )

        # Get test and reference sets.
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
            num_test = min(nr_test_samples, total)
            num_ref  = min(nr_ref_samples, max(0, total - num_test))

            test_idxs = list(range(num_test))
            ref_idxs  = list(range(num_test, num_test + num_ref))

            test_prompts = [dataset_handler.prompts[i] for i in test_idxs]
            test_labels  = [dataset_handler.y_true[i] for i in test_idxs]
            ref_prompts  = [dataset_handler.prompts[i] for i in ref_idxs]
            ref_labels   = [dataset_handler.y_true[i] for i in ref_idxs]

            print(f"[DEBUG] Fallback len(prompts_test) = {len(test_prompts)}")
            print(f"[DEBUG] Fallback len(prompts_ref)  = {len(ref_prompts)}")

        # Retrieve probe coefficients and contrastive pairs.
        df_all_probes = postprocess_df_probes(
            pd.read_pickle(file_path_probes),
            filter_error_type=error_type,
            filter_probe_token_pos=probe_token_pos,
            filter_inputs="activations",
        )

        ##########################################
        ####### Load best and worst weights ######
        ##########################################

        # Define task and metric mappings.
        tasks_metrics = {"regression": "RMSE", "classification": "AUCROC"}
        steering_options = ["best", "worst", "median"]

        print(f"[INFO] Steering {model_name} | {dataset_name}")
        probe_weights = {}
        probe_layers = {}

        for task, metric in tasks_metrics.items():
            for steer_flag in steering_options:
                probe_weights[(task, steer_flag)] = {
                    int(i): weights
                    for i, weights in zip(
                        range(nr_layers),
                        get_best_coefficients(
                            df_all_probes,
                            dataset_name=probe_dataset_name,
                            task=task,
                            metric=metric,
                            mode=steer_flag,
                        ),
                    )
                }
                
                probe_layers[(task, steer_flag)] = get_best_layer(
                    df_all_probes,
                    task=task,
                    dataset_name=probe_dataset_name,
                    metric=metric,
                    mode=steer_flag,
                )
                

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

        base_kwargs = {
            "model": model_handler.model,
            "tokenizer": model_handler.tokenizer,
            "dataset_info": task_config.dataset_info,
            "tokenizer_kwargs": task_config.tokenizer_kwargs,
            "save_dir": save_dir_steering,
        }
        layers_settings = {
            "all_layers": list(range(nr_layers)),
            # "best_layer": [probe_best_layer],
            # "last_layer": [nr_layers],
        }
        token_pos_settings = {
            "all_token_pos": "all",
            # "generation_token_pos": "generation",  # FIXME.
            # , "specific", "probe_position"] "probe_token_pos": probe_token_pos.replace("_", "") if "exact" in probe_token_pos else "last",
        }
        derive_settings = {
            # "with_logit_only": {
            #     "derive_with_sigmoid": False,
            #     "derive_with_logit": True,
            # },
            # "with_sigmoid_only": {
            #    "derive_with_sigmoid": True,
            #    "derive_with_logit": False,
            # },
            "with_both": {
                "derive_with_sigmoid": True,
                "derive_with_logit": True,
            },
            # "with_none": {
            #    "derive_with_sigmoid": False,
            #    "derive_with_logit": False,
            # },
        }
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

                for derive_key, derive_kwargs in derive_settings.items():

                    for k, sets_k in sets.items():

                        ######################################
                        ######## MERA steering methods #######
                        ######################################

                        mera_methods = [
                            (
                                f"optimal_probe_{eta}_{layer_key}_{token_pos_key}_derive_all_{derive_key}",
                                ("regression", "best"),
                                "optimal_probe",
                            ),
                            (
                                f"optimal_logistic_probe_{eta}_{layer_key}_{token_pos_key}_derive_all_{derive_key}",
                                ("classification", "best"),
                                "optimal_probe",
                            ),
                            (
                                f"optimal_contrastive_{eta}_{layer_key}_{token_pos_key}_derive_all_{derive_key}",
                                ("regression", "best"),
                                "optimal_contrastive",
                            ),
                            (
                                f"sub_optimal_probe_{eta}_{layer_key}_{token_pos_key}_derive_all_{derive_key}",
                                ("regression", "worst"),
                                "optimal_probe",
                            ),
                            (
                                f"median_optimal_probe_{eta}_{layer_key}_{token_pos_key}_derive_all_{derive_key}",
                                ("regression", "median"),
                                "optimal_probe",
                            ),
                        ]

                        # best_alpha_last, best_alpha_exact, best_metric_last, best_metric_exact, _ = get_best_alpha_from_searches(model_name.split("/")[1], dataset_name, threshold=threshold, method_name=method_name_ours)
                        kwargs_mera = {
                            "eta": eta,
                            "alpha_range": list(np.linspace(1e-3, 0.99, 10)),
                            # "refine_best_alpha": False,  # FIXME
                            "ref_prompts": ref_prompts,
                            "ref_labels": ref_labels,
                            "derive_with_sigmoid": derive_kwargs["derive_with_sigmoid"],
                            "derive_with_logit": derive_kwargs["derive_with_logit"],
                            "derive_with_all": True,
                            "apply_token_pos_to_steer": token_pos_to_steer,
                            "apply_layers_to_steer": layers_to_steer,
                            "objective_key": objective_key,
                            # "nr_samples": 210 if "mmlu" in dataset_name else 250,
                            "best_alpha_last": None,  # FIXME
                            "best_alpha_exact": None,  # FIXME.
                        }

                        for method_name, setting, mode in mera_methods:

                            kwargs_mera["probe_weights"] = probe_weights[
                                (setting[0], setting[1])
                            ]

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
                            "apply_layers_to_steer": layers_to_steer,
                        }
                        for method_name, setting, mode in base_additive_methods:
                            kwargs_additive["probe_weights"] = probe_weights[
                                (setting[0], setting[1])
                            ]
                            kwargs_additive["mode"] = mode 
                            benchmark_list[method_name] = kwargs_additive

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

        wandb.init(
            project="MERA",
            name=f"{dataset_name}-{model_name.split('/')[1]}-{fname}",
            config={
                "dataset_name": dataset_name,
                "model_name": model_name.split("/")[1],
                "nr_test_samples": task_config.nr_test_samples,
                "nr_ref_samples": task_config.nr_ref_samples,
                "nr_layers": nr_layers,
                "file_path_probes": file_path_probes,
                "file_path_acts": file_path_acts,
                "top_k_sets": top_k_sets,
                "probe_token_pos": probe_token_pos,
                "error_type": error_type,
                # "threshold": threshold,
            },
        )

        for ix, (steering_key, steering_kwargs) in enumerate(benchmark_list_filtered.items()):

            # if steering_key.startswith(("no_steering", "prompt_steering")): # FIXME
            #    continue

            # Update table logging keys.
            steering_kwargs["logging_calibration_table_key"] = f"{steering_key}"

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
                        "steering_key": steering_key_with_target,
                        "dataset_name": dataset_name,
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
