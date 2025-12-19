import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# -----------------------------
# Paths and config
# -----------------------------

ROOT = Path("/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/steering_outputs")

# Default configurations (can be overridden in function calls)
DEFAULT_DATASETS = {
    "MMLU-HS"      : "mmlu_high_school",
    "MMLU-PROF"    : "mmlu_professional",
    "ARC-EASY"     : "ARC-Easy",
    "ARC-CHALLENGE": "ARC-Challenge",
    "SMS-SPAM"     : "sms_spam",
    "YES-NO"       : "sujet_finance_yesno_5k"
}

DEFAULT_MODELS = {
    "Apertus-8B-2509"          : "Apertus-8B-2509",
    "Apertus-8B-Instruct-2509" : "Apertus-8B-Instruct-2509",
    "Llama-3.1-8B-Instruct" : "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B" : "Llama-3.1-8B",
}

# METHODS: for MERA we give a priority list of prefixes
# Try multiple variants in order of preference
DEFAULT_METHODS = {
    "Baseline (prompting)" : ["prompt_steering"],
    "MERA"            : [
        "optimal_probe_1.0_all_layers_all_token_pos_derive_all_with_both",
        "optimal_probe_1.0_all_layers_all_token_pos",  # fallback without derive_all_with_both
    ],
    "MERA logistic"   : [
        "optimal_logistic_probe_1.0_all_layers_all_token_pos_derive_all_with_both",
        "optimal_logistic_probe_1.0_all_layers_all_token_pos",  # fallback
    ],
    "MERA contrastive": [
        "optimal_contrastive_1.0_all_layers_all_token_pos_derive_all_with_both",
        "optimal_contrastive_1.0_all_layers_all_token_pos",  # fallback
    ],
    "Vanilla Contrastive": [
        "vanilla_contrastive_1.0_all_layers_all_token_pos_derive_all_with_both",
        "vanilla_contrastive_1.0_all_layers_all_token_pos",  # fallback
        "vanilla_contrastive",  # fallback - matches any vanilla_contrastive_* variant
    ],
    "Baseline Error (Additive)"        : ["additive_probe_1.0_all_layers_all_token_pos"],
}


# -----------------------------
# Helpers
# -----------------------------

def is_mixture_dataset(folder_name: str) -> bool:
    """Check if a folder name represents a mixture dataset.
    
    Mixture datasets have the format:
    dataset1+dataset2+..._steered_on_target_dataset
    """
    # Extract just the folder name if it's a path
    name = folder_name.split("/")[-1]
    return "+" in name and "_steered_on_" in name


def create_mixture_label(folder_name: str) -> str:
    """Create a readable label for a mixture dataset folder name.
    
    Example:
        mmlu_high_school+mmlu_professional+..._steered_on_mmlu_professional
        -> "Mixture→Mmlu-Professional"
    """
    if "_steered_on_" in folder_name:
        parts = folder_name.split("_steered_on_")
        target = parts[-1] if len(parts) > 1 else "unknown"
        return f"Mixture→{target.replace('_', '-').title()}"
    return folder_name.replace("_", "-").title()


def discover_datasets(model_folder: str, root_path: Path = ROOT) -> Dict[str, str]:
    """Discover all available datasets (including mixtures) for a given model.
    
    Looks for datasets in two places:
    1. Top-level folders in ROOT (e.g., mmlu_high_school, mixture_steered_on_target)
    2. Subfolders inside dataset folders (e.g., mmlu_high_school/mixture_steered_on_target)
    """
    if not root_path.exists():
        return {}
    
    discovered = {}
    
    # 1. Check top-level folders
    for item in root_path.iterdir():
        if not item.is_dir():
            continue
        
        dataset_name = item.name
        steering_path = item / model_folder / "steering"
        
        if steering_path.exists():
            # Create a readable label
            if is_mixture_dataset(dataset_name):
                # Extract the target dataset from mixture name
                if "_steered_on_" in dataset_name:
                    parts = dataset_name.split("_steered_on_")
                    target = parts[-1] if len(parts) > 1 else "unknown"
                    # Check if it's multi-dataset (has +) or single-dataset steering
                    if "+" in dataset_name:
                        label = f"Mixture→{target.replace('_', '-').title()}"
                    else:
                        label = f"Steered→{target.replace('_', '-').title()}"
                else:
                    label = f"Mixture: {dataset_name[:50]}..."
            else:
                # Find matching label from DEFAULT_DATASETS
                label = None
                for k, v in DEFAULT_DATASETS.items():
                    if v == dataset_name:
                        label = k
                        break
                if label is None:
                    label = dataset_name.replace("_", "-").upper()
            discovered[label] = dataset_name
    
    # 2. Check inside dataset folders for mixture subfolders
    for item in root_path.iterdir():
        if not item.is_dir():
            continue
        
        # Skip if this is already a mixture at top level
        if is_mixture_dataset(item.name):
            continue
        
        # Look for subfolders that might be mixtures
        try:
            for subitem in item.iterdir():
                if not subitem.is_dir():
                    continue
                
                subfolder_name = subitem.name
                # Check if it's a mixture (has _steered_on_ or + pattern)
                if "_steered_on_" in subfolder_name or is_mixture_dataset(subfolder_name):
                    steering_path = subitem / model_folder / "steering"
                    if steering_path.exists():
                        # Create label
                        if "_steered_on_" in subfolder_name:
                            parts = subfolder_name.split("_steered_on_")
                            target = parts[-1] if len(parts) > 1 else "unknown"
                            if "+" in subfolder_name:
                                label = f"Mixture→{target.replace('_', '-').title()}"
                            else:
                                label = f"Steered→{target.replace('_', '-').title()}"
                        else:
                            label = f"Mixture: {subfolder_name[:50]}..."
                        # Store as nested path: parent/subfolder
                        full_path = f"{item.name}/{subfolder_name}"
                        discovered[label] = full_path
        except (PermissionError, OSError):
            # Skip if we can't read the directory
            continue
    
    return discovered


def load_steering_results(dataset_folder: str, model_folder: str, root_path: Path = ROOT) -> pd.DataFrame:
    """Load and concat all *_steering_all_results.pkl for a (dataset, model).
    
    dataset_folder can be:
    - A simple folder name: "mmlu_high_school"
    - A nested path: "mmlu_high_school/mmlu_high_school_steered_on_mmlu_high_school"
    """
    # Handle nested paths (parent/subfolder format)
    if "/" in dataset_folder:
        # Split and join as path components
        path_parts = dataset_folder.split("/")
        steering_dir = root_path / Path(*path_parts) / model_folder / "steering"
    else:
        steering_dir = root_path / dataset_folder / model_folder / "steering"
    if not steering_dir.exists():
        print(f"[WARN] Missing steering dir: {steering_dir}")
        return pd.DataFrame()

    pkl_paths = sorted(steering_dir.glob("*_steering_all_results.pkl"))
    
    if not pkl_paths:
        print(f"[WARN] No steering PKLs found in {steering_dir}")
        return pd.DataFrame()

    dfs = []
    for p in pkl_paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        df = pd.DataFrame(obj)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # de-duplicate just in case
    needed = {"steering_key", "dataset_name", "alpha_calibration_token_pos_target"}
    if needed <= set(df.columns):
        df = df.drop_duplicates(
            subset=list(needed),
            keep="last",
        )

    return df


def pick_row_single(df: pd.DataFrame, key_prefix: str, calib: str | None):
    """Pick row whose steering_key starts with key_prefix (or equals for no_steering)."""
    if key_prefix == "no_steering":
        mask = df["steering_key"].eq("no_steering")
    else:
        mask = df["steering_key"].astype(str).str.startswith(key_prefix)
        if calib is not None and "alpha_calibration_token_pos_target" in df.columns:
            # Try both exact match and checking if calib is in the value
            calib_mask = df["alpha_calibration_token_pos_target"].fillna("").astype(str)
            # Check if calibration target matches (could be "exact" or contain "exact")
            calib_match = calib_mask.eq(calib) | calib_mask.str.contains(calib, na=False)
            mask &= calib_match

    sel = df[mask]
    if sel.empty:
        raise ValueError("no match")
    if len(sel) > 1:
        # just take the first, but warn
        print(f"[INFO] Multiple rows for {key_prefix}, taking first.")
    return sel.iloc[0]


def pick_row_multi(df: pd.DataFrame, key_prefixes: List[str], calib_target: Optional[str] = None, default_calib: str = "both") -> pd.Series:
    """
    Try several prefixes in order (for MERA). For optimal_* we use calib_target (or default_calib if not provided).
    Returns the first successful row.
    """
    if calib_target is None:
        calib_target = default_calib
    last_err = None
    for prefix in key_prefixes:
        try:
            calib = calib_target if prefix.startswith("optimal_") else None
            return pick_row_single(df, prefix, calib)
        except Exception as e:
            last_err = e
            continue
    raise last_err or ValueError("no match in any prefix")


def analyze_steering_results(
    models: Optional[Dict[str, str]] = None,
    datasets: Optional[Dict[str, str]] = None,
    methods: Optional[Dict[str, List[str]]] = None,
    token_position: str = "both",
    steering_type: Optional[str] = "logit",
    calib_target: str = "both",
    n_test: int = 250,
    root_path: Optional[Path] = None,
    include_mixtures: bool = False,
    plot: bool = True,
    save_path: Optional[str] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Analyze steering results and optionally create plots.
    
    Parameters:
    -----------
    models : dict, optional
        Dictionary mapping model labels to model folder names.
        Default: DEFAULT_MODELS
    datasets : dict, optional
        Dictionary mapping dataset labels to dataset folder names.
        Default: DEFAULT_DATASETS
    methods : dict, optional
        Dictionary mapping method labels to lists of steering key prefixes.
        Default: DEFAULT_METHODS
    token_position : str
        Which token position to collect: "exact", "last", or "both"
        Default: "both"
    steering_type : str or None
        Filter by steering type: "linear", "logit", or None for all
        Default: "logit"
    calib_target : str
        Calibration target for optimal_* methods: "exact", "last", or "both"
        Default: "both"
    n_test : int
        Expected number of test samples to filter by
        Default: 250
    root_path : Path, optional
        Root path for steering outputs. Default: ROOT
    plot : bool
        Whether to create and show plots
        Default: True
    save_path : str, optional
        Path to save the plot. If provided, the plot will be saved to this location.
        Can be a full path or just a filename (will save in current directory).
        If None, plot is only displayed (not saved).
        Default: None
    debug : bool
        Whether to print debug information
        Default: True
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with steering results (SPI values, error percentiles, etc.)
    """
    # Use defaults if not provided
    if models is None:
        models = DEFAULT_MODELS
    if datasets is None:
        datasets = DEFAULT_DATASETS
    if methods is None:
        methods = DEFAULT_METHODS
    if root_path is None:
        root_path = ROOT
    
    if debug:
        print(f"Using steering type: {steering_type if steering_type else 'all'}")
        print(f"Token position: {token_position}")
        print(f"Calibration target: {calib_target}")
    
    def load_steering_results_local(dataset_folder: str, model_folder: str) -> pd.DataFrame:
        """Load and concat all *_steering_all_results.pkl for a (dataset, model).
        
        Supports both regular datasets and mixture datasets.
        Mixture datasets have folder names like:
        mmlu_high_school+mmlu_professional+ARC-Challenge+ARC-Easy+sms_spam+sujet_finance_yesno_5k_steered_on_mmlu_professional
        """
        # Handle nested paths (parent/subfolder format) if needed
        if "/" in dataset_folder:
            path_parts = dataset_folder.split("/")
            steering_dir = root_path / Path(*path_parts) / model_folder / "steering"
        else:
            # Regular path - works for both regular and mixture datasets
            steering_dir = root_path / dataset_folder / model_folder / "steering"
        if not steering_dir.exists():
            if debug:
                print(f"[WARN] Missing steering dir: {steering_dir}")
            return pd.DataFrame()

        pkl_paths = sorted(steering_dir.glob("*_steering_all_results.pkl"))
        
        if steering_type is not None:
            filtered_paths = []
            for p in pkl_paths:
                if f"_{steering_type}_" in p.name or "derive_all_with_both" in p.name:
                    filtered_paths.append(p)
            pkl_paths = filtered_paths
            if not pkl_paths:
                if debug:
                    print(f"[WARN] No steering PKLs found for type '{steering_type}' in {steering_dir}")
                return pd.DataFrame()
        
        if not pkl_paths:
            if debug:
                print(f"[WARN] No steering PKLs found in {steering_dir}")
            return pd.DataFrame()

        dfs = []
        for p in pkl_paths:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            df = pd.DataFrame(obj)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        needed = {"steering_key", "dataset_name", "alpha_calibration_token_pos_target"}
        if needed <= set(df.columns):
            df = df.drop_duplicates(subset=list(needed), keep="last")
        return df
    
    # Collect SPI + percentile data
    records = []
    
    # Process datasets - supports both regular and mixture datasets
    for ds_label, ds_folder in datasets.items():
        # If user provided a mixture folder name but no custom label, create one
        if is_mixture_dataset(ds_folder) and ds_label == ds_folder:
            ds_label = create_mixture_label(ds_folder)
            if debug:
                print(f"[INFO] Auto-generated label for mixture: {ds_label}")
        for model_label, model_folder in models.items():
            df = load_steering_results_local(ds_folder, model_folder)
            if df.empty:
                continue

            if "nr_test_samples" in df.columns:
                df = df[df["nr_test_samples"] == n_test]

            # Debug: print available steering_keys for first dataset
            if debug and ds_label == list(datasets.keys())[0] and model_label == list(models.keys())[0]:
                print(f"\n[DEBUG] Available steering_keys for {ds_label} / {model_label}:")
                if "steering_key" in df.columns:
                    for key in sorted(df["steering_key"].unique()):
                        print(f"  - {key}")
                if "alpha_calibration_token_pos_target" in df.columns:
                    print(f"\n[DEBUG] Available calibration targets:")
                    for calib in sorted(df["alpha_calibration_token_pos_target"].unique()):
                        print(f"  - {calib}")

            for method_label, prefixes in methods.items():
                rows_to_use = {}
                
                if token_position in ["exact", "both"]:
                    try:
                        rows_to_use["exact"] = pick_row_multi(df, prefixes, calib_target="exact", default_calib=calib_target)
                    except Exception as e:
                        if debug and ds_label == list(datasets.keys())[0] and model_label == list(models.keys())[0]:
                            print(f"[WARN] Missing {method_label} (exact) for {ds_label} / {model_label}")
                        rows_to_use["exact"] = None
                
                if token_position in ["last", "both"]:
                    try:
                        rows_to_use["last"] = pick_row_multi(df, prefixes, calib_target="last", default_calib=calib_target)
                    except Exception as e:
                        if debug and ds_label == list(datasets.keys())[0] and model_label == list(models.keys())[0]:
                            print(f"[WARN] Missing {method_label} (last) for {ds_label} / {model_label}")
                        rows_to_use["last"] = None
                
                # Skip if no rows found for required token positions
                if token_position == "exact" and rows_to_use.get("exact") is None:
                    continue
                if token_position == "last" and rows_to_use.get("last") is None:
                    continue
                if token_position == "both" and rows_to_use.get("exact") is None and rows_to_use.get("last") is None:
                    continue
                
                row_exact = rows_to_use.get("exact")
                row_last = rows_to_use.get("last")
                row = row_exact if row_exact is not None else row_last
                if row is None:
                    continue

                spi_exact = np.nan
                spi_last = np.nan
                
                if rows_to_use.get("exact") is not None:
                    row_exact_data = rows_to_use["exact"]
                    # Try multiple possible column names (with and without prefix)
                    spi_exact = np.nan
                    for key in ["SPI Exact", "SPI_Exact", "overall_evaluation/SPI Exact", 
                               "overall_evaluation/SPI_Exact", "SPI Exact Exact"]:
                        if key in row_exact_data:
                            spi_exact = row_exact_data[key]
                            break
                    
                    # If still not found, try to find any key containing "SPI" and "Exact"
                    if pd.isna(spi_exact):
                        for key in row_exact_data.keys():
                            if "SPI" in str(key) and "Exact" in str(key):
                                spi_exact = row_exact_data[key]
                                if debug and ds_label == list(datasets.keys())[0] and model_label == list(models.keys())[0]:
                                    print(f"[DEBUG] Found SPI Exact using key: {key}")
                                break
                    
                    # Debug: print available keys if still not found
                    if pd.isna(spi_exact) and debug and ds_label == list(datasets.keys())[0] and model_label == list(models.keys())[0]:
                        available_keys = [k for k in row_exact_data.keys() if "SPI" in str(k)]
                        if available_keys:
                            print(f"[DEBUG] Available SPI keys for exact: {available_keys}")
                        else:
                            print(f"[DEBUG] Row for 'exact' has keys (first 20): {list(row_exact_data.keys())[:20]}")
                
                if rows_to_use.get("last") is not None:
                    row_last_data = rows_to_use["last"]
                    # Try multiple possible column names (with and without prefix)
                    spi_last = np.nan
                    for key in ["SPI Last", "SPI_Last", "overall_evaluation/SPI Last", 
                               "overall_evaluation/SPI_Last", "SPI Last Last"]:
                        if key in row_last_data:
                            spi_last = row_last_data[key]
                            break
                    
                    # If still not found, try to find any key containing "SPI" and "Last"
                    if pd.isna(spi_last):
                        for key in row_last_data.keys():
                            if "SPI" in str(key) and "Last" in str(key):
                                spi_last = row_last_data[key]
                                if debug and ds_label == list(datasets.keys())[0] and model_label == list(models.keys())[0]:
                                    print(f"[DEBUG] Found SPI Last using key: {key}")
                                break
                    
                    # Debug: print available keys if still not found
                    if pd.isna(spi_last) and debug and ds_label == list(datasets.keys())[0] and model_label == list(models.keys())[0]:
                        available_keys = [k for k in row_last_data.keys() if "SPI" in str(k)]
                        if available_keys:
                            print(f"[DEBUG] Available SPI keys for last: {available_keys}")
                        else:
                            print(f"[DEBUG] Row for 'last' has keys (first 20): {list(row_last_data.keys())[:20]}")
                
                if prefixes == ["no_steering"]:
                    if pd.isna(spi_exact) or spi_exact == "":
                        spi_exact = 0.0
                    if pd.isna(spi_last) or spi_last == "":
                        spi_last = 0.0

                rec = {"Dataset": ds_label, "Model": model_label, "Method": method_label}
                
                if token_position in ["exact", "both"]:
                    rec["SPI Exact"] = float(spi_exact)
                if token_position in ["last", "both"]:
                    rec["SPI Last"] = float(spi_last)
                
                # error percentiles
                if token_position in ["exact", "both"] and rows_to_use.get("exact") is not None:
                    row_exact = rows_to_use["exact"]
                    rec["Err25 Exact"] = row_exact.get("Error Exact 25th Percentile", np.nan)
                    rec["Err75 Exact"] = row_exact.get("Error Exact 75th Percentile", np.nan)
                    rec["Err90 Exact"] = row_exact.get("Error Exact 90th Percentile", np.nan)
                    rec["Err95 Exact"] = row_exact.get("Error Exact 95th Percentile", np.nan)
                if token_position in ["last", "both"] and rows_to_use.get("last") is not None:
                    row_last = rows_to_use["last"]
                    rec["Err25 Last"] = row_last.get("Error Last 25th Percentile", np.nan)
                    rec["Err75 Last"] = row_last.get("Error Last 75th Percentile", np.nan)
                    rec["Err90 Last"] = row_last.get("Error Last 90th Percentile", np.nan)
                    rec["Err95 Last"] = row_last.get("Error Last 95th Percentile", np.nan)
                
                records.append(rec)

    res = pd.DataFrame(records)
    
    if debug:
        print(f"\nResults shape: {res.shape}")
        print(res.head(30))
    
    # Plot if requested
    if plot and not res.empty:
        _plot_results(res, models, datasets, methods, token_position, save_path=save_path)
    
    return res


def _plot_results(
    res: pd.DataFrame,
    models: Dict[str, str],
    datasets: Dict[str, str],
    methods: Dict[str, List[str]],
    token_position: str,
    save_path: Optional[str] = None,
):
    """
    Internal function to create plots from results DataFrame.
    
    Parameters:
    -----------
    res : pd.DataFrame
        Results dataframe with SPI values
    models : dict
        Dictionary mapping model labels to model folder names
    datasets : dict
        Dictionary mapping dataset labels to dataset folder names
    methods : dict
        Dictionary mapping method labels to lists of steering key prefixes
    token_position : str
        Token position used: "exact", "last", or "both"
    save_path : str, optional
        Path to save the plot. If None, plot is only displayed.
    """
    methods_order = list(methods.keys())
    models_order = list(models.keys())
    datasets_order = list(datasets.keys())

    spi_columns = []
    if token_position in ["exact", "both"]:
        spi_columns.append("SPI Exact")
    if token_position in ["last", "both"]:
        spi_columns.append("SPI Last")

    if not spi_columns:
        raise ValueError("token_position must be 'exact', 'last', or 'both'")

    n_plots = len(spi_columns)
    fig, axes = plt.subplots(1, n_plots * 2, figsize=(18 * n_plots, 5))

    plot_idx = 0
    for spi_col in spi_columns:
        token_pos_label = spi_col.replace("SPI ", "")
        
        # Model-specific view
        ax0 = axes[plot_idx]
        grouped_model = (
            res.groupby(["Model", "Method"])[spi_col]
               .mean()
               .unstack("Method")
               .reindex(models_order)
        )

        x = np.arange(len(models_order))
        width = 0.09
        k = len(methods_order)
        offsets = (np.arange(k) - (k - 1) / 2) * width

        for i, method in enumerate(methods_order):
            if method not in grouped_model.columns:
                continue
            vals = grouped_model[method].values
            ax0.bar(x + offsets[i], vals, width=width, label=method)

        ax0.axhline(0, linewidth=1)
        ax0.set_xticks(x)
        ax0.set_xticklabels(models_order, ha="center")
        ax0.set_ylabel("Steering Performance Impact (SPI)")
        ax0.set_title(f"Model-specific View ({token_pos_label})")

        # Dataset-specific view
        ax1 = axes[plot_idx + 1]
        grouped_ds = (
            res.groupby(["Dataset", "Method"])[spi_col]
               .mean()
               .unstack("Method")
               .reindex(datasets_order)
        )

        x = np.arange(len(datasets_order))
        for i, method in enumerate(methods_order):
            if method not in grouped_ds.columns:
                continue
            vals = grouped_ds[method].values
            ax1.bar(x + offsets[i], vals, width=width, label=method)

        ax1.axhline(0, linewidth=1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets_order, ha="center")
        ax1.set_ylabel(f"SPI ({token_pos_label})")
        ax1.set_title(f"Dataset-specific View ({token_pos_label})")
        
        plot_idx += 2

    for ax in axes:
        ax.legend().remove()

    handles_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles_labels[0],
        handles_labels[1],
        loc="upper center",
        ncol=len(methods_order),
        bbox_to_anchor=(0.5, 1.12),
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure the file has an extension (default to .png)
        if not save_path_obj.suffix:
            save_path = str(save_path_obj) + ".png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def get_spi_dataframe(
    models: Optional[Dict[str, str]] = None,
    datasets: Optional[Dict[str, str]] = None,
    methods: Optional[Dict[str, List[str]]] = None,
    token_positions: List[str] = ["exact", "last"],
    probe_types: List[str] = ["linear", "logit"],
    calib_target: str = "both",
    n_test: int = 250,
    root_path: Optional[Path] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Get comprehensive SPI dataframe for all combinations of models, datasets, 
    probe types (linear/logit), and token positions.
    
    Parameters:
    -----------
    models : dict, optional
        Dictionary mapping model labels to model folder names.
        Default: DEFAULT_MODELS
    datasets : dict, optional
        Dictionary mapping dataset labels to dataset folder names.
        Default: DEFAULT_DATASETS
    methods : dict, optional
        Dictionary mapping method labels to lists of steering key prefixes.
        Default: DEFAULT_METHODS
    token_positions : list of str
        List of token positions to collect: ["exact"], ["last"], or ["exact", "last"]
        Default: ["exact", "last"]
    probe_types : list of str
        List of probe types to collect: ["linear"], ["logit"], or ["linear", "logit"]
        Default: ["linear", "logit"]
    calib_target : str
        Calibration target for optimal_* methods: "exact", "last", or "both"
        Default: "both"
    n_test : int
        Expected number of test samples to filter by
        Default: 250
    root_path : Path, optional
        Root path for steering outputs. Default: ROOT
    debug : bool
        Whether to print debug information
        Default: True
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - Model: Model name
        - Dataset: Dataset name
        - Probe Type: "linear" or "logit"
        - Token Position: "exact" or "last"
        - Method: Steering method name
        - SPI: SPI value for the given token position
        - Additional error percentile columns if available
    """
    # Use defaults if not provided
    if models is None:
        models = DEFAULT_MODELS
    if datasets is None:
        datasets = DEFAULT_DATASETS
    if methods is None:
        methods = DEFAULT_METHODS
    if root_path is None:
        root_path = ROOT
    
    all_records = []
    
    # Loop through all combinations
    for probe_type in probe_types:
        if debug:
            print(f"\n{'='*60}")
            print(f"Processing probe type: {probe_type}")
            print(f"{'='*60}")
        
        for token_pos in token_positions:
            if debug:
                print(f"\nProcessing token position: {token_pos}")
            
            # Call analyze_steering_results for this combination
            # Set plot=False to avoid creating plots for each combination
            df = analyze_steering_results(
                models=models,
                datasets=datasets,
                methods=methods,
                token_position=token_pos,
                steering_type=probe_type,
                calib_target=calib_target,
                n_test=n_test,
                root_path=root_path,
                plot=False,  # Don't plot individual combinations
                debug=debug,
            )
            
            if df.empty:
                if debug:
                    print(f"  No data found for {probe_type} / {token_pos}")
                continue
            
            # Add probe type and token position columns
            df["Probe Type"] = probe_type
            df["Token Position"] = token_pos
            
            # Reshape the dataframe to have one row per (Model, Dataset, Probe Type, Token Position, Method)
            # If we have both "SPI Exact" and "SPI Last", we need to handle them separately
            if token_pos == "exact" and "SPI Exact" in df.columns:
                df_reshaped = df.copy()
                df_reshaped["SPI"] = df_reshaped["SPI Exact"]
                # Drop the other SPI column if it exists
                if "SPI Last" in df_reshaped.columns:
                    df_reshaped = df_reshaped.drop(columns=["SPI Last"])
                all_records.append(df_reshaped)
            elif token_pos == "last" and "SPI Last" in df.columns:
                df_reshaped = df.copy()
                df_reshaped["SPI"] = df_reshaped["SPI Last"]
                # Drop the other SPI column if it exists
                if "SPI Exact" in df_reshaped.columns:
                    df_reshaped = df_reshaped.drop(columns=["SPI Exact"])
                all_records.append(df_reshaped)
            else:
                # If neither SPI column exists, skip or handle gracefully
                if debug:
                    print(f"  Warning: No SPI column found for {probe_type} / {token_pos}")
                continue
    
    # Combine all records
    if not all_records:
        if debug:
            print("\nNo data found for any combination!")
        return pd.DataFrame()
    
    result_df = pd.concat(all_records, ignore_index=True)
    
    # Reorder columns for better readability
    priority_columns = ["Model", "Dataset", "Probe Type", "Token Position", "Method", "SPI"]
    other_columns = [col for col in result_df.columns if col not in priority_columns]
    column_order = priority_columns + other_columns
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in result_df.columns]
    result_df = result_df[column_order]
    
    if debug:
        print(f"\n{'='*60}")
        print(f"Final dataframe shape: {result_df.shape}")
        print(f"Columns: {list(result_df.columns)}")
        print(f"\nFirst few rows:")
        print(result_df.head(20))
        print(f"\n{'='*60}")
    
    return result_df


def get_spi_difference_same_vs_mixture(
    same_datasets: Optional[Dict[str, str]] = None,
    mixture_datasets: Optional[Dict[str, str]] = None,
    models: Optional[Dict[str, str]] = None,
    methods: Optional[Dict[str, List[str]]] = None,
    token_positions: List[str] = ["exact", "last"],
    probe_types: List[str] = ["linear", "logit"],
    calib_target: str = "both",
    n_test: int = 250,
    root_path: Optional[Path] = None,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Get the difference in SPI between same-dataset training and mixture training.
    
    For each (model, dataset, probe_type, token_position, method) combination:
    - Same-dataset: SPI from same_datasets (trained on same dataset, steered on same dataset)
    - Mixture: SPI from mixture_datasets (trained on mixture, steered on target dataset)
    - Difference: SPI_mixture - SPI_same
    
    Parameters:
    -----------
    same_datasets : dict, optional
        Dictionary mapping dataset labels to folder names for same-dataset training.
        Default: DEFAULT_DATASETS
    mixture_datasets : dict, optional
        Dictionary mapping dataset labels to folder names for mixture training.
        Should contain mixture folders like "mmlu_high_school+mmlu_professional_steered_on_mmlu_professional"
        Default: None (will try to auto-discover)
    models : dict, optional
        Dictionary mapping model labels to model folder names.
        Default: DEFAULT_MODELS
    methods : dict, optional
        Dictionary mapping method labels to lists of steering key prefixes.
        Default: DEFAULT_METHODS
    token_positions : list of str
        List of token positions to collect: ["exact"], ["last"], or ["exact", "last"]
        Default: ["exact", "last"]
    probe_types : list of str
        List of probe types to collect: ["linear"], ["logit"], or ["linear", "logit"]
        Default: ["linear", "logit"]
    calib_target : str
        Calibration target for optimal_* methods: "exact", "last", or "both"
        Default: "both"
    n_test : int
        Expected number of test samples to filter by
        Default: 250
    root_path : Path, optional
        Root path for steering outputs. Default: ROOT
    debug : bool
        Whether to print debug information
        Default: True
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - Model: Model name
        - Dataset: Dataset name (target dataset being steered on)
        - Probe Type: "linear" or "logit"
        - Token Position: "exact" or "last"
        - Method: Steering method name
        - SPI Same: SPI value from same-dataset training
        - SPI Mixture: SPI value from mixture training
        - SPI Difference: SPI_mixture - SPI_same
    """
    # Use defaults if not provided
    if models is None:
        models = DEFAULT_MODELS
    if same_datasets is None:
        same_datasets = DEFAULT_DATASETS
    if methods is None:
        methods = DEFAULT_METHODS
    if root_path is None:
        root_path = ROOT
    
    if debug:
        print("=" * 80)
        print("Calculating SPI differences: Same-dataset vs Mixture training")
        print("=" * 80)
    
    # Step 1: Get SPI data for same-dataset training
    if debug:
        print("\n[Step 1] Collecting SPI data for same-dataset training...")
    df_same = get_spi_dataframe(
        models=models,
        datasets=same_datasets,
        methods=methods,
        token_positions=token_positions,
        probe_types=probe_types,
        calib_target=calib_target,
        n_test=n_test,
        root_path=root_path,
        debug=debug,
    )
    
    if df_same.empty:
        if debug:
            print("[WARN] No same-dataset data found")
        return pd.DataFrame()
    
    # Step 2: Get SPI data for mixture training
    if debug:
        print("\n[Step 2] Collecting SPI data for mixture training...")
    
    if mixture_datasets is None:
        if debug:
            print("[INFO] Auto-discovering mixture datasets...")
        # Auto-discover mixture folders
        mixture_datasets = {}
        if root_path.exists():
            for item in root_path.iterdir():
                if not item.is_dir():
                    continue
                if is_mixture_dataset(item.name):
                    label = create_mixture_label(item.name)
                    mixture_datasets[label] = item.name
        
        if not mixture_datasets:
            if debug:
                print("[WARN] No mixture datasets found")
            return pd.DataFrame()
    
    df_mixture = get_spi_dataframe(
        models=models,
        datasets=mixture_datasets,
        methods=methods,
        token_positions=token_positions,
        probe_types=probe_types,
        calib_target=calib_target,
        n_test=n_test,
        root_path=root_path,
        debug=debug,
    )
    
    if df_mixture.empty:
        if debug:
            print("[WARN] No mixture data found")
        return pd.DataFrame()
    
    # Step 3: Merge and calculate differences
    if debug:
        print("\n[Step 3] Calculating differences...")
    
    # Rename SPI columns for clarity
    df_same_prep = df_same.copy().rename(columns={"SPI": "SPI Same"})
    df_mixture_prep = df_mixture.copy().rename(columns={"SPI": "SPI Mixture"})
    
    # Match by dictionary keys
    # Extract target dataset key from mixture dataset keys (e.g., "Mix→MMLU-HS" -> "MMLU-HS")
    mixture_key_to_target_key = {}
    for mix_key in mixture_datasets.keys():
        # Try to extract target key from mixture key
        # Common patterns: "Mix→MMLU-HS", "Mixture→MMLU-Prof", etc.
        if "→" in mix_key:
            target_key = mix_key.split("→")[-1]
        elif "->" in mix_key:
            target_key = mix_key.split("->")[-1]
        elif mix_key.startswith("Mix"):
            # Try to extract after "Mix" prefix
            target_key = mix_key.replace("Mix", "").replace("→", "").replace("->", "").strip()
        else:
            # If no pattern found, try to match by checking if key exists in same_datasets
            target_key = None
            for same_key in same_datasets.keys():
                if same_key in mix_key or mix_key in same_key:
                    target_key = same_key
                    break
        
        if target_key and target_key in same_datasets:
            mixture_key_to_target_key[mix_key] = target_key
        elif debug:
            print(f"[WARN] Could not match mixture key '{mix_key}' to any same_datasets key")
    
    if debug:
        print(f"Matched {len(mixture_key_to_target_key)} mixture datasets to same datasets:")
        for mix_key, target_key in mixture_key_to_target_key.items():
            print(f"  {mix_key} -> {target_key}")
    
    # For df_same: Dataset column contains labels (keys from same_datasets, e.g., "MMLU-HS")
    # This is already the target key we want to match on
    df_same_prep["Target Dataset Key"] = df_same_prep["Dataset"]
    
    # For df_mixture: Dataset column contains mixture labels (keys from mixture_datasets, e.g., "Mix→MMLU-HS")
    # Map to target dataset key
    df_mixture_prep["Target Dataset Key"] = df_mixture_prep["Dataset"].map(mixture_key_to_target_key)
    
    # Remove rows where matching failed
    df_mixture_prep = df_mixture_prep.dropna(subset=["Target Dataset Key"])
    
    # Merge on: Model, Target Dataset Key, Probe Type, Token Position, Method
    merge_keys = ["Model", "Target Dataset Key", "Probe Type", "Token Position", "Method"]
    
    # Merge
    df_merged = pd.merge(
        df_same_prep[merge_keys + ["SPI Same"]],
        df_mixture_prep[merge_keys + ["SPI Mixture"]],
        on=merge_keys,
        how="inner",  # Only keep rows where both exist
    )
    
    # Calculate difference
    df_merged["SPI Difference"] = df_merged["SPI Mixture"] - df_merged["SPI Same"]
    
    # Rename Target Dataset Key back to Dataset for consistency
    df_merged = df_merged.rename(columns={"Target Dataset Key": "Dataset"})
    
    # Reorder columns
    column_order = ["Model", "Dataset", "Probe Type", "Token Position", "Method", 
                   "SPI Same", "SPI Mixture", "SPI Difference"]
    df_merged = df_merged[[col for col in column_order if col in df_merged.columns]]
    
    if debug:
        print(f"\n[SUCCESS] Calculated differences for {len(df_merged)} combinations")
        print(f"  Rows: {len(df_merged)}")
        print(f"  Columns: {list(df_merged.columns)}")
    
    return df_merged


def plot_spi_difference_heatmap_matrix(
    df: pd.DataFrame,
    steering_methods: List[str],
    probe_types: List[str],
    token_position: str,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
    fmt: str = ".2f",
    show_colorbar: bool = True,
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Create a matrix of heatmaps showing SPI differences (mixture - same) with steering methods as rows and probe types as columns.
    Each subplot shows models vs datasets for that specific combination.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from get_spi_difference_same_vs_mixture() with columns: Model, Dataset, Probe Type, 
        Token Position, Method, SPI Difference
    steering_methods : list of str
        List of steering methods to plot (rows of the matrix)
        e.g., ["MERA", "MERA contrastive", "Vanilla Contrastive"]
    probe_types : list of str
        List of probe types to plot (columns of the matrix)
        e.g., ["linear", "logit"]
    token_position : str
        Token position to use: "exact" or "last"
    save_path : str, optional
        Path to save the heatmap matrix. If None, plot is only displayed.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-sized based on grid.
    cmap : str
        Colormap name (default: "RdYlGn" for red-yellow-green)
    vmin : float, optional
        Minimum value for color scale. If None, auto-determined.
    vmax : float, optional
        Maximum value for color scale. If None, auto-determined.
    annotate : bool
        Whether to annotate cells with SPI difference values (default: True)
    fmt : str
        Format string for annotations (default: ".2f")
    show_colorbar : bool
        Whether to show a shared colorbar for all subplots (default: True)
    title : str, optional
        Overall title for the figure. If None, auto-generated.
    show : bool
        Whether to display the plot using plt.show() (default: True)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    n_rows = len(steering_methods)
    n_cols = len(probe_types)
    
    # Auto-size figure if not provided
    if figsize is None:
        # Estimate based on number of models and datasets
        sample_df = df[
            (df["Method"].isin(steering_methods)) &
            (df["Probe Type"].isin(probe_types)) &
            (df["Token Position"] == token_position)
        ]
        if not sample_df.empty:
            n_models = len(sample_df["Model"].unique())
            n_datasets = len(sample_df["Dataset"].unique())
            width = max(12, n_cols * max(6, n_datasets * 1.0))
            height = max(8, n_rows * max(5, n_models * 0.6))
            figsize = (width, height)
        else:
            figsize = (n_cols * 6, n_rows * 5)
    
    # Auto-determine vmin/vmax if not provided
    if vmin is None or vmax is None:
        filtered_df = df[
            (df["Method"].isin(steering_methods)) &
            (df["Probe Type"].isin(probe_types)) &
            (df["Token Position"] == token_position)
        ]
        if not filtered_df.empty and "SPI Difference" in filtered_df.columns:
            if vmin is None:
                vmin = filtered_df["SPI Difference"].min()
            if vmax is None:
                vmax = filtered_df["SPI Difference"].max()
        else:
            vmin = vmin or -1.0
            vmax = vmax or 1.0
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Plot each combination
    for row_idx, method in enumerate(steering_methods):
        for col_idx, probe_type in enumerate(probe_types):
            ax = axes[row_idx, col_idx]
            
            # Determine which labels to show
            show_xlabel = (row_idx == n_rows - 1)
            show_ylabel = (col_idx == 0)
            show_title = False
            
            # Filter data for this combination
            filtered_df = df[
                (df["Method"] == method) &
                (df["Probe Type"] == probe_type) &
                (df["Token Position"] == token_position)
            ]
            
            if filtered_df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Pivot for heatmap: models (rows) vs datasets (columns)
            heatmap_data = filtered_df.pivot_table(
                values="SPI Difference",
                index="Model",
                columns="Dataset",
                aggfunc="first"
            )
            
            # Plot heatmap using the existing function but with SPI Difference
            # We'll use plot_spi_heatmap_on_axis but need to adapt it
            if HAS_SEABORN:
                x_ticklabels = heatmap_data.columns if show_xlabel else False
                y_ticklabels = heatmap_data.index if show_ylabel else False
                
                sns.heatmap(
                    heatmap_data,
                    annot=annotate,
                    fmt=fmt,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    center=0,  # Center at 0 for differences
                    cbar_kws={"label": "SPI Difference"},
                    ax=ax,
                    linewidths=0.5,
                    linecolor='gray',
                    cbar=False,
                    xticklabels=x_ticklabels,
                    yticklabels=y_ticklabels,
                    annot_kws={"size": 12, "weight": "bold"} if annotate else None,
                )
                
                if show_xlabel:
                    ax.tick_params(axis='x', labelsize=14)
                if show_ylabel:
                    ax.tick_params(axis='y', labelsize=14)
            else:
                # Fallback to matplotlib
                im = ax.imshow(
                    heatmap_data.values,
                    cmap=cmap,
                    aspect='auto',
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='nearest'
                )
                
                ax.set_xticks(np.arange(len(heatmap_data.columns)))
                ax.set_yticks(np.arange(len(heatmap_data.index)))
                if show_xlabel:
                    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=14)
                else:
                    ax.set_xticklabels([], fontsize=14)
                if show_ylabel:
                    ax.set_yticklabels(heatmap_data.index, fontsize=14)
                else:
                    ax.set_yticklabels([], fontsize=14)
                
                if annotate:
                    for i in range(len(heatmap_data.index)):
                        for j in range(len(heatmap_data.columns)):
                            value = heatmap_data.iloc[i, j]
                            if not pd.isna(value):
                                text_color = "white" if abs(value) > (abs(vmax - vmin) * 0.5) else "black"
                                ax.text(j, i, f"{value:{fmt}}",
                                       ha="center", va="center",
                                       color=text_color, fontweight='bold', fontsize=12)
    
    # Add row labels (steering methods) on the left - rotated 90 degrees (vertical)
    for row_idx, method in enumerate(steering_methods):
        axes[row_idx, 0].set_ylabel(method, fontsize=12, fontweight='bold', rotation=90, ha='center', va='center')
    
    # Add column labels (probe types) on the top
    for col_idx, probe_type in enumerate(probe_types):
        axes[0, col_idx].set_title(f"{probe_type}", fontsize=12, fontweight='bold', pad=15)
    
    # Adjust spacing
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.15, right=0.90, top=0.95, bottom=0.1)
    
    # Add shared colorbar if requested
    if show_colorbar:
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("SPI Difference", rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set overall title
    if title is None:
        title = f"SPI Difference Heatmap Matrix (token_position: {token_position})\nMixture - Same Dataset"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Save if path provided
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_obj.suffix:
            save_path = str(save_path_obj) + ".png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Difference heatmap matrix saved to: {save_path}")
    
    # Only show if requested
    if show:
        plt.show()
    
    return fig


def plot_spi_heatmap_on_axis(
    ax,
    df: pd.DataFrame,
    method: str,
    probe_type: str,
    token_position: str,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
    fmt: str = ".2f",
    title: Optional[str] = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_title: bool = True,
    value_column: Optional[str] = None,
):
    """
    Create a heatmap of SPI values on a given axis for models (rows) vs datasets (columns).
    Designed for use in subplot figures. Supports both regular SPI and SPI difference dataframes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    df : pd.DataFrame
        DataFrame from get_spi_dataframe() or get_spi_difference_same_vs_mixture() with columns: 
        Model, Dataset, Probe Type, Token Position, Method, SPI (or SPI Difference)
    method : str
        Steering method to filter by (e.g., "MERA", "MERA logistic")
    probe_type : str
        Probe type to filter by: "linear" or "logit"
    token_position : str
        Token position to filter by: "exact" or "last"
    cmap : str
        Colormap name (default: "RdYlGn" for red-yellow-green)
    vmin : float, optional
        Minimum value for color scale. If None, auto-determined from data or defaults to -1.0.
    vmax : float, optional
        Maximum value for color scale. If None, auto-determined from data or defaults to 1.0.
    annotate : bool
        Whether to annotate cells with SPI values (default: True)
    fmt : str
        Format string for annotations (default: ".2f")
    title : str, optional
        Custom title for the heatmap. If None, auto-generated.
    show_xlabel : bool
        Whether to show x-axis labels (datasets) (default: True)
    show_ylabel : bool
        Whether to show y-axis labels (models) (default: True)
    show_title : bool
        Whether to show subplot title (default: True)
    value_column : str, optional
        Column name to use for values. If None, auto-detects "SPI Difference" or "SPI".
    
    Returns:
    --------
    matplotlib.axes.Axes
        The axis object
    """
    # Filter the dataframe
    filtered_df = df[
        (df["Method"] == method) &
        (df["Probe Type"] == probe_type) &
        (df["Token Position"] == token_position)
    ].copy()
    
    if filtered_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                transform=ax.transAxes, fontsize=12)
        if show_title:
            ax.set_title(title or f"{method}\n({probe_type})", fontsize=10)
        return ax
    
    # Pivot to create matrix: models (rows) x datasets (columns)
    heatmap_data = filtered_df.pivot_table(
        index="Model",
        columns="Dataset",
        values="SPI",
        aggfunc="mean"  # In case there are duplicates
    )
    
    # Sort models and datasets for consistent ordering
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)
    
    if HAS_SEABORN:
        # Use seaborn for nicer heatmap
        # Set tick labels based on show flags
        x_ticklabels = heatmap_data.columns if show_xlabel else False
        y_ticklabels = heatmap_data.index if show_ylabel else False
        
        sns.heatmap(
            heatmap_data,
            annot=annotate,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            cbar_kws={"label": "SPI"},
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
            cbar=False,  # Don't show colorbar on individual subplots
            xticklabels=x_ticklabels,
            yticklabels=y_ticklabels,
            annot_kws={"size": 12, "weight": "bold"} if annotate else None,  # Larger annotation font
        )
        
        # Set larger font sizes for axis labels after creating heatmap
        if show_xlabel:
            ax.tick_params(axis='x', labelsize=14)
        if show_ylabel:
            ax.tick_params(axis='y', labelsize=14)
    else:
        # Fallback to matplotlib
        im = ax.imshow(
            heatmap_data.values,
            cmap=cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        if show_xlabel:
            ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=14)
        else:
            ax.set_xticklabels([], fontsize=14)
        if show_ylabel:
            ax.set_yticklabels(heatmap_data.index, fontsize=14)
        else:
            ax.set_yticklabels([], fontsize=14)
        
        # Annotate cells if requested
        if annotate:
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    value = heatmap_data.iloc[i, j]
                    if not pd.isna(value):
                        text_color = "white" if abs(value) > 0.5 else "black"
                        ax.text(j, i, f"{value:{fmt}}",
                               ha="center", va="center",
                               color=text_color, fontweight='bold', fontsize=12)
    
    # Set title only if requested
    if show_title:
        if title is None:
            title = f"{method}\n({probe_type})"
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    
    # Hide x and y labels if not requested
    if not show_xlabel:
        ax.set_xlabel("")
    if not show_ylabel:
        ax.set_ylabel("")
    
    return ax


def plot_spi_heatmap(
    df: pd.DataFrame,
    method: str,
    probe_type: str,
    token_position: str,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
    fmt: str = ".2f",
    title: Optional[str] = None,
    show: bool = True,
):
    """
    Create a heatmap of SPI values for models (rows) vs datasets (columns).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from get_spi_dataframe() with columns: Model, Dataset, Probe Type, 
        Token Position, Method, SPI
    method : str
        Steering method to filter by (e.g., "MERA", "MERA logistic")
    probe_type : str
        Probe type to filter by: "linear" or "logit"
    token_position : str
        Token position to filter by: "exact" or "last"
    save_path : str, optional
        Path to save the heatmap. If None, plot is only displayed.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-sized based on data.
    cmap : str
        Colormap name (default: "RdYlGn" for red-yellow-green)
    vmin : float
        Minimum value for color scale (default: -1.0). SPI values range from -1 to 1.
    vmax : float
        Maximum value for color scale (default: 1.0). SPI values range from -1 to 1.
    annotate : bool
        Whether to annotate cells with SPI values (default: True)
    fmt : str
        Format string for annotations (default: ".2f")
    title : str, optional
        Custom title for the heatmap. If None, auto-generated.
    show : bool
        Whether to display the plot using plt.show() (default: True).
        Set to False if you want to manually control display (e.g., in notebooks).
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Filter the dataframe
    filtered_df = df[
        (df["Method"] == method) &
        (df["Probe Type"] == probe_type) &
        (df["Token Position"] == token_position)
    ].copy()
    
    if filtered_df.empty:
        raise ValueError(
            f"No data found for method='{method}', probe_type='{probe_type}', "
            f"token_position='{token_position}'. "
            f"Available combinations:\n"
            f"  Methods: {df['Method'].unique()}\n"
            f"  Probe Types: {df['Probe Type'].unique()}\n"
            f"  Token Positions: {df['Token Position'].unique()}"
        )
    
    # Pivot to create matrix: models (rows) x datasets (columns)
    heatmap_data = filtered_df.pivot_table(
        index="Model",
        columns="Dataset",
        values="SPI",
        aggfunc="mean"  # In case there are duplicates
    )
    
    # Sort models and datasets for consistent ordering
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)
    
    # Auto-size figure if not provided
    if figsize is None:
        n_models = len(heatmap_data.index)
        n_datasets = len(heatmap_data.columns)
        figsize = (max(8, n_datasets * 1.2), max(6, n_models * 0.8))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if HAS_SEABORN:
        # Use seaborn for nicer heatmap
        # Fixed color scale from -1 to 1 for SPI values
        sns.heatmap(
            heatmap_data,
            annot=annotate,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,  # Fixed at -1.0
            vmax=vmax,  # Fixed at 1.0
            center=0,  # Center colormap at 0
            cbar_kws={"label": "SPI"},
            ax=ax,
            linewidths=0.5,
            linecolor='gray'
        )
    else:
        # Fallback to matplotlib
        # Fixed color scale from -1 to 1 for SPI values
        im = ax.imshow(
            heatmap_data.values,
            cmap=cmap,
            aspect='auto',
            vmin=vmin,  # Fixed at -1.0
            vmax=vmax,  # Fixed at 1.0
            interpolation='nearest'
        )
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
        ax.set_yticklabels(heatmap_data.index)
        
        # Add colorbar with fixed range
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("SPI", rotation=270, labelpad=20)
        
        # Annotate cells if requested
        if annotate:
            for i in range(len(heatmap_data.index)):
                for j in range(len(heatmap_data.columns)):
                    value = heatmap_data.iloc[i, j]
                    if not pd.isna(value):
                        # Choose text color based on value for better visibility
                        # Use white text for values far from 0, black for values near 0
                        text_color = "white" if abs(value) > 0.5 else "black"
                        text = ax.text(j, i, f"{value:{fmt}}",
                                     ha="center", va="center",
                                     color=text_color, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel("Dataset", fontsize=12, fontweight='bold')
    ax.set_ylabel("Model", fontsize=12, fontweight='bold')
    
    if title is None:
        title = f"SPI Heatmap: {method} ({probe_type}, {token_position})"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_obj.suffix:
            save_path = str(save_path_obj) + ".png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    # Only show if requested (prevents double display in notebooks)
    if show:
        plt.show()
    
    return fig


def plot_spi_heatmap_matrix(
    df: pd.DataFrame,
    steering_methods: List[str],
    probe_types: List[str],
    token_position: str,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
    fmt: str = ".2f",
    show_colorbar: bool = True,
    title: Optional[str] = None,
    show: bool = True,
):
    """
    Create a matrix of heatmaps with steering methods as rows and probe types as columns.
    Each subplot shows models vs datasets for that specific combination.
    Supports both regular SPI dataframes and SPI difference dataframes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame from get_spi_dataframe() or get_spi_difference_same_vs_mixture() with columns: 
        Model, Dataset, Probe Type, Token Position, Method, SPI (or SPI Difference)
    steering_methods : list of str
        List of steering methods to plot (rows of the matrix)
        e.g., ["MERA", "MERA logistic", "MERA contrastive"]
    probe_types : list of str
        List of probe types to plot (columns of the matrix)
        e.g., ["linear", "logit"]
    token_position : str
        Token position to use: "exact" or "last"
    save_path : str, optional
        Path to save the heatmap matrix. If None, plot is only displayed.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-sized based on grid.
    cmap : str
        Colormap name (default: "RdYlGn" for red-yellow-green)
    vmin : float, optional
        Minimum value for color scale. If None, auto-determined from data or defaults to -1.0.
    vmax : float, optional
        Maximum value for color scale. If None, auto-determined from data or defaults to 1.0.
    annotate : bool
        Whether to annotate cells with SPI values (default: True)
    fmt : str
        Format string for annotations (default: ".2f")
    show_colorbar : bool
        Whether to show a shared colorbar for all subplots (default: True)
    title : str, optional
        Overall title for the figure. If None, auto-generated.
    show : bool
        Whether to display the plot using plt.show() (default: True)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    n_rows = len(steering_methods)
    n_cols = len(probe_types)
    
    # Auto-size figure if not provided
    if figsize is None:
        # Estimate based on number of models and datasets
        # Get sample data to estimate dimensions
        sample_df = df[
            (df["Method"].isin(steering_methods)) &
            (df["Probe Type"].isin(probe_types)) &
            (df["Token Position"] == token_position)
        ]
        if not sample_df.empty:
            n_models = len(sample_df["Model"].unique())
            n_datasets = len(sample_df["Dataset"].unique())
            # Base size per subplot, adjusted for grid
            width = max(12, n_cols * max(6, n_datasets * 1.0))
            height = max(8, n_rows * max(5, n_models * 0.6))
            figsize = (width, height)
        else:
            figsize = (n_cols * 6, n_rows * 5)
    
    # Auto-determine vmin/vmax if not provided
    if vmin is None or vmax is None:
        sample_df = df[
            (df["Method"].isin(steering_methods)) &
            (df["Probe Type"].isin(probe_types)) &
            (df["Token Position"] == token_position)
        ]
        # Determine which column to use
        value_col = "SPI Difference" if "SPI Difference" in sample_df.columns else "SPI"
        if not sample_df.empty and value_col in sample_df.columns:
            if vmin is None:
                vmin = sample_df[value_col].min()
            if vmax is None:
                vmax = sample_df[value_col].max()
        else:
            vmin = vmin or -1.0
            vmax = vmax or 1.0
    
    # Create figure with subplots
    # Use gridspec or adjust spacing to remove gaps between columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Plot each combination
    for row_idx, method in enumerate(steering_methods):
        for col_idx, probe_type in enumerate(probe_types):
            ax = axes[row_idx, col_idx]
            
            # Determine which labels to show
            # Show x-axis labels (datasets) only on bottom row
            show_xlabel = (row_idx == n_rows - 1)
            # Show y-axis labels (models) only on leftmost column
            show_ylabel = (col_idx == 0)
            # Don't show individual subplot titles
            show_title = False
            
            # Plot heatmap on this axis
            # Auto-detect if this is a difference dataframe
            value_col = None
            if "SPI Difference" in df.columns:
                value_col = "SPI Difference"
            
            plot_spi_heatmap_on_axis(
                ax=ax,
                df=df,
                method=method,
                probe_type=probe_type,
                token_position=token_position,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                annotate=annotate,
                fmt=fmt,
                title=None,
                show_xlabel=show_xlabel,
                show_ylabel=show_ylabel,
                show_title=show_title,
                value_column=value_col,
            )
    
    # Add row labels (steering methods) on the left - rotated 90 degrees (vertical)
    for row_idx, method in enumerate(steering_methods):
        axes[row_idx, 0].set_ylabel(method, fontsize=12, fontweight='bold', rotation=90, ha='center', va='center')
    
    # Add column labels (probe types) on the top
    for col_idx, probe_type in enumerate(probe_types):
        axes[0, col_idx].set_title(f"{probe_type}", fontsize=12, fontweight='bold', pad=15)
    
    # Adjust spacing FIRST to remove gaps between subplots
    # wspace controls horizontal spacing (smaller = less space between columns)
    # hspace controls vertical spacing (smaller = less space between rows)
    # Reduce right margin to allow colorbar to be further right
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.15, right=0.90, top=0.95, bottom=0.1)
    
    # Add shared colorbar if requested - position on the far right
    if show_colorbar:
        # Create a mappable for the colorbar
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Position colorbar on the far right side of the entire figure
        # Use fig.add_axes to place it outside the subplot grid
        # [left, bottom, width, height] in figure coordinates (0-1)
        # Move it more to the right (increase left value to 0.95 or 0.96)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])  # Further to the right
        cbar = fig.colorbar(sm, cax=cbar_ax)
        # Determine label based on whether we're plotting differences
        label = "SPI Difference" if "SPI Difference" in df.columns else "SPI"
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set overall title - position it higher
    if title is None:
        if "SPI Difference" in df.columns:
            title = f"SPI Difference Heatmap Matrix (token_position: {token_position})\nMixture - Same Dataset"
        else:
            title = f"SPI Heatmap Matrix (token_position: {token_position})"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Save if path provided
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not save_path_obj.suffix:
            save_path = str(save_path_obj) + ".png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap matrix saved to: {save_path}")
    
    # Only show if requested
    if show:
        plt.show()
    
    return fig

