"""
Plotting utilities for probe analysis.
Contains functions for plotting RMSE and accuracy comparisons across layers.
"""

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import colorsys
import numpy as np
from typing import Dict, List, Optional
import pandas as pd


def _adjust_lightness(color, factor=1.0):
    """factor > 1 -> lighter, < 1 -> darker."""
    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0, min(1, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


def _get_color_for_model(model_name: str) -> str:
    """
    Get a color for a model name. Uses predefined colors for known models,
    or generates a deterministic color for unknown models.
    """
    predefined_colors = {
        "Apertus-8B-Instruct-2509": "tab:orange",
        "Llama-3.1-8B-Instruct": "tab:blue",
        "Apertus-8B-2509": "tab:red",
        "Llama-3.1-8B": "tab:purple",
    }
    
    if model_name in predefined_colors:
        return predefined_colors[model_name]
    
    # Generate a deterministic color for unknown models using hash
    # Use a palette of distinct colors
    color_palette = [
        "tab:green",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "tab:orange",
        "tab:blue",
        "tab:red",
        "tab:purple",
    ]
    
    # Use hash of model name to pick a color deterministically
    hash_value = hash(model_name)
    color_index = abs(hash_value) % len(color_palette)
    return color_palette[color_index]


def _get_plot_config():
    """Shared configuration for plot colors and labels."""
    return {
        "llm_base_color": {
            "Apertus-8B-Instruct-2509": "tab:orange",
            "Llama-3.1-8B-Instruct": "tab:blue",
            "Apertus-8B-2509": "tab:red",
            "Llama-3.1-8B": "tab:purple",
        },
        "llm_label": {
            "Apertus-8B-Instruct-2509": "Apertus-Instruct",
            "Llama-3.1-8B-Instruct": "Llama-Instruct",
        },
        "probe_lightness": {
            "L-0.05": 0.7,  # darker
            "L-0.1": 0.9,   # slightly darker
            "L-0.25": 1.2,  # slightly lighter
        },
    }


def _plot_mse_lines_on_axis(
    ax,
    results_by_llm: dict,
    models_to_plot,
    token_positions,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
):
    """Internal helper to plot RMSE lines on a given axis."""
    config = _get_plot_config()
    llm_base_color = config["llm_base_color"]
    probe_lightness = config["probe_lightness"]
    
    for llm_name, df in results_by_llm.items():
        base = _get_color_for_model(llm_name)

        for model in models_to_plot:
            light_factor = probe_lightness.get(model, 1.0)
            color = _adjust_lightness(base, light_factor)

            for token_pos in token_positions:
                mask = (df["Model"] == model) & (df["Token-Pos"] == token_pos)
                group = df[mask]
                if group.empty:
                    continue

                # Always compute standard_error from y_pred and y_test if available
                # This replaces std across attempts with standard error of residuals
                if "y_pred" in group.columns and "y_test" in group.columns:
                    # Compute standard_error if not already present
                    if "standard_error" not in group.columns:
                        group = compute_residual_statistics(group)
                    
                    # Use standard_error from residuals (computed from y_test - y_pred)
                    layer_stats = group.groupby("Layer")[["RMSE", "Dummy-RMSE", "standard_error"]].agg({
                        "RMSE": "mean",
                        "Dummy-RMSE": "mean",
                        "standard_error": "mean"  # Average standard_error across attempts for each layer
                    })
                    layer_averages = layer_stats[["RMSE", "Dummy-RMSE"]]
                    # Use standard_error for error bars (instead of std across attempts)
                    se_values = layer_stats["standard_error"]
                    layer_errors = pd.DataFrame({
                        "RMSE": se_values,
                        "Dummy-RMSE": se_values  # Use same standard_error for dummy
                    }, index=layer_stats.index)
                else:
                    # Fallback: if y_pred/y_test not available, use std across attempts
                    layer_stats = group.groupby("Layer")[["RMSE", "Dummy-RMSE"]].agg(["mean", "std"])
                    layer_averages = layer_stats.xs("mean", axis=1, level=1)
                    layer_errors = layer_stats.xs("std", axis=1, level=1)
                    layer_errors = layer_errors.fillna(0)
                
                # Filter layers by start_layer and end_layer if specified
                if start_layer is not None:
                    layer_averages = layer_averages[layer_averages.index >= start_layer]
                    layer_errors = layer_errors[layer_errors.index >= start_layer]
                if end_layer is not None:
                    layer_averages = layer_averages[layer_averages.index <= end_layer]
                    layer_errors = layer_errors[layer_errors.index <= end_layer]
                
                if layer_averages.empty:
                    continue

                # Sort by layer index to ensure correct plotting order
                layer_averages = layer_averages.sort_index()
                layer_errors = layer_errors.sort_index()

                # RMSE: solid line with shaded error region
                ax.plot(
                    layer_averages.index,
                    layer_averages["RMSE"],
                    color=color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=1.0,
                )
                # Add shaded region for standard error (from residuals or std across attempts)
                if (layer_errors["RMSE"] > 0).any():
                    ax.fill_between(
                        layer_averages.index,
                        layer_averages["RMSE"] - layer_errors["RMSE"],
                        layer_averages["RMSE"] + layer_errors["RMSE"],
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                    )
                # Dummy-RMSE: dashed line with shaded error region
                ax.plot(
                    layer_averages.index,
                    layer_averages["Dummy-RMSE"],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                )
                # Add shaded region for dummy standard error (lighter, only if error > 0)
                if (layer_errors["Dummy-RMSE"] > 0).any():
                    ax.fill_between(
                        layer_averages.index,
                        layer_averages["Dummy-RMSE"] - layer_errors["Dummy-RMSE"],
                        layer_averages["Dummy-RMSE"] + layer_errors["Dummy-RMSE"],
                        color=color,
                        alpha=0.1,
                        linewidth=0,
                    )


def _create_legend_handles_labels(results_by_llm: dict, models_to_plot):
    """Create legend handles and labels for the plot."""
    config = _get_plot_config()
    llm_base_color = config["llm_base_color"]
    llm_label = config["llm_label"]
    probe_lightness = config["probe_lightness"]
    
    legend_handles = []
    legend_labels = []

    # One solid line per (LLM, probe model) combo
    for llm_name in results_by_llm.keys():
        base = _get_color_for_model(llm_name)
        for model in models_to_plot:
            light_factor = probe_lightness.get(model, 1.0)
            color = _adjust_lightness(base, light_factor)
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle="-", linewidth=2.0)
            )
            legend_labels.append(f"{llm_label.get(llm_name, llm_name)} {model}")

    # One dashed line entry explaining the dummy
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8)
    )
    legend_labels.append("Dummy RMSE")
    
    return legend_handles, legend_labels


def plot_rmse_comparison_multi(
    results_by_llm: dict,
    models_to_plot,
    token_positions,
    title_suffix: str = "",
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None
):
    """
    Plot RMSE comparison across layers for multiple LLMs and probe models.
    
    Args:
        results_by_llm: Dict mapping LLM model names to dataframes
        models_to_plot: List of probe model names to plot (e.g., ["L-0.1", "L-0.25"])
        token_positions: List of token positions to plot (e.g., ["exact", "last"])
        title_suffix: Title for the plot
        start_layer: Optional start layer (inclusive). If None, plots from first layer.
        end_layer: Optional end layer (inclusive). If None, plots to last layer.
    
    Returns:
        matplotlib.axes.Axes: The axis object for further customization
    """
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # Plot the lines
    _plot_mse_lines_on_axis(
        ax, results_by_llm, models_to_plot, token_positions, start_layer, end_layer
    )

    # Create and add legend
    legend_handles, legend_labels = _create_legend_handles_labels(
        results_by_llm, models_to_plot
    )
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        title=None,
        frameon=True,
    )

    # Styling
    plt.title(title_suffix, fontsize=16, pad=14)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
    
    return ax


def plot_rmse_on_axis(
    ax,
    results_by_llm: dict,
    models_to_plot,
    token_positions,
    title_suffix: str = "",
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    add_legend: bool = True,
):
    """
    Plot RMSE comparison on a given axis for multiple LLMs and probe models.
    Designed for use in subplot figures.
    
    Args:
        ax: matplotlib.axes.Axes object to plot on
        results_by_llm: Dict mapping LLM model names to dataframes
        models_to_plot: List of probe model names to plot (e.g., ["L-0.1", "L-0.25"])
        token_positions: List of token positions to plot (e.g., ["exact", "last"])
        title_suffix: Title for the plot
        start_layer: Optional start layer (inclusive). If None, plots from first layer.
        end_layer: Optional end layer (inclusive). If None, plots to last layer.
        add_legend: Whether to add a legend to this axis (default: True)
    
    Returns:
        tuple: (legend_handles, legend_labels) for use in shared legends
    """
    # Plot the lines
    _plot_mse_lines_on_axis(
        ax, results_by_llm, models_to_plot, token_positions, start_layer, end_layer
    )

    # Create legend handles and labels
    legend_handles, legend_labels = _create_legend_handles_labels(
        results_by_llm, models_to_plot
    )

    if add_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            title=None,
            frameon=True,
        )

    # Styling
    ax.set_title(title_suffix, fontsize=16, pad=14)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    return legend_handles, legend_labels


def _plot_accuracy_lines_on_axis(
    ax,
    results_by_llm: dict,
    models_to_plot,
    token_positions,
    accuracy_metric: str = "ACC",
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
):
    """Internal helper to plot accuracy lines on a given axis."""
    config = _get_plot_config()
    llm_base_color = config["llm_base_color"]
    
    for llm_name, df in results_by_llm.items():
        # Filter for classification tasks only (accuracy is only available for classification)
        df_class = df[df["Task"] == "classification"].copy()
        if df_class.empty:
            print(f"⚠️ Warning: No classification data found for {llm_name}")
            continue
            
        base = _get_color_for_model(llm_name)

        for model in models_to_plot:
            # For logistic regression, use base color; for other models, adjust lightness
            if model == "LogReg-l1":
                color = base
            else:
                probe_lightness = config["probe_lightness"]
                light_factor = probe_lightness.get(model, 1.0)
                color = _adjust_lightness(base, light_factor)

            for token_pos in token_positions:
                mask = (df_class["Model"] == model) & (df_class["Token-Pos"] == token_pos)
                group = df_class[mask]
                if group.empty:
                    continue

                # Get accuracy metric (ACC or BACC)
                metric_col = accuracy_metric
                dummy_metric_col = f"Dummy-{accuracy_metric}"
                
                # Check if columns exist
                if metric_col not in group.columns or dummy_metric_col not in group.columns:
                    print(f"⚠️ Warning: {metric_col} or {dummy_metric_col} not found in data")
                    continue
                
                layer_stats = group.groupby("Layer")[[metric_col, dummy_metric_col]].agg(["mean", "std"])
                layer_averages = layer_stats.xs("mean", axis=1, level=1)
                layer_stds = layer_stats.xs("std", axis=1, level=1)
                
                # Fill NaN std values with 0 (occurs when only one sample per layer)
                layer_stds = layer_stds.fillna(0)
                
                # Filter layers by start_layer and end_layer if specified
                if start_layer is not None:
                    layer_averages = layer_averages[layer_averages.index >= start_layer]
                    layer_stds = layer_stds[layer_stds.index >= start_layer]
                if end_layer is not None:
                    layer_averages = layer_averages[layer_averages.index <= end_layer]
                    layer_stds = layer_stds[layer_stds.index <= end_layer]
                
                if layer_averages.empty:
                    continue

                # Sort by layer index to ensure correct plotting order
                layer_averages = layer_averages.sort_index()
                layer_stds = layer_stds.sort_index()

                # Accuracy: solid line with shaded error region
                ax.plot(
                    layer_averages.index,
                    layer_averages[metric_col],
                    color=color,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=1.0,
                )
                # Add shaded region for standard deviation (only if std > 0)
                if (layer_stds[metric_col] > 0).any():
                    ax.fill_between(
                        layer_averages.index,
                        layer_averages[metric_col] - layer_stds[metric_col],
                        layer_averages[metric_col] + layer_stds[metric_col],
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                    )
                # Dummy-Accuracy: dashed line with shaded error region
                ax.plot(
                    layer_averages.index,
                    layer_averages[dummy_metric_col],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                )
                # Add shaded region for dummy standard deviation (lighter, only if std > 0)
                if (layer_stds[dummy_metric_col] > 0).any():
                    ax.fill_between(
                        layer_averages.index,
                        layer_averages[dummy_metric_col] - layer_stds[dummy_metric_col],
                        layer_averages[dummy_metric_col] + layer_stds[dummy_metric_col],
                        color=color,
                        alpha=0.1,
                        linewidth=0,
                    )


def _create_accuracy_legend_handles_labels(results_by_llm: dict, models_to_plot):
    """Create legend handles and labels for the accuracy plot."""
    config = _get_plot_config()
    llm_base_color = config["llm_base_color"]
    llm_label = config["llm_label"]
    probe_lightness = config["probe_lightness"]
    
    legend_handles = []
    legend_labels = []

    # One solid line per (LLM, probe model) combo
    for llm_name in results_by_llm.keys():
        base = _get_color_for_model(llm_name)
        for model in models_to_plot:
            if model == "LogReg-l1":
                color = base
            else:
                light_factor = probe_lightness.get(model, 1.0)
                color = _adjust_lightness(base, light_factor)
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle="-", linewidth=2.0)
            )
            legend_labels.append(f"{llm_label.get(llm_name, llm_name)} {model}")

    # One dashed line entry explaining the dummy
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8)
    )
    legend_labels.append("Dummy Accuracy")
    
    return legend_handles, legend_labels


def plot_accuracy_comparison_multi(
    results_by_llm: dict,
    models_to_plot,
    token_positions,
    title_suffix: str = "",
    accuracy_metric: str = "ACC",
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None
):
    """
    Plot accuracy comparison across layers for multiple LLMs and probe models.
    
    Args:
        results_by_llm: Dict mapping LLM model names to dataframes
        models_to_plot: List of probe model names to plot (e.g., ["LogReg-l1"])
        token_positions: List of token positions to plot (e.g., ["exact", "last"])
        title_suffix: Title for the plot
        accuracy_metric: Either "ACC" (accuracy) or "BACC" (balanced accuracy). Default: "ACC"
        start_layer: Optional start layer (inclusive). If None, plots from first layer.
        end_layer: Optional end layer (inclusive). If None, plots to last layer.
    
    Returns:
        matplotlib.axes.Axes: The axis object for further customization
    """
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # Plot the lines
    _plot_accuracy_lines_on_axis(
        ax, results_by_llm, models_to_plot, token_positions, accuracy_metric, start_layer, end_layer
    )

    # Create and add legend
    legend_handles, legend_labels = _create_accuracy_legend_handles_labels(
        results_by_llm, models_to_plot
    )
    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        title=None,
        frameon=True,
    )

    # Styling
    metric_label = "Accuracy" if accuracy_metric == "ACC" else "Balanced Accuracy"
    plt.title(title_suffix, fontsize=16, pad=14)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel(metric_label, fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
    
    return ax


def plot_accuracy_on_axis(
    ax,
    results_by_llm: dict,
    models_to_plot,
    token_positions,
    title_suffix: str = "",
    accuracy_metric: str = "ACC",
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    add_legend: bool = True,
):
    """
    Plot accuracy comparison on a given axis for multiple LLMs and probe models.
    Designed for use in subplot figures.
    
    Args:
        ax: matplotlib.axes.Axes object to plot on
        results_by_llm: Dict mapping LLM model names to dataframes
        models_to_plot: List of probe model names to plot (e.g., ["LogReg-l1"])
        token_positions: List of token positions to plot (e.g., ["exact", "last"])
        title_suffix: Title for the plot
        accuracy_metric: Either "ACC" (accuracy) or "BACC" (balanced accuracy). Default: "ACC"
        start_layer: Optional start layer (inclusive). If None, plots from first layer.
        end_layer: Optional end layer (inclusive). If None, plots to last layer.
        add_legend: Whether to add a legend to this axis (default: True)
    
    Returns:
        tuple: (legend_handles, legend_labels) for use in shared legends
    """
    # Plot the lines
    _plot_accuracy_lines_on_axis(
        ax, results_by_llm, models_to_plot, token_positions, accuracy_metric, start_layer, end_layer
    )

    # Create legend handles and labels
    legend_handles, legend_labels = _create_accuracy_legend_handles_labels(
        results_by_llm, models_to_plot
    )

    if add_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            fontsize=10,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            title=None,
            frameon=True,
        )

    # Styling
    metric_label = "Accuracy" if accuracy_metric == "ACC" else "Balanced Accuracy"
    ax.set_title(title_suffix, fontsize=16, pad=14)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    return legend_handles, legend_labels


def compute_residual_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute residuals, standard deviation of residuals, and standard error for all rows.
    
    Args:
        df: DataFrame with 'y_pred' and 'y_test' columns (each containing arrays)
    
    Returns:
        DataFrame with added columns: 'residuals', 'residuals_std', 'standard_error'
    """
    # Ensure y_pred and y_test are numpy arrays
    if 'y_pred' not in df.columns or 'y_test' not in df.columns:
        raise ValueError("DataFrame must contain 'y_pred' and 'y_test' columns")
    
    # Compute residuals: y_test - y_pred
    df = df.copy()
    df['residuals'] = df.apply(
        lambda row: np.array(row['y_test']) - np.array(row['y_pred']), 
        axis=1
    )
    
    # Compute standard deviation of residuals (sample std, ddof=1)
    df['residuals_std'] = df['residuals'].apply(lambda res: np.std(res, ddof=1))
    
    # Compute standard error: std / sqrt(n)
    df['standard_error'] = df.apply(
        lambda row: row['residuals_std'] / np.sqrt(len(row['residuals'])) 
        if len(row['residuals']) > 0 else 0.0,
        axis=1
    )
    
    return df


# Backward compatibility aliases (keep old function names working)
plot_mse_comparison_multi = plot_rmse_comparison_multi
plot_mse_on_axis = plot_rmse_on_axis

