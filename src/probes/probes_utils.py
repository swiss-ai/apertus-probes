import pandas as pd
import numpy as np
from typing import Optional


def postprocess_df_probes(
    df,
    filter_error_type: Optional[str] = "sm",
    filter_probe_token_pos: Optional[str] = "last",
    filter_inputs: Optional[str] = "activations",
) -> pd.DataFrame:
    """Process the probe performance result per LM model and apply filters at the end."""

    # Filter the df.
    if "Match-Type" in df.columns:
        df["Token-Pos"] = df["Match-Type"]
    for col, val, allowed in [
        ("Error-Type", filter_error_type, {"sm", "cm"}),
        ("Token-Pos", filter_probe_token_pos, {"last", "exact"}),
        ("Inputs", filter_inputs, {"activations", "encodings"}),
    ]:
        if val is not None:
            df[col] = df[col].str.lower()
            val = val.lower()
            assert val in allowed and val in df[col].unique()
            df = df[df[col] == val]

    # Dtypes and names.
    df["Layer"] = df["Layer"].astype(int)

    # Get into arrays.
    df["Residuals"] = df["Residuals"].apply(lambda x: np.array(x))
    df["Coefficients"] = df["Coefficients"].apply(lambda x: np.array(x))
    df["Nonzero-Features"] = df["Nonzero-Features"].apply(lambda x: np.array(x))
    df["Nonzero-Features-Count"] = df["Nonzero-Features"].apply(lambda x: len(x))
    df["y_pred"] = df["y_pred"].apply(lambda x: np.array(x))
    df["y_test"] = df["y_test"].apply(lambda x: np.array(x))

    # Add comparison columns.
    for metric in ["MSE", "RMSE"]:
        df[f"{metric}-Better-Than-Dummy"] = (
            df[f"Dummy-{metric}"] > df[f"{metric}"]
        )  # lower is better
        df[f"{metric}-Delta-Dummy"] = df[f"Dummy-{metric}"] - df[f"{metric}"]

    for metric in ["AUCROC", "Accuracy"]:
        df[f"{metric}-Better-Than-Dummy"] = (
            df[f"Dummy-{metric}"] < df[f"{metric}"]
        )  # higher is better
        df[f"{metric}-Delta-Dummy"] = df[f"Dummy-{metric}"] - df[f"{metric}"]

    # Sort!
    df.sort_values(["Layer", "Inputs", "Model", "Token-Pos"], inplace=True)

    return df


def get_best_layer(df, task: str, metric: str, mode: str = "best") -> int | None:
    df = df[df["Task"] == task].copy()
    if len(df) == 0:
        return None

    # direction: regression metrics (RMSE) lower better, classification (AUCROC) higher better
    lower_is_better = (task == "regression")

    # pick best row per layer first (best alpha/model within that layer)
    df_layer_best = (
        df.sort_values(["Layer", metric], ascending=[True, lower_is_better])
          .groupby("Layer", sort=False)
          .head(1)
    )
    if len(df_layer_best) == 0:
        return None

    # now choose layer across layers
    if mode == "best":
        row = df_layer_best.sort_values(metric, ascending=lower_is_better).iloc[0]
    elif mode == "worst":
        row = df_layer_best.sort_values(metric, ascending=not lower_is_better).iloc[0]
    elif mode == "median":
        df_sorted = df_layer_best.sort_values(metric, ascending=lower_is_better)
        row = df_sorted.iloc[len(df_sorted) // 2]
    else:
        raise ValueError(mode)

    return int(row["Layer"])


def get_best_coefficients(
    df,
    dataset_name: Optional[str] = None,
    task: str = "regression",
    metric: str = "RMSE",
    nr_rows: int = 1,
    get_values: bool = True,
    mode: str = "best",  # "best", "worst", or "median"
):
    """Finds best, worst, or median coefficients and intercepts based on the given metric.
    
    Returns:
        If get_values=True: tuple of (coefficients_array, intercepts_array)
        If get_values=False: DataFrame with selected rows
    """
    # CRITICAL: Filter by task first to ensure we only select models of the correct task type
    df = df[df["Task"] == task].copy()
    
    ascending_order = True if task == "regression" else False
    sorted_df = df.sort_values(
        by=["Dataset", "Layer", metric],
        ascending=[True, True, ascending_order],
    )
    grouped = sorted_df.groupby(["Dataset", "Layer"], sort=False)
    
    # Select rows first (keeping all columns including Dataset and Layer)
    if mode == "worst":
        df_selected = grouped.tail(nr_rows)
    elif mode == "best":
        df_selected = grouped.head(nr_rows)
    elif mode == "median":
        selected_rows = []
        for (dataset, layer), group in grouped:
            mid_start = max(0, (len(group) - nr_rows) // 2)
            mid_end = (len(group) + nr_rows) // 2
            selected_rows.append(group.iloc[mid_start:mid_end])
        df_selected = pd.concat(selected_rows, ignore_index=True)
    else:
        raise ValueError(f"Unknown mode={mode}")
    
    # Now keep only what we need (but keep Dataset/Layer!)
    df_selected = df_selected[["Dataset", "Layer", "Coefficients", "Intercept", "Model"]]

    if dataset_name is not None:
        df_selected = df_selected.loc[
            (df_selected["Dataset"].str.contains(dataset_name))
        ]
    
    # CRITICAL: Sort by Layer to ensure coefficients/intercepts are in layer order (0, 1, 2, ...)
    # This is essential because steering_run.py uses zip(range(nr_layers), coefficients)
    # which assumes coefficients[i] corresponds to layer i
    df_selected = df_selected.sort_values("Layer")
    
    if get_values:
        coefficients = df_selected.Coefficients.values
        # Extract intercepts, defaulting to 0.0 if None
        intercepts = df_selected.Intercept.values
        intercepts_processed = [
            float(intercept) if intercept is not None and not pd.isna(intercept) else 0.0 
            for intercept in intercepts
        ]
        # Extract model names
        models = df_selected.Model.values
        return coefficients, intercepts_processed, models

    return df_selected
