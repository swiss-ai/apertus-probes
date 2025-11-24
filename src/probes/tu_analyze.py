import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import os
BASE_DIR = "/capstor/store/cscs/swissai/infra01/apertus_probes"
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
def get_paths_for_dataset(dataset_name: str, variant: str = "leave"):
    paths = {}
    if dataset_name == "mmlu_high_school":
        paths["Apertus-8B-Instruct-2509"] = os.path.join(
            BASE_DIR, "mmlu_high_school", "Apertus-8B-Instruct-2509", "df_probes_hs.pkl"
        )
        llama_fname = f"df_probes_{variant}.pkl"
        paths["Llama-3.1-8B-Instruct"] = os.path.join(
            BASE_DIR, "mmlu_high_school", "Llama-3.1-8B-Instruct", llama_fname
        )
    elif dataset_name == "mmlu_professional":
        paths["Apertus-8B-Instruct-2509"] = os.path.join(
            BASE_DIR, "mmlu_professional", "Apertus-8B-Instruct-2509", "df_probes_pro.pkl"
        )
        llama_fname = f"df_probes_{variant}.pkl"
        paths["Llama-3.1-8B-Instruct"] = os.path.join(
            BASE_DIR, "mmlu_professional", "Llama-3.1-8B-Instruct", llama_fname
        )
    elif dataset_name == "ARC-Challenge":
        aper_fname = f"df_probes_{variant}.pkl"
        llama_fname = f"df_probes_{variant}.pkl"
        paths["Apertus-8B-Instruct-2509"] = os.path.join(
            BASE_DIR, "ARC-Challenge", "Apertus-8B-Instruct-2509", aper_fname
        )
        paths["Llama-3.1-8B-Instruct"] = os.path.join(
            BASE_DIR, "ARC-Challenge", "Llama-3.1-8B-Instruct", llama_fname
        )
    elif dataset_name == "ARC-Easy":
        aper_fname = f"df_probes_{variant}.pkl"
        llama_fname = f"df_probes_{variant}.pkl"
        paths["Apertus-8B-Instruct-2509"] = os.path.join(
            BASE_DIR, "ARC-Easy", "Apertus-8B-Instruct-2509", aper_fname
        )
        paths["Llama-3.1-8B-Instruct"] = os.path.join(
            BASE_DIR, "ARC-Easy", "Llama-3.1-8B-Instruct", llama_fname
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
    return paths
def plot_rmse_comparison_multi(
    results_by_llm: dict,
    models_to_plot,
    token_positions,
    title_suffix: str = ""
):
    """
    results_by_llm: {'Apertus-8B-Instruct-2509': df, 'Llama-3.1-8B-Instruct': df}
    """
    plt.figure(figsize=(14, 6))
    color_map = {
        "L-0":   "tab:blue",
        "L-0.25": "tab:green",
        "L-0.5": "tab:purple",
    }
    # Apertus: solid, Llama: dashed
    llm_style = {
        "Apertus-8B-Instruct-2509": "-",
        "Llama-3.1-8B-Instruct": "--",
    }
    legend_dict = {
        "Apertus-8B-Instruct-2509": "Apertus",
        "Llama-3.1-8B-Instruct": "Llama",
    }
    for llm_name, df in results_by_llm.items():
        for model in models_to_plot:
            for token_pos in token_positions:
                mask = (df["Model"] == model) & (df["Token-Pos"] == token_pos)
                group = df[mask]
                if group.empty:
                    continue
                layer_averages = group.groupby("Layer")[["RMSE", "Dummy-RMSE"]].mean()
                base_color = color_map.get(model, "black")
                linestyle = llm_style.get(llm_name, "-")
                # RMSE (bold lines)
                plt.plot(
                    layer_averages.index,
                    layer_averages["RMSE"],
                    color=base_color,
                    linestyle=linestyle,
                    linewidth=2.5,
                    alpha=1.0,
                )
                # Dummy-RMSE (light lines)
                plt.plot(
                    layer_averages.index,
                    layer_averages["Dummy-RMSE"],
                    color=base_color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.4,
                )
    plt.title(f"{title_suffix}", fontsize=16)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.ylim(0, 10)
    plt.grid(True, linestyle="--", alpha=0.7)
    # RMSE vs Dummy
    legend_rmse = [
        Line2D([0], [0], color="black", linewidth=3, alpha=1.0),
        Line2D([0], [0], color="black", linewidth=3, alpha=0.3),
    ]
    labels_rmse = ["RMSE (bold)", "Dummy-RMSE (light)"]
    # Apertus vs Llama
    legend_model = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=3),
        Line2D([0], [0], color="black", linestyle="--", linewidth=3),
    ]
    labels_model = ["Apertus (solid)", "Llama (dashed)"]
    # Colours
    legend_reg = [
        Line2D([0], [0], color="tab:blue", linewidth=3),
        Line2D([0], [0], color="tab:green", linewidth=3),
        Line2D([0], [0], color="tab:purple", linewidth=3),
    ]
    labels_reg = ["L-0", "L-0.25", "L-0.5"]
    ax = plt.gca()
    leg1 = ax.legend(
        legend_rmse,
        labels_rmse,
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        title="Error type",
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        legend_model,
        labels_model,
        fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.6),
        borderaxespad=0.0,
        title="Model",
    )
    ax.add_artist(leg2)
    ax.legend(
        legend_reg,
        labels_reg,
        fontsize=10,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.2),
        borderaxespad=0.0,
        title="Regularization",
    )
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
dataset_name = "mmlu_professional"   # or mmlu_professional / ARC-Challenge / ARC-Easy
dataset_in_title = "MMLU Professional"
variant = "leave"                   # or "transform"
paths = get_paths_for_dataset(dataset_name, variant=variant)
results_by_llm = {model: load_pkl(p) for model, p in paths.items()}
models_to_plot = ["L-0", "L-0.25", "L-0.5"]
# exact position
plot_rmse_comparison_multi(
    results_by_llm,
    models_to_plot=models_to_plot,
    token_positions=["exact"],
    title_suffix=f"RMSE by layers with {dataset_in_title} dataset and exact token position",
)
# last token
plot_rmse_comparison_multi(
    results_by_llm,
    models_to_plot=models_to_plot,
    token_positions=["last"],
    title_suffix=f"RMSE by layers with {dataset_in_title} dataset and last token position",
)