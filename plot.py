import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster environments
import matplotlib.pyplot as plt
import os

path = os.path.join(os.environ["SCRATCH"], "mera-runs", "sms_spam", "Apertus-8B-Instruct-2509", "df_probes_demo.pkl")
with open(path, "rb") as f:
    df_probes_demo = pickle.load(f)

print("Loaded df_probes_demo:")
print(df_probes_demo)

# Extract classification task rows from df_probes_demo
df_classification = df_probes_demo[df_probes_demo['Task'] == 'classification']

print("Classification task DataFrame:")
print(df_classification.head())
print(f"\nClassification DataFrame shape: {df_classification.shape}")
print(f"\nColumns: {df_classification.columns.tolist()}")

# The columns of interest for classification metrics
classification_cols = ["Accuracy", "AUCROC", "Accuracy (Balanced)", 
                       "Dummy-Accuracy", "Dummy-AUCROC", "Dummy-Accuracy (Balanced)"]

# Find which columns exist in the dataframe
classification_cols_present = [col for col in classification_cols if col in df_classification.columns]
print(f"\nAvailable classification metric columns: {classification_cols_present}")

# We'll group by the columns that uniquely identify a layer
group_cols = []
for col in ["LLM_model", "Dataset", "Task", "Layer"]:
    if col in df_classification.columns:
        group_cols.append(col)

# If 'Layer' or similar is not present, fall back to finding layer column
if not group_cols or "Layer" not in group_cols:
    for col in df_classification.columns:
        if "layer" in col.lower():
            group_cols.append(col)
            break

print(f"\nGrouping by: {group_cols}")

# Average classification metrics by layer
if classification_cols_present:
    avg_classification_df = (
        df_classification[group_cols + classification_cols_present]
        .groupby(group_cols)
        .mean(numeric_only=True)
        .reset_index()
    )
    print("\nAverage Classification metrics by layer:")
    print(avg_classification_df)
    
    # Find which column is the 'layer' column for plotting
    layer_col = None
    for col in avg_classification_df.columns:
        if "layer" in col.lower() or col == "Layer":
            layer_col = col
            break
    
    if layer_col is not None:
        # Convert layer to numeric if it's not already
        if avg_classification_df[layer_col].dtype == 'object':
            # Try to extract numeric layer number
            avg_classification_df['Layer_num'] = avg_classification_df[layer_col].str.extract('(\d+)').astype(float)
            layer_col = 'Layer_num'
        
        # Plot accuracy metrics
        plot_cols = [col for col in ["Accuracy", "Dummy-Accuracy"] if col in classification_cols_present]
        if plot_cols:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Accuracy
            ax1 = axes[0]
            for col in plot_cols:
                ax1.plot(avg_classification_df[layer_col], avg_classification_df[col], 
                        marker='o', label=col, linewidth=2)
            ax1.set_xlabel("Layer", fontsize=12)
            ax1.set_ylabel("Accuracy", fontsize=12)
            ax1.set_title("Accuracy by Layer", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: All available metrics
            ax2 = axes[1]
            for col in classification_cols_present:
                if col not in plot_cols:  # Plot other metrics
                    ax2.plot(avg_classification_df[layer_col], avg_classification_df[col], 
                            marker='o', label=col, linewidth=2, alpha=0.7)
            ax2.set_xlabel("Layer", fontsize=12)
            ax2.set_ylabel("Metric Value", fontsize=12)
            ax2.set_title("Other Classification Metrics by Layer", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot in multiple locations for easy access
            scratch_path = os.path.join(os.environ["SCRATCH"], "mera-runs", "classification_metrics.png")
            project_path = os.path.join(os.path.dirname(__file__), "classification_metrics.png")
            
            plt.savefig(scratch_path, dpi=150)
            plt.savefig(project_path, dpi=150)
            plt.close()  # Close to free memory
            
            print(f"\nPlot saved to:")
            print(f"  - {scratch_path}")
            print(f"  - {project_path}")
            print(f"\nTo view in VS Code:")
            print(f"  1. Open the file explorer (Ctrl+Shift+E)")
            print(f"  2. Navigate to: classification_metrics.png (in this project folder)")
            print(f"  3. Click on the file - VS Code will display it in the editor")
            print(f"  Or use Ctrl+P and type: classification_metrics.png")
        else:
            print("No Accuracy columns found for plotting.")
    else:
        print("Could not find a layer column to plot against.")
else:
    print("No classification metric columns found in the dataframe.")


