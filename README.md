# apertus-probes

A project for training and evaluating linear probes on language model activations to predict model errors (hallucinations). This codebase supports multiple datasets, models, and probe configurations, with capabilities for mixed-dataset training, cross-dataset evaluation, and activation steering.

## Project Overview

This project implements a pipeline for:
1. **Extracting and caching activations** from language models on various tasks
2. **Training linear probes** (Lasso/Linear Regression/Logistic Regression) on these activations to predict model errors
3. **Evaluating probe performance** across layers, token positions, and datasets
4. **Analyzing results** with comprehensive visualization tools
5. **Steering model behavior** using trained probes (optional)

The probes learn to predict error metrics (like softmax error or cross-entropy) from hidden layer activations, enabling analysis of where and how models encode uncertainty and potential errors.

## Supported Models

The project supports the following model aliases (defined in `submit.sh`):

- `apertus-instruct` → `swiss-ai/Apertus-8B-Instruct-2509`
- `apertus-base` → `swiss-ai/Apertus-8B-2509`
- `llama-instruct` → `meta-llama/Llama-3.1-8B-Instruct`
- `llama-base` → `meta-llama/Llama-3.1-8B`

You can also use full model names directly with `--model_name`.

## Supported Datasets

- `sms_spam` - SMS spam classification
- `mmlu_high_school` - MMLU high school level questions
- `mmlu_professional` - MMLU professional level questions
- `ARC-Easy` - ARC Easy science questions
- `ARC-Challenge` - ARC Challenge science questions
- `sujet_finance_yesno_5k` - Finance yes/no questions

## Quick Start

### Prerequisites

1. **Environment Setup**: The project uses a container environment. Copy the sample config:
   ```bash
   cp probes.toml ~/.edf/probes.toml
   ```

2. **Load Compute Node** (if using SLURM):
   ```bash
   srun -A infra01 --environment=$HOME/.edf/probes.toml --pty bash
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Workflow

The typical workflow consists of three main steps:

#### 1. Cache Model Activations

Extract and save activations from a language model on a dataset:

```bash
./submit.sh cache --model apertus-instruct --dataset_name mmlu_professional
```

Or run locally:
```bash
./submit.sh --local cache --model apertus-instruct --dataset_name mmlu_professional
```

This extracts activations for all layers and saves them to `$SCRATCH/mera-runs/`.

#### 2. Postprocess Cached Data

Aggregate and organize the cached activations:

```bash
./submit.sh postprocess --model apertus-instruct --dataset_name mmlu_professional
```

#### 3. Train Probes

Train linear probes on the cached activations. You can train on:
- **Single datasets**
- **Mixed datasets** (multiple datasets combined)
- **Cross-dataset** (train on one, test on another)

**Example: Single dataset**
```bash
./submit.sh run_probes --model apertus-instruct --datasets mmlu_professional
```

**Example: Mixed datasets**
```bash
./submit.sh run_probes --model apertus-instruct --datasets mmlu_professional ARC-Challenge sms_spam --save-name linear_intercept --alphas 0.02 0.05
```

**Example: Cross-dataset probes**
```bash
./submit.sh cross_dataset_probes --model apertus-instruct --train-dataset mmlu_professional --test-dataset ARC-Challenge
```

### Analysis and Visualization

Use the Jupyter notebooks in `src/probes/` to analyze results:

- `analyze.ipynb` - Main analysis notebook with plotting utilities
- `plot_utils.py` - Comprehensive plotting functions for RMSE and accuracy comparisons

The notebooks support:
- Comparing probe performance across layers
- Evaluating different probe models (Lasso with various alpha values)
- Analyzing exact vs. last token positions
- Visualizing mixed dataset results
- Cross-dataset generalization analysis

## Main Scripts

### `submit.sh` - Main Submission Script

The primary interface for running experiments. It handles:
- Model and dataset validation
- SLURM job submission (or local execution)
- Automatic job naming
- Parameter forwarding

**Usage:**
```bash
./submit.sh <task> --model <alias> --dataset_name <dataset> [options]
```

**Tasks:**
- `cache` - Extract and cache model activations
- `postprocess` - Postprocess cached activations
- `run_probes` - Train probes (single or mixed datasets)
- `cross_dataset_probes` - Train cross-dataset probes

**Common Options:**
- `--local` - Run locally instead of submitting to SLURM
- `--time <HH:MM:SS>` - Time limit for SLURM jobs
- `--gpus <N>` - Number of GPUs
- `--list-models` - List available model aliases
- `--list-datasets` - List available datasets
- `--show-defaults <task>` - Show default parameters for a task

**Examples:**
```bash
# List available options
./submit.sh --list-models
./submit.sh --list-datasets
./submit.sh --show-defaults run_probes

# Run experiments
./submit.sh cache --model apertus-instruct --dataset_name mmlu_professional
./submit.sh run_probes --model llama-instruct --datasets ARC-Challenge ARC-Easy --alphas 0.02 0.05 --max-workers 40
```

### `train_probes_all.sh`

Example script showing various probe training commands (commented out). Useful as a reference for different experiment configurations.

## Key Parameters

### Probe Training (`run_probes`)

- `--datasets` - One or more dataset names (space-separated)
- `--alphas` - Lasso regularization values (e.g., `0.02 0.05 0.1`)
- `--save-name` - Suffix for output files (e.g., `linear_intercept`, `logit_intercept`)
- `--token-pos` - Token positions: `exact`, `last`, or `both` (default: `exact`)
- `--error-type` - Error metric: `SM` (softmax) or `CE` (cross-entropy) (default: `SM`)
- `--max-workers` - Parallel workers for training (default: 25)
- `--seed` - Random seed (default: 52)
- `--nr-attempts` - Number of train/test splits per layer (default: 5)
- `--transform-targets` - Apply logit transformation to targets (default: enabled)
- `--normalize-features` - Standardize features (default: enabled)

### Caching (`cache`)

- `--batch-size` - Batch size for processing (default: 1)
- `--device` - Device to use (default: `cuda:0`)
- `--nr-samples` - Number of samples to process (default: 2000)

## Project Structure

```
apertus-probes/
├── src/
│   ├── cache/           # Activation caching and extraction
│   │   ├── cache_run.py
│   │   ├── cache_postprocess.py
│   │   └── cache_utils.py
│   ├── probes/          # Probe training and evaluation
│   │   ├── probes_train.py
│   │   ├── probes_core.py
│   │   ├── run_probes.py
│   │   ├── run_cross_dataset_probes.py
│   │   ├── plot_utils.py
│   │   └── analyze.ipynb
│   ├── steering/        # Activation steering (optional)
│   │   └── steering_run.py
│   └── tasks/           # Dataset and task handlers
│       └── task_handler.py
├── scripts/             # Dataset-specific scripts
├── submit.sh           # Main submission script
├── train_probes_all.sh # Example commands
├── probes.toml         # Environment configuration
└── requirements.txt    # Python dependencies
```

## Output Locations

All outputs are saved to `$SCRATCH/mera-runs/`:

- **Cached activations**: `$SCRATCH/mera-runs/<dataset>/<model>/`
- **Probe results**: `$SCRATCH/mera-runs/mix/<dataset>/<model>/df_probes_*.pkl`
- **Cross-dataset probes**: `$SCRATCH/mera-runs/cross_dataset/<train>_to_<test>/<model>/`

## Probe Types

The project supports several probe model types:

- **Lasso Regression** (`L-<alpha>`) - For regression tasks (predicting error values)
  - Examples: `L-0.02`, `L-0.05`, `L-0.1`, `L-0.25`, `L-0.5`
- **Linear Regression** (`L-0`) - Unregularized regression
- **Logistic Regression** (`LogReg-l1`) - For classification tasks

Probes can predict:
- **Regression targets**: Softmax error (SM), Cross-entropy (CE)
- **Classification targets**: Accuracy, AUC-ROC (for classification tasks)

## Analysis Workflow

1. **Train probes** using `submit.sh run_probes`
2. **Load results** in `analyze.ipynb`:
   ```python
   from plot_utils import plot_rmse_comparison_multi, plot_rmse_on_axis
   ```
3. **Visualize** using plotting functions:
   - Single dataset comparisons
   - Multi-dataset comparisons
   - Layer-wise performance
   - Token position analysis (exact vs. last)

## Tips

- Use `--local` flag for quick testing before submitting large jobs
- Check job status with `squeue -u $USER`
- View logs in `logs/` directory
- For mixed datasets, the probe is trained on concatenated data from all specified datasets
- Cross-dataset probes evaluate generalization: train on one dataset, test on another

## Environment

The project is designed for use on CSCS systems with:
- Container environment (specified in `probes.toml`)
- SLURM job scheduler
- Access to `/iopsstor` and `/capstor` storage

For local development, use the `--local` flag with `submit.sh`.

## License

[Add license information if applicable]

## Citation

[Add citation information if applicable]
