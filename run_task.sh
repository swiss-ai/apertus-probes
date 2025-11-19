#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -J apertus_task
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH --environment=probes

# Unified script to run cache, postprocess, or train_probes tasks
# 
# This script is designed to run under sbatch. Use submit.sh to run tasks
# either locally or with sbatch.
#
# This script expects to run under SLURM (via sbatch). It will:
# - Update the job name to include task, model, and dataset
# - Create custom log files with descriptive names
# - Run the specified task
#
# Use submit.sh instead of calling this directly.

# Parameters:
#   --model <alias>             Short model alias (use --list-models to see available aliases)
#   --model_name <full_name>    Full model name (e.g., "swiss-ai/Apertus-8B-Instruct-2509")
#   --dataset_name <name>       Dataset name (required)
#
# Optional parameters:
#   --nr_samples <int>          Number of samples for cache task (default: 2000)
#   --nr_layers <int>           Number of layers for postprocess/train_probes (default: 32)
#   --batch_size <int>          Batch size for cache task (default: 1)
#   --device <str>              Device for cache task (default: cuda:0)
#   --token_pos <str>           Token positions for train_probes (default: "" "_exact")
#   --process_saes <str>        Process SAEs for train_probes (default: False)
#   --transform_targets <str>   Transform targets for train_probes (default: True)
#   --save_name <str>           Save name for train_probes (default: dataset_name)

set -euo pipefail

# Get the absolute path of the script directory
# This works even when the script is called via sbatch with a relative path
# When sbatch runs the script, BASH_SOURCE[0] might point to a copy/symlink
# in /var/spool/slurmd/job*/ directory, so we need to handle this case
SCRIPT_PATH="${BASH_SOURCE[0]}"

# Check if we're running under SLURM and the script path is in SLURM's temp directory
# If so, --chdir should have set us to the correct directory, so use pwd
if [ -n "${SLURM_JOB_ID:-}" ] && [[ "$SCRIPT_PATH" == /var/spool/slurmd/* ]]; then
    # Running under SLURM with script in temp directory
    # --chdir in sbatch ensures we're already in the project directory
    ROOT="$(pwd)"
    echo "[INFO] Running under SLURM, using current working directory: $ROOT"
else
    # Normal execution - detect script location
    # Try to resolve if it's a symlink
    if [ -L "$SCRIPT_PATH" ]; then
        # Try readlink -f first (GNU), fallback to readlink (BSD)
        if command -v readlink >/dev/null 2>&1; then
            # Try to get absolute path
            if readlink -f "$SCRIPT_PATH" >/dev/null 2>&1; then
                SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
            else
                # BSD readlink (macOS) - need to resolve manually
                TARGET="$(readlink "$SCRIPT_PATH")"
                if [ "${TARGET:0:1}" != "/" ]; then
                    # Relative symlink
                    SCRIPT_PATH="$(cd "$(dirname "$SCRIPT_PATH")" && cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
                else
                    # Absolute symlink
                    SCRIPT_PATH="$TARGET"
                fi
            fi
        fi
    fi
    # Get absolute path of script directory
    ROOT="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
fi

# Change to project directory
cd "$ROOT" || {
    echo "Error: Could not change to directory: $ROOT"
    exit 1
}

# Available tasks
TASKS=("cache" "postprocess" "train_probes")

# Model aliases - map short names to full model names
declare -A MODELS
MODELS["apertus"]="swiss-ai/Apertus-8B-Instruct-2509"
MODELS["apertus-8b"]="swiss-ai/Apertus-8B-Instruct-2509"
MODELS["llama"]="meta-llama/Llama-3.1-8B-Instruct"
# Add more models as needed:
# MODELS["qwen2.5-3b"]="Qwen/Qwen2.5-3B-Instruct"
# MODELS["llama3.2-1b"]="meta-llama/Llama-3.2-1B-Instruct"

# Available datasets (must match dataset_info keys in src/tasks/task_handler.py)
DATASETS=(
    "sms_spam"
    "mmlu_pro_natural_science"
    "mmlu_high_school"
    "mmlu_professional"
    "ARC-Easy"
    "ARC-Challenge"
)

# Function to check if dataset is valid
is_valid_dataset() {
    local dataset="$1"
    for valid_dataset in "${DATASETS[@]}"; do
        if [ "$dataset" == "$valid_dataset" ]; then
            return 0
        fi
    done
    return 1
}

# Get task from first argument
TASK="${1:-}"

if [ -z "$TASK" ]; then
    echo "Error: Task parameter is required"
    echo "Usage: run_task.sh <task> --model <alias> --dataset_name <dataset> [options]"
    echo "Available tasks: ${TASKS[*]}"
    echo "Note: This script is designed to run under sbatch. Use submit.sh instead."
    exit 1
fi

# Validate task
if [[ ! " ${TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "Error: Invalid task '$TASK'"
    echo "Available tasks: ${TASKS[*]}"
    exit 1
fi

shift  # Remove task from arguments

# Parse arguments
MODEL_NAME=""
MODEL_ALIAS=""
DATASET_NAME=""
NR_SAMPLES="2000"
NR_LAYERS="32"
BATCH_SIZE="1"
DEVICE="cuda:0"
TOKEN_POS_ARGS=()
PROCESS_SAES="False"
TRANSFORM_TARGETS="True"
SAVE_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_ALIAS="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --nr_samples)
            NR_SAMPLES="$2"
            shift 2
            ;;
        --nr_layers)
            NR_LAYERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --token_pos)
            # Handle multiple token positions - consume all following non-flag arguments
            shift
            TOKEN_POS_ARGS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TOKEN_POS_ARGS+=("$1")
                shift
            done
            ;;
        --process_saes)
            PROCESS_SAES="$2"
            shift 2
            ;;
        --transform_targets)
            TRANSFORM_TARGETS="$2"
            shift 2
            ;;
        --save_name)
            SAVE_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resolve model name from alias if provided
if [ -n "$MODEL_ALIAS" ]; then
    if [ -n "$MODEL_NAME" ]; then
        echo "Error: Cannot use both --model and --model_name. Use only one."
        exit 1
    fi
    if [ -z "${MODELS[$MODEL_ALIAS]:-}" ]; then
        echo "Error: Unknown model alias '$MODEL_ALIAS'"
        exit 1
    fi
    MODEL_NAME="${MODELS[$MODEL_ALIAS]}"
fi

# Validate required parameters
if [ -z "$MODEL_NAME" ] || [ -z "$DATASET_NAME" ]; then
    echo "Error: --model <alias> or --model_name <full_name> and --dataset_name are required"
    exit 1
fi

# Validate dataset
if ! is_valid_dataset "$DATASET_NAME"; then
    echo "Error: Invalid dataset '$DATASET_NAME'"
    exit 1
fi

# Set default save_name if not provided (use dataset_name)
if [ -z "$SAVE_NAME" ] && [ "$TASK" == "train_probes" ]; then
    SAVE_NAME="$DATASET_NAME"
fi

# Set SCRATCH if not already set
if [ -z "${SCRATCH:-}" ]; then
    SCRATCH="/iopsstor/scratch/cscs/$USER"
fi

# Create custom log file name and update job name (if running under sbatch)
# Extract short model name from full path (e.g., "Apertus-8B-Instruct-2509" from "swiss-ai/Apertus-8B-Instruct-2509")
MODEL_SHORT=$(basename "$MODEL_NAME")
# Clean dataset name for filename (replace spaces/special chars)
DATASET_CLEAN=$(echo "$DATASET_NAME" | tr ' ' '_' | tr '/' '_')

# Create descriptive job name (max 100 chars for SLURM)
JOB_NAME="${TASK}_${MODEL_SHORT}_${DATASET_CLEAN}"
# Truncate if too long
if [ ${#JOB_NAME} -gt 100 ]; then
    JOB_NAME="${JOB_NAME:0:100}"
fi

# Try to update the job name in SLURM (works better for running jobs than pending)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update job="$SLURM_JOB_ID" name="$JOB_NAME" 2>/dev/null || true
    
    LOG_FILE="$ROOT/logs/${TASK}_${MODEL_SHORT}_${DATASET_CLEAN}_${SLURM_JOB_ID}.out"
    mkdir -p "$ROOT/logs" || {
        echo "Warning: Could not create logs directory at $ROOT/logs"
        echo "Falling back to default SLURM log only"
        LOG_FILE=""
    }
    # Redirect all output to custom log file (in addition to default SLURM log)
    # Using tee to write to both the custom log and stdout (which SLURM captures)
    if [ -n "$LOG_FILE" ]; then
        exec 1> >(tee -a "$LOG_FILE")
        exec 2> >(tee -a "$LOG_FILE" >&2)
    fi
fi

echo "Running $TASK task"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "----------------------------------------"

# Execute the appropriate task
case $TASK in
    cache)
        echo "Cache parameters:"
        echo "  --batch_size: $BATCH_SIZE"
        echo "  --device: $DEVICE"
        echo ""
        python3 "$ROOT/src/cache/cache_run.py" \
            --cache_dir "$SCRATCH/mera-runs/processed_datasets/" \
            --save_dir "$SCRATCH/mera-runs/" \
            --model_name "$MODEL_NAME" \
            --dataset_names "$DATASET_NAME" \
            --batch_size "$BATCH_SIZE" \
            --n_devices 1 \
            --flexible_match \
            --no-overwrite \
            --device "$DEVICE"
        ;;
    
    postprocess)
        echo "Postprocess parameters:"
        echo "  --nr_layers: $NR_LAYERS"
        echo ""
        python3 "$ROOT/src/cache/cache_postprocess.py" \
            --save_dir "$SCRATCH/mera-runs/" \
            --dataset_name "$DATASET_NAME" \
            --model_name "$MODEL_NAME" \
            --nr_layers "$NR_LAYERS"
        ;;
    
    train_probes)
        echo "Train probes parameters:"
        echo "  --nr_layers: $NR_LAYERS"
        # Use TOKEN_POS_ARGS if set, otherwise default to empty and _exact
        if [ ${#TOKEN_POS_ARGS[@]} -eq 0 ]; then
            TOKEN_POS_ARGS=("" "_exact")
        fi
        echo "  --token_pos: ${TOKEN_POS_ARGS[*]}"
        echo "  --process_saes: $PROCESS_SAES"
        echo "  --transform_targets: $TRANSFORM_TARGETS"
        echo "  --save_name: $SAVE_NAME"
        echo ""
        python3 "$ROOT/src/probes/probes_train.py" \
            --save_dir "$SCRATCH/mera-runs/" \
            --dataset_names "$DATASET_NAME" \
            --model_names "$MODEL_NAME" \
            --nr_layers "$NR_LAYERS" \
            --token_pos "${TOKEN_POS_ARGS[@]}" \
            --process_saes "$PROCESS_SAES" \
            --transform_targets "$TRANSFORM_TARGETS" \
            --save_name "$SAVE_NAME"
        ;;
esac

echo ""
echo "Task $TASK completed successfully!"

