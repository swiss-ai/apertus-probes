#!/usr/bin/env bash
#SBATCH -A infra01
#SBATCH -p normal
#SBATCH -t 02:00:00
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
#   --dataset_name <name>       Dataset name (required for cache/postprocess/train_probes)
#   --datasets <name1> [name2]  Dataset names for run_probes (space-separated, can be multiple)
#
# Optional parameters:
#   --nr_samples <int>          Number of samples for cache task (default: 2000)
#   --nr_layers <int>           Number of layers for postprocess/train_probes (default: 32)
#   --batch_size <int>          Batch size for cache task (default: 1)
#   --device <str>              Device for cache task (default: cuda:0)
#   --token_pos <str>           Token positions for train_probes (default: "" "_exact")
#   --process_saes <str>        Process SAEs for train_probes (default: False)
#   --transform_targets <str>   Transform targets for train_probes (default: True)
#   --save_name <str>           Save name for train_probes/run_probes (default: dataset_name or "")
#   --error_type <str>          Error type for run_probes: SM or CE (default: SM)
#   --seed <int>                Random seed for run_probes (default: 52)
#   --nr_attempts <int>         Number of attempts for run_probes (default: 5)
#   --max_trials <int>          Max refits for run_probes (default: 5)
#   --max_workers <int>         Max threads for run_probes (default: 25)
#   --alphas <float> [float]    L1 alphas for run_probes (default: 0.5 0.25 0.1 0.05)
#   --no_transform_targets     Disable logit transform for run_probes
#   --no_normalize_features     Disable feature normalization for run_probes

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
TASKS=("cache" "postprocess" "train_probes" "run_probes" "cross_dataset_probes" "steer_multi")

# Model aliases - map short names to full model names
declare -A MODELS
MODELS["apertus-instruct"]="swiss-ai/Apertus-8B-Instruct-2509"
MODELS["apertus-base"]="swiss-ai/Apertus-8B-2509"
MODELS["llama-instruct"]="meta-llama/Llama-3.1-8B-Instruct"
MODELS["llama-base"]="meta-llama/Llama-3.1-8B"

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
    "sujet_finance_yesno_5k"
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

# Function to show default values for a task
show_defaults() {
    local task="$1"
    
    case "$task" in
        cache)
            echo "Default values for 'cache' task:"
            echo ""
            echo "  --cache_dir: /capstor/store/cscs/swissai/infra01/apertus_probes/processed_datasets"
            echo "  --save_dir: /capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs"
            echo "  --nr_samples: 2000"
            echo "  --batch_size: 1"
            echo "  --device: cuda:0"
            echo ""
            ;;
        postprocess)
            echo "Default values for 'postprocess' task:"
            echo ""
            echo "  --nr_layers: 32"
            echo ""
            ;;
        train_probes)
            echo "Default values for 'train_probes' task:"
            echo ""
            echo "  --nr_layers: 32"
            echo "  --token_pos: \"\" \"_exact\" (both positions)"
            echo "  --process_saes: False"
            echo "  --transform_targets: True"
            echo "  --save_name: <dataset_name>"
            echo ""
            ;;
        run_probes)
            echo "Default values for 'run_probes' task:"
            echo ""
            echo "  --save-dir: /capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs"
            echo "  --save-name: \"\" (empty, uses dataset mixture name)"
            echo "  --error-type: SM"
            echo "  --token-pos: both"
            echo "  --seed: 52"
            echo "  --nr-attempts: 5"
            echo "  --max-trials: 5"
            echo "  --max-workers: 25"
            echo "  --alphas: 0.5 0.25 0.1 0.05"
            echo "  --transform-targets: enabled (logit transform)"
            echo "  --normalize-features: disabled (use --no-normalize-features to enable)"
            echo ""
            echo "Required parameters:"
            echo "  --datasets: <name1> [name2] ... (space-separated, at least one required)"
            echo "  --model-name: <model_name> (required)"
            echo ""
            ;;
        cross_dataset_probes)
            echo "Default values for 'cross_dataset_probes' task:"
            echo ""
            echo "  --save-dir: /capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs"
            echo "  --save-name: \"\" (empty, uses '{train_dataset}_to_{test_dataset}')"
            echo "  --error-type: SM"
            echo "  --token-pos: exact"
            echo "  --seed: 52"
            echo "  --max-trials: 5"
            echo "  --max-workers: 25"
            echo "  --alphas: 0.5 0.25 0.1 0.05"
            echo "  --transform-targets: enabled (logit transform)"
            echo "  --normalize-features: disabled (use --no-normalize-features to enable)"
            echo ""
            echo "Required parameters:"
            echo "  --train-dataset: <dataset_name> (dataset to train on)"
            echo "  --test-dataset: <dataset_name> (dataset to test on)"
            echo "  --model-name: <model_name> (required)"
            echo ""
            ;;
        steer_multi)
            echo "Default values for 'steer_multi' task:"
            echo ""
            echo "  --cache-dir: /capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/"
            echo "  --save-dir: /capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/steering_outputs/"
            echo "  --regression-model-type: linear (options: linear or logit)"
            echo "  --top-k: 50 100 200"
            echo "  --token-pos: exact"
            echo "  --error-type: sm"
            echo "  --objective: Accuracy"
            echo "  --probe-file: df_probes_both"
            echo ""
            echo "Required parameters:"
            echo "  --model: <alias> or --model_name <full_name> (required)"
            echo "  --dataset_name: <dataset_name> (required)"
            echo ""
            echo "This task automatically runs 4 predefined configurations:"
            echo "  Config 1: no_steering optimal_probe"
            echo "  Config 2: vanilla_contrastive prompt_steering"
            echo "  Config 3: optimal_logistic_probe additive_probe"
            echo "  Config 4: optimal_contrastive"
            echo ""
            echo "Fname patterns are auto-generated as: <dataset>_<regression_model_type>_<config_num>"
            echo "  (e.g., mmlu_high_school_linear_1, mmlu_high_school_linear_2, etc.)"
            echo ""
            ;;
        *)
            echo "Available tasks: ${TASKS[*]}"
            echo ""
            echo "Usage: run_task.sh <task> --show-defaults"
            echo ""
            echo "To see defaults for all tasks, run:"
            for t in "${TASKS[@]}"; do
                echo "  run_task.sh $t --show-defaults"
            done
            ;;
    esac
}

# Get task from first argument
TASK="${1:-}"

if [ -z "$TASK" ]; then
    echo "Error: Task parameter is required"
    echo "Usage: run_task.sh <task> --model <alias> --dataset_name <dataset> [options]"
    echo "Available tasks: ${TASKS[*]}"
    echo "Note: This script is designed to run under sbatch. Use submit.sh instead."
    echo ""
    echo "To see default values for a task:"
    echo "  run_task.sh <task> --show-defaults"
    exit 1
fi

# Validate task
if [[ ! " ${TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "Error: Invalid task '$TASK'"
    echo "Available tasks: ${TASKS[*]}"
    exit 1
fi

shift  # Remove task from arguments

# Check if user wants to see defaults
if [ "${1:-}" == "--show-defaults" ] || [ "${1:-}" == "--list-defaults" ]; then
    show_defaults "$TASK"
    exit 0
fi

# Parse arguments
MODEL_NAME=""
MODEL_ALIAS=""
DATASET_NAME=""
USER_DATASETS=()  # For run_probes (multiple datasets provided by user)
TRAIN_DATASET=""  # For cross_dataset_probes
TEST_DATASET=""   # For cross_dataset_probes
NR_SAMPLES="2000"
NR_LAYERS="32"
BATCH_SIZE="1"
DEVICE="cuda:0"
TOKEN_POS_ARGS=()
PROCESS_SAES="False"
TRANSFORM_TARGETS="True"
SAVE_NAME=""
# run_probes parameters
ERROR_TYPE="SM"
SEED="52"
NR_ATTEMPTS="5"
MAX_TRIALS="5"
MAX_WORKERS="25"
ALPHAS=()
NO_TRANSFORM_TARGETS=false
NORMALIZE_FEATURES=false
USE_LOGIT=false
TOKEN_POS_RUN_PROBES="both"
# cache task parameters
OVERWRITE=""  # Empty means use default (True), "--overwrite" or "--no-overwrite" will be passed
FLEXIBLE_MATCH=""  # Empty means use default (True)
RUN_ACTS=""  # Empty means use default (True)
RUN_SAES=""  # Empty means use default (False)
# steer_multi parameters
FNAME_PATTERNS=()  # Array of fname patterns (e.g., mmlu_hs_1 mmlu_hs_2)
STEERING_METHODS=()  # Array of steering methods per config
REGRESSION_MODEL_TYPE="linear"  # linear or logit
TOP_K="50 100 200"
TOKEN_POS_STEER="exact"
ERROR_TYPE_STEER="sm"
OBJECTIVE="Accuracy"
PROBE_FILE="df_probes_both"
CACHE_DIR_STEER="/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/"
SAVE_DIR_STEER="/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/steering_outputs/"
WANDB_KEY="private"

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
        --datasets)
            # Handle multiple datasets - consume all following non-flag arguments
            shift
            USER_DATASETS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                USER_DATASETS+=("$1")
                shift
            done
            ;;
        --train-dataset)
            TRAIN_DATASET="$2"
            shift 2
            ;;
        --test-dataset)
            TEST_DATASET="$2"
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
        --save_name|--save-name)
            SAVE_NAME="$2"
            shift 2
            ;;
        --error_type|--error-type)
            ERROR_TYPE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --nr_attempts|--nr-attempts)
            NR_ATTEMPTS="$2"
            shift 2
            ;;
        --max_trials|--max-trials)
            MAX_TRIALS="$2"
            shift 2
            ;;
        --max_workers|--max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --alphas)
            # Handle multiple alphas - consume all following non-flag arguments
            shift
            ALPHAS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ALPHAS+=("$1")
                shift
            done
            ;;
        --no_transform_targets|--no-transform-targets)
            NO_TRANSFORM_TARGETS=true
            shift
            ;;
        --normalize_features|--normalize-features)
            NORMALIZE_FEATURES=true
            shift
            ;;
        --no_normalize_features|--no-normalize-features)
            # Deprecated: kept for backward compatibility, but does nothing (default is now False)
            shift
            ;;
        --use_logit|--use-logit)
            USE_LOGIT=true
            shift
            ;;
        --token_pos|--token-pos)
            # For run_probes, token_pos is a single value, not multiple
            TOKEN_POS_RUN_PROBES="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        --no-overwrite)
            OVERWRITE="--no-overwrite"
            shift
            ;;
        --flexible_match|--flexible-match)
            FLEXIBLE_MATCH="--flexible-match"
            shift
            ;;
        --no-flexible-match)
            FLEXIBLE_MATCH="--no-flexible-match"
            shift
            ;;
        --run_acts|--run-acts)
            RUN_ACTS="--run-acts"
            shift
            ;;
        --no-run-acts)
            RUN_ACTS="--no-run-acts"
            shift
            ;;
        --run_saes|--run-saes)
            RUN_SAES="--run-saes"
            shift
            ;;
        --no-run-saes)
            RUN_SAES="--no-run-saes"
            shift
            ;;
        --fname-patterns)
            # Handle multiple fname patterns
            shift
            FNAME_PATTERNS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FNAME_PATTERNS+=("$1")
                shift
            done
            ;;
        --steering-methods)
            # Handle multiple steering methods
            shift
            STEERING_METHODS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STEERING_METHODS+=("$1")
                shift
            done
            ;;
        --regression-model-type)
            REGRESSION_MODEL_TYPE="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --token-pos-steer)
            TOKEN_POS_STEER="$2"
            shift 2
            ;;
        --error-type-steer)
            ERROR_TYPE_STEER="$2"
            shift 2
            ;;
        --objective)
            OBJECTIVE="$2"
            shift 2
            ;;
        --probe-file)
            PROBE_FILE="$2"
            shift 2
            ;;
        --cache-dir-steer)
            CACHE_DIR_STEER="$2"
            shift 2
            ;;
        --save-dir-steer)
            SAVE_DIR_STEER="$2"
            shift 2
            ;;
        --wandb-key)
            WANDB_KEY="$2"
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

# Validate required parameters based on task
if [ "$TASK" == "steer_multi" ]; then
    # For steer_multi, need model and dataset (configs are predefined)
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: --model <alias> or --model_name <full_name> is required for steer_multi"
        exit 1
    fi
    if [ -z "$DATASET_NAME" ]; then
        echo "Error: --dataset_name is required for steer_multi"
        exit 1
    fi
    # Validate dataset
    if ! is_valid_dataset "$DATASET_NAME"; then
        echo "Error: Invalid dataset '$DATASET_NAME'"
        exit 1
    fi
elif [ "$TASK" == "run_probes" ]; then
    # For run_probes, need model and datasets (can be multiple)
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: --model <alias> or --model_name <full_name> is required for run_probes"
        exit 1
    fi
    if [ ${#USER_DATASETS[@]} -eq 0 ] && [ -z "$DATASET_NAME" ]; then
        echo "Error: --datasets <name1> [name2] ... is required for run_probes"
        exit 1
    fi
    # If DATASET_NAME is set but USER_DATASETS is empty, use DATASET_NAME
    if [ ${#USER_DATASETS[@]} -eq 0 ] && [ -n "$DATASET_NAME" ]; then
        USER_DATASETS=("$DATASET_NAME")
    fi
    # Validate all datasets
    for dataset in "${USER_DATASETS[@]}"; do
        if ! is_valid_dataset "$dataset"; then
            echo "Error: Invalid dataset '$dataset'"
            exit 1
        fi
    done
elif [ "$TASK" == "cross_dataset_probes" ]; then
    # For cross_dataset_probes, need model, train_dataset, and test_dataset
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: --model <alias> or --model_name <full_name> is required for cross_dataset_probes"
        exit 1
    fi
    if [ -z "$TRAIN_DATASET" ]; then
        echo "Error: --train-dataset <name> is required for cross_dataset_probes"
        exit 1
    fi
    if [ -z "$TEST_DATASET" ]; then
        echo "Error: --test-dataset <name> is required for cross_dataset_probes"
        exit 1
    fi
    # Validate datasets
    if ! is_valid_dataset "$TRAIN_DATASET"; then
        echo "Error: Invalid train dataset '$TRAIN_DATASET'"
        exit 1
    fi
    if ! is_valid_dataset "$TEST_DATASET"; then
        echo "Error: Invalid test dataset '$TEST_DATASET'"
        exit 1
    fi
else
    # For other tasks, need model and single dataset
    if [ -z "$MODEL_NAME" ] || [ -z "$DATASET_NAME" ]; then
        echo "Error: --model <alias> or --model_name <full_name> and --dataset_name are required"
        exit 1
    fi
    # Validate dataset
    if ! is_valid_dataset "$DATASET_NAME"; then
        echo "Error: Invalid dataset '$DATASET_NAME'"
        exit 1
    fi
fi

# Set default save_name if not provided
if [ -z "$SAVE_NAME" ]; then
    if [ "$TASK" == "train_probes" ]; then
        SAVE_NAME="$DATASET_NAME"
    elif [ "$TASK" == "run_probes" ]; then
        # For run_probes, save_name defaults to empty (will use dataset mixture name)
        SAVE_NAME=""
    elif [ "$TASK" == "cross_dataset_probes" ]; then
        # For cross_dataset_probes, save_name defaults to empty (will use '{train_dataset}_to_{test_dataset}')
        SAVE_NAME=""
    fi
fi

# Set default alphas if not provided for run_probes or cross_dataset_probes
if [ ${#ALPHAS[@]} -eq 0 ] && ([ "$TASK" == "run_probes" ] || [ "$TASK" == "cross_dataset_probes" ]); then
    ALPHAS=(0.5 0.25 0.1 0.05)
fi

# Set default save directory
DEFAULT_SAVE_DIR="/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs"
DEFAULT_CACHE_DIR="/capstor/store/cscs/swissai/infra01/apertus_probes/processed_datasets"
if [ -z "${SCRATCH:-}" ]; then
    SCRATCH="/iopsstor/scratch/cscs/$USER"
fi

# Create custom log file name and update job name (if running under sbatch)
# Extract short model name from full path (e.g., "Apertus-8B-Instruct-2509" from "swiss-ai/Apertus-8B-Instruct-2509")
MODEL_SHORT=$(basename "$MODEL_NAME")
# Clean dataset name(s) for filename (replace spaces/special chars)
if [ "$TASK" == "run_probes" ]; then
    # For run_probes, join datasets with +
    DATASET_CLEAN=$(IFS=+; echo "${USER_DATASETS[*]}" | tr ' ' '_' | tr '/' '_')
elif [ "$TASK" == "cross_dataset_probes" ]; then
    # For cross_dataset_probes, use train_to_test format
    TRAIN_CLEAN=$(echo "$TRAIN_DATASET" | tr ' ' '_' | tr '/' '_')
    TEST_CLEAN=$(echo "$TEST_DATASET" | tr ' ' '_' | tr '/' '_')
    DATASET_CLEAN="${TRAIN_CLEAN}_to_${TEST_CLEAN}"
else
    DATASET_CLEAN=$(echo "$DATASET_NAME" | tr ' ' '_' | tr '/' '_')
fi

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
if [ "$TASK" == "run_probes" ]; then
    echo "Datasets: ${USER_DATASETS[*]}"
elif [ "$TASK" == "cross_dataset_probes" ]; then
    echo "Train dataset: $TRAIN_DATASET"
    echo "Test dataset: $TEST_DATASET"
else
    echo "Dataset: $DATASET_NAME"
fi
echo "----------------------------------------"

# Execute the appropriate task
case $TASK in
    cache)
        echo "Cache parameters:"
        echo "  --cache_dir: $DEFAULT_CACHE_DIR"
        echo "  --save_dir: $DEFAULT_SAVE_DIR"
        echo "  --batch_size: $BATCH_SIZE"
        echo "  --device: $DEVICE"
        [ -n "$OVERWRITE" ] && echo "  $OVERWRITE"
        [ -n "$FLEXIBLE_MATCH" ] && echo "  $FLEXIBLE_MATCH"
        [ -n "$RUN_ACTS" ] && echo "  $RUN_ACTS"
        [ -n "$RUN_SAES" ] && echo "  $RUN_SAES"
        echo ""
        python3 "$ROOT/src/cache/cache_run.py" \
            --cache_dir "$DEFAULT_CACHE_DIR/" \
            --save_dir "$DEFAULT_SAVE_DIR/" \
            --model_name "$MODEL_NAME" \
            --dataset_names "$DATASET_NAME" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            $OVERWRITE \
            $FLEXIBLE_MATCH \
            $RUN_ACTS \
            $RUN_SAES
        ;;
    
    postprocess)
        echo "Postprocess parameters:"
        echo "  --save_dir: $DEFAULT_SAVE_DIR"
        echo "  --nr_layers: $NR_LAYERS"
        echo ""
        python3 "$ROOT/src/cache/cache_postprocess.py" \
            --save_dir "$DEFAULT_SAVE_DIR/" \
            --dataset_name "$DATASET_NAME" \
            --model_name "$MODEL_NAME" \
            --nr_layers "$NR_LAYERS"
        ;;
    
    # train_probes)
    #     echo "Train probes parameters:"
    #     echo "  --nr_layers: $NR_LAYERS"
    #     # Use TOKEN_POS_ARGS if set, otherwise default to empty and _exact
    #     if [ ${#TOKEN_POS_ARGS[@]} -eq 0 ]; then
    #         TOKEN_POS_ARGS=("" "_exact")
    #     fi
    #     echo "  --token_pos: ${TOKEN_POS_ARGS[*]}"
    #     echo "  --process_saes: $PROCESS_SAES"
    #     echo "  --transform_targets: $TRANSFORM_TARGETS"
    #     echo "  --save_name: $SAVE_NAME"
    #     echo ""
    #     python3 "$ROOT/src/probes/probes_train.py" \
    #         --save_dir "$SCRATCH/mera-runs/" \
    #         --dataset_names "$DATASET_NAME" \
    #         --model_names "$MODEL_NAME" \
    #         --nr_layers "$NR_LAYERS" \
    #         --token_pos "${TOKEN_POS_ARGS[@]}" \
    #         --process_saes "$PROCESS_SAES" \
    #         --transform_targets "$TRANSFORM_TARGETS" \
    #         --save_name "$SAVE_NAME"
    #     ;;
    train_probes)
        echo "Train probes parameters:"
        # USE TOKEN_POS_ARGS if set, otherwise default to both
        if [ ${#TOKEN_POS_ARGS[@]} -eq 0 ]; then
            TOKEN_POS_ARGS=("" "_exact")
        fi
        echo "  --token_pos: ${TOKEN_POS_ARGS[*]}"
        echo "  --transform_targets: $TRANSFORM_TARGETS"
        echo "  --save_name: $SAVE_NAME"
        echo ""
        ;;
    
    run_probes)
        echo "Run probes parameters:"
        echo "  --datasets: ${USER_DATASETS[*]}"
        echo "  --model-name: $MODEL_SHORT"
        echo "  --save-dir: $DEFAULT_SAVE_DIR"
        echo "  --save-name: $SAVE_NAME"
        echo "  --token-pos: $TOKEN_POS_RUN_PROBES"
        echo "  --seed: $SEED"
        echo "  --nr-attempts: $NR_ATTEMPTS"
        echo "  --max-trials: $MAX_TRIALS"
        echo "  --max-workers: $MAX_WORKERS"
        echo "  --error-type: $ERROR_TYPE"
        echo "  --alphas: ${ALPHAS[*]}"
        if [ "$NO_TRANSFORM_TARGETS" = true ]; then
            echo "  --no-transform-targets: enabled"
        fi
        if [ "$NORMALIZE_FEATURES" = true ]; then
            echo "  --normalize-features: enabled"
        fi
        echo ""
        
        # Build command
        RUN_PROBES_CMD=(
            python3 "$ROOT/src/probes/run_probes.py"
            --datasets "${USER_DATASETS[@]}"
            --model-name "$MODEL_SHORT"
            --save-dir "$DEFAULT_SAVE_DIR"
            --token-pos "$TOKEN_POS_RUN_PROBES"
            --seed "$SEED"
            --nr-attempts "$NR_ATTEMPTS"
            --max-trials "$MAX_TRIALS"
            --max-workers "$MAX_WORKERS"
            --error-type "$ERROR_TYPE"
        )
        
        # Add optional save-name
        if [ -n "$SAVE_NAME" ]; then
            RUN_PROBES_CMD+=(--save-name "$SAVE_NAME")
        fi
        
        # Add alphas
        if [ ${#ALPHAS[@]} -gt 0 ]; then
            RUN_PROBES_CMD+=(--alphas "${ALPHAS[@]}")
        fi
        
        # Add boolean flags
        if [ "$NO_TRANSFORM_TARGETS" = true ]; then
            RUN_PROBES_CMD+=(--no-transform-targets)
        fi
        if [ "$NORMALIZE_FEATURES" = true ]; then
            RUN_PROBES_CMD+=(--normalize-features)
        fi
        if [ "$USE_LOGIT" = true ]; then
            RUN_PROBES_CMD+=(--use-logit)
        fi
        
        # Execute
        "${RUN_PROBES_CMD[@]}"
        ;;
    
    cross_dataset_probes)
        echo "Cross-dataset probes parameters:"
        echo "  --train-dataset: $TRAIN_DATASET"
        echo "  --test-dataset: $TEST_DATASET"
        echo "  --model-name: $MODEL_SHORT"
        echo "  --save-dir: $DEFAULT_SAVE_DIR"
        echo "  --save-name: $SAVE_NAME"
        echo "  --token-pos: $TOKEN_POS_RUN_PROBES"
        echo "  --seed: $SEED"
        echo "  --max-trials: $MAX_TRIALS"
        echo "  --max-workers: $MAX_WORKERS"
        echo "  --error-type: $ERROR_TYPE"
        echo "  --alphas: ${ALPHAS[*]}"
        if [ "$NO_TRANSFORM_TARGETS" = true ]; then
            echo "  --no-transform-targets: enabled"
        fi
        if [ "$NORMALIZE_FEATURES" = true ]; then
            echo "  --normalize-features: enabled"
        fi
        echo ""
        
        # Build command
        CROSS_DATASET_CMD=(
            python3 "$ROOT/src/probes/run_cross_dataset_probes.py"
            --train-dataset "$TRAIN_DATASET"
            --test-dataset "$TEST_DATASET"
            --model-name "$MODEL_SHORT"
            --save-dir "$DEFAULT_SAVE_DIR"
            --token-pos "$TOKEN_POS_RUN_PROBES"
            --seed "$SEED"
            --max-trials "$MAX_TRIALS"
            --max-workers "$MAX_WORKERS"
            --error-type "$ERROR_TYPE"
        )
        
        # Add optional save-name
        if [ -n "$SAVE_NAME" ]; then
            CROSS_DATASET_CMD+=(--save-name "$SAVE_NAME")
        fi
        
        # Add alphas
        if [ ${#ALPHAS[@]} -gt 0 ]; then
            CROSS_DATASET_CMD+=(--alphas "${ALPHAS[@]}")
        fi
        
        # Add boolean flags
        if [ "$NO_TRANSFORM_TARGETS" = true ]; then
            CROSS_DATASET_CMD+=(--no-transform-targets)
        fi
        if [ "$NORMALIZE_FEATURES" = true ]; then
            CROSS_DATASET_CMD+=(--normalize-features)
        fi
        if [ "$USE_LOGIT" = true ]; then
            CROSS_DATASET_CMD+=(--use-logit)
        fi
        
        # Execute
        "${CROSS_DATASET_CMD[@]}"
        ;;
    
    steer_multi)
        echo "Steer multi parameters:"
        echo "  Model: $MODEL_NAME"
        echo "  Dataset: $DATASET_NAME"
        echo "  Regression model type: $REGRESSION_MODEL_TYPE"
        echo "  Top-k: $TOP_K"
        echo "  Token pos: $TOKEN_POS_STEER"
        echo "  Error type: $ERROR_TYPE_STEER"
        echo "  Objective: $OBJECTIVE"
        echo "  Probe file: $PROBE_FILE"
        echo "  Cache dir: $CACHE_DIR_STEER"
        echo "  Save dir: $SAVE_DIR_STEER"
        echo ""
        
        # Create logs directory
        LOGS_DIR="$ROOT/logs"
        mkdir -p "$LOGS_DIR"
        
        # Set wandb to online mode
        export WANDB_MODE=online
        
        # Create unique job tag for log files (use SLURM_JOB_ID if available, otherwise timestamp)
        if [ -n "${SLURM_JOB_ID:-}" ]; then
            JOB_TAG="${SLURM_JOB_ID}"
        else
            JOB_TAG="$(date +%Y%m%d_%H%M%S)_$$"
        fi
        
        # GPU assignments (cycle through if more configs than GPUs)
        # Determine number of GPUs available from SLURM or detect locally
        if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
            # Running under SLURM - use the allocated GPUs
            NUM_GPUS=$SLURM_GPUS_ON_NODE
            echo "[INFO] Running under SLURM with $NUM_GPUS GPU(s), job ID: $JOB_TAG"
        else
            # Running locally - try to detect available GPUs
            if command -v nvidia-smi &> /dev/null; then
                NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
                echo "[INFO] Detected $NUM_GPUS GPU(s) locally, job tag: $JOB_TAG"
            else
                NUM_GPUS=4  # Default fallback
                echo "[WARN] Could not detect GPUs, defaulting to $NUM_GPUS, job tag: $JOB_TAG"
            fi
        fi
        GPUS=($(seq 0 $((NUM_GPUS - 1))))
        
        # Define configurations (same as steer_multi_gpu_all.sh)
        # Generate fname based on dataset name and regression model type
        DATASET_SHORT=$(echo "$DATASET_NAME" | tr ' ' '_' | tr '/' '_' | tr '[:upper:]' '[:lower:]')
        
        # Configuration 1: no_steering optimal_probe
        CONFIG_1_FNAME="${DATASET_SHORT}_${REGRESSION_MODEL_TYPE}_1"
        CONFIG_1_METHODS=("no_steering" "optimal_probe")
        
        # Configuration 2: vanilla_contrastive prompt_steering
        CONFIG_2_FNAME="${DATASET_SHORT}_${REGRESSION_MODEL_TYPE}_2"
        CONFIG_2_METHODS=("vanilla_contrastive" "prompt_steering")
        
        # Configuration 3: optimal_logistic_probe additive_probe
        CONFIG_3_FNAME="${DATASET_SHORT}_${REGRESSION_MODEL_TYPE}_3"
        CONFIG_3_METHODS=("optimal_logistic_probe" "additive_probe")
        
        # Configuration 4: optimal_contrastive
        CONFIG_4_FNAME="${DATASET_SHORT}_${REGRESSION_MODEL_TYPE}_4"
        CONFIG_4_METHODS=("optimal_contrastive")
        
        CONFIGS=(
            "$CONFIG_1_FNAME:${CONFIG_1_METHODS[*]}"
            "$CONFIG_2_FNAME:${CONFIG_2_METHODS[*]}"
            "$CONFIG_3_FNAME:${CONFIG_3_METHODS[*]}"
            "$CONFIG_4_FNAME:${CONFIG_4_METHODS[*]}"
        )
        
        echo "[INFO] Will run ${#CONFIGS[@]} config(s) across GPU(s): ${GPUS[*]}"
        
        # Function to run steering from config
        run_steering_from_config() {
            local fname="$1"
            local gpu_id="$2"
            shift 2
            local methods=("$@")
            
            echo "========================================"
            echo "Running $fname on GPU $gpu_id"
            echo "  Model: $MODEL_NAME"
            echo "  Methods: ${methods[*]}"
            echo "  Probe file: $PROBE_FILE"
            echo "  Regression model type: $REGRESSION_MODEL_TYPE"
            echo "========================================"
            
            python3 "$ROOT/src/steering/steering_run.py" \
                --fname "$fname" \
                --cache_dir "$CACHE_DIR_STEER" \
                --save_dir "$SAVE_DIR_STEER" \
                --model_names "$MODEL_NAME" \
                --dataset_names "$DATASET_NAME" \
                --steering_methods "${methods[@]}" \
                --top_k_sets $TOP_K \
                --probe_token_pos "$TOKEN_POS_STEER" \
                --error_type "$ERROR_TYPE_STEER" \
                --objective_key "$OBJECTIVE" \
                --probe_file_name "$PROBE_FILE" \
                --regression-model-type "$REGRESSION_MODEL_TYPE" \
                --device "cuda:$gpu_id" \
                --wandb_key "$WANDB_KEY" \
                > "${LOGS_DIR}/steering_${fname}_gpu${gpu_id}_${JOB_TAG}.log" 2>&1 &
            
            local pid=$!
            echo "Started $fname on GPU $gpu_id (PID: $pid)"
        }
        
        # Launch all configurations
        echo "========================================"
        echo "Launching ${#CONFIGS[@]} steering configurations"
        echo "========================================"
        
        PIDS=()
        for i in "${!CONFIGS[@]}"; do
            config_line="${CONFIGS[$i]}"
            fname="${config_line%%:*}"
            methods_str="${config_line#*:}"
            # Convert methods string back to array
            read -ra methods <<< "$methods_str"
            
            gpu_idx=$((i % ${#GPUS[@]}))
            gpu_id="${GPUS[$gpu_idx]}"
            
            run_steering_from_config "$fname" "$gpu_id" "${methods[@]}"
            
            PIDS+=($!)
            sleep 2  # Small delay between launches
        done
        
        echo ""
        echo "All steering jobs launched!"
        echo "GPU assignments:"
        for i in "${!CONFIGS[@]}"; do
            config_line="${CONFIGS[$i]}"
            fname="${config_line%%:*}"
            gpu_idx=$((i % ${#GPUS[@]}))
            gpu_id="${GPUS[$gpu_idx]}"
            echo "  GPU $gpu_id: $fname"
        done
        echo ""
        echo "Waiting for completion..."
        wait
        
        echo ""
        echo "All steering jobs completed!"
        ;;

esac

echo ""
echo "Task $TASK completed successfully!"

