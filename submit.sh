#!/usr/bin/env bash
# Easy-to-use wrapper script for submitting jobs with sbatch
# This script automatically sets descriptive job names based on task, model, and dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Load model aliases (same as in run_task.sh)
declare -A MODELS
MODELS["apertus-instruct"]="swiss-ai/Apertus-8B-Instruct-2509"
MODELS["apertus-base"]="swiss-ai/Apertus-8B-2509"
MODELS["llama-instruct"]="meta-llama/Llama-3.1-8B-Instruct"
MODELS["llama-base"]="meta-llama/Llama-3.1-8B"

# Available tasks
TASKS=("cache" "postprocess" "train_probes" "run_probes" "cross_dataset_probes")

# Available datasets (same as in run_task.sh)
DATASETS=(
    "sms_spam"
    "mmlu_pro_natural_science"
    "mmlu_high_school"
    "mmlu_professional"
    "ARC-Easy"
    "ARC-Challenge"
    "sujet_finance_yesno_5k"
)

# Function to list available models
list_models() {
    echo "Available model aliases:"
    echo ""
    for alias in "${!MODELS[@]}"; do
        printf "  %-20s -> %s\n" "$alias" "${MODELS[$alias]}"
    done | sort
    echo ""
    echo "Usage: --model <alias> or --model_name <full_name>"
}

# Function to list available datasets
list_datasets() {
    echo "Available datasets:"
    echo ""
    for dataset in "${DATASETS[@]}"; do
        echo "  $dataset"
    done | sort
    echo ""
}

# Function to show default values for a task
show_defaults() {
    local task="$1"
    
    case "$task" in
        cache)
            echo "Default values for 'cache' task:"
            echo ""
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
            echo "  --save-dir: \$SCRATCH/mera-runs"
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
            echo "  --save-dir: \$SCRATCH/mera-runs"
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
        *)
            echo "Available tasks: ${TASKS[*]}"
            echo ""
            echo "Usage: submit.sh --show-defaults <task>"
            echo "   or: submit.sh <task> --show-defaults"
            echo ""
            echo "To see defaults for all tasks, run:"
            for t in "${TASKS[@]}"; do
                echo "  submit.sh --show-defaults $t"
            done
            ;;
    esac
}

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

# Parse arguments
TASK=""
MODEL_ALIAS=""
MODEL_NAME=""
DATASET_NAME=""
USER_DATASETS=()  # For run_probes (multiple datasets provided by user)
TRAIN_DATASET=""  # For cross_dataset_probes
TEST_DATASET=""   # For cross_dataset_probes
SBATCH_ARGS=()
SCRIPT_ARGS=()
SHOW_STATUS=false
SHOW_HELP=false
RUN_LOCAL=false
SHOW_DEFAULTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_ALIAS="$2"
            SCRIPT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            SCRIPT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            SCRIPT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --datasets)
            # Handle multiple datasets - consume all following non-flag arguments
            shift
            USER_DATASETS=()
            SCRIPT_ARGS+=("--datasets")
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                USER_DATASETS+=("$1")
                SCRIPT_ARGS+=("$1")
                shift
            done
            ;;
        --train-dataset)
            TRAIN_DATASET="$2"
            SCRIPT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --test-dataset)
            TEST_DATASET="$2"
            SCRIPT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --time|-t)
            SBATCH_ARGS+=("--time=$2")
            shift 2
            ;;
        --gpus|-g)
            SBATCH_ARGS+=("--gpus-per-node=$2")
            shift 2
            ;;
        --account|-A)
            SBATCH_ARGS+=("--account=$2")
            shift 2
            ;;
        --partition|-p)
            SBATCH_ARGS+=("--partition=$2")
            shift 2
            ;;
        --status|--check|-s)
            SHOW_STATUS=true
            shift
            ;;
        --local|--run-local)
            RUN_LOCAL=true
            shift
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        --list-models)
            list_models
            exit 0
            ;;
        --list-datasets)
            list_datasets
            exit 0
            ;;
        --show-defaults|--list-defaults)
            SHOW_DEFAULTS=true
            shift
            ;;
        cache|postprocess|train_probes|run_probes|cross_dataset_probes)
            TASK="$1"
            SCRIPT_ARGS+=("$1")
            shift
            ;;
        *)
            SCRIPT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Show defaults
if [ "$SHOW_DEFAULTS" = true ]; then
    if [ -n "$TASK" ]; then
        show_defaults "$TASK"
    else
        echo "Error: Task is required to show defaults"
        echo "Usage: submit.sh --show-defaults <task>"
        echo "   or: submit.sh <task> --show-defaults"
        echo ""
        echo "Available tasks: ${TASKS[*]}"
        echo ""
        echo "Examples:"
        echo "  submit.sh --show-defaults run_probes"
        echo "  submit.sh run_probes --show-defaults"
    fi
    exit 0
fi

# Show help
if [ "$SHOW_HELP" = true ]; then
    cat << EOF
Usage: submit.sh <task> --model <alias> --dataset_name <dataset> [options]

Submit jobs to SLURM or run locally with automatic parameter validation.

Commands:
  --list-models             List available model aliases
  --list-datasets           List available datasets
  --show-defaults           Show default values for a task
  --status, -s              Show your job status

Arguments:
  <task>                    Task to run: cache, postprocess, train_probes, or run_probes
  --model <alias>           Model alias (use --list-models to see options)
  --dataset_name <dataset>  Dataset name (required for cache/postprocess/train_probes)
  --datasets <name1> [name2] Dataset names for run_probes (space-separated, can be multiple)
  
Execution modes:
  --local, --run-local      Run locally instead of submitting with sbatch

SBATCH options (only used when not using --local):
  --time, -t <time>         Time limit (e.g., 06:00:00)
  --gpus, -g <num>          Number of GPUs (default: 1)
  --account, -A <account>   Account name (default: infra01)
  --partition, -p <part>    Partition (default: normal)

Other options:
  --help, -h                Show this help message

Examples:
  # Submit with sbatch (default):
  ./submit.sh cache --model apertus-instruct --dataset_name "mmlu_professional"
  ./submit.sh postprocess --model apertus-instruct --dataset_name "ARC-Challenge" --time 12:00:00
  ./submit.sh run_probes --model apertus-instruct --datasets sms_spam mmlu_professional
  
  # Run locally:
  ./submit.sh --local cache --model apertus-instruct --dataset_name "mmlu_professional"
  ./submit.sh --local train_probes --model apertus-instruct --dataset_name "sms_spam"
  ./submit.sh --local run_probes --model apertus-instruct --datasets sms_spam mmlu_professional
  
  # List options:
  ./submit.sh --list-models
  ./submit.sh --list-datasets
  ./submit.sh --show-defaults run_probes

The script automatically creates descriptive job names like:
  cache_Apertus-8B-Instruct-2509_mmlu_professional
  postprocess_Apertus-8B-Instruct-2509_ARC-Challenge
EOF
    exit 0
fi

# Show status
if [ "$SHOW_STATUS" = true ] && [ -z "$TASK" ]; then
    echo "Your SLURM jobs:"
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R"
    exit 0
fi

# Validate required arguments
if [ -z "$TASK" ]; then
    echo "Error: Task is required"
    echo "Usage: submit.sh <task> --model <alias> --dataset_name <dataset> [options]"
    echo "Available tasks: ${TASKS[*]}"
    echo "Run './submit.sh --help' for more information"
    exit 1
fi

# Validate task
if [[ ! " ${TASKS[*]} " =~ " ${TASK} " ]]; then
    echo "Error: Invalid task '$TASK'"
    echo "Available tasks: ${TASKS[*]}"
    exit 1
fi

# Validate required arguments based on task
if [ "$TASK" == "run_probes" ]; then
    # For run_probes, need datasets (can be multiple)
    if [ ${#USER_DATASETS[@]} -eq 0 ] && [ -z "$DATASET_NAME" ]; then
        echo "Error: --datasets <name1> [name2] ... is required for run_probes"
        echo "Usage: submit.sh run_probes --model <alias> --datasets <name1> [name2] ... [options]"
        exit 1
    fi
    # If DATASET_NAME is set but USER_DATASETS is empty, use DATASET_NAME
    if [ ${#USER_DATASETS[@]} -eq 0 ] && [ -n "$DATASET_NAME" ]; then
        USER_DATASETS=("$DATASET_NAME")
        # Add to SCRIPT_ARGS if not already there
        if [[ ! " ${SCRIPT_ARGS[@]} " =~ " --datasets " ]]; then
            SCRIPT_ARGS+=("--datasets" "$DATASET_NAME")
        fi
    fi
    # Validate all datasets
    for dataset in "${USER_DATASETS[@]}"; do
        if ! is_valid_dataset "$dataset"; then
            echo "Error: Invalid dataset '$dataset'"
            echo "Available datasets:"
            list_datasets
            exit 1
        fi
    done
elif [ "$TASK" == "cross_dataset_probes" ]; then
    # For cross_dataset_probes, need train_dataset and test_dataset
    if [ -z "$TRAIN_DATASET" ]; then
        echo "Error: --train-dataset <name> is required for cross_dataset_probes"
        echo "Usage: submit.sh cross_dataset_probes --model <alias> --train-dataset <name> --test-dataset <name> [options]"
        exit 1
    fi
    if [ -z "$TEST_DATASET" ]; then
        echo "Error: --test-dataset <name> is required for cross_dataset_probes"
        echo "Usage: submit.sh cross_dataset_probes --model <alias> --train-dataset <name> --test-dataset <name> [options]"
        exit 1
    fi
    # Validate datasets
    if ! is_valid_dataset "$TRAIN_DATASET"; then
        echo "Error: Invalid train dataset '$TRAIN_DATASET'"
        echo "Available datasets:"
        list_datasets
        exit 1
    fi
    if ! is_valid_dataset "$TEST_DATASET"; then
        echo "Error: Invalid test dataset '$TEST_DATASET'"
        echo "Available datasets:"
        list_datasets
        exit 1
    fi
else
    # For other tasks, need single dataset
    if [ -z "$DATASET_NAME" ]; then
        echo "Error: --dataset_name is required"
        echo "Usage: submit.sh <task> --model <alias> --dataset_name <dataset> [options]"
        exit 1
    fi
    # Validate dataset
    if ! is_valid_dataset "$DATASET_NAME"; then
        echo "Error: Invalid dataset '$DATASET_NAME'"
        echo "Available datasets:"
        for dataset in "${DATASETS[@]}"; do
            echo "  $dataset"
        done | sort
        exit 1
    fi
fi

# Get model name from alias if needed
if [ -n "$MODEL_ALIAS" ] && [ -z "$MODEL_NAME" ]; then
    if [ -n "${MODELS[$MODEL_ALIAS]:-}" ]; then
        MODEL_NAME="${MODELS[$MODEL_ALIAS]}"
    else
        echo "Error: Unknown model alias '$MODEL_ALIAS'"
        echo "Available model aliases:"
        for alias in "${!MODELS[@]}"; do
            printf "  %-20s -> %s\n" "$alias" "${MODELS[$alias]}"
        done | sort
        echo ""
        echo "Use --model_name <full_name> to use a model not in the alias list"
        exit 1
    fi
fi

if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model <alias> or --model_name <full_name> is required"
    echo ""
    echo "Available model aliases:"
    for alias in "${!MODELS[@]}"; do
        printf "  %-20s -> %s\n" "$alias" "${MODELS[$alias]}"
    done | sort
    echo ""
    echo "Usage: submit.sh <task> --model <alias> --dataset_name <dataset> [options]"
    exit 1
fi

# Create descriptive job name
MODEL_SHORT=$(basename "$MODEL_NAME")
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
JOB_NAME="${TASK}_${MODEL_SHORT}_${DATASET_CLEAN}"

# Truncate if too long (SLURM limit is 100 chars)
if [ ${#JOB_NAME} -gt 100 ]; then
    JOB_NAME="${JOB_NAME:0:100}"
fi

# Set default time to 6 hours if not specified
TIME_SPECIFIED=false
for arg in "${SBATCH_ARGS[@]}"; do
    if [[ "$arg" == --time=* ]]; then
        TIME_SPECIFIED=true
        break
    fi
done
if [ "$TIME_SPECIFIED" = false ] && [ "$RUN_LOCAL" = false ]; then
    SBATCH_ARGS+=("--time=06:00:00")
fi

# Execute locally or with sbatch
if [ "$RUN_LOCAL" = true ]; then
    # Run locally
    echo "========================================"
    echo "Running locally..."
    echo "  Task:    $TASK"
    echo "  Model:   $MODEL_NAME"
    if [ "$TASK" == "run_probes" ]; then
        echo "  Datasets: ${USER_DATASETS[*]}"
    elif [ "$TASK" == "cross_dataset_probes" ]; then
        echo "  Train dataset: $TRAIN_DATASET"
        echo "  Test dataset: $TEST_DATASET"
    else
        echo "  Dataset: $DATASET_NAME"
    fi
    echo "========================================"
    echo ""
    
    # Run the task script directly (bypass sbatch)
    bash "$SCRIPT_DIR/run_task.sh" "${SCRIPT_ARGS[@]}"
else
    # Submit with sbatch
    echo "========================================"
    echo "Submitting job to SLURM..."
    echo "  Task:    $TASK"
    echo "  Model:   $MODEL_NAME"
    if [ "$TASK" == "run_probes" ]; then
        echo "  Datasets: ${USER_DATASETS[*]}"
    elif [ "$TASK" == "cross_dataset_probes" ]; then
        echo "  Train dataset: $TRAIN_DATASET"
        echo "  Test dataset: $TEST_DATASET"
    else
        echo "  Dataset: $DATASET_NAME"
    fi
    echo "  Job name: $JOB_NAME"
    echo "========================================"
    
    JOB_ID=$(sbatch --job-name="$JOB_NAME" --chdir="$SCRIPT_DIR" "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_task.sh" "${SCRIPT_ARGS[@]}" 2>&1 | grep -oP '\d+' | tail -1)
    
    if [ -n "$JOB_ID" ]; then
        echo "Job submitted successfully!"
        echo "  Job ID: $JOB_ID"
        echo "  Job name: $JOB_NAME"
        echo ""
        
        if [ "$SHOW_STATUS" = true ]; then
            sleep 1
            echo "Job status:"
            squeue -j "$JOB_ID" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "  (Job may not appear in queue yet)"
        else
            echo "Use 'squeue -u $USER' to check job status"
            echo "Use 'tail -f logs/${JOB_NAME}_${JOB_ID}.out' to follow the log"
        fi
    else
        echo "Error: Failed to submit job"
        exit 1
    fi
fi

