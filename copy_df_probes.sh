#!/usr/bin/env bash
# Script to copy all .pkl and .jsonl files from scratch (mera-runs and processed_datasets) to /capstor/store/cscs/swissai/infra01/apertus_probes
#
# Usage:
#   ./copy_df_probes.sh [destination_folder] [options]
#
# Options:
#   --source <path>        Source base directory (default: \$SCRATCH)
#                         Will copy .pkl and .jsonl files from both source/mera-runs/ and source/processed_datasets/
#   --preserve-structure   Preserve directory structure in destination
#   --dry-run              Show what would be copied without actually copying
#   --help                 Show this help message
#
# Default source: \$SCRATCH (typically /iopsstor/scratch/cscs/\$USER)
# Default destination: /capstor/store/cscs/swissai/infra01/apertus_probes

set -euo pipefail

# Default values
PRESERVE_STRUCTURE=true
DRY_RUN=false
SOURCE_BASE_DIR=""
DEFAULT_DEST_DIR="/capstor/store/cscs/swissai/infra01/apertus_probes"

# Parse arguments
DEST_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE_BASE_DIR="$2"
            shift 2
            ;;
        --preserve-structure)
            PRESERVE_STRUCTURE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: $0 [destination_folder] [options]

Copy all .pkl and .jsonl files from scratch (mera-runs and processed_datasets) to a destination folder.

Arguments:
  [destination_folder]    Optional destination directory (default: /capstor/store/cscs/swissai/infra01/apertus_probes)

Options:
  --source <path>         Source base directory (default: \$SCRATCH)
                         Will copy .pkl and .jsonl files from both source/mera-runs/ and source/processed_datasets/
                         Default: \$SCRATCH (typically /iopsstor/scratch/cscs/\$USER)
  --preserve-structure    Preserve directory structure in destination
                          (default: all files copied to flat structure)
  --dry-run               Show what would be copied without actually copying
  --help, -h              Show this help message

Examples:
  # Copy all .pkl files to default destination (flat structure):
  ./copy_df_probes.sh

  # Copy to custom destination:
  ./copy_df_probes.sh /path/to/destination

  # Copy with preserved directory structure:
  ./copy_df_probes.sh --preserve-structure

  # Use custom source base directory:
  ./copy_df_probes.sh --source /custom/path
  # (default source is \$SCRATCH, which contains mera-runs/ and processed_datasets/)

  # Dry run to see what would be copied:
  ./copy_df_probes.sh --dry-run
EOF
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [ -z "$DEST_DIR" ]; then
                DEST_DIR="$1"
            else
                echo "Error: Multiple destination directories specified"
                exit 1
            fi
            shift
            ;;
    esac
done

# Use default destination if not provided
if [ -z "$DEST_DIR" ]; then
    DEST_DIR="$DEFAULT_DEST_DIR"
    echo "Using default destination: $DEST_DIR"
fi

# Remove trailing slash from destination directory to avoid double slashes
DEST_DIR="${DEST_DIR%/}"

# Set default source base directory if not provided (use SCRATCH)
if [ -z "$SOURCE_BASE_DIR" ]; then
    if [ -z "${SCRATCH:-}" ]; then
        SCRATCH="/iopsstor/scratch/cscs/$USER"
    fi
    SOURCE_BASE_DIR="$SCRATCH"
fi

# Remove trailing slash from source base directory to avoid double slashes
SOURCE_BASE_DIR="${SOURCE_BASE_DIR%/}"

# Define source directories to copy from (in scratch)
SOURCE_MERA_RUNS="${SOURCE_BASE_DIR}/mera-runs"
SOURCE_PROCESSED="${SOURCE_BASE_DIR}/processed_datasets"

# Check if source base directory exists
if [ ! -d "$SOURCE_BASE_DIR" ]; then
    echo "Error: Source base directory does not exist: $SOURCE_BASE_DIR"
    exit 1
fi

# Check which source directories exist
SOURCES_TO_COPY=()
if [ -d "$SOURCE_MERA_RUNS" ]; then
    SOURCES_TO_COPY+=("$SOURCE_MERA_RUNS:mera-runs")
fi
if [ -d "$SOURCE_PROCESSED" ]; then
    SOURCES_TO_COPY+=("$SOURCE_PROCESSED:processed_datasets")
fi

if [ ${#SOURCES_TO_COPY[@]} -eq 0 ]; then
    echo "Error: Neither $SOURCE_MERA_RUNS nor $SOURCE_PROCESSED exist"
    exit 1
fi

# Create destination directory if it doesn't exist (unless dry-run)
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$DEST_DIR"
    if [ "$PRESERVE_STRUCTURE" = true ]; then
        # Create the base structure
        mkdir -p "$DEST_DIR"
    fi
fi

echo "========================================"
echo "Copying all .pkl and .jsonl files"
echo "========================================"
echo "Source base: $SOURCE_BASE_DIR"
echo "Destination: $DEST_DIR"
echo "Preserve structure: $PRESERVE_STRUCTURE"
echo "Dry run:    $DRY_RUN"
echo ""

# Find all .pkl and .jsonl files from all source directories, excluding those in activations folders
FILES=()
for source_info in "${SOURCES_TO_COPY[@]}"; do
    IFS=':' read -r source_dir source_name <<< "$source_info"
    echo "Scanning: $source_dir"
    while IFS= read -r -d '' file; do
        # Store with source name prefix for later path reconstruction
        FILES+=("${source_name}:${file}")
    done < <(find "$source_dir" -type d -name "activations" -prune -o \( -type f \( -name "*.pkl" -o -name "*.jsonl" \) -print0 \) 2>/dev/null || true)
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .pkl or .jsonl files found in $SOURCE_BASE_DIR"
    exit 0
fi

echo "Found ${#FILES[@]} file(s) to copy"
echo ""

# Copy files
COPIED=0
SKIPPED=0
FAILED=0

for file_info in "${FILES[@]}"; do
    # Split source name and file path
    IFS=':' read -r source_name file <<< "$file_info"
    
    # Determine the base source directory
    if [ "$source_name" = "mera-runs" ]; then
        SOURCE_DIR="$SOURCE_MERA_RUNS"
    else
        SOURCE_DIR="$SOURCE_PROCESSED"
    fi
    
    # Get relative path from source directory
    rel_path="${file#$SOURCE_DIR/}"
    
    if [ "$PRESERVE_STRUCTURE" = true ]; then
        # Preserve directory structure, including source subdirectory name
        dest_file="${DEST_DIR}/${source_name}/${rel_path}"
        dest_dir="$(dirname "$dest_file")"
    else
        # Flatten structure: use filename with dataset and model info
        filename="$(basename "$file")"
        # Try to extract dataset and model from path
        # Path format: .../dataset_name/model_name/*.pkl or .../dataset_name/*.jsonl
        if [[ "$rel_path" =~ ^([^/]+)/([^/]+)/.+\.(pkl|jsonl)$ ]]; then
            dataset="${BASH_REMATCH[1]}"
            model="${BASH_REMATCH[2]}"
            # Remove extension, add source name, dataset and model, then add extension back
            extension="${filename##*.}"
            base_name="${filename%.*}"
            dest_file="$DEST_DIR/${source_name}_${dataset}_${model}_${base_name}.${extension}"
        elif [[ "$rel_path" =~ ^([^/]+)/.+\.(pkl|jsonl)$ ]]; then
            # Path format: .../dataset_name/*.jsonl (no model subdirectory)
            dataset="${BASH_REMATCH[1]}"
            extension="${filename##*.}"
            base_name="${filename%.*}"
            dest_file="$DEST_DIR/${source_name}_${dataset}_${base_name}.${extension}"
        else
            # Fallback: use source name and filename
            dest_file="$DEST_DIR/${source_name}_${filename}"
        fi
        dest_dir="$DEST_DIR"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "Would copy: ${source_name}/${rel_path}"
        echo "         -> $dest_file"
    else
        # Create destination directory if needed
        mkdir -p "$dest_dir"
        
        # Check if file already exists
        if [ -f "$dest_file" ]; then
            # Compare file sizes or modification times to decide if we should skip
            if [ "$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)" = "$(stat -f%z "$dest_file" 2>/dev/null || stat -c%s "$dest_file" 2>/dev/null)" ]; then
                echo "Skipping (already exists): ${source_name}/${rel_path}"
                ((SKIPPED++)) || true
                continue
            fi
        fi
        
        # Copy the file
        if cp "$file" "$dest_file" 2>/dev/null; then
            echo "Copied: ${source_name}/${rel_path} -> $(basename "$dest_file")"
            ((COPIED++)) || true
        else
            echo "Error: Failed to copy ${source_name}/${rel_path}"
            ((FAILED++)) || true
        fi
    fi
done

echo ""
echo "========================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. ${#FILES[@]} file(s) would be copied."
else
    echo "Copy complete!"
    echo "  Copied:   $COPIED"
    echo "  Skipped:  $SKIPPED"
    echo "  Failed:   $FAILED"
    echo "  Total:    ${#FILES[@]}"
fi
echo "========================================"

