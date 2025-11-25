#!/usr/bin/env bash
# Script to copy all .pkl files from all datasets and models to a specific folder
#
# Usage:
#   ./copy_df_probes.sh <destination_folder> [options]
#
# Options:
#   --source <path>        Source directory (default: $SCRATCH/mera-runs/)
#   --preserve-structure   Preserve directory structure in destination
#   --dry-run              Show what would be copied without actually copying
#   --help                 Show this help message

set -euo pipefail

# Default values
PRESERVE_STRUCTURE=true
DRY_RUN=false
SOURCE_DIR=""

# Parse arguments
DEST_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE_DIR="$2"
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
Usage: $0 <destination_folder> [options]

Copy all .pkl files from all datasets and models to a specific folder.

Arguments:
  <destination_folder>    Destination directory where files will be copied

Options:
  --source <path>         Source directory to search (default: \$SCRATCH/mera-runs/)
  --preserve-structure    Preserve directory structure in destination
                          (default: all files copied to flat structure)
  --dry-run               Show what would be copied without actually copying
  --help, -h              Show this help message

Examples:
  # Copy all .pkl files to a folder (flat structure):
  ./copy_df_probes.sh /path/to/destination

  # Copy with preserved directory structure:
  ./copy_df_probes.sh /path/to/destination --preserve-structure

  # Use custom source directory:
  ./copy_df_probes.sh /path/to/destination --source /custom/path/mera-runs

  # Dry run to see what would be copied:
  ./copy_df_probes.sh /path/to/destination --dry-run
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

# Validate destination directory
if [ -z "$DEST_DIR" ]; then
    echo "Error: Destination directory is required"
    echo "Usage: $0 <destination_folder> [options]"
    echo "Use --help for more information"
    exit 1
fi

# Remove trailing slash from destination directory to avoid double slashes
DEST_DIR="${DEST_DIR%/}"

# Set default source directory if not provided
if [ -z "$SOURCE_DIR" ]; then
    if [ -z "${SCRATCH:-}" ]; then
        SCRATCH="/iopsstor/scratch/cscs/$USER"
    fi
    SOURCE_DIR="$SCRATCH/mera-runs"
fi

# Remove trailing slash from source directory to avoid double slashes
SOURCE_DIR="${SOURCE_DIR%/}"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $SOURCE_DIR"
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
echo "Copying all .pkl files"
echo "========================================"
echo "Source:      $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo "Preserve structure: $PRESERVE_STRUCTURE"
echo "Dry run:    $DRY_RUN"
echo ""

# Find all .pkl files, excluding those in activations folders
FILES=()
while IFS= read -r -d '' file; do
    FILES+=("$file")
done < <(find "$SOURCE_DIR" -type d -name "activations" -prune -o -type f -name "*.pkl" -print0 2>/dev/null || true)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .pkl files found in $SOURCE_DIR"
    exit 0
fi

echo "Found ${#FILES[@]} file(s) to copy"
echo ""

# Copy files
COPIED=0
SKIPPED=0
FAILED=0

for file in "${FILES[@]}"; do
    # Get relative path from source directory
    rel_path="${file#$SOURCE_DIR/}"
    
    if [ "$PRESERVE_STRUCTURE" = true ]; then
        # Preserve directory structure
        dest_file="${DEST_DIR}/${rel_path}"
        dest_dir="$(dirname "$dest_file")"
    else
        # Flatten structure: use filename with dataset and model info
        filename="$(basename "$file")"
        # Try to extract dataset and model from path
        # Path format: .../dataset_name/model_name/*.pkl
        if [[ "$rel_path" =~ ^([^/]+)/([^/]+)/.+\.pkl$ ]]; then
            dataset="${BASH_REMATCH[1]}"
            model="${BASH_REMATCH[2]}"
            # Remove extension, add dataset and model, then add extension back
            base_name="${filename%.pkl}"
            dest_file="$DEST_DIR/${dataset}_${model}_${base_name}.pkl"
        else
            # Fallback: just use filename
            dest_file="$DEST_DIR/$filename"
        fi
        dest_dir="$DEST_DIR"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "Would copy: $file"
        echo "         -> $dest_file"
    else
        # Create destination directory if needed
        mkdir -p "$dest_dir"
        
        # Check if file already exists
        if [ -f "$dest_file" ]; then
            # Compare file sizes or modification times to decide if we should skip
            if [ "$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)" = "$(stat -f%z "$dest_file" 2>/dev/null || stat -c%s "$dest_file" 2>/dev/null)" ]; then
                echo "Skipping (already exists): $rel_path"
                ((SKIPPED++)) || true
                continue
            fi
        fi
        
        # Copy the file
        if cp "$file" "$dest_file" 2>/dev/null; then
            echo "Copied: $rel_path -> $(basename "$dest_file")"
            ((COPIED++)) || true
        else
            echo "Error: Failed to copy $rel_path"
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

