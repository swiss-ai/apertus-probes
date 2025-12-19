#!/usr/bin/env python3
"""
Reorganize steering output folders by moving models from steered_on folders into their base dataset folders.

For example:
- mmlu_high_school_steered_on_mmlu_high_school/Apertus-8B-2509
  -> mmlu_high_school/Apertus-8B-2509 (or merge if exists)
"""

import shutil
from pathlib import Path
from typing import Optional, List

ROOT = Path("/capstor/store/cscs/swissai/infra01/apertus_probes/mera-runs/steering_outputs")


def extract_target_dataset(folder_name: str) -> Optional[str]:
    """Extract target dataset name from folder name like 'dataset_steered_on_target'."""
    if "_steered_on_" not in folder_name:
        return None
    
    # Skip mixture datasets (those with +)
    if "+" in folder_name:
        return None
    
    parts = folder_name.split("_steered_on_")
    if len(parts) != 2:
        return None
    
    return parts[1]  # Return the target dataset name


def merge_directories(source: Path, target: Path, dry_run: bool = True) -> List[str]:
    """
    Merge contents from source directory into target directory.
    Returns list of actions taken.
    """
    actions = []
    
    if not source.exists():
        actions.append(f"  [SKIP] Source does not exist: {source}")
        return actions
    
    if not target.exists():
        if dry_run:
            actions.append(f"  [CREATE] Would create: {target}")
        else:
            target.mkdir(parents=True, exist_ok=True)
            actions.append(f"  [CREATE] Created: {target}")
    
    # Copy/merge all items from source to target
    for item in source.iterdir():
        target_item = target / item.name
        
        if item.is_dir():
            if target_item.exists():
                # Directory exists, merge recursively
                if dry_run:
                    actions.append(f"  [MERGE] Would merge {item.name}/ into existing {target_item}/")
                else:
                    actions.append(f"  [MERGE] Merging {item.name}/ into {target_item}/")
                    # Recursively merge subdirectories
                    for subitem in item.rglob("*"):
                        rel_path = subitem.relative_to(item)
                        target_subitem = target_item / rel_path
                        if subitem.is_dir():
                            target_subitem.mkdir(parents=True, exist_ok=True)
                        else:
                            target_subitem.parent.mkdir(parents=True, exist_ok=True)
                            if target_subitem.exists():
                                actions.append(f"    [OVERWRITE] {rel_path}")
                            shutil.copy2(subitem, target_subitem)
            else:
                # Directory doesn't exist, copy it
                if dry_run:
                    actions.append(f"  [COPY] Would copy directory {item.name}/ to {target_item}/")
                else:
                    # Manual recursive copy for Python < 3.8 compatibility
                    target_item.mkdir(parents=True, exist_ok=True)
                    for subitem in item.rglob("*"):
                        rel_path = subitem.relative_to(item)
                        target_subitem = target_item / rel_path
                        if subitem.is_dir():
                            target_subitem.mkdir(parents=True, exist_ok=True)
                        else:
                            target_subitem.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(subitem, target_subitem)
                    actions.append(f"  [COPY] Copied directory {item.name}/ to {target_item}/")
        else:
            # It's a file
            if target_item.exists():
                if dry_run:
                    actions.append(f"  [OVERWRITE] Would overwrite {item.name}")
                else:
                    shutil.copy2(item, target_item)
                    actions.append(f"  [OVERWRITE] Overwrote {item.name}")
            else:
                if dry_run:
                    actions.append(f"  [COPY] Would copy file {item.name} to {target_item}")
                else:
                    target_item.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_item)
                    actions.append(f"  [COPY] Copied file {item.name} to {target_item}")
    
    return actions


def reorganize_folders(dry_run: bool = True, delete_source: bool = False):
    """
    Reorganize steering folders by moving models from steered_on folders to base dataset folders.
    
    Args:
        dry_run: If True, only show what would be done without making changes
        delete_source: If True, delete source folders after successful copy (only if not dry_run)
    """
    if not ROOT.exists():
        print(f"Error: Root directory does not exist: {ROOT}")
        return
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Reorganizing steering folders in: {ROOT}")
    print("=" * 80)
    
    # Find all steered_on folders (excluding mixtures)
    steered_folders = []
    for item in ROOT.iterdir():
        if not item.is_dir():
            continue
        
        target_dataset = extract_target_dataset(item.name)
        if target_dataset:
            steered_folders.append((item, target_dataset))
    
    print(f"Found {len(steered_folders)} steered_on folders to process:\n")
    
    all_actions = []
    for source_folder, target_dataset_name in steered_folders:
        target_folder = ROOT / target_dataset_name
        
        print(f"\nProcessing: {source_folder.name}")
        print(f"  Target: {target_folder.name}")
        
        if not target_folder.exists():
            print(f"  [WARN] Target folder does not exist: {target_folder}")
            print(f"  [SKIP] Skipping this folder")
            continue
        
        # Process each model in the source folder
        for model_dir in source_folder.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            target_model_dir = target_folder / model_name
            
            print(f"\n  Model: {model_name}")
            actions = merge_directories(model_dir, target_model_dir, dry_run=dry_run)
            all_actions.extend(actions)
            for action in actions:
                print(action)
        
        # Optionally delete source folder after successful copy
        if not dry_run and delete_source:
            # Check if source folder is now empty or only has empty model dirs
            has_content = False
            for item in source_folder.iterdir():
                if item.is_file() or (item.is_dir() and any(item.rglob("*"))):
                    has_content = True
                    break
            
            if not has_content:
                print(f"\n  [DELETE] Source folder is empty, would delete: {source_folder}")
                # Uncomment to actually delete:
                # shutil.rmtree(source_folder)
                # print(f"  [DELETE] Deleted: {source_folder}")
    
    print("\n" + "=" * 80)
    print(f"Summary: Processed {len(steered_folders)} folders")
    if dry_run:
        print("\nThis was a DRY RUN. No files were modified.")
        print("Run with dry_run=False to perform the actual reorganization.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reorganize steering output folders by moving models from steered_on folders into base dataset folders"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the reorganization (default is dry-run)"
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source folders after successful copy (only if --execute is set)"
    )
    
    args = parser.parse_args()
    
    reorganize_folders(dry_run=not args.execute, delete_source=args.delete_source)
