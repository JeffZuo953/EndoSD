#!/usr/bin/env python3
"""
Script to clean old cache files and regenerate with correct depth scaling.

Usage:
    python regenerate_cache.py /data/ziyi/multitask/data/SCARED/cache
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def clean_cache_directory(cache_dir: str, dry_run: bool = False):
    """
    Clean cache directory by removing all .pt files and cache lists.

    Args:
        cache_dir: Path to cache directory
        dry_run: If True, only show what would be deleted without actually deleting
    """
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    print(f"Scanning cache directory: {cache_dir}")

    # Find all .pt files
    pt_files = list(Path(cache_dir).rglob("*.pt"))

    # Find cache list files
    cache_lists = list(Path(cache_dir).glob("*_cache.txt"))

    # Find report files
    report_files = list(Path(cache_dir).glob("depth_statistics_report*"))
    report_files.extend(Path(cache_dir).glob("*_depth_analysis*"))

    total_files = len(pt_files) + len(cache_lists) + len(report_files)

    if total_files == 0:
        print("No cache files found to clean.")
        return

    print(f"\nFound files to clean:")
    print(f"  - {len(pt_files)} .pt cache files")
    print(f"  - {len(cache_lists)} cache list files")
    print(f"  - {len(report_files)} report files")
    print(f"  Total: {total_files} files")

    if dry_run:
        print("\n[DRY RUN] Would delete these files:")
        for f in list(pt_files)[:5] + list(cache_lists) + list(report_files):
            print(f"  - {f}")
        if len(pt_files) > 5:
            print(f"  - ... and {len(pt_files) - 5} more .pt files")
        print("\nRun without --dry-run to actually delete.")
        return

    # Ask for confirmation
    response = input(f"\nAre you sure you want to delete {total_files} files? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Delete files
    print("\nDeleting files...")
    deleted = 0

    for pt_file in pt_files:
        try:
            pt_file.unlink()
            deleted += 1
        except Exception as e:
            print(f"Error deleting {pt_file}: {e}")

    for cache_list in cache_lists:
        try:
            cache_list.unlink()
            deleted += 1
        except Exception as e:
            print(f"Error deleting {cache_list}: {e}")

    for report_file in report_files:
        try:
            report_file.unlink()
            deleted += 1
        except Exception as e:
            print(f"Error deleting {report_file}: {e}")

    print(f"\nDeleted {deleted} files successfully.")

    # Remove empty directories
    print("\nRemoving empty directories...")
    removed_dirs = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir, topdown=False):
        if dirpath == cache_dir:
            continue
        if not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
                removed_dirs += 1
            except Exception as e:
                print(f"Error removing directory {dirpath}: {e}")

    if removed_dirs > 0:
        print(f"Removed {removed_dirs} empty directories.")

    print("\nCache cleaning complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Clean old cache files before regenerating with correct depth scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be deleted
  python regenerate_cache.py /data/ziyi/multitask/data/SCARED/cache --dry-run

  # Actually clean the cache
  python regenerate_cache.py /data/ziyi/multitask/data/SCARED/cache

After cleaning, regenerate cache with:
  python -m dataset.cache_utils_data
        """
    )
    parser.add_argument(
        'cache_dir',
        type=str,
        help='Path to cache directory to clean'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    try:
        clean_cache_directory(args.cache_dir, args.dry_run)

        if not args.dry_run:
            print("\n" + "="*80)
            print("Next steps:")
            print("="*80)
            print("1. Verify the cache directory is clean")
            print("2. Regenerate cache with corrected depth scaling:")
            print("   cd /data/ziyi/multitask/code/DepthAnythingV2/multitask_moe_lora")
            print("   python -m dataset.cache_utils_data")
            print("3. Verify the new statistics are correct:")
            print("   python -m dataset.analyze_cached_depth /data/ziyi/multitask/data/SCARED/cache/train_cache.txt")
            print("="*80)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
