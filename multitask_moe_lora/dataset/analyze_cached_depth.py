#!/usr/bin/env python3
"""
Script to analyze depth distribution from cached .pt files.
Reads a txt file containing paths to .pt files and generates statistics.

Usage:
    python analyze_cached_depth.py <filelist.txt> [--output <report_path>]

Example:
    python analyze_cached_depth.py /data/ziyi/multitask/data/SCARED/cache/train_cache.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm


class CachedDepthAnalyzer:
    """Analyzes depth distribution from cached .pt files."""

    def __init__(self):
        self.depth_values = []
        self.valid_pixel_counts = []
        self.total_pixel_counts = []
        self.min_depths = []
        self.max_depths = []
        self.mean_depths = []
        self.file_paths = []
        self.failed_files = []

    def process_pt_file(self, pt_path: str) -> bool:
        """
        Process a single .pt file and extract depth statistics.

        Args:
            pt_path: Path to the .pt file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the cached data
            data = torch.load(pt_path, map_location='cpu')

            # Check if depth exists
            if 'depth' not in data:
                print(f"Warning: No 'depth' key in {pt_path}")
                return False

            depth = data['depth']

            # Convert to numpy if it's a tensor
            if isinstance(depth, torch.Tensor):
                depth_np = depth.numpy()
            else:
                depth_np = np.array(depth)

            # Flatten depth data
            depth_flat = depth_np.flatten()

            # Calculate valid pixels (non-zero values)
            valid_mask = depth_flat > 0
            valid_depths = depth_flat[valid_mask]

            # Record statistics
            self.total_pixel_counts.append(len(depth_flat))
            self.valid_pixel_counts.append(len(valid_depths))

            if len(valid_depths) > 0:
                min_val = float(np.min(valid_depths))
                max_val = float(np.max(valid_depths))
                mean_val = float(np.mean(valid_depths))

                self.min_depths.append(min_val)
                self.max_depths.append(max_val)
                self.mean_depths.append(mean_val)

                # Sample depth values for global statistics
                # To avoid memory issues, sample at most 10000 values per file
                if len(valid_depths) > 10000:
                    sample_indices = np.random.choice(len(valid_depths), 10000, replace=False)
                    sampled_depths = valid_depths[sample_indices]
                    self.depth_values.extend(sampled_depths.tolist())
                else:
                    self.depth_values.extend(valid_depths.tolist())
            else:
                self.min_depths.append(0.0)
                self.max_depths.append(0.0)
                self.mean_depths.append(0.0)

            self.file_paths.append(pt_path)
            return True

        except Exception as e:
            print(f"Error processing {pt_path}: {e}")
            self.failed_files.append(pt_path)
            return False

    def process_filelist(self, filelist_path: str):
        """
        Process all .pt files listed in a text file.

        Args:
            filelist_path: Path to text file containing .pt file paths
        """
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"Filelist not found: {filelist_path}")

        with open(filelist_path, 'r') as f:
            pt_files = [line.strip() for line in f if line.strip()]

        print(f"Found {len(pt_files)} files to process")
        print("Processing files...")

        for pt_path in tqdm(pt_files, desc="Analyzing depth data"):
            self.process_pt_file(pt_path)

        if self.failed_files:
            print(f"\nWarning: {len(self.failed_files)} files failed to process")

    def generate_report(self, output_path: str):
        """
        Generate depth statistics report.

        Args:
            output_path: Base path for output files (without extension)
        """
        if len(self.depth_values) == 0:
            print("Error: No depth data collected, cannot generate report.")
            return

        depth_array = np.array(self.depth_values)

        # Calculate global statistics
        # Note: For clipped data, all valid pixels may have the same max value
        all_depth_array = np.array(self.depth_values)

        global_stats = {
            "total_samples": len(self.file_paths),
            "total_pixels": sum(self.total_pixel_counts),
            "total_valid_pixels": sum(self.valid_pixel_counts),
            "valid_pixel_ratio": sum(self.valid_pixel_counts) / sum(self.total_pixel_counts),
            "valid_depth_min": float(np.min(all_depth_array)),
            "valid_depth_max": float(np.max(all_depth_array)),
            "valid_depth_mean": float(np.mean(all_depth_array)),
            "valid_depth_median": float(np.median(all_depth_array)),
            "valid_depth_std": float(np.std(all_depth_array)),
            "per_sample_min_depth": float(min(self.min_depths)),
            "per_sample_max_depth": float(max(self.max_depths)),
            "per_sample_mean_depth_avg": float(np.mean(self.mean_depths)),
            "per_sample_mean_depth_std": float(np.std(self.mean_depths)),
        }

        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {f"p{p}": float(np.percentile(depth_array, p)) for p in percentiles}

        # Calculate per-sample statistics
        per_sample_stats = []
        for i, path in enumerate(self.file_paths):
            per_sample_stats.append({
                "file_path": path,
                "total_pixels": self.total_pixel_counts[i],
                "valid_pixels": self.valid_pixel_counts[i],
                "valid_ratio": self.valid_pixel_counts[i] / self.total_pixel_counts[i] if self.total_pixel_counts[i] > 0 else 0,
                "min_depth": self.min_depths[i],
                "max_depth": self.max_depths[i],
                "mean_depth": self.mean_depths[i],
            })

        # Build complete report
        report = {
            "global_statistics": global_stats,
            "percentiles": percentile_values,
            "per_sample_statistics": per_sample_stats,
        }

        if self.failed_files:
            report["failed_files"] = self.failed_files

        # Save JSON report
        json_path = output_path + '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Generate readable text report
        txt_path = output_path + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Cached Depth Data Analysis Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("Global Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total samples: {global_stats['total_samples']}\n")
            f.write(f"Total pixels: {global_stats['total_pixels']:,}\n")
            f.write(f"Valid pixels: {global_stats['total_valid_pixels']:,}\n")
            f.write(f"Valid pixel ratio: {global_stats['valid_pixel_ratio']:.2%}\n\n")

            f.write("Valid Depth Statistics (pixels > 0):\n")
            f.write(f"  Min: {global_stats['valid_depth_min']:.6f} m\n")
            f.write(f"  Max: {global_stats['valid_depth_max']:.6f} m\n")
            f.write(f"  Mean: {global_stats['valid_depth_mean']:.6f} m\n")
            f.write(f"  Median: {global_stats['valid_depth_median']:.6f} m\n")
            f.write(f"  Std: {global_stats['valid_depth_std']:.6f} m\n\n")

            f.write("Per-Sample Statistics:\n")
            f.write(f"  Min depth across samples: {global_stats['per_sample_min_depth']:.6f} m\n")
            f.write(f"  Max depth across samples: {global_stats['per_sample_max_depth']:.6f} m\n")
            f.write(f"  Mean of sample means: {global_stats['per_sample_mean_depth_avg']:.6f} m\n")
            f.write(f"  Std of sample means: {global_stats['per_sample_mean_depth_std']:.6f} m\n\n")

            f.write("Depth Distribution Percentiles:\n")
            f.write("-" * 80 + "\n")
            for p in percentiles:
                f.write(f"P{p:2d}: {percentile_values[f'p{p}']:.6f}\n")
            f.write("\n")

            f.write("Sample Statistics Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Min depth across samples: {min(self.min_depths):.6f}\n")
            f.write(f"Max depth across samples: {max(self.max_depths):.6f}\n")
            f.write(f"Mean of sample means: {np.mean(self.mean_depths):.6f}\n")
            f.write(f"Std of sample means: {np.std(self.mean_depths):.6f}\n\n")

            f.write("First 10 Samples:\n")
            f.write("-" * 80 + "\n")
            for i, stats in enumerate(per_sample_stats[:10]):
                f.write(f"\nSample {i+1}: {os.path.basename(stats['file_path'])}\n")
                f.write(f"  Valid pixels: {stats['valid_pixels']:,} / {stats['total_pixels']:,} ({stats['valid_ratio']:.2%})\n")
                f.write(f"  Depth range: [{stats['min_depth']:.6f}, {stats['max_depth']:.6f}]\n")
                f.write(f"  Mean depth: {stats['mean_depth']:.6f}\n")

            if len(per_sample_stats) > 10:
                f.write(f"\n... (remaining {len(per_sample_stats) - 10} samples in JSON report)\n")

            if self.failed_files:
                f.write(f"\n\nFailed Files ({len(self.failed_files)}):\n")
                f.write("-" * 80 + "\n")
                for failed_file in self.failed_files[:10]:
                    f.write(f"  {failed_file}\n")
                if len(self.failed_files) > 10:
                    f.write(f"  ... and {len(self.failed_files) - 10} more\n")

        print(f"\n{'='*80}")
        print("Analysis Report Generated:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
        print(f"{'='*80}\n")

        print("Global Statistics Summary:")
        print(f"  Samples: {global_stats['total_samples']}")
        print(f"  Valid pixels: {global_stats['valid_pixel_ratio']:.2%}")
        print(f"  Valid depth range: [{global_stats['valid_depth_min']:.6f}, {global_stats['valid_depth_max']:.6f}] m")
        print(f"  Valid depth mean: {global_stats['valid_depth_mean']:.6f} m")
        print(f"  Valid depth median: {global_stats['valid_depth_median']:.6f} m")
        print(f"  Per-sample mean avg: {global_stats['per_sample_mean_depth_avg']:.6f} m")

        if self.failed_files:
            print(f"\nWarning: {len(self.failed_files)} files failed (see report for details)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze depth distribution from cached .pt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_cached_depth.py /path/to/train_cache.txt
  python analyze_cached_depth.py /path/to/train_cache.txt --output ./depth_report
        """
    )
    parser.add_argument(
        'filelist',
        type=str,
        help='Path to text file containing list of .pt files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for report files (without extension). Default: same directory as filelist'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        filelist_dir = os.path.dirname(args.filelist)
        filelist_name = os.path.splitext(os.path.basename(args.filelist))[0]
        output_path = os.path.join(filelist_dir, f"{filelist_name}_depth_analysis")

    # Create analyzer and process files
    analyzer = CachedDepthAnalyzer()

    try:
        analyzer.process_filelist(args.filelist)
        analyzer.generate_report(output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
