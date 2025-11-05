#!/usr/bin/env python3
"""
Compute depth-value histograms for PT cache files.

The script traverses one or more cache roots, finds every ``*.pt`` file below a
``cache_pt`` directory, and aggregates depth statistics per dataset. Depth
values are bucketised into 1 mm (0.001 m) bins over [0, 0.3). Exact zeros and
values greater than or equal to the maximum bucket edge are tracked in their
own buckets. Results are written as a CSV file that can be consumed by plotting
utilities such as ``plot_violin_from_csv.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class DatasetHistogram:
    """Container for aggregated histogram values."""

    zero_count: int
    bin_counts: np.ndarray  # shape: (num_bins,)
    ge_max_count: int
    total_count: int
    nan_count: int
    negative_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate depth histograms from cached PT samples."
    )
    parser.add_argument(
        "--cache-root",
        action="append",
        dest="cache_roots",
        type=Path,
        help="Root directory containing cache_pt folders (repeatable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("depth_histograms.csv"),
        help="Destination CSV file (default: depth_histograms.csv).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.001,
        help="Histogram bin width in metres (default: 0.001 = 1 mm).",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=0.3,
        help="Upper bound (exclusive) for regular bins in metres (default: 0.3).",
    )
    parser.add_argument(
        "--zero-eps",
        type=float,
        default=1e-6,
        help="Values with absolute depth <= zero_eps are counted as zero.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip cache roots that do not exist instead of failing.",
    )
    return parser.parse_args()


def resolve_cache_roots(args: argparse.Namespace) -> List[Path]:
    if args.cache_roots:
        roots = list(dict.fromkeys(args.cache_roots))
    else:
        # Heuristic default: sibling data directory relative to repository root.
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root.parent / "data"
        roots = [candidate]

    resolved: List[Path] = []
    for root in roots:
        if root.exists():
            resolved.append(root.resolve())
        elif not args.allow_missing:
            raise FileNotFoundError(f"Cache root {root} does not exist.")
    if not resolved:
        raise RuntimeError("No valid cache roots found.")
    return resolved


def iter_pt_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            path_parts = Path(dirpath).parts
            if "cache_pt" not in path_parts:
                continue
            for filename in filenames:
                if not filename.endswith(".pt"):
                    continue
                yield Path(dirpath) / filename


def infer_dataset_label(pt_path: Path, record: dict) -> str:
    if isinstance(record, dict):
        name = record.get("dataset_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        src = record.get("source_type")
        if isinstance(src, str) and src.strip():
            seq = record.get("sequence")
            if isinstance(seq, str) and seq.strip():
                return f"{src.strip()}:{seq.strip()}"
            return src.strip()

    parts = pt_path.parts
    if "cache_pt" in parts:
        idx = parts.index("cache_pt")
        before = parts[idx - 1] if idx > 0 else "cache_pt"
        after = parts[idx + 1] if idx + 1 < len(parts) else ""
        if after and after not in {"images"}:
            return f"{before}/{after}"
        return before
    return pt_path.parent.name


def load_depth_tensor(pt_path: Path) -> Tuple[np.ndarray, dict]:
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected data format in {pt_path}")
    depth = data.get("depth")
    if depth is None:
        raise KeyError(f"No 'depth' entry found in {pt_path}")
    if torch.is_tensor(depth):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth)
    if depth_np.ndim == 3 and depth_np.shape[0] == 1:
        depth_np = depth_np[0]
    if depth_np.ndim != 2:
        raise ValueError(f"Depth tensor in {pt_path} has shape {depth_np.shape}, expected 2D.")
    return depth_np.astype(np.float64, copy=False), data


class HistogramAccumulator:
    def __init__(self, bin_size: float, max_depth: float, zero_eps: float) -> None:
        ratio = max_depth / bin_size
        if not math.isclose(ratio, round(ratio)):
            raise ValueError("max_depth must be an integer multiple of bin_size.")
        self.bin_size = bin_size
        self.max_depth = max_depth
        self.zero_eps = zero_eps
        self.num_bins = int(round(ratio))
        self.bin_edges = np.linspace(0.0, max_depth, self.num_bins + 1, dtype=np.float64)
        self.stats: Dict[str, DatasetHistogram] = {}

    def add_sample(self, dataset: str, depth_values: np.ndarray) -> None:
        flat = depth_values.reshape(-1)
        finite_mask = np.isfinite(flat)
        nan_count = int(finite_mask.size - finite_mask.sum())
        clean = flat[finite_mask]

        zero_mask = np.abs(clean) <= self.zero_eps
        zero_count = int(zero_mask.sum())

        positive = clean[~zero_mask]
        if positive.size == 0:
            hist_bins = np.zeros(self.num_bins, dtype=np.int64)
            ge_count = 0
            negative_count = 0
        else:
            negative_mask = positive < 0
            negative_count = int(negative_mask.sum())
            positive = positive[~negative_mask]

            if positive.size == 0:
                hist_bins = np.zeros(self.num_bins, dtype=np.int64)
                ge_count = 0
            else:
                ge_mask = positive >= self.max_depth
                ge_count = int(ge_mask.sum())
                mid = positive[~ge_mask]
                if mid.size:
                    hist_bins, _ = np.histogram(mid, bins=self.bin_edges)
                else:
                    hist_bins = np.zeros(self.num_bins, dtype=np.int64)

        stats = self.stats.get(dataset)
        if stats is None:
            stats = DatasetHistogram(
                zero_count=0,
                bin_counts=np.zeros(self.num_bins, dtype=np.int64),
                ge_max_count=0,
                total_count=0,
                nan_count=0,
                negative_count=0,
            )
            self.stats[dataset] = stats

        stats.zero_count += zero_count
        stats.bin_counts += hist_bins
        stats.ge_max_count += ge_count
        stats.total_count += clean.size
        stats.nan_count += nan_count
        stats.negative_count += negative_count

    def write_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset",
            "bin_label",
            "bin_start",
            "bin_end",
            "count",
            "total_count",
            "nan_count",
            "negative_count",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for dataset in sorted(self.stats.keys()):
                stats = self.stats[dataset]
                total = stats.total_count
                nan_count = stats.nan_count
                neg_count = stats.negative_count

                writer.writerow(
                    {
                        "dataset": dataset,
                        "bin_label": "zero",
                        "bin_start": 0.0,
                        "bin_end": 0.0,
                        "count": stats.zero_count,
                        "total_count": total,
                        "nan_count": nan_count,
                        "negative_count": neg_count,
                    }
                )

                for idx, count in enumerate(stats.bin_counts):
                    if count == 0:
                        continue
                    start = float(self.bin_edges[idx])
                    end = float(self.bin_edges[idx + 1])
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "bin_label": f"{start:.3f}-{end:.3f}",
                            "bin_start": start,
                            "bin_end": end,
                            "count": int(count),
                            "total_count": total,
                            "nan_count": nan_count,
                            "negative_count": neg_count,
                        }
                    )

                writer.writerow(
                    {
                        "dataset": dataset,
                        "bin_label": f">={self.max_depth:.3f}",
                        "bin_start": float(self.max_depth),
                        "bin_end": "",
                        "count": stats.ge_max_count,
                        "total_count": total,
                        "nan_count": nan_count,
                        "negative_count": neg_count,
                    }
                )


def main() -> None:
    args = parse_args()
    cache_roots = resolve_cache_roots(args)

    if not args.quiet:
        print("Scanning cache roots:", ", ".join(str(r) for r in cache_roots))

    accumulator = HistogramAccumulator(
        bin_size=args.bin_size,
        max_depth=args.max_depth,
        zero_eps=args.zero_eps,
    )

    pt_files = list(iter_pt_files(cache_roots))
    if not pt_files:
        raise RuntimeError("No PT files found under the provided roots.")

    for idx, pt_path in enumerate(pt_files, start=1):
        try:
            depth_values, record = load_depth_tensor(pt_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skipping {pt_path}: {exc}", file=sys.stderr)
            continue

        dataset = infer_dataset_label(pt_path, record)
        accumulator.add_sample(dataset, depth_values)

        if not args.quiet and idx % 50 == 0:
            print(f"Processed {idx}/{len(pt_files)} files...", end="\r")

    if not args.quiet:
        print(f"Processed {len(pt_files)} files. Writing CSV to {args.output}")

    accumulator.write_csv(args.output)


if __name__ == "__main__":
    main()
