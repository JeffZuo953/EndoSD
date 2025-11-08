#!/usr/bin/env python3
"""
Summarize camera intrinsics for cached PT datasets.

Usage:
    python tools/inspect_intrinsics_summary.py \\
        --dataset EndoVis2018=/data/.../EndoVis2018/cache_pt/all_cache.txt \\
        --dataset StereoMIS=/data/.../LS/StereoMIS/cache/cache_pt/all_cache.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect intrinsics/normalization across cached datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        action="append",
        required=True,
        help="Dataset specification in the form NAME=FILELIST_PATH. Can be repeated.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Maximum number of samples to inspect per dataset (default: 10).",
    )
    return parser.parse_args()


def load_dataset_entries(name: str, filelist_path: Path, limit: int) -> List[Dict[str, torch.Tensor]]:
    from dataset.cache_utils import DepthCacheDataset  # lazy import with repo path

    ds = DepthCacheDataset(str(filelist_path), dataset_type="LS", dataset_name=name)
    entries: List[Dict[str, torch.Tensor]] = []
    total = min(len(ds), limit)
    for idx in range(total):
        entries.append(ds[idx])
    return entries


def compute_stats(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    fx: List[float] = []
    fy: List[float] = []
    cx: List[float] = []
    cy: List[float] = []
    width_raw: List[float] = []
    height_raw: List[float] = []
    width_img: List[float] = []
    height_img: List[float] = []

    for sample in samples:
        intr = sample.get("camera_intrinsics")
        if intr is None:
            intr = sample.get("intrinsics")
        if intr is None:
            continue
        intr = torch.as_tensor(intr, dtype=torch.float32)
        fx.append(float(intr[0, 0]))
        fy.append(float(intr[1, 1]))
        cx.append(float(intr[0, 2]))
        cy.append(float(intr[1, 2]))

        size_raw = sample.get("camera_size")
        if size_raw is None:
            size_raw = sample.get("camera_size_original")
        if size_raw is not None:
            size_raw = torch.as_tensor(size_raw, dtype=torch.float32)
            width_raw.append(float(size_raw[0]))
            height_raw.append(float(size_raw[1]))

        size_img = sample.get("camera_image_size")
        if size_img is not None:
            size_img = torch.as_tensor(size_img, dtype=torch.float32)
            width_img.append(float(size_img[0]))
            height_img.append(float(size_img[1]))

    if not fx:
        return {}

    def mean_range(values: List[float]) -> Tuple[float, float, float]:
        arr = np.array(values, dtype=np.float64)
        return float(arr.mean()), float(arr.min()), float(arr.max())

    stats = {
        "samples": len(fx),
        "fx": mean_range(fx),
        "fy": mean_range(fy),
        "cx": mean_range(cx),
        "cy": mean_range(cy),
        "width_raw": mean_range(width_raw) if width_raw else (math.nan, math.nan, math.nan),
        "height_raw": mean_range(height_raw) if height_raw else (math.nan, math.nan, math.nan),
        "width_img": mean_range(width_img) if width_img else (math.nan, math.nan, math.nan),
        "height_img": mean_range(height_img) if height_img else (math.nan, math.nan, math.nan),
    }

    # Derived normalized values using raw size averages.
    if width_raw and height_raw:
        raw_w = stats["width_raw"][0]
        raw_h = stats["height_raw"][0]
        stats["cx_norm"] = (stats["cx"][0] / raw_w, stats["cx"][1] / raw_w, stats["cx"][2] / raw_w)
        stats["cy_norm"] = (stats["cy"][0] / raw_h, stats["cy"][1] / raw_h, stats["cy"][2] / raw_h)
    else:
        stats["cx_norm"] = (math.nan, math.nan, math.nan)
        stats["cy_norm"] = (math.nan, math.nan, math.nan)
    return stats


def format_triplet(triplet: Tuple[float, float, float], precision: int = 2) -> str:
    mean, vmin, vmax = triplet
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f} [{vmin:.{precision}f}, {vmax:.{precision}f}]"


def main() -> None:
    args = parse_args()

    datasets: Dict[str, Path] = {}
    for spec in args.dataset:
        if "=" not in spec:
            raise ValueError(f"Dataset spec must be NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        datasets[name.strip()] = Path(path.strip())

    rows = []
    for name, path in datasets.items():
        if not path.exists():
            print(f"[WARN] Filelist not found for {name}: {path}")
            continue
        samples = load_dataset_entries(name, path, args.samples)
        stats = compute_stats(samples)
        if not stats:
            print(f"[WARN] No valid samples for {name}.")
            continue
        rows.append((name, stats))

    if not rows:
        print("No datasets processed.")
        return

    header = [
        "dataset",
        "samples",
        "raw_size(mean[min,max])",
        "img_size(mean[min,max])",
        "fx(mean[min,max])",
        "fy(mean[min,max])",
        "cx(mean[min,max])",
        "cy(mean[min,max])",
        "cx_norm(mean[min,max])",
        "cy_norm(mean[min,max])",
    ]
    print(" | ".join(header))
    for name, stats in rows:
        raw_size = f"{stats['width_raw'][0]:.1f}x{stats['height_raw'][0]:.1f}" if not math.isnan(stats['width_raw'][0]) else "N/A"
        raw_range = f"[{stats['width_raw'][1]:.1f},{stats['width_raw'][2]:.1f}]/[{stats['height_raw'][1]:.1f},{stats['height_raw'][2]:.1f}]" if not math.isnan(stats['width_raw'][0]) else ""
        img_size = f"{stats['width_img'][0]:.1f}x{stats['height_img'][0]:.1f}" if not math.isnan(stats['width_img'][0]) else "N/A"
        img_range = f"[{stats['width_img'][1]:.1f},{stats['width_img'][2]:.1f}]/[{stats['height_img'][1]:.1f},{stats['height_img'][2]:.1f}]" if not math.isnan(stats['width_img'][0]) else ""
        row = [
            name,
            str(stats["samples"]),
            f"{raw_size} {raw_range}",
            f"{img_size} {img_range}",
            format_triplet(stats["fx"]),
            format_triplet(stats["fy"]),
            format_triplet(stats["cx"]),
            format_triplet(stats["cy"]),
            format_triplet(stats["cx_norm"], precision=3),
            format_triplet(stats["cy_norm"], precision=3),
        ]
        print(" | ".join(row))


if __name__ == "__main__":
    # Ensure repo root is importable
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    main()
