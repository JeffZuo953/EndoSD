#!/usr/bin/env python3
"""
Inspect cached datasets to confirm camera normalization (camera_intrinsics_norm) vs. raw intrinsics.
Prints per-dataset summary showing raw fx/fy/cx/cy, normalized values, and the width/height used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check normalization of cached camera intrinsics.")
    parser.add_argument(
        "--filelist",
        type=Path,
        action="append",
        required=True,
        help="Path(s) to cache filelists (.txt). Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Max samples per filelist (default 2000; set -1 for all).",
    )
    return parser.parse_args()


def load_entries(paths: Iterable[Path], limit: int | None) -> Iterable[str]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                entry = line.strip()
                if entry:
                    yield entry


def summarize_entries(entries: Iterable[str]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        try:
            payload = torch.load(entry, map_location="cpu")
        except FileNotFoundError:
            continue

        dataset_name = payload.get("dataset_name") or payload.get("source_type") or "unknown"

        K = payload.get("camera_intrinsics") or payload.get("intrinsics")
        K = torch.as_tensor(K, dtype=torch.float32)

        width_height_raw = payload.get("camera_size") or payload.get("camera_size_original")
        width_height_img = payload.get("camera_image_size")
        norm = payload.get("camera_intrinsics_norm")

        if any(item is None for item in (width_height_raw, width_height_img, norm)):
            continue

        width_height_raw = torch.as_tensor(width_height_raw, dtype=torch.float32)
        width_height_img = torch.as_tensor(width_height_img, dtype=torch.float32)
        norm = torch.as_tensor(norm, dtype=torch.float32)

        stats = buckets[dataset_name]
        stats["fx"].append(float(K[0, 0]))
        stats["fy"].append(float(K[1, 1]))
        stats["cx"].append(float(K[0, 2]))
        stats["cy"].append(float(K[1, 2]))
        stats["fx_norm"].append(float(norm[0]))
        stats["fy_norm"].append(float(norm[1]))
        stats["cx_norm"].append(float(norm[2]))
        stats["cy_norm"].append(float(norm[3]))
        stats["width_raw"].append(float(width_height_raw[0]))
        stats["height_raw"].append(float(width_height_raw[1]))
        stats["width_img"].append(float(width_height_img[0]))
        stats["height_img"].append(float(width_height_img[1]))

    summary = {}
    for name, stats in buckets.items():
        summary[name] = {
            "samples": len(stats["fx"]),
            "fx_mean": float(np.mean(stats["fx"])),
            "fy_mean": float(np.mean(stats["fy"])),
            "cx_mean": float(np.mean(stats["cx"])),
            "cy_mean": float(np.mean(stats["cy"])),
            "cx_min": float(np.min(stats["cx"])),
            "cx_max": float(np.max(stats["cx"])),
            "cy_min": float(np.min(stats["cy"])),
            "cy_max": float(np.max(stats["cy"])),
            "width_raw": float(np.median(stats["width_raw"])),
            "height_raw": float(np.median(stats["height_raw"])),
            "width_img": float(np.median(stats["width_img"])),
            "height_img": float(np.median(stats["height_img"])),
            "fx_norm_mean": float(np.mean(stats["fx_norm"])),
            "fy_norm_mean": float(np.mean(stats["fy_norm"])),
            "cx_norm_range": (float(np.min(stats["cx_norm"])), float(np.max(stats["cx_norm"]))),
            "cy_norm_range": (float(np.min(stats["cy_norm"])), float(np.max(stats["cy_norm"]))),
        }
    return summary


def main() -> None:
    args = parse_args()
    limit = None if args.limit is None or args.limit < 0 else args.limit
    entries = load_entries(args.filelist, limit)
    summary = summarize_entries(entries)

    if not summary:
        print("No samples found.")
        return

    header = [
        "dataset",
        "samples",
        "width_raw",
        "height_raw",
        "width_img",
        "height_img",
        "cx_mean",
        "cy_mean",
        "cx_range",
        "cy_range",
        "fx_mean",
        "fy_mean",
        "cx_norm_range",
        "cy_norm_range",
    ]
    row_format = "{:<18} {:>8} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.2f} {:>10.2f} {:>18} {:>18} {:>10.2f} {:>10.2f} {:>18} {:>18}"
    print(" ".join(header))
    for dataset, stats in sorted(summary.items()):
        cx_norm_range = f"{stats['cx_norm_range'][0]:.3f}-{stats['cx_norm_range'][1]:.3f}"
        cy_norm_range = f"{stats['cy_norm_range'][0]:.3f}-{stats['cy_norm_range'][1]:.3f}"
        cx_range = f"{stats['cx_min']:.2f}-{stats['cx_max']:.2f}"
        cy_range = f"{stats['cy_min']:.2f}-{stats['cy_max']:.2f}"
        row = row_format.format(
            dataset,
            stats["samples"],
            stats["width_raw"],
            stats["height_raw"],
            stats["width_img"],
            stats["height_img"],
            stats["cx_mean"],
            stats["cy_mean"],
            cx_range,
            cy_range,
            stats["fx_mean"],
            stats["fy_mean"],
            cx_norm_range,
            cy_norm_range,
        )
        print(row)


if __name__ == "__main__":
    main()
