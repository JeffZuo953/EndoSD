#!/usr/bin/env python3
"""Compute depth value statistics for cached .pt datasets."""
import argparse
import os
import sys
from typing import Dict, Iterable

import torch

DEFAULT_BASE = os.environ.get("BASE_DATA_PATH", "/data/ziyi/multitask")
DEFAULT_HOME_SSD = os.environ.get("HOME_SSD_PATH", os.path.join(os.path.expanduser("~"), "ssde"))


def _rewrite(path: str) -> str:
    if not path:
        return path
    path = path.replace("${BASE_DATA_PATH}", DEFAULT_BASE)
    path = path.replace("${HOME_SSD_PATH}", DEFAULT_HOME_SSD)
    path = path.replace("$BASE_DATA_PATH", DEFAULT_BASE)
    path = path.replace("$HOME_SSD_PATH", DEFAULT_HOME_SSD)
    if path.startswith("~/"):
        path = os.path.expanduser(path)
    return path


DATASET_FILELISTS: Dict[str, str] = {
    "endonerf": os.path.join(DEFAULT_BASE, "data/LS/EndoNeRF/cache_pt/all_cache.txt"),
    "endovis2018": os.path.join(DEFAULT_BASE, "data/LS/EndoVis2018/cache_pt/all_cache.txt"),
    "endovis2017": os.path.join(DEFAULT_BASE, "data/LS/EndoVis2017/Endovis2017_seg_depth/cache/train_cache.txt"),
}


def _iter_cache_paths(filelist: str) -> Iterable[str]:
    filelist = _rewrite(filelist)
    if not os.path.exists(filelist):
        raise FileNotFoundError(f"Cache filelist not found: {filelist}")
    with open(filelist, "r") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            yield _rewrite(path)


def _update_stats(stats: Dict[str, float], tensor: torch.Tensor) -> None:
    tensor = tensor.flatten()
    if tensor.numel() == 0:
        return
    stats["min"] = min(stats["min"], tensor.min().item())
    stats["max"] = max(stats["max"], tensor.max().item())
    stats["sum"] += float(tensor.sum().item())
    stats["sum_sq"] += float((tensor.double().pow(2).sum()).item())
    stats["count"] += int(tensor.numel())


def analyze_dataset(name: str, filelist: str) -> Dict[str, float]:
    stats = {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "sum_sq": 0.0, "count": 0}
    num_files = 0
    for pt_path in _iter_cache_paths(filelist):
        if not os.path.exists(pt_path):
            continue
        payload = torch.load(pt_path, map_location="cpu")
        depth = payload.get("depth")
        if depth is None:
            continue
        depth = depth.to(torch.float32)
        mask = payload.get("valid_mask")
        if mask is not None:
            mask = mask.to(torch.bool)
            if mask.shape != depth.shape:
                mask = None
        if mask is not None:
            depth = depth[mask]
        depth = depth[torch.isfinite(depth)]
        if depth.numel() == 0:
            continue
        _update_stats(stats, depth)
        num_files += 1
    if stats["count"] == 0:
        raise RuntimeError(f"No depth values found for dataset '{name}'. Checked {num_files} cache files.")
    stats["mean"] = stats["sum"] / stats["count"]
    stats["std"] = (stats["sum_sq"] / stats["count"] - stats["mean"] ** 2) ** 0.5
    stats["files"] = num_files
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute depth value ranges for cached datasets.")
    parser.add_argument("--datasets", nargs="*", default=["endonerf", "endovis2018", "endovis2017"],
                        help="Dataset keys to evaluate (default: all supported).")
    args = parser.parse_args()

    results = {}
    for key in args.datasets:
        norm_key = key.lower()
        if norm_key not in DATASET_FILELISTS:
            print(f"[WARN] Unknown dataset key '{key}', skipping.", file=sys.stderr)
            continue
        filelist = DATASET_FILELISTS[norm_key]
        stats = analyze_dataset(norm_key, filelist)
        results[norm_key] = stats

    if not results:
        print("No valid datasets processed.", file=sys.stderr)
        sys.exit(1)

    print("Depth Range Statistics")
    print("======================")
    for name, stats in results.items():
        print(f"Dataset: {name}")
        print(f"  Files processed : {stats['files']}")
        print(f"  Count           : {stats['count']}")
        print(f"  Min depth (m)   : {stats['min']:.6f}")
        print(f"  Max depth (m)   : {stats['max']:.6f}")
        print(f"  Mean depth (m)  : {stats['mean']:.6f}")
        print(f"  Std depth  (m)  : {stats['std']:.6f}")
        print("")


if __name__ == "__main__":
    main()
