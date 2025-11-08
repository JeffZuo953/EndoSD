#!/usr/bin/env python3
"""
Evaluate intrinsic-parameter baselines (random, zeros, centered) against cached PT samples.

Usage:
    python tools/eval_intrinsics_baselines.py \
        --filelist /path/to/cache_pt/all_cache.txt \
        --output intrinsics_baselines.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


BASELINES = ("random", "zeros", "center")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate intrinsic baselines on cached PT samples.")
    parser.add_argument(
        "--filelist",
        type=Path,
        action="append",
        required=True,
        help="Path(s) to txt filelists containing PT cache entries (one per line). "
        "Can be repeated to concatenate multiple sets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("intrinsics_baselines.csv"),
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples (per filelist) for quick evaluation.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=16,
        help="Grid resolution used for reprojection error sampling (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the random baseline.",
    )
    return parser.parse_args()


def load_filelist(path: Path, limit: Optional[int] = None) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        entries = [line.strip() for line in f if line.strip()]
    if limit is not None:
        entries = entries[:limit]
    return entries


def create_bucket() -> Dict[str, object]:
    bucket: Dict[str, object] = defaultdict(float)
    bucket["count"] = 0.0
    bucket["f_pair_samples"] = []
    bucket["c_pair_samples"] = []
    bucket["principal_samples"] = []
    bucket["principal_pct_samples"] = []
    bucket["reproj_samples"] = []
    return bucket


def accumulate(bucket: Dict[str, object], sample: Dict[str, float]) -> None:
    bucket["fx_abs"] += sample["fx_abs"]
    bucket["fy_abs"] += sample["fy_abs"]
    bucket["fx_pct"] += sample["fx_pct"]
    bucket["fy_pct"] += sample["fy_pct"]
    bucket["fx_sq"] += sample["fx_abs"] ** 2
    bucket["fy_sq"] += sample["fy_abs"] ** 2

    bucket["cx_abs"] += sample["cx_abs"]
    bucket["cy_abs"] += sample["cy_abs"]
    bucket["cx_pct"] += sample["cx_pct"]
    bucket["cy_pct"] += sample["cy_pct"]
    bucket["cx_sq"] += sample["cx_abs"] ** 2
    bucket["cy_sq"] += sample["cy_abs"] ** 2

    bucket["principal_dist"] += sample["principal_dist"]
    bucket["principal_sq"] += sample["principal_dist"] ** 2
    bucket["principal_pct"] += sample["principal_pct"]
    bucket["principal_pct_sq"] += sample["principal_pct"] ** 2

    reproj_err = sample.get("reproj_err")
    if reproj_err is not None:
        bucket["reproj_sum"] += reproj_err
        bucket["reproj_samples"].append(reproj_err)

    bucket["count"] += 1
    bucket["f_pair_samples"].append(sample["f_pair"])
    bucket["c_pair_samples"].append(sample["c_pair"])
    bucket["principal_samples"].append(sample["principal_dist"])
    bucket["principal_pct_samples"].append(sample["principal_pct"])


def median_or_none(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(statistics.median(values))


def finalize(bucket: Dict[str, object]) -> Dict[str, Optional[float]]:
    count = max(float(bucket.get("count", 0.0)), 1.0)
    fx_abs_mean = bucket.get("fx_abs", 0.0) / count
    fy_abs_mean = bucket.get("fy_abs", 0.0) / count
    fx_pct_mean = bucket.get("fx_pct", 0.0) / count
    fy_pct_mean = bucket.get("fy_pct", 0.0) / count
    cx_abs_mean = bucket.get("cx_abs", 0.0) / count
    cy_abs_mean = bucket.get("cy_abs", 0.0) / count
    cx_pct_mean = bucket.get("cx_pct", 0.0) / count
    cy_pct_mean = bucket.get("cy_pct", 0.0) / count
    principal_mean = bucket.get("principal_dist", 0.0) / count
    principal_pct_mean = bucket.get("principal_pct", 0.0) / count

    fx_rmse = np.sqrt(max(bucket.get("fx_sq", 0.0) + bucket.get("fy_sq", 0.0), 0.0) / max(2.0 * count, 1e-6))
    c_rmse = np.sqrt(max(bucket.get("cx_sq", 0.0) + bucket.get("cy_sq", 0.0), 0.0) / max(2.0 * count, 1e-6))
    principal_rmse = np.sqrt(max(bucket.get("principal_sq", 0.0), 0.0) / max(count, 1e-6))
    principal_pct_rmse = np.sqrt(max(bucket.get("principal_pct_sq", 0.0), 0.0) / max(count, 1e-6))

    reproj_sum = bucket.get("reproj_sum", 0.0)
    reproj_mean = (reproj_sum / count) if reproj_sum else None

    return {
        "samples": bucket.get("count", 0.0),
        "fx_abs": fx_abs_mean,
        "fy_abs": fy_abs_mean,
        "fx_pct": fx_pct_mean,
        "fy_pct": fy_pct_mean,
        "fx_rmse": fx_rmse,
        "cx_abs": cx_abs_mean,
        "cy_abs": cy_abs_mean,
        "cx_pct": cx_pct_mean,
        "cy_pct": cy_pct_mean,
        "c_rmse": c_rmse,
        "principal_dist": principal_mean,
        "principal_pct": principal_pct_mean,
        "principal_rmse": principal_rmse,
        "principal_pct_rmse": principal_pct_rmse,
        "principal_median": median_or_none(bucket["principal_samples"]),
        "principal_pct_median": median_or_none(bucket["principal_pct_samples"]),
        "reproj_mean": reproj_mean,
        "reproj_median": median_or_none(bucket["reproj_samples"]),
    }


def compute_reprojection_error(
    fx_gt: float,
    fy_gt: float,
    cx_gt: float,
    cy_gt: float,
    fx_pred: float,
    fy_pred: float,
    cx_pred: float,
    cy_pred: float,
    width: int,
    height: int,
    grid: int,
    cache: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray]],
) -> float:
    if width <= 1 or height <= 1:
        return 0.0
    steps_x = min(grid, width)
    steps_y = min(grid, height)
    cache_key = (steps_x, steps_y, width, height)
    grid_x, grid_y = cache.get(cache_key, (None, None))
    if grid_x is None or grid_y is None:
        xs = np.linspace(0.0, max(width - 1, 1), num=steps_x, dtype=np.float64)
        ys = np.linspace(0.0, max(height - 1, 1), num=steps_y, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        cache[cache_key] = (grid_x, grid_y)

    fx_gt_safe = max(abs(fx_gt), 1e-6)
    fy_gt_safe = max(abs(fy_gt), 1e-6)
    dir_x = (grid_x - cx_gt) / fx_gt_safe
    dir_y = (grid_y - cy_gt) / fy_gt_safe

    u_pred = fx_pred * dir_x + cx_pred
    v_pred = fy_pred * dir_y + cy_pred
    err = np.sqrt((u_pred - grid_x) ** 2 + (v_pred - grid_y) ** 2)
    return float(err.mean())


def evaluate_sample(
    K_gt: torch.Tensor,
    width: int,
    height: int,
    fx_pred: float,
    fy_pred: float,
    cx_pred: float,
    cy_pred: float,
    grid: int,
    cache: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    fx_gt = float(K_gt[0, 0])
    fy_gt = float(K_gt[1, 1])
    cx_gt = float(K_gt[0, 2])
    cy_gt = float(K_gt[1, 2])

    fx_abs = abs(fx_pred - fx_gt)
    fy_abs = abs(fy_pred - fy_gt)
    cx_abs = abs(cx_pred - cx_gt)
    cy_abs = abs(cy_pred - cy_gt)

    fx_pct = fx_abs / max(abs(fx_gt), 1e-6)
    fy_pct = fy_abs / max(abs(fy_gt), 1e-6)
    cx_pct = cx_abs / max(width, 1e-6)
    cy_pct = cy_abs / max(height, 1e-6)

    principal_dist = float(np.hypot(cx_abs, cy_abs))
    principal_pct = float(np.hypot(cx_pct, cy_pct))
    f_pair = float(np.sqrt(0.5 * (fx_abs * fx_abs + fy_abs * fy_abs)))
    c_pair = float(np.sqrt(0.5 * (cx_abs * cx_abs + cy_abs * cy_abs)))

    reproj_err = compute_reprojection_error(
        fx_gt,
        fy_gt,
        cx_gt,
        cy_gt,
        fx_pred,
        fy_pred,
        cx_pred,
        cy_pred,
        width,
        height,
        grid,
        cache,
    )

    return {
        "fx_abs": fx_abs,
        "fy_abs": fy_abs,
        "fx_pct": fx_pct,
        "fy_pct": fy_pct,
        "cx_abs": cx_abs,
        "cy_abs": cy_abs,
        "cx_pct": cx_pct,
        "cy_pct": cy_pct,
        "principal_dist": principal_dist,
        "principal_pct": principal_pct,
        "f_pair": f_pair,
        "c_pair": c_pair,
        "reproj_err": reproj_err,
    }


def baseline_prediction(
    baseline: str,
    width: int,
    height: int,
    rng: random.Random,
) -> Tuple[float, float, float, float]:
    longest = max(width, height)
    if baseline == "zeros":
        return 0.0, 0.0, 0.0, 0.0
    if baseline == "center":
        fx = fy = longest / 2.0
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy
    # random baseline
    fx = rng.uniform(0.0, longest * 2.0)
    fy = rng.uniform(0.0, longest * 2.0)
    cx = rng.uniform(0.0, width)
    cy = rng.uniform(0.0, height)
    return fx, fy, cx, cy


def main() -> None:
    args = parse_args()
    all_entries: List[str] = []
    for filelist in args.filelist:
        entries = load_filelist(filelist, args.limit)
        all_entries.extend(entries)

    if not all_entries:
        raise RuntimeError("No cache entries found from the provided filelists.")

    rng = random.Random(args.seed)
    reproj_cache: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
    buckets = {name: create_bucket() for name in BASELINES}

    for entry in all_entries:
        try:
            payload = torch.load(entry, map_location="cpu")
        except FileNotFoundError:
            continue

        intrinsics = payload.get("camera_intrinsics") or payload.get("intrinsics")
        if intrinsics is None:
            continue
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)

        image_tensor = payload.get("image")
        if image_tensor is None:
            continue
        if torch.is_tensor(image_tensor):
            height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        else:
            height = int(image_tensor["height"])  # fallback if dict
            width = int(image_tensor["width"])

        for baseline in BASELINES:
            fx_pred, fy_pred, cx_pred, cy_pred = baseline_prediction(baseline, width, height, rng)
            sample_metrics = evaluate_sample(
                intrinsics,
                width,
                height,
                fx_pred,
                fy_pred,
                cx_pred,
                cy_pred,
                args.grid_size,
                reproj_cache,
            )
            accumulate(buckets[baseline], sample_metrics)

    metrics = {name: finalize(bucket) for name, bucket in buckets.items()}

    headers = [
        ("baseline", ""),
        ("samples", "↑"),
        ("fx_abs", "↓"),
        ("fy_abs", "↓"),
        ("fx_pct", "↓"),
        ("fy_pct", "↓"),
        ("fx_rmse", "↓"),
        ("cx_abs", "↓"),
        ("cy_abs", "↓"),
        ("cx_pct", "↓"),
        ("cy_pct", "↓"),
        ("c_rmse", "↓"),
        ("principal_dist", "↓"),
        ("principal_pct", "↓"),
        ("principal_rmse", "↓"),
        ("principal_pct_rmse", "↓"),
        ("principal_median", "↓"),
        ("principal_pct_median", "↓"),
        ("reproj_mean", "↓"),
        ("reproj_median", "↓"),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"{name} ({arrow})" if arrow else name for name, arrow in headers])
        for baseline in BASELINES:
            row = [baseline]
            baseline_metrics = metrics[baseline]
            for name, _arrow in headers[1:]:
                value = baseline_metrics.get(name)
                row.append("" if value is None else f"{value:.6f}")
            writer.writerow(row)

    print(f"Wrote baseline metrics to {args.output}")


if __name__ == "__main__":
    main()
