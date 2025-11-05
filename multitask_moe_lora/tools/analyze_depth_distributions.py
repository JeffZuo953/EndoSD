#!/usr/bin/env python3
"""
Sample cached depth tensors for FM datasets and compute depth distribution statistics.

Outputs:
  - status/{dataset}_hist.csv  : aggregated histogram in [0, 0.3] with 0.02 step.
  - status/{dataset}_quantiles.csv : per-sample 5% / 95% depth quantiles.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

sample_limit_env = os.environ.get("ANALYSIS_SAMPLE_LIMIT")
if sample_limit_env is None or sample_limit_env.strip().lower() == "none":
    SAMPLE_LIMIT = None
else:
    try:
        SAMPLE_LIMIT = int(sample_limit_env)
        if SAMPLE_LIMIT <= 0:
            SAMPLE_LIMIT = None
    except ValueError:
        SAMPLE_LIMIT = 300
HIST_MIN = 0.0
HIST_MAX = 0.3
HIST_STEP = 0.02

# Dataset -> filelist paths (train + val bundles from FM setup)
DATASET_FILELISTS = {
    "SCARED": "/home/ziyi/ssde/data/LS/SCARED/cache/train_all_cache.txt",
    "StereoMIS": "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/all_cache.txt",
    "EndoVis2017": "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/all_cache.txt",
    "EndoVis2018_train": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/train_cache.txt",
    "EndoVis2018_val": "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/val_cache.txt",
    "C3VDv2": "/data/ziyi/multitask/data/NO/c3vdv2/cache/cache.txt",
    "Kidney3D": "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
    "SimCol": "/home/ziyi/ssde/data/simcol/cache/train_all_cache.txt",
    "dVPN": "/home/ziyi/ssde/data/dVPN/cache/train_all_cache.txt",
    "hamlyn": "/data/ziyi/multitask/data/LS/hamlyn/cache_pt/all_cache.txt",
    "EndoNeRF": "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/all_cache.txt",
    "C3VD": "/data/ziyi/multitask/data/NO/c3vd/cache/all_cache.txt",
    "EndoMapper": "/data/ziyi/multitask/data/NO/endomapper_sim/cache/all_cache.txt",
}


def read_filelist(path: Path) -> List[str]:
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def select_sample(paths: Sequence[str], limit: Optional[int]) -> List[str]:
    if limit is None or len(paths) <= limit:
        return list(paths)
    # evenly spaced sampling for determinism
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[i] for i in indices]


def _to_numpy_bool(data: Any) -> np.ndarray:
    if data is None:
        raise ValueError
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(bool)


def extract_valid_depths(path: str) -> np.ndarray:
    payload = torch.load(path, map_location="cpu")
    depth = payload.get("depth")
    if depth is None:
        return np.empty(0, dtype=np.float32)
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np[0]

    valid_mask = None
    for key in ("depth_valid_mask", "valid_mask"):
        if key in payload:
            try:
                valid_mask = _to_numpy_bool(payload[key])
                break
            except ValueError:
                continue

    if valid_mask is None:
        valid_mask = depth_np > 0.0

    if depth_np.shape != valid_mask.shape:
        valid_mask = np.broadcast_to(valid_mask, depth_np.shape)

    values = depth_np[valid_mask]
    return values


def ensure_status_dir() -> Path:
    status_dir = Path("status")
    status_dir.mkdir(exist_ok=True)
    return status_dir


def write_histogram(status_dir: Path, dataset: str, bins: np.ndarray, counts: np.ndarray) -> None:
    outfile = status_dir / f"{dataset}_hist.csv"
    with outfile.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_start", "bin_end", "count"])
        for start, end, cnt in zip(bins[:-1], bins[1:], counts):
            writer.writerow([f"{start:.3f}", f"{end:.3f}", int(cnt)])


def write_quantiles(status_dir: Path, dataset: str, rows: Iterable[Tuple[str, float, float]]) -> None:
    outfile = status_dir / f"{dataset}_quantiles.csv"
    with outfile.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "quantile_5", "quantile_95"])
        for fname, q05, q95 in rows:
            q05_str = "" if math.isnan(q05) else f"{q05:.6f}"
            q95_str = "" if math.isnan(q95) else f"{q95:.6f}"
            writer.writerow([fname, q05_str, q95_str])


def main() -> None:
    status_dir = ensure_status_dir()

    summary_lines = []
    for dataset, filelist_path in DATASET_FILELISTS.items():
        path_obj = Path(filelist_path)
        if not path_obj.exists():
            summary_lines.append(f"{dataset}: filelist missing -> {filelist_path}")
            continue

        entries = read_filelist(path_obj)
        if not entries:
            summary_lines.append(f"{dataset}: empty filelist -> {filelist_path}")
            continue

        sample_paths = select_sample(entries, SAMPLE_LIMIT)
        quantile_rows = []
        max_depth_seen = HIST_MIN

        progress = tqdm(sample_paths, desc=f"{dataset}", unit="file")
        for cache_path in progress:
            if not cache_path:
                continue
            values = extract_valid_depths(cache_path)
            if values.size == 0:
                quantile_rows.append((cache_path, math.nan, math.nan))
                continue
            q05 = float(np.quantile(values, 0.05))
            q95 = float(np.quantile(values, 0.95))
            quantile_rows.append((cache_path, q05, q95))
            max_depth_seen = max(max_depth_seen, float(values.max()))

        if max_depth_seen <= HIST_MIN:
            max_depth_for_bins = HIST_MIN + HIST_STEP
        else:
            steps = math.ceil((max_depth_seen - HIST_MIN) / HIST_STEP)
            max_depth_for_bins = HIST_MIN + steps * HIST_STEP

        bin_edges = np.arange(HIST_MIN, max_depth_for_bins + HIST_STEP * 0.5, HIST_STEP, dtype=np.float32)
        total_hist = np.zeros(len(bin_edges) - 1, dtype=np.int64)

        for cache_path in sample_paths:
            if not cache_path:
                continue
            values = extract_valid_depths(cache_path)
            if values.size == 0:
                continue
            hist, _ = np.histogram(values, bins=bin_edges)
            total_hist += hist

        write_histogram(status_dir, dataset, bin_edges, total_hist)
        write_quantiles(status_dir, dataset, quantile_rows)

        valid_entries = sum(0 if math.isnan(q05) else 1 for _, q05, _ in quantile_rows)
        summary_line = f"{dataset}: sampled={len(sample_paths)}, valid_quantiles={valid_entries}, max_depth={max_depth_seen:.3f}"
        summary_lines.append(summary_line)
        print(summary_line)

    summary_path = status_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("\n".join(summary_lines))

    print("Analysis complete. Summary:")
    for line in summary_lines:
        print("  ", line)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
