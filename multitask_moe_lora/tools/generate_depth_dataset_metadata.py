#!/usr/bin/env python3
"""
Compute depth distribution statistics for cached FM datasets and materialize results.

Outputs (written to DEPTH_METADATA_DIR or /dataset_metadata by default):
  - depth_histogram_bins.csv : merged histogram with 0.005 m bins.
  - depth_summary_stats.csv  : dataset-level summary (quartiles, mean, variance).
  - plot_depth_statistics.py : helper script for violin/box plots.

Available dataset names for --datasets (pass as comma-separated or repeated flag):
  EndoMapper, EndoNeRF, EndoVis2017, EndoVis2018, EndoVis2018_train, EndoVis2018_val,
  SCARED, StereoMIS, C3VD, C3VDv2, Kidney3D, SimCol, dVPN, hamlyn
Use --workers to enable multi-process sampling during statistics collection.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

HIST_MIN = 0.0
HIST_STEP = 0.005
DEFAULT_OUTPUT_DIR = os.environ.get("DEPTH_METADATA_DIR", "/dataset_metadata")
OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)
PLOTTER_TEMPLATE = Path(__file__).with_name("depth_metadata_plotter.py")
PLOTTER_DEST_NAME = "plot_depth_statistics.py"

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

DATASET_FILELISTS = {
    "SCARED": "/home/ziyi/ssde/data/LS/SCARED/cache/train_all_cache.txt",
    "StereoMIS": "/home/ziyi/ssde/000/abdo/StereoMIS_SA/cache_pt/all_cache.txt",
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

DATASET_ALIASES = {
    "EndoVis2018_train": "EndoVis2018",
    "EndoVis2018_val": "EndoVis2018",
}
ALL_DATASET_NAMES = sorted(
    set(DATASET_FILELISTS.keys()) | set(DATASET_ALIASES.values())
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute depth distribution statistics for cached datasets."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        action="append",
        help=(
            "Datasets to process (comma-separated). "
            f"Choices: {', '.join(ALL_DATASET_NAMES)}. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (set to 1 for sequential).",
    )
    return parser.parse_args()


def normalize_dataset_selection(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return list(DATASET_FILELISTS.keys())
    selected: List[str] = []
    for chunk in raw:
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        selected.extend(parts)
    valid_raw = set(DATASET_FILELISTS.keys())
    valid_alias = set(DATASET_ALIASES.values())
    resolved: List[str] = []
    for name in selected:
        if name in valid_raw:
            resolved.append(name)
            continue
        # allow aggregated alias -> expand to matching raw names
        matched = [
            raw_name
            for raw_name, alias in DATASET_ALIASES.items()
            if alias == name
        ]
        if matched:
            resolved.extend(matched)
            continue
        if name in valid_alias:
            # alias with no expansion (safety fallback)
            continue
        raise ValueError(
            f"Unknown dataset '{name}'. Allowed names: {', '.join(ALL_DATASET_NAMES)}"
        )
    # remove duplicates while preserving order
    seen = set()
    ordered = []
    for name in resolved:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def read_filelist(path: Path) -> List[str]:
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def select_sample(paths: Sequence[str], limit: Optional[int]) -> List[str]:
    if limit is None or len(paths) <= limit:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[i] for i in indices]


def _to_numpy_bool(data: torch.Tensor) -> np.ndarray:
    arr = data.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(bool)


def extract_valid_depths(path: str) -> np.ndarray:
    try:
        payload = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        print(f"Missing cache file: {path}")
        return np.empty(0, dtype=np.float32)
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
            except Exception:
                continue
    if valid_mask is None:
        valid_mask = depth_np > 0.0
    if depth_np.shape != valid_mask.shape:
        valid_mask = np.broadcast_to(valid_mask, depth_np.shape)

    return depth_np[valid_mask]


def process_cache_file(cache_path: str) -> Tuple[List[Tuple[int, int]], int, float, float, int]:
    values = extract_valid_depths(cache_path)
    if values.size == 0:
        return ([], 0, 0.0, 0.0, 0)
    values64 = values.astype(np.float64)
    indices = np.floor(values64 / HIST_STEP).astype(np.int64)
    unique, counts = np.unique(indices, return_counts=True)
    bins = [(int(idx), int(cnt)) for idx, cnt in zip(unique.tolist(), counts.tolist())]
    total_values = int(values64.size)
    total_sum = float(values64.sum())
    total_sq_sum = float(np.square(values64).sum())
    return (bins, total_values, total_sum, total_sq_sum, 1)


def quantile_from_hist(
    bin_starts: np.ndarray, counts: np.ndarray, quantile: float
) -> float:
    total = counts.sum()
    if total == 0:
        return math.nan
    target = quantile * (total - 1)
    cumulative = np.cumsum(counts)
    idx = int(np.searchsorted(cumulative, target, side="right"))
    idx = min(idx, len(bin_starts) - 1)
    prev_cum = cumulative[idx - 1] if idx > 0 else 0.0
    within_bin = target - prev_cum
    count_in_bin = counts[idx]
    if count_in_bin <= 0:
        return bin_starts[idx]
    fraction = within_bin / count_in_bin
    return bin_starts[idx] + fraction * HIST_STEP


def format_float(value: float) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def ensure_output_dir() -> Path:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create output directory {OUTPUT_DIR}. "
            "Set DEPTH_METADATA_DIR to a writable path."
        ) from exc
    return OUTPUT_DIR


def write_histogram_csv(rows: List[Tuple[str, float, float, int]], outdir: Path) -> Path:
    outfile = outdir / "depth_histogram_bins.csv"
    with outfile.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "bin_start", "bin_end", "count"])
        for dataset, start, end, count in rows:
            writer.writerow(
                [
                    dataset,
                    f"{start:.6f}",
                    f"{end:.6f}",
                    int(count),
                ]
            )
    return outfile


def write_summary_csv(rows: List[Dict[str, str]], outdir: Path) -> Path:
    outfile = outdir / "depth_summary_stats.csv"
    headers = [
        "dataset",
        "sampled_files",
        "value_count",
        "quantile_25",
        "quantile_50",
        "quantile_75",
        "mean",
        "variance",
    ]
    with outfile.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return outfile


def copy_plotter_script(outdir: Path) -> Optional[Path]:
    if not PLOTTER_TEMPLATE.exists():
        return None
    destination = outdir / PLOTTER_DEST_NAME
    shutil.copyfile(PLOTTER_TEMPLATE, destination)
    destination.chmod(0o755)
    return destination


def main() -> None:
    args = parse_args()
    selected_datasets = normalize_dataset_selection(args.datasets)
    outdir = ensure_output_dir()
    aggregates: Dict[str, Dict[str, object]] = {}

    for dataset in selected_datasets:
        filelist = DATASET_FILELISTS[dataset]
        path_obj = Path(filelist)
        if not path_obj.exists():
            print(f"{dataset}: missing filelist -> {filelist}")
            continue

        all_entries = read_filelist(path_obj)
        if not all_entries:
            print(f"{dataset}: empty filelist -> {filelist}")
            continue

        sample_paths = select_sample(all_entries, SAMPLE_LIMIT)
        bin_store: Dict[int, int] = defaultdict(int)

        total_values = 0
        total_sum = 0.0
        total_sq_sum = 0.0
        valid_files = 0
        worker_count = max(1, args.workers)

        if worker_count == 1:
            progress = tqdm(sample_paths, desc=f"{dataset}", unit="file")
            for cache_path in progress:
                if not cache_path:
                    continue
                bins_data, value_count, sum_values, sq_sum_values, valid_flag = process_cache_file(
                    cache_path
                )
                if value_count == 0:
                    continue
                valid_files += valid_flag
                total_values += value_count
                total_sum += sum_values
                total_sq_sum += sq_sum_values
                for idx, cnt in bins_data:
                    bin_store[idx] += cnt
            progress.close()
        else:
            progress = tqdm(total=len(sample_paths), desc=f"{dataset}", unit="file")
            chunksize = max(1, len(sample_paths) // (worker_count * 4))
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = executor.map(
                    process_cache_file, sample_paths, chunksize=chunksize
                )
                for bins_data, value_count, sum_values, sq_sum_values, valid_flag in results:
                    progress.update(1)
                    if value_count == 0:
                        continue
                    valid_files += valid_flag
                    total_values += value_count
                    total_sum += sum_values
                    total_sq_sum += sq_sum_values
                    for idx, cnt in bins_data:
                        bin_store[idx] += cnt
            progress.close()

        if total_values == 0 or not bin_store:
            print(f"{dataset}: no valid depth samples collected.")
            continue

        mean = total_sum / total_values
        variance = total_sq_sum / total_values - mean * mean
        variance = max(0.0, variance)

        output_name = DATASET_ALIASES.get(dataset, dataset)
        agg = aggregates.setdefault(
            output_name,
            {
                "bins": defaultdict(int),
                "sampled_files": 0,
                "value_count": 0,
                "sum": 0.0,
                "sq_sum": 0.0,
            },
        )
        bins_dict: defaultdict[int, int] = agg["bins"]  # type: ignore[assignment]
        for idx, cnt in bin_store.items():
            bins_dict[int(idx)] += int(cnt)
        agg["sampled_files"] = int(agg["sampled_files"]) + len(sample_paths)
        agg["value_count"] = int(agg["value_count"]) + total_values
        agg["sum"] = float(agg["sum"]) + total_sum
        agg["sq_sum"] = float(agg["sq_sum"]) + total_sq_sum

        print(
            f"{dataset}: files={len(sample_paths)}, valid_files={valid_files}, "
            f"values={total_values}, mean={mean:.4f}"
        )

    if not aggregates:
        print("No histogram data collected; exiting.")
        return

    histogram_rows: List[Tuple[str, float, float, int]] = []
    summary_rows: List[Dict[str, str]] = []

    for dataset_name in sorted(aggregates.keys()):
        agg = aggregates[dataset_name]
        bins_dict: defaultdict[int, int] = agg["bins"]  # type: ignore[assignment]
        if not bins_dict:
            continue
        bin_indices = np.array(sorted(bins_dict.keys()), dtype=np.int64)
        counts = np.array([bins_dict[idx] for idx in bin_indices], dtype=np.int64)
        bin_starts = bin_indices.astype(np.float64) * HIST_STEP + HIST_MIN
        bin_ends = bin_starts + HIST_STEP

        for start, end, count in zip(bin_starts, bin_ends, counts):
            histogram_rows.append((dataset_name, start, end, int(count)))

        q25 = quantile_from_hist(bin_starts, counts, 0.25)
        q50 = quantile_from_hist(bin_starts, counts, 0.50)
        q75 = quantile_from_hist(bin_starts, counts, 0.75)
        total_values = int(agg["value_count"])
        if total_values == 0:
            mean = float("nan")
            variance = float("nan")
        else:
            total_sum = float(agg["sum"])
            total_sq_sum = float(agg["sq_sum"])
            mean = total_sum / total_values
            variance = total_sq_sum / total_values - mean * mean
            variance = max(0.0, variance)

        summary_rows.append(
            {
                "dataset": dataset_name,
                "sampled_files": str(int(agg["sampled_files"])),
                "value_count": str(total_values),
                "quantile_25": format_float(q25),
                "quantile_50": format_float(q50),
                "quantile_75": format_float(q75),
                "mean": format_float(mean),
                "variance": format_float(variance),
            }
        )

    hist_path = write_histogram_csv(histogram_rows, outdir)
    summary_path = write_summary_csv(summary_rows, outdir)
    plotter_path = copy_plotter_script(outdir)

    print(f"Wrote histogram CSV to {hist_path}")
    print(f"Wrote summary CSV to {summary_path}")
    if plotter_path:
        print(f"Copied plotter script to {plotter_path}")
    else:
        print("Plotter template missing; skipping copy.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
