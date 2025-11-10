#!/usr/bin/env python3
"""
Compute depth distribution statistics for cached FM datasets and materialize results.

Outputs (written to --output-dir, defaulting to DEPTH_METADATA_DIR or /dataset_metadata):
  - depth_histogram_bins.csv : merged histogram with 0.005 m bins.
  - depth_summary_stats.csv  : dataset-level summary (quartiles, mean, variance).
  - plot_depth_statistics.py : helper script for violin/box plots.

Available dataset names for --datasets (comma-separated or repeated flag):
  SCARED, EndoVis2017, EndoVis2017_train, EndoVis2017_eval, EndoVis2018,
  EndoSynth, C3VDv2, Kidney3D, SimCol, dVPN, EndoNeRF, C3VD, EndoMapper,
  StereoMIS, hamlyn
Use --workers to enable multi-process sampling during statistics collection.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.append(str(PACKAGE_PARENT))

try:
    from multitask_moe_lora.util.data_utils import _build_native_depth_dataset
except Exception:  # pragma: no cover - fallback for minimal envs
    _build_native_depth_dataset = None

DEFAULT_HIST_MIN = 0.0
DEFAULT_HIST_STEP = 0.005
HIST_MIN = DEFAULT_HIST_MIN
HIST_STEP = DEFAULT_HIST_STEP
DEFAULT_OUTPUT_DIR = Path(os.environ.get("DEPTH_METADATA_DIR", "/dataset_metadata"))
PLOTTER_TEMPLATE = Path(__file__).with_name("depth_metadata_plotter.py")
PLOTTER_DEST_NAME = "plot_depth_statistics.py"

DATASET_FILELISTS: Dict[str, List[str]] = {
    "SCARED": [
        "/home/ziyi/ssde/data/LS/SCARED/cache/train_all_cache.txt",
    ],
    "EndoVis2017": [
        "/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/train_cache.txt",
        "/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/eval_cache.txt",
    ],
    "EndoVis2017_train": [
        "/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/train_cache.txt",
    ],
    "EndoVis2017_eval": [
        "/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth/cache/eval_cache.txt",
    ],
    "EndoVis2018": [
        "/data/ziyi/multitask/data/LS/EndoVis2018/cache_pt/all_cache.txt",
    ],
    "EndoSynth": [
        "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/train_cache.txt",
        "/data/ziyi/multitask/data/LS/EndoSynth/cache_pt/val_cache.txt",
    ],
    "C3VDv2": [
        "/data/ziyi/multitask/data/NO/c3vdv2/cache/cache.txt",
    ],
    "Kidney3D": [
        "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
        "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/val_cache.txt",
    ],
    "Kidney3D_train": [
        "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
    ],
    "Kidney3D_eval": [
        "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/val_cache.txt",
    ],
    "SimCol": [
        "/home/ziyi/ssde/data/simcol/cache/train_all_cache.txt",
    ],
    "dVPN": [
        "/home/ziyi/ssde/data/dVPN/cache/train_all_cache.txt",
    ],
    "EndoNeRF": [
        "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/all_cache.txt",
    ],
    "C3VD": [
        "/data/ziyi/multitask/data/NO/c3vd/cache/all_cache.txt",
    ],
    "EndoMapper": [
        "/data/ziyi/multitask/data/NO/endomapper_sim/cache/all_cache.txt",
    ],
}

NATIVE_DATASETS: Dict[str, Dict[str, Any]] = {
    "StereoMIS": {
        "dataset": "StereoMIS",
        "dataset_type": "LS",
        "name": "StereoMIS",
        "params": {
            "root_dir": "/data/ziyi/multitask/data/LS/StereoMIS",
            "split": "all",
            "size": [518, 518],
            # raw StereoMIS depth npy files store millimeters; use a large clamp so we can
            # rescale to meters later without flattening everything to the old 0.3 m cap
            "max_depth": 2000.0,
        },
    },
    "hamlyn": {
        "dataset": "hamlyn",
        "dataset_type": "LS",
        "name": "hamlyn",
        "params": {
            "filelist_path": "~/ssde/000/abdo/hamlyn_data/filelists/all.txt",
            "rootpath": "~/ssde/000/abdo/hamlyn_data",
            "mode": "eval",
            "size": [518, 518],
            "max_depth": 0.3,
        },
    },
}

ALL_DATASET_NAMES = sorted(
    set(DATASET_FILELISTS.keys()) | set(NATIVE_DATASETS.keys())
)
DEFAULT_NATIVE_IMAGE_SIZE = 518
DEFAULT_NATIVE_MAX_DEPTH = 0.3


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
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help=(
            "Equidistant sample count per dataset; omit or set <=0 to process all "
            "available cache entries."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset_metadata",
        help=(
            "Directory to store generated metadata (defaults ./dataset_metadata)."
        ),
    )
    parser.add_argument(
        "--hist-step",
        type=float,
        default=0.005,
        help="Histogram bin width in meters (must be > 0).",
    )
    parser.add_argument(
        "--hist-min",
        type=float,
        default=0.0,
        help="Minimum depth value represented by the histogram bins.",
    )
    return parser.parse_args()


def normalize_dataset_selection(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return list(ALL_DATASET_NAMES)
    selected: List[str] = []
    for chunk in raw:
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        selected.extend(parts)
    valid_names = set(ALL_DATASET_NAMES)
    resolved: List[str] = []
    for name in selected:
        if name not in valid_names:
            raise ValueError(
                f"Unknown dataset '{name}'. Allowed names: {', '.join(ALL_DATASET_NAMES)}"
            )
        resolved.append(name)
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


def _to_numpy_bool(data: Any) -> np.ndarray:
    if data is None:
        raise ValueError("Cannot convert None to boolean mask.")
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
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


def histogram_from_values(values: np.ndarray) -> Tuple[List[Tuple[int, int]], int, float, float]:
    if values.size == 0:
        return ([], 0, 0.0, 0.0)
    values64 = values.astype(np.float64, copy=False)
    indices = np.floor((values64 - HIST_MIN) / HIST_STEP).astype(np.int64)
    unique, counts = np.unique(indices, return_counts=True)
    bins = [(int(idx), int(cnt)) for idx, cnt in zip(unique.tolist(), counts.tolist())]
    total_values = int(values64.size)
    total_sum = float(values64.sum())
    total_sq_sum = float(np.square(values64).sum())
    return (bins, total_values, total_sum, total_sq_sum)


def process_cache_file(cache_path: str) -> Tuple[List[Tuple[int, int]], int, float, float, int]:
    values = extract_valid_depths(cache_path)
    if values.size == 0:
        return ([], 0, 0.0, 0.0, 0)
    bins, total_values, total_sum, total_sq_sum = histogram_from_values(values)
    return (bins, total_values, total_sum, total_sq_sum, 1)


@dataclass
class DatasetStats:
    bins: Dict[int, int]
    total_values: int
    total_sum: float
    total_sq_sum: float
    sampled_units: int
    valid_units: int


def select_indices(total_count: int, limit: Optional[int]) -> List[int]:
    if total_count <= 0:
        return []
    if limit is None or total_count <= limit:
        return list(range(total_count))
    return np.linspace(0, total_count - 1, num=limit, dtype=int).tolist()


def extract_values_from_sample(sample: Dict[str, Any]) -> np.ndarray:
    depth = sample.get("depth")
    if depth is None:
        return np.empty(0, dtype=np.float32)
    if torch.is_tensor(depth):
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
    else:
        depth_np = np.asarray(depth, dtype=np.float32)
    if depth_np.ndim == 3:
        depth_np = depth_np[0]

    valid_mask = None
    for key in ("depth_valid_mask", "valid_mask"):
        if key in sample:
            try:
                valid_mask = _to_numpy_bool(sample[key])
                break
            except ValueError:
                continue
    if valid_mask is None:
        valid_mask = depth_np > 0.0
    if depth_np.shape != valid_mask.shape:
        valid_mask = np.broadcast_to(valid_mask, depth_np.shape)
    return depth_np[valid_mask]


def adjust_values_for_dataset(dataset: str, values: np.ndarray) -> np.ndarray:
    """Apply dataset-specific scaling or normalization before statistics."""
    if values.size == 0:
        return values
    if dataset.lower() == "stereomis":
        return values / 1000.0
    return values


def merge_dataset_stats(
    dataset_name: str, stats: DatasetStats, aggregates: Dict[str, Dict[str, object]]
) -> None:
    agg = aggregates.setdefault(
        dataset_name,
        {
            "bins": defaultdict(int),
            "sampled_files": 0,
            "value_count": 0,
            "sum": 0.0,
            "sq_sum": 0.0,
        },
    )
    bins_dict: defaultdict[int, int] = agg["bins"]  # type: ignore[assignment]
    for idx, cnt in stats.bins.items():
        bins_dict[int(idx)] += int(cnt)
    agg["sampled_files"] = int(agg["sampled_files"]) + int(stats.sampled_units)
    agg["value_count"] = int(agg["value_count"]) + int(stats.total_values)
    agg["sum"] = float(agg["sum"]) + float(stats.total_sum)
    agg["sq_sum"] = float(agg["sq_sum"]) + float(stats.total_sq_sum)


def process_cache_dataset(
    dataset: str, filelists: Sequence[str], worker_count: int, sample_limit: Optional[int]
) -> Optional[DatasetStats]:
    all_entries: List[str] = []
    missing_paths = True
    for filelist in filelists:
        path_obj = Path(filelist)
        if not path_obj.exists():
            print(f"{dataset}: missing filelist -> {filelist}")
            continue
        missing_paths = False
        entries = read_filelist(path_obj)
        if not entries:
            print(f"{dataset}: empty filelist -> {filelist}")
            continue
        all_entries.extend(entries)
    if missing_paths and not all_entries:
        return None
    if not all_entries:
        print(f"{dataset}: no cache entries collected.")
        return None

    sample_paths = select_sample(all_entries, sample_limit)
    if not sample_paths:
        print(f"{dataset}: sampled path list is empty.")
        return None

    bin_store: Dict[int, int] = defaultdict(int)
    total_values = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    valid_files = 0
    worker_count = max(1, worker_count)

    def _process_sequential() -> None:
        nonlocal total_values, total_sum, total_sq_sum, valid_files
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

    def _process_parallel(current_workers: int) -> None:
        nonlocal total_values, total_sum, total_sq_sum, valid_files
        progress = tqdm(total=len(sample_paths), desc=f"{dataset}", unit="file")
        chunksize = max(1, len(sample_paths) // (current_workers * 4) or 1)
        with ProcessPoolExecutor(max_workers=current_workers) as executor:
            results = executor.map(process_cache_file, sample_paths, chunksize=chunksize)
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

    if worker_count == 1:
        _process_sequential()
    else:
        try:
            _process_parallel(worker_count)
        except (PermissionError, OSError) as exc:
            print(f"{dataset}: multiprocess workers unavailable ({exc}); falling back to single worker.")
            _process_sequential()

    if total_values == 0 or not bin_store:
        print(f"{dataset}: no valid depth samples collected.")
        return None

    mean = total_sum / total_values
    print(
        f"{dataset}: files={len(sample_paths)}, valid_files={valid_files}, "
        f"values={total_values}, mean={mean:.4f}"
    )
    return DatasetStats(
        bins=bin_store,
        total_values=int(total_values),
        total_sum=float(total_sum),
        total_sq_sum=float(total_sq_sum),
        sampled_units=len(sample_paths),
        valid_units=valid_files,
    )


def process_native_dataset(
    dataset: str, entry: Dict[str, Any], sample_limit: Optional[int]
) -> Optional[DatasetStats]:
    if _build_native_depth_dataset is None:
        raise RuntimeError(
            "Native dataset support is unavailable because util.data_utils "
            "could not be imported."
        )
    ds = _build_native_depth_dataset(entry, DEFAULT_NATIVE_IMAGE_SIZE, DEFAULT_NATIVE_MAX_DEPTH)
    total_len = len(ds)
    if total_len == 0:
        print(f"{dataset}[native]: dataset is empty.")
        return None

    sample_indices = select_indices(total_len, sample_limit)
    if not sample_indices:
        print(f"{dataset}[native]: no indices sampled.")
        return None

    bin_store: Dict[int, int] = defaultdict(int)
    total_values = 0
    total_sum = 0.0
    total_sq_sum = 0.0
    valid_samples = 0

    progress = tqdm(sample_indices, desc=f"{dataset}[native]", unit="sample")
    for idx in progress:
        try:
            sample = ds[idx]
        except Exception as exc:
            print(f"{dataset}[native]: failed to load index {idx}: {exc}")
            continue
        values = extract_values_from_sample(sample)
        if values.size == 0:
            continue
        values = adjust_values_for_dataset(dataset, values)
        valid_samples += 1
        bins_data, value_count, sum_values, sq_sum_values = histogram_from_values(values)
        total_values += value_count
        total_sum += sum_values
        total_sq_sum += sq_sum_values
        for bin_idx, cnt in bins_data:
            bin_store[bin_idx] += cnt
    progress.close()

    if total_values == 0 or not bin_store:
        print(f"{dataset}[native]: no valid depth samples collected.")
        return None

    mean = total_sum / total_values
    print(
        f"{dataset}[native]: samples={len(sample_indices)}, valid_samples={valid_samples}, "
        f"values={total_values}, mean={mean:.4f}"
    )
    return DatasetStats(
        bins=bin_store,
        total_values=int(total_values),
        total_sum=float(total_sum),
        total_sq_sum=float(total_sq_sum),
        sampled_units=len(sample_indices),
        valid_units=valid_samples,
    )


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


def ensure_output_dir(path: Path) -> Path:
    outdir = path.expanduser()
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create output directory {outdir}. "
            "Provide a writable path via --output-dir."
        ) from exc
    return outdir


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
    sample_limit: Optional[int] = args.sample_limit
    if sample_limit is not None and sample_limit <= 0:
        sample_limit = None
    global HIST_STEP, HIST_MIN
    if args.hist_step is not None and args.hist_step > 0:
        HIST_STEP = float(args.hist_step)
    else:
        HIST_STEP = DEFAULT_HIST_STEP
    HIST_MIN = float(args.hist_min)
    selected_datasets = normalize_dataset_selection(args.datasets)
    outdir = ensure_output_dir(Path(args.output_dir))
    aggregates: Dict[str, Dict[str, object]] = {}

    for dataset in selected_datasets:
        collected = False
        if dataset in DATASET_FILELISTS:
            stats = process_cache_dataset(
                dataset, DATASET_FILELISTS[dataset], args.workers, sample_limit
            )
            if stats is not None:
                merge_dataset_stats(dataset, stats, aggregates)
                collected = True
        if dataset in NATIVE_DATASETS:
            stats = process_native_dataset(dataset, NATIVE_DATASETS[dataset], sample_limit)
            if stats is not None:
                merge_dataset_stats(dataset, stats, aggregates)
                collected = True
        if not collected:
            print(f"{dataset}: no data sources processed.")

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
