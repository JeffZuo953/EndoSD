#!/usr/bin/env python3
"""
Plot violin and box plots for depth distributions reconstructed from histogram CSVs.

The script samples synthetic depth values from per-dataset histograms to avoid
loading the original tensors. Dataset selection is configurable via --datasets and
is derived at runtime from the available CSV files.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

HIST_STEP_DEFAULT = 0.005
SAMPLES_PER_DATASET = 5000
SUMMARY_FILENAME_DEFAULT = "depth_summary_stats.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot depth statistics (violin/box) from histogram CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/dataset_metadata"),
        help="Directory containing histogram/summary CSVs.",
    )
    parser.add_argument(
        "--histogram",
        type=str,
        default="depth_histogram_bins.csv",
        help="Histogram CSV filename (within data-dir unless absolute).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=SUMMARY_FILENAME_DEFAULT,
        help="Summary stats CSV filename (within data-dir unless absolute).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        action="append",
        help="Datasets to plot (comma-separated). Default: all datasets in the CSVs.",
    )
    parser.add_argument(
        "--hist-step",
        type=float,
        default=HIST_STEP_DEFAULT,
        help="Histogram bin width used when sampling.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLES_PER_DATASET,
        help="Max number of synthetic samples per dataset.",
    )
    parser.add_argument(
        "--violin-output",
        type=str,
        default="depth_violin.png",
        help="Filename for the violin plot (within data-dir unless absolute).",
    )
    parser.add_argument(
        "--box-output",
        type=str,
        default="depth_boxplot.png",
        help="Filename for the box plot (within data-dir unless absolute).",
    )
    return parser.parse_args()


def resolve_path(base: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    return path if path.is_absolute() else base / path


def gather_dataset_names(histogram_path: Path, summary_path: Path | None) -> List[str]:
    if not histogram_path.exists():
        raise FileNotFoundError(f"Histogram CSV not found: {histogram_path}")

    ordered: List[str] = []
    seen: set[str] = set()

    def add_from_csv(path: Path) -> None:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if "dataset" not in reader.fieldnames:
                raise ValueError(f"'dataset' column missing from {path}")
            for row in reader:
                name = row["dataset"].strip()
                if not name or name in seen:
                    continue
                ordered.append(name)
                seen.add(name)

    if summary_path:
        if summary_path.exists():
            add_from_csv(summary_path)
        else:
            print(f"Warning: Summary CSV not found: {summary_path}")

    add_from_csv(histogram_path)
    return ordered


def normalize_dataset_selection(raw: List[str] | None, available: List[str]) -> List[str]:
    if not available:
        raise ValueError("No datasets discovered in the provided CSV files.")
    if not raw:
        return list(available)
    selections: List[str] = []
    for chunk in raw:
        selections.extend([p.strip() for p in chunk.split(",") if p.strip()])
    valid_inputs = set(available)
    seen = set()
    ordered: List[str] = []
    for name in selections:
        if name not in valid_inputs:
            raise ValueError(
                f"Unknown dataset '{name}'. Allowed names: {', '.join(available)}"
            )
        if name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    return ordered


def load_histograms(
    csv_path: Path, hist_step: float, selected: List[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Histogram CSV not found: {csv_path}")
    if not selected:
        return {}

    accumulators: Dict[str, defaultdict[int, int]] = {}
    selected_set = set(selected)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"].strip()
            if dataset not in selected_set:
                continue
            count = int(row["count"])
            if count <= 0:
                continue
            bin_start = float(row["bin_start"])
            bin_index = int(round(bin_start / hist_step))
            acc = accumulators.setdefault(dataset, defaultdict(int))
            acc[bin_index] += count

    datasets: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset, bins in accumulators.items():
        indices = np.array(sorted(bins.keys()), dtype=np.int64)
        starts = indices.astype(np.float64) * hist_step
        ends = starts + hist_step
        counts = np.array([bins[idx] for idx in indices], dtype=np.int64)
        datasets[dataset] = {"starts": starts, "ends": ends, "counts": counts}
    return datasets


def sample_from_histogram(
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    counts: np.ndarray,
    hist_step: float,
    rng: np.random.Generator,
    max_samples: int,
) -> np.ndarray:
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.float32)
    sample_size = min(total, max_samples)
    probabilities = counts / total
    chosen_bins = rng.choice(len(bin_starts), size=sample_size, p=probabilities)
    offsets = rng.random(sample_size) * hist_step
    return bin_starts[chosen_bins] + offsets


def prepare_datasets(
    histogram_path: Path,
    hist_step: float,
    selected: List[str],
    max_samples: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    histograms = load_histograms(histogram_path, hist_step, selected)
    sampled: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for dataset in selected:
        entry = histograms.get(dataset)
        if entry is None:
            missing.append(dataset)
            continue
        data = sample_from_histogram(
            entry["starts"], entry["ends"], entry["counts"], hist_step, rng, max_samples
        )
        if data.size == 0:
            continue
        sampled[dataset] = data
    if missing:
        print(
            "Warning: datasets not found in histogram CSV and skipped: "
            + ", ".join(missing)
        )
    return sampled


def plot_violin(data: Dict[str, np.ndarray], output: Path) -> None:
    if not data:
        print("No data for violin plot; skipping.")
        return
    labels = list(data.keys())
    values = [data[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    parts = ax.violinplot(values, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.7)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Depth")
    ax.set_title("Depth Distribution Violin Plot")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_box(data: Dict[str, np.ndarray], output: Path) -> None:
    if not data:
        print("No data for box plot; skipping.")
        return
    labels = list(data.keys())
    values = [data[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    boxplot_kwargs = {"showmeans": True}
    try:
        # Matplotlib 3.9 renamed 'labels' to 'tick_labels'; prefer the new name if available.
        ax.boxplot(values, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(values, labels=labels, **boxplot_kwargs)
    ax.set_ylabel("Depth")
    ax.set_title("Depth Distribution Box Plot")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    histogram_path = resolve_path(data_dir, args.histogram)
    summary_path = resolve_path(data_dir, args.summary) if args.summary else None
    violin_path = resolve_path(data_dir, args.violin_output)
    box_path = resolve_path(data_dir, args.box_output)
    available_datasets = gather_dataset_names(histogram_path, summary_path)
    selected = normalize_dataset_selection(args.datasets, available_datasets)

    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_samples = prepare_datasets(
        histogram_path,
        args.hist_step,
        selected=selected,
        max_samples=args.samples,
    )
    plot_violin(dataset_samples, violin_path)
    plot_box(dataset_samples, box_path)
    print(f"Plots saved to {violin_path} and {box_path}")


if __name__ == "__main__":
    main()
