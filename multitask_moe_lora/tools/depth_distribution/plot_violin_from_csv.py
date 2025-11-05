#!/usr/bin/env python3
"""
Generate violin plots from cached depth histograms.

The script consumes the CSV emitted by ``compute_depth_histograms.py``,
reconstructs approximate depth samples via weighted sampling, and renders a
violin plot per dataset.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


class HistogramRecord:
    __slots__ = ("zero", "bins", "ge", "total", "nan", "neg", "ge_threshold")

    def __init__(self) -> None:
        self.zero = 0
        self.bins: List[Tuple[float, float, int]] = []
        self.ge = 0
        self.total = 0
        self.nan = 0
        self.neg = 0
        self.ge_threshold = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot violin charts from depth histogram CSV data."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CSV produced by compute_depth_histograms.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (PNG/PDF/etc). If omitted, shows the figure.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200_000,
        help="Maximum number of samples per dataset (0 => use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling within bins.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="Figure width in inches (default: 10).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=6.0,
        help="Figure height in inches (default: 6).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving to file (default: 150).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Depth Distribution per Dataset",
        help="Plot title.",
    )
    return parser.parse_args()


def load_histograms(csv_path: Path) -> Dict[str, HistogramRecord]:
    records: "OrderedDict[str, HistogramRecord]" = OrderedDict()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            rec = records.setdefault(dataset, HistogramRecord())

            count = int(row["count"])
            rec.total = int(row.get("total_count", rec.total or 0))
            rec.nan = int(row.get("nan_count", rec.nan or 0))
            rec.neg = int(row.get("negative_count", rec.neg or 0))

            label = row["bin_label"]
            if label == "zero":
                rec.zero = count
            elif label.startswith(">="):
                rec.ge = count
                try:
                    rec.ge_threshold = float(label.replace(">=", ""))
                except ValueError:
                    rec.ge_threshold = None
            else:
                start = float(row["bin_start"])
                end = float(row["bin_end"])
                rec.bins.append((start, end, count))

    return records


def sample_from_histogram(
    record: HistogramRecord,
    rng: np.random.Generator,
    max_samples: int,
) -> np.ndarray:
    components: List[Tuple[str, float, float, int]] = []
    components.append(("zero", 0.0, 0.0, record.zero))
    components.extend(("bin", start, end, count) for start, end, count in record.bins)
    ge_threshold = record.ge_threshold if record.ge_threshold is not None else (
        record.bins[-1][1] if record.bins else 0.3
    )
    components.append(("ge", ge_threshold, ge_threshold, record.ge))

    total_count = sum(count for _, _, _, count in components)
    if total_count == 0:
        return np.empty((0,), dtype=np.float32)

    sample_cap = total_count if max_samples <= 0 else min(total_count, max_samples)
    weights = np.array([count for _, _, _, count in components], dtype=np.float64)
    weights_sum = weights.sum()
    if weights_sum == 0:
        return np.empty((0,), dtype=np.float32)
    weights /= weights_sum
    draw_counts = rng.multinomial(sample_cap, weights)

    samples: List[np.ndarray] = []
    for (kind, start, end, _), draw_count in zip(components, draw_counts):
        if draw_count == 0:
            continue
        if kind == "zero":
            samples.append(np.zeros(draw_count, dtype=np.float32))
        elif kind == "ge":
            samples.append(np.full(draw_count, start, dtype=np.float32))
        else:
            values = rng.uniform(low=start, high=end, size=draw_count).astype(np.float32)
            samples.append(values)
    if not samples:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(samples, axis=0)


def create_violin_plot(
    histogram_samples: Dict[str, np.ndarray],
    width: float,
    height: float,
    dpi: int,
    title: str,
    output_path: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    datasets = list(histogram_samples.keys())
    data = [histogram_samples[name] for name in datasets]

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    violin = ax.violinplot(
        data,
        showmeans=True,
        showextrema=False,
        showmedians=True,
    )
    for body in violin["bodies"]:
        body.set_alpha(0.6)

    ax.set_xticks(range(1, len(datasets) + 1))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    histograms = load_histograms(args.csv)
    if not histograms:
        raise RuntimeError(f"No datasets found in {args.csv}")

    samples = OrderedDict()
    for dataset, record in histograms.items():
        samples[dataset] = sample_from_histogram(record, rng, args.max_samples)

    create_violin_plot(
        samples,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        title=args.title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

