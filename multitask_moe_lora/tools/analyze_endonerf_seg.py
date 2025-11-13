#!/usr/bin/env python3
"""
Quick utility to analyze EndoNeRF segmentation label distributions.

Example:
    python tools/analyze_endonerf_seg.py --seg-root /data/.../EndoNeRF_seg
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def _discover_sequence_dirs(seg_root: Path) -> Iterable[Tuple[str, Path]]:
    for sequence_dir in sorted(seg_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        mask_root = sequence_dir / "mask_label_1_6"
        if not mask_root.exists():
            mask_root = sequence_dir
        if mask_root.is_dir():
            yield sequence_dir.name, mask_root


def analyze_masks(seg_root: Path, limit: int | None = None) -> Dict[str, Dict[str, int]]:
    total_counts: Counter = Counter()
    per_sequence: Dict[str, Counter] = defaultdict(Counter)
    processed = 0

    sequences = list(_discover_sequence_dirs(seg_root))
    if not sequences:
        raise FileNotFoundError(f"No sequences with masks found under {seg_root}")

    for seq_name, mask_root in sequences:
        mask_paths = sorted(mask_root.glob("*.png"))
        if not mask_paths:
            continue
        for mask_path in mask_paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            values, counts = np.unique(mask, return_counts=True)
            local_counter = Counter(dict(zip(values.tolist(), counts.tolist())))
            total_counts.update(local_counter)
            per_sequence[seq_name].update(local_counter)
            processed += 1
            if limit and processed >= limit:
                break
        if limit and processed >= limit:
            break

    summary = {
        "total_pixels": sum(total_counts.values()),
        "label_histogram": dict(sorted(total_counts.items())),
        "sequences": {
            seq: {
                "total_pixels": sum(counter.values()),
                "label_histogram": dict(sorted(counter.items())),
            }
            for seq, counter in per_sequence.items()
        },
        "processed_masks": processed,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze EndoNeRF segmentation label distribution.")
    parser.add_argument("--seg-root", type=Path, required=True, help="Path to EndoNeRF_seg directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of mask images to analyze")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    seg_root = args.seg_root.expanduser().resolve()
    if not seg_root.exists():
        raise FileNotFoundError(f"Segmentation root not found: {seg_root}")

    print(f"[analyze_endonerf_seg] scanning {seg_root}")
    summary = analyze_masks(seg_root, limit=args.limit)

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[analyze_endonerf_seg] saved summary to {args.output}")


if __name__ == "__main__":
    main()
