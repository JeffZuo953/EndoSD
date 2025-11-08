#!/usr/bin/env python3
"""
Aggregate EndoVis2017 rectified camera intrinsics.

This scans the `Endovis2017_seg_depth` dataset layout, parses the `P1`
projection matrix from every `camera/frameXXX.txt`, verifies per-sequence
consistency, and writes a CSV summary under `dataset_metadata/`.

Example:
    python tools/extract_endovis2017_rectified_intrinsics.py \
        --root /data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/ziyi/multitask/data/LS/EndoVis2017/Endovis2017_seg_depth")
DEFAULT_OUTPUT = REPO_ROOT / "dataset_metadata" / "endovis2017_rectified_intrinsics.csv"


@dataclass
class SequenceIntrinsics:
    split: str
    sequence: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    max_abs_diff: float
    sample_count: int


def _iter_sequences(root: Path) -> Iterable[Tuple[str, Path]]:
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name.lower()
        if split_name not in {"train", "test"}:
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            yield split_name, seq_dir


def _parse_p1_matrix(camera_file: Path) -> np.ndarray:
    with camera_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    p1_lines: List[str] = []
    capture = False
    for raw in lines:
        line = raw.strip()
        if not capture and line.startswith("P1"):
            capture = True
            continue
        if capture:
            if not line or line.startswith("P2:") or line.startswith("Q:"):
                break
            p1_lines.append(line)
            if len(p1_lines) >= 3:
                break
    if len(p1_lines) != 3:
        raise ValueError(f"Failed to parse P1 block from {camera_file}")
    rows: List[List[float]] = []
    for line in p1_lines:
        cleaned = line.replace("[", " ").replace("]", " ").strip()
        if not cleaned:
            continue
        rows.append([float(token) for token in cleaned.split()])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (3, 4):
        raise ValueError(f"P1 matrix of unexpected shape {matrix.shape} in {camera_file}")
    return matrix[:, :3]


def _read_image_shape(left_dir: Path) -> Tuple[int, int]:
    images = sorted(left_dir.glob("frame*.png"))
    if not images:
        raise FileNotFoundError(f"No frames found under {left_dir}")
    sample = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise FileNotFoundError(f"Failed to read {images[0]}")
    height, width = sample.shape[:2]
    return width, height


def collect_intrinsics(root: Path) -> Tuple[List[SequenceIntrinsics], Dict[str, List[str]]]:
    summaries: List[SequenceIntrinsics] = []
    issues: Dict[str, List[str]] = {"missing_cam": [], "missing_left": [], "inconsistent": []}

    for split_name, seq_dir in _iter_sequences(root):
        rectified_root = seq_dir / "output_rectified"
        camera_dir = rectified_root / "camera"
        left_dir = rectified_root / "left"

        if not camera_dir.exists():
            issues["missing_cam"].append(seq_dir.as_posix())
            continue
        if not left_dir.exists():
            issues["missing_left"].append(seq_dir.as_posix())
            continue

        cam_files = sorted(camera_dir.glob("frame*.txt"))
        if not cam_files:
            issues["missing_cam"].append(seq_dir.as_posix())
            continue

        base = _parse_p1_matrix(cam_files[0])
        diffs = []
        for cam_path in cam_files[1:]:
            mat = _parse_p1_matrix(cam_path)
            diff = np.abs(mat - base).max()
            diffs.append(float(diff))
        max_diff = max(diffs, default=0.0)
        if max_diff > 1e-3:
            issues["inconsistent"].append(f"{seq_dir.name} (max diff {max_diff:.2e})")

        width, height = _read_image_shape(left_dir)
        summaries.append(
            SequenceIntrinsics(
                split=split_name,
                sequence=seq_dir.name,
                width=width,
                height=height,
                fx=float(base[0, 0]),
                fy=float(base[1, 1]),
                cx=float(base[0, 2]),
                cy=float(base[1, 2]),
                max_abs_diff=max_diff,
                sample_count=len(cam_files),
            )
        )

    return summaries, issues


def save_csv(summaries: List[SequenceIntrinsics], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["split", "sequence", "width", "height", "fx", "fy", "cx", "cy", "max_abs_diff", "samples"]
        )
        for item in sorted(summaries, key=lambda s: (s.split, s.sequence)):
            writer.writerow(
                [
                    item.split,
                    item.sequence,
                    item.width,
                    item.height,
                    f"{item.fx:.9f}",
                    f"{item.fy:.9f}",
                    f"{item.cx:.9f}",
                    f"{item.cy:.9f}",
                    f"{item.max_abs_diff:.3e}",
                    item.sample_count,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EndoVis2017 rectified intrinsics.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Path to Endovis2017_seg_depth root")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV (defaults to dataset_metadata/endovis2017_rectified_intrinsics.csv)",
    )
    parser.add_argument(
        "--issues-json",
        type=Path,
        default=None,
        help="Optional JSON path to record missing/inconsistent sequences.",
    )
    args = parser.parse_args()

    summaries, issues = collect_intrinsics(args.root)
    if not summaries:
        raise SystemExit("No sequences processed. Check the --root path.")

    save_csv(summaries, args.output)
    print(f"[INFO] Saved intrinsics for {len(summaries)} sequences to {args.output}")

    issue_counts = {key: len(val) for key, val in issues.items()}
    print("[INFO] Issues summary:", issue_counts)
    if args.issues_json:
        args.issues_json.parent.mkdir(parents=True, exist_ok=True)
        with args.issues_json.open("w", encoding="utf-8") as f:
            json.dump({"issues": issues, "counts": issue_counts}, f, indent=2)


if __name__ == "__main__":
    main()
