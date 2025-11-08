#!/usr/bin/env python3
"""
Extract per-sequence camera intrinsics for EndoVis2018 rectified data.

The rectification files placed under ``output_rectified/camera`` contain projection
matrices (P1, P2) for each frame. This utility parses the P1 matrix for every frame,
checks whether the intrinsics remain constant within a sequence, and writes the
aggregated parameters to ``dataset_metadata/endovis2018_intrinsics.csv``.

Usage:
    python tools/extract_endovis2018_intrinsics.py \
        --rectified-root /data/ziyi/multitask/data/LS/EndoVis2018/output_rectified

The script will report any sequences whose P1 varies across frames and emit the
aggregated intrinsics table. Optional validation of segmentation masks can be
performed via ``--check-masks``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

RE_CAMERA_FILE = re.compile(r"seq[_-]?(\d+)_frame(\d+)\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class SequenceEntry:
    sequence: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _parse_size(value: str) -> Tuple[int, int]:
    """
    Parse the ``size: (W, H)`` line emitted by OpenCV's stereoRectify.
    """
    match = re.search(r"\((\d+)\s*,\s*(\d+)\)", value)
    if not match:
        raise ValueError(f"Unable to parse image size from line: {value!r}")
    width, height = match.groups()
    return int(width), int(height)


def _parse_p1_matrix(lines: Iterable[str]) -> np.ndarray:
    """
    Given an iterator positioned right after a ``P1:`` line, parse the next three
    bracketed rows into a 3x4 projection matrix.
    """
    matrix_rows: List[np.ndarray] = []
    for _ in range(3):
        try:
            row = next(lines)
        except StopIteration as exc:
            raise ValueError("Unexpected end of file while parsing P1 matrix.") from exc
        row_clean = row.strip().lstrip("[").rstrip("]")
        values = np.fromstring(row_clean, sep=" ")
        if values.size != 4:
            raise ValueError(f"Expected 4 values per P1 row, got {values.size}: {row.strip()}")
        matrix_rows.append(values)
    matrix = np.vstack(matrix_rows)
    if matrix.shape != (3, 4):
        raise ValueError(f"Unexpected P1 matrix shape {matrix.shape}")
    return matrix


def parse_camera_file(path: Path) -> Tuple[str, SequenceEntry]:
    """
    Parse a single ``camera/*.txt`` file and return the sequence id along with
    the extracted intrinsics.
    """
    match = RE_CAMERA_FILE.search(path.name)
    if not match:
        raise ValueError(f"Unrecognised camera filename pattern: {path.name}")
    seq_digits = match.group(1)

    width = height = None
    p1_matrix: Optional[np.ndarray] = None

    with path.open("r", encoding="utf-8") as f:
        lines = iter(f.readlines())
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("size"):
                width, height = _parse_size(stripped)
            elif stripped.startswith("P1:"):
                p1_matrix = _parse_p1_matrix(lines)
            if width is not None and p1_matrix is not None:
                break

    if width is None or height is None:
        raise ValueError(f"Failed to locate 'size:' line in {path}")
    if p1_matrix is None:
        raise ValueError(f"Failed to parse P1 matrix in {path}")

    intrinsic = p1_matrix[:, :3]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    entry = SequenceEntry(sequence=seq_digits, width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return seq_digits, entry


def gather_intrinsics(camera_dir: Path) -> Dict[str, List[SequenceEntry]]:
    """
    Aggregate intrinsics for all files in ``camera_dir``.
    """
    if not camera_dir.exists():
        raise FileNotFoundError(f"Camera directory not found: {camera_dir}")
    seq_entries: Dict[str, List[SequenceEntry]] = defaultdict(list)
    for txt_path in sorted(camera_dir.glob("*.txt")):
        seq_id, entry = parse_camera_file(txt_path)
        seq_entries[seq_id].append(entry)
    if not seq_entries:
        raise RuntimeError(f"No camera calibration files found under {camera_dir}")
    return seq_entries


def summarise_sequences(seq_entries: Dict[str, List[SequenceEntry]]) -> Dict[str, SequenceEntry]:
    """
    Check that intrinsics are constant per sequence and resolve a single entry.
    """
    resolved: Dict[str, SequenceEntry] = {}
    for seq, entries in sorted(seq_entries.items(), key=lambda x: int(x[0])):
        if not entries:
            continue
        base = entries[0]
        mismatch = False
        for other in entries[1:]:
            if (
                not math.isclose(base.fx, other.fx, rel_tol=1e-6, abs_tol=1e-3)
                or not math.isclose(base.fy, other.fy, rel_tol=1e-6, abs_tol=1e-3)
                or not math.isclose(base.cx, other.cx, rel_tol=1e-6, abs_tol=1e-3)
                or not math.isclose(base.cy, other.cy, rel_tol=1e-6, abs_tol=1e-3)
                or base.width != other.width
                or base.height != other.height
            ):
                mismatch = True
                break
        if mismatch:
            print(f"[WARN] Sequence {seq} has varying P1 matrices ({len(entries)} samples).")
        resolved[seq] = base
    return resolved


def write_metadata(entries: Dict[str, SequenceEntry], output_csv: Path) -> None:
    """
    Persist the aggregated intrinsics to a CSV file.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "width", "height", "fx", "fy", "cx", "cy"])
        for seq, entry in sorted(entries.items(), key=lambda x: int(x[0])):
            writer.writerow(
                [
                    seq,
                    entry.width,
                    entry.height,
                    f"{entry.fx:.8f}",
                    f"{entry.fy:.8f}",
                    f"{entry.cx:.8f}",
                    f"{entry.cy:.8f}",
                ]
            )
    print(f"Wrote intrinsics for {len(entries)} sequences to {output_csv}")


def _check_mask_values(mask_dir: Path) -> None:
    """
    Optionally validate that segmentation masks are confined to the 0-9 range.
    """
    from PIL import Image  # Lazy import to avoid mandatory dependency.

    if not mask_dir.exists():
        print(f"[WARN] Mask directory not found: {mask_dir}")
        return

    unique_values: set[int] = set()
    samples_checked = 0
    for mask_path in mask_dir.glob("*.png"):
        with Image.open(mask_path) as img:
            arr = np.array(img, dtype=np.uint16)
        unique_values.update(np.unique(arr).tolist())
        samples_checked += 1
        if samples_checked >= 512:
            break

    if not unique_values:
        print(f"[WARN] No mask samples found under {mask_dir}")
        return

    min_val = min(unique_values)
    max_val = max(unique_values)
    print(f"Mask value check ({samples_checked} samples): min={min_val}, max={max_val}, values={sorted(unique_values)}")
    if max_val > 9:
        print("[WARN] Detected mask ids > 9; investigate remapping requirements.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate EndoVis2018 intrinsics from rectified camera files.")
    parser.add_argument(
        "--rectified-root",
        type=Path,
        required=True,
        help="Path to 'output_rectified' directory containing the 'camera' subdirectory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to dataset_metadata/endovis2018_intrinsics.csv within the repo.",
    )
    parser.add_argument(
        "--check-masks",
        action="store_true",
        help="If set, sample masks from 'left_mask_reid' to verify label range.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rectified_root = args.rectified_root
    camera_dir = rectified_root / "camera"
    sequences = gather_intrinsics(camera_dir)
    resolved = summarise_sequences(sequences)

    output_path = args.output
    if output_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        output_path = repo_root / "dataset_metadata" / "endovis2018_intrinsics.csv"

    write_metadata(resolved, output_path)

    if args.check_masks:
        mask_dir = rectified_root / "left_mask_reid"
        _check_mask_values(mask_dir)


if __name__ == "__main__":
    main()
