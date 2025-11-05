#!/usr/bin/env python
"""
compare_pth.py  
A utility script to compare two PyTorch ``.pth``/``.pt`` checkpoint files.

It reports
1. Keys that are **only** present in the first file.
2. Keys that are **only** present in the second file.
3. For keys that exist in *both* files, the mean absolute difference between the two tensors.

Usage (from the project root):

    python compare_pth.py path/to/first.pth path/to/second.pth [--topk 20]

``--topk`` controls how many of the most different parameters (by mean abs diff) to show. Default is 20.

The script is intentionally dependency-light (only requires ``torch``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List

import torch


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return a flat state-dict.

    The checkpoint might be:
        * A raw state-dict (common for ``model.state_dict()`` saves).
        * A dict containing a ``"state_dict"`` key (produced by many training scripts).
        * Something else entirely (rare). In such cases we try to treat the object itself
          as the state-dict if possible.
    """
    # PyTorch >=2.6 defaults to ``weights_only=True`` which is safer but can
    # fail for some older checkpoints that contain non-tensor pickled objects.
    # We first try the safe path and fall back (with a warning) if required.
    try:
        obj: Any = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as err:  # noqa: BLE001
        print(
            "[info] Safe load with weights_only=True failed, retrying with weights_only=False (trusted load).",
            file=sys.stderr,
        )
        obj = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        # Common patterns.
        # Common nesting keys used by various training frameworks
        candidate_keys = [
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "module",
        ]
        for ck in candidate_keys:
            if ck in obj and isinstance(obj[ck], dict):
                return obj[ck]  # type: ignore[return-value]
        # If no known key, assume the dict itself is a state-dict.
        return obj  # type: ignore[return-value]

    # Fallback – give up with a helpful error.
    raise TypeError(f"{path} did not contain a usable state_dict. Got type {type(obj)} instead.")


def _mean_abs_diff(t1: torch.Tensor, t2: torch.Tensor) -> float:
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {tuple(t1.shape)} vs {tuple(t2.shape)}")
    # Cast to float32 to avoid dtype issues (e.g., Long) when computing mean
    return (t1.to(torch.float32) - t2.to(torch.float32)).abs().mean().item()


def compare_state_dicts(
    sd1: Dict[str, torch.Tensor],
    sd2: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], List[Tuple[str, float]]]:
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    only_in_1 = sorted(keys1 - keys2)
    only_in_2 = sorted(keys2 - keys1)

    shared = keys1 & keys2
    diffs: List[Tuple[str, float]] = []
    mean1: Dict[str, float] = {}
    mean2: Dict[str, float] = {}

    for k in sorted(shared):
        v1, v2 = sd1[k], sd2[k]
        if isinstance(v1, torch.Tensor):
            mean1[k] = v1.to(torch.float32).mean().item()
        if isinstance(v2, torch.Tensor):
            mean2[k] = v2.to(torch.float32).mean().item()
        if not (isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor)):
            continue
        if v1.shape != v2.shape:
            continue
        diff = _mean_abs_diff(v1, v2)
        diffs.append((k, diff))
    return only_in_1, only_in_2, diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PyTorch checkpoints (pth files).")
    parser.add_argument("file1", type=Path, help="Path to the first .pth file")
    parser.add_argument("file2", type=Path, help="Path to the second .pth file")
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many of the largest mean-absolute-difference parameters to display (default: 20)",
    )

    args = parser.parse_args()

    if not args.file1.exists() or not args.file2.exists():
        sys.exit("One or both checkpoint files do not exist.")

    sd1 = _load_state_dict(args.file1)
    sd2 = _load_state_dict(args.file2)

    only_in_1, only_in_2, diffs = compare_state_dicts(sd1, sd2)

    # Compute per-file tensor means
    mean1 = {k: v.to(torch.float32).mean().item() for k, v in sd1.items() if isinstance(v, torch.Tensor)}
    mean2 = {k: v.to(torch.float32).mean().item() for k, v in sd2.items() if isinstance(v, torch.Tensor)}

    print("\n=== Keys only in", args.file1, "(", len(only_in_1), ") ===")
    for k in only_in_1:
        print(f"  {k} (shape: {tuple(sd1[k].shape)})")

    print("\n=== Keys only in", args.file2, "(", len(only_in_2), ") ===")
    for k in only_in_2:
        print(f"  {k} (shape: {tuple(sd2[k].shape)})")

    # Generate a Markdown table that is also well-aligned in plain text
    all_keys = sorted(set(sd1.keys()) | set(sd2.keys()))
    diff_dict = {k: d for k, d in diffs}

    f1_name = args.file1.name
    f2_name = args.file2.name

    # --- Pre-calculate all cell contents and column widths ---
    headers = ["Key", f"[{f1_name}] Shape", f"Mean", f"[{f2_name}] Shape", f"Mean", "MeanAbsDiff"]

    rows = []
    for k in all_keys:
        s1 = sd1.get(k)
        s2 = sd2.get(k)
        s1_shape = str(tuple(s1.shape)) if hasattr(s1, 'shape') else ""
        s2_shape = str(tuple(s2.shape)) if hasattr(s2, 'shape') else ""
        m1 = f"{mean1.get(k, ''):.6f}" if k in mean1 else ""
        m2 = f"{mean2.get(k, ''):.6f}" if k in mean2 else ""
        diff = f"{diff_dict.get(k, ''):.6f}" if k in diff_dict else ""
        if hasattr(s1, 'shape') and hasattr(s2, 'shape') and s1.shape != s2.shape:
            diff = "Shape Mismatch"
        rows.append([k, s1_shape, m1, s2_shape, m2, diff])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # --- Print Aligned Markdown Table ---
    def print_row(row_data, col_widths, is_header=False):
        cells = [f" {cell:<{width}} " for width, cell in zip(col_widths, row_data)]
        print(f"|{'|'.join(cells)}|")

    print("\n\n## Comparison Table\n")
    print_row(headers, widths)

    # Print separator
    sep_cells = [f":{'-' * (w + 1)}:" for w in widths]
    print(f"|{'|'.join(sep_cells)}|")

    for row in rows:
        print_row(row, widths)


if __name__ == "__main__":
    main()
