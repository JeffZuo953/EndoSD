#!/usr/bin/env python3
"""Visualize demo ground-truth samples into composite PNGs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Dict

import cv2
import numpy as np
import torch


# Ensure we can reuse the shared palette helper
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util.palette import get_palette  # type: ignore


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize demo GT samples into composite PNGs")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more .pt files or text files listing .pt samples (as used by demo validation)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/data/ziyi/multitask/gt_sample",
        help="Directory where visualization PNGs will be written",
    )
    parser.add_argument(
        "--base-prefix",
        type=str,
        default="/data/ziyi/multitask",
        help="Optional base prefix to trim from image paths when recreating subdirectories",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples to export (useful for smoke tests)",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional cap on number of visuals saved per dataset (identified by output subdirectory)",
    )
    return parser.parse_args()


def compute_dataset_key_from_path(path: Path, base_prefix: Path | None) -> str:
    rel: Path | str
    try:
        if base_prefix is not None:
            rel = path.relative_to(base_prefix)
        else:
            rel = path
    except Exception:
        resolved = path.resolve()
        if base_prefix is not None:
            try:
                rel = resolved.relative_to(base_prefix.resolve())
            except Exception:
                rel = resolved
        else:
            rel = resolved

    parts = list(Path(rel).parts if isinstance(rel, (str, bytes)) else rel.parts)
    if parts and parts[0] == "data":
        parts = parts[1:]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if parts:
        return parts[0]
    return Path(rel).stem if isinstance(rel, (str, bytes)) else path.stem


def collect_pt_files(paths: Sequence[str],
                     base_prefix: Path | None,
                     limit_per_dataset: int | None) -> tuple[List[Path], Set[str]]:
    pt_files: List[Path] = []
    expected_keys: Set[str] = set()
    local_counts: Dict[str, int] = {}
    for item in paths:
        raw_path = Path(item).expanduser()
        if not raw_path.exists():
            print(f"[WARN] Input not found: {raw_path}")
            continue
        dataset_key_input = compute_dataset_key_from_path(raw_path.parent, base_prefix)
        expected_keys.add(dataset_key_input)
        if raw_path.suffix.lower() == ".txt":
            remaining = None
            if limit_per_dataset is not None:
                current = local_counts.get(dataset_key_input, 0)
                remaining = max(limit_per_dataset - current, 0)
                if remaining == 0:
                    continue
            with raw_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    pt_files.append(Path(line).expanduser().resolve())
                    if limit_per_dataset is not None:
                        local_counts[dataset_key_input] = local_counts.get(dataset_key_input, 0) + 1
                        remaining -= 1
                        if remaining == 0:
                            break
        elif raw_path.suffix.lower() in {".pt", ".pth"}:
            if limit_per_dataset is not None and local_counts.get(dataset_key_input, 0) >= limit_per_dataset:
                continue
            pt_files.append(raw_path.resolve())
            if limit_per_dataset is not None:
                local_counts[dataset_key_input] = local_counts.get(dataset_key_input, 0) + 1
        else:
            print(f"[WARN] Unsupported input type (skipped): {raw_path}")
    # Deduplicate while preserving order
    unique: List[Path] = []
    seen = set()
    for path in pt_files:
        if path in seen:
            continue
        unique.append(path)
        seen.add(path)
    return unique, expected_keys


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor into uint8 RGB image."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Unexpected image tensor shape: {tuple(tensor.shape)}")
    arr = tensor.numpy()
    arr = arr * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))  # HWC RGB


def palette_colorize(mask: np.ndarray) -> np.ndarray:
    """Map integer segmentation mask to RGB using training palette."""
    palette = get_palette()
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    num_colors = len(palette)
    for idx in range(num_colors):
        color[mask == idx] = palette[idx]
    return color


def ensure_uint16_color(image: np.ndarray) -> np.ndarray:
    """Promote an 8-bit RGB/BGR image to 16-bit for consistent tiling."""
    if image.dtype == np.uint16:
        return image
    return (image.astype(np.uint16) * 257).clip(0, 65535).astype(np.uint16)


def prepare_depth_tile(depth: np.ndarray, max_depth: float | None, mask: np.ndarray | None) -> np.ndarray:
    if max_depth is None or max_depth <= 0:
        max_depth = float(depth.max() if depth.size else 1.0)
    if max_depth <= 0:
        max_depth = 1.0
    normalized = np.clip(depth / max_depth, 0.0, 1.0)
    if mask is not None:
        normalized = normalized * mask.astype(np.float32)
    depth_u16 = (normalized * 65535.0).round().astype(np.uint16)
    return np.repeat(depth_u16[:, :, None], 3, axis=2)


def prepare_mask_tile(mask: np.ndarray, valid_color: bool = False) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if valid_color:
        mask_u16 = np.where(mask, 65535, 0).astype(np.uint16)
        return np.repeat(mask_u16[:, :, None], 3, axis=2)
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    return ensure_uint16_color(np.repeat(mask_u8[:, :, None], 3, axis=2))


def sanitize_path_for_filename(path: Path) -> str:
    sanitized = str(path).replace(os.sep, "_")
    if os.altsep:
        sanitized = sanitized.replace(os.altsep, "_")
    sanitized = sanitized.replace(":", "_")
    return sanitized.strip("_") or path.name


def determine_output_dir(image_path: Path | None, output_root: Path, base_prefix: Path | None) -> Path:
    if image_path is None:
        return output_root
    try:
        if base_prefix is not None:
            rel = image_path.resolve().relative_to(base_prefix.resolve())
            return output_root / rel.parent
    except Exception:
        pass
    return output_root


def tile_panels(panels: Sequence[np.ndarray]) -> np.ndarray:
    if len(panels) != 4:
        raise ValueError("Expected exactly four panels: image, segmentation, valid mask, depth")
    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def visualize_sample(pt_path: Path,
                     output_root: Path,
                     base_prefix: Path | None,
                     limit_per_dataset: int | None,
                     saved_counter: Dict[str, int]) -> Path | None:
    try:
        sample = torch.load(pt_path, map_location="cpu")
    except Exception as exc:
        print(f"[ERROR] Failed to load {pt_path}: {exc}")
        return None

    image_tensor = sample.get("image")
    if image_tensor is None:
        print(f"[WARN] Sample missing image tensor: {pt_path}")
        return None
    depth_tensor = sample.get("depth")
    valid_mask_tensor = sample.get("valid_mask")
    seg_mask_tensor = sample.get("semseg_mask")

    image_rgb = tensor_to_image(image_tensor.cpu())
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if valid_mask_tensor is None:
        valid_mask = np.ones(image_bgr.shape[:2], dtype=bool)
    else:
        valid_mask = valid_mask_tensor.cpu().numpy().astype(bool)

    if depth_tensor is None:
        depth_np = np.zeros(image_bgr.shape[:2], dtype=np.float32)
        max_depth = 1.0
    else:
        depth_np = depth_tensor.cpu().numpy().astype(np.float32)
        max_depth = float(sample.get("max_depth", 0.0))
    depth_tile = prepare_depth_tile(depth_np, max_depth, valid_mask)

    if seg_mask_tensor is not None:
        seg_np = seg_mask_tensor.cpu().numpy().astype(np.int32)
        seg_rgb = palette_colorize(seg_np)
        seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)
    else:
        seg_bgr = np.zeros_like(image_bgr, dtype=np.uint8)

    valid_tile = prepare_mask_tile(valid_mask, valid_color=True)

    panels = [
        ensure_uint16_color(image_bgr),
        ensure_uint16_color(seg_bgr),
        valid_tile,
        depth_tile,
    ]
    composite = tile_panels(panels)

    image_path_str = sample.get("image_path")
    image_path = Path(image_path_str).expanduser() if isinstance(image_path_str, str) else None
    out_dir = determine_output_dir(image_path, output_root, base_prefix)
    stem_source = image_path if image_path is not None else pt_path
    dataset_key = compute_dataset_key_from_path(stem_source.parent if stem_source is not None else pt_path.parent,
                                                base_prefix)
    if limit_per_dataset is not None and saved_counter.get(dataset_key, 0) >= limit_per_dataset:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    output_name = sanitize_path_for_filename(stem_source.with_suffix("")) + ".png"
    output_path = out_dir / output_name
    cv2.imwrite(str(output_path), composite)
    if limit_per_dataset is not None:
        saved_counter[dataset_key] = saved_counter.get(dataset_key, 0) + 1
    return output_path


def main() -> None:
    args = parse_args()
    base_prefix: Path | None = None
    if args.base_prefix:
        base_prefix = Path(args.base_prefix).expanduser().resolve()

    pt_files, expected_keys = collect_pt_files(args.inputs, base_prefix, args.limit_per_dataset)
    if not pt_files:
        print("[WARN] No .pt samples discovered from inputs.")
        return

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    per_dataset_counter: dict[str, int] = {}
    target_keys = expected_keys if args.limit_per_dataset is not None else set()
    for pt_path in pt_files:
        if args.max_samples is not None and processed >= args.max_samples:
            break
        result = visualize_sample(
            pt_path,
            output_root,
            base_prefix,
            args.limit_per_dataset,
            per_dataset_counter,
        )
        if result is not None:
            processed += 1
            print(f"[INFO] Saved {result}")
        if args.limit_per_dataset is not None and target_keys:
            if all(per_dataset_counter.get(key, 0) >= args.limit_per_dataset for key in target_keys):
                break

    print(f"[INFO] Completed {processed} visualizations -> {output_root}")


if __name__ == "__main__":
    main()
