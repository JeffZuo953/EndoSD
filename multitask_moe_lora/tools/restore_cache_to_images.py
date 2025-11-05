#!/usr/bin/env python3
"""
Restore cached .pt samples (produced by dataset/cache_utils.py) back to image / GT files.

Usage example:
    python tools/restore_cache_to_images.py \
        --input /data/ziyi/multitask/data/LS \
        --input /data/ziyi/multitask/data/NO \
        --output /data/ziyi/multitask/restored_pt

The script will scan recursively for *.pt files under the provided --input roots,
load each sample, reverse the normalization, and export:
    - RGB image (.png)
    - Depth ground-truth (.npy, optional .png if requested)
    - Segmentation mask (.png) when available

Directory structure under --output mirrors the relative path of the original
image file inside the corresponding dataset root (LS / NO).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore cached .pt samples into image/mask/depth files.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more root directories to scan for .pt files (e.g. data/LS data/NO).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root directory where restored files will be written.",
    )
    parser.add_argument(
        "--include-depth-png",
        action="store_true",
        help="Additionally save depth maps as 16-bit PNG (scaled by max_depth).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be processed without writing outputs.",
    )
    return parser.parse_args()


def discover_pt_files(roots: Iterable[Path]) -> list[Path]:
    pt_files: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] Input root does not exist: {root}")
            continue
        for path in root.rglob("*.pt"):
            pt_files.append(path)
    pt_files.sort()
    return pt_files


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized CHW tensor (float32) back to uint8 RGB image.
    """
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Unexpected image tensor shape: {tuple(tensor.shape)}")
    arr = tensor.numpy()
    arr = arr * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    return arr


def make_output_paths(
    image_path: Path,
    pt_path: Path,
    input_roots: list[Path],
    output_root: Path,
) -> tuple[Path, str]:
    """
    Determine destination directory based on image path; fall back to PT path.
    Returns (destination_directory, domain_tag).
    """
    for root in input_roots:
        if image_path.is_absolute():
            try:
                rel = image_path.relative_to(root)
                domain = root.name
                return output_root / domain / rel.parent, domain
            except ValueError:
                continue
    # Fallback: mirror structure under PT directory
    rel_pt = pt_path.parent
    domain = pt_path.anchor if pt_path.is_absolute() else input_roots[0].name
    return output_root / domain / rel_pt, domain


def save_depth(
    depth: np.ndarray,
    max_depth: float,
    out_dir: Path,
    stem: str,
    save_png: bool,
) -> None:
    npy_path = out_dir / f"{stem}_depth.npy"
    np.save(npy_path, depth.astype(np.float32))
    if save_png:
        if max_depth <= 0:
            scale = 1.0
        else:
            scale = 65535.0 / max_depth
        depth_png = np.clip(depth * scale, 0, 65535).astype(np.uint16)
        png_path = out_dir / f"{stem}_depth.png"
        cv2.imwrite(str(png_path), depth_png)


def save_semseg(mask: np.ndarray, out_dir: Path, stem: str) -> None:
    mask_uint8 = mask.astype(np.uint8)
    png_path = out_dir / f"{stem}_semseg.png"
    cv2.imwrite(str(png_path), mask_uint8)


def process_pt_file(
    pt_path: Path,
    input_roots: list[Path],
    output_root: Path,
    save_depth_png: bool,
    dry_run: bool,
) -> None:
    sample = torch.load(pt_path, map_location="cpu")

    # Prefer stored original paths when available
    image_path_str: Optional[str] = sample.get("raw_image_path") or sample.get("image_path")
    image_path = Path(image_path_str) if image_path_str else pt_path.with_suffix(".png")

    dest_dir, _ = make_output_paths(image_path, pt_path, input_roots, output_root)
    if dry_run:
        print(f"[DRY] {pt_path} -> {dest_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    stem = Path(image_path).stem

    # Restore RGB image
    image_tensor: torch.Tensor = sample["image"].cpu()
    image_rgb = tensor_to_image(image_tensor)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_ext = Path(image_path).suffix.lower()
    if img_ext not in {".png", ".jpg", ".jpeg"}:
        img_ext = ".png"
    img_path = dest_dir / f"{stem}{img_ext}"
    cv2.imwrite(str(img_path), image_bgr)

    # Depth (if any)
    if "depth" in sample:
        depth = sample["depth"].cpu().numpy()
        max_depth = float(sample.get("max_depth", 0.0))
        save_depth(depth, max_depth, dest_dir, stem, save_depth_png)

    # Semantic mask
    if "semseg_mask" in sample:
        mask = sample["semseg_mask"].cpu().numpy()
        save_semseg(mask, dest_dir, stem)


def main() -> None:
    args = parse_args()
    input_roots = [Path(p).expanduser().resolve() for p in args.input]
    output_root = Path(args.output).expanduser().resolve()

    pt_files = discover_pt_files(input_roots)
    print(f"Discovered {len(pt_files)} *.pt files under {len(input_roots)} input roots.")
    if not pt_files:
        return

    for pt_path in tqdm(pt_files, desc="Restoring PT samples"):
        try:
            process_pt_file(
                pt_path=pt_path,
                input_roots=input_roots,
                output_root=output_root,
                save_depth_png=args.include_depth_png,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to process {pt_path}: {exc}")


if __name__ == "__main__":
    main()
