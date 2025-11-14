#!/usr/bin/env python3
"""Batch visualize segmentation predictions by overlaying class colors on RGB frames."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

DEFAULT_COLORS: List[tuple[int, int, int]] = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay segmentation masks on RGB frames.")
    parser.add_argument("--seg-root", required=True,
                        help="Directory containing predicted seg PNGs (e.g. runs/.../seg_png)")
    parser.add_argument("--data-root", required=True,
                        help="Original dataset root containing RGB images")
    parser.add_argument("--output-root", required=True,
                        help="Directory to save visualization overlays")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha (default: 0.4)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of images to process")
    return parser.parse_args()


def _resolve_rgb_path(mask_path: Path, seg_root: Path, data_root: Path) -> Path:
    rel = mask_path.relative_to(seg_root)
    rel_str = str(rel)
    if "/depth/" not in rel_str:
        raise ValueError(f"Cannot map mask path {rel} to RGB path (missing /depth/ segment)")
    rel_str = rel_str.replace("/depth/", "/left/", 1)
    rel_str = rel_str.replace("_depth.png", ".png")
    return data_root / rel_str


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_color_map() -> np.ndarray:
    colors = np.array(DEFAULT_COLORS, dtype=np.uint8)
    return colors


def main() -> None:
    args = _parse_args()
    seg_root = Path(args.seg_root).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.output_root).resolve()
    colors = _build_color_map()

    mask_paths = sorted(seg_root.rglob("*.png"))
    if not mask_paths:
        raise SystemExit(f"No PNG masks found under {seg_root}")

    processed = 0
    for mask_path in mask_paths:
        rgb_path = _resolve_rgb_path(mask_path, seg_root, data_root)
        if not rgb_path.exists():
            print(f"[WARN] Missing RGB image for {mask_path.relative_to(seg_root)} -> {rgb_path}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[WARN] Failed to load mask {mask_path}")
            continue
        mask = np.clip(mask, 0, len(colors) - 1)
        color_mask = colors[mask]

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[WARN] Failed to load image {rgb_path}")
            continue
        if color_mask.shape[:2] != rgb.shape[:2]:
            color_mask = cv2.resize(color_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(rgb, 1.0, color_mask, args.alpha, 0)

        vis_rel = mask_path.relative_to(seg_root)
        vis_path = out_root / vis_rel
        _ensure_parent(vis_path)
        cv2.imwrite(str(vis_path), overlay)
        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"Saved {processed} overlay images to {out_root}")


if __name__ == "__main__":
    main()
