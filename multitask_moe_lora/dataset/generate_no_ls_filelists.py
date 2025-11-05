#!/usr/bin/env python3
"""
Generate file lists for datasets located under /data/ziyi/multitask/data/NO and /data/ziyi/multitask/data/LS.

Currently supported dataset types:
    - kidney3d:  Kidney3D-CT-depth-seg (RGB + EXR depth + PNG masks)

The generated text files can be consumed by FileListSegDepthDataset and fed into cache_utils.generate_dataset_cache
to build training caches compatible with the multitask pipeline.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import cv2
import numpy as np


def _iter_kidney3d_samples(root: Path) -> Iterable[Tuple[Path, Path, Path]]:
    """
    Iterate over Kidney3D samples yielding (rgb_path, depth_path, mask_path).

    Expected directory layout:
        root/
          <case_id>/
            withLaser/
              image/Image0001.png
              depth/Image0001.exr
              mask/Image0001.png
            noLaser/
              ...
    """
    if not root.exists():
        raise FileNotFoundError(f"Kidney3D root does not exist: {root}")

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        for variant in ("withLaser", "noLaser"):
            sub_dir = case_dir / variant
            image_dir = sub_dir / "image"
            depth_dir = sub_dir / "depth"
            mask_dir = sub_dir / "mask"
            if not (image_dir.exists() and depth_dir.exists() and mask_dir.exists()):
                continue

            for image_file in sorted(image_dir.glob("*")):
                if not image_file.is_file():
                    continue
                stem = image_file.stem
                depth_file = depth_dir / f"{stem}.exr"
                mask_file = mask_dir / f"{stem}.png"
                if not (depth_file.exists() and mask_file.exists()):
                    # Skip incomplete triplets
                    continue
                yield image_file, depth_file, mask_file


def _write_filelist(entries: Sequence[Tuple[Path, Path, Path]], output: Path, meta: Dict[str, float | str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for image_path, depth_path, mask_path in entries:
            meta_str = " ".join(f"{k}={v}" for k, v in meta.items())
            f.write(f"{image_path} {depth_path} {mask_path} {meta_str}\n")


def generate_kidney3d_filelists(root: Path,
                                train_output: Path,
                                val_output: Path | None,
                                val_ratio: float,
                                max_depth: float,
                                seed: int,
                                val_cases: Optional[Sequence[str]] = None) -> None:
    samples = list(_iter_kidney3d_samples(root))
    if not samples:
        raise RuntimeError(f"No samples found under {root}. Please verify the dataset layout.")

    meta = {"max_depth": max_depth, "source_type": "NO"}
    if val_cases:
        val_cases_set = {case.strip() for case in val_cases if case.strip()}
        train_entries: List[Tuple[Path, Path, Path]] = []
        val_entries: List[Tuple[Path, Path, Path]] = []
        for image_path, depth_path, mask_path in samples:
            case_id = image_path.parts[len(root.parts)]
            if case_id in val_cases_set:
                val_entries.append((image_path, depth_path, mask_path))
            else:
                train_entries.append((image_path, depth_path, mask_path))

        if not train_entries:
            raise RuntimeError("Kidney3D: train split empty when applying val_cases.")
        if val_output and not val_entries:
            raise RuntimeError("Kidney3D: val split empty when applying val_cases.")

        _write_filelist(train_entries, train_output, meta)
        if val_output:
            _write_filelist(val_entries, val_output, meta)
        return

    random.Random(seed).shuffle(samples)
    if val_output is None or val_ratio <= 0:
        _write_filelist(samples, train_output, meta)
        return

    val_count = max(1, int(len(samples) * val_ratio))
    val_entries = samples[:val_count]
    train_entries = samples[val_count:]

    _write_filelist(train_entries, train_output, meta)
    _write_filelist(val_entries, val_output, meta)


_ENDOVIS2017_LABEL_MAP: Dict[str, int] = {
    "bipolar_forceps": 1,
    "maryland_bipolar_forceps": 1,
    "prograsp_forceps": 2,
    "large_needle_driver": 3,
    "vessel_sealer": 4,
    "grasping_retractor": 5,
    "monopolar_curved_scissors": 6,
    "other": 7,
}


def _resolve_endovis2017_label(name: str) -> Optional[int]:
    name = name.replace("_labels", "")
    name = name.lower()
    if name.startswith("left_") or name.startswith("right_"):
        name = name.split("_", 1)[1]
    if name.endswith("_left") or name.endswith("_right"):
        name = name.rsplit("_", 1)[0]
    return _ENDOVIS2017_LABEL_MAP.get(name)


def _collect_endovis2017(root: Path) -> List[Tuple[Path, Path, List[Tuple[Path, int]]]]:
    image_root = root / "image"
    depth_root = root / "depth"
    if not image_root.exists() or not depth_root.exists():
        raise FileNotFoundError(f"EndoVis2017 root missing image/depth folders: {root}")

    samples: List[Tuple[Path, Path, List[Tuple[Path, int]]]] = []
    for left_dir in image_root.glob("**/left_frames"):
        relative = left_dir.relative_to(image_root)
        depth_dir = depth_root / relative
        if not depth_dir.exists():
            continue

        mask_root = left_dir.parent / "ground_truth"
        mask_dirs = [d for d in mask_root.glob("*") if d.is_dir()] if mask_root.exists() else []
        if not mask_dirs:
            continue

        for image_path in sorted(left_dir.glob("frame*.png")):
            frame_token = image_path.stem
            depth_path = depth_dir / f"{frame_token}_depth.npy"
            if not depth_path.exists():
                continue

            mask_sources: List[Tuple[Path, int]] = []
            for mdir in mask_dirs:
                class_id = _resolve_endovis2017_label(mdir.name)
                if class_id is None:
                    continue
                candidate = mdir / f"{frame_token}.png"
                if candidate.exists():
                    mask_sources.append((candidate, class_id))
            if not mask_sources:
                continue

            samples.append((image_path, depth_path, mask_sources))

    return samples


def _ensure_combined_mask(image_path: Path,
                          mask_sources: List[Tuple[Path, int]],
                          cache_root: Path,
                          image_root: Path) -> Path:
    relative = image_path.relative_to(image_root)
    output_path = cache_root / relative.parent / f"{image_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        sample_mask = None
        for src_path, class_id in mask_sources:
            mask_img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                continue
            if mask_img.ndim == 3:
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            if sample_mask is None:
                sample_mask = np.zeros_like(mask_img, dtype=np.uint8)
            sample_mask[mask_img > 0] = class_id

        if sample_mask is None:
            raise RuntimeError(f"Failed to build combined mask for {image_path}")
        cv2.imwrite(str(output_path), sample_mask)

    return output_path


def generate_endovis2017_filelists(root: Path,
                                   train_output: Path,
                                   val_output: Path,
                                   max_depth: float,
                                   val_ratio: float,
                                   seed: int) -> None:
    samples = _collect_endovis2017(root)
    if not samples:
        raise RuntimeError("EndoVis2017: no samples with masks found")

    random.Random(seed).shuffle(samples)
    val_count = max(1, int(len(samples) * val_ratio))
    val_entries = samples[:val_count]
    train_entries = samples[val_count:]

    if not train_entries or not val_entries:
        raise RuntimeError("EndoVis2017: not enough samples after split")

    mask_cache = train_output.parent / "masks"
    image_root = root / "image"
    meta = {"max_depth": max_depth, "source_type": "LS"}
    def convert(entries: List[Tuple[Path, Path, List[Tuple[Path, int]]]]) -> List[Tuple[Path, Path, Path]]:
        converted = []
        for img_path, depth_path, mask_sources in entries:
            mask_path = _ensure_combined_mask(img_path, mask_sources, mask_cache, image_root)
            converted.append((img_path, depth_path, mask_path))
        return converted

    _write_filelist(convert(train_entries), train_output, meta)
    _write_filelist(convert(val_entries), val_output, meta)


def _iter_endovis2018(root: Path, split: str) -> Iterable[Tuple[Path, Path, Path]]:
    image_root = root / "EndoVis2018_ISINet_tool" / "EndoVis_2018_ISINet_tool" / split
    depth_root = root / "depth" / "EndoVis2018_Scene_seg"
    if not image_root.exists():
        raise FileNotFoundError(f"EndoVis2018 split folder not found: {image_root}")

    image_dir = image_root / "images"
    ann_dir = image_root / "annotations"
    if not image_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"EndoVis2018 requires images/annotations under {image_root}")

    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = ann_dir / image_path.name
        if not mask_path.exists():
            continue

        parts = image_path.stem.split("_")
        if len(parts) < 3:
            continue
        seq_id = parts[1]
        frame_token = parts[-1].replace("frame", "")

        depth_dir = depth_root.glob(f"**/seq_{seq_id}")
        depth_folder = None
        for candidate in depth_dir:
            left_frames = candidate / "left_frames"
            if left_frames.exists():
                depth_folder = left_frames
                break
        if depth_folder is None:
            continue

        depth_path = depth_folder / f"frame{frame_token}_depth.npy"
        if not depth_path.exists():
            continue

        yield image_path, depth_path, mask_path


def generate_endovis2018_filelists(root: Path,
                                   train_output: Path,
                                   val_output: Path,
                                   max_depth: float) -> None:
    train_entries = list(_iter_endovis2018(root, "train"))
    val_entries = list(_iter_endovis2018(root, "val"))
    if not train_entries or not val_entries:
        raise RuntimeError("EndoVis2018: insufficient samples in train/val splits")

    meta = {"max_depth": max_depth, "source_type": "LS"}
    _write_filelist(train_entries, train_output, meta)
    _write_filelist(val_entries, val_output, meta)


def _iter_endonerf(root: Path) -> Iterable[Tuple[Path, Path, Path]]:
    base = root
    if (base / "endonerf").exists():
        base = base / "endonerf"

    for sequence_dir in sorted(base.iterdir()):
        if not sequence_dir.is_dir():
            continue
        image_dir = sequence_dir / "images"
        depth_dir = sequence_dir / "depth"
        mask_dir = sequence_dir / "gt_masks"
        if not (image_dir.exists() and depth_dir.exists() and mask_dir.exists()):
            continue

        for img_path in sorted(image_dir.glob("*.png")):
            frame_id = img_path.stem
            depth_path = depth_dir / f"frame-{frame_id}.depth.png"
            mask_path = mask_dir / f"{frame_id}.png"
            if not depth_path.exists():
                # try alternative naming like same filename
                alt_depth = depth_dir / f"{frame_id}.png"
                if alt_depth.exists():
                    depth_path = alt_depth
                else:
                    continue
            if not mask_path.exists():
                continue
            yield img_path, depth_path, mask_path


def generate_endonerf_filelists(root: Path,
                                train_output: Path,
                                val_output: Path,
                                val_ratio: float,
                                max_depth: float,
                                seed: int) -> None:
    samples = list(_iter_endonerf(root))
    if not samples:
        raise RuntimeError("EndoNeRF: no samples found")

    random.Random(seed).shuffle(samples)
    if val_ratio <= 0:
        val_ratio = 0.1

    val_count = max(1, int(len(samples) * val_ratio))
    val_entries = samples[:val_count]
    train_entries = samples[val_count:]

    if not train_entries or not val_entries:
        raise RuntimeError("EndoNeRF: not enough samples after split")

    depth_scale = 255.0 / max_depth
    meta = {"max_depth": max_depth, "depth_scale": depth_scale, "source_type": "LS"}
    _write_filelist(train_entries, train_output, meta)
    _write_filelist(val_entries, val_output, meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset file lists for NO/LS datasets.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    kidney_parser = subparsers.add_parser("kidney3d", help="Kidney3D-CT-depth-seg dataset")
    kidney_parser.add_argument("--root", type=Path, required=True,
                               help="Path to Kidney3D-CT-depth-seg root directory.")
    kidney_parser.add_argument("--train-output", type=Path, required=True,
                               help="Output path for the training file list.")
    kidney_parser.add_argument("--val-output", type=Path, default=None,
                               help="Optional output path for validation file list.")
    kidney_parser.add_argument("--val-ratio", type=float, default=0.1,
                               help="Validation split ratio (default: 0.1). Ignored if --val-output is not provided or --val-cases is set.")
    kidney_parser.add_argument("--val-cases", type=str, default=None,
                               help="Comma separated case ids (e.g., '04,08') to force into validation split.")
    kidney_parser.add_argument("--max-depth", type=float, default=0.05,
                               help="Maximum depth clipping value stored in the file list.")
    kidney_parser.add_argument("--seed", type=int, default=42,
                               help="Random seed for train/val split.")

    ev17_parser = subparsers.add_parser("endovis2017", help="EndoVis2017 depth+seg dataset")
    ev17_parser.add_argument("--root", type=Path, required=True, help="Path to EndoVis2017 dataset root")
    ev17_parser.add_argument("--train-output", type=Path, required=True, help="Output file for training file list")
    ev17_parser.add_argument("--val-output", type=Path, required=True, help="Output file for validation file list")
    ev17_parser.add_argument("--max-depth", type=float, default=0.3)
    ev17_parser.add_argument("--val-ratio", type=float, default=0.1)
    ev17_parser.add_argument("--seed", type=int, default=42)

    ev18_parser = subparsers.add_parser("endovis2018", help="EndoVis2018 depth+seg dataset")
    ev18_parser.add_argument("--root", type=Path, required=True, help="Path to EndoVis2018 dataset root")
    ev18_parser.add_argument("--train-output", type=Path, required=True, help="Output file for training file list")
    ev18_parser.add_argument("--val-output", type=Path, required=True, help="Output file for validation file list")
    ev18_parser.add_argument("--max-depth", type=float, default=0.3)

    nerf_parser = subparsers.add_parser("endonrf", help="EndoNeRF synthetic dataset")
    nerf_parser.add_argument("--root", type=Path, required=True, help="Path to EndoNeRF dataset root")
    nerf_parser.add_argument("--train-output", type=Path, required=True, help="Output file for training file list")
    nerf_parser.add_argument("--val-output", type=Path, required=True, help="Output file for validation file list")
    nerf_parser.add_argument("--val-ratio", type=float, default=0.1)
    nerf_parser.add_argument("--max-depth", type=float, default=0.3)
    nerf_parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "kidney3d":
        val_cases = None
        if getattr(args, "val_cases", None):
            val_cases = [item.strip() for item in args.val_cases.split(",") if item.strip()]
        generate_kidney3d_filelists(
            root=args.root,
            train_output=args.train_output,
            val_output=args.val_output,
            val_ratio=args.val_ratio,
            max_depth=args.max_depth,
            seed=args.seed,
            val_cases=val_cases,
        )
    elif args.dataset == "endovis2017":
        generate_endovis2017_filelists(
            root=args.root,
            train_output=args.train_output,
            val_output=args.val_output,
            max_depth=args.max_depth,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif args.dataset == "endovis2018":
        generate_endovis2018_filelists(
            root=args.root,
            train_output=args.train_output,
            val_output=args.val_output,
            max_depth=args.max_depth,
        )
    elif args.dataset == "endonrf":
        generate_endonerf_filelists(
            root=args.root,
            train_output=args.train_output,
            val_output=args.val_output,
            val_ratio=args.val_ratio,
            max_depth=args.max_depth,
            seed=args.seed,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset type: {args.dataset}")


if __name__ == "__main__":
    main()
