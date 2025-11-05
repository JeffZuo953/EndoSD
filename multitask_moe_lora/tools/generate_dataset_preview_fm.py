#!/usr/bin/env python3
"""
Generate preview visualizations and basic integrity statistics for the
requested FD depth datasets. For each dataset the script:
  * Collects the list of cached .pt samples (or raw image/depth pairs)
  * Checks that the referenced files exist
  * Samples five evenly spaced entries, converts RGB/depth to preview images
  * Saves the side-by-side RGB/depth visualizations under FM/dataset_preview
  * Records lightweight integrity metrics in FM/dataset_preview/report.json

This script focuses on depth-only data (no LoRA heads) and is intended to
support rapid sanity checks before launching legacy-depth training.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # isort: skip

from dataset.utils import compute_valid_mask, _normalize_dataset_key


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INVALID_DEPTH_COLOR = np.array([220, 32, 32], dtype=np.uint8)
FALLBACK_BACKGROUND = np.array([30, 30, 30], dtype=np.uint8)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "FM" / "dataset_preview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def to_image_uint8(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().float().numpy()
    else:
        array = tensor.astype(np.float32)

    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    # Assume tensor was normalized with ImageNet statistics.
    array = array * IMAGENET_STD + IMAGENET_MEAN
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    return array


def detect_invalid_rgb(rgb: np.ndarray, threshold: int = 16, dataset_key: Optional[str] = None) -> np.ndarray:
    """Detect invalid pixels based on near-black RGB values with dataset-specific rules."""
    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb

    dataset_norm = _normalize_dataset_key(dataset_key) if dataset_key else None

    def compute_near_black(thresh: int) -> np.ndarray:
        channel_max = rgb_u8.max(axis=-1)
        channel_min = rgb_u8.min(axis=-1)
        fully_low = (rgb_u8 <= thresh).all(axis=-1)
        clustered_low = ((channel_max - channel_min) <= 2) & (channel_max <= thresh + 2)
        return fully_low | clustered_low

    if dataset_norm in {"endovis2017", "endovis2018"}:
        near_black = compute_near_black(threshold)
        height, width, _ = rgb_u8.shape
        horiz_border = max(1, min(width, int(round(width * 0.30))))
        vert_border = max(1, min(height, int(round(height * 0.10))))
        rows = np.arange(height).reshape(-1, 1)
        cols = np.arange(width).reshape(1, -1)
        edge_mask = (rows < vert_border) | (rows >= height - vert_border) | \
                    (cols < horiz_border) | (cols >= width - horiz_border)
        return near_black & edge_mask

    if dataset_norm in {"hamlyn", "c3vd", "c3vdv2"}:
        near_black = compute_near_black(3)
        height, width, _ = rgb_u8.shape
        third_h = max(1, int(round(height / 3.0)))
        third_w = max(1, int(round(width / 3.0)))
        rows = np.arange(height).reshape(-1, 1)
        cols = np.arange(width).reshape(1, -1)
        border_mask = (rows < third_h) | (rows >= height - third_h) | \
                      (cols < third_w) | (cols >= width - third_w)
        return near_black & border_mask

    # Default: do not mark pixels invalid based on RGB alone
    return np.zeros(rgb_u8.shape[:2], dtype=bool)


def render_depth_color(
    depth_data: torch.Tensor | np.ndarray | None,
    max_depth: float | None,
    valid_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    if depth_data is None:
        return None

    if isinstance(depth_data, torch.Tensor):
        depth = depth_data.detach().cpu().float().numpy()
    else:
        depth = depth_data.astype(np.float32)

    finite_mask = np.isfinite(depth)
    if valid_mask is not None:
        valid_mask = valid_mask.astype(bool)
        if valid_mask.shape != depth.shape:
            raise ValueError("Valid mask and depth shape mismatch.")
        effective_mask = finite_mask & valid_mask
    else:
        effective_mask = finite_mask

    if not effective_mask.any():
        canvas = np.tile(FALLBACK_BACKGROUND, (depth.shape[0], depth.shape[1], 1))
        if valid_mask is None:
            invalid_mask = ~finite_mask
        else:
            invalid_mask = (~valid_mask) | (~finite_mask)
        canvas[invalid_mask] = INVALID_DEPTH_COLOR
        return canvas.astype(np.uint8)

    valid_depths = depth[effective_mask]
    if valid_depths.size == 0:
        return None

    if not max_depth or max_depth <= 0 or not math.isfinite(max_depth):
        max_depth = float(valid_depths.max())
        if max_depth <= 0:
            max_depth = 1.0

    depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_norm[effective_mask] = np.clip(depth[effective_mask] / max_depth, 0.0, 1.0)
    cmap = plt.get_cmap("magma")
    colored = (cmap(depth_norm)[:, :, :3] * 255.0).astype(np.uint8)

    depth_color = np.tile(FALLBACK_BACKGROUND, (depth.shape[0], depth.shape[1], 1))
    depth_color[effective_mask] = colored[effective_mask]
    depth_color[~effective_mask] = INVALID_DEPTH_COLOR

    return depth_color


def compose_preview(rgb: np.ndarray, depth_color: Optional[np.ndarray]) -> Image.Image:
    rgb_img = Image.fromarray(rgb)

    if depth_color is None:
        return rgb_img

    depth_img = Image.fromarray(depth_color)
    if depth_img.size != rgb_img.size:
        depth_img = depth_img.resize(rgb_img.size, resample=Image.BILINEAR)

    combined = Image.new("RGB", (rgb_img.width * 2, rgb_img.height))
    combined.paste(rgb_img, (0, 0))
    combined.paste(depth_img, (rgb_img.width, 0))
    return combined


def evenly_sample(items: Sequence[str], count: int) -> List[str]:
    if not items:
        return []
    if len(items) <= count:
        return list(items)

    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    picked = []
    last_idx = -1
    for idx in indices:
        idx = int(idx)
        if idx == last_idx:
            idx = min(idx + 1, len(items) - 1)
        picked.append(items[idx])
        last_idx = idx
    return picked


def load_pt_sample(path: Path, fallback_dataset_name: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, float], Optional[float]]:
    data = torch.load(path, map_location="cpu")
    image_tensor = data.get("image")
    depth_tensor = data.get("depth")
    max_depth = data.get("max_depth")

    max_depth_value = None
    if isinstance(max_depth, (int, float)):
        max_depth_value = float(max_depth)
    elif isinstance(max_depth, torch.Tensor):
        max_depth_value = float(max_depth.item())

    rgb = to_image_uint8(image_tensor)
    depth_np: Optional[np.ndarray]
    if depth_tensor is None:
        depth_np = None
    elif isinstance(depth_tensor, torch.Tensor):
        depth_np = depth_tensor.detach().cpu().float().numpy()
    else:
        depth_np = np.asarray(depth_tensor, dtype=np.float32)

    dataset_name_raw = data.get("dataset_name") or fallback_dataset_name
    dataset_key = _normalize_dataset_key(dataset_name_raw) if dataset_name_raw else _normalize_dataset_key(path.as_posix())
    if not dataset_key:
        dataset_key = None

    valid_mask_tensor = data.get("valid_mask")
    if isinstance(valid_mask_tensor, torch.Tensor):
        valid_mask = valid_mask_tensor.detach().cpu().numpy().astype(bool)
    elif valid_mask_tensor is not None:
        valid_mask = np.asarray(valid_mask_tensor, dtype=bool)
    else:
        valid_mask = None

    depth_valid_tensor = data.get("depth_valid_mask")
    if isinstance(depth_valid_tensor, torch.Tensor):
        depth_valid_mask = depth_valid_tensor.detach().cpu().numpy().astype(bool)
    elif depth_valid_tensor is not None:
        depth_valid_mask = np.asarray(depth_valid_tensor, dtype=bool)
    else:
        depth_valid_mask = None

    seg_valid_tensor = data.get("seg_valid_mask")
    if isinstance(seg_valid_tensor, torch.Tensor):
        seg_valid_mask = seg_valid_tensor.detach().cpu().numpy().astype(bool)
    elif seg_valid_tensor is not None:
        seg_valid_mask = np.asarray(seg_valid_tensor, dtype=bool)
    else:
        seg_valid_mask = None

    invalid_rgb = detect_invalid_rgb(rgb, dataset_key=dataset_key)

    if depth_np is not None:
        finite = np.isfinite(depth_np)
        mask_source = depth_valid_mask if depth_valid_mask is not None else valid_mask
        if mask_source is not None:
            combined_valid = finite & mask_source
        else:
            combined_valid = finite & (~invalid_rgb)
    else:
        mask_source = valid_mask if valid_mask is not None else (~invalid_rgb)
        combined_valid = mask_source.copy() if isinstance(mask_source, np.ndarray) else mask_source

    has_nan_image = bool(torch.isnan(image_tensor).any().item()) if isinstance(image_tensor, torch.Tensor) else bool(np.isnan(image_tensor).any())

    if depth_np is not None:
        finite = np.isfinite(depth_np)
        combined_valid = combined_valid & finite
        valid_depth_values = depth_np[combined_valid]
        if valid_depth_values.size > 0:
            depth_min = float(valid_depth_values.min())
            depth_max = float(valid_depth_values.max())
        else:
            depth_min = float("nan")
            depth_max = float("nan")
        has_nan_depth = bool(np.isnan(depth_np).any())
        valid_ratio = float(combined_valid.astype(np.float32).mean())
    else:
        depth_min = float("nan")
        depth_max = float("nan")
        has_nan_depth = False
        combined_valid = ~invalid_rgb
        valid_ratio = float(combined_valid.astype(np.float32).mean())

    stats = {
        "image_min": float(image_tensor.min().item()) if isinstance(image_tensor, torch.Tensor) else float(np.min(image_tensor)),
        "image_max": float(image_tensor.max().item()) if isinstance(image_tensor, torch.Tensor) else float(np.max(image_tensor)),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "has_nan_image": has_nan_image,
        "has_nan_depth": has_nan_depth,
        "valid_ratio": valid_ratio,
        "invalid_ratio": 1.0 - valid_ratio,
    }

    if depth_valid_mask is not None:
        stats["depth_valid_ratio"] = float(np.mean(depth_valid_mask.astype(np.float32)))
    if seg_valid_mask is not None:
        stats["seg_valid_ratio"] = float(np.mean(seg_valid_mask.astype(np.float32)))

    return rgb, depth_np, combined_valid.astype(bool), stats, max_depth_value


def parse_filelist_line(line: str) -> Tuple[Path, Optional[Path]]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Invalid filelist line: {line}")

    image_path = Path(parts[0])
    depth_path = Path(parts[1])
    return image_path, depth_path


def load_raw_image_depth(image_path: Path,
                         depth_path: Path,
                         dataset_name: Optional[str] = None,
                         min_depth: float = 0.0,
                         default_max_depth: float = 0.3) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, float], Optional[float]]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image, dtype=np.uint8)

    depth_array: Optional[np.ndarray] = None
    if depth_path.suffix.lower() == ".npy":
        depth_array = np.load(depth_path).astype(np.float32)
    elif depth_path.suffix.lower() in (".png", ".tif", ".tiff"):
        depth_raw = Image.open(depth_path)
        depth_array = np.array(depth_raw, dtype=np.float32)
    elif depth_path.suffix.lower() == ".exr":
        import cv2
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        depth_array = depth.astype(np.float32) if depth is not None else None

    dataset_key = _normalize_dataset_key(dataset_name) if dataset_name else None

    if depth_array is not None and depth_array.size > 0:
        finite_mask = np.isfinite(depth_array)
        finite_values = depth_array[finite_mask]
        if finite_values.size > 0:
            max_depth_est = float(finite_values.max())
            max_depth_value = max_depth_est if math.isfinite(max_depth_est) else default_max_depth
        else:
            max_depth_value = default_max_depth

        image_norm = torch.from_numpy(((rgb.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD).permute(2, 0, 1).contiguous()
        depth_tensor = torch.from_numpy(depth_array.astype(np.float32))
        valid_mask_tensor = compute_valid_mask(
            image_norm,
            depth_tensor,
            min_depth=min_depth,
            max_depth=max_depth_value,
            dataset_name=dataset_name or dataset_key,
        )
        combined_valid = valid_mask_tensor.cpu().numpy()

        valid_depth_values = depth_array[combined_valid]
        if valid_depth_values.size > 0:
            depth_min = float(valid_depth_values.min())
            depth_max = float(valid_depth_values.max())
        else:
            depth_min = float("nan")
            depth_max = float("nan")
        has_nan_depth = bool(np.isnan(depth_array).any())
    else:
        combined_valid = ~detect_invalid_rgb(rgb, dataset_key=dataset_key)
        depth_min = float("nan")
        depth_max = float("nan")
        has_nan_depth = False
        max_depth_value = None

    rgb_norm = rgb.astype(np.float32) / 255.0
    image_min = float(rgb_norm.min())
    image_max = float(rgb_norm.max())

    valid_ratio = float(combined_valid.astype(np.float32).mean())

    stats = {
        "image_min": image_min,
        "image_max": image_max,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "has_nan_depth": has_nan_depth,
        "has_nan_image": False,
        "valid_ratio": valid_ratio,
        "invalid_ratio": 1.0 - valid_ratio,
    }

    return rgb, depth_array, combined_valid.astype(bool), stats, max_depth_value


def collect_pt_entries(list_paths: Sequence[Path]) -> Tuple[List[Path], Dict[str, int]]:
    found: List[Path] = []
    missing = 0
    for list_path in list_paths:
        if not list_path.exists():
            missing += 1
            continue
        with open(list_path, "r") as f:
            for line in f:
                entry = line.strip()
                if not entry:
                    continue
                sample_path = Path(entry)
                if not sample_path.is_absolute():
                    sample_path = (list_path.parent / sample_path).resolve()
                if sample_path.exists():
                    found.append(sample_path)
                else:
                    missing += 1
    return found, {"missing_files": missing}


def collect_filelist_entries(list_paths: Sequence[Path]) -> Tuple[List[Tuple[Path, Path]], Dict[str, int]]:
    pairs: List[Tuple[Path, Path]] = []
    missing_files = 0
    for list_path in list_paths:
        if not list_path.exists():
            missing_files += 1
            continue
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    image_path, depth_path = parse_filelist_line(line)
                except ValueError:
                    missing_files += 1
                    continue
                if not image_path.is_absolute():
                    image_path = (list_path.parent / image_path).resolve()
                if not depth_path.is_absolute():
                    depth_path = (list_path.parent / depth_path).resolve()

                if image_path.exists() and depth_path.exists():
                    pairs.append((image_path, depth_path))
                else:
                    missing_files += 1
    return pairs, {"missing_files": missing_files}


def collect_hamlyn_pairs(root: Path) -> Tuple[List[Tuple[Path, Path]], Dict[str, int]]:
    image_root = root / "image"
    depth_root = root / "depth"
    pairs: List[Tuple[Path, Path]] = []
    missing = 0

    if not image_root.exists() or not depth_root.exists():
        return pairs, {"missing_files": 1}

    for seq_dir in sorted(image_root.glob("rectified*")):
        depth_dir = depth_root / seq_dir.name
        if not depth_dir.exists():
            missing += 1
            continue
        depth_files = {p.stem: p for p in depth_dir.glob("*.npy")}
        for image_path in sorted(seq_dir.glob("*.jpg")):
            base = image_path.stem
            depth_path = depth_files.get(base)
            if depth_path is None:
                missing += 1
                continue
            pairs.append((image_path, depth_path))
    return pairs, {"missing_files": missing}


DATASETS: List[Dict[str, object]] = [
    {
        "name": "SCARED",
        "category": "Train/LS",
        "mode": "pt_cache",
        "list_paths": ["/home/ziyi/ssde/data/LS/SCARED/cache/train_all_cache.txt"],
    },
    {
        "name": "dVPN",
        "category": "Train/LS",
        "mode": "pt_cache",
        "list_paths": [
            "/home/ziyi/ssde/data/dVPN/cache/train_all_cache.txt",
        ],
    },
    {
        "name": "StereoMIS",
        "category": "Train/LS",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/LS/StereoMIS/cache_pt/all_cache.txt",
        ],
    },
    {
        "name": "EndoVis2017",
        "category": "Train/LS",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/LS/EndoVis2017/cache_pt/all_cache.txt",
        ],
    },
    {
        "name": "EndoVis2018-ISINet",
        "category": "Train/LS",
        "mode": "filelist",
        "list_paths": [
            "/data/ziyi/multitask/data/LS/EndoVis2018/filelists/all.txt",
        ],
    },
    {
        "name": "C3VDv2",
        "category": "Train/NO",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/NO/c3vdv2/cache/cache.txt",
        ],
    },
    {
        "name": "SimCol",
        "category": "Train/NO",
        "mode": "pt_cache",
        "list_paths": [
            "/home/ziyi/ssde/data/simcol/cache/train_all_cache.txt",
        ],
    },
    {
        "name": "Kidney3D",
        "category": "Train/NO",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/NO/Kidney3D-CT-depth-seg/cache_pt/train_cache.txt",
        ],
    },
    {
        "name": "Hamlyn Dataset",
        "category": "Eval/LS",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/LS/hamlyn/cache_pt/all_cache.txt",
        ],
    },
    {
        "name": "EndoNeRF",
        "category": "Eval/LS",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/LS/EndoNeRF/cache_pt/all_cache.txt",
        ],
    },
    {
        "name": "C3VD",
        "category": "Eval/NO",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/NO/c3vd/cache/all_cache.txt",
        ],
    },
    {
        "name": "EndoMapper",
        "category": "Eval/NO",
        "mode": "pt_cache",
        "list_paths": [
            "/data/ziyi/multitask/data/NO/endomapper_sim/cache/all_cache.txt",
        ],
    },
]


def main() -> None:
    report: List[Dict[str, object]] = []

    for dataset_cfg in DATASETS:
        name = dataset_cfg["name"]
        mode = dataset_cfg["mode"]
        category = dataset_cfg.get("category", "Unknown")
        safe_name = name.replace(" ", "_").replace("/", "_")
        entry: Dict[str, object] = {
            "name": name,
            "category": category,
            "mode": mode,
            "preview_images": [],
            "status": "ok",
        }

        try:
            if mode == "missing":
                expected_path = dataset_cfg.get("expected")
                entry["status"] = "missing"
                entry["message"] = f"Expected dataset path not found: {expected_path}"
                report.append(entry)
                continue

            if mode == "pt_cache":
                list_paths = [
                    Path(p).resolve() if str(p).startswith("/") else (ROOT / p).resolve()
                    for p in dataset_cfg.get("list_paths", [])
                ]
                samples, stats_meta = collect_pt_entries(list_paths)
            elif mode == "filelist":
                list_paths = [
                    Path(p).resolve() if str(p).startswith("/") else (ROOT / p).resolve()
                    for p in dataset_cfg.get("list_paths", [])
                ]
                samples_pairs, stats_meta = collect_filelist_entries(list_paths)
                samples = samples_pairs
            elif mode == "hamlyn":
                root = Path(dataset_cfg["root"]).resolve()
                samples_pairs, stats_meta = collect_hamlyn_pairs(root)
                samples = samples_pairs
            else:
                raise ValueError(f"Unsupported mode for dataset {name}: {mode}")

            entry.update(stats_meta)
            entry["total_entries"] = len(samples)

            if len(samples) == 0:
                entry["status"] = "empty"
                report.append(entry)
                continue

            sampled_items = evenly_sample(samples, 5)

            depth_stats: List[Dict[str, float]] = []

            for idx, sample in enumerate(sampled_items):
                if mode == "pt_cache":
                    sample_path = Path(sample)
                    rgb, depth_array, valid_mask_np, stats, sample_max_depth = load_pt_sample(sample_path, fallback_dataset_name=name)
                else:
                    if isinstance(sample, tuple):
                        image_path, depth_path = sample
                    else:
                        raise TypeError("Unexpected sample type for non-pt cache")
                    rgb, depth_array, valid_mask_np, stats, sample_max_depth = load_raw_image_depth(Path(image_path), Path(depth_path), dataset_name=name)

                depth_color = render_depth_color(depth_array, sample_max_depth, valid_mask_np if depth_array is not None else None)

                preview_image = compose_preview(rgb, depth_color)
                out_path = OUTPUT_DIR / f"{safe_name}_{idx + 1}.png"
                preview_image.save(out_path)

                entry["preview_images"].append(str(out_path.relative_to(ROOT)))
                depth_stats.append(stats)

            if depth_stats:
                summary: Dict[str, float | bool] = {}

                def collect_min(key: str) -> float:
                    values = [stat[key] for stat in depth_stats if key in stat and isinstance(stat[key], (int, float)) and math.isfinite(stat[key])]
                    return float(min(values)) if values else float("nan")

                def collect_max(key: str) -> float:
                    values = [stat[key] for stat in depth_stats if key in stat and isinstance(stat[key], (int, float)) and math.isfinite(stat[key])]
                    return float(max(values)) if values else float("nan")

                summary["image_min"] = collect_min("image_min")
                summary["image_max"] = collect_max("image_max")
                summary["depth_min"] = collect_min("depth_min")
                summary["depth_max"] = collect_max("depth_max")
                summary["has_nan_image"] = any(stat.get("has_nan_image", False) for stat in depth_stats)
                summary["has_nan_depth"] = any(stat.get("has_nan_depth", False) for stat in depth_stats)

                valid_values = [stat["valid_ratio"] for stat in depth_stats if "valid_ratio" in stat]
                if valid_values:
                    summary["valid_ratio_avg"] = float(np.mean(valid_values))
                    summary["valid_ratio_min"] = float(min(valid_values))
                    summary["valid_ratio_max"] = float(max(valid_values))
                    summary["invalid_ratio_avg"] = 1.0 - summary["valid_ratio_avg"]

                entry["stats_sampled"] = summary

        except Exception as exc:
            entry["status"] = "error"
            entry["message"] = str(exc)

        report.append(entry)

    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Dataset preview report saved to {report_path}")


if __name__ == "__main__":
    main()
