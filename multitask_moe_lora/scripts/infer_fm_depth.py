#!/usr/bin/env python3
"""
Foundation Depth (FM) inference helper.

Runs depth + camera-head inference over the FM datasets, saves float16 depth
maps (plus predicted fx/fy/cx/cy) while mirroring dataset directory structure,
and produces a CSV camera report with δ25% / δ50% focal-length accuracy.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PARENT_ROOT = REPO_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from multitask_moe_lora.util.config import TrainingConfig
from multitask_moe_lora.util.data_utils import (  # noqa: E402
    _make_collate_fn,
    create_datasets,
    summarize_loader_composition,
)
from multitask_moe_lora.util.model_setup import (  # noqa: E402
    create_and_setup_model,
    load_weights_from_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FM depth inference and export float16 npz predictions."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the .pth checkpoint to load.",
    )
    parser.add_argument(
        "--output-root",
        default="/data/ziyi/result",
        help="Root directory used to mirror dataset directory structures.",
    )
    parser.add_argument(
        "--dataset-config-name",
        default="fd_depth_fm_v1",
        help="Dataset config defined in util.data_utils (default: fd_depth_fm_v1).",
    )
    parser.add_argument(
        "--dataset-include",
        default="hamlyn,EndoNeRF,C3VD,EndoMapper",
        help="Comma separated dataset names to keep for inference.",
    )
    parser.add_argument(
        "--dataset-modality",
        default="fd",
        choices=["fd", "mt"],
        help="Dataset modality flag used by data_utils.",
    )
    parser.add_argument(
        "--path-transform-name",
        default="none",
        help="Optional path transform name used by data_utils (default: none).",
    )
    parser.add_argument(
        "--local-cache-dir",
        default=None,
        help="Optional local cache root for dataset .pt files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers (default: 8).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=518,
        help="Input resolution when datasets need resizing (default: 518).",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=0.3,
        help="Depth upper bound; predictions beyond this value are nulled.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=1e-6,
        help="Depth lower bound passed to the model.",
    )
    parser.add_argument(
        "--encoder",
        default="vitb",
        help="Backbone encoder (default: vitb).",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=64,
        help="Feature dimension for heads (default: 64).",
    )
    parser.add_argument(
        "--camera-head-mode",
        default="simple",
        choices=["none", "simple", "prolike", "vggtlike", "vggt-like"],
        help="Camera head architecture (default: simple).",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma separated CUDA device indices. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES mask applied before CUDA init (e.g., '0,1').",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use torch.cuda.amp autocast for inference.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing npz files that already exist on disk.",
    )
    parser.add_argument(
        "--log-dir",
        default="runs/infer_fm_depth",
        help="Directory for inference logs.",
    )
    parser.add_argument(
        "--camera-report",
        default=None,
        help="CSV path for camera metrics (default: <output-root>/camera_report.csv).",
    )
    return parser.parse_args()


def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "infer.log"
    formatter = logging.Formatter("[%(asctime)s][%(levelname)5s] %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("fm_infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _parse_name_list(raw_value: Optional[str]) -> Optional[List[str]]:
    if not raw_value:
        return None
    entries = [item.strip() for item in raw_value.split(",") if item.strip()]
    return entries or None


def _build_collate_with_meta(stride: int):
    base_collate = _make_collate_fn(stride)

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = base_collate(batch)
        meta_list: List[Dict[str, Any]] = []
        for item in batch:
            meta_list.append(
                {
                    "height": int(item["image"].shape[-2]),
                    "width": int(item["image"].shape[-1]),
                    "dataset_name": item.get("dataset_name"),
                    "source_type": item.get("source_type"),
                    "original_image_path": item.get("original_image_path"),
                    "cache_path": item.get("image_path"),
                    "depth_path": item.get("depth_path"),
                }
            )
        result["meta"] = meta_list
        return result

    return collate


def _derive_relative_path(meta: Dict[str, Any], dataset_tokens: Sequence[str]) -> str:
    dataset_name = str(meta.get("dataset_name") or meta.get("source_type") or "dataset")
    dataset_name = dataset_name.strip() or "dataset"
    canonical = dataset_name
    needles = [dataset_name.lower()]
    needles.extend(token.lower() for token in dataset_tokens if token)

    candidates = [
        meta.get("original_image_path"),
        meta.get("depth_path"),
        meta.get("cache_path"),
    ]
    rel_candidate: Optional[str] = None
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.expanduser(str(candidate)))
        lowered = normalized.lower()
        match_offset = -1
        for needle in needles:
            idx = lowered.find(needle)
            if idx != -1:
                match_offset = idx + len(needle)
                break
        if match_offset != -1:
            suffix = normalized[match_offset:].lstrip("/\\")
            rel_candidate = os.path.join(canonical, suffix)
            break
        rel_candidate = os.path.join(canonical, os.path.basename(normalized))
        break

    rel_candidate = rel_candidate or os.path.join(canonical, f"sample_{meta.get('cache_path', 'unknown')}")
    base, _ = os.path.splitext(rel_candidate)
    safe = base.replace(":", "_")
    safe = safe.replace("\\", "/").lstrip("/")
    return f"{safe}.npz"


def _prepare_config(args: argparse.Namespace, logger: logging.Logger) -> TrainingConfig:
    config = TrainingConfig()
    config.encoder = args.encoder
    config.features = args.features
    config.num_classes = 1
    config.min_depth = float(args.min_depth)
    config.max_depth = float(args.max_depth)
    config.seg_head_type = "none"
    config.disable_seg_head = True
    config.camera_head_mode = args.camera_head_mode.lower()
    config.mode = "original"
    config.bs = args.batch_size
    config.val_bs = args.batch_size
    config.img_size = args.img_size
    config.dataset_config_name = args.dataset_config_name
    config.dataset_modality = args.dataset_modality
    config.path_transform_name = None if args.path_transform_name.lower() == "none" else args.path_transform_name
    include_list = _parse_name_list(args.dataset_include)
    config.val_dataset_include = include_list
    config.train_dataset_include = None
    config.local_cache_dir = args.local_cache_dir
    config.val_sample_step = 1
    config.val_min_samples_per_dataset = 0
    config.save_path = str(Path(args.log_dir).resolve())
    config.resume_from = args.checkpoint
    config.resume_full_state = False
    config.mixed_precision = args.use_amp
    logger.info("Config prepared: dataset=%s, include=%s", config.dataset_config_name, include_list)
    return config


def _build_loader(dataset, batch_size: int, num_workers: int, encoder: str) -> DataLoader:
    stride = 16 if "dinov3" in encoder.lower() else 14
    collate = _build_collate_with_meta(stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )


def _prepare_model(
    config: TrainingConfig,
    logger: logging.Logger,
    device_ids: Sequence[int],
) -> torch.nn.Module:
    torch.cuda.set_device(device_ids[0])
    model = create_and_setup_model(config, logger)
    if len(device_ids) > 1:
        logger.info("Using DataParallel across GPUs: %s", ",".join(map(str, device_ids)))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        logger.info("Using single GPU: %d", device_ids[0])
    load_weights_from_checkpoint(
        model=model,
        optimizer_depth=None,
        optimizer_seg=None,
        optimizer_camera=None,
        scheduler_depth=None,
        scheduler_seg=None,
        scheduler_camera=None,
        config=config,
        logger=logger,
    )
    model.eval()
    return model


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float16, copy=False)


def _resolve_width_height(
    idx: int,
    camera_sizes: Optional[np.ndarray],
    camera_size_mask: Optional[np.ndarray],
    meta: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    width = None
    height = None
    if camera_sizes is not None and 0 <= idx < len(camera_sizes):
        valid = True
        if camera_size_mask is not None and idx < len(camera_size_mask):
            valid = bool(camera_size_mask[idx])
        if valid:
            width = float(camera_sizes[idx][0])
            height = float(camera_sizes[idx][1])
    if (width is None or width <= 0) and meta.get("width"):
        width = float(meta["width"])
    if (height is None or height <= 0) and meta.get("height"):
        height = float(meta["height"])
    if width is None or width <= 0:
        width = height
    if height is None or height <= 0:
        height = width
    return width, height


def _extract_camera_values_from_norm(
    idx: int,
    norm_tensor: Optional[torch.Tensor],
    width: Optional[float],
    height: Optional[float],
) -> Optional[Tuple[float, float, float, float]]:
    if norm_tensor is None or width is None or height is None or width <= 0 or height <= 0:
        return None
    if idx >= norm_tensor.size(0):
        return None
    vec = norm_tensor[idx]
    fx = float(vec[0].item()) * width
    fy = float(vec[1].item()) * height
    cx = float(vec[2].item()) * width
    cy = float(vec[3].item()) * height
    return fx, fy, cx, cy


def _extract_camera_values_from_raw(
    idx: int,
    raw_tensor: Optional[torch.Tensor],
) -> Optional[Tuple[float, float, float, float]]:
    if raw_tensor is None or idx >= raw_tensor.size(0):
        return None
    mat = raw_tensor[idx]
    if mat.dim() == 1 and mat.numel() >= 4:
        return tuple(float(x) for x in mat[:4])
    if mat.dim() == 2 and mat.size(0) >= 3 and mat.size(1) >= 3:
        fx = float(mat[0, 0].item())
        fy = float(mat[1, 1].item())
        cx = float(mat[0, 2].item())
        cy = float(mat[1, 2].item())
        return fx, fy, cx, cy
    return None


def _resolve_camera_prediction(
    idx: int,
    camera_pred_norm: Optional[torch.Tensor],
    camera_sizes: Optional[np.ndarray],
    camera_size_mask: Optional[np.ndarray],
    meta: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float]]:
    width, height = _resolve_width_height(idx, camera_sizes, camera_size_mask, meta)
    return _extract_camera_values_from_norm(idx, camera_pred_norm, width, height)


def _resolve_camera_ground_truth(
    idx: int,
    gt_raw: Optional[torch.Tensor],
    gt_norm: Optional[torch.Tensor],
    camera_sizes: Optional[np.ndarray],
    camera_size_mask: Optional[np.ndarray],
    meta: Dict[str, Any],
    raw_mask: Optional[np.ndarray],
    norm_mask: Optional[np.ndarray],
) -> Optional[Tuple[float, float, float, float]]:
    valid_raw = True
    if raw_mask is not None and idx < len(raw_mask):
        valid_raw = bool(raw_mask[idx])
    if valid_raw:
        values = _extract_camera_values_from_raw(idx, gt_raw)
        if values is not None:
            return values

    valid_norm = True
    if norm_mask is not None and idx < len(norm_mask):
        valid_norm = bool(norm_mask[idx])
    if not valid_norm:
        return None
    width, height = _resolve_width_height(idx, camera_sizes, camera_size_mask, meta)
    return _extract_camera_values_from_norm(idx, gt_norm, width, height)


def _update_camera_stats(
    stats: Dict[str, Dict[str, float]],
    dataset_name: str,
    pred_fx: float,
    pred_fy: float,
    gt_fx: float,
    gt_fy: float,
) -> None:
    dataset_key = (dataset_name or "unknown").strip() or "unknown"
    entry = stats.setdefault(
        dataset_key,
        {"name": dataset_key, "total": 0, "delta25": 0, "delta50": 0},
    )
    gt_focal = max(1e-6, 0.5 * (abs(gt_fx) + abs(gt_fy)))
    pred_focal = 0.5 * (abs(pred_fx) + abs(pred_fy))
    rel_error = abs(pred_focal - gt_focal) / gt_focal
    entry["total"] += 1
    if rel_error < 0.25:
        entry["delta25"] += 1
    if rel_error < 0.50:
        entry["delta50"] += 1


def _write_camera_report(stats: Dict[str, Dict[str, float]], report_path: Path) -> None:
    if not stats:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = OrderedDict(sorted(stats.items(), key=lambda kv: kv[0].lower()))
    total_entry = {"name": "ALL", "total": 0, "delta25": 0, "delta50": 0}
    for entry in ordered.values():
        total_entry["total"] += entry["total"]
        total_entry["delta25"] += entry["delta25"]
        total_entry["delta50"] += entry["delta50"]
    ordered["__all__"] = total_entry

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "total_images", "delta25_pct", "delta50_pct", "delta25_count", "delta50_count"])
        for entry in ordered.values():
            name = entry["name"]
            total = int(entry["total"])
            d25 = int(entry["delta25"])
            d50 = int(entry["delta50"])
            pct25 = (d25 / total * 100.0) if total > 0 else 0.0
            pct50 = (d50 / total * 100.0) if total > 0 else 0.0
            writer.writerow(
                [
                    name,
                    total,
                    f"{pct25:.2f}",
                    f"{pct50:.2f}",
                    d25,
                    d50,
                ]
            )


def run_inference(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    logger = _setup_logger(log_dir)

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        logger.info("Set CUDA_VISIBLE_DEVICES=%s", args.cuda_devices)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FM inference.")

    gpu_ids = (
        [int(idx) for idx in args.gpu_ids.split(",")]
        if args.gpu_ids
        else list(range(torch.cuda.device_count()))
    )
    if not gpu_ids:
        raise RuntimeError("No GPU ids available or provided.")

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_tokens = _parse_name_list(args.dataset_include) or []
    camera_report_path = Path(args.camera_report) if args.camera_report else output_root / "camera_report.csv"

    config = _prepare_config(args, logger)
    train_depth_dataset, val_depth_dataset, _, _ = create_datasets(config)
    del train_depth_dataset
    if val_depth_dataset is None:
        raise RuntimeError("Failed to build validation dataset for depth inference.")
    loader = _build_loader(val_depth_dataset, args.batch_size, args.num_workers, args.encoder)
    summaries = summarize_loader_composition(loader)
    if summaries:
        logger.info("Loaded %d datasets (%d samples)", len(summaries), summaries[0]["total"])
        for entry in summaries:
            logger.info("  %s (%s): %d", entry["name"], entry.get("dataset_type", "NA"), entry["count"])

    model = _prepare_model(config, logger, gpu_ids)
    amp_dtype = torch.float16
    saved = 0
    camera_stats: Dict[str, Dict[str, float]] = {}

    def _call_model(batch_images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run the model while handling micro-batches smaller than the DataParallel replica count.

        When batch size < #GPUs, DataParallel would try to scatter an empty tensor to some replica,
        causing it to call forward() without positional args. We fall back to single-GPU execution
        on the first device in that case.
        """
        if isinstance(model, torch.nn.DataParallel):
            num_replicas = len(model.device_ids)
            if batch_images.size(0) < num_replicas:
                primary_device = model.device_ids[0]
                batch_images = batch_images.to(primary_device, non_blocking=True)
                with torch.cuda.device(primary_device):
                    return model.module(batch_images, task="depth")
        return model(batch_images, task="depth")

    with torch.no_grad():
        progress = tqdm(total=len(loader), desc="Inference", unit="batch")
        for batch in loader:
            images = batch["image"].cuda(non_blocking=True)
            gt_depth = batch.get("depth")
            if gt_depth is not None:
                gt_depth = gt_depth.cuda(non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
                outputs = _call_model(images)
                pred_depth = outputs["depth"]
            camera_pred_norm = outputs.get("camera_intrinsics_norm")
            if camera_pred_norm is not None:
                camera_pred_norm = camera_pred_norm.detach().cpu()
            if pred_depth.dim() == 3:
                pred_depth = pred_depth.unsqueeze(1)
            meta_list: List[Dict[str, Any]] = batch["meta"]

            camera_sizes = batch.get("camera_size")
            camera_size_mask = batch.get("camera_size_mask")
            camera_sizes_np = camera_sizes.cpu().numpy() if camera_sizes is not None else None
            camera_size_mask_np = (
                camera_size_mask.cpu().numpy().astype(bool) if camera_size_mask is not None else None
            )
            gt_camera_raw = batch.get("camera_intrinsics")
            if gt_camera_raw is not None:
                gt_camera_raw = gt_camera_raw.detach().cpu()
            gt_camera_norm = batch.get("camera_intrinsics_norm")
            if gt_camera_norm is not None:
                gt_camera_norm = gt_camera_norm.detach().cpu()
            gt_cam_mask = batch.get("camera_intrinsics_mask")
            gt_cam_mask_np = gt_cam_mask.cpu().numpy().astype(bool) if gt_cam_mask is not None else None
            gt_cam_norm_mask = batch.get("camera_intrinsics_norm_mask")
            gt_cam_norm_mask_np = (
                gt_cam_norm_mask.cpu().numpy().astype(bool) if gt_cam_norm_mask is not None else None
            )

            for idx in range(pred_depth.shape[0]):
                meta = meta_list[idx]
                h = meta["height"]
                w = meta["width"]
                pred_slice = pred_depth[idx, 0, :h, :w]
                pred_slice = torch.nan_to_num(pred_slice, nan=0.0, posinf=0.0, neginf=0.0)
                if gt_depth is not None:
                    gt_slice = gt_depth[idx, 0, :h, :w]
                    invalid = (gt_slice <= 0) | (~torch.isfinite(gt_slice))
                    pred_slice = pred_slice.clone()
                    pred_slice[invalid] = 0.0
                above_max = pred_slice > args.max_depth
                if above_max.any():
                    pred_slice = pred_slice.clone()
                    pred_slice[above_max] = 0.0

                pred_fx = pred_fy = pred_cx = pred_cy = np.nan
                if camera_pred_norm is not None:
                    pred_vals = _resolve_camera_prediction(
                        idx,
                        camera_pred_norm,
                        camera_sizes_np,
                        camera_size_mask_np,
                        meta,
                    )
                    if pred_vals is not None:
                        pred_fx, pred_fy, pred_cx, pred_cy = pred_vals

                gt_vals = _resolve_camera_ground_truth(
                    idx,
                    gt_camera_raw,
                    gt_camera_norm,
                    camera_sizes_np,
                    camera_size_mask_np,
                    meta,
                    gt_cam_mask_np,
                    gt_cam_norm_mask_np,
                )
                dataset_name = meta.get("dataset_name") or meta.get("source_type") or "unknown"
                if (
                    gt_vals is not None
                    and not np.isnan(pred_fx)
                    and not np.isnan(pred_fy)
                ):
                    _update_camera_stats(camera_stats, str(dataset_name), pred_fx, pred_fy, gt_vals[0], gt_vals[1])

                rel_path = _derive_relative_path(meta, dataset_tokens)
                output_path = output_root / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if args.skip_existing and output_path.exists():
                    continue
                depth_np = _tensor_to_numpy(pred_slice)
                np.savez_compressed(
                    output_path,
                    depth=depth_np,
                    cam_fx=np.float32(pred_fx),
                    cam_fy=np.float32(pred_fy),
                    cam_cx=np.float32(pred_cx),
                    cam_cy=np.float32(pred_cy),
                )
                saved += 1

            progress.update(1)
        progress.close()

    logger.info("Finished inference. Saved %d files (root=%s).", saved, output_root)
    if camera_stats:
        _write_camera_report(camera_stats, camera_report_path)
        logger.info("Camera report saved to %s", camera_report_path)
    else:
        logger.warning("No valid camera statistics collected; camera report skipped.")


if __name__ == "__main__":
    run_inference(_parse_args())
