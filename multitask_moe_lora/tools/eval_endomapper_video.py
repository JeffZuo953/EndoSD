#!/usr/bin/env python3
"""
Run depth-only evaluation on EndoMapper using the training pipeline helpers,
then export RGB | GT | Pred | Error mosaics (invalid/ignored pixels in yellow)
and assemble them into a video for quick visual inspection.

All key knobs (checkpoint, dataset selections, sampling, output paths, etc.)
are declared at the top so the script can be edited without juggling shell
arguments.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from eval_depth_with_training_pipeline import (  # noqa: E402
    _build_eval_loaders,
    _build_eval_parser,
    _prepare_config,
    _setup_logger,
)
from multitask_moe_lora.util.model_setup import (  # noqa: E402
    create_and_setup_model,
    load_weights_from_checkpoint,
)
from multitask_moe_lora.util.validation import validate_and_visualize  # noqa: E402

# -----------------------------------------------------------------------------
# User-configurable knobs
# -----------------------------------------------------------------------------
CHECKPOINT = Path(
    "/data/ziyi/multitask/save/FM/fd_vits_fd_depth_fm_v1_camera_simple_train300_20250112_123456/checkpoints/best_depth.pth"
)
SAVE_ROOT = Path("/data/ziyi/multitask/save/eval_simple_endomapper")
RUN_NAME = "endomapper_simple_vits"
DATASET_CONFIG = "fd_depth_fm_v1"
DATASET_MODALITY = "fd"
TRAIN_INCLUDE = "EndoMapper"
VAL_INCLUDE = "EndoMapper"
VAL_SAMPLE_STEP = 1  # 1 = keep all samples, -1 = evenly-spaced sampler
VAL_MIN_SAMPLES = 0
IMG_SIZE = 518
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
ENCODER = "vits"
FEATURES = 64
CAMERA_HEAD_MODE = "simple"
CAMERA_LOSS_TYPE = "l1"
MODE = "original"
EVAL_EPOCH_TAG = 19  # controls visualization export cadence in validate_and_visualize
NUM_WORKERS = 8
VIDEO_FPS = 10
VIDEO_NAME = "endomapper_eval.mp4"

# -----------------------------------------------------------------------------
# Helpers for frame export
# -----------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INVALID_COLOR = np.array([0, 255, 255], dtype=np.uint8)  # yellow in BGR


def _build_arg_list(save_path: Path) -> List[str]:
    """Construct the pseudo-CLI argument list for the training parser."""
    return [
        "--encoder",
        ENCODER,
        "--features",
        str(FEATURES),
        "--camera-head-mode",
        CAMERA_HEAD_MODE,
        "--camera-loss-type",
        CAMERA_LOSS_TYPE,
        "--dataset-config-name",
        DATASET_CONFIG,
        "--dataset-modality",
        DATASET_MODALITY,
        "--path-transform-name",
        "none",
        "--train-dataset-include",
        TRAIN_INCLUDE,
        "--val-dataset-include",
        VAL_INCLUDE,
        "--val-sample-step",
        str(VAL_SAMPLE_STEP),
        "--val-min-samples-per-dataset",
        str(VAL_MIN_SAMPLES),
        "--img-size",
        str(IMG_SIZE),
        "--bs",
        str(BATCH_SIZE),
        "--val-bs",
        str(VAL_BATCH_SIZE),
        "--mode",
        MODE,
        "--disable-seg-head",
        "--depth-only",
        "--save-path",
        str(save_path),
        "--checkpoint",
        str(CHECKPOINT),
        "--eval-epoch-tag",
        str(EVAL_EPOCH_TAG),
        "--num-workers",
        str(NUM_WORKERS),
    ]


def _denorm_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized BCHW tensor (single sample) to uint8 BGR."""
    arr = tensor.detach().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Image tensor expects shape [C,H,W], got {arr.shape}")
    arr = np.transpose(arr, (1, 2, 0))
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    return arr_u8[:, :, ::-1]  # RGB -> BGR


def _extract_valid_mask(batch: dict, idx: int) -> torch.Tensor:
    """Pick the best-available valid mask and squeeze to [H,W]."""
    mask = None
    if "depth_valid_mask" in batch:
        mask = batch["depth_valid_mask"][idx]
    elif "valid_mask" in batch:
        mask = batch["valid_mask"][idx]
    if mask is None:
        mask = torch.ones_like(batch["depth"][idx], dtype=torch.bool)
    if mask.ndim == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask.bool().cpu()


def _colorize_depth(depth: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    """Map depth tensor to a magma colormap, invalid pixels -> yellow."""
    depth_np = depth.detach().cpu().numpy()
    mask_np = valid_mask.numpy().astype(bool)
    if depth_np.ndim == 3:
        depth_np = depth_np[0]
    valid_values = depth_np[mask_np] if mask_np.any() else depth_np
    min_val = float(valid_values.min()) if valid_values.size else 0.0
    max_val = float(valid_values.max()) if valid_values.size else 1.0
    scale = max(max_val - min_val, 1e-6)
    depth_norm = np.clip((depth_np - min_val) / scale, 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).round().astype(np.uint8)
    color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    color[~mask_np] = INVALID_COLOR
    return color


def _colorize_error(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    """Compute |pred-gt| heatmap; invalid pixels -> yellow."""
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    if pred_np.ndim == 3:
        pred_np = pred_np[0]
    if gt_np.ndim == 3:
        gt_np = gt_np[0]
    mask_np = valid_mask.numpy().astype(bool)
    err = np.abs(pred_np - gt_np)
    valid_vals = err[mask_np]
    if valid_vals.size == 0:
        err_norm = np.zeros_like(err)
    else:
        min_val = float(valid_vals.min())
        max_val = float(valid_vals.max())
        scale = max(max_val - min_val, 1e-6)
        err_norm = np.clip((err - min_val) / scale, 0.0, 1.0)
    err_uint8 = (err_norm * 255.0).round().astype(np.uint8)
    color = cv2.applyColorMap(err_uint8, cv2.COLORMAP_TURBO)
    color[~mask_np] = INVALID_COLOR
    return color


def _clean_frame_dir(frame_dir: Path) -> None:
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)


def _export_frames_and_video(
    model: torch.nn.Module,
    depth_loader,
    output_dir: Path,
    video_fps: int,
    logger,
) -> None:
    """Generate side-by-side mosaics and convert them into a video."""
    frame_dir = output_dir / "frames"
    _clean_frame_dir(frame_dir)
    video_path = output_dir / VIDEO_NAME

    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for idx, batch in enumerate(depth_loader):
            images = batch["image"].cuda(non_blocking=True)
            preds = model(images, task="depth")["depth"].cpu()
            rgb = _denorm_to_bgr(batch["image"][0])
            gt = batch["depth"][0]
            mask = _extract_valid_mask(batch, 0)

            rgb_with_mask = rgb.copy()
            mask_np = mask.numpy().astype(bool)
            rgb_with_mask[~mask_np] = INVALID_COLOR

            pred_map = preds[0]
            pred_color = _colorize_depth(pred_map, mask)
            gt_color = _colorize_depth(gt, mask)
            err_color = _colorize_error(pred_map, gt, mask)

            tile = np.concatenate([rgb_with_mask, gt_color, pred_color, err_color], axis=1)
            frame_path = frame_dir / f"frame_{idx:05d}.png"
            cv2.imwrite(str(frame_path), tile)

    _frames_to_video(frame_dir, video_path, video_fps, logger)


def _frames_to_video(frame_dir: Path, video_path: Path, fps: int, logger) -> None:
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        logger.warning("No frames found in %s; skipping video export.", frame_dir)
        return
    first = cv2.imread(str(frames[0]))
    if first is None:
        logger.error("Failed to read the first frame %s", frames[0])
        return
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        img = cv2.imread(str(frame))
        if img is None:
            logger.warning("Skipping unreadable frame %s", frame)
            continue
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        writer.write(img)
    writer.release()
    logger.info("Video written to %s", video_path)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    save_path = SAVE_ROOT / RUN_NAME
    save_path.mkdir(parents=True, exist_ok=True)

    parser = _build_eval_parser()
    args = parser.parse_args(_build_arg_list(save_path))
    config = _prepare_config(args)

    logger = _setup_logger(save_path)
    logger.info("=== Custom EndoMapper eval script ===")
    logger.info("Checkpoint: %s", CHECKPOINT)
    logger.info("Save path:  %s", save_path)
    logger.info("Datasets:   train=%s, val=%s", TRAIN_INCLUDE, VAL_INCLUDE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this eval script.")
    torch.cuda.set_device(0)

    depth_loader, _ = _build_eval_loaders(config, args.num_workers, logger, skip_seg=True)

    model = create_and_setup_model(config, logger)
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

    logger.info("=== Depth metrics ===")
    _ = validate_and_visualize(model, depth_loader, "depth", args.eval_epoch_tag, config, writer=None, logger=logger)

    logger.info("Exporting per-frame mosaics + video...")
    _export_frames_and_video(model, depth_loader, save_path, VIDEO_FPS, logger)
    logger.info("Eval + visualization finished.")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
