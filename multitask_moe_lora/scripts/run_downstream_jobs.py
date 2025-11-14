#!/usr/bin/env python3
"""
Custom downstream evaluation utility for:
  1. dVRK_demo left camera images (depth eval + segmentation + size estimation)
  2. Polyp_Size_Videos (sampled frames, depth + segmentation + size estimation)

Outputs (per dataset) are stored under the requested save root while preserving
the original directory layout for depth/segmentation artifacts.
"""

from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import sys
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = REPO_ROOT.parent
for entry in (str(PARENT_ROOT), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from multitask_moe_lora.util.config import TrainingConfig
from multitask_moe_lora.util.model_setup import create_and_setup_model, load_weights_from_checkpoint
from multitask_moe_lora.dataset.transform import Resize, NormalizeImage, PrepareForNet
from util.metric import eval_depth


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str
    notes: Optional[str] = None

    def as_dict(self) -> Dict[str, float]:
        data = asdict(self)
        return data


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_relpath(path: Path) -> Path:
    # Drop drive letters or network prefixes if any
    return Path(*[p for p in path.parts if p not in (Path(path.anchor).name if path.anchor else "", "")])


def _depth_to_colormap(depth: np.ndarray, max_depth: float) -> np.ndarray:
    normalized = np.clip(depth / max(max_depth, 1e-6), 0.0, 1.0)
    normalized = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return colored


SEGMENT_COLORS = np.array([
    [0, 0, 0],
    [0, 165, 255],     # orange for class 1
    [0, 255, 0],       # green
    [255, 0, 0],       # blue
    [255, 255, 0],     # yellow (extra safety)
], dtype=np.uint8)


def _colorize_seg(seg_mask: np.ndarray) -> np.ndarray:
    idx = np.clip(seg_mask.astype(np.int32), 0, len(SEGMENT_COLORS) - 1)
    return SEGMENT_COLORS[idx]


def _connected_component_mask(mask: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    if num_labels <= 1:
        return None, 0
    best_area = 0
    best_id = 0
    for label_id in range(1, num_labels):
        area = int((labels == label_id).sum())
        if area > best_area:
            best_area = area
            best_id = label_id
    if best_area == 0:
        return None, 0
    component_mask = (labels == best_id).astype(np.uint8)
    return component_mask, best_area


def _estimate_size(points_xyz: np.ndarray) -> float:
    if points_xyz.shape[0] < 10:
        return 0.0
    pts = points_xyz - points_xyz.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    proj = pts @ eigvecs
    lengths = proj.max(axis=0) - proj.min(axis=0)
    return float(np.max(lengths))


def _mask_points_3d(depth: np.ndarray, mask: np.ndarray, intrinsics: Intrinsics) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    z = depth[ys, xs]
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    return pts


def _prepare_transform(img_size: int, ensure_multiple_of: int = 14) -> Compose:
    return Compose([
        Resize(
            width=img_size,
            height=img_size,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=ensure_multiple_of,
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


class MultitaskInferencer:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        encoder: str = "vitb",
        features: int = 64,
        num_classes: int = 4,
        max_depth: float = 0.3,
        seg_input_type: str = "from_depth",
        seg_head_type: str = "linear",
        mode: str = "endounid",
        img_size: int = 518,
        use_semantic_tokens: bool = True,
        semantic_token_count: int = 10,
    ):
        self.logger = logging.getLogger("downstream_infer")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        config = TrainingConfig()
        config.encoder = encoder
        config.features = features
        config.num_classes = num_classes
        config.max_depth = max_depth
        config.seg_input_type = seg_input_type
        config.seg_head_type = seg_head_type
        config.mode = mode
        config.num_experts = 8
        config.top_k = 2
        config.lora_r = 4
        config.lora_alpha = 8
        config.use_semantic_tokens = use_semantic_tokens
        config.semantic_token_count = semantic_token_count
        config.resume_from = str(checkpoint)
        config.resume_full_state = False
        config.bs = 1
        config.seg_bs = 1
        config.val_bs = 1
        config.img_size = img_size
        config.save_path = str(checkpoint.parent)
        config.mixed_precision = False
        config.frozen_backbone = False
        config.dinov3_repo_path = str((REPO_ROOT.parent / "dinov3").resolve())

        if device == "cpu" or not torch.cuda.is_available():
            raise RuntimeError("CUDA device is required for this inference helper.")

        self.model = create_and_setup_model(config, self.logger)
        load_weights_from_checkpoint(
            model=self.model,
            optimizer_depth=None,
            optimizer_seg=None,
            optimizer_camera=None,
            scheduler_depth=None,
            scheduler_seg=None,
            scheduler_camera=None,
            config=config,
            logger=self.logger,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        ensure_multiple_of = 16 if "dinov3" in encoder else 14
        self.transform = _prepare_transform(img_size, ensure_multiple_of)
        self.max_depth = max_depth

    def infer(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
        data = self.transform({"image": image_rgb})
        tensor = torch.from_numpy(data["image"]).unsqueeze(0).to(self.device, non_blocking=True)

        with torch.no_grad():
            depth_out = self.model(tensor, task="depth")["depth"]
            if depth_out.ndim == 3:
                depth_out = depth_out.unsqueeze(1)
            depth_resized = F.interpolate(depth_out, size=(h, w), mode="bilinear", align_corners=False)
            depth_map = depth_resized.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

            seg_out = self.model(tensor, task="seg")["seg"]
            seg_resized = F.interpolate(seg_out, size=(h, w), mode="bilinear", align_corners=False)
            seg_logits = seg_resized.squeeze(0)
            seg_probs = seg_logits.softmax(dim=0).cpu().numpy().astype(np.float32)
            seg_mask = np.argmax(seg_probs, axis=0).astype(np.uint8)

        return depth_map, seg_mask, seg_probs


def _collect_left_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for left_dir in root.rglob("left_imgs"):
        for img_path in sorted(left_dir.glob("*.png")):
            images.append(img_path)
    return images


def _relative_to(path: Path, base: Path) -> Path:
    return Path(os.path.relpath(path, base))


def _save_numpy(array: np.ndarray, path: Path) -> None:
    _ensure_dir(path)
    np.save(path, array)


def _save_image(img: np.ndarray, path: Path) -> None:
    _ensure_dir(path)
    cv2.imwrite(str(path), img)


def _load_stereo_intrinsics(config_path: Path, target_hw: Tuple[int, int]) -> Intrinsics:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    res_x = float(cfg["StereoLeft"]["res_x"])
    res_y = float(cfg["StereoLeft"]["res_y"])
    scale_x = target_hw[1] / res_x
    scale_y = target_hw[0] / res_y
    fx = float(cfg["StereoLeft"]["fc_x"]) * scale_x
    fy = float(cfg["StereoLeft"]["fc_y"]) * scale_y
    cx = float(cfg["StereoLeft"]["cc_x"]) * scale_x
    cy = float(cfg["StereoLeft"]["cc_y"]) * scale_y
    return Intrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=target_hw[1],
        height=target_hw[0],
        source=str(config_path),
        notes=f"Scaled from {int(res_x)}x{int(res_y)} to {target_hw[1]}x{target_hw[0]}",
    )


def _default_intrinsics(width: int, height: int, source: str) -> Intrinsics:
    fx = fy = 0.9 * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return Intrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        source=source,
        notes="Assumed pinhole intrinsics (no calibration provided)",
    )


def _write_json(data: Dict, path: Path) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _append_jsonl(records: List[Dict], path: Path) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _compute_depth_metrics(pred: np.ndarray, gt: np.ndarray) -> Optional[Dict[str, float]]:
    valid_mask = gt > 0
    if not np.any(valid_mask):
        return None
    pred_flat = pred[valid_mask]
    gt_flat = gt[valid_mask]
    metrics = eval_depth(pred_flat, gt_flat)
    diff = pred_flat - gt_flat
    metrics["mae"] = float(np.mean(np.abs(diff)))
    metrics["mse"] = float(np.mean(diff ** 2))
    metrics["rmse_mm"] = float(math.sqrt(metrics["mse"]) * 1000.0)
    metrics["mae_mm"] = metrics["mae"] * 1000.0
    metrics["count"] = int(pred_flat.size)
    return metrics


def process_dvrk(
    inferencer: MultitaskInferencer,
    root: Path,
    save_root: Path,
    intrinsics: Intrinsics,
    target_label: int = 1,
) -> Dict[str, float]:
    images = _collect_left_images(root)
    size_records: List[Dict] = []
    metric_records: List[Dict] = []

    depth_sum: Dict[str, float] = {}
    sample_count = 0

    for img_path in images:
        rel_path = _relative_to(img_path, root)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        depth_map, seg_mask, seg_probs = inferencer.infer(img_bgr)

        depth_save = save_root / "depth_npy" / rel_path.with_suffix(".npy")
        depth_vis_save = save_root / "depth_vis" / rel_path.with_suffix(".png")
        seg_mask_save = save_root / "seg_mask" / rel_path.with_suffix(".png")
        seg_vis_save = save_root / "seg_vis" / rel_path.with_suffix(".png")

        _save_numpy(depth_map.astype(np.float32), depth_save)
        _save_image(_depth_to_colormap(depth_map, inferencer.max_depth), depth_vis_save)
        _save_image(seg_mask, seg_mask_save)

        color_seg = _colorize_seg(seg_mask)
        overlay = cv2.addWeighted(img_bgr, 0.4, color_seg, 0.6, 0)

        mask_class = (seg_mask == target_label).astype(np.uint8)
        component_mask, area_px = _connected_component_mask(mask_class)
        component_size_mm = 0.0
        avg_depth = float(depth_map[mask_class > 0].mean()) if area_px > 0 else 0.0
        if component_mask is not None:
            pts = _mask_points_3d(depth_map, component_mask, intrinsics)
            if pts is not None:
                component_size_mm = _estimate_size(pts) * 1000.0
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
                cv2.putText(
                    overlay,
                    f"size~{component_size_mm:.1f}mm",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        _save_image(overlay, seg_vis_save)

        size_records.append({
            "dataset": "dVRK_demo",
            "relative_path": str(rel_path),
            "mask_pixels": int(area_px),
            "avg_depth_m": float(avg_depth),
            "size_mm": float(component_size_mm),
        })

        depth_filename = img_path.parent.parent / "depth_imgs" / img_path.name.replace("frame_", "depth_").replace(".png", ".npy")
        if depth_filename.exists():
            gt_depth = np.load(depth_filename)
            metrics = _compute_depth_metrics(depth_map, gt_depth)
            if metrics:
                metrics["relative_path"] = str(rel_path)
                metric_records.append(metrics)
                sample_count += 1
                for k, v in metrics.items():
                    if k in {"relative_path", "count"}:
                        continue
                    depth_sum[k] = depth_sum.get(k, 0.0) + float(v)

    if metric_records:
        averaged = {k: v / sample_count for k, v in depth_sum.items()}
        averaged["samples"] = sample_count
    else:
        averaged = {}

    _append_jsonl(size_records, save_root / "size_predictions.jsonl")
    _append_jsonl(metric_records, save_root / "depth_metrics.jsonl")
    _write_json(averaged, save_root / "depth_metrics_summary.json")
    _write_json(intrinsics.as_dict(), save_root / "intrinsics.json")
    return averaged


def _sample_frame_indices(frame_count: int, target: int) -> List[int]:
    if frame_count == 0:
        return []
    if frame_count <= target:
        return list(range(frame_count))
    indices = np.linspace(0, frame_count - 1, target)
    return sorted({int(round(i)) for i in indices})


def process_polyp(
    inferencer: MultitaskInferencer,
    video_root: Path,
    save_root: Path,
    frames_per_video: int,
    target_label: int = 3,
    keep_labels: Optional[List[int]] = None,
) -> None:
    size_records: List[Dict] = []
    intrinsics_cache: Dict[str, Intrinsics] = {}

    for video_path in sorted(video_root.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_name = video_path.stem
        intr = intrinsics_cache.get(video_name)
        if intr is None:
            intr = _default_intrinsics(width, height, f"{video_name}_assumed")
            intrinsics_cache[video_name] = intr

        indices = _sample_frame_indices(frame_count, frames_per_video)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            depth_map, seg_mask, seg_probs = inferencer.infer(frame)
            rel_path = Path(video_name) / f"frame_{idx:05d}.png"

            depth_save = save_root / "depth_npy" / rel_path.with_suffix(".npy")
            depth_vis_save = save_root / "depth_vis" / rel_path.with_suffix(".png")
            seg_mask_save = save_root / "seg_mask" / rel_path.with_suffix(".png")
            seg_vis_save = save_root / "seg_vis" / rel_path.with_suffix(".png")

            _save_numpy(depth_map.astype(np.float32), depth_save)
            _save_image(_depth_to_colormap(depth_map, inferencer.max_depth), depth_vis_save)
            vis_mask = seg_mask.copy()
            if keep_labels:
                allowed = np.isin(vis_mask, keep_labels)
                vis_mask = np.where(allowed, vis_mask, 0).astype(np.uint8)
            _save_image(vis_mask, seg_mask_save)

            color_seg = _colorize_seg(vis_mask)
            overlay = cv2.addWeighted(frame, 0.4, color_seg, 0.6, 0)

            mask_class = (seg_mask == target_label).astype(np.uint8)
            component_mask, area_px = _connected_component_mask(mask_class)
            component_size_mm = 0.0
            avg_depth = float(depth_map[mask_class > 0].mean()) if area_px > 0 else 0.0
            if component_mask is not None:
                pts = _mask_points_3d(depth_map, component_mask, intr)
                if pts is not None:
                    component_size_mm = _estimate_size(pts) * 1000.0
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 1)
                    cv2.putText(
                        overlay,
                        f"size~{component_size_mm:.1f}mm",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
            _save_image(overlay, seg_vis_save)

            size_records.append({
                "dataset": "Polyp_Size_Videos",
                "video": video_name,
                "frame_index": idx,
                "relative_path": str(rel_path),
                "mask_pixels": int(area_px),
                "avg_depth_m": float(avg_depth),
                "size_mm": float(component_size_mm),
            })
        cap.release()

    _append_jsonl(size_records, save_root / "size_predictions.jsonl")
    intr_dump = {name: intr.as_dict() for name, intr in intrinsics_cache.items()}
    _write_json(intr_dump, save_root / "intrinsics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream depth/seg inference + size estimations.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--save-root", default="~/downstreamjobs/save", help="Output root.")
    parser.add_argument("--dvrk-root", default="~/downstreamjobs/dVRK_demo", help="dVRK data root.")
    parser.add_argument("--polyp-videos-root", default="~/downstreamjobs/Polyp_Size_Dataset/Polyp_Size_Videos", help="Polyp video root.")
    parser.add_argument("--frames-per-video", type=int, default=50, help="Number of frames sampled per video.")
    parser.add_argument("--dvrk-mask-label", type=int, default=1, help="Segmentation label used for dVRK size estimation.")
    parser.add_argument("--polyp-mask-label", type=int, default=3, help="Segmentation label used for Polyp size estimation.")
    parser.add_argument("--polyp-keep-labels", default="", help="Comma-separated labels to keep for Polyp visualization (others set to 0).")
    parser.add_argument("--dvrk-output-name", default="dVRK_demo", help="Subdirectory under save root for dVRK outputs.")
    parser.add_argument("--polyp-output-name", default="Polyp_Size_Videos", help="Subdirectory under save root for Polyp outputs.")
    parser.add_argument("--skip-dvrk", action="store_true", help="Skip dVRK processing.")
    parser.add_argument("--skip-polyp", action="store_true", help="Skip Polyp processing.")
    parser.add_argument("--img-size", type=int, default=518, help="Model input resolution.")
    parser.add_argument("--max-depth", type=float, default=0.3, help="Depth normalization upper bound.")
    parser.add_argument("--device", default="cuda", help="Inference device (cuda or cpu).")
    parser.add_argument("--intr-config", default="StereoCalibration.ini", help="Stereo calibration file for dVRK left cam.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(os.path.expanduser(args.checkpoint)).resolve()
    save_root = Path(os.path.expanduser(args.save_root)).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    inferencer = MultitaskInferencer(
        checkpoint=checkpoint,
        device=args.device if torch.cuda.is_available() else "cpu",
        num_classes=4,
        max_depth=args.max_depth,
        img_size=args.img_size,
    )

    dvrk_summary = {}
    polyp_summary = {}

    if not args.skip_dvrk:
        dvrk_root = Path(os.path.expanduser(args.dvrk_root)).resolve()
        intr_config = Path(args.intr_config).resolve()
        sample_img = next(iter(_collect_left_images(dvrk_root)), None)
        if sample_img is None:
            raise RuntimeError("No dVRK left_imgs found.")
        sample = cv2.imread(str(sample_img), cv2.IMREAD_COLOR)
        if sample is None:
            raise RuntimeError(f"Failed to load sample image: {sample_img}")
        dvrk_intr = _load_stereo_intrinsics(intr_config, (sample.shape[0], sample.shape[1]))
        dvrk_save = save_root / args.dvrk_output_name
        dvrk_summary = process_dvrk(inferencer, dvrk_root, dvrk_save, dvrk_intr, target_label=args.dvrk_mask_label)
    else:
        dvrk_save = save_root / args.dvrk_output_name

    if not args.skip_polyp:
        polyp_root = Path(os.path.expanduser(args.polyp_videos_root)).resolve()
        polyp_save = save_root / args.polyp_output_name
        keep_labels = None
        if args.polyp_keep_labels.strip():
            keep_labels = [int(x) for x in args.polyp_keep_labels.split(",") if x.strip()]
        process_polyp(
            inferencer,
            polyp_root,
            polyp_save,
            args.frames_per_video,
            target_label=args.polyp_mask_label,
            keep_labels=keep_labels,
        )
    else:
        polyp_save = save_root / args.polyp_output_name

    summary_path = save_root / "summary.json"
    summary_payload = {
        "checkpoint": str(checkpoint),
        "dvrk_metrics": dvrk_summary,
        "outputs": {
            "dvrk_root": str(dvrk_save),
            "polyp_root": str(polyp_save),
        },
    }
    _write_json(summary_payload, summary_path)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
