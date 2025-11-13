#!/usr/bin/env python3
"""
验证评估模块
包含验证逻辑、指标计算和结果可视化功能
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
import logging
from typing import Dict, Any, Optional, List
import math
import statistics
from collections import Counter
from pathlib import Path
import torch.distributed as dist

from .config import TrainingConfig
from .loss import SiLogLoss
from .metric import SegMetric, eval_depth
from .data_utils import summarize_loader_composition
from .palette import get_palette

LS_SEG_CLASS_WHITELIST = [0, 1, 2, 3, 5, 6, 7, 8]
NO_SEG_CLASS_WHITELIST = [0, 1, 2, 3, 4]
COMBINED_SEG_CLASS_WHITELIST = sorted(set(LS_SEG_CLASS_WHITELIST + NO_SEG_CLASS_WHITELIST))
# Try to import simple save helpers; fallback to legacy names if needed
try:
    from .visualize import save_depth_prediction, save_seg_prediction
except ImportError:
    try:
        # Legacy API fallback
        from .visualize import save_depth_output, save_seg_output  # type: ignore
    except Exception:
        # As a last resort, define no-op functions to avoid import-time failure
        def save_depth_prediction(*args, **kwargs):  # type: ignore
            return None

        def save_seg_prediction(*args, **kwargs):  # type: ignore
            return None
    else:
        def save_depth_prediction(pred, filename, outdir, colormap: str = 'gray'):  # type: ignore
            return save_depth_output(
                pred,
                filename,
                outdir,
                norm_type='min-max',
                colormap=colormap,
                save_img=True,
                save_pt=False,
                save_npz=False,
            )

        def save_seg_prediction(pred, filename, outdir):  # type: ignore
            return save_seg_output(
                pred,
                filename,
                outdir,
                save_img=True,
                save_pt=False,
                save_npz=False,
            )
except Exception:
    def save_depth_prediction(*args, **kwargs):  # type: ignore
        return None

    def save_seg_prediction(*args, **kwargs):  # type: ignore
        return None


def _persist_segmentation_metrics(save_path: Optional[str],
                                  epoch: int,
                                  metrics: Dict[str, Dict[str, float]],
                                  logger: logging.Logger) -> None:
    if not save_path or not metrics:
        return
    record = {
        "epoch": int(epoch),
        "metrics": metrics,
    }
    out_path = Path(save_path) / "segmentation_metrics.jsonl"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved per-dataset segmentation metrics to %s", out_path)
    except OSError as exc:
        logger.warning("Failed to write segmentation metrics to %s: %s", out_path, exc)


class DepthValidator:
    """深度估计验证器"""

    _MASKED_DATASETS = {"scard", "scared", "dvpn", "stereomis", "endovis2017", "endovis2018"}

    def __init__(self, config: TrainingConfig):
        self.config = config
        min_depth_cfg = float(getattr(self.config, 'min_depth', 1e-6) or 1e-6)
        self._min_safe_depth = max(min_depth_cfg, 1e-5)
        self.criterion = SiLogLoss(eps=self._min_safe_depth)
        self.pred_folder = None

    def setup_output_folder(self, save_path: str) -> None:
        """禁用逐 batch 深度图保存（保留占位）"""
        self.pred_folder = None

    def validate_batch(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> tuple[float, Optional[Dict[str, float]]]:
        """
        验证单个批次
        
        Args:
            model: 模型
            batch: 批次数据
            
        Returns:
            (loss, metrics) - metrics在有有效像素时返回，否则为None
        """
        input_img = batch["image"].cuda()
        target_gt = batch["depth"].cuda()

        outputs = model(input_img, task='depth')
        pred = outputs['depth']

        # 先根据batch中提供的有效性掩码（优先 depth_valid_mask）构建基础掩码
        base_mask: Optional[torch.Tensor] = None
        if "depth_valid_mask" in batch:
            base_mask = batch["depth_valid_mask"].cuda()
        elif "valid_mask" in batch:
            base_mask = batch["valid_mask"].cuda()

        if base_mask is not None:
            if base_mask.dim() == 2:
                base_mask = base_mask.unsqueeze(0).unsqueeze(0)
            elif base_mask.dim() == 3:
                base_mask = base_mask.unsqueeze(1)
            base_mask = base_mask.to(torch.bool)
        # 使用与训练阶段一致的深度范围约束
        range_mask = (target_gt > 0) & (target_gt >= self._min_safe_depth) & (target_gt <= self.config.max_depth)
        valid_mask_4d = range_mask if base_mask is None else (range_mask & base_mask)
        dataset_mask = self._extract_dataset_mask(batch, valid_mask_4d.device, valid_mask_4d.shape[0])
        if dataset_mask is not None:
            valid_mask_4d = valid_mask_4d & dataset_mask
        loss = self.criterion(pred, target_gt, valid_mask_4d)

        # 确保维度一致：pred是3D [B,H,W]，target_gt是4D [B,1,H,W]
        target_gt_squeezed = target_gt.squeeze(1)  # [B,1,H,W] -> [B,H,W]
        valid_mask = valid_mask_4d.squeeze(1)  # [B,1,H,W] -> [B,H,W]

        metrics = None
        if torch.any(valid_mask):
            metrics = eval_depth(pred[valid_mask], target_gt_squeezed[valid_mask])

        return loss.item(), metrics

    def save_prediction(self, pred: torch.Tensor, epoch: int) -> None:
        """保存预测结果"""
        if self.pred_folder is not None:
            save_depth_prediction(pred, f'epoch_{epoch}.png', self.pred_folder)

    def _normalize_meta_list(self, value: Any, batch_size: int) -> list:
        if value is None:
            return [None] * batch_size
        if isinstance(value, (list, tuple)):
            entries = list(value)
            if len(entries) == batch_size:
                return entries
            if len(entries) == 1:
                return entries * batch_size
            if len(entries) < batch_size:
                entries.extend([entries[-1]] * (batch_size - len(entries)))
            return entries[:batch_size]
        return [value] * batch_size

    def _extract_dataset_mask(self, batch: Dict[str, Any], device: torch.device, batch_size: int) -> Optional[torch.Tensor]:
        dataset_entries = self._normalize_meta_list(batch.get("dataset_name"), batch_size)
        source_entries = self._normalize_meta_list(batch.get("source_type"), batch_size)

        apply_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for idx in range(batch_size):
            name = dataset_entries[idx]
            source = source_entries[idx]
            matched = False
            if name is not None:
                matched = str(name).lower() in self._MASKED_DATASETS
            if not matched and source is not None:
                matched = str(source).lower() in self._MASKED_DATASETS
            apply_flags[idx] = matched

        if not bool(apply_flags.any()):
            return None

        if "depth_valid_mask" in batch:
            mask_tensor = batch["depth_valid_mask"]
        elif "valid_mask" in batch:
            mask_tensor = batch["valid_mask"]
        else:
            return None

        if not torch.is_tensor(mask_tensor):
            mask_tensor = torch.as_tensor(mask_tensor)

        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)
        elif mask_tensor.dim() != 4:
            return None

        mask_tensor = mask_tensor.to(device=device, dtype=torch.bool)
        if mask_tensor.shape[0] != batch_size:
            return None

        merged_mask = torch.ones_like(mask_tensor, dtype=torch.bool, device=device)
        merged_mask[apply_flags] = mask_tensor[apply_flags]
        return merged_mask


class SegValidator:
    """语义分割验证器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metric = SegMetric(config.num_classes)
        self.criterion = CrossEntropyLoss(ignore_index=255)
        self.pred_folder = None

    def setup_output_folder(self, save_path: str) -> None:
        """禁用分割图像保存（仅保持接口一致）。"""
        self.pred_folder = None

    def reset_metrics(self) -> None:
        """重置指标"""
        self.metric.reset()

    def validate_batch(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> float:
        """
        验证单个批次
        
        Args:
            model: 模型
            batch: 批次数据
            
        Returns:
            loss
        """
        input_img = batch["image"].cuda()
        target_gt = batch["semseg_mask"].cuda().long()

        outputs = model(input_img, task='seg')
        pred = outputs['seg']

        # 确保分割标签在有效范围内
        ignore_idx = 255  # CrossEntropyLoss的默认ignore_index
        valid_class_mask = (target_gt >= 0) & (target_gt < self.config.num_classes)
        ignore_mask = (target_gt == ignore_idx)
        valid_mask = valid_class_mask | ignore_mask

        # 将无效标签设置为ignore_idx
        target_gt = torch.where(valid_mask, target_gt, torch.tensor(ignore_idx, device=target_gt.device))

        # 若没有任何有效像素，避免NaN，返回0损失
        if int(((target_gt != ignore_idx) & (target_gt >= 0) & (target_gt < self.config.num_classes)).sum().item()) == 0:
            loss = (pred.sum() * 0.0).to(pred.dtype)
        else:
            loss = self.criterion(pred, target_gt)

        # 更新指标
        self.metric.update(pred.argmax(dim=1), target_gt)

        return loss.item()

    def save_prediction(self, pred: torch.Tensor, epoch: int) -> None:
        """训练期间不再保存分割预测，保持空实现。"""
        return

    def get_metrics(self) -> Dict[str, float]:
        """获取指标"""
        return self.metric.get_scores()


def _ensure_list(batch: Dict[str, Any], key: str, default_value: str, count: int) -> List[str]:
    value = batch.get(key)
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return [default_value for _ in range(count)]
    return [value for _ in range(count)]


def _sanitize_name(name: Any) -> str:
    if name is None:
        return "unknown"
    name_str = str(name)
    return name_str.replace("/", "_").replace(" ", "_")


def _tensor_to_visual_image(image: torch.Tensor) -> np.ndarray:
    """Convert tensor image [C,H,W] or [H,W] to uint8 BGR image."""
    img = image.detach().cpu()
    if img.ndim == 3:
        if img.shape[0] >= 3:
            img = img[:3]
        img_np = img.permute(1, 2, 0).numpy()
    elif img.ndim == 2:
        img_np = img.unsqueeze(-1).numpy()
    else:
        img_np = img.squeeze().numpy()
        if img_np.ndim == 1:
            img_np = img_np[:, None]

    img_np = img_np.astype(np.float32)
    min_val = float(np.min(img_np))
    max_val = float(np.max(img_np))
    if max_val - min_val > 1e-6:
        img_np = (img_np - min_val) / (max_val - min_val)
    else:
        img_np = np.zeros_like(img_np, dtype=np.float32)

    if img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)

    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)


def _tensor_to_depth_colormap(depth: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> np.ndarray:
    """Convert depth tensor [1,H,W] to colored visualization."""
    depth_np = depth.detach().cpu().squeeze().numpy().astype(np.float32)
    if depth_np.ndim != 2:
        depth_np = np.squeeze(depth_np)
        if depth_np.ndim != 2:
            depth_np = depth_np.reshape(depth_np.shape[-2], depth_np.shape[-1])

    mask_np = None
    if valid_mask is not None:
        mask_np = valid_mask.detach().cpu().squeeze().numpy().astype(bool)
        if mask_np.shape != depth_np.shape:
            mask_np = None

    if mask_np is not None and mask_np.any():
        valid_values = depth_np[mask_np]
        min_val = float(valid_values.min())
        max_val = float(valid_values.max())
    else:
        min_val = float(depth_np.min()) if depth_np.size > 0 else 0.0
        max_val = float(depth_np.max()) if depth_np.size > 0 else 1.0

    scale = max(max_val - min_val, 1e-6)
    depth_norm = (depth_np - min_val) / scale
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    if mask_np is not None:
        depth_norm = depth_norm * mask_np.astype(np.float32)

    depth_uint8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)


def _tensor_to_segmentation_colormap(mask: torch.Tensor) -> np.ndarray:
    """Convert segmentation mask [H,W] to color visualization, ignore index=255."""
    mask_np = mask.detach().cpu().numpy().astype(np.int32)
    ignore_mask = mask_np == 255
    mask_np = np.where(ignore_mask, 0, mask_np)

    try:
        palette = get_palette()
    except Exception:
        palette = []
    if not palette:
        palette = [(i * 50) % 255 for i in range(3)]
        palette = [palette for _ in range(256)]
    palette_arr = np.array(palette, dtype=np.uint8)
    if palette_arr.ndim == 1:
        palette_arr = np.tile(palette_arr, (256, 1))

    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(mask_np)
    palette_len = len(palette_arr)
    for label in unique_labels:
        idx = label % palette_len
        color[mask_np == label] = palette_arr[idx]
    color[ignore_mask] = np.array([128, 128, 128], dtype=np.uint8)
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


def _export_depth_visuals(visuals: Dict[str, Dict[str, Any]], save_path: str, epoch: int, logger: logging.Logger) -> None:
    if not visuals:
        return
    epoch_dir = os.path.join(save_path, "visualizations", f"epoch_{epoch:04d}", "depth")
    for dataset_key, payload in visuals.items():
        samples = payload.get("samples", [])
        if not samples:
            continue
        dataset_dir = os.path.join(epoch_dir, dataset_key or "unknown")
        os.makedirs(dataset_dir, exist_ok=True)
        for idx, sample in enumerate(samples, start=1):
            base = f"sample{idx:02d}"
            image_path = os.path.join(dataset_dir, f"{base}_image.png")
            pred_path = os.path.join(dataset_dir, f"{base}_pred.png")
            gt_path = os.path.join(dataset_dir, f"{base}_gt.png")
            cv2.imwrite(image_path, _tensor_to_visual_image(sample["image"]))
            cv2.imwrite(pred_path, _tensor_to_depth_colormap(sample["pred"], sample.get("valid_mask")))
            cv2.imwrite(gt_path, _tensor_to_depth_colormap(sample["gt"], sample.get("valid_mask")))
    logger.info("Saved depth visualizations to %s", epoch_dir)


def _export_seg_visuals(visuals: Dict[str, Dict[str, Any]], save_path: str, epoch: int, logger: logging.Logger) -> None:
    if not visuals:
        return
    epoch_dir = os.path.join(save_path, "visualizations", f"epoch_{epoch:04d}", "seg")
    for dataset_key, payload in visuals.items():
        samples = payload.get("samples", [])
        if not samples:
            continue
        dataset_dir = os.path.join(epoch_dir, dataset_key or "unknown")
        os.makedirs(dataset_dir, exist_ok=True)
        for idx, sample in enumerate(samples, start=1):
            base = f"sample{idx:02d}"
            image_path = os.path.join(dataset_dir, f"{base}_image.png")
            pred_path = os.path.join(dataset_dir, f"{base}_pred.png")
            gt_path = os.path.join(dataset_dir, f"{base}_gt.png")
            cv2.imwrite(image_path, _tensor_to_visual_image(sample["image"]))
            cv2.imwrite(pred_path, _tensor_to_segmentation_colormap(sample["pred"]))
            cv2.imwrite(gt_path, _tensor_to_segmentation_colormap(sample["gt"]))
    logger.info("Saved segmentation visualizations to %s", epoch_dir)


def save_demo_predictions(model: torch.nn.Module,
                          depth_loader: Optional[DataLoader],
                          seg_loader: Optional[DataLoader],
                          config: TrainingConfig,
                          epoch: int,
                          logger: logging.Logger,
                          max_samples_per_source: Optional[int] = None) -> None:
    """保存 demo 预测结果於 save_path/demo/<dataset>/epoch_<N>/..."""

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    demo_root = config.demo_output_root or os.path.join(config.save_path, "demo")
    epoch_dir_name = f"epoch_{epoch + 1}"

    model.eval()
    with torch.no_grad():
        if depth_loader is not None:
            saved_counts: dict[str, int] = {}
            observed_sources: set[str] = set()
            for batch in depth_loader:
                images = batch["image"].cuda()
                outputs = model(images, task='depth')["depth"].detach().cpu()

                source_types = _ensure_list(batch, "source_type", "unknown", outputs.shape[0])
                image_paths = _ensure_list(batch, "image_path", "sample", outputs.shape[0])

                for idx in range(outputs.shape[0]):
                    dataset_name = _sanitize_name(source_types[idx])
                    base_name = os.path.splitext(os.path.basename(image_paths[idx]))[0]
                    observed_sources.add(dataset_name)
                    if max_samples_per_source is not None and saved_counts.get(dataset_name, 0) >= max_samples_per_source:
                        continue
                    target_dir = os.path.join(demo_root, dataset_name, epoch_dir_name, "depth")
                    os.makedirs(target_dir, exist_ok=True)

                    image_tensor = batch["image"][idx].detach().cpu()
                    image_rgb = _denormalize_image(image_tensor)
                    depth_tensor = outputs[idx]
                    valid_mask = None
                    if "valid_mask" in batch:
                        valid_mask = batch["valid_mask"][idx].detach().cpu()
                    if "depth" in batch:
                        depth_gt = batch["depth"][idx].detach().cpu()
                    else:
                        depth_gt = None

                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    depth_pred_color = cv2.cvtColor(
                        _depth_to_colormap(depth_tensor, valid_mask, config.max_depth),
                        cv2.COLOR_RGB2BGR,
                    )
                    tiles = [image_bgr, depth_pred_color]
                    if depth_gt is not None:
                        depth_gt_color = cv2.cvtColor(
                            _depth_to_colormap(depth_gt, valid_mask, config.max_depth),
                            cv2.COLOR_RGB2BGR,
                        )
                        tiles.append(depth_gt_color)
                    depth_vis = np.concatenate(tiles, axis=1)
                    vis_path = os.path.join(target_dir, f"{base_name}_depth_vis.png")
                    cv2.imwrite(vis_path, depth_vis)

                    if max_samples_per_source is not None:
                        saved_counts[dataset_name] = saved_counts.get(dataset_name, 0) + 1
                if max_samples_per_source is not None and observed_sources and all(
                    saved_counts.get(name, 0) >= max_samples_per_source for name in observed_sources
                ):
                    break

        if seg_loader is not None:
            saved_counts_seg: dict[str, int] = {}
            observed_sources_seg: set[str] = set()
            for batch in seg_loader:
                images = batch["image"].cuda()
                preds = model(images, task='seg')["seg"].detach()

                source_types = _ensure_list(batch, "source_type", "unknown", preds.shape[0])
                image_paths = _ensure_list(batch, "image_path", "sample", preds.shape[0])

                for idx in range(preds.shape[0]):
                    dataset_name = _sanitize_name(source_types[idx])
                    base_name = os.path.splitext(os.path.basename(image_paths[idx]))[0]
                    observed_sources_seg.add(dataset_name)
                    if max_samples_per_source is not None and saved_counts_seg.get(dataset_name, 0) >= max_samples_per_source:
                        continue
                    target_dir = os.path.join(demo_root, dataset_name, epoch_dir_name, "seg")
                    os.makedirs(target_dir, exist_ok=True)
                    image_tensor = batch["image"][idx].detach().cpu()
                    image_rgb = _denormalize_image(image_tensor)
                    pred_mask = preds[idx:idx + 1].argmax(dim=1).squeeze(0).cpu()
                    pred_color = _seg_to_color(pred_mask, config.num_classes)
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    tiles = [image_bgr, pred_color[:, :, ::-1]]

                    if "semseg_mask" in batch:
                        gt_mask = batch["semseg_mask"][idx].detach().cpu()
                        gt_color = _seg_to_color(gt_mask, config.num_classes)
                        tiles.append(gt_color[:, :, ::-1])

                    seg_vis = np.concatenate(tiles, axis=1)
                    vis_path = os.path.join(target_dir, f"{base_name}_seg_vis.png")
                    cv2.imwrite(vis_path, seg_vis)

                    if max_samples_per_source is not None:
                        saved_counts_seg[dataset_name] = saved_counts_seg.get(dataset_name, 0) + 1

                if max_samples_per_source is not None and observed_sources_seg and all(
                    saved_counts_seg.get(name, 0) >= max_samples_per_source for name in observed_sources_seg
                ):
                    break

    logger.info(f"Saved demo predictions for epoch {epoch + 1} -> {demo_root}")


def _select_primary_dataset(names: Optional[Any]) -> Optional[str]:
    if names is None:
        return None
    if isinstance(names, torch.Tensor):
        try:
            names = names.detach().cpu().tolist()
        except Exception:
            names = names.tolist()
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if isinstance(names, (set, tuple)):
        names = list(names)
    if isinstance(names, list):
        filtered = [name for name in names if name is not None]
        if not filtered:
            return None
        counter = Counter(filtered)
        primary, _ = counter.most_common(1)[0]
        return primary
    return names


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_BASE_SEGMENT_PALETTE = np.array(
    [
        [52, 63, 92],
        [87, 108, 138],
        [132, 153, 178],
        [176, 191, 206],
        [76, 103, 103],
        [121, 146, 139],
        [168, 188, 171],
        [210, 221, 205],
        [94, 86, 101],
        [136, 128, 142],
        [178, 170, 184],
        [214, 207, 219],
        [96, 100, 80],
        [139, 145, 122],
        [182, 188, 168],
        [219, 222, 206],
        [115, 90, 90],
        [155, 125, 126],
        [194, 166, 168],
        [226, 204, 204],
    ],
    dtype=np.uint8,
)
SEGMENT_COLOR_CACHE: Dict[int, np.ndarray] = {}


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized CHW tensor back to HWC uint8 image.
    """
    arr = image_tensor.cpu().numpy()
    arr = arr * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def _get_segment_color_map(num_classes: int) -> np.ndarray:
    if num_classes in SEGMENT_COLOR_CACHE:
        return SEGMENT_COLOR_CACHE[num_classes]
    base = _BASE_SEGMENT_PALETTE
    if num_classes <= base.shape[0]:
        palette = base[:max(num_classes, 1)]
    else:
        repeats = int(np.ceil(num_classes / base.shape[0]))
        palette = np.tile(base, (repeats, 1))[:num_classes]
    SEGMENT_COLOR_CACHE[num_classes] = palette
    return palette


def _depth_to_colormap(depth_tensor: torch.Tensor,
                       valid_mask: Optional[torch.Tensor],
                       max_depth: float) -> np.ndarray:
    depth_np = depth_tensor.squeeze().cpu().numpy()
    mask_np = valid_mask.squeeze().cpu().numpy().astype(bool) if valid_mask is not None else None

    if max_depth <= 0:
        if mask_np is not None and mask_np.any():
            max_depth = float(np.max(depth_np[mask_np]))
        else:
            max_depth = float(np.max(depth_np))
        if max_depth <= 0:
            max_depth = 1.0

    norm = np.clip(depth_np / max_depth, 0.0, 1.0)
    colored = plt.get_cmap("viridis")(norm)[:, :, :3]
    if mask_np is not None:
        colored[~mask_np] = 0.0
    return (colored * 255).astype(np.uint8)


def _seg_to_color(mask_tensor: torch.Tensor,
                  num_classes: int,
                  ignore_index: int = 255) -> np.ndarray:
    label_np = mask_tensor.squeeze().cpu().numpy().astype(np.int32)
    effective_classes = num_classes if num_classes > 0 else int(label_np[label_np != ignore_index].max() + 1) if np.any(label_np != ignore_index) else 1
    palette = _get_segment_color_map(effective_classes)
    safe_indices = np.mod(label_np.clip(min=0), palette.shape[0])
    color_indices = safe_indices
    color_img = palette[color_indices]
    if ignore_index is not None:
        ignore_mask = (label_np == ignore_index)
        color_img[ignore_mask] = 0
    return color_img


def _log_depth_samples_to_tensorboard(writer: SummaryWriter,
                                      samples: Dict[str, Dict[str, torch.Tensor]],
                                      epoch: int,
                                      max_depth: float) -> None:
    for name, sample in samples.items():
        image = _denormalize_image(sample["image"])
        pred_color = _depth_to_colormap(sample["pred"], sample.get("valid_mask"), max_depth)
        gt_color = _depth_to_colormap(sample["gt"], sample.get("valid_mask"), max_depth)
        h = min(image.shape[0], pred_color.shape[0], gt_color.shape[0])
        image = image[:h]
        pred_color = pred_color[:h]
        gt_color = gt_color[:h]
        combined = np.concatenate([image, pred_color, gt_color], axis=1)
        writer.add_image(f"ValidationSamples/Depth/{name}", combined, epoch, dataformats="HWC")


def _log_seg_samples_to_tensorboard(writer: SummaryWriter,
                                    samples: Dict[str, Dict[str, torch.Tensor]],
                                    epoch: int,
                                    num_classes: int) -> None:
    for name, sample in samples.items():
        image = _denormalize_image(sample["image"])
        pred_color = _seg_to_color(sample["pred"], num_classes)
        gt_color = _seg_to_color(sample["gt"], num_classes)
        h = min(image.shape[0], pred_color.shape[0], gt_color.shape[0])
        image = image[:h]
        pred_color = pred_color[:h]
        gt_color = gt_color[:h]
        combined = np.concatenate([image, pred_color, gt_color], axis=1)
        writer.add_image(f"ValidationSamples/Seg/{name}", combined, epoch, dataformats="HWC")


def _build_progress_tracker(loader: DataLoader) -> Optional[Dict[str, Any]]:
    summary = summarize_loader_composition(loader)
    if not summary:
        return None
    totals = {entry['name']: entry['count'] for entry in summary}
    types = {entry['name']: entry.get('dataset_type', 'unknown') for entry in summary}
    return {
        "totals": totals,
        "types": types,
        "counts": {name: 0 for name in totals},
        "started": set(),
        "milestones": {name: set() for name in totals},
        "completed": set(),
    }


def _log_dataset_progress(logger: logging.Logger,
                          phase: str,
                          task: str,
                          epoch: int,
                          names: Optional[Any],
                          tracker: Optional[Dict[str, Any]],
                          rank: int) -> None:
    if rank != 0 or tracker is None or names is None:
        return
    if isinstance(names, torch.Tensor):
        names = names.tolist()
    if isinstance(names, str):
        names_list = [names]
    else:
        names_list = list(names)
    if not names_list:
        return
    primary = _select_primary_dataset(names_list)
    if primary is None:
        return
    increment = sum(1 for n in names_list if n == primary)
    if increment <= 0:
        increment = len(names_list)
    tracker['counts'][primary] = tracker['counts'].get(primary, 0) + increment
    processed = tracker['counts'][primary]
    total = tracker['totals'].get(primary, "unknown")
    dataset_type = tracker['types'].get(primary, "unknown")
    if primary not in tracker['started']:
        logger.info(f"[{phase}][{task}][Epoch {epoch}] Started dataset {primary} ({dataset_type}) 0/{total}")
        tracker['started'].add(primary)
    if isinstance(total, int) and total > 0:
        progress = processed / total
        if progress >= 1.0 and primary not in tracker['completed']:
            logger.info(f"[{phase}][{task}][Epoch {epoch}] Finished dataset {primary} ({dataset_type}) ({processed}/{total})")
            tracker['completed'].add(primary)
        else:
            milestone = math.floor(progress * 10) / 10.0
            if milestone >= 0.1 and milestone < 1.0:
                seen = tracker['milestones'].setdefault(primary, set())
                if milestone not in seen:
                    seen.add(milestone)
                    percent = int(milestone * 100)
                    logger.info(f"[{phase}][{task}][Epoch {epoch}] {primary} ({dataset_type}) progress {percent}% ({processed}/{total})")


def validate_and_visualize(model: torch.nn.Module,
                           val_loader: DataLoader,
                           task_type: str,
                           epoch: int,
                           config: TrainingConfig,
                           writer: SummaryWriter,
                           logger: logging.Logger,
                           dataset_name: Optional[str] = None) -> Optional[Dict[str, Dict[str, float]]]:
    """
   分布式验证和可视化函数
   """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    model.eval()

    task_display = "Depth" if task_type == 'depth' else "Seg"
    tracker = _build_progress_tracker(val_loader) if rank == 0 else None

    total_loss_local = 0.0
    batch_count_local = 0

    tb_depth_samples: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    tb_seg_samples: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    temp_validator = None
    if task_type == 'depth':
        temp_validator = DepthValidator(config)
        tb_depth_samples = {} if rank == 0 else None
    elif task_type == 'seg':
        temp_validator = SegValidator(config)
        tb_seg_samples = {} if rank == 0 else None

    # 禁止训练过程写入可视化图片，避免产生 pred_depth/pred_seg 等目录
    save_visuals = False
    depth_visual_samples: Optional[Dict[str, Dict[str, Any]]] = None
    seg_visual_samples: Optional[Dict[str, Dict[str, Any]]] = None

    if rank == 0 and temp_validator is not None:
        temp_validator.setup_output_folder(config.save_path)

    alias_map = {
        'kidney': 'NO',
        'colon': 'NO',
        'no': 'NO',
        'ls': 'LS',
        'endovis2017': 'LS',
        'endovis2018': 'LS',
        'endonerf': 'LS',
    }

    # --- 按任务类型初始化临时统计结构 ---
    if task_type == 'depth':
        # [0]: valid_pixels, [1]: thresh1.25, [2]: thresh1.25^2, [3]: thresh1.25^3,
        # [4]: abs_rel_sum, [5]: sq_rel_sum, [6]: diff_sq_sum, [7]: diff_log_sq_sum,
        # [8]: log10_sum, [9]: diff_log_sum, [10]: abs_diff_sum (m), [11]: within_2cm_count
        def _create_depth_stats() -> torch.Tensor:
            return torch.zeros(12, device='cuda')

        depth_stats = {
            "NO": _create_depth_stats(),
            "LS": _create_depth_stats(),
            "combined": _create_depth_stats()
        }
        dataset_depth_stats: Dict[str, torch.Tensor] = {}

        CAMERA_NUMERIC_KEYS = [
            "fx_abs",
            "fy_abs",
            "fx_pct",
            "fy_pct",
            "fx_sq",
            "fy_sq",
            "cx_abs",
            "cy_abs",
            "cx_pct",
            "cy_pct",
            "cx_sq",
            "cy_sq",
            "principal_dist",
            "principal_sq",
            "principal_pct",
            "principal_pct_sq",
            "reproj_sum",
        ]
        CAMERA_SAMPLE_KEYS = [
            "f_pair_samples",
            "c_pair_samples",
            "principal_samples",
            "principal_pct_samples",
            "reproj_samples",
        ]
        _REPROJ_SAMPLES = 16

        _reproj_grid_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

        def _create_camera_metrics_bucket() -> Dict[str, Any]:
            bucket: Dict[str, Any] = {key: 0.0 for key in CAMERA_NUMERIC_KEYS}
            bucket["count"] = 0.0
            for key in CAMERA_SAMPLE_KEYS:
                bucket[key] = []
            return bucket

        def _accumulate_camera_bucket(dest: Dict[str, Any], src: Dict[str, Any]) -> None:
            for key in CAMERA_NUMERIC_KEYS:
                dest[key] += float(src.get(key, 0.0))
            dest["count"] += float(src.get("count", 0.0))
            for key in CAMERA_SAMPLE_KEYS:
                dest[key].extend(src.get(key, []))

        def _update_camera_metrics(bucket: Dict[str, Any], sample: Dict[str, float]) -> None:
            bucket["fx_abs"] += sample["fx_abs"]
            bucket["fy_abs"] += sample["fy_abs"]
            bucket["fx_pct"] += sample["fx_pct"]
            bucket["fy_pct"] += sample["fy_pct"]
            bucket["fx_sq"] += sample["fx_abs"] ** 2
            bucket["fy_sq"] += sample["fy_abs"] ** 2
            bucket["cx_abs"] += sample["cx_abs"]
            bucket["cy_abs"] += sample["cy_abs"]
            bucket["cx_pct"] += sample["cx_pct"]
            bucket["cy_pct"] += sample["cy_pct"]
            bucket["cx_sq"] += sample["cx_abs"] ** 2
            bucket["cy_sq"] += sample["cy_abs"] ** 2
            bucket["principal_dist"] += sample["principal_dist"]
            bucket["principal_sq"] += sample["principal_dist"] ** 2
            bucket["principal_pct"] += sample["principal_pct"]
            bucket["principal_pct_sq"] += sample["principal_pct"] ** 2
            reproj_val = sample.get("reproj_err")
            if reproj_val is not None:
                bucket["reproj_sum"] += reproj_val
                bucket["reproj_samples"].append(reproj_val)
            bucket["count"] += 1
            bucket["f_pair_samples"].append(sample["f_pair"])
            bucket["c_pair_samples"].append(sample["c_pair"])
            bucket["principal_samples"].append(sample["principal_dist"])
            bucket["principal_pct_samples"].append(sample["principal_pct"])

        def _compute_camera_sample_metrics(
            fx_gt: float,
            fy_gt: float,
            cx_gt: float,
            cy_gt: float,
            fx_pred: float,
            fy_pred: float,
            cx_pred: float,
            cy_pred: float,
            width: int,
            height: int,
        ) -> Dict[str, float]:
            fx_abs = abs(fx_pred - fx_gt)
            fy_abs = abs(fy_pred - fy_gt)
            cx_abs = abs(cx_pred - cx_gt)
            cy_abs = abs(cy_pred - cy_gt)
            fx_pct = fx_abs / max(abs(fx_gt), 1e-6)
            fy_pct = fy_abs / max(abs(fy_gt), 1e-6)
            cx_pct = cx_abs / max(float(width), 1e-6)
            cy_pct = cy_abs / max(float(height), 1e-6)
            principal_dist = math.hypot(cx_abs, cy_abs)
            principal_pct = math.hypot(cx_pct, cy_pct)
            f_pair = math.sqrt(0.5 * (fx_abs ** 2 + fy_abs ** 2))
            c_pair = math.sqrt(0.5 * (cx_abs ** 2 + cy_abs ** 2))
            reproj_err = _compute_reprojection_error(
                fx_gt=fx_gt,
                fy_gt=fy_gt,
                cx_gt=cx_gt,
                cy_gt=cy_gt,
                fx_pred=fx_pred,
                fy_pred=fy_pred,
                cx_pred=cx_pred,
                cy_pred=cy_pred,
                width=width,
                height=height,
            )
            return {
                "fx_abs": fx_abs,
                "fy_abs": fy_abs,
                "fx_pct": fx_pct,
                "fy_pct": fy_pct,
                "cx_abs": cx_abs,
                "cy_abs": cy_abs,
                "cx_pct": cx_pct,
                "cy_pct": cy_pct,
                "principal_dist": principal_dist,
                "principal_pct": principal_pct,
                "f_pair": f_pair,
                "c_pair": c_pair,
                "reproj_err": reproj_err,
            }

        _TB_SAMPLES_PER_DATASET = 4

        def _collect_depth_samples(store: Dict[str, List[Dict[str, torch.Tensor]]],
                                   batch: Dict[str, Any],
                                   pred: torch.Tensor,
                                   valid_mask: torch.Tensor,
                                   dataset_name: Optional[str],
                                   sample_idx: int,
                                   batch_idx: int,
                                   config: TrainingConfig) -> None:
            """
            Cache a few representative samples per dataset for potential downstream visualization.
            """
            safe_name = _sanitize_name(dataset_name) or "unknown"
            samples = store.setdefault(safe_name, [])
            if len(samples) >= _TB_SAMPLES_PER_DATASET:
                return

            sample = {
                "image": batch["image"][sample_idx].detach().cpu(),
                "pred": pred[sample_idx].detach().cpu(),
                "mask": valid_mask[sample_idx].detach().cpu(),
                "meta": {
                    "dataset": dataset_name or "unknown",
                    "batch_idx": batch_idx,
                    "epoch": epoch,
                    "img_size": config.img_size,
                },
            }
            samples.append(sample)

        def _finalize_camera_metrics(bucket: Dict[str, Any]) -> Dict[str, float]:
            count = max(bucket.get("count", 0.0), 1.0)
            fx_abs_mean = bucket["fx_abs"] / count
            fy_abs_mean = bucket["fy_abs"] / count
            cx_abs_mean = bucket["cx_abs"] / count
            cy_abs_mean = bucket["cy_abs"] / count
            cx_pct_mean = bucket["cx_pct"] / count
            cy_pct_mean = bucket["cy_pct"] / count
            fx_pct_mean = bucket["fx_pct"] / count
            fy_pct_mean = bucket["fy_pct"] / count
            principal_mean = bucket["principal_dist"] / count
            principal_pct_mean = bucket["principal_pct"] / count
            reproj_sum = bucket.get("reproj_sum", 0.0)
            reproj_mean = (reproj_sum / count) if reproj_sum > 0 else None

            fx_rmse = math.sqrt(max(bucket["fx_sq"] + bucket["fy_sq"], 0.0) / max(2.0 * count, 1e-6))
            cx_rmse = math.sqrt(max(bucket["cx_sq"] + bucket["cy_sq"], 0.0) / max(2.0 * count, 1e-6))
            principal_rmse = math.sqrt(max(bucket["principal_sq"], 0.0) / max(count, 1e-6))
            principal_pct_rmse = math.sqrt(max(bucket["principal_pct_sq"], 0.0) / max(count, 1e-6))

            def _median(samples: List[float]) -> Optional[float]:
                if not samples:
                    return None
                return float(statistics.median(samples))

            metrics = {
                "fx_abs": fx_abs_mean,
                "fy_abs": fy_abs_mean,
                "fx_pct": fx_pct_mean,
                "fy_pct": fy_pct_mean,
                "fx_rmse": fx_rmse,
                "cx_abs": cx_abs_mean,
                "cy_abs": cy_abs_mean,
                "cx_pct": cx_pct_mean,
                "cy_pct": cy_pct_mean,
                "c_rmse": cx_rmse,
                "principal_dist": principal_mean,
                "principal_rmse": principal_rmse,
                "principal_pct": principal_pct_mean,
                "principal_pct_rmse": principal_pct_rmse,
                "reproj_mean": reproj_mean,
                "median_f_abs": _median(bucket["f_pair_samples"]),
                "median_c_abs": _median(bucket["c_pair_samples"]),
                "principal_median": _median(bucket["principal_samples"]),
                "principal_pct_median": _median(bucket["principal_pct_samples"]),
                "reproj_median": _median(bucket["reproj_samples"]),
            }
            return metrics

        def _compute_reprojection_error(
            fx_gt: float,
            fy_gt: float,
            cx_gt: float,
            cy_gt: float,
            fx_pred: float,
            fy_pred: float,
            cx_pred: float,
            cy_pred: float,
            width: int,
            height: int,
            grid: int = _REPROJ_SAMPLES,
        ) -> float:
            if width <= 1 or height <= 1:
                return 0.0
            steps_x = min(grid, width)
            steps_y = min(grid, height)
            cache_key = (steps_x, steps_y, width, height)
            grid_x, grid_y = _reproj_grid_cache.get(cache_key, (None, None))
            if grid_x is None or grid_y is None:
                xs = np.linspace(0.0, max(width - 1, 1), num=steps_x, dtype=np.float64)
                ys = np.linspace(0.0, max(height - 1, 1), num=steps_y, dtype=np.float64)
                grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
                _reproj_grid_cache[cache_key] = (grid_x, grid_y)
            fx_gt_safe = max(abs(fx_gt), 1e-6)
            fy_gt_safe = max(abs(fy_gt), 1e-6)
            dir_x = (grid_x - cx_gt) / fx_gt_safe
            dir_y = (grid_y - cy_gt) / fy_gt_safe
            u_pred = fx_pred * dir_x + cx_pred
            v_pred = fy_pred * dir_y + cy_pred
            err = np.sqrt((u_pred - grid_x) ** 2 + (v_pred - grid_y) ** 2)
            return float(err.mean())

        camera_metrics_total = _create_camera_metrics_bucket()
        camera_metrics_by_dataset: Dict[str, Dict[str, Any]] = {}

        def _update_depth_stats(stats_tensor: torch.Tensor,
                                pred_valid: torch.Tensor,
                                gt_valid: torch.Tensor) -> None:
            if pred_valid.numel() == 0:
                return
            eps = 1e-6
            pred_clamped = pred_valid.clamp(min=eps)
            gt_clamped = gt_valid.clamp(min=eps)

            thresh = torch.maximum(gt_clamped / pred_clamped, pred_clamped / gt_clamped)
            diff = pred_clamped - gt_clamped
            diff_sq = diff.square()
            abs_diff = torch.abs(diff)
            diff_log = torch.log(pred_clamped) - torch.log(gt_clamped)

            stats_tensor[0] += pred_clamped.numel()
            stats_tensor[1] += (thresh < 1.25).sum()
            stats_tensor[2] += (thresh < 1.25 ** 2).sum()
            stats_tensor[3] += (thresh < 1.25 ** 3).sum()
            stats_tensor[4] += torch.sum(abs_diff / gt_clamped)
            stats_tensor[5] += torch.sum(diff_sq / gt_clamped)
            stats_tensor[6] += torch.sum(diff_sq)
            stats_tensor[7] += torch.sum(diff_log.square())
            stats_tensor[8] += torch.sum(torch.abs(torch.log10(pred_clamped) - torch.log10(gt_clamped)))
            stats_tensor[9] += torch.sum(diff_log)
            stats_tensor[10] += torch.sum(abs_diff)
            stats_tensor[11] += (abs_diff <= 0.02).sum()

    elif task_type == 'seg':
        seg_metrics: Dict[str, SegMetric] = {
            "NO": SegMetric(config.num_classes),
            "LS": SegMetric(config.num_classes),
            "combined": SegMetric(config.num_classes),
        }
        dataset_seg_metrics: Dict[str, SegMetric] = {}

        def _resolve_dataset_metric(dataset_label: Optional[str], raw_source: str) -> SegMetric:
            name_key = (dataset_label or raw_source or "unknown").strip().lower()
            if name_key not in dataset_seg_metrics:
                dataset_seg_metrics[name_key] = SegMetric(config.num_classes)
            return dataset_seg_metrics[name_key]

    def _process_validation_batch(batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        input_img = batch["image"].cuda()
        outputs: Optional[Dict[str, torch.Tensor]] = None
        loss = torch.tensor(0.0, device=input_img.device)

        if task_type == 'depth':
            _log_dataset_progress(logger, "Val", task_display, epoch, batch.get("dataset_name"), tracker, rank)
            target_gt = batch["depth"].cuda()
            outputs = model(input_img, task='depth')
            pred = outputs['depth']
            valid_mask_4d = (target_gt > 0) & (target_gt >= config.min_depth) & (target_gt <= config.max_depth)
            loss = temp_validator.criterion(pred, target_gt, valid_mask_4d)

            if batch_idx == 0 and rank == 0:
                temp_validator.save_prediction(pred, epoch)

            source_types = batch.get("source_type", None)
            dataset_names = batch.get("dataset_name", None)
            for j in range(pred.shape[0]):
                valid_mask = valid_mask_4d[j].squeeze(0)
                if valid_mask.numel() == 0 or valid_mask.sum().item() == 0:
                    continue
                pred_slice = pred[j].squeeze(0)
                gt_slice = target_gt[j].squeeze(0)

                if pred_slice.shape != valid_mask.shape:
                    logger.error(
                        "[Validation] Shape mismatch before masking (task=%s, dataset=%s, sample=%d, pred=%s, mask=%s)",
                        task_type,
                        dataset_names[j] if dataset_names and j < len(dataset_names) else None,
                        j,
                        tuple(pred_slice.shape),
                        tuple(valid_mask.shape),
                    )
                    raise RuntimeError(
                        f"Pred/mask shape mismatch: pred={tuple(pred_slice.shape)} mask={tuple(valid_mask.shape)}"
                    )

                pred_valid = pred_slice[valid_mask]
                gt_valid = gt_slice[valid_mask]

                dataset_raw = None
                if dataset_names and j < len(dataset_names):
                    dataset_raw = dataset_names[j]
                dataset_name = str(dataset_raw) if dataset_raw else None
                if dataset_name:
                    stats_tensor_ds = dataset_depth_stats.setdefault(dataset_name, _create_depth_stats())
                    _update_depth_stats(stats_tensor_ds, pred_valid, gt_valid)

                raw_type = (source_types[j] if source_types and j < len(source_types) else "combined")
                mapped = alias_map.get(str(raw_type).lower(), None)
                source_type = mapped if mapped is not None else ("combined" if raw_type not in depth_stats else raw_type)

                if source_type not in depth_stats:
                    logger.warning(f"Unknown source_type '{raw_type}' encountered during depth validation. Falling back to 'combined'.")
                    source_type = "combined"

                _update_depth_stats(depth_stats[source_type], pred_valid, gt_valid)
                if source_type != "combined":
                    _update_depth_stats(depth_stats["combined"], pred_valid, gt_valid)

                camera_batch = batch.get('camera_intrinsics')
                camera_mask = batch.get('camera_intrinsics_mask')
                if camera_mask is not None and not torch.is_tensor(camera_mask):
                    camera_mask = torch.as_tensor(camera_mask, dtype=torch.bool)
                pred_camera_batch = outputs.get('camera_intrinsics') if outputs else None
                if (
                    camera_batch is not None
                    and pred_camera_batch is not None
                    and j < len(camera_batch)
                    and j < pred_camera_batch.size(0)
                ):
                    if camera_mask is not None:
                        if camera_mask.dim() > 1:
                            mask_row = camera_mask[j].reshape(-1)[0]
                        else:
                            mask_row = camera_mask[j]
                        if not bool(mask_row):
                            continue
                    image_tensor = batch.get("image")
                    if image_tensor is None:
                        continue
                    height_px = int(image_tensor.shape[-2])
                    width_px = int(image_tensor.shape[-1])
                    pred_camera = pred_camera_batch[j].detach().cpu()
                    gt_camera_entry = camera_batch[j]
                    if torch.is_tensor(gt_camera_entry):
                        gt_camera = gt_camera_entry.cpu()
                    else:
                        gt_camera = torch.as_tensor(gt_camera_entry, dtype=torch.float32)

                    fx_pred = float(pred_camera[0, 0])
                    fy_pred = float(pred_camera[1, 1])
                    cx_pred = float(pred_camera[0, 2])
                    cy_pred = float(pred_camera[1, 2])

                    fx_gt = float(gt_camera[0, 0])
                    fy_gt = float(gt_camera[1, 1])
                    cx_gt = float(gt_camera[0, 2])
                    cy_gt = float(gt_camera[1, 2])

                    row_key = dataset_name or "unknown"
                    bucket = camera_metrics_by_dataset.setdefault(row_key, _create_camera_metrics_bucket())

                    camera_sample = _compute_camera_sample_metrics(
                        fx_gt=fx_gt,
                        fy_gt=fy_gt,
                        cx_gt=cx_gt,
                        cy_gt=cy_gt,
                        fx_pred=fx_pred,
                        fy_pred=fy_pred,
                        cx_pred=cx_pred,
                        cy_pred=cy_pred,
                        width=width_px,
                        height=height_px,
                    )

                    _update_camera_metrics(bucket, camera_sample)
                    _update_camera_metrics(camera_metrics_total, camera_sample)
                    bucket["count"] += 1
                    camera_metrics_total["count"] += 1

                if tb_depth_samples is not None and rank == 0:
                    _collect_depth_samples(tb_depth_samples, batch, pred, valid_mask_4d, dataset_name, j, batch_idx, config)

        elif task_type == 'seg':
            _log_dataset_progress(logger, "Val", task_display, epoch, batch.get("dataset_name"), tracker, rank)
            target_gt = batch["semseg_mask"].cuda().long()
            outputs = model(input_img, task='seg')
            pred = outputs['seg']
            dataset_names = batch.get("dataset_name", None)

            ignore_idx = 255
            valid_class_mask = (target_gt >= 0) & (target_gt < config.num_classes)
            ignore_mask = (target_gt == ignore_idx)
            valid_mask = valid_class_mask | ignore_mask
            sanitized_target = torch.where(valid_mask, target_gt, torch.full_like(target_gt, ignore_idx))

            loss = temp_validator.criterion(pred, sanitized_target)

            if batch_idx == 0 and rank == 0:
                temp_validator.save_prediction(pred, epoch)

            pred_labels = pred.argmax(dim=1)
            for j in range(pred.shape[0]):
                source_list = batch.get("source_type", [])
                raw_type = source_list[j] if j < len(source_list) else "combined"
                mapped = alias_map.get(str(raw_type).lower(), None)
                source_type = mapped if mapped is not None else ("combined" if raw_type not in seg_metrics else raw_type)

                if source_type not in seg_metrics:
                    logger.warning(f"Unknown source_type '{raw_type}' encountered during seg validation. Falling back to 'combined'.")
                    source_type = "combined"

                per_sample_loss = loss.item() / max(pred.shape[0], 1)
                seg_metrics[source_type].update(sanitized_target[j].unsqueeze(0), pred_labels[j].unsqueeze(0), batch_loss=per_sample_loss)
                if source_type != "combined":
                    seg_metrics["combined"].update(sanitized_target[j].unsqueeze(0), pred_labels[j].unsqueeze(0), batch_loss=per_sample_loss)

                if tb_seg_samples is not None:
                    display_name = _sanitize_name(raw_type)
                    if display_name.lower() == "combined":
                        display_name = _sanitize_name(source_type)
                    if display_name and display_name not in tb_seg_samples:
                        tb_seg_samples[display_name] = {
                            "image": batch["image"][j].detach().cpu(),
                            "pred": pred_labels[j].detach().cpu(),
                            "gt": sanitized_target[j].detach().cpu(),
                        }

                if seg_visual_samples is not None:
                    dataset_raw = None
                    if dataset_names and j < len(dataset_names):
                        dataset_raw = dataset_names[j]
                    if dataset_raw is None:
                        dataset_raw = raw_type
                    dataset_label = str(dataset_raw)
                    dataset_key = _sanitize_name(dataset_label) or "unknown"
                    entry = seg_visual_samples.setdefault(dataset_key, {"label": dataset_label, "samples": []})
                    if len(entry["samples"]) < 3:
                        entry["samples"].append({
                            "image": batch["image"][j].detach().cpu(),
                            "pred": pred_labels[j].detach().cpu(),
                            "gt": sanitized_target[j].detach().cpu(),
                        })

        return loss

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                loss = _process_validation_batch(batch, i)
            except Exception as exc:
                if config.tolerate_validation_errors:
                    dataset_info = batch.get("dataset_name") if isinstance(batch, dict) else None
                    logger.error(
                        "Validation batch failed (task=%s, dataset=%s, batch_idx=%d): %s",
                        task_type,
                        dataset_info,
                        i,
                        exc,
                    )
                    logger.debug("Validation exception details", exc_info=exc)
                    continue
                raise

            total_loss_local += loss.item()
            batch_count_local += 1


    # 同步所有进程
    if dist.is_initialized() and world_size > 1:
        dist.barrier()

    # 在进入主进程聚合前，先执行跨进程规约
    if task_type == 'depth' and dist.is_initialized() and world_size > 1:
        for stats_tensor in depth_stats.values():
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

        gathered_totals: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_totals, camera_metrics_total)
        merged_total = _create_camera_metrics_bucket()
        for entry in gathered_totals:
            if entry:
                _accumulate_camera_bucket(merged_total, entry)
        camera_metrics_total = merged_total

        gathered_camera_buckets: List[Optional[Dict[str, Dict[str, Any]]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_camera_buckets, camera_metrics_by_dataset)
        merged_camera_metrics: Dict[str, Dict[str, Any]] = {}
        for entry in gathered_camera_buckets:
            if not entry:
                continue
            for dataset_key, bucket in entry.items():
                accumulator = merged_camera_metrics.setdefault(dataset_key, _create_camera_metrics_bucket())
                _accumulate_camera_bucket(accumulator, bucket)
        camera_metrics_by_dataset = merged_camera_metrics
    elif task_type == 'seg' and dist.is_initialized() and world_size > 1:
        for metric_calc in seg_metrics.values():
            dist.all_reduce(metric_calc.confusion_matrix, op=dist.ReduceOp.SUM)

            total_loss_tensor_tmp = torch.tensor(metric_calc.total_loss, device='cuda')
            dist.all_reduce(total_loss_tensor_tmp, op=dist.ReduceOp.SUM)
            metric_calc.total_loss = total_loss_tensor_tmp.item()

            batch_tensor_tmp = torch.tensor(metric_calc.number_of_batches, device='cuda')
            dist.all_reduce(batch_tensor_tmp, op=dist.ReduceOp.SUM)
            metric_calc.number_of_batches = batch_tensor_tmp.item()

            total_pixels_tensor_tmp = torch.tensor(metric_calc.total_pixels, device='cuda')
            dist.all_reduce(total_pixels_tensor_tmp, op=dist.ReduceOp.SUM)
            metric_calc.total_pixels = total_pixels_tensor_tmp.item()

            correct_pixels_tensor_tmp = torch.tensor(metric_calc.correct_pixels, device='cuda')
            dist.all_reduce(correct_pixels_tensor_tmp, op=dist.ReduceOp.SUM)
            metric_calc.correct_pixels = correct_pixels_tensor_tmp.item()

    # 收集所有进程的损失
    total_loss_tensor = torch.tensor(total_loss_local, device='cuda')
    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

    batch_count_tensor = torch.tensor(batch_count_local, device='cuda')
    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        # --- 在主进程上处理和计算聚合后的指标 ---
        all_metrics: Dict[str, Dict[str, float]] = {}
        domain_metrics: Dict[str, Dict[str, float]] = {}
        dataset_metrics: Dict[str, Dict[str, float]] = {}

        # 计算平均损失
        total_batches = max(int(batch_count_tensor.item()), 1)
        avg_loss = total_loss_tensor.item() / total_batches
        if writer is not None:
            writer.add_scalar(f'Validation_Loss/{task_type}', avg_loss, epoch)

        if task_type == 'depth':
            # 计算指标
            for name, stats_tensor in depth_stats.items():
                stats_cpu = stats_tensor.detach().cpu()
                valid_pixels = stats_cpu[0].item()
                if valid_pixels <= 0:
                    logger.warning(f"No valid pixels for depth validation on '{name}'.")
                    continue

                thresh1 = stats_cpu[1].item() / valid_pixels
                thresh2 = stats_cpu[2].item() / valid_pixels
                thresh3 = stats_cpu[3].item() / valid_pixels
                absrel = stats_cpu[4].item() / valid_pixels
                sq_rel = stats_cpu[5].item() / valid_pixels
                rmse = float(np.sqrt(stats_cpu[6].item() / valid_pixels))
                rmse_log = float(np.sqrt(stats_cpu[7].item() / valid_pixels))
                log10_val = stats_cpu[8].item() / valid_pixels
                mean_diff_log = stats_cpu[9].item() / valid_pixels
                mae_cm = (stats_cpu[10].item() / valid_pixels) * 100.0
                acc_2cm = stats_cpu[11].item() / valid_pixels
                silog_inner = stats_cpu[7].item() / valid_pixels - 0.5 * (mean_diff_log ** 2)
                silog_inner = max(silog_inner, 0.0)
                silog = float(np.sqrt(silog_inner))

                metrics = {
                    'd1': thresh1,
                    'd2': thresh2,
                    'd3': thresh3,
                    'absrel': absrel,
                    'sq_rel': sq_rel,
                    'rmse': rmse,
                    'rmse_log': rmse_log,
                    'log10': log10_val,
                    'silog': silog,
                    'mae_cm': mae_cm,
                    'acc_2cm': acc_2cm,
                }
                domain_metrics[name] = metrics
                all_metrics[name] = metrics
                display_name = name.upper() if name != 'combined' else name.capitalize()
                logger.info(
                    f"[{display_name}] Depth Validation Epoch {epoch} - absrel: {metrics.get('absrel', 0):.4f}, "
                    f"rmse: {metrics.get('rmse', 0):.4f}, mae_cm: {metrics.get('mae_cm', 0):.3f}, "
                    f"acc@2cm: {metrics.get('acc_2cm', 0):.4f}"
                )
                for k, v in metrics.items():
                    if writer is not None:
                        writer.add_scalar(f'Metrics/Depth_{display_name}/{k}', v, epoch)
                    logger.info(f"    {k}: {v:.6f}")

            for dataset_name, stats_tensor in dataset_depth_stats.items():
                stats_cpu = stats_tensor.detach().cpu()
                valid_pixels = stats_cpu[0].item()
                if valid_pixels <= 0:
                    continue

                thresh1 = stats_cpu[1].item() / valid_pixels
                thresh2 = stats_cpu[2].item() / valid_pixels
                thresh3 = stats_cpu[3].item() / valid_pixels
                absrel = stats_cpu[4].item() / valid_pixels
                sq_rel = stats_cpu[5].item() / valid_pixels
                rmse = float(np.sqrt(stats_cpu[6].item() / valid_pixels))
                rmse_log = float(np.sqrt(stats_cpu[7].item() / valid_pixels))
                log10_val = stats_cpu[8].item() / valid_pixels
                mean_diff_log = stats_cpu[9].item() / valid_pixels
                mae_cm = (stats_cpu[10].item() / valid_pixels) * 100.0
                acc_2cm = stats_cpu[11].item() / valid_pixels
                silog_inner = stats_cpu[7].item() / valid_pixels - 0.5 * (mean_diff_log ** 2)
                silog_inner = max(silog_inner, 0.0)
                silog = float(np.sqrt(silog_inner))

                metrics = {
                    'd1': thresh1,
                    'd2': thresh2,
                    'd3': thresh3,
                    'absrel': absrel,
                    'sq_rel': sq_rel,
                    'rmse': rmse,
                    'rmse_log': rmse_log,
                    'log10': log10_val,
                    'silog': silog,
                    'mae_cm': mae_cm,
                    'acc_2cm': acc_2cm,
                }
                dataset_metrics[dataset_name] = metrics
                safe_name = _sanitize_name(dataset_name)
                logger.info(
                    f"[Dataset:{dataset_name}] Depth Validation Epoch {epoch} - absrel: {metrics.get('absrel', 0):.4f}, "
                    f"rmse: {metrics.get('rmse', 0):.4f}, mae_cm: {metrics.get('mae_cm', 0):.3f}, "
                    f"acc@2cm: {metrics.get('acc_2cm', 0):.4f}"
                )
                for k, v in metrics.items():
                    if writer is not None:
                        writer.add_scalar(f'Metrics/Depth_Dataset/{safe_name}/{k}', v, epoch)
                    logger.info(f"    {k}: {v:.6f}")

            if camera_metrics_total["count"] > 0:
                overall_camera = _finalize_camera_metrics(camera_metrics_total)
                all_metrics['camera_overall'] = overall_camera
                reproj_mean = overall_camera.get('reproj_mean')
                reproj_str = f"{reproj_mean:.2f}" if reproj_mean is not None else "N/A"
                overall_principal_median = overall_camera.get('principal_median')
                overall_principal_median_str = (
                    f"{overall_principal_median:.2f}" if overall_principal_median is not None else "N/A"
                )
                overall_principal_pct_median = overall_camera.get('principal_pct_median')
                overall_principal_pct_median_str = (
                    f"{overall_principal_pct_median:.4f}" if overall_principal_pct_median is not None else "N/A"
                )
                logger.info(
                    f"[Camera] Depth Validation Epoch {epoch} - fx_abs: {overall_camera['fx_abs']:.2f}, "
                    f"fy_abs: {overall_camera['fy_abs']:.2f}, fx_pct: {overall_camera['fx_pct']:.4f}, "
                    f"fy_pct: {overall_camera['fy_pct']:.4f}, f_rmse_px: {overall_camera['fx_rmse']:.2f}, "
                    f"cx_abs: {overall_camera['cx_abs']:.2f}, cy_abs: {overall_camera['cy_abs']:.2f}, "
                    f"cx_pct: {overall_camera['cx_pct']:.4f}, cy_pct: {overall_camera['cy_pct']:.4f}, "
                    f"principal_px: {overall_camera['principal_dist']:.2f}, principal_pct: {overall_camera['principal_pct']:.4f}, "
                    f"principal_rmse_px: {overall_camera['principal_rmse']:.2f}, "
                    f"principal_pct_rmse: {overall_camera['principal_pct_rmse']:.4f}, "
                    f"principal_median_px: {overall_principal_median_str}, "
                    f"principal_pct_median: {overall_principal_pct_median_str}, "
                    f"c_rmse_px: {overall_camera['c_rmse']:.2f}, reproj_px: {reproj_str}"
                )
                if writer is not None:
                    for key, value in overall_camera.items():
                        if value is None:
                            continue
                        writer.add_scalar(f'Metrics/Camera/overall_{key}', value, epoch)

            if camera_metrics_by_dataset:
                for dataset_name, bucket in camera_metrics_by_dataset.items():
                    metrics_cam = _finalize_camera_metrics(bucket)
                    dataset_metrics.setdefault(dataset_name, {}).update({
                        'camera_fx_abs': metrics_cam['fx_abs'],
                        'camera_fy_abs': metrics_cam['fy_abs'],
                        'camera_fx_pct': metrics_cam['fx_pct'],
                        'camera_fy_pct': metrics_cam['fy_pct'],
                        'camera_principal_dist': metrics_cam['principal_dist'],
                        'camera_principal_pct': metrics_cam['principal_pct'],
                        'camera_c_rmse': metrics_cam['c_rmse'],
                        'camera_f_rmse': metrics_cam['fx_rmse'],
                        'camera_principal_rmse': metrics_cam['principal_rmse'],
                        'camera_principal_pct_rmse': metrics_cam['principal_pct_rmse'],
                        'camera_cx_pct': metrics_cam['cx_pct'],
                        'camera_cy_pct': metrics_cam['cy_pct'],
                        'camera_principal_median': metrics_cam.get('principal_median'),
                        'camera_principal_pct_median': metrics_cam.get('principal_pct_median'),
                        'camera_reproj': metrics_cam.get('reproj_mean'),
                    })
                    safe_name = _sanitize_name(dataset_name)
                    reproj_mean = metrics_cam.get('reproj_mean')
                    reproj_str = f"{reproj_mean:.2f}" if reproj_mean is not None else "N/A"
                    principal_median = metrics_cam.get('principal_median')
                    principal_median_str = f"{principal_median:.2f}" if principal_median is not None else "N/A"
                    principal_pct_median = metrics_cam.get('principal_pct_median')
                    principal_pct_median_str = (
                        f"{principal_pct_median:.4f}" if principal_pct_median is not None else "N/A"
                    )
                    logger.info(
                        f"[Dataset:{dataset_name}] Camera Metrics Epoch {epoch} - fx_abs: {metrics_cam['fx_abs']:.2f}, "
                        f"fy_abs: {metrics_cam['fy_abs']:.2f}, fx_pct: {metrics_cam['fx_pct']:.4f}, "
                        f"fy_pct: {metrics_cam['fy_pct']:.4f}, f_rmse_px: {metrics_cam['fx_rmse']:.2f}, "
                        f"cx_abs: {metrics_cam['cx_abs']:.2f}, cy_abs: {metrics_cam['cy_abs']:.2f}, "
                        f"cx_pct: {metrics_cam['cx_pct']:.4f}, cy_pct: {metrics_cam['cy_pct']:.4f}, "
                        f"principal_px: {metrics_cam['principal_dist']:.2f}, principal_pct: {metrics_cam['principal_pct']:.4f}, "
                        f"principal_rmse_px: {metrics_cam['principal_rmse']:.2f}, "
                        f"principal_pct_rmse: {metrics_cam['principal_pct_rmse']:.4f}, "
                        f"principal_median_px: {principal_median_str}, "
                        f"principal_pct_median: {principal_pct_median_str}, "
                        f"c_rmse_px: {metrics_cam['c_rmse']:.2f}, reproj_px: {reproj_str}"
                    )
                    if writer is not None:
                        for key, value in metrics_cam.items():
                            if value is None:
                                continue
                            writer.add_scalar(f'Metrics/Camera_Dataset/{safe_name}/{key}', value, epoch)

            if tb_depth_samples and logger is not None:
                logger.debug("[Validation] 深度样本已生成，按要求跳过 TensorBoard 记录。")

            if depth_visual_samples:
                _export_depth_visuals(depth_visual_samples, config.save_path, epoch + 1, logger)

        elif task_type == 'seg':
            seg_metrics_payload: Dict[str, Dict[str, float]] = {}
            for name, metric_calc in seg_metrics.items():
                seg_scores = metric_calc.get_scores()
                all_metrics[name] = seg_scores
                seg_metrics_payload[name] = {
                    k: float(v)
                    for k, v in seg_scores.items()
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))
                }
                display_name = _sanitize_name(name) or name
                logger.info(f"[Dataset:{display_name}] Seg Validation Epoch {epoch} - mIoU: {seg_scores.get('miou', 0):.4f}, mDice: {seg_scores.get('mdice', 0):.4f}")
                for k, v in seg_scores.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        writer.add_scalar(f'Metrics/Seg_Dataset/{display_name}/{k}', v, epoch)
                logger.info(f"    Overall Accuracy: {seg_scores.get('acc_overall', 0):.4f}, Mean IoU: {seg_scores.get('miou', 0):.4f}")
            _persist_segmentation_metrics(config.save_path, epoch, seg_metrics_payload, logger)

            if tb_seg_samples and logger is not None:
                logger.debug("[Validation] 分割样本已生成，按要求跳过 TensorBoard 记录。")

            if seg_visual_samples:
                _export_seg_visuals(seg_visual_samples, config.save_path, epoch + 1, logger)

        if task_type == 'depth':
            all_metrics['domains'] = domain_metrics
            all_metrics['datasets'] = dataset_metrics

        return all_metrics

    return None


def run_initial_evaluation(model: torch.nn.Module, val_depth_loader: DataLoader, val_seg_loader: DataLoader, config: TrainingConfig, writer: SummaryWriter,
                           logger: logging.Logger) -> None:
    """
                       运行初始评估
                       """
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        logger.info("Performing initial evaluation...")

    # 同步点，确保所有进程都准备好
    if dist.is_initialized():
        dist.barrier()

    logger.info("--- Initial Depth Validation ---")
    validate_and_visualize(model, val_depth_loader, 'depth', -1, config, writer, logger)

    if dist.is_initialized():
        dist.barrier()

    if val_seg_loader is not None:
        logger.info("--- Initial Segmentation Validation ---")
        validate_and_visualize(model, val_seg_loader, 'seg', -1, config, writer, logger)
    elif rank == 0:
        logger.info("Skip initial segmentation validation (no segmentation loader).")

    if rank == 0:
        logger.info("Initial evaluation finished.")


def run_epoch_validation(model: torch.nn.Module, val_depth_loader: DataLoader, val_seg_loader: DataLoader, epoch: int, config: TrainingConfig, writer: SummaryWriter,
                         logger: logging.Logger) -> tuple:
    """
   运行epoch验证（分布式安全）
   """
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 深度验证
    if rank == 0:
        logger.info("--- Depth Validation ---")
    if dist.is_initialized():
        dist.barrier()
    depth_metrics = validate_and_visualize(model, val_depth_loader, 'depth', epoch, config, writer, logger)

    # 分割验证
    if val_seg_loader is not None:
        if rank == 0:
            logger.info("--- Segmentation Validation ---")
        if dist.is_initialized():
            dist.barrier()
        seg_metrics = validate_and_visualize(model, val_seg_loader, 'seg', epoch, config, writer, logger)
    else:
        seg_metrics = {}
        if rank == 0:
            logger.info("Skip segmentation validation (no segmentation loader).")

    # 只有主进程返回指标
    if rank == 0:
        return depth_metrics, seg_metrics
    else:
        return None, None
