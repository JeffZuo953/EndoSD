#!/usr/bin/env python3
"""
多任务训练器模块
"""

import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from typing import Optional, Any, Dict, Tuple
from itertools import cycle
import math
from collections import Counter

from .config import TrainingConfig
from .loss import SiLogLoss
from .train_utils import clear_cuda_cache, log_training_progress, autocast
from .data_utils import summarize_loader_composition
from .base_trainer import BaseTrainer
from .loss_weighter import LossWeighter


class MultiTaskTrainer(BaseTrainer):
    """多任务训练器，支持独立或统一优化器"""

    _MASKED_DATASETS = {"scard", "scared", "dvpn", "stereomis", "endovis2017", "endovis2018"}

    def __init__(self,
                 model: torch.nn.Module,
                 config: TrainingConfig,
                 logger: logging.Logger,
                 writer: Any,
                 loss_weighter: LossWeighter,
                 rank: int,
                 optimizer_depth: Optional[AdamW] = None,
                 optimizer_seg: Optional[AdamW] = None,
                 optimizer_camera: Optional[AdamW] = None,
                 scheduler_depth: Optional[Any] = None,
                 scheduler_seg: Optional[Any] = None,
                 scheduler_camera: Optional[Any] = None,
                 optimizer_unified: Optional[AdamW] = None,
                 scheduler_unified: Optional[Any] = None):
        super().__init__(model, config, logger, writer)
        self.rank = rank
        # global step counter for per-step logging
        self.global_step = 0

        self.optimizer_depth = optimizer_depth
        self.optimizer_seg = optimizer_seg
        self.optimizer_camera = optimizer_camera
        self.scheduler_depth = scheduler_depth
        self.scheduler_seg = scheduler_seg
        self.scheduler_camera = scheduler_camera
        self.optimizer_unified = optimizer_unified
        self.scheduler_unified = scheduler_unified

        self.loss_weighter = loss_weighter
        min_depth_cfg = float(getattr(self.config, 'min_depth', 1e-6) or 1e-6)
        self._min_safe_depth = max(min_depth_cfg, 1e-5)
        self.depth_criterion = SiLogLoss(eps=self._min_safe_depth).cuda()
        self.camera_criterion = torch.nn.L1Loss(reduction='mean').cuda()
        # Keep reduction='mean' but handle all-ignore batches explicitly before calling
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()
        self.ignore_index = 255
        self._latest_camera_loss: float = 0.0
        self._camera_eps = 1e-6
        self._camera_loss_type = getattr(self.config, 'camera_loss_type', 'l1').lower()
        self._enable_nan_checks = bool(int(os.environ.get("DEPTHANYTHING_CHECK_NAN", "1")))
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        camera_head = getattr(actual_model, "camera_head", None)
        self._camera_head_params = [p for p in camera_head.parameters() if p.requires_grad] if camera_head is not None else []
        base_camera_weight = float(getattr(self.config, 'camera_loss_weight', 0.0))
        backbone_scale = float(getattr(self.config, 'camera_backbone_loss_scale', 1.0))
        head_scale = float(getattr(self.config, 'camera_head_loss_scale', backbone_scale))
        self._camera_backbone_weight = base_camera_weight * backbone_scale
        self._camera_head_weight = base_camera_weight * head_scale
        self._camera_grad_scale = 1.0
        if self._camera_backbone_weight > 0 and self._camera_head_weight > 0:
            self._camera_grad_scale = self._camera_head_weight / self._camera_backbone_weight
        first_param = next((p for p in self.model.parameters() if p is not None), None)
        self._primary_device = first_param.device if first_param is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._camera_debug_counts: Counter = Counter()
        self._camera_issue_log_limit = 5
        self._save_bad_batches = os.environ.get("FM_DEBUG_SAVE_BAD_BATCH", "0") == "1"
        self._save_bad_snapshots = os.environ.get("FM_DEBUG_SAVE_BAD_SNAPSHOT", "0") == "1"
        self._bad_batch_dir: Optional[Path] = None
        self._last_depth_batch: Optional[Dict[str, Any]] = None
        if self._save_bad_batches and self.rank == 0:
            try:
                self._bad_batch_dir = Path(self.config.save_path) / "debug_bad_batches"
                self._bad_batch_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.logger.warning("Failed to prepare debug batch directory: %s", exc)
                self._save_bad_batches = False
        self._bad_snapshot_dir: Optional[Path] = None
        if self._save_bad_snapshots and self.rank == 0:
            try:
                self._bad_snapshot_dir = Path(self.config.save_path) / "debug_bad_snapshots"
                self._bad_snapshot_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.logger.warning("Failed to prepare debug snapshot directory: %s", exc)
                self._save_bad_snapshots = False

    def _describe_tensor(self, tensor: Optional[torch.Tensor], label: str) -> str:
        if tensor is None:
            return f"{label}:None"
        try:
            data = tensor.detach()
        except Exception:
            return f"{label}:unavailable"
        finite_mask = torch.isfinite(data)
        if not finite_mask.any():
            return f"{label}:all_non_finite"
        data = data[finite_mask]
        return f"{label}:min={data.min().item():.6f},max={data.max().item():.6f},mean={data.mean().item():.6f}"

    def _log_nan_debug(self,
                       task: str,
                       batch: Dict[str, Any],
                       stage: str,
                       loss_value: Optional[torch.Tensor],
                       outputs: Optional[Dict[str, torch.Tensor]]) -> None:
        if not self._enable_nan_checks:
            return
        dataset = batch.get("dataset_name")
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0] if dataset else None
        source_type = batch.get("source_type")
        if isinstance(source_type, (list, tuple)):
            source_type = source_type[0] if source_type else None
        msg = [f"[NaN-{stage}] task={task}", f"dataset={dataset}", f"source={source_type}"]
        if loss_value is not None:
            msg.append(self._describe_tensor(loss_value, "loss"))
        depth_gt = batch.get("depth")
        if torch.is_tensor(depth_gt):
            msg.append(self._describe_tensor(depth_gt, "depth_gt"))
        if outputs:
            depth_pred = outputs.get("depth")
            if depth_pred is not None:
                msg.append(self._describe_tensor(depth_pred, "depth_pred"))
            seg_pred = outputs.get("seg")
            if seg_pred is not None:
                msg.append(self._describe_tensor(seg_pred, "seg_pred"))
            camera_pred = outputs.get("camera_intrinsics_norm")
            if camera_pred is not None:
                msg.append(self._describe_tensor(camera_pred, "camera_pred"))
        self.logger.error(" | ".join(filter(None, msg)))

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

    def _masked_dataset_flags(self,
                              dataset_entries: list,
                              source_entries: list,
                              batch_size: int,
                              device: torch.device) -> torch.Tensor:
        flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for idx in range(batch_size):
            entry = dataset_entries[idx]
            source = source_entries[idx]
            matched = False
            if entry is not None:
                matched = str(entry).lower() in self._MASKED_DATASETS
            if not matched and source is not None:
                matched = str(source).lower() in self._MASKED_DATASETS
            flags[idx] = matched
        return flags

    def _format_dataset_meta(self, name: Optional[Any], source: Optional[Any]) -> str:
        if name and source:
            return f"{name}({source})"
        return str(name or source or "unknown")

    def _extract_dataset_mask_tensor(self,
                                     batch: Dict[str, Any],
                                     device: torch.device,
                                     batch_size: int,
                                     dataset_entries: list,
                                     source_entries: list) -> Optional[torch.Tensor]:
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

        current_size = mask_tensor.shape[0]
        if current_size == batch_size:
            return mask_tensor

        if current_size == 1:
            return mask_tensor.expand(batch_size, -1, -1, -1).contiguous()

        if current_size < batch_size:
            if self.rank == 0:
            missing_indices = range(current_size, batch_size)
            missing_meta = ", ".join(
                self._format_dataset_meta(dataset_entries[i], source_entries[i]) for i in missing_indices
            )
            self.logger.warning(
                "Dataset mask smaller than batch (%s vs %s). Padding missing entries with all-True mask. Missing samples: %s",
                mask_tensor.shape, (batch_size, *mask_tensor.shape[1:]), missing_meta or "unknown"
            )
            pad_shape = (batch_size - current_size, *mask_tensor.shape[1:])
            pad = torch.ones(pad_shape, dtype=torch.bool, device=device)
            return torch.cat([mask_tensor, pad], dim=0)

        # current_size > batch_size : trim extra entries (shouldn't happen but keep safe)
        if self.rank == 0:
            extra_indices = range(batch_size, current_size)
            extra_meta = ", ".join(
                self._format_dataset_meta(dataset_entries[i], source_entries[i] if i < len(source_entries) else None)
                for i in extra_indices if i < len(dataset_entries)
            )
            self.logger.warning(
                "Dataset mask larger than batch (%s vs %s). Truncating extra entries. Extra samples: %s",
                mask_tensor.shape, (batch_size, *mask_tensor.shape[1:]), extra_meta or "unknown"
            )
        return mask_tensor[:batch_size]

    def _apply_dataset_masks(self, batch: Dict[str, Any], mask_depth: torch.Tensor) -> torch.Tensor:
        batch_size = mask_depth.shape[0]
        dataset_entries = self._normalize_meta_list(batch.get("dataset_name"), batch_size)
        source_entries = self._normalize_meta_list(batch.get("source_type"), batch_size)
        selector = self._masked_dataset_flags(dataset_entries, source_entries, batch_size, mask_depth.device)
        if not bool(selector.any()):
            return mask_depth

        dataset_mask = self._extract_dataset_mask_tensor(
            batch, mask_depth.device, batch_size, dataset_entries, source_entries
        )
        if dataset_mask is None:
            if self.rank == 0:
                dataset_meta = batch.get("dataset_name") or batch.get("source_type")
                self.logger.warning("Requested dataset mask but none found for batch meta: %s", dataset_meta)
            return mask_depth

        if dataset_mask.shape[0] != batch_size:
            if self.rank == 0:
                self.logger.warning("Dataset mask batch mismatch: mask=%s, batch=%s", dataset_mask.shape, mask_depth.shape)
            return mask_depth

        updated_mask = mask_depth.clone()
        updated_mask[selector] = updated_mask[selector] & dataset_mask[selector]
        return updated_mask

    def _rescale_camera_head_grads(self, camera_loss_applied: bool) -> None:
        if not camera_loss_applied:
            return
        if not self._camera_head_params:
            return
        scale = getattr(self, "_camera_grad_scale", 1.0)
        if scale <= 0 or not math.isfinite(scale):
            return
        if math.isclose(scale, 1.0, rel_tol=1e-5, abs_tol=1e-7):
            return
        for param in self._camera_head_params:
            if param.grad is not None:
                param.grad.mul_(scale)

    @staticmethod
    def _select_primary_dataset(names: Optional[Any]) -> Optional[str]:
        if names is None:
            return None
        if isinstance(names, (list, tuple)):
            if not names:
                return None
            counter = Counter(names)
            primary, _ = counter.most_common(1)[0]
            return primary
        return names

    def _build_progress_tracker(self, data_loader: DataLoader) -> Optional[Dict[str, Any]]:
        if data_loader is None:
            return None
        summary = summarize_loader_composition(data_loader)
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

    def _log_dataset_progress(self,
                              phase: str,
                              task: str,
                              epoch: int,
                              names: Optional[Any],
                              tracker: Optional[Dict[str, Any]]) -> None:
        if self.rank != 0 or tracker is None or names is None:
            return
        if isinstance(names, torch.Tensor):
            names = names.tolist()
        if isinstance(names, str):
            names_list = [names]
        else:
            names_list = list(names)
        if not names_list:
            return
        primary = self._select_primary_dataset(names_list)
        if primary is None:
            return
        increment = sum(1 for n in names_list if n == primary)
        if increment <= 0:
            increment = len(names_list)
        tracker['counts'][primary] = tracker['counts'].get(primary, 0) + increment
        processed = tracker['counts'][primary]
        total = tracker['totals'].get(primary, "unknown")
        dataset_type = tracker['types'].get(primary, "unknown")
        if dataset_type == "unknown":
            for key, value in tracker['types'].items():
                if not isinstance(key, str):
                    continue
                base = key.split('[', 1)[0]
                if base == primary or primary in key:
                    dataset_type = value
                    break
        if primary not in tracker['started']:
            self.logger.info(f"[{phase}][{task}][Epoch {epoch}] Started dataset {primary} ({dataset_type}) 0/{total}")
            tracker['started'].add(primary)
        if isinstance(total, int) and total > 0:
            progress = processed / total
            if progress >= 1.0 and primary not in tracker['completed']:
                self.logger.info(f"[{phase}][{task}][Epoch {epoch}] Finished dataset {primary} ({dataset_type}) ({processed}/{total})")
                tracker['completed'].add(primary)
            else:
                milestone = math.floor(progress * 10) / 10.0
                if milestone >= 0.1 and milestone < 1.0:
                    seen = tracker['milestones'].setdefault(primary, set())
                    if milestone not in seen:
                        seen.add(milestone)
                        percent = int(milestone * 100)
                        self.logger.info(f"[{phase}][{task}][Epoch {epoch}] {primary} ({dataset_type}) progress {percent}% ({processed}/{total})")

    def _to_float_tensor(self, value: Any, device: torch.device) -> torch.Tensor:
        """Convert arbitrary values (lists, numpy arrays, tensors) to float32 tensors on the desired device."""
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        return tensor.to(device=device, dtype=torch.float32, non_blocking=True)

    def _to_bool_tensor(self, value: Any, device: torch.device) -> torch.Tensor:
        """Convert arbitrary values into boolean tensors on the desired device."""
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        return tensor.to(device=device, dtype=torch.bool, non_blocking=True)

    def _align_camera_mask(self, tensor: torch.Tensor, batch_size: int, name: str) -> Optional[torch.Tensor]:
        """Ensure boolean masks broadcast to [batch] shape."""
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2 and tensor.size(1) == 1:
            tensor = tensor.squeeze(1)
        elif tensor.dim() != 1:
            self.logger.debug(f"[Camera] {name} mask has incompatible shape {tuple(tensor.shape)}.")
            return None
        rows = tensor.size(0)
        if rows == batch_size:
            return tensor.contiguous()
        if rows == 1 and batch_size > 1:
            return tensor.expand(batch_size).contiguous()
        self.logger.debug(f"[Camera] {name} mask rows={rows} mismatch batch={batch_size}; skipping.")
        return None

    def _extract_camera_mask(
        self,
        batch: Dict[str, Any],
        key: str,
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        value = batch.get(key)
        if value is None:
            return None
        mask_tensor = self._to_bool_tensor(value, device)
        return self._align_camera_mask(mask_tensor, batch_size, key)

    @staticmethod
    def _combine_masks(primary: Optional[torch.Tensor], secondary: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if primary is None:
            return secondary
        if secondary is None:
            return primary
        if primary.size(0) != secondary.size(0):
            return None
        combined = primary & secondary
        if not combined.any():
            return None
        return combined

    def _filter_tensor_with_mask(
        self,
        tensor: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        name: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tensor is None:
            return None, None
        if mask is None:
            return tensor, None
        if tensor.size(0) != mask.size(0):
            self.logger.debug(f"[Camera] {name} mask size mismatch ({mask.size(0)} vs {tensor.size(0)}).")
            return None, None
        if not mask.any():
            self.logger.debug(f"[Camera] {name} mask contains no valid entries.")
            return None, None
        return tensor[mask], mask

    def _filter_prediction_with_mask(
        self,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor],
        name: str,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return pred
        if pred.size(0) != mask.size(0):
            self.logger.debug(f"[Camera] Prediction mask mismatch for {name}.")
            return None
        if not mask.any():
            return None
        return pred[mask]

    def _filter_camera_size_subset(
        self,
        camera_size: Optional[torch.Tensor],
        size_mask: Optional[torch.Tensor],
        selection_mask: Optional[torch.Tensor],
        require_all: bool = False,
    ) -> Optional[torch.Tensor]:
        if camera_size is None:
            return None
        size_tensor = camera_size if selection_mask is None else camera_size[selection_mask]
        if size_mask is None:
            return size_tensor
        if selection_mask is not None:
            size_mask = size_mask[selection_mask]
        if not size_mask.any():
            return None
        if not size_mask.all():
            return None
        return size_tensor

    def _align_camera_tensor(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        feature_dim: int,
        name: str,
    ) -> Optional[torch.Tensor]:
        """
        Ensure tensors used by the camera head have a [batch, feature_dim] shape (or broadcastable from batch=1).
        """
        if tensor.dim() == 1:
            if tensor.numel() < feature_dim:
                self.logger.debug(f"[Camera] {name} expects >= {feature_dim} values, got {tensor.numel()}.")
                return None
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2 or tensor.size(1) < feature_dim:
            self.logger.debug(f"[Camera] {name} shape {tuple(tensor.shape)} incompatible with feature_dim={feature_dim}.")
            return None
        tensor = tensor[:, :feature_dim]
        rows = tensor.size(0)
        if rows == batch_size:
            return tensor.contiguous()
        if rows == 1 and batch_size > 1:
            return tensor.expand(batch_size, feature_dim).contiguous()
        self.logger.debug(
            f"[Camera] {name} batch mismatch (rows={rows}, batch={batch_size}); skipping camera supervision this step."
        )
        return None

    def _extract_camera_size_tensor(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        size_keys = ("camera_size", "camera_image_size", "camera_size_original", "camera_original_image_size")
        for key in size_keys:
            value = batch.get(key)
            if value is None:
                continue
            size_tensor = self._to_float_tensor(value, device)
            aligned = self._align_camera_tensor(size_tensor, batch_size, 2, key)
            if aligned is not None:
                return aligned
        return None

    def _prepare_camera_targets(
        self,
        batch: Dict[str, Any],
        camera_pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Produce aligned prediction, normalized GT intrinsics, and optional camera-size tensors.
        """
        batch_size = camera_pred.size(0)
        device = camera_pred.device
        camera_size = self._extract_camera_size_tensor(batch, device, batch_size)
        camera_size_mask = self._extract_camera_mask(batch, 'camera_size_mask', batch_size, device)

        gt_norm_value = batch.get('camera_intrinsics_norm')
        if gt_norm_value is not None:
            gt_norm_tensor = self._align_camera_tensor(
                self._to_float_tensor(gt_norm_value, device),
                batch_size,
                4,
                'camera_intrinsics_norm',
            )
            if gt_norm_tensor is not None:
                norm_mask = self._extract_camera_mask(batch, 'camera_intrinsics_norm_mask', batch_size, device)
                gt_norm_tensor, selection_mask = self._filter_tensor_with_mask(
                    gt_norm_tensor, norm_mask, 'camera_intrinsics_norm'
                )
                if gt_norm_tensor is not None:
                    pred_tensor = self._filter_prediction_with_mask(camera_pred, selection_mask, 'camera_intrinsics_norm')
                    if pred_tensor is not None:
                        size_tensor = self._filter_camera_size_subset(camera_size, camera_size_mask, selection_mask)
                        return pred_tensor, gt_norm_tensor.contiguous(), size_tensor

        raw_value = batch.get('camera_intrinsics')
        if raw_value is not None:
            raw_tensor = self._to_float_tensor(raw_value, device)
            raw_mask = self._extract_camera_mask(batch, 'camera_intrinsics_mask', batch_size, device)
            combined_mask = self._combine_masks(raw_mask, camera_size_mask)
            raw_tensor, selection_mask = self._filter_tensor_with_mask(raw_tensor, combined_mask, 'camera_intrinsics')
            if raw_tensor is None:
                return None

            if raw_tensor.dim() == 3 and raw_tensor.size(-1) == 3 and raw_tensor.size(-2) == 3:
                if camera_size is None:
                    self.logger.debug("[Camera] Missing camera_size, cannot convert raw intrinsics to normalized form.")
                    return None
                size_tensor = self._filter_camera_size_subset(camera_size, camera_size_mask, selection_mask, require_all=True)
                if size_tensor is None:
                    self.logger.debug("[Camera] Missing valid camera_size rows for raw intrinsics.")
                    return None
                width = torch.clamp(size_tensor[:, 0:1], min=1e-6)
                height = torch.clamp(size_tensor[:, 1:2], min=1e-6)
                fx_norm = raw_tensor[:, 0, 0:1] / width
                fy_norm = raw_tensor[:, 1, 1:2] / height
                cx_norm = raw_tensor[:, 0, 2:3] / width
                cy_norm = raw_tensor[:, 1, 2:3] / height
                gt_tensor = torch.cat([fx_norm, fy_norm, cx_norm, cy_norm], dim=1)
                pred_tensor = self._filter_prediction_with_mask(camera_pred, selection_mask, 'camera_intrinsics')
                if pred_tensor is not None:
                    return pred_tensor, gt_tensor.contiguous(), size_tensor
            else:
                gt_tensor = self._align_camera_tensor(raw_tensor, batch_size, 4, 'camera_intrinsics')
                if gt_tensor is not None:
                    gt_tensor, selection_mask = self._filter_tensor_with_mask(gt_tensor, raw_mask, 'camera_intrinsics')
                    if gt_tensor is not None:
                        pred_tensor = self._filter_prediction_with_mask(camera_pred, selection_mask, 'camera_intrinsics')
                        if pred_tensor is not None:
                            size_tensor = self._filter_camera_size_subset(camera_size, camera_size_mask, selection_mask)
                            return pred_tensor, gt_tensor.contiguous(), size_tensor

        return None

    def train_epoch(self, train_depth_loader: DataLoader, train_seg_loader: DataLoader, epoch: int) -> tuple:
        from .train_utils import set_epoch_for_samplers
        set_epoch_for_samplers(train_depth_loader, train_seg_loader, epoch)
        self.logger.info(f"===========> Epoch: {epoch}/{self.config.epochs}")
        self.model.train()

        depth_tracker = self._build_progress_tracker(train_depth_loader) if train_depth_loader is not None else None
        seg_tracker = self._build_progress_tracker(train_seg_loader) if train_seg_loader is not None else None

        if self.optimizer_unified:
            avg_depth_loss, avg_seg_loss, avg_camera_loss = self._train_epoch_unified(
                train_depth_loader, train_seg_loader, epoch, depth_tracker, seg_tracker
            )
        else:
            avg_depth_loss, avg_camera_loss = self._train_task_epoch(train_depth_loader, 'depth', epoch, depth_tracker)
            if train_seg_loader is not None:
                avg_seg_loss, _ = self._train_task_epoch(train_seg_loader, 'seg', epoch, seg_tracker)
            else:
                avg_seg_loss = 0.0

        self._latest_camera_loss = avg_camera_loss

        if self.rank == 0:
            if self.writer is not None:
                self.writer.add_scalar("train/depth_loss", avg_depth_loss, epoch)
                if not getattr(self.config, "disable_seg_head", False):
                    self.writer.add_scalar("train/seg_loss", avg_seg_loss, epoch)
                if self.config.camera_head_mode.lower() != 'none':
                    self.writer.add_scalar("train/camera_loss", avg_camera_loss, epoch)

                if getattr(self.loss_weighter, "log_vars", None) is not None:
                    bounds = getattr(self.loss_weighter, 'log_var_bounds', (-float('inf'), float('inf')))
                    clamp_enabled = all(map(math.isfinite, bounds))

                    raw_log_var_depth = self.loss_weighter.log_vars[0].item()
                    raw_log_var_seg = self.loss_weighter.log_vars[1].item()
                    log_var_depth = max(bounds[0], min(bounds[1], raw_log_var_depth)) if clamp_enabled else raw_log_var_depth
                    log_var_seg = max(bounds[0], min(bounds[1], raw_log_var_seg)) if clamp_enabled else raw_log_var_seg

                    self.writer.add_scalar("train/uwl_log_var_depth", log_var_depth, epoch)
                    self.writer.add_scalar("train/uwl_log_var_seg", log_var_seg, epoch)

                    if clamp_enabled:
                        weight_factor_depth = math.exp(-log_var_depth)
                        weight_factor_seg = math.exp(-log_var_seg)
                    else:
                        weight_factor_depth = torch.exp(-self.loss_weighter.log_vars[0]).item()
                        weight_factor_seg = torch.exp(-self.loss_weighter.log_vars[1]).item()

                    self.writer.add_scalar("train/weight_depth", weight_factor_depth, epoch)
                    if not getattr(self.config, "disable_seg_head", False):
                        self.writer.add_scalar("train/weight_seg", weight_factor_seg, epoch)

            if self.config.camera_head_mode.lower() != 'none':
                msg = f"Epoch {epoch} - Avg Depth Loss: {avg_depth_loss:.4f}"
                if not getattr(self.config, "disable_seg_head", False):
                    msg += f", Avg Seg Loss: {avg_seg_loss:.4f}"
                msg += f", Avg Camera Loss: {avg_camera_loss:.4f}"
                self.logger.info(msg)
            else:
                if not getattr(self.config, "disable_seg_head", False):
                    self.logger.info(f"Epoch {epoch} - Avg Depth Loss: {avg_depth_loss:.4f}, Avg Seg Loss: {avg_seg_loss:.4f}")
                else:
                    self.logger.info(f"Epoch {epoch} - Avg Depth Loss: {avg_depth_loss:.4f}")

        return avg_depth_loss, avg_seg_loss

    def _record_camera_issue(self, reason: str, batch: Optional[Dict[str, Any]], extra: Optional[str] = None) -> None:
        self._camera_debug_counts[reason] += 1
        if self.rank != 0 or self._camera_debug_counts[reason] > self._camera_issue_log_limit:
            return
        dataset = None
        if batch is not None:
            dataset = batch.get("dataset_name")
            if isinstance(dataset, (list, tuple)) and dataset:
                dataset = dataset[0]
        msg = f"[CameraDebug] reason={reason}"
        if dataset:
            msg += f", dataset={dataset}"
        if extra:
            msg += f", detail={extra}"
        self.logger.warning(msg)

    def _train_epoch_unified(self,
                             train_depth_loader: DataLoader,
                             train_seg_loader: Optional[DataLoader],
                             epoch: int,
                             depth_tracker: Optional[Dict[str, Any]],
                             seg_tracker: Optional[Dict[str, Any]]) -> tuple:
        if train_depth_loader is None:
            raise ValueError("Unified optimizer requires a depth dataloader.")

        seg_enabled = train_seg_loader is not None and not getattr(self.config, "disable_seg_head", False)

        total_depth_loss, total_seg_loss = 0.0, 0.0
        total_camera_loss = 0.0
        depth_batches, seg_batches, camera_batches = 0, 0, 0

        def _step_optimizer(i: int) -> bool:
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.unscale_(self.optimizer_unified)

            max_norm = getattr(self.config, 'clip_grad_norm', 1.0)
            if max_norm and max_norm > 0:
                params_to_clip = [p for p in self.model.parameters() if p.requires_grad]
                if getattr(self.loss_weighter, "log_vars", None) is not None:
                    params_to_clip.append(self.loss_weighter.log_vars)
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=max_norm)

            named_params = list(self.model.named_parameters())
            if getattr(self.loss_weighter, "log_vars", None) is not None:
                named_params.append(("loss_weighter.log_vars", self.loss_weighter.log_vars))
            bad_params = self._sanitize_gradients(named_params)
            local_skip = bool(bad_params)
            if local_skip and any("camera_head" in name for name in bad_params):
                snippet = ", ".join(bad_params[:3])
                self._record_camera_issue("camera_bad_grad", None, extra=snippet)
            if local_skip and self.rank == 0:
                self.logger.warning(
                    f"Skipping optimizer step at epoch {epoch}, iter {i} due to non-finite gradients: {bad_params}"
                )

            skip_step = local_skip
            if dist.is_initialized():
                skip_tensor = torch.tensor(1 if skip_step else 0, device=self._primary_device)
                dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                skip_step = bool(skip_tensor.item())
            if skip_step and not local_skip:
                self._record_camera_issue("external_bad_grad", None)

            if skip_step:
                if self.config.mixed_precision and self.scaler is not None:
                    self.scaler.update()
                self.optimizer_unified.zero_grad(set_to_none=True)
                if getattr(self.loss_weighter, "log_vars", None) is not None and self.loss_weighter.log_vars.grad is not None:
                    self.loss_weighter.log_vars.grad.zero_()
                if self._save_bad_batches:
                    self._dump_bad_batch(epoch, i, bad_params)
                if self._save_bad_snapshots:
                    self._dump_bad_snapshot(epoch, i)
                return False

            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.step(self.optimizer_unified)
                self.loss_weighter.sanitize_parameters()
                self.scaler.update()
            else:
                self.optimizer_unified.step()
                self.loss_weighter.sanitize_parameters()
            return True

        if seg_enabled:
            len_depth = len(train_depth_loader)
            len_seg = len(train_seg_loader)
            if len_depth > len_seg:
                long_loader, short_loader = train_depth_loader, train_seg_loader
                is_depth_long = True
            else:
                long_loader, short_loader = train_seg_loader, train_depth_loader
                is_depth_long = False

            short_loader_iter = iter(cycle(short_loader))

            for i, long_batch in enumerate(long_loader):
                batch_depth, batch_seg = (long_batch, next(short_loader_iter)) if is_depth_long else (next(short_loader_iter), long_batch)

                self.optimizer_unified.zero_grad()
                depth_stats = self._run_depth_step(batch_depth, depth_tracker, epoch)
                loss_depth, weighted_loss_depth, camera_loss, camera_loss_applied = depth_stats

                self._log_dataset_progress("Train", "Seg", epoch, batch_seg.get("dataset_name"), seg_tracker)
                with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                    img_seg = batch_seg["image"].cuda()
                    gt_seg = batch_seg["semseg_mask"].cuda().long()
                    pred_seg = self.model(img_seg, task='seg')['seg']
                    pred_seg = torch.nan_to_num(pred_seg, nan=0.0).clamp_(-100.0, 100.0)
                    valid_mask = (gt_seg >= 0) & (gt_seg < self.config.num_classes)
                    if int(valid_mask.sum().item()) == 0:
                        loss_seg = (pred_seg.sum() * 0.0).to(pred_seg.dtype)
                    else:
                        gt_seg = torch.where(valid_mask, gt_seg, torch.tensor(self.ignore_index, device=gt_seg.device))
                        loss_seg = self.seg_criterion(pred_seg, gt_seg)
                    weighted_loss_seg = self.loss_weighter.get_loss(
                        torch.tensor(0.0, device=loss_seg.device), loss_seg, task='seg')

                self._backward_depth(weighted_loss_depth, camera_loss_applied)
                if self.config.mixed_precision and self.scaler is not None:
                    self.scaler.scale(weighted_loss_seg).backward()
                else:
                    weighted_loss_seg.backward()

                total_depth_loss += loss_depth.item()
                total_seg_loss += loss_seg.item()
                depth_batches += 1
                seg_batches += 1
                if camera_loss is not None:
                    total_camera_loss += camera_loss.item()
                    camera_batches += 1

                if not _step_optimizer(i):
                    continue

                self._log_step_uwl()
                self.global_step += 1
        else:
            for i, batch_depth in enumerate(train_depth_loader):
                self.optimizer_unified.zero_grad()
                depth_stats = self._run_depth_step(batch_depth, depth_tracker, epoch)
                loss_depth, weighted_loss_depth, camera_loss, camera_loss_applied = depth_stats
                self._backward_depth(weighted_loss_depth, camera_loss_applied)

                total_depth_loss += loss_depth.item()
                depth_batches += 1
                if camera_loss is not None:
                    total_camera_loss += camera_loss.item()
                    camera_batches += 1

                if not _step_optimizer(i):
                    continue

                self._log_step_uwl()
                self.global_step += 1

        avg_depth_loss = total_depth_loss / depth_batches if depth_batches > 0 else 0.0
        avg_seg_loss = total_seg_loss / seg_batches if seg_batches > 0 else 0.0
        avg_camera_loss = total_camera_loss / camera_batches if camera_batches > 0 else 0.0
        return avg_depth_loss, avg_seg_loss, avg_camera_loss

    def _run_depth_step(self,
                        batch_depth: Dict[str, Any],
                        depth_tracker: Optional[Dict[str, Any]],
                        epoch: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool]:
        self._log_dataset_progress("Train", "Depth", epoch, batch_depth.get("dataset_name"), depth_tracker)
        self._last_depth_batch = batch_depth
        camera_loss = None
        camera_loss_applied = False
        min_depth_cfg = float(getattr(self.config, 'min_depth', 1e-6) or 1e-6)
        min_safe_depth = getattr(self, '_min_safe_depth', max(min_depth_cfg, 1e-5))
        max_depth_cfg = float(getattr(self.config, 'max_depth', 1.0) or 1.0)
        with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
            img_depth = batch_depth["image"].cuda()
            torch.nan_to_num_(img_depth, nan=0.0, posinf=0.0, neginf=0.0)
            if img_depth.dtype.is_floating_point:
                img_depth = img_depth.clamp_(-10.0, 10.0)
            raw_gt_depth = batch_depth["depth"].cuda()
            gt_depth = torch.nan_to_num(raw_gt_depth, nan=0.0, posinf=max_depth_cfg, neginf=0.0)
            gt_depth = gt_depth.clamp_(min=min_safe_depth, max=max_depth_cfg)
            outputs_depth = self.model(img_depth, task='depth')
            pred_depth = outputs_depth['depth']
            pred_depth = torch.nan_to_num(pred_depth, nan=0.0, posinf=max_depth_cfg * 1.25, neginf=0.0)
            pred_depth = pred_depth.clamp_(min=min_safe_depth, max=max_depth_cfg * 1.25)
            mask_depth = (gt_depth >= min_safe_depth) & (gt_depth <= max_depth_cfg) & torch.isfinite(gt_depth)
            mask_depth = self._apply_dataset_masks(batch_depth, mask_depth)
            valid_depth = int(mask_depth.sum().item())
            if valid_depth == 0:
                loss_depth = (pred_depth.sum() * 0.0).to(pred_depth.dtype)
            else:
                loss_depth = self.depth_criterion(pred_depth, gt_depth, mask_depth)

            if self.config.camera_head_mode.lower() != 'none' and 'camera_intrinsics_norm' in outputs_depth:
                camera_pred = outputs_depth['camera_intrinsics_norm']
                camera_pred = torch.nan_to_num(camera_pred, nan=0.0, posinf=1.0, neginf=0.0)
                camera_batch = self._prepare_camera_targets(batch_depth, camera_pred)
                if camera_batch is None:
                    self._record_camera_issue("missing_targets", batch_depth)
                else:
                    pred_aligned, gt_camera, camera_size = camera_batch
                    camera_valid = True
                    pred_aligned = torch.nan_to_num(pred_aligned, nan=0.0, posinf=1.0, neginf=0.0)
                    gt_camera = torch.nan_to_num(gt_camera, nan=0.0, posinf=1.0, neginf=0.0)
                    if not torch.isfinite(pred_aligned).all() or not torch.isfinite(gt_camera).all():
                        camera_valid = False
                        self._record_camera_issue("non_finite_tensor", batch_depth)
                    if camera_size is not None and not torch.isfinite(camera_size).all():
                        camera_valid = False
                        self._record_camera_issue("non_finite_size", batch_depth)
                    if camera_valid:
                        camera_loss = self._compute_camera_loss(pred_aligned, gt_camera, camera_size)
                        if not torch.isfinite(camera_loss):
                            self._record_camera_issue("non_finite_loss", batch_depth)
                            camera_loss = torch.tensor(0.0, device=pred_aligned.device)
                        elif self._camera_backbone_weight > 0.0:
                            loss_depth = loss_depth + self._camera_backbone_weight * camera_loss
                            camera_loss_applied = True

        with autocast(enabled=False, **self.autocast_kwargs):
            weighted_loss_depth = self.loss_weighter.get_loss(
                loss_depth, torch.tensor(0.0, device=loss_depth.device), task='depth'
            )

        return loss_depth, weighted_loss_depth, camera_loss, camera_loss_applied

    def _backward_depth(self, weighted_loss_depth: torch.Tensor, camera_loss_applied: bool) -> None:
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(weighted_loss_depth).backward()
            self._rescale_camera_head_grads(camera_loss_applied)
        else:
            weighted_loss_depth.backward()
            self._rescale_camera_head_grads(camera_loss_applied)

    def _sanitize_gradients(self, named_params):
        bad = []
        for name, param in named_params:
            if param is None or not param.requires_grad or param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                bad.append(name)
        return bad

    def _snapshot_depth_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        tensor_keys = (
            "image",
            "depth",
            "valid_mask",
            "depth_valid_mask",
            "semseg_mask",
            "camera_intrinsics",
            "camera_intrinsics_norm",
            "camera_size",
        )
        meta_keys = (
            "dataset_name",
            "source_type",
            "image_path",
            "depth_path",
            "camera_path",
            "max_depth",
        )
        for key in tensor_keys:
            value = batch.get(key)
            if torch.is_tensor(value):
                try:
                    snapshot[key] = value.detach().cpu()
                except Exception:
                    pass
        for key in meta_keys:
            if key in batch:
                snapshot[key] = batch[key]
        return snapshot

    def _dump_bad_batch(self, epoch: int, iteration: int, bad_params: Optional[Any]) -> None:
        if not self._save_bad_batches or self.rank != 0 or self._bad_batch_dir is None:
            return
        batch = getattr(self, "_last_depth_batch", None)
        if not isinstance(batch, dict):
            return
        snapshot = self._snapshot_depth_batch(batch)
        snapshot["_meta"] = {
            "epoch": epoch,
            "iteration": iteration,
            "bad_params": bad_params,
        }
        filename = f"bad_batch_e{epoch:03d}_i{iteration:05d}.pt"
        try:
            torch.save(snapshot, self._bad_batch_dir / filename)
            self.logger.warning("Saved debug batch to %s", self._bad_batch_dir / filename)
        except Exception as exc:
            self.logger.warning("Failed to save debug batch: %s", exc)

    def _dump_bad_snapshot(self, epoch: int, iteration: int) -> None:
        if not self._save_bad_snapshots or self.rank != 0 or self._bad_snapshot_dir is None:
            return
        filename = f"bad_snapshot_e{epoch:03d}_i{iteration:05d}.pth"
        try:
            state_dict = self.model.state_dict()
            torch.save({"state_dict": state_dict, "epoch": epoch, "iteration": iteration}, self._bad_snapshot_dir / filename)
            self.logger.warning("Saved debug snapshot to %s", self._bad_snapshot_dir / filename)
        except Exception as exc:
            self.logger.warning("Failed to save debug snapshot: %s", exc)

    def _compute_camera_loss(self, camera_pred: torch.Tensor, camera_gt: torch.Tensor, camera_size: Optional[torch.Tensor]) -> torch.Tensor:
        if camera_size is not None:
            if camera_size.dim() == 1:
                camera_size = camera_size.unsqueeze(0)
            width = camera_size[:, 0:1].to(camera_pred.device)
            height = camera_size[:, 1:2].to(camera_pred.device)
        else:
            width = None
            height = None

        if width is not None and height is not None:
            pred_fx = camera_pred[:, 0:1] * width
            pred_fy = camera_pred[:, 1:2] * height
            pred_cx = camera_pred[:, 2:3] * width
            pred_cy = camera_pred[:, 3:4] * height

            gt_fx = camera_gt[:, 0:1] * width
            gt_fy = camera_gt[:, 1:2] * height
            gt_cx = camera_gt[:, 2:3] * width
            gt_cy = camera_gt[:, 3:4] * height
        else:
            pred_fx, pred_fy, pred_cx, pred_cy = camera_pred.split(1, dim=1)
            gt_fx, gt_fy, gt_cx, gt_cy = camera_gt.split(1, dim=1)

        loss_type = getattr(self, '_camera_loss_type', 'l1')
        rel_fx = self._camera_component_loss(pred_fx, gt_fx, loss_type)
        rel_fy = self._camera_component_loss(pred_fy, gt_fy, loss_type)
        rel_cx = self._camera_component_loss(pred_cx, gt_cx, loss_type)
        rel_cy = self._camera_component_loss(pred_cy, gt_cy, loss_type)

        return (rel_fx + rel_fy + rel_cx + rel_cy).mean()

    def _camera_component_loss(self, pred: torch.Tensor, gt: torch.Tensor, loss_type: str) -> torch.Tensor:
        eps = self._camera_eps
        if loss_type == 'l2':
            denom = torch.square(gt).clamp_min(eps)
            return torch.square(pred - gt) / denom
        return torch.abs(pred - gt) / (torch.abs(gt) + eps)

    def _log_step_uwl(self) -> None:
        """Log UWL variables and weights per step under a separate category.
        Uses a single add_scalars call to minimize overhead. Only rank 0 logs.
        """
        if self.rank != 0 or self.writer is None:
            return
        if getattr(self.loss_weighter, "log_vars", None) is None:
            return

        try:
            # Fetch both log_vars in one CPU sync to reduce overhead
            log_vars = self.loss_weighter.log_vars.detach().cpu().tolist()
            if not isinstance(log_vars, list) or len(log_vars) < 2:
                return

            bounds = getattr(self.loss_weighter, 'log_var_bounds', (-float('inf'), float('inf')))
            clamp_enabled = all(map(math.isfinite, bounds))

            log_var_depth = float(log_vars[0])
            log_var_seg = float(log_vars[1])
            if clamp_enabled:
                log_var_depth = max(bounds[0], min(bounds[1], log_var_depth))
                log_var_seg = max(bounds[0], min(bounds[1], log_var_seg))

            try:
                weight_depth = math.exp(-log_var_depth)
            except OverflowError:
                weight_depth = float('inf')
            try:
                weight_seg = math.exp(-log_var_seg)
            except OverflowError:
                weight_seg = float('inf')

            prefix = "train_step/uwl"
            self.writer.add_scalar(f"{prefix}/log_var_depth", log_var_depth, self.global_step)
            self.writer.add_scalar(f"{prefix}/log_var_seg", log_var_seg, self.global_step)
            self.writer.add_scalar(f"{prefix}/weight_depth", weight_depth, self.global_step)
            self.writer.add_scalar(f"{prefix}/weight_seg", weight_seg, self.global_step)

        except Exception as e:
            # Avoid training interruption due to logging issues
            self.logger.debug(f"Step UWL logging skipped due to error: {e}")

    def _train_task_epoch(self, data_loader: DataLoader, task: str, epoch: int, tracker: Optional[Dict[str, Any]]) -> tuple:
        if data_loader is None:
            self.logger.info(f"--- Skipping {task.capitalize()} Head (no loader) ---")
            return 0.0, 0.0

        optimizer = self.optimizer_depth if task == 'depth' else self.optimizer_seg
        extra_optimizer = self.optimizer_camera if task == 'depth' else None
        criterion = self.depth_criterion if task == 'depth' else self.seg_criterion

        self.logger.info(f"--- Training {task.capitalize()} Head ---")
        total_loss = 0.0
        total_camera_loss = 0.0
        camera_batches = 0
        task_label = "Depth" if task == 'depth' else "Seg"

        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            if extra_optimizer is not None:
                extra_optimizer.zero_grad()
            input_img = batch["image"].cuda()

            self._log_dataset_progress("Train", task_label, epoch, batch.get("dataset_name"), tracker)

            camera_loss = None
            camera_loss_applied = False
            with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                outputs = None
                if task == 'depth':
                    depth_gt = batch["depth"].cuda()
                    outputs = self.model(input_img, task='depth')
                    depth_pred = outputs['depth']

                    # Sanitize depth predictions to avoid propagating NaNs/Infs into the loss.
                    min_depth_cfg = float(getattr(self.config, 'min_depth', 0.0) or 0.0)
                    max_depth_cfg = getattr(self.config, 'max_depth', None)
                    nan_to_num_kwargs = {'nan': 0.0}
                    if max_depth_cfg is not None:
                        max_depth_val = float(max_depth_cfg)
                        if math.isfinite(max_depth_val):
                            nan_to_num_kwargs['posinf'] = max_depth_val
                    depth_pred = torch.nan_to_num(depth_pred, **nan_to_num_kwargs)
                    if math.isfinite(min_depth_cfg):
                        depth_pred = depth_pred.clamp_min_(min_depth_cfg)
                    else:
                        depth_pred = depth_pred.clamp_min_(0.0)
                    if max_depth_cfg is not None:
                        max_depth_val = float(max_depth_cfg)
                        if math.isfinite(max_depth_val):
                            depth_pred = depth_pred.clamp_max_(max_depth_val * 1.25)

                    valid_mask = (depth_gt > 0) & (depth_gt >= self.config.min_depth) & (depth_gt <= self.config.max_depth)
                    loss = criterion(depth_pred, depth_gt, valid_mask)

                    if self.config.camera_head_mode.lower() != 'none' and 'camera_intrinsics_norm' in outputs:
                        camera_pred = outputs['camera_intrinsics_norm']
                        camera_batch = self._prepare_camera_targets(batch, camera_pred)
                        if camera_batch is not None:
                            pred_aligned, gt_camera, camera_size = camera_batch
                            camera_loss = self._compute_camera_loss(pred_aligned, gt_camera, camera_size)
                            if self._camera_backbone_weight > 0.0:
                                loss = loss + self._camera_backbone_weight * camera_loss
                                camera_loss_applied = True
                else:
                    seg_gt = batch["semseg_mask"].cuda().long()
                    outputs = self.model(input_img, task='seg')
                    seg_pred = outputs['seg']
                    seg_pred = torch.nan_to_num(seg_pred, nan=0.0)
                    seg_pred = seg_pred.clamp_(-100.0, 100.0)
                    valid_mask = (seg_gt >= 0) & (seg_gt < self.config.num_classes)
                    valid_count = int(valid_mask.sum().item())
                    if valid_count == 0:
                        loss = (seg_pred.sum() * 0.0).to(seg_pred.dtype)
                    else:
                        seg_gt = torch.where(valid_mask, seg_gt, torch.tensor(self.ignore_index, device=seg_gt.device))
                        loss = criterion(seg_pred, seg_gt)

                raw_weight = self.config.depth_loss_weight if task == 'depth' else self.config.seg_loss_weight
                safe_weight = float(raw_weight) if isinstance(raw_weight, (int, float)) and raw_weight > 0 else 1.0
                loss = loss * torch.tensor(safe_weight, device=loss.device, dtype=loss.dtype)

            if not torch.isfinite(loss).item():
                self.logger.warning(f"Non-finite {task} loss detected at step {i}; skipping backward/optimizer step.")
                self._log_nan_debug(task, batch, "loss", loss, outputs)
                optimizer.zero_grad(set_to_none=True)
                if extra_optimizer is not None:
                    extra_optimizer.zero_grad(set_to_none=True)
                continue

            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self._rescale_camera_head_grads(camera_loss_applied)
                self.scaler.unscale_(optimizer)
                if extra_optimizer is not None:
                    self.scaler.unscale_(extra_optimizer)
            else:
                loss.backward()
                self._rescale_camera_head_grads(camera_loss_applied)

            max_norm = getattr(self.config, 'clip_grad_norm', 1.0)
            if max_norm and max_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=max_norm)

            if self.config.mixed_precision and self.scaler is not None:
                grads_ok = True
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            grads_ok = False
                            break
                if grads_ok:
                    self.scaler.step(optimizer)
                    if extra_optimizer is not None:
                        extra_optimizer.step()
                else:
                    self.logger.warning(f"Detected non-finite gradients ({task}). Skipping optimizer step.")
                    self._log_nan_debug(task, batch, "grad", loss, outputs)
                    optimizer.zero_grad(set_to_none=True)
                    if extra_optimizer is not None:
                        extra_optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                optimizer.step()
                if extra_optimizer is not None:
                    extra_optimizer.step()

            total_loss += loss.item()
            if camera_loss is not None:
                total_camera_loss += camera_loss.item()
                camera_batches += 1
            log_training_progress(self.logger, epoch, self.config.epochs, task, i, len(data_loader), loss.item())

        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        avg_camera = total_camera_loss / camera_batches if camera_batches > 0 else 0.0
        return avg_loss, avg_camera

    def step_schedulers(self) -> None:
        if self.scheduler_unified:
            self.scheduler_unified.step()
        else:
            if self.scheduler_depth:
                self.scheduler_depth.step()
            if self.scheduler_seg:
                self.scheduler_seg.step()
            if self.scheduler_camera:
                self.scheduler_camera.step()

    def get_current_lr(self) -> Any:
        if self.scheduler_unified:
            return self.scheduler_unified.get_last_lr()[0]
        else:
            depth_lr = self.scheduler_depth.get_last_lr()[0] if self.scheduler_depth else None
            seg_lr = self.scheduler_seg.get_last_lr()[0] if self.scheduler_seg else None
            camera_lr = self.scheduler_camera.get_last_lr()[0] if self.scheduler_camera else None
            return depth_lr, seg_lr, camera_lr
