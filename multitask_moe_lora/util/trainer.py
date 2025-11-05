#!/usr/bin/env python3
"""
多任务训练器模块
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from typing import Optional, Any, Dict
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
        self.depth_criterion = SiLogLoss().cuda()
        self.camera_criterion = torch.nn.L1Loss(reduction='mean').cuda()
        # Keep reduction='mean' but handle all-ignore batches explicitly before calling
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()
        self.ignore_index = 255
        self._latest_camera_loss: float = 0.0

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

    def train_epoch(self, train_depth_loader: DataLoader, train_seg_loader: DataLoader, epoch: int) -> tuple:
        from .train_utils import set_epoch_for_samplers
        set_epoch_for_samplers(train_depth_loader, train_seg_loader, epoch)
        self.logger.info(f"===========> Epoch: {epoch}/{self.config.epochs}")
        self.model.train()

        depth_tracker = self._build_progress_tracker(train_depth_loader) if train_depth_loader is not None else None
        seg_tracker = self._build_progress_tracker(train_seg_loader) if train_seg_loader is not None else None

        if self.optimizer_unified and train_seg_loader is not None:
            avg_depth_loss, avg_seg_loss, avg_camera_loss = self._train_epoch_unified(
                train_depth_loader, train_seg_loader, epoch, depth_tracker, seg_tracker
            )
        else:
            avg_depth_loss, avg_camera_loss = self._train_task_epoch(train_depth_loader, 'depth', epoch, depth_tracker)
            if train_seg_loader is not None:
                avg_seg_loss, _ = self._train_task_epoch(train_seg_loader, 'seg', epoch, seg_tracker)
            else:
                avg_seg_loss = 0.0

        self.loss_weighter.update_weights(avg_depth_loss, avg_seg_loss)
        self._latest_camera_loss = avg_camera_loss

        if self.rank == 0:
            self.writer.add_scalar("train/depth_loss", avg_depth_loss, epoch)
            self.writer.add_scalar("train/seg_loss", avg_seg_loss, epoch)
            if self.config.camera_head_mode.lower() != 'none':
                self.writer.add_scalar("train/camera_loss", avg_camera_loss, epoch)

            # 记录损失权重
            if self.loss_weighter.strategy == 'uwl':
                bounds = getattr(self.loss_weighter, 'log_var_bounds', (-float('inf'), float('inf')))
                clamp_enabled = all(map(math.isfinite, bounds))

                raw_log_var_depth = self.loss_weighter.log_vars[0].item()
                raw_log_var_seg = self.loss_weighter.log_vars[1].item()
                log_var_depth = max(bounds[0], min(bounds[1], raw_log_var_depth)) if clamp_enabled else raw_log_var_depth
                log_var_seg = max(bounds[0], min(bounds[1], raw_log_var_seg)) if clamp_enabled else raw_log_var_seg

                self.writer.add_scalar("train/uwl_log_var_depth", log_var_depth, epoch)
                self.writer.add_scalar("train/uwl_log_var_seg", log_var_seg, epoch)

                # 数值稳定：记录经过裁剪后的权重因子 exp(-log_var)
                if clamp_enabled:
                    weight_factor_depth = math.exp(-log_var_depth)
                    weight_factor_seg = math.exp(-log_var_seg)
                else:
                    weight_factor_depth = torch.exp(-self.loss_weighter.log_vars[0]).item()
                    weight_factor_seg = torch.exp(-self.loss_weighter.log_vars[1]).item()

                self.writer.add_scalar("train/weight_depth", weight_factor_depth, epoch)
                self.writer.add_scalar("train/weight_seg", weight_factor_seg, epoch)

            else:  # fixed 和 dwa
                self.writer.add_scalar("train/weight_depth", self.config.depth_loss_weight, epoch)
                self.writer.add_scalar("train/weight_seg", self.config.seg_loss_weight, epoch)

            if self.config.camera_head_mode.lower() != 'none':
                self.logger.info(
                    f"Epoch {epoch} - Avg Depth Loss: {avg_depth_loss:.4f}, Avg Seg Loss: {avg_seg_loss:.4f}, "
                    f"Avg Camera Loss: {avg_camera_loss:.4f}"
                )
            else:
                self.logger.info(f"Epoch {epoch} - Avg Depth Loss: {avg_depth_loss:.4f}, Avg Seg Loss: {avg_seg_loss:.4f}")

        return avg_depth_loss, avg_seg_loss

    def _train_epoch_unified(self,
                             train_depth_loader: DataLoader,
                             train_seg_loader: DataLoader,
                             epoch: int,
                             depth_tracker: Optional[Dict[str, Any]],
                             seg_tracker: Optional[Dict[str, Any]]) -> tuple:
        if train_depth_loader is None or train_seg_loader is None:
            raise ValueError("Unified optimizer requires both depth and segmentation loaders.")
        total_depth_loss, total_seg_loss = 0, 0
        total_camera_loss = 0.0
        depth_batches, seg_batches, camera_batches = 0, 0, 0

        len_depth = len(train_depth_loader)
        len_seg = len(train_seg_loader)

        # 使用itertools.cycle来处理不同长度的数据加载器，避免重新创建迭代器
        if len_depth > len_seg:
            long_loader, short_loader = train_depth_loader, train_seg_loader
            is_depth_long = True
        else:
            long_loader, short_loader = train_seg_loader, train_depth_loader
            is_depth_long = False

        short_loader_iter = iter(cycle(short_loader))

        for i, long_batch in enumerate(long_loader):
            short_batch = next(short_loader_iter)

            if is_depth_long:
                batch_depth, batch_seg = long_batch, short_batch
            else:
                batch_depth, batch_seg = short_batch, long_batch

            self.optimizer_unified.zero_grad()

            self._log_dataset_progress("Train", "Depth", epoch, batch_depth.get("dataset_name"), depth_tracker)
            # 深度任务
            camera_loss = None
            with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                img_depth = batch_depth["image"].cuda()
                gt_depth = batch_depth["depth"].cuda()
                outputs_depth = self.model(img_depth, task='depth')
                pred_depth = outputs_depth['depth']
                mask_depth = (gt_depth > 0) & (gt_depth >= self.config.min_depth) & (gt_depth <= self.config.max_depth)
                loss_depth = self.depth_criterion(pred_depth, gt_depth, mask_depth)

                if self.config.camera_head_mode.lower() != 'none' and \
                        'camera_intrinsics_norm' in outputs_depth and \
                        batch_depth.get('camera_intrinsics_norm') is not None:
                    gt_camera = batch_depth['camera_intrinsics_norm'].cuda()
                    camera_pred = outputs_depth['camera_intrinsics_norm']
                    camera_loss = self.camera_criterion(camera_pred, gt_camera)
                    loss_depth = loss_depth + float(self.config.camera_loss_weight) * camera_loss

                weighted_loss_depth = self.loss_weighter.get_loss(
                    loss_depth, torch.tensor(0.0, device=loss_depth.device), task='depth'
                )

            # Backward for depth loss (AMP-safe)
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(weighted_loss_depth).backward()
            else:
                weighted_loss_depth.backward()

            self._log_dataset_progress("Train", "Seg", epoch, batch_seg.get("dataset_name"), seg_tracker)
            # 分割任务
            with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                img_seg = batch_seg["image"].cuda()
                gt_seg = batch_seg["semseg_mask"].cuda().long()
                pred_seg = self.model(img_seg, task='seg')['seg']

                # 强化数值稳定性：先清理 NaN/Inf，再截断绝对值过大的 logits
                pred_seg = torch.nan_to_num(pred_seg, nan=0.0)
                pred_seg = pred_seg.clamp_(-100.0, 100.0)

                # 若本批次没有任何有效标签（全部为ignore_index或越界），跳过损失以避免NaN
                valid_mask = (gt_seg >= 0) & (gt_seg < self.config.num_classes)
                valid_count = int(valid_mask.sum().item())
                if valid_count == 0:
                    loss_seg = (pred_seg.sum() * 0.0).to(pred_seg.dtype)
                else:
                    # 将越界标签设为ignore_index，保证CrossEntropyLoss稳定
                    gt_seg = torch.where(valid_mask, gt_seg, torch.tensor(self.ignore_index, device=gt_seg.device))
                    loss_seg = self.seg_criterion(pred_seg, gt_seg)
                weighted_loss_seg = self.loss_weighter.get_loss(torch.tensor(0.0, device=loss_seg.device), loss_seg, task='seg')

            # Backward for seg loss (AMP-safe)
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(weighted_loss_seg).backward()
            else:
                weighted_loss_seg.backward()

            # 统一更新：在AMP下先unscale，再裁剪；非AMP直接裁剪
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.unscale_(self.optimizer_unified)

            # 梯度裁剪
            max_norm = getattr(self.config, 'clip_grad_norm', 1.0)
            params_to_clip = [p for p in self.model.parameters() if p.requires_grad]
            if getattr(self.loss_weighter, 'strategy', None) == 'uwl':
                params_to_clip.append(self.loss_weighter.log_vars)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=max_norm)

            named_params = list(self.model.named_parameters())
            if getattr(self.loss_weighter, 'strategy', None) == 'uwl':
                named_params.append(("loss_weighter.log_vars", self.loss_weighter.log_vars))
            bad_params = self._sanitize_gradients(named_params)
            if bad_params and self.rank == 0:
                self.logger.debug(f"Sanitized non-finite gradients for: {bad_params}")

            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.step(self.optimizer_unified)
                self.loss_weighter.sanitize_parameters()
                self.scaler.update()
            else:
                self.optimizer_unified.step()
                self.loss_weighter.sanitize_parameters()

            # Step-level UWL logging (compact, single call) and step increment
            self._log_step_uwl()
            self.global_step += 1

            total_depth_loss += loss_depth.item()
            total_seg_loss += loss_seg.item()
            depth_batches += 1
            seg_batches += 1
            if camera_loss is not None:
                total_camera_loss += camera_loss.item()
                camera_batches += 1

        avg_depth_loss = total_depth_loss / depth_batches if depth_batches > 0 else 0
        avg_seg_loss = total_seg_loss / seg_batches if seg_batches > 0 else 0
        avg_camera_loss = total_camera_loss / camera_batches if camera_batches > 0 else 0.0

        return avg_depth_loss, avg_seg_loss, avg_camera_loss

    def _sanitize_gradients(self, named_params):
        bad = []
        for name, param in named_params:
            if param is None or not param.requires_grad or param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                bad.append(name)
        return bad

    def _log_step_uwl(self) -> None:
        """Log UWL variables and weights per step under a separate category.
        Uses a single add_scalars call to minimize overhead. Only rank 0 logs.
        """
        if self.rank != 0 or self.writer is None:
            return
        if getattr(self.loss_weighter, 'strategy', None) != 'uwl':
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
            with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                if task == 'depth':
                    depth_gt = batch["depth"].cuda()
                    outputs = self.model(input_img, task='depth')
                    depth_pred = outputs['depth']
                    valid_mask = (depth_gt > 0) & (depth_gt >= self.config.min_depth) & (depth_gt <= self.config.max_depth)
                    loss = criterion(depth_pred, depth_gt, valid_mask)

                    if self.config.camera_head_mode.lower() != 'none' and \
                            'camera_intrinsics_norm' in outputs and \
                            batch.get('camera_intrinsics_norm') is not None:
                        gt_camera = batch['camera_intrinsics_norm'].cuda()
                        camera_pred = outputs['camera_intrinsics_norm']
                        camera_loss = self.camera_criterion(camera_pred, gt_camera)
                        loss = loss + float(self.config.camera_loss_weight) * camera_loss
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

            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                if extra_optimizer is not None:
                    self.scaler.unscale_(extra_optimizer)
            else:
                loss.backward()

            max_norm = getattr(self.config, 'clip_grad_norm', 1.0)
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
