#!/usr/bin/env python3
"""
多任务训练器模块
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from typing import Optional, Any
from itertools import cycle
import math

from .config import TrainingConfig
from .loss import SiLogLoss
from .train_utils import clear_cuda_cache, log_training_progress, autocast
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
                 scheduler_depth: Optional[Any] = None,
                 scheduler_seg: Optional[Any] = None,
                 optimizer_unified: Optional[AdamW] = None,
                 scheduler_unified: Optional[Any] = None):
        super().__init__(model, config, logger, writer)
        self.rank = rank
        # global step counter for per-step logging
        self.global_step = 0

        self.optimizer_depth = optimizer_depth
        self.optimizer_seg = optimizer_seg
        self.scheduler_depth = scheduler_depth
        self.scheduler_seg = scheduler_seg
        self.optimizer_unified = optimizer_unified
        self.scheduler_unified = scheduler_unified

        self.loss_weighter = loss_weighter
        self.depth_criterion = SiLogLoss().cuda()
        # Keep reduction='mean' but handle all-ignore batches explicitly before calling
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean').cuda()
        self.ignore_index = 255

    def train_epoch(self, train_depth_loader: DataLoader, train_seg_loader: DataLoader, epoch: int) -> tuple:
        from .train_utils import set_epoch_for_samplers
        set_epoch_for_samplers(train_depth_loader, train_seg_loader, epoch)
        self.logger.info(f"===========> Epoch: {epoch}/{self.config.epochs}")
        self.model.train()

        if self.optimizer_unified:
            avg_depth_loss, avg_seg_loss = self._train_epoch_unified(train_depth_loader, train_seg_loader, epoch)
        else:
            avg_depth_loss = self._train_task_epoch(train_depth_loader, 'depth', epoch)
            avg_seg_loss = self._train_task_epoch(train_seg_loader, 'seg', epoch)

        self.loss_weighter.update_weights(avg_depth_loss, avg_seg_loss)

        if self.rank == 0:
            self.writer.add_scalar("train/depth_loss", avg_depth_loss, epoch)
            self.writer.add_scalar("train/seg_loss", avg_seg_loss, epoch)

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

            self.logger.info(f"Epoch {epoch} - Avg Depth Loss: {avg_depth_loss:.4f}, Avg Seg Loss: {avg_seg_loss:.4f}")

        return avg_depth_loss, avg_seg_loss

    def _train_epoch_unified(self, train_depth_loader: DataLoader, train_seg_loader: DataLoader, epoch: int) -> tuple:
        total_depth_loss, total_seg_loss = 0, 0
        depth_batches, seg_batches = 0, 0

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

            # 深度任务
            with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                img_depth = batch_depth["image"].cuda()
                gt_depth = batch_depth["depth"].cuda()
                pred_depth = self.model(img_depth, task='depth')['depth']
                mask_depth = (gt_depth > 0) & (gt_depth >= self.config.min_depth) & (gt_depth <= self.config.max_depth)
                loss_depth = self.depth_criterion(pred_depth, gt_depth, mask_depth)
                weighted_loss_depth = self.loss_weighter.get_loss(loss_depth, torch.tensor(0.0, device=loss_depth.device), task='depth')

            # Backward for depth loss (AMP-safe)
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(weighted_loss_depth).backward()
            else:
                weighted_loss_depth.backward()

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

            # 额外的数值健壮性：裁剪后检查梯度是否为有限值，若存在NaN/Inf则跳过本次step
            grads_ok = True
            bad_params = []
            named_params = list(self.model.named_parameters())
            if getattr(self.loss_weighter, 'strategy', None) == 'uwl':
                named_params.append(("loss_weighter.log_vars", self.loss_weighter.log_vars))
            for name, p in named_params:
                if p is None or not p.requires_grad or p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    grads_ok = False
                    bad_params.append(name)

            if self.config.mixed_precision and self.scaler is not None:
                if grads_ok:
                    self.scaler.step(self.optimizer_unified)
                    self.loss_weighter.sanitize_parameters()
                else:
                    self.logger.warning(f"Detected non-finite gradients (unified) in: {bad_params}. Skipping optimizer step.")
                    self.optimizer_unified.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                if grads_ok:
                    self.optimizer_unified.step()
                    self.loss_weighter.sanitize_parameters()
                else:
                    self.logger.warning(f"Detected non-finite gradients (unified) in: {bad_params}. Skipping optimizer step.")
                    self.optimizer_unified.zero_grad(set_to_none=True)

            # Step-level UWL logging (compact, single call) and step increment
            self._log_step_uwl()
            self.global_step += 1

            total_depth_loss += loss_depth.item()
            total_seg_loss += loss_seg.item()
            depth_batches += 1
            seg_batches += 1

        avg_depth_loss = total_depth_loss / depth_batches if depth_batches > 0 else 0
        avg_seg_loss = total_seg_loss / seg_batches if seg_batches > 0 else 0

        return avg_depth_loss, avg_seg_loss

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

            # Single writer call with grouped series
            self.writer.add_scalars(
                "train_step/uwl",
                {
                    "log_var_depth": log_var_depth,
                    "log_var_seg": log_var_seg,
                    "weight_depth": weight_depth,
                    "weight_seg": weight_seg,
                },
                global_step=self.global_step,
            )

        except Exception as e:
            # Avoid training interruption due to logging issues
            self.logger.debug(f"Step UWL logging skipped due to error: {e}")

    def _train_task_epoch(self, data_loader: DataLoader, task: str, epoch: int) -> float:
        optimizer = self.optimizer_depth if task == 'depth' else self.optimizer_seg
        criterion = self.depth_criterion if task == 'depth' else self.seg_criterion

        self.logger.info(f"--- Training {task.capitalize()} Head ---")
        total_loss = 0
        total_intrinsics_loss = 0.0
        intrinsics_loss_steps = 0

        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            input_img = batch["image"].cuda()

            with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                if task == 'depth':
                    depth_gt = batch["depth"].cuda()
                    outputs = self.model(input_img, task='depth')
                    depth_pred = outputs['depth']
                    valid_mask = (depth_gt > 0) & (depth_gt >= self.config.min_depth) & (depth_gt <= self.config.max_depth)
                    loss = criterion(depth_pred, depth_gt, valid_mask)
                    intrinsics_weight = float(getattr(self.config, 'intrinsics_loss_weight', 0.0) or 0.0)
                    if (
                        intrinsics_weight > 0
                        and 'intrinsics' in outputs
                        and 'intrinsics' in batch
                    ):
                        intrinsics_gt = batch["intrinsics"].cuda()
                        fx_gt = intrinsics_gt[:, 0, 0]
                        fy_gt = intrinsics_gt[:, 1, 1]
                        intrinsics_target = torch.stack([fx_gt, fy_gt], dim=-1)
                        intrinsics_pred = torch.nan_to_num(outputs['intrinsics'], nan=0.0)
                        intrinsics_loss = F.l1_loss(intrinsics_pred, intrinsics_target)
                        loss = loss + intrinsics_weight * intrinsics_loss
                        total_intrinsics_loss += intrinsics_loss.detach().item()
                        intrinsics_loss_steps += 1
                else:  # seg
                    seg_gt = batch["semseg_mask"].cuda().long()
                    outputs = self.model(input_img, task='seg')
                    seg_pred = outputs['seg']

                    # 数值稳定性：先移除 NaN/Inf，再限制 logits 幅度
                    seg_pred = torch.nan_to_num(seg_pred, nan=0.0)
                    seg_pred = seg_pred.clamp_(-100.0, 100.0)

                    # 处理全部为ignore/越界的批次，避免CrossEntropy内部出现无效均值
                    valid_mask = (seg_gt >= 0) & (seg_gt < self.config.num_classes)
                    valid_count = int(valid_mask.sum().item())
                    if valid_count == 0:
                        loss = (seg_pred.sum() * 0.0).to(seg_pred.dtype)
                    else:
                        seg_gt = torch.where(valid_mask, seg_gt, torch.tensor(self.ignore_index, device=seg_gt.device))
                        loss = criterion(seg_pred, seg_gt)

                # 使用 'fixed' 或 'dwa' 权重
                raw_weight = self.config.depth_loss_weight if task == 'depth' else self.config.seg_loss_weight
                # 数值保护：若权重无效则回退到1.0
                if not (isinstance(raw_weight, (int, float)) and raw_weight > 0):
                    safe_weight = 1.0
                else:
                    safe_weight = float(raw_weight)
                loss = loss * torch.tensor(safe_weight, device=loss.device, dtype=loss.dtype)

            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
            else:
                loss.backward()

            # 梯度裁剪 (在 unscale 之后, step 之前)
            max_norm = getattr(self.config, 'clip_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=max_norm)

            if self.config.mixed_precision and self.scaler is not None:
                # 裁剪后检查梯度是否为有限值，若存在NaN/Inf则跳过本次step
                grads_ok = True
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            grads_ok = False
                            break
                if grads_ok:
                    self.scaler.step(optimizer)
                else:
                    self.logger.warning(f"Detected non-finite gradients ({task}). Skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                optimizer.step()

            total_loss += loss.item()
            log_training_progress(self.logger, epoch, self.config.epochs, task, i, len(data_loader), loss.item())

        if (
            task == 'depth'
            and intrinsics_loss_steps > 0
            and getattr(self, 'writer', None) is not None
            and self.rank == 0
        ):
            avg_intrinsics_loss = total_intrinsics_loss / intrinsics_loss_steps
            self.writer.add_scalar("train/intrinsics_loss", avg_intrinsics_loss, epoch)

        return total_loss / len(data_loader) if len(data_loader) > 0 else 0

    def step_schedulers(self) -> None:
        if self.scheduler_unified:
            self.scheduler_unified.step()
        else:
            self.scheduler_depth.step()
            self.scheduler_seg.step()

    def get_current_lr(self) -> Any:
        if self.scheduler_unified:
            return self.scheduler_unified.get_last_lr()[0]
        else:
            return self.scheduler_depth.get_last_lr()[0], self.scheduler_seg.get_last_lr()[0]
