#!/usr/bin/env python3
"""
验证评估模块
包含验证逻辑、指标计算和结果可视化功能
"""

import os
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
import torch.distributed as dist

from .config import TrainingConfig
from .loss import SiLogLoss
from .metric import SegMetric, eval_depth
# Try to import simple save helpers; fallback to legacy names if needed
try:
    from .visualize import save_depth_prediction, save_seg_prediction
except ImportError:
    try:
        # Legacy API fallback
        from .visualize import save_depth_output, save_seg_output  # type: ignore

        def save_depth_prediction(pred, filename, outdir, colormap: str = 'gray'):
            return save_depth_output(pred, filename, outdir,
                                     norm_type='min-max', colormap=colormap,
                                     save_img=True, save_pt=False, save_npz=False)

        def save_seg_prediction(pred, filename, outdir):
            return save_seg_output(pred, filename, outdir,
                                   save_img=True, save_pt=False, save_npz=False)
    except Exception:
        # As a last resort, define no-op functions to avoid import-time failure
        def save_depth_prediction(*args, **kwargs):
            return None

        def save_seg_prediction(*args, **kwargs):
            return None


class DepthValidator:
    """深度估计验证器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.criterion = SiLogLoss()
        self.pred_folder = None

    def setup_output_folder(self, save_path: str) -> None:
        """设置输出文件夹"""
        self.pred_folder = os.path.join(save_path, 'pred_depth')
        os.makedirs(self.pred_folder, exist_ok=True)

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

        # 使用与原始模型完全一致的掩码逻辑
        valid_mask_4d = (target_gt > 0) & (target_gt >= self.config.min_depth) & (target_gt <= self.config.max_depth)
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


class SegValidator:
    """语义分割验证器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metric = SegMetric(config.num_classes)
        self.criterion = CrossEntropyLoss(ignore_index=255)
        self.pred_folder = None

    def setup_output_folder(self, save_path: str) -> None:
        """设置输出文件夹"""
        self.pred_folder = os.path.join(save_path, 'pred_seg')
        os.makedirs(self.pred_folder, exist_ok=True)

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
        """保存预测结果"""
        if self.pred_folder is not None:
            save_seg_prediction(pred, f'epoch_{epoch}.png', self.pred_folder)

    def get_metrics(self) -> Dict[str, float]:
        """获取指标"""
        return self.metric.get_scores()


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

    local_results = []
    total_loss_local = 0

    temp_validator = None
    if task_type == 'depth':
        temp_validator = DepthValidator(config)
    elif task_type == 'seg':
        temp_validator = SegValidator(config)

    if rank == 0 and temp_validator is not None:
        temp_validator.setup_output_folder(config.save_path)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_img = batch["image"].cuda()
            outputs, loss = None, torch.tensor(0.0)

            if task_type == 'depth':
                target_gt = batch["depth"].cuda()
                outputs = model(input_img, task='depth')
                pred = outputs['depth']
                valid_mask_4d = (target_gt > 0) & (target_gt >= config.min_depth) & (target_gt <= config.max_depth)
                loss = temp_validator.criterion(pred, target_gt, valid_mask_4d)

                # 保存第一个batch的预测结果（仅rank 0）
                if i == 0 and rank == 0:
                    temp_validator.save_prediction(pred, epoch)

                # 收集用于指标计算的数据
                source_types = batch.get("source_type", None)
                for j in range(pred.shape[0]):
                    valid_mask = valid_mask_4d[j].squeeze(0)
                    if valid_mask.numel() == 0 or valid_mask.sum().item() == 0:
                        continue  # 跳过没有有效像素的样本，避免后续指标计算出现空数组

                    local_results.append({
                        'pred': pred[j][valid_mask].cpu().numpy(),
                        'gt': target_gt[j].squeeze(0)[valid_mask].cpu().numpy(),
                        'source_type': source_types[j] if source_types and j < len(source_types) else "combined"
                    })

            elif task_type == 'seg':
                target_gt = batch["semseg_mask"].cuda()
                outputs = model(input_img, task='seg')
                pred = outputs['seg']
                loss = temp_validator.criterion(pred, target_gt)

                # 保存第一个batch的预测结果（仅rank 0）
                if i == 0 and rank == 0:
                    temp_validator.save_prediction(pred, epoch)

                # 收集用于指标计算的数据
                pred_labels = pred.argmax(dim=1)
                for j in range(pred.shape[0]):
                    local_results.append({'pred': pred_labels[j].cpu(), 'gt': target_gt[j].cpu(), 'source_type': batch.get("source_type", [])[j]})

            total_loss_local += loss.item()

    # 同步所有进程
    dist.barrier()

    # 收集所有进程的损失
    total_loss_tensor = torch.tensor(total_loss_local, device='cuda')
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)

    # 收集所有进程的结果
    gathered_results: List[List[Dict[str, Any]]] = [None] * world_size
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        # --- 在主进程上处理和计算聚合后的指标 ---
        all_metrics = {}

        # 计算平均损失
        avg_loss = total_loss_tensor.item() / (len(val_loader) * world_size)
        writer.add_scalar(f'Validation_Loss/{task_type}', avg_loss, epoch)

        # 扁平化聚合结果
        flat_results = [item for sublist in gathered_results for item in sublist]

        if task_type == 'depth':
            # 按数据集类型分类结果
            categorized_results = {"kidney": {'preds': [], 'gts': []}, "colon": {'preds': [], 'gts': []}, "combined": {'preds': [], 'gts': []}}
            for res in flat_results:
                source_type = res.get('source_type', 'combined')
                if source_type not in categorized_results:
                    logger.warning(f"Unknown source_type '{source_type}' encountered during depth validation. Falling back to 'combined'.")
                    source_type = 'combined'
                categorized_results[source_type]['preds'].append(res['pred'])
                categorized_results[source_type]['gts'].append(res['gt'])
                if source_type != 'combined':
                    categorized_results['combined']['preds'].append(res['pred'])
                    categorized_results['combined']['gts'].append(res['gt'])

            # 计算指标
            for name, data in categorized_results.items():
                if not data['preds']:
                    continue
                preds_all = np.concatenate(data['preds'])
                gts_all = np.concatenate(data['gts'])
                if preds_all.size == 0:
                    logger.warning(f"No valid pixels for depth validation on '{name}'.")
                    continue

                metrics = eval_depth(torch.from_numpy(preds_all), torch.from_numpy(gts_all))
                all_metrics[name] = metrics
                logger.info(f"[{name.capitalize()}] Depth Validation Epoch {epoch} - absrel: {metrics.get('absrel', 0):.4f}, rmse: {metrics.get('rmse', 0):.4f}")
                for k, v in metrics.items():
                    writer.add_scalar(f'Metrics/Depth_{name}/{k}', v, epoch)
                    logger.info(f"    {k}: {v:.6f}")

        elif task_type == 'seg':
            # 按数据集类型分类结果
            metrics_calculators = {"kidney": SegMetric(config.num_classes), "colon": SegMetric(config.num_classes), "combined": SegMetric(config.num_classes)}
            for res in flat_results:
                source_type = res.get('source_type', 'combined')
                if source_type not in metrics_calculators:
                    logger.warning(f"Unknown source_type '{source_type}' encountered during seg validation. Falling back to 'combined'.")
                    source_type = 'combined'
                metrics_calculators[source_type].update(res['pred'].unsqueeze(0), res['gt'].unsqueeze(0))
                if source_type != 'combined':
                    metrics_calculators['combined'].update(res['pred'].unsqueeze(0), res['gt'].unsqueeze(0))

            # 计算指标
            for name, metric_calc in metrics_calculators.items():
                seg_scores = metric_calc.get_scores()
                all_metrics[name] = seg_scores
                logger.info(f"[{name.capitalize()}] Seg Validation Epoch {epoch} - mIoU: {seg_scores.get('miou', 0):.4f}, mDice: {seg_scores.get('mdice', 0):.4f}")
                for k, v in seg_scores.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        writer.add_scalar(f'Metrics/Seg_{name}/{k}', v, epoch)
                logger.info(f"    Overall Accuracy: {seg_scores.get('acc_overall', 0):.4f}, Mean IoU: {seg_scores.get('miou', 0):.4f}")

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

    logger.info("--- Initial Segmentation Validation ---")
    validate_and_visualize(model, val_seg_loader, 'seg', -1, config, writer, logger)

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
    if rank == 0:
        logger.info("--- Segmentation Validation ---")
    if dist.is_initialized():
        dist.barrier()
    seg_metrics = validate_and_visualize(model, val_seg_loader, 'seg', epoch, config, writer, logger)

    # 只有主进程返回指标
    if rank == 0:
        return depth_metrics, seg_metrics
    else:
        return None, None
