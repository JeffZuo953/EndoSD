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
from typing import Dict, Any, Optional

from .config import TrainingConfig
from .loss import SiLogLoss
from .metric import SegMetric, eval_depth
from .visualize import save_depth_prediction, save_seg_prediction


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
        target_gt = batch["semseg_mask"].cuda()
        
        outputs = model(input_img, task='seg')
        pred = outputs['seg']
        
        # 确保分割标签在有效范围内
        ignore_idx = 255  # CrossEntropyLoss的默认ignore_index
        valid_class_mask = (target_gt >= 0) & (target_gt < self.config.num_classes)
        ignore_mask = (target_gt == ignore_idx)
        valid_mask = valid_class_mask | ignore_mask
        
        # 将无效标签设置为ignore_idx
        target_gt = torch.where(valid_mask, target_gt, torch.tensor(ignore_idx, device=target_gt.device))
        
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
                          logger: logging.Logger) -> Optional[Dict[str, float]]:
    """
    验证和可视化函数
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        task_type: 任务类型 ('depth' 或 'seg')
        epoch: 当前epoch
        config: 训练配置
        writer: TensorBoard writer
        logger: 日志记录器
        
    Returns:
        验证指标字典（如果适用）
    """
    import os
    
    # 检查分布式训练的rank，只有主进程执行验证
    rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    
    # 只有主进程（rank 0）执行验证
    if rank != 0:
        return None
    
    model.eval()
    
    total_loss = 0
    
    if task_type == 'depth':
        validator = DepthValidator(config)
        validator.setup_output_folder(config.save_path)
        metric_depth_val = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                loss, metrics = validator.validate_batch(model, batch)
                total_loss += loss
                
                if metrics is not None:
                    metric_depth_val.append(metrics)
                
                # 保存第一个批次的预测结果
                if i == 0:
                    input_img = batch["image"].cuda()
                    outputs = model(input_img, task='depth')
                    pred = outputs['depth']
                    validator.save_prediction(pred, epoch)
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        writer.add_scalar(f'Validation_Loss/{task_type}', avg_loss, epoch)
        
        if metric_depth_val:
            # metric_depth_val is a list of dicts
            keys = metric_depth_val[0].keys()
            avg_metrics = {k: np.mean([d[k] for d in metric_depth_val]) for k in keys}
            
            logger.info(f"Depth Validation Epoch {epoch} - Loss: {avg_loss:.4f}, absrel: {avg_metrics.get('absrel', 0):.4f}, rmse: {avg_metrics.get('rmse', 0):.4f}")

            # 记录所有深度指标到TensorBoard
            for k, v in avg_metrics.items():
                writer.add_scalar(f'Metrics/Depth_{k}', v, epoch)
            
            logger.info("  Detailed Depth Metrics:")
            for k, v in avg_metrics.items():
                logger.info(f"    {k}: {v:.6f}")
            
            return avg_metrics
    
    elif task_type == 'seg':
        # 为每个数据集类型创建一个验证器
        validators = {
            "kidney": SegValidator(config),
            "colon": SegValidator(config),
            "combined": SegValidator(config)
        }
        
        for validator in validators.values():
            validator.setup_output_folder(config.save_path)
            
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                input_img = batch["image"].cuda()
                target_gt = batch["semseg_mask"].cuda()
                source_types = batch.get("source_type", [])

                outputs = model(input_img, task='seg')
                pred = outputs['seg']

                # 计算一次损失 (使用 combined validator)
                loss = validators["combined"].criterion(pred, target_gt)
                total_loss += loss.item()

                # 根据 source_type 将每个样本分发给对应的 validator
                for j in range(pred.shape[0]):
                    sample_type = source_types[j] if j < len(source_types) else None
                    sample_pred = pred[j].unsqueeze(0)
                    sample_gt = target_gt[j].unsqueeze(0)
                    
                    if sample_type in validators:
                        validators[sample_type].metric.update(sample_pred.argmax(dim=1), sample_gt)
                    
                    # 所有样本都计入 combined
                    validators["combined"].metric.update(sample_pred.argmax(dim=1), sample_gt)

                # 只为第一个批次保存一次预测结果
                if i == 0:
                    validators["combined"].save_prediction(pred, epoch)
        
        # --- 计算并记录每个数据集的指标 ---
        final_combined_scores = {}
        for name, validator in validators.items():
            seg_scores = validator.get_metrics()
            if name == "combined":
                final_combined_scores = seg_scores
                avg_loss = seg_scores.get('avg_loss', total_loss / len(val_loader) if len(val_loader) > 0 else 0)
                writer.add_scalar(f'Validation_Loss/{task_type}', avg_loss, epoch)
            
            logger.info(f"Seg Validation Epoch {epoch} [{name}] - mIoU: {seg_scores.get('miou', 0):.4f}, mDice: {seg_scores.get('mdice', 0):.4f}")

            # 记录指标到 TensorBoard，并添加数据集前缀
            for k, v in seg_scores.items():
                if isinstance(v, (int, float)) and not np.isnan(v) and k not in ['avg_loss', 'total_pixels_N']:
                    if 'class' in k:
                        parts = k.split('_')
                        metric_name = parts[0]
                        class_id = parts[-1]
                        writer.add_scalar(f'Metrics/Seg_{name}_Class_{class_id}/{metric_name}', v, epoch)
                    else:
                        writer.add_scalar(f'Metrics/Seg_{name}/{k}', v, epoch)
            
            # 记录详细日志
            logger.info(f"  [{name}] Overall Accuracy: {seg_scores.get('acc_overall', 0):.4f}")
            logger.info(f"  [{name}] Mean IoU: {seg_scores.get('miou', 0):.4f}")
        
        # 记录一次总像素数
        if epoch == -1 and 'total_pixels_N' in final_combined_scores:
            writer.add_scalar('Metrics/Seg_Total_Pixels_N', final_combined_scores['total_pixels_N'], epoch)
            
        return final_combined_scores

    return None


def run_initial_evaluation(model: torch.nn.Module,
                           val_depth_loader: DataLoader,
                           val_seg_loader: DataLoader, # Now this is the combined loader
                           config: TrainingConfig,
                           writer: SummaryWriter,
                           logger: logging.Logger) -> None:
    """
    运行初始评估
    
    Args:
        model: 模型
        val_depth_loader: 深度验证数据加载器
        val_seg_loader: 合并后的分割验证数据加载器
        config: 训练配置
        writer: TensorBoard writer
        logger: 日志记录器
    """
    import os
    
    rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    
    if rank == 0:
        logger.info("Performing initial evaluation...")
        validate_and_visualize(model, val_depth_loader, 'depth', -1, config, writer, logger)
        validate_and_visualize(model, val_seg_loader, 'seg', -1, config, writer, logger)
        logger.info("Initial evaluation finished.")


def run_epoch_validation(model: torch.nn.Module,
                         val_depth_loader: DataLoader,
                         val_seg_loader: DataLoader, # Now this is the combined loader
                         epoch: int,
                         config: TrainingConfig,
                         writer: SummaryWriter,
                         logger: logging.Logger) -> tuple:
    """
    运行epoch验证
    
    Args:
        model: 模型
        val_depth_loader: 深度验证数据加载器
        val_seg_loader: 合并后的分割验证数据加载器
        epoch: 当前epoch
        config: 训练配置
        writer: TensorBoard writer
        logger: 日志记录器
        
    Returns:
        (depth_metrics, seg_metrics)
    """
    import os
    
    rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    
    if rank == 0:
        depth_metrics = validate_and_visualize(model, val_depth_loader, 'depth', epoch, config, writer, logger)
        seg_metrics = validate_and_visualize(model, val_seg_loader, 'seg', epoch, config, writer, logger)
        return depth_metrics, seg_metrics
    else:
        return None, None