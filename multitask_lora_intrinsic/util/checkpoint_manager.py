#!/usr/bin/env python3
"""
检查点管理模块
"""

import os
import logging
from typing import Dict, Any, Optional
import torch

from .config import TrainingConfig
from .train_utils import save_checkpoint
from .csv_logger import CSVLogger
# Metrics/Depth_combined/absrel
# Metrics/Seg_combined/miou


class CheckpointManager:
    """负责管理和保存训练检查点"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_depth: Optional[torch.optim.Optimizer],
                 optimizer_seg: Optional[torch.optim.Optimizer],
                 scheduler_depth: Optional[Any],
                 scheduler_seg: Optional[Any],
                 config: TrainingConfig,
                 logger: logging.Logger,
                 rank: int = 0,
                 optimizer_unified: Optional[torch.optim.Optimizer] = None,
                 scheduler_unified: Optional[Any] = None):
        """
        初始化 CheckpointManager
        
        Args:
            model: 训练模型
            optimizer_depth: 深度优化器
            optimizer_seg: 分割优化器
            scheduler_depth: 深度学习率调度器
            scheduler_seg: 分割学习率调度器
            config: 训练配置
            logger: 日志记录器
            rank: 分布式训练的进程rank
        """
        self.model = model
        self.optimizer_depth = optimizer_depth
        self.optimizer_seg = optimizer_seg
        self.scheduler_depth = scheduler_depth
        self.scheduler_seg = scheduler_seg
        self.optimizer_unified = optimizer_unified
        self.scheduler_unified = scheduler_unified
        self.config = config
        self.logger = logger
        self.rank = rank
        
        # 初始化最佳指标跟踪器
        self.best_metrics = {
            "kidney": {"absrel": float('inf'), "miou": float('-inf'), "miou_minus_absrel": float('-inf')},
            "colon": {"absrel": float('inf'), "miou": float('-inf'), "miou_minus_absrel": float('-inf')},
            "combined": {"absrel": float('inf'), "miou": float('-inf'), "miou_minus_absrel": float('-inf')},
        }
        
        # 初始化CSV Logger
        csv_path = os.path.join(self.config.save_path, "best_metrics.csv")
        csv_columns = [
            'best_type', 'epoch', 'train_depth_loss', 'train_seg_loss',
            # Combined指标
            'val_depth_absrel_combined', 'val_seg_miou_combined', 'val_seg_mdice_combined',
            'val_seg_c0iou_combined', 'val_seg_c1iou_combined', 'val_seg_c2iou_combined', 'val_seg_c3iou_combined',
            # Kidney指标
            'val_depth_absrel_kidney', 'val_seg_miou_kidney', 'val_seg_mdice_kidney',
            'val_seg_c0iou_kidney', 'val_seg_c1iou_kidney', 'val_seg_c2iou_kidney', 'val_seg_c3iou_kidney',
            # Colon指标
            'val_depth_absrel_colon', 'val_seg_miou_colon', 'val_seg_mdice_colon',
            'val_seg_c0iou_colon', 'val_seg_c1iou_colon', 'val_seg_c2iou_colon', 'val_seg_c3iou_colon'
        ]
        self.csv_logger = CSVLogger(csv_path, csv_columns)

    def save(self, epoch: int, suffix: str):
        """调用底层的 save_checkpoint 函数"""
        save_checkpoint(
            model=self.model,
            optimizer_depth=self.optimizer_depth,
            optimizer_seg=self.optimizer_seg,
            scheduler_depth=self.scheduler_depth,
            scheduler_seg=self.scheduler_seg,
            optimizer_unified=self.optimizer_unified,
            scheduler_unified=self.scheduler_unified,
            epoch=epoch,
            save_path=self.config.save_path,
            suffix=suffix,
            rank=self.rank
        )

    def update_and_save(self,
                        epoch: int,
                        train_losses: Dict[str, float],
                        depth_metrics_all: Optional[Dict[str, Dict[str, float]]],
                        seg_metrics: Optional[Dict[str, Dict[str, float]]]):
        """
        根据当前epoch的指标更新并保存检查点
        
        Args:
            epoch (int): 当前epoch
            train_losses (Dict): 包含训练损失的字典
            depth_metrics_all (Optional[Dict]): 包含所有深度数据集指标的字典
            seg_metrics (Optional[Dict]): 分割指标字典
        """
        # 1. 保存最新的检查点
        self.save(epoch, "latest")

        # 2. 如果启用，则保存每个epoch的检查点
        if self.config.massive_checkpoint:
            self.save(epoch, f"epoch_{epoch + 1}")

        # 3. 基于指标的最佳检查点保存
        if depth_metrics_all and seg_metrics:
            for a_type in ["kidney", "colon", "combined"]:
                # 确保指标存在
                if a_type not in depth_metrics_all or depth_metrics_all[a_type] is None:
                    continue

                current_absrel = depth_metrics_all[a_type].get('absrel', float('inf'))
                current_miou = seg_metrics.get(a_type, {}).get('miou', float('-inf'))
                current_miou_minus_absrel = current_miou - current_absrel

                # 检查并保存 best_absrel
                if current_absrel < self.best_metrics[a_type]["absrel"]:
                    self.best_metrics[a_type]["absrel"] = current_absrel
                    self.save(epoch, f"best_absrel_{a_type}")
                    if self.rank == 0:
                        self.logger.info(f"New best absrel for {a_type}: {current_absrel:.4f} at epoch {epoch}")
                        self._log_best_to_csv(f"best_absrel_{a_type}", epoch, train_losses, depth_metrics_all, seg_metrics)

                # 检查并保存 best_miou
                if current_miou > self.best_metrics[a_type]["miou"]:
                    self.best_metrics[a_type]["miou"] = current_miou
                    self.save(epoch, f"best_miou_{a_type}")
                    if self.rank == 0:
                        self.logger.info(f"New best miou for {a_type}: {current_miou:.4f} at epoch {epoch}")
                        self._log_best_to_csv(f"best_miou_{a_type}", epoch, train_losses, depth_metrics_all, seg_metrics)
                
                # 检查并保存 best_miou_minus_absrel
                if current_miou_minus_absrel > self.best_metrics[a_type]["miou_minus_absrel"]:
                    self.best_metrics[a_type]["miou_minus_absrel"] = current_miou_minus_absrel
                    self.save(epoch, f"best_miou_minus_absrel_{a_type}")
                    if self.rank == 0:
                        self.logger.info(f"New best miou-absrel for {a_type}: {current_miou_minus_absrel:.4f} at epoch {epoch}")
                        self._log_best_to_csv(f"best_miou_minus_absrel_{a_type}", epoch, train_losses, depth_metrics_all, seg_metrics)

    def _log_best_to_csv(self,
                          best_type: str,
                          epoch: int,
                          train_losses: Dict[str, float],
                          depth_metrics_all: Dict[str, Dict[str, float]],
                          seg_metrics: Dict[str, Dict[str, float]]):
        """将最佳指标记录到CSV文件"""
        data_to_log = {
            'best_type': best_type,
            'epoch': epoch,
            'train_depth_loss': train_losses.get('depth', -1),
            'train_seg_loss': train_losses.get('seg', -1),
            # Combined指标
            'val_depth_absrel_combined': depth_metrics_all.get('combined', {}).get('absrel', -1),
            'val_seg_miou_combined': seg_metrics.get('combined', {}).get('miou', -1),
            'val_seg_mdice_combined': seg_metrics.get('combined', {}).get('mdice', -1),
            'val_seg_c0iou_combined': seg_metrics.get('combined', {}).get('iou_class_0', -1),
            'val_seg_c1iou_combined': seg_metrics.get('combined', {}).get('iou_class_1', -1),
            'val_seg_c2iou_combined': seg_metrics.get('combined', {}).get('iou_class_2', -1),
            'val_seg_c3iou_combined': seg_metrics.get('combined', {}).get('iou_class_3', -1),
            # Kidney指标
            'val_depth_absrel_kidney': depth_metrics_all.get('kidney', {}).get('absrel', -1),
            'val_seg_miou_kidney': seg_metrics.get('kidney', {}).get('miou', -1),
            'val_seg_mdice_kidney': seg_metrics.get('kidney', {}).get('mdice', -1),
            'val_seg_c0iou_kidney': seg_metrics.get('kidney', {}).get('iou_class_0', -1),
            'val_seg_c1iou_kidney': seg_metrics.get('kidney', {}).get('iou_class_1', -1),
            'val_seg_c2iou_kidney': seg_metrics.get('kidney', {}).get('iou_class_2', -1),
            'val_seg_c3iou_kidney': seg_metrics.get('kidney', {}).get('iou_class_3', -1),
            # Colon指标
            'val_depth_absrel_colon': depth_metrics_all.get('colon', {}).get('absrel', -1),
            'val_seg_miou_colon': seg_metrics.get('colon', {}).get('miou', -1),
            'val_seg_mdice_colon': seg_metrics.get('colon', {}).get('mdice', -1),
            'val_seg_c0iou_colon': seg_metrics.get('colon', {}).get('iou_class_0', -1),
            'val_seg_c1iou_colon': seg_metrics.get('colon', {}).get('iou_class_1', -1),
            'val_seg_c2iou_colon': seg_metrics.get('colon', {}).get('iou_class_2', -1),
            'val_seg_c3iou_colon': seg_metrics.get('colon', {}).get('iou_class_3', -1)
        }
        self.csv_logger.update(best_type, data_to_log)