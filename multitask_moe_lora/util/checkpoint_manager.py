#!/usr/bin/env python3
"""
检查点管理模块
"""

import logging
from typing import Dict, Any, Optional
import torch

from .config import TrainingConfig
from .train_utils import save_checkpoint
# Metrics/Depth_combined/absrel
# Metrics/Seg_combined/miou


class CheckpointManager:
    """负责管理和保存训练检查点"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_depth: Optional[torch.optim.Optimizer],
                 optimizer_seg: Optional[torch.optim.Optimizer],
                 optimizer_camera: Optional[torch.optim.Optimizer],
                 scheduler_depth: Optional[Any],
                 scheduler_seg: Optional[Any],
                 scheduler_camera: Optional[Any],
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
        self.optimizer_camera = optimizer_camera
        self.scheduler_depth = scheduler_depth
        self.scheduler_seg = scheduler_seg
        self.scheduler_camera = scheduler_camera
        self.optimizer_unified = optimizer_unified
        self.scheduler_unified = scheduler_unified
        self.config = config
        self.logger = logger
        self.rank = rank
        self._saved_checkpoints: Dict[str, int] = {}
        
        # 初始化最佳指标跟踪器
        self.best_metrics = {
            "NO": {"absrel": float('inf'), "miou": float('-inf'), "miou_minus_absrel": float('-inf')},
            "LS": {"absrel": float('inf'), "miou": float('-inf'), "miou_minus_absrel": float('-inf')},
            "combined": {"absrel": float('inf'), "miou": float('-inf'), "miou_minus_absrel": float('-inf')},
        }
        self.dataset_best_absrel: Dict[str, float] = {}

    def save(self, epoch: int, suffix: str):
        """调用底层的 save_checkpoint 函数"""
        save_checkpoint(
            model=self.model,
            optimizer_depth=self.optimizer_depth,
            optimizer_seg=self.optimizer_seg,
            optimizer_camera=getattr(self, "optimizer_camera", None),
            scheduler_depth=self.scheduler_depth,
            scheduler_seg=self.scheduler_seg,
            scheduler_camera=getattr(self, "scheduler_camera", None),
            optimizer_unified=self.optimizer_unified,
            scheduler_unified=self.scheduler_unified,
            epoch=epoch,
            save_path=self.config.save_path,
            suffix=suffix,
            rank=self.rank
        )
        self._saved_checkpoints[suffix] = epoch + 1

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

        # 2.1 每100个epoch保存一次完整检查点
        if (epoch + 1) % 100 == 0:
            self.save(epoch, f"full_epoch_{epoch + 1}")

        # 3. 基于指标的最佳检查点保存
        seg_metrics = seg_metrics or {}
        domain_metrics = {}
        dataset_metrics = {}
        if depth_metrics_all:
            domain_metrics = depth_metrics_all.get("domains", {})
            if not domain_metrics:
                for key in self.best_metrics.keys():
                    metrics = depth_metrics_all.get(key)
                    if metrics:
                        domain_metrics[key] = metrics
            dataset_metrics = depth_metrics_all.get("datasets", {})

        if domain_metrics:
            for a_type, metrics in domain_metrics.items():
                if metrics is None:
                    continue

                current_absrel = metrics.get('absrel', float('inf'))
                current_miou = seg_metrics.get(a_type, {}).get('miou', float('-inf'))
                current_miou_minus_absrel = current_miou - current_absrel

                # 检查并保存 best_absrel
                if current_absrel < self.best_metrics[a_type]["absrel"]:
                    self.best_metrics[a_type]["absrel"] = current_absrel
                    self.save(epoch, f"best_absrel_{a_type}")
                    if self.rank == 0:
                        self.logger.info(f"New best absrel for {a_type}: {current_absrel:.4f} at epoch {epoch}")

                # 检查并保存 best_miou
                if a_type in seg_metrics and 'miou' in seg_metrics[a_type]:
                    if current_miou > self.best_metrics[a_type]["miou"]:
                        self.best_metrics[a_type]["miou"] = current_miou
                        self.save(epoch, f"best_miou_{a_type}")
                        if self.rank == 0:
                            self.logger.info(f"New best miou for {a_type}: {current_miou:.4f} at epoch {epoch}")
                    
                    # 检查并保存 best_miou_minus_absrel
                    if current_miou_minus_absrel > self.best_metrics[a_type]["miou_minus_absrel"]:
                        self.best_metrics[a_type]["miou_minus_absrel"] = current_miou_minus_absrel
                        self.save(epoch, f"best_miou_minus_absrel_{a_type}")
                        if self.rank == 0:
                            self.logger.info(f"New best miou-absrel for {a_type}: {current_miou_minus_absrel:.4f} at epoch {epoch}")

        # 数据集级别最佳指标
        for dataset_name, metrics in dataset_metrics.items():
            if not metrics:
                continue
            current_absrel = metrics.get('absrel', float('inf'))
            if not isinstance(current_absrel, (int, float)) or current_absrel == float('inf'):
                continue

            prev_best = self.dataset_best_absrel.get(dataset_name, float('inf'))
            if current_absrel < prev_best:
                self.dataset_best_absrel[dataset_name] = current_absrel
                slug = self._slugify(dataset_name)
                suffix = f"best_absrel_dataset_{slug}"
                self.save(epoch, suffix)
                if self.rank == 0:
                    self.logger.info(f"New best dataset absrel for {dataset_name}: {current_absrel:.4f} at epoch {epoch}")

    @staticmethod
    def _slugify(name: str) -> str:
        sanitized = ''.join(ch.lower() if ch.isalnum() else '_' for ch in name)
        sanitized = sanitized.strip('_')
        return sanitized or "dataset"

    def log_saved_checkpoints(self) -> None:
        if self.rank != 0:
            return
        if not self._saved_checkpoints:
            self.logger.info("No checkpoints were saved during training.")
            return
        self.logger.info("Checkpoints saved（epoch为1-based）:")
        for suffix, epoch in self._saved_checkpoints.items():
            self.logger.info(f"  {suffix}: epoch {epoch}")
