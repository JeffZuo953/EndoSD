#!/usr//bin/env python3
"""
训练器基类
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
import logging
import os

from .config import TrainingConfig
from .train_utils import get_mixed_precision_components, set_epoch_for_samplers

class BaseTrainer:
    """训练器基类，包含通用功能"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 config: TrainingConfig,
                 logger: logging.Logger,
                 writer: SummaryWriter):
        """
        初始化 BaseTrainer
        
        Args:
            model: 模型
            config: 训练配置
            logger: 日志记录器
            writer: TensorBoard writer
        """
        self.model = model
        self.config = config
        self.logger = logger
        self.writer = writer
        
        self.rank = 0
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            self.rank = int(os.environ["SLURM_PROCID"])
            
        self.scaler, self.autocast_kwargs = get_mixed_precision_components(config.mixed_precision)

    def get_current_lr(self) -> tuple:
        """获取当前学习率"""
        raise NotImplementedError

    def step_schedulers(self) -> None:
        """更新学习率调度器"""
        raise NotImplementedError

    def log_learning_rates(self, epoch: int) -> None:
        """记录学习率"""
        if self.rank == 0:
            lrs = self.get_current_lr()
            if isinstance(lrs, tuple):
                depth_lr, seg_lr = lrs
                self.writer.add_scalar("train/depth_lr", depth_lr, epoch)
                self.writer.add_scalar("train/seg_lr", seg_lr, epoch)
                self.logger.info(f"Learning rates - Depth: {depth_lr:.2e}, Seg: {seg_lr:.2e}")
            else:
                self.writer.add_scalar("train/lr", lrs, epoch)
                self.logger.info(f"Learning rate: {lrs:.2e}")

    def train_epoch(self, *args, **kwargs):
        """训练一个epoch"""
        raise NotImplementedError