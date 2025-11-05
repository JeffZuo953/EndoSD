#!/usr/bin/env python3
"""
训练器模块
包含核心训练逻辑、优化器管理和训练循环
"""

import torch
import torch.nn.functional as F
import torch.nn.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import logging
from typing import Optional

from .config import TrainingConfig
from .loss import SiLogLoss
from .train_utils import (
    get_mixed_precision_components,
    clear_cuda_cache,
    log_training_progress,
    set_epoch_for_samplers,
    autocast
)


class MultiTaskTrainer:
    """多任务训练器"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer_depth: AdamW,
                 optimizer_seg: AdamW,
                 scheduler_depth: torch.optim.lr_scheduler.CosineAnnealingLR,
                 scheduler_seg: torch.optim.lr_scheduler.CosineAnnealingLR,
                 config: TrainingConfig,
                 logger: logging.Logger,
                 writer: SummaryWriter):
        """
        初始化训练器
        
        Args:
            model: 多任务模型
            optimizer_depth: 深度优化器
            optimizer_seg: 分割优化器
            scheduler_depth: 深度学习率调度器
            scheduler_seg: 分割学习率调度器
            config: 训练配置
            logger: 日志记录器
            writer: TensorBoard writer
        """
        import os
        
        self.model = model
        self.optimizer_depth = optimizer_depth
        self.optimizer_seg = optimizer_seg
        self.scheduler_depth = scheduler_depth
        self.scheduler_seg = scheduler_seg
        self.config = config
        self.logger = logger
        self.writer = writer
        
        # 检查分布式训练的rank
        self.rank = 0
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            self.rank = int(os.environ["SLURM_PROCID"])
        
        # 损失函数
        self.depth_criterion = SiLogLoss().cuda()
        self.seg_criterion = CrossEntropyLoss(ignore_index=255).cuda()
        
        # 混合精度组件
        self.scaler, self.autocast_kwargs = get_mixed_precision_components(config.mixed_precision)
    
    def train_depth_epoch(self, train_depth_loader: DataLoader, epoch: int) -> float:
        """
        训练深度估计头一个epoch
        
        Args:
            train_depth_loader: 深度训练数据加载器
            epoch: 当前epoch
            
        Returns:
            平均深度损失
        """
        self.logger.info("--- Training Depth Head ---")
        total_depth_loss = 0
        
        for i, batch in enumerate(train_depth_loader):
            self.optimizer_depth.zero_grad()
            
            input_img = batch["image"].cuda()
            depth_gt = batch["depth"].cuda()
            
            # 使用混合精度训练
            with autocast(enabled=self.config.mixed_precision, **self.autocast_kwargs):
                outputs = self.model(input_img, task='depth')
                depth_pred = outputs['depth']
                
                valid_mask = (depth_gt > 0) & (depth_gt >= self.config.min_depth) & (depth_gt <= self.config.max_depth)
                loss = self.depth_criterion(depth_pred, depth_gt, valid_mask)
            
            # 反向传播和优化
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer_depth)
                torch.nn.utils.clip_grad_norm_(self.model.module.depth_head.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer_depth)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.module.depth_head.parameters(), max_norm=1.0)
                self.optimizer_depth.step()
            
            total_depth_loss += loss.item()
            
            # 记录训练进度
            log_training_progress(self.logger, epoch, self.config.epochs, 'depth', 
                                i, len(train_depth_loader), loss.item())
        
        avg_depth_loss = total_depth_loss / len(train_depth_loader) if len(train_depth_loader) > 0 else 0
        return avg_depth_loss
    
    def train_seg_epoch(self, train_seg_loader: DataLoader, epoch: int) -> float:
        """
        训练语义分割头一个epoch（内存优化版本）
        
        Args:
            train_seg_loader: 分割训练数据加载器
            epoch: 当前epoch
            
        Returns:
            平均分割损失
        """
        self.logger.info("--- Training Segmentation Head ---")
        total_seg_loss = 0
        
        for i, batch in enumerate(train_seg_loader):
            self.optimizer_seg.zero_grad()
            
            input_img = batch["image"].cuda()
            seg_gt = batch["semseg_mask"].cuda()
            
            # 内存优化策略：
            # 1. 使用 torch.cuda.empty_cache() 清理缓存
            # 2. 使用梯度检查点减少内存使用
            # 3. 对于 from_depth 模式，优化特征提取
            
            clear_cuda_cache()  # 清理GPU缓存
            
            # 分割头训练使用 FP32 以避免 GradScaler 状态管理问题
            # 对于 from_depth 模式，使用梯度检查点
            if self.config.seg_input_type == 'from_depth':
                # 使用梯度检查点来减少内存使用
                def seg_forward_func(x):
                    return self.model(x, task='seg')
                
                outputs = torch.utils.checkpoint.checkpoint(seg_forward_func, input_img, use_reentrant=False)
            else:
                outputs = self.model(input_img, task='seg')
            
            seg_pred = outputs['seg']
            
            # 确保标签在有效范围内
            ignore_idx = 255
            valid_class_mask = (seg_gt >= 0) & (seg_gt < self.config.num_classes)
            ignore_mask = (seg_gt == ignore_idx)
            valid_mask = valid_class_mask | ignore_mask
            seg_gt = torch.where(valid_mask, seg_gt, torch.tensor(ignore_idx, device=seg_gt.device))
            
            loss = self.seg_criterion(seg_pred, seg_gt)
            
            # 分割头始终使用 FP32 训练
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.module.seg_head.parameters(), max_norm=1.0)
            self.optimizer_seg.step()
            
            # 清理中间变量和缓存
            del outputs, seg_pred
            clear_cuda_cache()
            
            total_seg_loss += loss.item()
            
            # 记录训练进度
            log_training_progress(self.logger, epoch, self.config.epochs, 'seg', 
                                i, len(train_seg_loader), loss.item())
        
        avg_seg_loss = total_seg_loss / len(train_seg_loader) if len(train_seg_loader) > 0 else 0
        return avg_seg_loss
    
    def train_epoch(self,
                   train_depth_loader: DataLoader,
                   train_seg_loader: DataLoader,
                   epoch: int) -> tuple:
        """
        训练一个完整的epoch
        
        Args:
            train_depth_loader: 深度训练数据加载器
            train_seg_loader: 分割训练数据加载器
            epoch: 当前epoch
            
        Returns:
            (avg_depth_loss, avg_seg_loss)
        """
        # 设置sampler的epoch，确保数据在不同epoch中被正确打乱
        set_epoch_for_samplers(train_depth_loader, train_seg_loader, epoch)
        self.logger.info(f"===========> Epoch: {epoch}/{self.config.epochs}")
        
        self.model.train()
        
        # 训练深度估计头
        avg_depth_loss = self.train_depth_epoch(train_depth_loader, epoch)
        
        # 训练语义分割头
        avg_seg_loss = self.train_seg_epoch(train_seg_loader, epoch)
        
        # 只有主进程记录损失到TensorBoard和输出日志
        if self.rank == 0:
            # 记录损失到TensorBoard
            self.writer.add_scalar("train/depth_loss", avg_depth_loss, epoch)
            self.writer.add_scalar("train/seg_loss", avg_seg_loss, epoch)
            
            # 记录epoch总结
            self.logger.info(f"Epoch {epoch} - Avg Depth Loss: {avg_depth_loss:.4f}")
            self.logger.info(f"Epoch {epoch} - Avg Seg Loss: {avg_seg_loss:.4f}")
        
        return avg_depth_loss, avg_seg_loss
    
    def step_schedulers(self) -> None:
        """更新学习率调度器"""
        self.scheduler_depth.step()
        self.scheduler_seg.step()
    
    def get_current_lr(self) -> tuple:
        """
        获取当前学习率
        
        Returns:
            (depth_lr, seg_lr)
        """
        depth_lr = self.scheduler_depth.get_last_lr()[0]
        seg_lr = self.scheduler_seg.get_last_lr()[0]
        return depth_lr, seg_lr
    
    def log_learning_rates(self, epoch: int) -> None:
        """记录学习率"""
        if self.rank == 0:  # 只有主进程记录学习率
            depth_lr, seg_lr = self.get_current_lr()
            self.writer.add_scalar("train/depth_lr", depth_lr, epoch)
            self.writer.add_scalar("train/seg_lr", seg_lr, epoch)
            self.logger.info(f"Learning rates - Depth: {depth_lr:.2e}, Seg: {seg_lr:.2e}")


def create_trainer(model: torch.nn.Module,
                  optimizer_depth: AdamW,
                  optimizer_seg: AdamW,
                  scheduler_depth: torch.optim.lr_scheduler.CosineAnnealingLR,
                  scheduler_seg: torch.optim.lr_scheduler.CosineAnnealingLR,
                  config: TrainingConfig,
                  logger: logging.Logger,
                  writer: SummaryWriter) -> MultiTaskTrainer:
    """
    创建训练器实例
    
    Args:
        model: 多任务模型
        optimizer_depth: 深度优化器
        optimizer_seg: 分割优化器
        scheduler_depth: 深度学习率调度器
        scheduler_seg: 分割学习率调度器
        config: 训练配置
        logger: 日志记录器
        writer: TensorBoard writer
        
    Returns:
        训练器实例
    """
    return MultiTaskTrainer(
        model=model,
        optimizer_depth=optimizer_depth,
        optimizer_seg=optimizer_seg,
        scheduler_depth=scheduler_depth,
        scheduler_seg=scheduler_seg,
        config=config,
        logger=logger,
        writer=writer
    )


def run_training_loop(trainer: MultiTaskTrainer,
                     train_depth_loader: DataLoader,
                     train_seg_loader: DataLoader,
                     val_depth_loader: DataLoader,
                     val_seg_loader: DataLoader,
                     config: TrainingConfig,
                     validation_fn) -> None:
    """
    运行完整的训练循环
    
    Args:
        trainer: 训练器
        train_depth_loader: 深度训练数据加载器
        train_seg_loader: 分割训练数据加载器
        val_depth_loader: 深度验证数据加载器
        val_seg_loader: 分割验证数据加载器
        config: 训练配置
        validation_fn: 验证函数
    """
    for epoch in range(config.epochs):
        # 训练一个epoch
        avg_depth_loss, avg_seg_loss = trainer.train_epoch(train_depth_loader, train_seg_loader, epoch)
        
        # 更新学习率
        trainer.step_schedulers()
        trainer.log_learning_rates(epoch)
        
        # 在每个epoch结束后进行验证
        validation_fn(trainer.model, val_depth_loader, val_seg_loader, epoch, config, trainer.writer, trainer.logger)