#!/usr/bin/env python3
"""
模型初始化模块
处理模型创建、预训练权重加载和分布式设置
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import logging
from typing import Tuple, Dict, Any, Optional

from .config import TrainingConfig
from ..depth_anything_v2.dpt_multitask import create_multitask_model


def load_checkpoint(model: torch.nn.Module,
                    optimizer_depth: torch.optim.Optimizer,
                    optimizer_seg: torch.optim.Optimizer,
                    scheduler_depth: torch.optim.lr_scheduler._LRScheduler,
                    scheduler_seg: torch.optim.lr_scheduler._LRScheduler,
                    config: TrainingConfig,
                    logger: logging.Logger) -> int:
    """
    加载完整的训练检查点
    
    Args:
        model: 模型
        optimizer_depth: 深度优化器
        optimizer_seg: 分割优化器
        scheduler_depth: 深度学习率调度器
        scheduler_seg: 分割学习率调度器
        config: 训练配置
        logger: 日志记录器
        
    Returns:
        start_epoch: 训练应开始的epoch
    """
    if not config.resume_from or not os.path.isfile(config.resume_from):
        logger.info("No checkpoint found to resume from, starting training from scratch.")
        return 0

    logger.info(f"Resuming training from checkpoint: {config.resume_from}")
    checkpoint = torch.load(config.resume_from, map_location='cpu')
    
    # 加载模型状态
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if 'optimizer_depth_state_dict' in checkpoint:
        optimizer_depth.load_state_dict(checkpoint['optimizer_depth_state_dict'])
    if 'optimizer_seg_state_dict' in checkpoint:
        optimizer_seg.load_state_dict(checkpoint['optimizer_seg_state_dict'])
        
    # 加载调度器状态 (可选)
    if 'scheduler_depth_state_dict' in checkpoint:
        scheduler_depth.load_state_dict(checkpoint['scheduler_depth_state_dict'])
    if 'scheduler_seg_state_dict' in checkpoint:
        scheduler_seg.load_state_dict(checkpoint['scheduler_seg_state_dict'])

    # 加载epoch
    start_epoch = checkpoint.get('epoch', -1) + 1
    
    logger.info(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}.")
    
    return start_epoch


def create_and_setup_model(config: TrainingConfig, logger: logging.Logger) -> torch.nn.Module:
    """
    创建并设置模型
    
    Args:
        config: 训练配置
        logger: 日志记录器
        
    Returns:
        设置好的模型
    """
    # 创建模型
    model = create_multitask_model(
        encoder=config.encoder, 
        num_classes=config.num_classes, 
        features=config.features, 
        max_depth=config.max_depth, 
        frozen_backbone=config.frozen_backbone, 
        seg_input_type=config.seg_input_type, 
        dinov3_repo_path=config.dinov3_repo_path
    )
    
    # 移动到GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    model.cuda(local_rank)
    
    # 启用SyncBatchNorm和DDP
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        find_unused_parameters=True, 
        broadcast_buffers=True
    )
    
    return model


def load_pretrained_weights(model: torch.nn.Module, 
                           config: TrainingConfig, 
                           logger: logging.Logger) -> None:
    """
    加载预训练权重
    
    Args:
        model: 模型
        config: 训练配置
        logger: 日志记录器
    """
    if not config.pretrained_from:
        logger.warning("No pretrained weights specified")
        return
    
    logger.info(f"Loading pretrained weights from: {config.pretrained_from}")
    checkpoint = torch.load(config.pretrained_from, map_location="cpu")
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 检查编码器类型兼容性
    _check_encoder_compatibility(model, state_dict, config, logger)
    
    # 检查backbone的实际配置
    _validate_backbone_config(model, config, logger)
    
    # 过滤和加载权重
    _filter_and_load_weights(model, state_dict, config, logger)
    
    # 验证和修复seg_head
    _validate_and_fix_seg_head(model, config, logger)


def _check_encoder_compatibility(model: torch.nn.Module, 
                                state_dict: Dict[str, torch.Tensor], 
                                config: TrainingConfig, 
                                logger: logging.Logger) -> Dict[str, torch.Tensor]:
    """检查编码器兼容性"""
    if 'backbone.pos_embed' in state_dict:
        checkpoint_pos_embed_size = state_dict['backbone.pos_embed'].shape[1]
        current_pos_embed_size = model.module.backbone.pos_embed.shape[1]
        
        logger.info(f"Encoder compatibility check:")
        logger.info(f"  Checkpoint pos_embed size: {checkpoint_pos_embed_size}")
        logger.info(f"  Current model pos_embed size: {current_pos_embed_size}")
        
        if checkpoint_pos_embed_size != current_pos_embed_size:
            logger.warning(f"Encoder type mismatch detected!")
            logger.warning(f"Checkpoint pos_embed size: {checkpoint_pos_embed_size}")
            logger.warning(f"Current model pos_embed size: {current_pos_embed_size}")
            logger.warning(f"Skipping backbone weights loading to avoid mismatch")
            
            # 过滤掉backbone相关的权重
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('backbone.'):
                    filtered_state_dict[k] = v
            state_dict = filtered_state_dict
            logger.info(f"Filtered out backbone weights. Remaining keys: {len(state_dict)}")
        else:
            logger.info(f"Encoder types match. Loading all compatible weights.")
    
    return state_dict


def _validate_backbone_config(model: torch.nn.Module, 
                             config: TrainingConfig, 
                             logger: logging.Logger) -> None:
    """验证backbone配置"""
    logger.info(f"Current model backbone embed_dim: {model.module.backbone.embed_dim}")
    logger.info(f"Current model encoder: {model.module.encoder}")
    
    # 检查backbone embed_dim与预期是否一致
    expected_embed_dims = {'vits': 384, 'vitb': 768, 'vitl': 1024, 'vitg': 1536}
    expected_embed_dim = expected_embed_dims.get(config.encoder, 384)
    
    if model.module.backbone.embed_dim != expected_embed_dim:
        logger.error(f"Model backbone embed_dim {model.module.backbone.embed_dim} doesn't match expected {expected_embed_dim} for encoder {config.encoder}")
        logger.error(f"This indicates the model was incorrectly created. Recreating model...")
        
        # 重新创建模型
        model = create_multitask_model(
            encoder=config.encoder, 
            num_classes=config.num_classes, 
            max_depth=config.max_depth, 
            frozen_backbone=False, 
            seg_input_type=config.seg_input_type
        )
        model.cuda()
        logger.info(f"Recreated model with correct encoder: {config.encoder}")
        logger.info(f"New model backbone embed_dim: {model.backbone.embed_dim}")


def _filter_and_load_weights(model: torch.nn.Module,
                            state_dict: Dict[str, torch.Tensor],
                            config: TrainingConfig,
                            logger: logging.Logger) -> None:
    """过滤并加载权重"""
    # 获取模型参数
    model_state_dict = model.module.state_dict()
    model_keys = set(model_state_dict.keys())
    
    # 需要排除的参数模式
    # 在评估模式下，我们不排除seg_head，允许加载其权重
    exclude_patterns = []
    
    # 过滤参数并加载
    filtered_state_dict = {}
    excluded_keys = []
    
    for k, v in state_dict.items():
        # 处理键名映射
        target_k = _map_parameter_key(k, model_state_dict, config)
        
        # 检查是否需要排除
        is_excluded = any(pattern in target_k for pattern in exclude_patterns)
        
        if target_k in model_state_dict and not is_excluded:
            filtered_state_dict[target_k] = v
        else:
            excluded_keys.append(k)
    
    # 加载过滤后的状态字典
    model.module.load_state_dict(filtered_state_dict, strict=False)
    
    # 记录加载结果
    _log_loading_results(filtered_state_dict, model_keys, excluded_keys, logger)


def _map_parameter_key(k: str, model_state_dict: Dict[str, torch.Tensor], config: TrainingConfig) -> str:
    """映射参数键名"""
    target_k = k
    
    # 处理各种前缀
    if k.startswith('module.pretrained.'):
        target_k = 'backbone.' + k[len('module.pretrained.'):]
    elif k.startswith('module.depth_head.'):
        target_k = k[len('module.'):]
    elif k.startswith('module.'):
        target_k = k[len('module.'):]
        if target_k.startswith('pretrained.'):
            target_k = 'backbone.' + target_k[len('pretrained.'):]
    elif k.startswith('pretrained.'):
        target_k = 'backbone.' + k[len('pretrained.'):]
    else:
        # 新增逻辑：处理无前缀的DINOv2 backbone权重
        # 尝试为不带任何已知前缀的键添加 'backbone.' 前缀
        potential_k = 'backbone.' + k
        if potential_k in model_state_dict:
            target_k = potential_k
        # DINOv3 weights are loaded directly, so no prefix modification is needed
        elif 'dinov3' in config.encoder:
            target_k = k
    
    return target_k


def _log_loading_results(filtered_state_dict: Dict[str, torch.Tensor],
                        model_keys: set,
                        excluded_keys: list,
                        logger: logging.Logger) -> None:
    """记录加载结果"""
    loaded_keys = list(filtered_state_dict.keys())
    new_keys = [k for k in model_keys if k not in loaded_keys]
    
    logger.info(f"Successfully loaded {len(loaded_keys)} parameters.")
    if loaded_keys:
        logger.info(f"Loaded parameters: {', '.join(loaded_keys[:5])}{'...' if len(loaded_keys) > 5 else ''}")
    
    logger.info(f"Skipped {len(excluded_keys)} parameters (non-matching keys).")
    if excluded_keys:
        logger.info(f"Skipped parameters: {', '.join(excluded_keys[:5])}{'...' if len(excluded_keys) > 5 else ''}")
    
    logger.info(f"Initialized {len(new_keys)} new parameters (not found in pretrained checkpoint).")
    if new_keys:
        logger.info(f"New parameters: {', '.join(new_keys[:5])}")


def _validate_and_fix_seg_head(model: torch.nn.Module, 
                              config: TrainingConfig, 
                              logger: logging.Logger) -> None:
    """验证和修复seg_head"""
    # 显式重置seg_head中BatchNorm层的running_mean和running_var
    if hasattr(model.module.seg_head, 'bn') and isinstance(model.module.seg_head.bn, torch.nn.modules.batchnorm.SyncBatchNorm):
        logger.info("Resetting seg_head's SyncBatchNorm running_mean and running_var.")
        model.module.seg_head.bn.reset_running_stats()
    
    # 确保seg_head的cls_seg层有正确的输出通道数
    if hasattr(model.module.seg_head, 'cls_seg'):
        expected_out_channels = config.num_classes
        actual_out_channels = model.module.seg_head.cls_seg.out_channels
        
        logger.info(f"seg_head output channels check:")
        logger.info(f"  Expected: {expected_out_channels}")
        logger.info(f"  Actual: {actual_out_channels}")
        
        if actual_out_channels != expected_out_channels:
            logger.warning(f"seg_head output channels mismatch! Reinitializing cls_seg layer...")
            
            # 重新创建分类层，确保正确的输出通道数
            in_channels = model.module.seg_head.cls_seg.in_channels
            model.module.seg_head.cls_seg = torch.nn.Conv2d(in_channels, expected_out_channels, kernel_size=1).cuda()
            model.module.seg_head.num_classes = expected_out_channels
            
            logger.info(f"Reinitialized cls_seg: {in_channels} -> {expected_out_channels} channels")


def setup_optimizers_and_schedulers(model: torch.nn.Module, 
                                   config: TrainingConfig, 
                                   logger: logging.Logger) -> Tuple[AdamW, AdamW, torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """
    设置优化器和学习率调度器
    
    Args:
        model: 模型
        config: 训练配置
        logger: 日志记录器
        
    Returns:
        (optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg)
    """
    # 确保backbone被冻结
    if config.frozen_backbone:
        logger.info("Backbone is frozen as per user request.")
        for param in model.module.backbone.parameters():
            param.requires_grad = False
    
    logger.info("Backbone is frozen. Creating separate optimizers for each head.")
    
    # 为深度和分割头设置学习率，提供向后兼容性
    lr_depth = getattr(config, 'lr_depth', config.lr)
    lr_seg = getattr(config, 'lr_seg', config.lr * 10)
    logger.info(f"Using LR for depth: {lr_depth}, LR for seg: {lr_seg}")

    # 为深度头创建优化器
    optimizer_depth = AdamW(model.module.depth_head.parameters(), lr=lr_depth, weight_decay=config.weight_decay)

    # 为分割头创建优化器
    optimizer_seg = AdamW(model.module.seg_head.parameters(), lr=lr_seg, weight_decay=config.weight_decay)
    
    # 为每个优化器创建学习率调度器
    scheduler_depth = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_depth, T_max=config.epochs)
    scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=config.epochs)
    
    return optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg


def setup_complete_model(config: TrainingConfig, logger: logging.Logger) -> Tuple[torch.nn.Module, AdamW, AdamW, torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.CosineAnnealingLR, int]:
    """
    完整的模型设置流程
    
    Args:
        config: 训练配置
        logger: 日志记录器
        
    Returns:
        (model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, start_epoch)
    """
    # 创建并设置模型
    model = create_and_setup_model(config, logger)
    
    # 设置优化器和调度器
    optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg = setup_optimizers_and_schedulers(model, config, logger)
    
    start_epoch = 0
    # 检查是否需要从检查点恢复
    if config.resume_from:
        start_epoch = load_checkpoint(model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, config, logger)
    # 否则，检查是否需要加载预训练权重
    elif config.pretrained_from:
        load_pretrained_weights(model, config, logger)
        
    return model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, start_epoch