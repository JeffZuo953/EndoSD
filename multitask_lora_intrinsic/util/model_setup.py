#!/usr/bin/env python3
"""
模型初始化模块
处理模型创建、预训练权重加载和分布式设置
"""

import os
import re
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
import logging
from typing import Tuple, Dict, Any, Optional

from .config import TrainingConfig
from ..depth_anything_v2.dpt_multitask import create_multitask_model
from .model_io import remap_checkpoint_keys


def load_weights_from_checkpoint(model: torch.nn.Module,
                                 optimizer_depth: Optional[torch.optim.Optimizer],
                                 optimizer_seg: Optional[torch.optim.Optimizer],
                                 scheduler_depth: Optional[torch.optim.lr_scheduler._LRScheduler],
                                 scheduler_seg: Optional[torch.optim.lr_scheduler._LRScheduler],
                                 config: TrainingConfig,
                                 logger: logging.Logger) -> int:
    """
    从检查点加载权重，并根据配置决定是否恢复完整训练状态。
    """
    if not config.resume_from or not os.path.isfile(config.resume_from):
        logger.info("No checkpoint specified, starting training from scratch.")
        return 0

    logger.info(f"Loading weights from checkpoint: {config.resume_from}")
    checkpoint = torch.load(config.resume_from, map_location='cpu')
    
    # 提取模型状态字典
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # First, remap keys for LoRA/MoE compatibility to standardize them
    state_dict = remap_checkpoint_keys(state_dict, model, config, logger)

    # Then, try to adapt positional embeddings on the remapped state_dict
    state_dict = _maybe_resize_pos_embed(model, state_dict, logger)
    
    # 过滤和加载权重
    _filter_and_load_weights(model, state_dict, config, logger)
    
    start_epoch = 0
    if config.resume_full_state:
        logger.info("Resuming full training state...")
        # 加载优化器状态
        if optimizer_depth and 'optimizer_depth_state_dict' in checkpoint:
            optimizer_depth.load_state_dict(checkpoint['optimizer_depth_state_dict'])
        if optimizer_seg and 'optimizer_seg_state_dict' in checkpoint:
            optimizer_seg.load_state_dict(checkpoint['optimizer_seg_state_dict'])
            
        # 加载调度器状态
        if scheduler_depth and 'scheduler_depth_state_dict' in checkpoint:
            scheduler_depth.load_state_dict(checkpoint['scheduler_depth_state_dict'])
        if scheduler_seg and 'scheduler_seg_state_dict' in checkpoint:
            scheduler_seg.load_state_dict(checkpoint['scheduler_seg_state_dict'])

        # 加载epoch
        start_epoch = checkpoint.get('epoch', -1) + 1
        logger.info(f"Successfully loaded full training state. Resuming from epoch {start_epoch}.")
    else:
        logger.info("Only model weights loaded. Optimizer, scheduler, and epoch are not restored.")
        # 验证和修复seg_head
        _validate_and_fix_seg_head(model, config, logger)

    return start_epoch


def _maybe_resize_pos_embed(model: torch.nn.Module,
                            state_dict: Dict[str, torch.Tensor],
                            logger: logging.Logger) -> Dict[str, torch.Tensor]:
    """
    If checkpoint pos_embed length doesn't match current model's, resize it with bicubic interpolation.
    Keeps the checkpoint's class token when dimensions match; otherwise falls back to model's class token.
    """
    try:
        actual_model = model.module if hasattr(model, 'module') else model
        
        # After remapping, the key should consistently be 'backbone.pos_embed'
        key_to_check = 'backbone.pos_embed'
        
        if key_to_check not in state_dict:
            return state_dict

        ckpt_pos = state_dict[key_to_check]
        model_pos = actual_model.backbone.pos_embed.detach().cpu()

        if ckpt_pos.shape == model_pos.shape:
            return state_dict  # perfectly matches

        logger.warning("Positional embedding size mismatch; resizing checkpoint pos_embed to current model.")
        logger.info(f"  ckpt pos_embed: {tuple(ckpt_pos.shape)} -> model pos_embed: {tuple(model_pos.shape)}")

        # Shapes: [1, 1+N, C]
        _, ckpt_tokens, dim = ckpt_pos.shape
        _, model_tokens, _ = model_pos.shape
        ckpt_grid = ckpt_tokens - 1
        model_grid = model_tokens - 1

        # Derive spatial sizes (assume square grid)
        import math
        s1 = int(round(math.sqrt(ckpt_grid)))
        s2 = int(round(math.sqrt(model_grid)))
        if s1 * s1 != ckpt_grid or s2 * s2 != model_grid:
            logger.warning(f"Non-square pos_embed detected (ckpt_grid={ckpt_grid}, model_grid={model_grid}). Skipping resize.")
            return state_dict

        # Split cls and patch embeddings
        ckpt_cls = ckpt_pos[:, :1]
        ckpt_patch = ckpt_pos[:, 1:]

        # [1, N, C] -> [1, C, H, W]
        ckpt_patch = ckpt_patch.transpose(1, 2).contiguous().view(1, dim, s1, s1).float()

        # Resize to target grid
        import torch.nn.functional as F
        resized = F.interpolate(ckpt_patch, size=(s2, s2), mode='bicubic', align_corners=False)

        # Back to [1, N, C]
        resized = resized.view(1, dim, s2 * s2).transpose(1, 2).contiguous()

        # Use checkpoint cls token if dim matches; else fall back to model's cls token
        if ckpt_cls.shape[-1] == dim:
            cls_token = ckpt_cls
        else:
            cls_token = model_pos[:, :1]

        new_pos = torch.cat([cls_token, resized], dim=1)
        # Ensure the resized pos_embed is stored under the model's key
        # Update the state dict in place
        state_dict[key_to_check] = new_pos
        logger.info("Resized checkpoint pos_embed successfully.")
    except Exception as e:
        logger.warning(f"Failed to resize pos_embed: {e}")
    return state_dict


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
        dinov3_repo_path=config.dinov3_repo_path,
        mode=config.mode,
        num_experts=config.num_experts,
        top_k=config.top_k,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha
    )
    
    # 检查是否在分布式环境中
    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
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
    else:
        # 非分布式环境（例如，评估脚本）
        logger.info("Not in a distributed environment. Loading model to cuda:0.")
        model.cuda()
    
    return model




def _check_encoder_compatibility(model: torch.nn.Module, 
                                state_dict: Dict[str, torch.Tensor], 
                                config: TrainingConfig, 
                                logger: logging.Logger) -> Dict[str, torch.Tensor]:
    """检查编码器兼容性"""
    actual_model = model.module if hasattr(model, 'module') else model
    if 'backbone.pos_embed' in state_dict:
        checkpoint_pos_embed_size = state_dict['backbone.pos_embed'].shape[1]
        current_pos_embed_size = actual_model.backbone.pos_embed.shape[1]
        
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
    actual_model = model.module if hasattr(model, 'module') else model
    logger.info(f"Current model backbone embed_dim: {actual_model.backbone.embed_dim}")
    logger.info(f"Current model encoder: {actual_model.encoder}")
    
    # 检查backbone embed_dim与预期是否一致
    expected_embed_dims = {'vits': 384, 'vitb': 768, 'vitl': 1024, 'vitg': 1536}
    expected_embed_dim = expected_embed_dims.get(config.encoder, 384)
    
    if actual_model.backbone.embed_dim != expected_embed_dim:
        logger.error(f"Model backbone embed_dim {actual_model.backbone.embed_dim} doesn't match expected {expected_embed_dim} for encoder {config.encoder}")
        logger.error(f"This indicates the model was incorrectly created. Recreating model...")
        
        # 重新创建模型
        model = create_multitask_model(
            encoder=config.encoder,
            num_classes=config.num_classes,
            max_depth=config.max_depth,
            frozen_backbone=False,
            seg_input_type=config.seg_input_type,
            mode="original",  # 使用原始模式重新创建模型
            num_experts=config.num_experts,
            top_k=config.top_k,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha
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
    actual_model = model.module if hasattr(model, 'module') else model
    model_state_dict = actual_model.state_dict()
    model_keys = set(model_state_dict.keys())
    
    # 需要排除的参数模式
    exclude_patterns = []

    # 打印前5个key用于调试
    logger.info("--- Checkpoint Keys (first 5) ---")
    for i, k in enumerate(state_dict.keys()):
        if i >= 5: break
        logger.info(f"  - {k}")
    
    logger.info("--- Model Keys (first 5) ---")
    for i, k in enumerate(model_keys):
        if i >= 5: break
        logger.info(f"  - {k}")
    logger.info("---------------------------------")
    
    # 过滤参数并加载
    filtered_state_dict = {}
    excluded_keys = []
    
    for k_ckpt, v in state_dict.items():
        target_k = None
        
        # 1. 直接匹配
        if k_ckpt in model_keys:
            target_k = k_ckpt
        
        # 2. 尝试移除 'module.' 前缀
        if target_k is None:
            k_no_module = re.sub(r'^module\.', '', k_ckpt)
            if k_no_module in model_keys:
                target_k = k_no_module
        
        # 3. 尝试为移除 'module.' 前缀后的 key 添加 'backbone.' 前缀
        if target_k is None:
            k_no_module = re.sub(r'^module\.', '', k_ckpt)
            potential_k = 'backbone.' + k_no_module
            if potential_k in model_keys:
                target_k = potential_k

        # 4. 仅尝试添加 'backbone.' 前缀
        if target_k is None:
            potential_k = 'backbone.' + k_ckpt
            if potential_k in model_keys:
                target_k = potential_k
        
        # 如果找到了匹配的 key
        if target_k:
            is_excluded = any(pattern in target_k for pattern in exclude_patterns)
            if not is_excluded:
                if v.shape == model_state_dict[target_k].shape:
                    filtered_state_dict[target_k] = v
                else:
                    logger.warning(f"Skipping {k_ckpt} -> {target_k} due to shape mismatch: "
                                   f"Checkpoint shape {v.shape}, Model shape {model_state_dict[target_k].shape}")
                    excluded_keys.append(k_ckpt)
            else:
                excluded_keys.append(k_ckpt)
        else:
            excluded_keys.append(k_ckpt)
    
    # 加载过滤后的状态字典
    actual_model.load_state_dict(filtered_state_dict, strict=False)
    
    # 记录加载结果
    _log_loading_results(filtered_state_dict, model_keys, excluded_keys, logger)


def _log_loading_results(filtered_state_dict: Dict[str, torch.Tensor],
                        model_keys: set,
                        excluded_keys: list,
                        logger: logging.Logger) -> None:
    """记录加载结果"""
    loaded_keys = set(filtered_state_dict.keys())
    new_keys = model_keys - loaded_keys
    
    logger.info(f"Successfully loaded {len(loaded_keys)} parameters.")
    
    # 过滤掉的、真正未被使用的key
    truly_skipped_keys = [k for k in excluded_keys if k not in loaded_keys]
    
    logger.info(f"Skipped {len(truly_skipped_keys)} parameters (non-matching or excluded).")
    if truly_skipped_keys:
        # 打印所有被跳过的键，以便调试
        logger.warning("--- Skipped Keys ---")
        for key in sorted(truly_skipped_keys):
            logger.warning(f"  - {key}")
        logger.warning("--------------------")

    logger.info(f"Initialized {len(new_keys)} new parameters (not found in checkpoint or skipped).")
    if new_keys:
        # 打印所有新初始化的键，以便调试
        logger.warning("--- New Initialized Keys ---")
        for key in sorted(list(new_keys)):
            logger.warning(f"  - {key}")
        logger.warning("--------------------------")


def _validate_and_fix_seg_head(model: torch.nn.Module, 
                              config: TrainingConfig, 
                              logger: logging.Logger) -> None:
    """验证和修复seg_head"""
    actual_model = model.module if hasattr(model, 'module') else model
    # 显式重置seg_head中BatchNorm层的running_mean和running_var
    if hasattr(actual_model.seg_head, 'bn') and isinstance(actual_model.seg_head.bn, torch.nn.modules.batchnorm.SyncBatchNorm):
        logger.info("Resetting seg_head's SyncBatchNorm running_mean and running_var.")
        actual_model.seg_head.bn.reset_running_stats()
    
    # 确保seg_head的cls_seg层有正确的输出通道数
    if hasattr(actual_model.seg_head, 'cls_seg'):
        expected_out_channels = config.num_classes
        actual_out_channels = actual_model.seg_head.cls_seg.out_channels
        
        logger.info(f"seg_head output channels check:")
        logger.info(f"  Expected: {expected_out_channels}")
        logger.info(f"  Actual: {actual_out_channels}")
        
        if actual_out_channels != expected_out_channels:
            logger.warning(f"seg_head output channels mismatch! Reinitializing cls_seg layer...")
            
            # 重新创建分类层，确保正确的输出通道数
            in_channels = actual_model.seg_head.cls_seg.in_channels
            actual_model.seg_head.cls_seg = torch.nn.Conv2d(in_channels, expected_out_channels, kernel_size=1).cuda()
            actual_model.seg_head.num_classes = expected_out_channels
            
            logger.info(f"Reinitialized cls_seg: {in_channels} -> {expected_out_channels} channels")


def setup_optimizers_and_schedulers(model: torch.nn.Module,
                                   config: TrainingConfig,
                                   logger: logging.Logger,
                                   loss_weighter: Any) -> Tuple[Optional[AdamW], Optional[AdamW], Optional[torch.optim.lr_scheduler.CosineAnnealingLR], Optional[torch.optim.lr_scheduler.CosineAnnealingLR], Optional[AdamW], Optional[torch.optim.lr_scheduler.CosineAnnealingLR]]:
    """
    设置优化器和学习率调度器
    - UWL策略: 返回一个统一的优化器和调度器
    - 其他策略: 返回两个独立的优化器和调度器
    """
    # ... (parameter freezing logic remains the same)

    if config.loss_weighting_strategy == 'uwl':
        logger.info("Using a unified optimizer for UWL strategy.")
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        trainable_params.append(loss_weighter.log_vars)
        
        optimizer = AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        return None, None, None, None, optimizer, scheduler
    else:
        logger.info("Using separate optimizers for depth and segmentation heads.")
        actual_model = model.module if hasattr(model, 'module') else model
        trainable_params_depth = list(actual_model.depth_head.parameters()) + list(actual_model.intrinsics_head.parameters())
        trainable_params_seg = list(actual_model.seg_head.parameters())

        for name, param in actual_model.backbone.named_parameters():
            if param.requires_grad:
                trainable_params_depth.append(param)
        
        # 为深度和分割头设置学习率，提供向后兼容性
        # Learning rates: fallback to base when overrides are None
        lr_depth_cfg = getattr(config, 'lr_depth', None)
        lr_seg_cfg = getattr(config, 'lr_seg', None)
        base_lr = float(getattr(config, 'lr', 1e-5))
        lr_depth = float(lr_depth_cfg) if (lr_depth_cfg is not None) else base_lr
        lr_seg = float(lr_seg_cfg) if (lr_seg_cfg is not None) else base_lr * 10.0
        logger.info(f"Using LR for depth: {lr_depth}, LR for seg: {lr_seg}")
        
        optimizer_depth = AdamW(trainable_params_depth, lr=lr_depth, weight_decay=config.weight_decay)
        optimizer_seg = AdamW(trainable_params_seg, lr=lr_seg, weight_decay=config.weight_decay)
        
        scheduler_depth = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_depth, T_max=config.epochs)
        scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=config.epochs)
        
        return optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, None, None


def setup_complete_model(config: TrainingConfig, logger: logging.Logger) -> Tuple[torch.nn.Module, Optional[AdamW], Optional[AdamW], Optional[torch.optim.lr_scheduler.CosineAnnealingLR], Optional[torch.optim.lr_scheduler.CosineAnnealingLR], Optional[AdamW], Optional[torch.optim.lr_scheduler.CosineAnnealingLR], Any, int]:
    """
    完整的模型设置流程
    """
    # 创建并设置模型
    model = create_and_setup_model(config, logger)
    
    # 设置优化器和调度器
    from .loss_weighter import LossWeighter
    loss_weighter = LossWeighter(config)
    optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, optimizer_unified, scheduler_unified = setup_optimizers_and_schedulers(model, config, logger, loss_weighter)
    
    # 从检查点加载权重
    # 注意：检查点加载逻辑需要根据优化器类型进行调整
    start_epoch = load_weights_from_checkpoint(
        model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, config, logger
    )
    
    return model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, optimizer_unified, scheduler_unified, loss_weighter, start_epoch
