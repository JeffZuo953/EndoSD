#!/usr/bin/env python3
"""
训练工具模块
包含环境设置、混合精度处理、检查点保存等通用工具函数
"""

import os
import sys
import random
import warnings
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any

from .config import TrainingConfig
from .utils import init_log
from .dist_helper import setup_distributed


# 混合精度兼容性处理
# 根据PyTorch版本选择正确的AMP API
try:
    # 优先使用新的 torch.amp API (PyTorch 1.10+)
    from torch.amp import GradScaler, autocast
    # torch.amp.autocast 需要 device_type 参数
    AUTOCAST_KWARGS = {'device_type': 'cuda'}
except ImportError:
    try:
        # 回退到旧的 torch.cuda.amp API
        from torch.cuda.amp import GradScaler, autocast
        # torch.cuda.amp.autocast 不接受 device_type 参数
        AUTOCAST_KWARGS = {}
    except ImportError:
        # 如果没有混合精度支持，创建一个dummy类
        class GradScaler:
            def __init__(self):
                pass

            def scale(self, loss):
                return loss

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        class autocast:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        AUTOCAST_KWARGS = {}


# 导出autocast供其他模块使用
__all__ = ['GradScaler', 'autocast', 'AUTOCAST_KWARGS', 'get_mixed_precision_components']


def setup_training_environment(config: TrainingConfig) -> tuple:
    """
    设置训练环境
    
    Args:
        config: 训练配置
        
    Returns:
        (rank, world_size, logger, writer)
    """
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    cudnn.deterministic = True
    cudnn.benchmark = False

    warnings.simplefilter("ignore")

    # 初始化日志
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    # 统一限制全局 logger 仅保留两个 handlers（step级、epoch级）
    if len(logger.handlers) > 2:
        extra_handlers = logger.handlers[2:]
        for handler in extra_handlers:
            handler.close()
        logger.handlers = logger.handlers[:2]

    # 分布式训练设置
    rank, world_size = setup_distributed(port=config.port)
    
    # 设置CUDA环境，确保每个进程绑定到正确的GPU
    setup_cuda_environment()

    # 创建保存目录
    os.makedirs(config.save_path, exist_ok=True)
    
    # 只有主进程创建TensorBoard writer和记录配置信息
    if rank == 0:
        # 创建TensorBoard writer
        writer = SummaryWriter(config.save_path)
        
        # 记录配置信息
        import pprint
        all_args = {**config.__dict__, "ngpus": world_size}
        logger.info("\n{}\n".format(pprint.pformat(all_args)))
        logger.info(f"Save path: {config.save_path}")
    else:
        # 非主进程使用dummy writer
        class DummyWriter:
            def add_scalar(self, *args, **kwargs): pass
            def add_image(self, *args, **kwargs): pass
            def add_histogram(self, *args, **kwargs): pass
            def close(self): pass
        writer = DummyWriter()

    # CUDA设置
    cudnn.enabled = True
    cudnn.benchmark = True

    return rank, world_size, logger, writer


def get_mixed_precision_components(use_mixed_precision: bool) -> tuple:
    """
    获取混合精度训练组件
    
    Args:
        use_mixed_precision: 是否使用混合精度
        
    Returns:
        (scaler, autocast_kwargs)
    """
    scaler = GradScaler() if use_mixed_precision else None
    return scaler, AUTOCAST_KWARGS


def save_checkpoint(model: torch.nn.Module,
                    optimizer_depth: Optional[torch.optim.Optimizer],
                    optimizer_seg: Optional[torch.optim.Optimizer],
                    optimizer_camera: Optional[torch.optim.Optimizer],
                    scheduler_depth: Optional[torch.optim.lr_scheduler._LRScheduler],
                    scheduler_seg: Optional[torch.optim.lr_scheduler._LRScheduler],
                    scheduler_camera: Optional[torch.optim.lr_scheduler._LRScheduler],
                    epoch: int,
                    save_path: str,
                    suffix: str,
                    rank: int = 0,
                    optimizer_unified: Optional[torch.optim.Optimizer] = None,
                    scheduler_unified: Optional[Any] = None) -> None:
    """
    保存训练检查点
    
    Args:
        model (torch.nn.Module): 模型
        optimizer_depth (torch.optim.Optimizer): 深度优化器
        optimizer_seg (torch.optim.Optimizer): 分割优化器
        scheduler_depth (torch.optim.lr_scheduler._LRScheduler): 深度学习率调度器
        scheduler_seg (torch.optim.lr_scheduler._LRScheduler): 分割学习率调度器
        epoch (int): 当前epoch
        save_path (str): 保存路径
        suffix (str): 检查点文件名后缀 (e.g., 'latest', 'best_absrel', 'epoch_100')
        rank (int): 进程rank，只有rank=0时才保存
    """
    if rank == 0:
        filter_seg = os.environ.get("FM_FILTER_SEG_HEAD", "0") == "1"

        full_state = model.module.state_dict()
        if filter_seg:
            filtered_state = {k: v for k, v in full_state.items() if not k.startswith("seg_head.")}
        else:
            filtered_state = full_state
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': filtered_state,
        }

        # 根据是否存在统一优化器，保存不同的优化器和调度器状态
        if optimizer_unified:
            checkpoint_data['optimizer_unified_state_dict'] = optimizer_unified.state_dict()
            if scheduler_unified:
                checkpoint_data['scheduler_unified_state_dict'] = scheduler_unified.state_dict()
        else:
            if optimizer_depth:
                checkpoint_data['optimizer_depth_state_dict'] = optimizer_depth.state_dict()
            if optimizer_seg:
                checkpoint_data['optimizer_seg_state_dict'] = optimizer_seg.state_dict()
            if optimizer_camera:
                checkpoint_data['optimizer_camera_state_dict'] = optimizer_camera.state_dict()
            if scheduler_depth:
                checkpoint_data['scheduler_depth_state_dict'] = scheduler_depth.state_dict()
            if scheduler_seg:
                checkpoint_data['scheduler_seg_state_dict'] = scheduler_seg.state_dict()
            if scheduler_camera:
                checkpoint_data['scheduler_camera_state_dict'] = scheduler_camera.state_dict()
        
        # 使用后缀构建文件名
        checkpoint_filename = f"checkpoint_{suffix}.pth"
        checkpoint_path = os.path.join(save_path, checkpoint_filename)
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path} (Epoch: {epoch})")


def cleanup_distributed() -> None:
    """清理分布式训练资源"""
    if dist.is_initialized():
        dist.destroy_process_group()


def log_batch_info(logger: logging.Logger, 
                  config: TrainingConfig) -> None:
    """
    记录批次大小信息
    
    Args:
        logger: 日志记录器
        config: 训练配置
    """
    seg_batch_size = config.seg_bs if config.seg_bs is not None else config.bs
    logger.info(f"Using batch sizes - Depth: {config.bs}, Segmentation: {seg_batch_size}")


def setup_cuda_environment() -> None:
    """设置CUDA环境"""
    if torch.cuda.is_available():
        # 设置CUDA设备
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        # 启用CUDA优化
        cudnn.enabled = True
        cudnn.benchmark = True
    else:
        raise RuntimeError("CUDA is not available, but this script requires GPU training")


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息
    
    Returns:
        设备信息字典
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        device_info['memory_allocated'] = torch.cuda.memory_allocated()
        device_info['memory_reserved'] = torch.cuda.memory_reserved()
        device_info['max_memory_allocated'] = torch.cuda.max_memory_allocated()
    
    return device_info


def log_device_info(logger: logging.Logger) -> None:
    """记录设备信息"""
    device_info = get_device_info()
    logger.info("Device Information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")


def clear_cuda_cache() -> None:
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_epoch_for_samplers(train_depth_loader, train_seg_loader, epoch: int) -> None:
    """
    为分布式采样器设置epoch
    
    Args:
        train_depth_loader: 深度训练数据加载器
        train_seg_loader: 分割训练数据加载器
        epoch: 当前epoch
    """
    if hasattr(train_depth_loader, 'sampler') and hasattr(train_depth_loader.sampler, 'set_epoch'):
        train_depth_loader.sampler.set_epoch(epoch)
    
    if hasattr(train_seg_loader, 'sampler') and hasattr(train_seg_loader.sampler, 'set_epoch'):
        train_seg_loader.sampler.set_epoch(epoch)


def log_training_progress(logger: logging.Logger, 
                         epoch: int, 
                         total_epochs: int,
                         task: str,
                         iteration: int,
                         total_iterations: int,
                         loss: float,
                         log_interval: int = 50) -> None:
    """
    记录训练进度
    
    Args:
        logger: 日志记录器
        epoch: 当前epoch
        total_epochs: 总epoch数
        task: 任务名称 ('depth' 或 'seg')
        iteration: 当前迭代
        total_iterations: 总迭代数
        loss: 当前损失
        log_interval: 日志记录间隔
    """
    if iteration % log_interval == 0:
        logger.info(f"  {task.capitalize()} Iter: {iteration}/{total_iterations}, Loss: {loss:.4f}")


def log_epoch_summary(logger: logging.Logger,
                     writer: SummaryWriter,
                     epoch: int,
                     avg_depth_loss: float,
                     avg_seg_loss: float) -> None:
    """
    记录epoch总结信息
    
    Args:
        logger: 日志记录器
        writer: TensorBoard writer
        epoch: 当前epoch
        avg_depth_loss: 平均深度损失
        avg_seg_loss: 平均分割损失
    """
    logger.info(f"Epoch {epoch} Summary:")
    logger.info(f"  Avg Depth Loss: {avg_depth_loss:.4f}")
    logger.info(f"  Avg Seg Loss: {avg_seg_loss:.4f}")
    
    # 记录到TensorBoard
    writer.add_scalar("train/depth_loss", avg_depth_loss, epoch)
    writer.add_scalar("train/seg_loss", avg_seg_loss, epoch)


def validate_training_setup(config: TrainingConfig) -> None:
    """
    验证训练设置
    
    Args:
        config: 训练配置
    """
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU training is required")
    
    # 检查分布式环境变量
    required_env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        print(f"Warning: Missing environment variables for distributed training: {missing_vars}")
        print("This might indicate single-GPU training or incorrect distributed setup")
    
    # 检查保存路径
    if not os.path.exists(os.path.dirname(config.save_path)) and os.path.dirname(config.save_path):
        raise ValueError(f"Parent directory of save_path does not exist: {os.path.dirname(config.save_path)}")
