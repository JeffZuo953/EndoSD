#!/usr/bin/env python3
"""
多任务训练脚本：同时训练深度估计和语义分割
基于 DepthAnythingV2 backbone，使用 CacheDataset 加载 inhouse 数据
重构版本 - 使用模块化架构
"""

# 添加父目录到 Python 路径，避免导入问题
# 导入重构后的模块
from .util.config import parse_and_validate_config
from .util.train_utils import setup_training_environment, cleanup_distributed, validate_training_setup
from .util.data_utils import setup_dataloaders, log_batch_info
from .util.model_setup import setup_complete_model
from .util.trainer import MultiTaskTrainer
from .util.validation import run_initial_evaluation, run_epoch_validation
from .util.checkpoint_manager import CheckpointManager
from .util.training_loop import run_training_loop
import torch


def main():
    """主训练函数 - 协调整个训练流程"""
    torch.autograd.set_detect_anomaly(True)
    try:
        # 1. 解析和验证配置
        config = parse_and_validate_config()
        
        # 2. 验证训练设置
        validate_training_setup(config)
        
        # 3. 设置训练环境
        rank, world_size, logger, writer = setup_training_environment(config)
        
        # 4. 记录批次大小信息
        log_batch_info(logger, config)
        
        # 5. 设置数据加载器
        train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader = setup_dataloaders(config)
        
        # 6. 设置模型、优化器和调度器
        model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, optimizer_unified, scheduler_unified, loss_weighter, start_epoch = setup_complete_model(config, logger)
        
        # 7. 创建训练器
        trainer = MultiTaskTrainer(
            model=model,
            optimizer_depth=optimizer_depth,
            optimizer_seg=optimizer_seg,
            optimizer_unified=optimizer_unified,
            scheduler_depth=scheduler_depth,
            scheduler_seg=scheduler_seg,
            scheduler_unified=scheduler_unified,
            loss_weighter=loss_weighter,
            config=config,
            logger=logger,
            writer=writer,
            rank=rank
        )
        
        # 8. 创建检查点管理器
        checkpoint_manager = CheckpointManager(
            model=model,
            optimizer_depth=optimizer_depth,
            optimizer_seg=optimizer_seg,
            optimizer_unified=optimizer_unified,
            scheduler_depth=scheduler_depth,
            scheduler_seg=scheduler_seg,
            scheduler_unified=scheduler_unified,
            config=config,
            logger=logger,
            rank=rank
        )
        
        # 9. 运行初始评估
        run_initial_evaluation(model, val_depth_loader, val_seg_loader, config, writer, logger)
        
        # 10. 运行训练循环
        run_training_loop(
            config=config,
            start_epoch=start_epoch,
            trainer=trainer,
            checkpoint_manager=checkpoint_manager,
            train_depth_loader=train_depth_loader,
            train_seg_loader=train_seg_loader,
            val_depth_loader=val_depth_loader,
            val_seg_loader=val_seg_loader,
            model=model,
            writer=writer,
            logger=logger,
            rank=rank,
            world_size=world_size,
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # 11. 清理资源
        cleanup_distributed()


if __name__ == "__main__":
    main()
