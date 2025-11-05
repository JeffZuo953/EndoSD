#!/usr/bin/env python3
"""
多任务训练脚本：同时训练深度估计和语义分割
基于 DepthAnythingV2 backbone，使用 CacheDataset 加载 inhouse 数据
重构版本 - 使用模块化架构
"""

# 添加父目录到 Python 路径，避免导入问题
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入重构后的模块
from multitask.util.config import parse_and_validate_config
from multitask.util.train_utils import setup_training_environment, save_checkpoint, cleanup_distributed, validate_training_setup
from multitask.util.data_utils import setup_dataloaders, log_batch_info
from multitask.util.model_setup import setup_complete_model
from multitask.util.trainer import create_trainer
from multitask.util.validation import run_initial_evaluation, run_epoch_validation


def main():
    """主训练函数 - 协调整个训练流程"""
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
        model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, start_epoch = setup_complete_model(config, logger)
        
        # 7. 创建训练器
        trainer = create_trainer(
            model=model,
            optimizer_depth=optimizer_depth,
            optimizer_seg=optimizer_seg,
            scheduler_depth=scheduler_depth,
            scheduler_seg=scheduler_seg,
            config=config,
            logger=logger,
            writer=writer
        )
        
        # 8. 运行初始评估
        # val_seg_loader 现在是合并后的加载器
        run_initial_evaluation(model, val_depth_loader, val_seg_loader, config, writer, logger)
        
        # 9. 定义验证函数
        def validation_fn(model, val_depth_loader, val_seg_loader, epoch, config, writer, logger):
            # val_seg_loader 现在是合并后的加载器
            return run_epoch_validation(model, val_depth_loader, val_seg_loader, epoch, config, writer, logger)
        
        # 10. 运行训练循环
        logger.info(f"Starting training from epoch {start_epoch}")
        for epoch in range(start_epoch, config.epochs):
            # 训练一个epoch
            avg_depth_loss, avg_seg_loss = trainer.train_epoch(train_depth_loader, train_seg_loader, epoch)

            # 更新学习率
            trainer.step_schedulers()
            trainer.log_learning_rates(epoch)
            
            # 在每个epoch结束后进行验证
            if (epoch + 1) % config.val_interval == 0:
                validation_fn(model, val_depth_loader, val_seg_loader, epoch, config, writer, logger)
            
            # 保存检查点
            if (epoch + 1) % config.save_interval == 0:
                save_checkpoint(model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, epoch, config.save_path, rank)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # 11. 清理资源
        cleanup_distributed()


if __name__ == "__main__":
    main()
