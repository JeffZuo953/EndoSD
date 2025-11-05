#!/usr/bin/env python3
"""
多任务评估脚本：仅评估深度估计和语义分割性能
基于 DepthAnythingV2 backbone，使用 CacheDataset 加载 inhouse 数据
"""

# 添加父目录到 Python 路径，避免导入问题
import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入重构后的模块
from multitask.util.config import parse_and_validate_config, create_parser, args_to_config
from multitask.util.train_utils import setup_training_environment, cleanup_distributed, validate_training_setup
from multitask_moe_lora.util.data_utils import setup_dataloaders, log_batch_info
from multitask.util.model_setup import setup_complete_model, load_pretrained_weights
from multitask.util.validation import run_initial_evaluation
from multitask_moe_lora.util.model_io import load_weights_from_dict

def create_evaluation_parser():
    """创建评估专用参数解析器"""
    parser = create_parser()
    
    # 添加评估专用参数
    parser.add_argument("--checkpoint-root", type=str, help="Root directory containing checkpoint files")
    parser.add_argument("--start-epoch", type=int, default=0, help="Start epoch number")
    parser.add_argument("--end-epoch", type=int, default=100, help="End epoch number")
    
    return parser

def evaluate_checkpoint(model, val_depth_loader, val_seg_loader, config, writer, logger, checkpoint_path, epoch):
    """评估单个checkpoint"""
    try:
        # 加载checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            checkpoint_state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            checkpoint_state_dict = checkpoint['model']
        else:
            checkpoint_state_dict = checkpoint

        # 使用支持字典的权重加载函数
        load_weights_from_dict(model.module, checkpoint_state_dict, source_info=checkpoint_path)
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")

        # 运行评估
        logger.info(f"Running evaluation for epoch {epoch}")
        run_initial_evaluation(model, val_depth_loader, val_seg_loader, config, writer, logger)

    except Exception as e:
        logger.error(f"Failed to evaluate checkpoint {checkpoint_path}: {str(e)}")

def main():
    """主评估函数 - 协调整个评估流程"""
    try:
        # 1. 解析评估参数
        parser = create_evaluation_parser()
        args = parser.parse_args()
        config = args_to_config(args)
        
        # 2. 验证训练设置
        validate_training_setup(config)
        
        # 3. 设置训练环境
        rank, world_size, logger, writer = setup_training_environment(config)
        
        # 4. 记录批次大小信息
        log_batch_info(logger, config)
        
        # 5. 设置数据加载器
        train_depth_loader, train_seg_loader, val_depth_loader, val_seg_loader = setup_dataloaders(config)
        
        # 6. 设置模型
        model, optimizer_depth, optimizer_seg, scheduler_depth, scheduler_seg, optimizer_unified, scheduler_unified, loss_weighter, start_epoch = setup_complete_model(config, logger)
        
        # 7. 遍历指定范围内的epoch进行评估
        if args.checkpoint_root and rank == 0:
            logger.info(f"开始多任务评估，epoch范围: {args.start_epoch} - {args.end_epoch}")
            for epoch in range(args.start_epoch, args.end_epoch + 1):
                checkpoint_path = os.path.join(args.checkpoint_root, f"checkpoint_epoch_{epoch}.pth")
                
                # 检查checkpoint文件是否存在
                if os.path.exists(checkpoint_path):
                    logger.info("==============================================================================")
                    logger.info(f"评估 checkpoint: {checkpoint_path}")
                    logger.info("==============================================================================")
                    
                    # 评估单个checkpoint
                    evaluate_checkpoint(model, val_depth_loader, val_seg_loader, config, writer, logger, checkpoint_path, epoch)
                    
                    logger.info(f"完成评估 checkpoint: {checkpoint_path}")
                    logger.info("")
                else:
                    logger.info(f"跳过不存在的checkpoint: {checkpoint_path}")
            
            logger.info("所有评估完成!")
        
        # 8. 如果只指定了pretrained-from路径，则只评估该checkpoint
        elif config.pretrained_from and rank == 0:
            logger.info(f"评估单个checkpoint: {config.pretrained_from}")
            evaluate_checkpoint(model, val_depth_loader, val_seg_loader, config, writer, logger, config.pretrained_from, -1)
            logger.info("评估完成!")
        
        # 9. 关闭tensorboard writer
        if rank == 0:
            writer.close()
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise
    finally:
        # 10. 清理资源
        cleanup_distributed()


if __name__ == "__main__":
    main()