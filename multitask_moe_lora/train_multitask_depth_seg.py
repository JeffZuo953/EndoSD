#!/usr/bin/env python3
"""
多任务训练脚本：同时训练深度估计和语义分割
基于 DepthAnythingV2 backbone，使用 CacheDataset 加载 inhouse 数据
重构版本 - 使用模块化架构
"""

# 统一使用绝对导入，确保通过 `python -m multitask_moe_lora.train_multitask_depth_seg` 调用稳定
from multitask_moe_lora.util.config import parse_and_validate_config
from multitask_moe_lora.util.train_utils import setup_training_environment, cleanup_distributed, validate_training_setup
from multitask_moe_lora.util.data_utils import setup_dataloaders, log_batch_info, summarize_loader_composition
from multitask_moe_lora.util.model_setup import setup_complete_model
from multitask_moe_lora.util.trainer import MultiTaskTrainer
from multitask_moe_lora.util.validation import run_initial_evaluation, run_epoch_validation
from multitask_moe_lora.util.checkpoint_manager import CheckpointManager
from multitask_moe_lora.util.training_loop import run_training_loop
import os as _os
_tmpdir = _os.environ.get("TMPDIR") or "/data/ziyi/tmp"
try:
    _os.makedirs(_tmpdir, exist_ok=True)
except Exception:
    _tmpdir = None
else:
    _os.environ.setdefault("TMPDIR", _tmpdir)
_os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
import torch


def main():
    """主训练函数 - 协调整个训练流程"""
    if _os.environ.get("FM_DEBUG_MODE", "0") in {"1", "true", "True"}:
        torch.autograd.set_detect_anomaly(True)
        print("--- 警告: 已启用 autograd 异常检测 仅在调试时开启，会降低训练速度---")
    logger = None
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

        auto_disable_seg = train_seg_loader is None
        if config.disable_seg_head or auto_disable_seg:
            if auto_disable_seg and not config.disable_seg_head:
                config.disable_seg_head = True
                if rank == 0:
                    logger.info("No segmentation training dataset detected. Segmentation head disabled automatically.")
            elif config.disable_seg_head and rank == 0:
                logger.info("Segmentation head disabled via flag; dropping segmentation loaders.")
            train_seg_loader = None
            val_seg_loader = None

        if rank == 0:
            def _log_loader_summary(loader, label: str) -> None:
                if loader is None:
                    logger.info(f"{label}: <empty>")
                    return
                summaries = summarize_loader_composition(loader)
                if not summaries:
                    logger.info(f"{label}: <empty>")
                    return
                dataset_names = [entry['name'] for entry in summaries]
                total_samples = summaries[0]['total']
                logger.info(f"{label}: {', '.join(dataset_names)} (total {total_samples} samples)")

            _log_loader_summary(train_depth_loader, "Train-Depth datasets")
            _log_loader_summary(train_seg_loader, "Train-Seg datasets")
            _log_loader_summary(val_depth_loader, "Val-Depth datasets")
            _log_loader_summary(val_seg_loader, "Val-Seg datasets")
        
        # 6. 设置模型、优化器和调度器
        setup_bundle = setup_complete_model(config, logger)
        model = setup_bundle["model"]
        optimizer_depth = setup_bundle["optimizer_depth"]
        optimizer_seg = setup_bundle["optimizer_seg"]
        optimizer_camera = setup_bundle["optimizer_camera"]
        optimizer_unified = setup_bundle["optimizer_unified"]
        scheduler_depth = setup_bundle["scheduler_depth"]
        scheduler_seg = setup_bundle["scheduler_seg"]
        scheduler_camera = setup_bundle["scheduler_camera"]
        scheduler_unified = setup_bundle["scheduler_unified"]
        loss_weighter = setup_bundle["loss_weighter"]
        start_epoch = setup_bundle["start_epoch"]
        
        # 7. 创建训练器
        trainer = MultiTaskTrainer(
            model=model,
            optimizer_depth=optimizer_depth,
            optimizer_seg=optimizer_seg,
            optimizer_camera=optimizer_camera,
            optimizer_unified=optimizer_unified,
            scheduler_depth=scheduler_depth,
            scheduler_seg=scheduler_seg,
            scheduler_camera=scheduler_camera,
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
            optimizer_camera=optimizer_camera,
            optimizer_unified=optimizer_unified,
            scheduler_depth=scheduler_depth,
            scheduler_seg=scheduler_seg,
            scheduler_camera=scheduler_camera,
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
        if logger is not None:
            logger.error(f"Training failed with error: {str(e)}")
        else:
            print(f"[ERROR] Training failed before logger initialization: {e}")
        raise
    finally:
        # 11. 清理资源
        cleanup_distributed()


if __name__ == "__main__":
    main()
