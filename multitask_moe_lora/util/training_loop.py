import torch
import torch.distributed as dist
from typing import Any, Dict, Optional

from .validation import run_epoch_validation


def run_training_loop(
    config: Any,
    start_epoch: int,
    trainer: Any,
    checkpoint_manager: Any,
    train_depth_loader: torch.utils.data.DataLoader,
    train_seg_loader: torch.utils.data.DataLoader,
    val_depth_loader: torch.utils.data.DataLoader,
    val_seg_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    writer: Optional[torch.utils.tensorboard.SummaryWriter],
    logger: Any,
    rank: int,
    world_size: int,
) -> None:
    """Executes the main training loop."""
    logger.info(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, config.epochs):
        # Train one epoch
        avg_depth_loss, avg_seg_loss = trainer.train_epoch(train_depth_loader, train_seg_loader, epoch)
        train_losses = {"depth": avg_depth_loss, "seg": avg_seg_loss}

        # Step schedulers
        trainer.step_schedulers()
        trainer.log_learning_rates(epoch)

        depth_metrics_all: Optional[Dict[str, Any]] = None
        seg_metrics: Optional[Dict[str, Any]] = None

        # Perform validation at specified intervals
        if (epoch + 1) % config.val_interval == 0:
            depth_metrics_all, seg_metrics = run_epoch_validation(model, val_depth_loader, val_seg_loader, epoch, config, writer, logger)

            # In distributed training, handle metrics across processes
            if world_size > 1:
                if rank != 0:
                    depth_metrics_all = None
                    seg_metrics = None

        # Update and save the checkpoint
        checkpoint_manager.update_and_save(epoch, train_losses, depth_metrics_all, seg_metrics)

    logger.info("Training loop completed.")
