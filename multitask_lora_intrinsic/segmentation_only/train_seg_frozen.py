# 添加父目录到 Python 路径，避免代码重复
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import pprint
import random
import warnings
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from dataset.inhouse_seg import InhouseSegDataset
from dataset.cache_utils import SegCacheDataset
from depth_anything_v2.dpt_seg_frozen import DepthAnythingV2_Seg_Frozen
from util.metric import SegMetric
from torch.nn import CrossEntropyLoss
from util.dist_helper import setup_distributed
from util.utils import init_log
from torch.utils.data.dataloader import default_collate

parser = argparse.ArgumentParser(description="Depth Anything V2 Frozen Encoder Training for Semantic Segmentation")

parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])  # Changed to ViT-Small
parser.add_argument(
    "--dataset",
    default="inhouse",
    choices=["inhouse"],
)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--epochs", default=20, type=int)  # 减少默认 epochs
parser.add_argument("--target-epochs", default=None, type=int, help="Target epochs for LR scheduling (if different from actual training epochs)")
parser.add_argument("--bs", default=1, type=int)  # 减少批处理大小
parser.add_argument("--lr", default=None, type=float)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--seg-head-type", default="BNHead", choices=["BNHead"])
parser.add_argument("--seg-input-type", default="last_four", choices=["last", "last_four", "from_depth"], help="Input type for segmentation head ('last', 'last_four', 'from_depth')")
parser.add_argument(
    "--pretrained-from",
    type=str,
    help="Path to the FULL DINOv2 pretrained weights (.pth)",
)
parser.add_argument("--seg-head-weights", type=str, default=None, help="Path to pretrained segmentation head weights (.pth)")
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)


def collate_fn_pad(batch):
    # 找出批次中图像和掩码的最大尺寸
    max_h = 0
    max_w = 0
    for item in batch:
        max_h = max(max_h, item["image"].shape[-2])
        max_w = max(max_w, item["image"].shape[-1])

    stride = 14
    if max_h % stride != 0:
        max_h = max_h + (stride - max_h % stride)
    if max_w % stride != 0:
        max_w = max_w + (stride - max_w % stride)

    images = []
    masks = []

    for item in batch:
        image = item["image"]
        mask = item["semseg_mask"]

        h, w = image.shape[-2:]

        pad_h = max_h - h
        pad_w = max_w - w

        padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0)
        images.append(padded_image)

        padded_mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=255)  # 255 is ignore_index
        masks.append(padded_mask)

    images_batch = torch.stack(images)
    masks_batch = torch.stack(masks)

    return {
        "image": images_batch,
        "semseg_mask": masks_batch,
    }


def main():
    args = parser.parse_args()

    # Determine project root based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    warnings.simplefilter("ignore")

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        # Ensure save_path is created correctly (it might be relative like ../exp/...)
        os.makedirs(args.save_path, exist_ok=True)
        all_args = {
            **vars(args),
            "ngpus": world_size,
            "project_root": project_root,
        }  # Add project_root to log
        logger.info("\n{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        logger.info(f"Save path: {args.save_path}")

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    # Construct dataset paths relative to project root
    if args.dataset == "inhouse":
        train_split_path = "/media/ExtHDD1/jianfu/data/seg_inhouse/cache/train_cache.txt"
        val_split_path = "/media/ExtHDD1/jianfu/data/seg_inhouse/cache/val_cache.txt"
        trainset = SegCacheDataset(filelist_path=train_split_path)
        valset = SegCacheDataset(filelist_path=val_split_path)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset,
        batch_size=args.bs,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler,
        collate_fn=collate_fn_pad,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        sampler=valsampler,
        collate_fn=default_collate if args.bs == 1 else collate_fn_pad,
    )

    local_rank = int(os.environ["LOCAL_RANK"])

    # --- Model Definition ---
    model_configs = {
        "vits": {
            "features": 64,
            # "out_channels": [384],
            "out_channels": [48, 96, 192, 384]
        },
        "vitb": {
            "features": 128,
            "out_channels": [96, 192, 384, 768]
        },
        "vitl": {
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        },
        "vitg": {
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536]
        },
    }
    model_config = model_configs[args.encoder]

    model = DepthAnythingV2_Seg_Frozen(
        encoder=args.encoder,
        num_classes=args.num_classes,
        features=model_config["features"],
        seg_head_type=args.seg_head_type,
        seg_input_type=args.seg_input_type,
    )

    # --- Model Setup for DDP ---
    # Move model to GPU *before* wrapping
    model.cuda(local_rank)

    # Convert BatchNorm layers *before* wrapping
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # --- Weight Loading (AFTER DDP wrapping) ---
    if args.pretrained_from:
        # Check if pretrained path is absolute or needs resolving relative to project root
        if not os.path.isabs(args.pretrained_from):
            # Adjust relative path assuming it's passed relative to metric_depth
            pretrained_path = os.path.abspath(os.path.join(script_dir, args.pretrained_from))
            logger.info(f"Resolving relative pretrained path: {args.pretrained_from} -> {pretrained_path}")
        else:
            pretrained_path = args.pretrained_from

        logger.info(f"Loading pretrained weights from: {pretrained_path}")
        if not os.path.exists(pretrained_path):
            logger.error(f"Pretrained file not found at resolved path: {pretrained_path}")
            raise FileNotFoundError(f"Pretrained file not found: {pretrained_path}")

        # Load checkpoint to CPU first to avoid GPU memory issues on rank 0
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 获取模型参数
        model_state_dict = model.state_dict()
        
        # 过滤并加载backbone权重
        filtered_state_dict = {}
        excluded_keys = []
        
        # 获取模型状态字典以进行检查
        model_state_dict = model.state_dict()
        
        for k, v in state_dict.items():
            # 默认目标键为原始键
            target_k = k
            
            # 移除 'module.' 前缀（如果存在）
            if k.startswith('module.'):
                k_no_module = k[len('module.'):]
            else:
                k_no_module = k

            # 移除 'pretrained.' 或 'backbone.' 前缀
            if k_no_module.startswith('pretrained.'):
                k_core = k_no_module[len('pretrained.'):]
            elif k_no_module.startswith('backbone.'):
                k_core = k_no_module[len('backbone.'):]
            else:
                k_core = k_no_module

            # 构建最终的目标键
            # 1. 始终以 'module.backbone.' 开头，因为模型被DDP包装
            # 2. 附加核心键名
            final_target_k = f"module.backbone.{k_core}"

            # 检查最终键是否存在于模型中
            if final_target_k in model_state_dict:
                filtered_state_dict[final_target_k] = v
            else:
                excluded_keys.append(k)
        
        # 加载过滤后的状态字典
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

        # 记录加载结果
        if rank == 0:
            loaded_keys = list(filtered_state_dict.keys())
            logger.info(f"Successfully loaded {len(loaded_keys)} parameters for backbone.")
            if loaded_keys:
                logger.info(f"  Loaded parameters sample: {', '.join(loaded_keys[:5])}{'...' if len(loaded_keys) > 5 else ''}")
            
            # 过滤掉不是模型一部分的意外键
            unexpected_keys_in_model = [k for k in unexpected_keys if k in model_state_dict]
            if unexpected_keys_in_model:
                logger.warning(f"Found {len(unexpected_keys_in_model)} unexpected keys in the model state_dict that were not in the checkpoint.")
                logger.warning(f"  Unexpected keys sample: {', '.join(unexpected_keys_in_model[:5])}{'...' if len(unexpected_keys_in_model) > 5 else ''}")

            # 报告模型中缺失的键 (主要是seg_head)
            missing_in_checkpoint = [k for k in missing_keys if k.startswith('seg_head.')]
            if missing_in_checkpoint:
                logger.info(f"Initialized {len(missing_in_checkpoint)} new parameters for seg_head (not found in pretrained checkpoint).")
                logger.info(f"  New seg_head parameters sample: {', '.join(missing_in_checkpoint[:5])}{'...' if len(missing_in_checkpoint) > 5 else ''}")
            
            # 报告真正缺失的backbone键 (如果有的话，说明出错了)
            missing_backbone_keys = [k for k in missing_keys if k.startswith('backbone.')]
            if missing_backbone_keys:
                logger.error(f"CRITICAL: Missing {len(missing_backbone_keys)} backbone keys during loading!")
                logger.error(f"  Missing backbone keys sample: {', '.join(missing_backbone_keys[:20])}{'...' if len(missing_backbone_keys) > 20 else ''}")

            if excluded_keys:
                logger.info(f"Skipped {len(excluded_keys)} parameters from checkpoint (e.g., depth_head, non-matching keys).")
                logger.info(f"  Skipped keys sample: {', '.join(excluded_keys[:5])}{'...' if len(excluded_keys) > 5 else ''}")
        
        # --- Load Seg Head Weights (if provided) ---
        if args.seg_head_weights:
            if not os.path.exists(args.seg_head_weights):
                logger.error(f"Seg head weights file not found: {args.seg_head_weights}")
                raise FileNotFoundError(f"Seg head weights file not found: {args.seg_head_weights}")

            logger.info(f"Loading pretrained seg head weights from: {args.seg_head_weights}")
            seg_head_checkpoint = torch.load(args.seg_head_weights, map_location="cpu")
            
            # Extract state dict
            if 'model_state_dict' in seg_head_checkpoint:
                seg_head_state_dict = seg_head_checkpoint['model_state_dict']
            elif 'model' in seg_head_checkpoint:
                seg_head_state_dict = seg_head_checkpoint['model']
            elif 'state_dict' in seg_head_checkpoint:
                seg_head_state_dict = seg_head_checkpoint['state_dict']
            else:
                seg_head_state_dict = seg_head_checkpoint

            # Remap keys from 'decode_head' to 'seg_head'
            remapped_seg_head_dict = {}
            model_state_dict = model.state_dict()
            
            for k, v in seg_head_state_dict.items():
                target_k = None
                if k.startswith('decode_head.'):
                    k_remap = k.replace('decode_head.', 'seg_head.')
                    if 'conv_seg' in k_remap:
                        target_k = k_remap.replace('conv_seg', 'cls_seg')
                    else:
                        target_k = k_remap
                elif 'conv_seg' in k:
                    target_k = k.replace('conv_seg', 'cls_seg')
                elif k.startswith('module.decode_head.'):
                    k_remap = k.replace('module.decode_head.', 'seg_head.')
                    if 'conv_seg' in k_remap:
                        target_k = k_remap.replace('conv_seg', 'cls_seg')
                    else:
                        target_k = k_remap

                if target_k:
                    # Add prefix for DDP
                    ddp_target_k = f"module.{target_k}"
                    if ddp_target_k in model_state_dict:
                        target_k = ddp_target_k

                    if target_k in model_state_dict:
                        # Check for shape mismatch before adding to the dict
                        if v.shape == model_state_dict[target_k].shape:
                            remapped_seg_head_dict[target_k] = v
                            if rank == 0:
                                logger.info(f"  - Remapping and loading key: {k} -> {target_k}")
                        else:
                            if rank == 0:
                                logger.warning(f"  - Skipped key due to shape mismatch: {target_k} (checkpoint {v.shape} vs model {model_state_dict[target_k].shape})")
                    else:
                        if rank == 0:
                            logger.warning(f"  - Skipped key (not in model): {target_k}")

            if remapped_seg_head_dict:
                model.load_state_dict(remapped_seg_head_dict, strict=False)
                if rank == 0:
                    logger.info(f"Successfully loaded {len(remapped_seg_head_dict)} parameters for seg_head.")
            else:
                if rank == 0:
                    logger.warning("Could not find any matching seg_head weights to load from the provided file.")

    # --- Criterion and Optimizer Setup ---
    # 默认使用简单的交叉熵损失
    criterion = CrossEntropyLoss(ignore_index=255).cuda(0)


    # Now setup optimizer, accessing parameters via model.module for DDP
    seg_head_params = []
    for name, param in model.module.seg_head.named_parameters():
        if param.requires_grad:
            seg_head_params.append(param)
            if rank == 0:
                logger.debug(f"Adding Seg head param to optimizer: {name}")
        else:
            if rank == 0:
                logger.warning(f"Seg head param {name} does not require grad.")

    if not seg_head_params:
        logger.error("No trainable parameters found in the Seg head!")

    optimizer = AdamW(seg_head_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    previous_best = {
        "Mean IoU": 0.0,
    }

    for epoch in range(args.epochs):
        if rank == 0:
            logger.info(f"===========> Epoch: {epoch}/{args.epochs}")
            log_metrics = {k: f"{v:.3f}" for k, v in previous_best.items()}
            logger.info(f"Previous best: {log_metrics}")

        trainloader.sampler.set_epoch(epoch + 1)

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img, mask = (
                sample["image"].cuda(),
                sample["semseg_mask"].cuda(),
            )

            if random.random() < 0.5:
                img = img.flip(-1)
                mask = mask.flip(-1)

            pred = model(img)
            loss = criterion(pred, mask.long())

            loss.backward()
            optimizer.step()

            # 清理 GPU 内存，防止内存碎片化
            if i % 50 == 0:  # 每 50 个 iteration 清理一次
                torch.cuda.empty_cache()

            total_loss += loss.item()

            if rank == 0:
                iters = epoch * len(trainloader) + i
                writer.add_scalar("train/loss", loss.item(), iters)
                writer.add_scalar("train/lr_head", optimizer.param_groups[0]["lr"], iters)

            if rank == 0 and i % 100 == 0:
                current_lr_head = optimizer.param_groups[0]["lr"]
                logger.info(f"Iter: {i}/{len(trainloader)}, LR Head: {current_lr_head:.7f}, Loss: {loss.item():.3f}")

        if rank == 0:
            avg_loss = total_loss / len(trainloader) if len(trainloader) > 0 else 0
            logger.info(f"Epoch {epoch} Average Train Loss: {avg_loss:.4f}")

        scheduler.step()

        model.eval()

        metric = SegMetric(args.num_classes, device='cuda')

        with torch.no_grad():
            for i, sample in enumerate(valloader):
                img, mask = sample["image"].cuda(), sample["semseg_mask"].cuda()

                # BNHead case - direct prediction
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                # Upsample prediction to match ground truth size
                h, w = mask.shape[-2:]  # Get original image size
                pred = F.interpolate(
                    pred.unsqueeze(1).float(),  # Add channel dimension and convert to float
                    size=(h, w),
                    mode='nearest').squeeze(1).long()  # Remove channel dimension and convert back to long

                metric.update(mask.cpu().numpy(), pred.cpu().numpy())

                # Save the first image's prediction for visualization
                if i == 0 and rank == 0:
                    # Get the first prediction in the batch
                    single_pred = pred[0].cpu().numpy()

                    # Create an empty RGB image
                    h, w = single_pred.shape
                    output_image = np.zeros((h, w, 3), dtype=np.uint8)

                    # Map 0 to black, 1 to white, 2 to gray
                    output_image[single_pred == 0] = [0, 0, 0]  # Black
                    output_image[single_pred == 1] = [255, 255, 255]  # White
                    output_image[single_pred == 2] = [128, 128, 128]  # Gray

                    # Save the image
                    save_path = os.path.join(args.save_path, 'pred')
                    os.makedirs(save_path, exist_ok=True)
                    save_image_path = os.path.join(save_path, f"epoch_{epoch}.png")
                    Image.fromarray(output_image).save(save_image_path)
                    logger.info(f"Saved prediction for epoch {epoch} to {save_image_path}")

                    # Save GT for epoch 0
                    if epoch == 0:
                        single_gt = mask[0].cpu().numpy()
                        gt_output_image = np.zeros((h, w, 3), dtype=np.uint8)
                        gt_output_image[single_gt == 0] = [0, 0, 0]
                        gt_output_image[single_gt == 1] = [255, 255, 255]
                        gt_output_image[single_gt == 2] = [128, 128, 128]
                        gt_save_path = os.path.join(args.save_path, 'gt')
                        os.makedirs(gt_save_path, exist_ok=True)
                        gt_image_path = os.path.join(gt_save_path, "000gt.png")
                        Image.fromarray(gt_output_image).save(gt_image_path)
                        logger.info(f"Saved GT for epoch 0 to {gt_image_path}")

        scores = metric.get_scores()
        if rank == 0:
            logger.info("================ Validation Results ================")
            logger.info(pprint.pformat(scores))
            logger.info("=====================================================")
            print()

            for name, val in scores.items():
                writer.add_scalar(f"eval/{name}", val, epoch)

            if scores["Mean IoU"] > previous_best["Mean IoU"]:
                previous_best["Mean IoU"] = scores["Mean IoU"]
                is_best = True
            else:
                is_best = False

            # 保存checkpoint
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "args": args,
            }
            latest_path = os.path.join(args.save_path, "latest_frozen_checkpoint.pth")
            torch.save(checkpoint, latest_path)
            logger.info(f"Saved latest frozen checkpoint to {latest_path}")

            if epoch % 10 == 0:
                epoch_path = os.path.join(args.save_path, f"frozen_checkpoint_epoch_{epoch}.pth")
                torch.save(checkpoint, epoch_path)
                logger.info(f"Saved epoch {epoch} frozen checkpoint to {epoch_path}")

            if is_best:
                best_path = os.path.join(args.save_path, "best_miou_checkpoint.pth")
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best mIoU checkpoint to {best_path}")

    if rank == 0:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
