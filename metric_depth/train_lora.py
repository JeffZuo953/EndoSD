import argparse
import logging
import os
import pprint
import random
import cv2
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.c3vd import C3VD
from dataset.simcol import Simcol
from depth_anything_v2.dpt_lora import DepthAnythingV2_LoRA
from dataset.endomapper import Endomapper
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log
from torch.utils.data.dataloader import default_collate

parser = argparse.ArgumentParser(
    description="Depth Anything V2 LoRA Training for Metric Depth Estimation")

parser.add_argument("--encoder",
                    default="vitl",
                    choices=["vits", "vitb", "vitl", "vitg"])
parser.add_argument(
    "--dataset",
    default="hypersim",
    choices=["hypersim", "vkitti", "c3vd", "simcol", "combined", "endomapper"],
)
parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--min-depth", default=0.001, type=float)
parser.add_argument("--max-depth", default=20, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--bs", default=2, type=int)
parser.add_argument("--lr", default=0.000005, type=float)
parser.add_argument(
    "--pretrained-from",
    type=str,
    help="Path to the FULL DINOv2 pretrained weights (.pth)",
)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)

parser.add_argument("--lora_r",
                    default=0,
                    type=int,
                    help="LoRA rank (0 means no LoRA)")
parser.add_argument("--lora_alpha",
                    default=1,
                    type=int,
                    help="LoRA alpha scaling factor")
parser.add_argument("--lora_dropout",
                    default=0.0,
                    type=float,
                    help="LoRA dropout probability")
parser.add_argument(
    "--lora_bias",
    default="none",
    choices=["none", "lora_only", "all"],
    help="Which bias parameters to train",
)
parser.add_argument(
    "--lora_head_lr_multiplier",
    default=10.0,
    type=float,
    help=
    "Multiplier for the DPT head learning rate compared to LoRA parameters",
)


def collate_fn_pad(batch):
    # 找出批次中图像和深度的最大尺寸
    max_h = 0
    max_w = 0
    for item in batch:
        max_h = max(max_h, item["image"].shape[-2])
        max_w = max(max_w, item["image"].shape[-1])

    # 确保最大尺寸是某个数的倍数（例如 14），如果模型有此要求
    # 如果你的 Resize 中去掉了 ensure_multiple_of，这里可能也需要类似处理，
    # 或者确保 Resize 后的尺寸总是能被后续层处理
    stride = 14  # 例如，ViT 可能需要 14 或 16 的倍数
    if max_h % stride != 0:
        max_h = max_h + (stride - max_h % stride)
    if max_w % stride != 0:
        max_w = max_w + (stride - max_w % stride)

    images = []
    depths = []
    valid_masks = []
    image_paths = []

    for item in batch:
        image = item["image"]
        depth = item["depth"]
        valid_mask = item["valid_mask"]

        h, w = image.shape[-2:]

        # 计算需要填充的量
        pad_h = max_h - h
        pad_w = max_w - w

        # F.pad 的参数是 (左, 右, 上, 下)
        # 对于图像，使用 0 填充
        padded_image = F.pad(image, (0, pad_w, 0, pad_h),
                             mode="constant",
                             value=0)
        images.append(padded_image)

        # 对于深度图，也使用 0 填充（通常 0 表示无效深度）
        padded_depth = F.pad(depth, (0, pad_w, 0, pad_h),
                             mode="constant",
                             value=0)
        depths.append(padded_depth)

        # 对于 valid_mask (布尔或 0/1)，填充 False 或 0
        padded_mask = F.pad(valid_mask.float(), (0, pad_w, 0, pad_h),
                            mode="constant",
                            value=0).bool()
        valid_masks.append(padded_mask)

        image_paths.append(item["image_path"])

    # 使用 torch.stack 将列表合并为批次张量
    images_batch = torch.stack(images)
    depths_batch = torch.stack(depths)
    valid_masks_batch = torch.stack(valid_masks)

    # 返回批次字典
    return {
        "image": images_batch,
        "depth": depths_batch,
        "valid_mask": valid_masks_batch,
        "image_path": image_paths,  # 列表形式
    }


def main():
    args = parser.parse_args()

    # Determine project root based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    warnings.simplefilter("ignore", np.RankWarning)

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
        logger.info(f"Use LoRA: {args.lora_r > 0}")
        if args.lora_r > 0:
            logger.info(
                f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}, dropout: {args.lora_dropout}, bias: {args.lora_bias}"
            )

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    # Construct dataset paths relative to project root
    if args.dataset == "hypersim":
        train_split_path = os.path.join(project_root,
                                        "dataset/splits/hypersim/train.txt")
        val_split_path = os.path.join(project_root,
                                      "dataset/splits/hypersim/val.txt")
        trainset = Hypersim(train_split_path, "train", size=size)
        valset = Hypersim(val_split_path, "val", size=size)
    elif args.dataset == "vkitti":
        train_split_path = os.path.join(project_root,
                                        "dataset/splits/vkitti2/train.txt")
        val_split_path = os.path.join(
            project_root,
            "dataset/splits/kitti/val.txt")  # Note: Uses kitti val split
        trainset = VKITTI2(train_split_path, "train", size=size)
        valset = KITTI(val_split_path, "val", size=size)
    elif args.dataset == "c3vd":
        train_split_path = os.path.join(
            project_root, "metric_depth/dataset/splits/c3vd/train.txt")
        val_split_path = os.path.join(
            project_root, "metric_depth/dataset/splits/c3vd/val.txt")
        trainset = C3VD(train_split_path, "train", size=size)
        valset = C3VD(val_split_path, "val", size=size)
    elif args.dataset == "simcol":
        train_split_path = os.path.join(project_root,
                                        "dataset/splits/simcol/train.txt")
        val_split_path = os.path.join(project_root,
                                      "dataset/splits/simcol/val.txt")
        trainset = Simcol(train_split_path, "train", size=size)
        valset = Simcol(val_split_path, "val", size=size)
    elif args.dataset == "endomapper":
        train_split_path = os.path.join(project_root,
                                        "dataset/splits/endomapper/train.txt")
        val_split_path = os.path.join(project_root,
                                      "dataset/splits/endomapper/val.txt")
        trainset = Endomapper(train_split_path, "train", size=size)
        valset = Endomapper(val_split_path, "val", size=size)
    elif args.dataset == "combined":
        # Combine datasets using paths relative to project root
        c3vd_train_path = os.path.join(project_root, "metric_depth",
                                       "dataset/splits/c3vd/train.txt")
        simcol_train_path = os.path.join(project_root, "metric_depth",
                                         "dataset/splits/simcol/train.txt")
        endomapper_train_path = os.path.join(
            project_root, "metric_depth",
            "dataset/splits/endomapper/train.txt")
        c3vd_val_path = os.path.join(project_root, "metric_depth",
                                     "dataset/splits/c3vd/val.txt")
        simcol_val_path = os.path.join(project_root, "metric_depth",
                                       "dataset/splits/simcol/val.txt")
        endomapper_val_path = os.path.join(
            project_root, "metric_depth", "dataset/splits/endomapper/val.txt")

        trainset = ConcatDataset([
            C3VD(c3vd_train_path, "train", size=size),
            # Simcol(simcol_train_path, "train", size=size),
            # Endomapper(endomapper_train_path, "train", size=size),
        ])
        valset = ConcatDataset([
            C3VD(c3vd_val_path, "val", size=size),
            # Simcol(simcol_val_path, "val", size=size),
            # Endomapper(endomapper_val_path, "val", size=size),
        ])
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
    model = DepthAnythingV2_LoRA(
        encoder=args.encoder,
        features=model_config["features"],
        out_channels=model_config["out_channels"],
        max_depth=args.max_depth,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
    )

    # --- Model Setup for DDP ---
    # Move model to GPU *before* wrapping
    model.cuda(local_rank)

    # Convert BatchNorm layers *before* wrapping
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap the model with DDP
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
            pretrained_path = os.path.abspath(
                os.path.join(script_dir, args.pretrained_from))
            logger.info(
                f"Resolving relative pretrained path: {args.pretrained_from} -> {pretrained_path}"
            )
        else:
            pretrained_path = args.pretrained_from

        logger.info(f"Loading pretrained weights from: {pretrained_path}")
        if not os.path.exists(pretrained_path):
            logger.error(
                f"Pretrained file not found at resolved path: {pretrained_path}"
            )
            raise FileNotFoundError(
                f"Pretrained file not found: {pretrained_path}")

        # Load checkpoint to CPU first to avoid GPU memory issues on rank 0
        full_state_dict = torch.load(pretrained_path, map_location="cpu")

        # --- Load Backbone Weights ---
        # Extract weights for the 'pretrained' part (backbone + LoRA layers potentially)
        backbone_weights = {
            k.replace("pretrained.", "", 1): v
            for k, v in full_state_dict.items() if k.startswith("pretrained.")
        }
        head_weights = {
            k.replace("depth_head.", "", 1): v
            for k, v in full_state_dict.items() if k.startswith("depth_head.")
        }

        # === 按evaluate的规则替换key ===
        import re
        patterns = {
            r"blocks.(\d+).attn.qkv.(weight|bias)":
            r"blocks.\1.attn.qkv.linear.\2",
            r"blocks.(\d+).attn.proj.(weight|bias)":
            r"blocks.\1.attn.proj.linear.\2",
            r"blocks.(\d+).mlp.fc1.(weight|bias)":
            r"blocks.\1.mlp.fc1.linear.\2",
            r"blocks.(\d+).mlp.fc2.(weight|bias)":
            r"blocks.\1.mlp.fc2.linear.\2",
        }
        merged_backbone_weights = backbone_weights.copy()
        for backbone_key in list(backbone_weights.keys()):
            for pattern, replacement in patterns.items():
                if re.match(pattern, backbone_key):
                    lora_key = re.sub(pattern, replacement, backbone_key)
                    merged_backbone_weights[
                        lora_key] = merged_backbone_weights.pop(backbone_key)
                    break

        # 加载权重
        model.module.pretrained.load_state_dict(merged_backbone_weights,
                                                strict=False)
        model.module.depth_head.load_state_dict(head_weights, strict=True)
    # --- Criterion and Optimizer Setup ---
    criterion = SiLogLoss().cuda(local_rank)

    # Now setup optimizer, accessing parameters via model.module
    params_to_optimize = []
    for name, param in model.module.pretrained.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            if rank == 0:
                logger.debug(f"Adding LoRA param to optimizer: {name}")

    dpt_head_params = []
    for name, param in model.module.depth_head.named_parameters():
        if param.requires_grad:
            dpt_head_params.append(param)
            if rank == 0:
                logger.debug(f"Adding DPT head param to optimizer: {name}")
        else:
            if rank == 0:
                logger.warning(f"DPT head param {name} does not require grad.")

    if not dpt_head_params:
        logger.error("No trainable parameters found in the DPT head!")

    optimizer = AdamW(
        [
            {
                "params": params_to_optimize,
                "lr": args.lr
            },
            {
                "params": dpt_head_params,
                "lr": args.lr * args.lora_head_lr_multiplier
            },
        ],
        weight_decay=0.01,
    )

    total_iters = args.epochs * len(trainloader)

    previous_best = {
        "d1": 0,
        "d2": 0,
        "d3": 0,
        "abs_rel": 100,
        "sq_rel": 100,
        "rmse": 100,
        "rmse_log": 100,
        "log10": 100,
        "silog": 100,
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

            img, depth, valid_mask = (
                sample["image"].cuda(),
                sample["depth"].cuda(),
                sample["valid_mask"].cuda(),
            )

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)

            mask = ((valid_mask == 1)
                    & (depth >= args.min_depth)
                    & (depth <= args.max_depth))
            if not mask.any():
                continue

            loss = criterion(pred, depth, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr_scale = (1 - iters / total_iters)**0.9
            optimizer.param_groups[0]["lr"] = args.lr * lr_scale
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]["lr"] = (
                    args.lr * args.lora_head_lr_multiplier * lr_scale)

            if rank == 0:
                writer.add_scalar("train/loss", loss.item(), iters)
                writer.add_scalar("train/lr_lora",
                                  optimizer.param_groups[0]["lr"], iters)
                if len(optimizer.param_groups) > 1:
                    writer.add_scalar("train/lr_head",
                                      optimizer.param_groups[1]["lr"], iters)

            if rank == 0 and i % 100 == 0:
                current_lr_lora = optimizer.param_groups[0]["lr"]
                current_lr_head = (optimizer.param_groups[1]["lr"]
                                   if len(optimizer.param_groups) > 1 else 0.0)
                logger.info(
                    f"Iter: {i}/{len(trainloader)}, LR LoRA: {current_lr_lora:.7f}, LR Head: {current_lr_head:.7f}, Loss: {loss.item():.3f}"
                )

        if rank == 0:
            avg_loss = total_loss / len(trainloader) if len(
                trainloader) > 0 else 0
            logger.info(f"Epoch {epoch} Average Train Loss: {avg_loss:.4f}")

        model.eval()

        results = {
            "d1": torch.tensor([0.0]).cuda(),
            "d2": torch.tensor([0.0]).cuda(),
            "d3": torch.tensor([0.0]).cuda(),
            "abs_rel": torch.tensor([0.0]).cuda(),
            "sq_rel": torch.tensor([0.0]).cuda(),
            "rmse": torch.tensor([0.0]).cuda(),
            "rmse_log": torch.tensor([0.0]).cuda(),
            "log10": torch.tensor([0.0]).cuda(),
            "silog": torch.tensor([0.0]).cuda(),
        }
        nsamples = torch.tensor([0.0]).cuda()

        with torch.no_grad():
            for i, sample in enumerate(valloader):
                img, depth = sample["image"].cuda(), sample["depth"].cuda()
                valid_mask = sample["valid_mask"].cuda()

                pred = model(img)

                pred_interpolated = F.interpolate(
                    pred[:, None],
                    depth.shape[-2:],
                    mode="bilinear",
                    align_corners=True).squeeze(1)

                if rank == 0 and (i == 0):
                    filename = sample["image_path"][0].split("/")[-2:]
                    os.makedirs(args.save_path, exist_ok=True)
                    os.makedirs(os.path.join(args.save_path, filename[-2]),
                                exist_ok=True)
                    output_path = os.path.join(args.save_path, filename[-2],
                                               f"{epoch}_{filename[-1]}")
                    # 可视化时使用原始图像
                    # visualize_depth(img, pred, depth, output_path)
                    pred_16bit = (pred / args.max_depth * 65535).to(
                        torch.uint16)

                    cv2.imwrite(output_path,
                                pred_16bit.squeeze().cpu().numpy())

                for b_idx in range(depth.shape[0]):
                    pred_single = pred_interpolated[b_idx]
                    depth_single = depth[b_idx]
                    valid_mask_single = valid_mask[b_idx]

                    eval_mask = ((valid_mask_single == 1)
                                 & (depth_single >= args.min_depth)
                                 & (depth_single <= args.max_depth))

                    if eval_mask.sum() < 10:
                        continue

                    cur_results = eval_depth(pred_single[eval_mask],
                                             depth_single[eval_mask])

                    for k in results.keys():
                        results[k] += cur_results[k]
                    nsamples += 1

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            if nsamples.item() > 0:
                final_results = {
                    k: (v / nsamples).item()
                    for k, v in results.items()
                }
                logger.info(
                    "================ Validation Results ================")
                logger.info(
                    "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".
                    format(*final_results.keys()))
                logger.info(
                    "{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}"
                    .format(*final_results.values()))
                logger.info(
                    "=====================================================")
                print()

                for name, metric in final_results.items():
                    writer.add_scalar(f"eval/{name}", metric, epoch)

                is_best_abs_rel = False
                for k in results.keys():
                    current_val = final_results[k]
                    if k in ["d1", "d2", "d3"]:
                        if current_val > previous_best[k]:
                            previous_best[k] = current_val
                    else:
                        if current_val < previous_best[k]:
                            previous_best[k] = current_val
                            if k == "abs_rel":
                                is_best_abs_rel = True
            else:
                logger.warning("No validation samples were evaluated.")
                is_best_abs_rel = False

            # 保存checkpoint
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
                "args": args,
            }
            latest_path = os.path.join(args.save_path,
                                       "latest_lora_checkpoint.pth")
            torch.save(checkpoint, latest_path)
            logger.info(f"Saved latest LoRA checkpoint to {latest_path}")

            if epoch % 10 == 0:
                epoch_path = os.path.join(
                    args.save_path, f"lora_checkpoint_epoch_{epoch}.pth")
                torch.save(checkpoint, epoch_path)
                logger.info(
                    f"Saved epoch {epoch} LoRA checkpoint to {epoch_path}")

            if is_best_abs_rel:
                best_path = os.path.join(args.save_path,
                                         "best_lora_abs_rel.pth")
                torch.save(checkpoint, best_path)
                logger.info(
                    f"Saved best abs_rel LoRA checkpoint to {best_path}")

    if rank == 0:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
