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
# 修改: 导入混合精度训练所需的库
from torch.amp import GradScaler, autocast

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.c3vd import C3VD
from dataset.inhouse import InHouse
from dataset.simcol import Simcol
from depth_anything_v2.dpt import DepthAnythingV2
from dataset.endomapper import Endomapper
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log
from torch.utils.data.dataloader import default_collate

C3VD_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/c3vd/train.txt"
C3VD_VAL_SPLIT = f"{os.environ['JIANFU']}/data/c3vd/test.txt"
ENDOMAPER_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/endomapper_sim/file_list.txt"
INHOUSE_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/inhouse/train.txt"
INHOUSE_VAL_SPLIT = f"{os.environ['JIANFU']}/data/inhouse/val.txt"
SIMCOL_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/simcol/train_paths.txt"
SIMCOL_VAL_SPLIT = f"{os.environ['JIANFU']}/data/simcol/val_paths.txt"

parser = argparse.ArgumentParser(description="Depth Anything V2 for Metric Depth Estimation")

parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
parser.add_argument(
    "--dataset",
    default="hypersim",
    choices=["hypersim", "vkitti", "c3vd", "simcol", "combined"],
)
parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--min-depth", default=0.001, type=float)
parser.add_argument("--max-depth", default=20, type=float)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--bs", default=2, type=int)
parser.add_argument("--lr", default=0.000005, type=float)
parser.add_argument("--pretrained-from", type=str)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
# 修改: 添加性能优化相关的参数
parser.add_argument("--num-workers", default=16, type=int, help="DataLoader使用的工作进程数")
parser.add_argument("--accumulation-steps", default=1, type=int, help="梯度累积步数")
parser.add_argument("--no-amp", action='store_true', help="禁用自动混合精度")
parser.add_argument("--no-compile", action='store_true', help="禁用torch.compile")


def collate_fn_pad(batch):
    # 这部分函数保持不变
    max_h = 0
    max_w = 0
    for item in batch:
        max_h = max(max_h, item['image'].shape[-2])
        max_w = max(max_w, item['image'].shape[-1])

    stride = 14
    if max_h % stride != 0:
        max_h = max_h + (stride - max_h % stride)
    if max_w % stride != 0:
        max_w = max_w + (stride - max_w % stride)

    images, depths, valid_masks, image_paths, max_depths = [], [], [], [], []

    for item in batch:
        image, depth, valid_mask, max_depth = item['image'], item['depth'], item['valid_mask'], item['max_depth']
        h, w = image.shape[-2:]
        pad_h, pad_w = max_h - h, max_w - w

        images.append(F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0))
        depths.append(F.pad(depth, (0, pad_w, 0, pad_h), mode='constant', value=0))
        valid_masks.append(F.pad(valid_mask.float(), (0, pad_w, 0, pad_h), mode='constant', value=0).bool())
        image_paths.append(item['image_path'])
        max_depths.append(max_depth)

    return {'image': torch.stack(images), 'depth': torch.stack(depths), 'valid_mask': torch.stack(valid_masks), 'image_path': image_paths, 'max_depth': max_depths}


def main():
    args = parser.parse_args()

    # PyTorch 2.0+ 允许使用TF32，在Ampere及更新架构的GPU上可以加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    warnings.simplefilter("ignore", np.polynomial.polyutils.RankWarning)
    logger = init_log("global", logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**vars(args), "ngpus": world_size}
        logger.info("训练配置:\n{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    if args.dataset == "hypersim":
        trainset = Hypersim("dataset/splits/hypersim/train.txt", "train", size=size)
    elif args.dataset == "vkitti":
        trainset = VKITTI2("dataset/splits/vkitti2/train.txt", "train", size=size)
    elif args.dataset == "c3vd":
        trainset = C3VD(
            C3VD_TRAIN_SPLIT,
            "train",
            size=size,
        )
    elif args.dataset == "simcol":
        trainset = Simcol(
            SIMCOL_TRAIN_SPLIT,
            "train",
            size=size,
        )
    elif args.dataset == "endomapper":
        trainset = Endomapper(
            ENDOMAPER_TRAIN_SPLIT,
            "train",
            size=size,
        )
    elif args.dataset == "combined":
        trainset = ConcatDataset([
            C3VD(
                C3VD_TRAIN_SPLIT,
                "train",
            ),
            Simcol(
                SIMCOL_TRAIN_SPLIT,
                "train",
            ),
            Endomapper(
                ENDOMAPER_TRAIN_SPLIT,
                "train",
            ),
            # InHouse(
            #     INHOUSE_TRAIN_SPLIT,
            #     "train",
            # )
        ])
    else:
        raise NotImplementedError

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, drop_last=False, sampler=trainsampler, collate_fn=collate_fn_pad)

    if args.dataset == "hypersim":
        valset = Hypersim("dataset/splits/hypersim/val.txt", "val", size=size)
    elif args.dataset == "vkitti":
        valset = KITTI("dataset/splits/kitti/val.txt", "val", size=size)
    elif args.dataset == "c3vd":
        valset = C3VD(C3VD_VAL_SPLIT, "val", size=size)
    elif args.dataset == "simcol":
        valset = Simcol(SIMCOL_VAL_SPLIT, "val", size=size)
    elif args.dataset == "endomapper":
        valset = Endomapper(ENDOMAPER_TRAIN_SPLIT, "val", size=size)
    elif args.dataset == "combined":
        valset = ConcatDataset([
            # C3VD("dataset/splits/c3vd/val.txt", "val", size=size),
            # Simcol("dataset/splits/simcol/val.txt", "val", size=size),
            InHouse(INHOUSE_VAL_SPLIT, "val")
        ])
    else:
        raise NotImplementedError

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=args.bs,
        # 修改: 开启 pin_memory
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
        sampler=valsampler,
        collate_fn=collate_fn_pad)

    local_rank = int(os.environ["LOCAL_RANK"])
    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384]
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768]
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536]
        },
    }
    model = DepthAnythingV2(**{**model_configs[args.encoder], "max_depth": 1})

    if args.pretrained_from:
        model.load_state_dict(
            {
                k: v for k, v in torch.load(args.pretrained_from, map_location="cpu").items() if "pretrained" in k
            },
            strict=False,
        )

    # 修改: 使用 torch.compile 加速模型 (PyTorch 2.0+), 提供了命令行开关
    if not args.no_compile and hasattr(torch, 'compile'):
        logger.info("正在编译模型 (PyTorch 2.x)...")
        model = torch.compile(model)
        logger.info("模型编译完成。")

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        # find_unused_parameters=False,
        find_unused_parameters=True,
    )

    # 微调设置：确保所有参数都可训练
    for name, param in model.named_parameters():
        param.requires_grad = True

    criterion = SiLogLoss().cuda(local_rank)
    optimizer = AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if "pretrained" in n and p.requires_grad],
                "lr": args.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if "pretrained" not in n and p.requires_grad],
                "lr": args.lr * 10.0
            },
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    total_iters = args.epochs * len(trainloader)

    # 修改: 初始化混合精度训练的 GradScaler
    # enabled 参数可以方便地通过命令行开关AMP
    scaler = GradScaler(enabled=not args.no_amp)

    previous_best = {"d1": 0, "d2": 0, "d3": 0, "abs_rel": 100, "sq_rel": 100, "rmse": 100, "rmse_log": 100, "log10": 100, "silog": 100}

    for epoch in range(args.epochs):
        if rank == 0:
            logger.info(f"===========> Epoch: {epoch}/{args.epochs}")

        trainloader.sampler.set_epoch(epoch + 1)
        model.train()

        for i, sample in enumerate(trainloader):
            # 梯度累积逻辑
            is_accumulation_step = (i + 1) % args.accumulation_steps == 0

            # 使用 non_blocking=True 配合 pin_memory 进一步加速
            img = sample["image"].cuda(non_blocking=True)
            depth = sample["depth"].cuda(non_blocking=True)
            valid_mask = sample["valid_mask"].cuda(non_blocking=True)

            max_depth_val = torch.tensor(sample["max_depth"], device=img.device).view(-1, 1, 1)

            if random.random() < 0.5:
                img, depth, valid_mask = img.flip(-1).contiguous(), depth.flip(-1).contiguous(), valid_mask.flip(-1).contiguous()

            # 修改: 使用 autocast 上下文管理器进行前向传播
            with autocast(device_type="cuda", enabled=not args.no_amp):
                pred = model(img) * max_depth_val
                loss = criterion(
                    pred,
                    depth,
                    (valid_mask == 1) & (depth >= args.min_depth) & (depth <= max_depth_val),
                )
                # 修改: 标准化loss以进行梯度累积
                loss = loss / args.accumulation_steps

            # 修改: 使用 scaler 缩放 loss 并进行反向传播
            scaler.scale(loss).backward()

            # 修改: 在累积了足够步数后，才更新优化器
            if is_accumulation_step:
                # 在原地对优化器的参数进行梯度反缩放
                scaler.unscale_(optimizer)
                # 可选：梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if rank == 0 and i % 100 == 0:
                # 注意：这里打印的Loss是为梯度累积而缩小的Loss，所以乘回去以显示真实值
                logger.info(f"Iter: {i}/{len(trainloader)}, Loss: {loss.item() * args.accumulation_steps:.3f}")

        # 调整学习率
        lr = args.lr * (1 - (epoch * len(trainloader) + i) / total_iters)**0.9
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * 10.0

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

        for i, sample in enumerate(valloader):
            img = sample["image"].cuda(non_blocking=True).float()
            depth = sample["depth"].cuda(non_blocking=True)
            valid_mask = sample["valid_mask"].cuda(non_blocking=True)
            max_depth_list = sample["max_depth"]
            B = img.shape[0]

            # 修改: 在验证时也使用 autocast (无需scaler)
            with torch.no_grad(), autocast(device_type="cuda", enabled=not args.no_amp):
                max_depth_tensor = torch.tensor(max_depth_list, device=img.device).view(B, 1, 1)
                preds_batch = model(img) * max_depth_tensor

            for j in range(B):
                pred_j, depth_j, valid_mask_j, current_max_depth_j = \
                    preds_batch[j], depth[j], valid_mask[j], max_depth_list[j]
                if rank == 0 and i == 0 and j == 0:
                    pred_j_mean = torch.mean(pred_j)
                    pred_j_max = torch.max(pred_j)
                    pred_j_min = torch.min(pred_j)
                    pred_j_median = torch.median(pred_j)  # torch.median 接受张量

                    depth_j_mean = torch.mean(depth_j)
                    depth_j_max = torch.max(depth_j)
                    depth_j_min = torch.min(depth_j)
                    depth_j_median = torch.median(depth_j)  # torch.median 接受张量

                    logger.info(f"Pred_j - Mean: {pred_j_mean:.4f}, Max: {pred_j_max:.4f}, Min: {pred_j_min:.4f}, Median: {pred_j_median:.4f}")
                    logger.info(f"Depth_j - Mean: {depth_j_mean:.4f}, Max: {depth_j_max:.4f}, Min: {depth_j_min:.4f}, Median: {depth_j_median:.4f}")
                    logger.info(f"current_max_depth_j: {current_max_depth_j}")
                h, w = depth_j.shape
                pred_j_resized = F.interpolate(pred_j.unsqueeze(0).unsqueeze(0), (h, w), mode="bilinear", align_corners=True).squeeze()

                if rank == 0 and i == 0 and j == 0:
                    filename = sample["image_path"][j].split("/")[-2:]
                    os.makedirs(os.path.join(args.save_path, filename[-2]), exist_ok=True)
                    output_path = os.path.join(args.save_path, filename[-2], f'epoch_{epoch}_{filename[-1]}')
                    pred_16bit = (pred_j_resized / current_max_depth_j * 65535).to(torch.uint16)
                    cv2.imwrite(output_path, pred_16bit.cpu().numpy())

                final_valid_mask_j = (valid_mask_j == 1) & (depth_j >= args.min_depth) & (depth_j <= current_max_depth_j)
                if final_valid_mask_j.sum() < 10:
                    continue

                cur_results = eval_depth(pred_j_resized[final_valid_mask_j], depth_j[final_valid_mask_j])
                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            logger.info("==========================================================================================")
            logger.info("{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(*tuple(results.keys())))
            logger.info("{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}".format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info("==========================================================================================")
            print()

            for name, metric in results.items():
                writer.add_scalar(f"eval/{name}", (metric / nsamples).item(), epoch)

        if rank == 0:

            is_best_abs_rel = False
            for k in results.keys():
                current_val = results[k]
                if k in ["d1", "d2", "d3"]:
                    if current_val > previous_best[k]:
                        previous_best[k] = current_val
                else:
                    if current_val < previous_best[k]:
                        previous_best[k] = current_val
                        if k == "abs_rel":
                            is_best_abs_rel = True

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))

            # 每隔10个epoch保存模型
            if epoch % 10 == 0:
                torch.save(
                    checkpoint,
                    os.path.join(args.save_path, f"checkpoint_epoch_{epoch}.pth"),
                )

            # 由于previous_best已经在前面更新，直接使用previous_best['abs_rel']作为判断标准
            if is_best_abs_rel:
                torch.save(checkpoint, os.path.join(args.save_path, "best_abs_rel.pth"))


if __name__ == "__main__":
    main()
