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
# from dataset.c3vd import C3VD
from dataset.cache_utils import CacheDataset
# from dataset.inhouse import InHouse
# from dataset.simcol import Simcol
from depth_anything_v2.dpt import DepthAnythingV2
# from dataset.endomapper import Endomapper
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log

# C3VD_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/c3vd/cache/train/train_cache.txt"
# C3VD_VAL_SPLIT = f"{os.environ['JIANFU']}/data/c3vd/cache/test/test_cache.txt"
# ENDOMAPER_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/endomapper_sim/cache/train/train_cache.txt"
# INHOUSE_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/inhouse/cache/train_cache.txt"
# INHOUSE_VAL_SPLIT = f"{os.environ['JIANFU']}/data/inhouse/cache/val_cache.txt"
# SIMCOL_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/simcol/cache/train_cache.txt"
# SIMCOL_VAL_SPLIT = f"{os.environ['JIANFU']}/data/simcol/cache/val_cache.txt"
C3VD_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/c3vd/cache/train/train_cache.txt"
C3VD_VAL_SPLIT = f"{os.environ['JIANFU']}/data/c3vd/cache/test/test_cache.txt"
ENDOMAPER_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/endomapper_sim/cache/train/train_cache.txt"
INHOUSE_TRAIN_SPLIT = "/home/jianfu/data/inhouse/cache/train_cache.txt"
INHOUSE_VAL_SPLIT = "/home/jianfu/data/inhouse/cache/val_cache.txt"
SIMCOL_TRAIN_SPLIT = f"{os.environ['JIANFU']}/data/simcol/cache/train_cache.txt"
SIMCOL_VAL_SPLIT = f"{os.environ['JIANFU']}/data/simcol/cache/val_cache.txt"

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

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank, world_size = setup_distributed(port=args.port)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0
    if rank == 0:
        all_args = {**vars(args), "ngpus": world_size}
        logger.info("训练配置:\n{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.enabled = True
    cudnn.benchmark = True
    warnings.simplefilter("ignore", np.polynomial.polyutils.RankWarning)

    size = (args.img_size, args.img_size)
    if args.dataset == "hypersim":
        trainset = Hypersim("dataset/splits/hypersim/train.txt", "train", size=size)
    elif args.dataset == "vkitti":
        trainset = VKITTI2("dataset/splits/vkitti2/train.txt", "train", size=size)
    # elif args.dataset == "c3vd":
    #     trainset = C3VD(
    #         C3VD_TRAIN_SPLIT,
    #         "train",
    #         size=size,
    #     )
    # elif args.dataset == "simcol":
    #     trainset = Simcol(
    #         SIMCOL_TRAIN_SPLIT,
    #         "train",
    #         size=size,
    #     )
    # elif args.dataset == "endomapper":
    #     trainset = Endomapper(
    #         ENDOMAPER_TRAIN_SPLIT,
    #         "train",
    #         size=size,
    #     )
    elif args.dataset == "combined":
        trainset = ConcatDataset([
            # C3VD(
            #     C3VD_TRAIN_SPLIT,
            #     "train",
            # ),
            # Simcol(
            #     SIMCOL_TRAIN_SPLIT,
            #     "train",
            # ),
            # Endomapper(
            #     ENDOMAPER_TRAIN_SPLIT,
            #     "train",
            # ),
            # CacheDataset(C3VD_TRAIN_SPLIT, 0.1),
            # CacheDataset(SIMCOL_TRAIN_SPLIT, 0.2),
            # CacheDataset(ENDOMAPER_TRAIN_SPLIT, 0.2),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            CacheDataset(INHOUSE_TRAIN_SPLIT, 0.05),
            # InHouse(
            #     INHOUSE_TRAIN_SPLIT,
            #     "train",
            # )
        ])
    else:
        raise NotImplementedError

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(
        trainset,
        batch_size=args.bs,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
        sampler=trainsampler,
        collate_fn=collate_fn_pad,
    )

    if args.dataset == "hypersim":
        valset = Hypersim("dataset/splits/hypersim/val.txt", "val", size=size)
    elif args.dataset == "vkitti":
        valset = KITTI("dataset/splits/kitti/val.txt", "val", size=size)
    # elif args.dataset == "c3vd":
    #     valset = C3VD(C3VD_VAL_SPLIT, "val", size=size)
    # elif args.dataset == "simcol":
    #     valset = Simcol(SIMCOL_VAL_SPLIT, "val", size=size)
    # elif args.dataset == "endomapper":
    #     valset = Endomapper(ENDOMAPER_TRAIN_SPLIT, "val", size=size)
    elif args.dataset == "combined":
        valset = ConcatDataset([
            # C3VD("dataset/splits/c3vd/val.txt", "val", size=size),
            # Simcol("dataset/splits/simcol/val.txt", "val", size=size),
            # InHouse(INHOUSE_VAL_SPLIT, "val")
            CacheDataset(INHOUSE_VAL_SPLIT, 0.05)
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

    unused_param_names = ["pretrained.mask_token", "depth_head.scratch.refinenet4"]

    logger.info("正在设置参数组，将冻结以下未使用的参数:")
    for name in unused_param_names:
        logger.info(f"- {name}")

    def is_param_unused(name: str) -> bool:
        for unused_name in unused_param_names:
            if unused_name in name:
                return True
        return False

    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        # 如果参数是已知的无用参数
        if is_param_unused(name):
            param.requires_grad = False  # 明确地冻结它
            continue  # 跳过，不将其添加到优化器中

        # 对于需要训练的参数，确保它们是可训练的
        param.requires_grad = True

        # 根据名称将其分类到不同的学习率组
        if "pretrained" in name:
            pretrained_params.append(param)
        else:
            new_params.append(param)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )

    if not args.no_compile and hasattr(torch, 'compile'):
        logger.info("正在编译模型 (PyTorch 2.x)...")
        model = torch.compile(model)
        logger.info("模型编译完成。")

    criterion = SiLogLoss().cuda(local_rank)

    optimizer_param_groups = [
        {
            "params": pretrained_params,
            "lr": args.lr
        },
        {
            "params": new_params,
            "lr": args.lr * 10.0
        },
    ]

    # 5. 用筛选后的参数列表初始化优化器
    optimizer = AdamW(
        optimizer_param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        fused=True,
    )

    logger.info(f"优化器初始化完成。Pretrained参数数量: {len(pretrained_params)}, New参数数量: {len(new_params)}")

    total_iters = args.epochs * len(trainloader)

    # 修改: 初始化混合精度训练的 GradScaler
    # enabled 参数可以方便地通过命令行开关AMP
    scaler = GradScaler(enabled=not args.no_amp)

    previous_best = {"d1": 0, "d2": 0, "d3": 0, "abs_rel": 100, "sq_rel": 100, "rmse": 100, "rmse_log": 100, "log10": 100, "silog": 100}

    for epoch in range(args.epochs):
        if rank == 0:
            import sys

            # 定义你想“吃掉”多少GB的内存。这个值应该接近你机器的总物理内存。
            # 例如，如果机器有64GB内存，你可以尝试分配50GB。
            # 你可以通过 `free -h` 命令查看总内存和可用内存。
            GB_TO_EAT = 45

            print(f"Attempting to allocate ~{GB_TO_EAT} GB of memory to flush file cache...")

            try:
                # 创建一个巨大的bytearray对象来占用内存
                # 1024**3 = 1GB
                eater = bytearray(GB_TO_EAT * 1024**3)
                # 访问一下内存，防止被优化掉
                eater[-1] = 0
                print(f"Successfully allocated memory. Cache should be significantly reduced.")
            except MemoryError:
                # 如果你申请的内存超过了可用内存，会报这个错。
                # 这恰好说明我们的目的达到了：系统已无更多内存，缓存已被尽可能地清空。
                print("MemoryError caught. This is expected and means we've successfully used up available RAM.")
                print("The file cache has likely been flushed.")
            except Exception as e:
                print(f"An error occurred: {e}")

            print("Script finished. You can now run your benchmark.")

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
                    output_path = os.path.join(args.save_path, filename[-2], f'epoch_{epoch}_{filename[-1]}.png')
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
