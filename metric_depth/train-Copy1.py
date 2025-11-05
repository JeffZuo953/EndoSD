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
from depth_anything_v2.dpt import DepthAnythingV2
from dataset.endomapper import Endomapper
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log
from torch.utils.data.dataloader import default_collate


parser = argparse.ArgumentParser(
    description="Depth Anything V2 for Metric Depth Estimation"
)

parser.add_argument(
    "--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
)
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


def collate_fn_pad(batch):
    # 找出批次中图像和深度的最大尺寸
    max_h = 0
    max_w = 0
    for item in batch:
        max_h = max(max_h, item['image'].shape[-2])
        max_w = max(max_w, item['image'].shape[-1])
        
    # 确保最大尺寸是某个数的倍数（例如 14），如果模型有此要求
    # 如果你的 Resize 中去掉了 ensure_multiple_of，这里可能也需要类似处理，
    # 或者确保 Resize 后的尺寸总是能被后续层处理
    stride = 14 # 例如，ViT 可能需要 14 或 16 的倍数
    if max_h % stride != 0:
        max_h = max_h + (stride - max_h % stride)
    if max_w % stride != 0:
        max_w = max_w + (stride - max_w % stride)

    images = []
    depths = []
    valid_masks = []
    image_paths = []

    for item in batch:
        image = item['image']
        depth = item['depth']
        valid_mask = item['valid_mask']
        
        h, w = image.shape[-2:]
        
        # 计算需要填充的量
        pad_h = max_h - h
        pad_w = max_w - w
        
        # F.pad 的参数是 (左, 右, 上, 下)
        # 对于图像，使用 0 填充
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        images.append(padded_image)
        
        # 对于深度图，也使用 0 填充（通常 0 表示无效深度）
        padded_depth = F.pad(depth, (0, pad_w, 0, pad_h), mode='constant', value=0)
        depths.append(padded_depth)
        
        # 对于 valid_mask (布尔或 0/1)，填充 False 或 0
        padded_mask = F.pad(valid_mask.float(), (0, pad_w, 0, pad_h), mode='constant', value=0).bool()
        valid_masks.append(padded_mask)

        image_paths.append(item['image_path'])

    # 使用 torch.stack 将列表合并为批次张量
    images_batch = torch.stack(images)
    depths_batch = torch.stack(depths)
    valid_masks_batch = torch.stack(valid_masks)
    
    # 返回批次字典
    return {
        'image': images_batch,
        'depth': depths_batch,
        'valid_mask': valid_masks_batch,
        'image_path': image_paths # 列表形式
    }


def main():
    args = parser.parse_args()

    warnings.simplefilter("ignore", np.RankWarning)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
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
            "dataset/splits/c3vd/train.txt",
            "train",
            size=size,
        )
    elif args.dataset == "simcol":
        trainset = Simcol(
            "dataset/splits/simcol/train.txt",
            "train",
            size=size,
        )
    elif args.dataset == "endomapper":
        trainset = Endomapper(
            "dataset/splits/endomapper/train.txt",
            "train",
            size=size,
        )
    elif args.dataset == "combined":
        trainset = ConcatDataset(
            [
                C3VD(
                    "dataset/splits/c3vd/train.txt",
                    "train",
                    size=size,
                ),
                Simcol(
                    "dataset/splits/simcol/train.txt",
                    "train",
                    size=size,
                ),
                Endomapper(
                    "dataset/splits/endomapper/train.txt",
                    "train",
                    size=size,
                ),
            ]
        )
    else:
        raise NotImplementedError
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset,
        batch_size=args.bs,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler,
        collate_fn=collate_fn_pad
    )

    if args.dataset == "hypersim":
        valset = Hypersim("dataset/splits/hypersim/val.txt", "val", size=size)
    elif args.dataset == "vkitti":
        valset = KITTI("dataset/splits/kitti/val.txt", "val", size=size)
    elif args.dataset == "c3vd":
        valset = C3VD("dataset/splits/c3vd/val.txt", "val", size=size)
    elif args.dataset == "simcol":
        valset = Simcol("dataset/splits/simcol/val.txt", "val", size=size)
    elif args.dataset == "endomapper":
        valset = Endomapper("dataset/splits/endomapper/val.txt", "val", size=size)
    elif args.dataset == "combined":
        valset = ConcatDataset(
            [
                C3VD("dataset/splits/c3vd/val.txt", "val", size=size),
                Simcol("dataset/splits/simcol/val.txt", "val", size=size),
            ]
        )
    else:
        raise NotImplementedError
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=valsampler,
        collate_fn=collate_fn_pad if args.bs > 1 else default_collate
    )

    local_rank = int(os.environ["LOCAL_RANK"])

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }
    model = DepthAnythingV2(
        **{**model_configs[args.encoder], "max_depth": args.max_depth}
    )

    if args.pretrained_from:
        model.load_state_dict(
            {
                k: v
                for k, v in torch.load(args.pretrained_from, map_location="cpu").items()
                if "pretrained" in k
            },
            strict=False,
        )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    for name, param in model.named_parameters():
        if "pretrained" in name:
            param.requires_grad = False

    criterion = SiLogLoss().cuda(local_rank)

    optimizer = AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if ("pretrained" in name and param.requires_grad)
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if ("pretrained" not in name and param.requires_grad)
                ],
                "lr": args.lr * 10.0,
            },
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
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
            logger.info(
                "===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}".format(
                    epoch,
                    args.epochs,
                    previous_best["d1"],
                    previous_best["d2"],
                    previous_best["d3"],
                )
            )
            logger.info(
                "===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, "
                "log10: {:.3f}, silog: {:.3f}".format(
                    epoch,
                    args.epochs,
                    previous_best["abs_rel"],
                    previous_best["sq_rel"],
                    previous_best["rmse"],
                    previous_best["rmse_log"],
                    previous_best["log10"],
                    previous_best["silog"],
                )
            )

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

            loss = criterion(
                pred,
                depth,
                (valid_mask == 1)
                & (depth >= args.min_depth)
                & (depth <= args.max_depth),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            if rank == 0:
                writer.add_scalar("train/loss", loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info(
                    "Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}".format(
                        i,
                        len(trainloader),
                        optimizer.param_groups[0]["lr"],
                        loss.item(),
                    )
                )

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

            img, depth, valid_mask = (
                sample["image"].cuda().float(),
                sample["depth"].cuda()[0],
                sample["valid_mask"].cuda()[0],
            )

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(
                    pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
                )[0, 0]
            if rank ==0 and (i == 0 or i == 3000):
                filename = sample["image_path"][0].split("/")[-2:]
                os.makedirs(args.save_path, exist_ok=True)
                os.makedirs(os.path.join(args.save_path, filename[-2]), exist_ok=True)
                output_path = os.path.join(args.save_path, filename[-2], f'{epoch}_{filename[-1]}')
                
                # 可视化时使用原始图像
                # visualize_depth(img, pred, depth, output_path)
                pred_16bit = (pred / args.max_depth * 65535).to(torch.uint16)
                cv2.imwrite(output_path, pred_16bit.cpu().numpy())
            
            valid_mask = (
                (valid_mask == 1)
                & (depth >= args.min_depth)
                & (depth <= args.max_depth)
            )

            if valid_mask.sum() < 10:
                continue

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            logger.info(
                "=========================================================================================="
            )
            logger.info(
                "{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(
                    *tuple(results.keys())
                )
            )
            logger.info(
                "{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}".format(
                    *tuple([(v / nsamples).item() for v in results.values()])
                )
            )
            logger.info(
                "=========================================================================================="
            )
            print()

            for name, metric in results.items():
                writer.add_scalar(f"eval/{name}", (metric / nsamples).item(), epoch)
        
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

        if rank == 0:
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
