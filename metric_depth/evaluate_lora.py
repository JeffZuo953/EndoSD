import argparse
import logging
import os
import pprint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2
import re

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.c3vd import C3VD
from dataset.simcol import Simcol
from dataset.endomapper import Endomapper
from depth_anything_v2.dpt_lora import DepthAnythingV2_LoRA
from util.metric import eval_depth
from util.utils import init_log
from collections import OrderedDict


def strip_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    return new_state_dict


def verify_model_weights_loaded(model: torch.nn.Module,
                                processed_state_dict_to_load: OrderedDict,
                                logger: logging.Logger = None):

    if logger is None:
        log_fn = print
    else:
        log_fn = logger.info

    model_state_dict = model.state_dict()  # 这将在模型的当前设备上
    model_keys = set(model_state_dict.keys())

    loaded_keys = set(processed_state_dict_to_load.keys())

    missing_keys = sorted(list(model_keys - loaded_keys))
    unexpected_keys = sorted(list(loaded_keys - model_keys))

    mismatched_shape_keys = []

    common_keys = model_keys.intersection(loaded_keys)
    for key in sorted(list(common_keys)):
        model_tensor = model_state_dict[key]
        loaded_tensor = processed_state_dict_to_load[key]
        if model_tensor.shape != loaded_tensor.shape:
            mismatched_shape_keys.append({
                "key": key,
                "model_shape": model_tensor.shape,
                "loaded_shape": loaded_tensor.shape
            })

    all_loaded_perfectly = not missing_keys and not unexpected_keys and not mismatched_shape_keys

    log_fn("--- 权重加载验证开始 ---")
    if missing_keys:
        log_fn(f"  缺失的键 (模型期望，但检查点中未找到): {len(missing_keys)}")
        for k in missing_keys:
            log_fn(f"    - {k}")
    if unexpected_keys:
        log_fn(f"  意外的键 (检查点中存在，但模型不期望): {len(unexpected_keys)}")
        for k in unexpected_keys:
            log_fn(f"    - {k}")
    if mismatched_shape_keys:
        log_fn(f"  形状不匹配的键: {len(mismatched_shape_keys)}")
        for item in mismatched_shape_keys:
            log_fn(
                f"    - 键: {item['key']}, 模型形状: {item['model_shape']}, 加载形状: {item['loaded_shape']}"
            )

    if all_loaded_perfectly:
        log_fn("权重加载验证: 所有模型权重与检查点中的键和形状匹配。")
    else:
        log_fn("权重加载验证: 发现潜在问题。请检查以上日志。")
    log_fn("--- 权重加载验证结束 ---")

    return all_loaded_perfectly, missing_keys, unexpected_keys, mismatched_shape_keys


parser = argparse.ArgumentParser(
    description="Depth Anything V2 LoRA Evaluation for Metric Depth Estimation"
)

parser.add_argument("--encoder",
                    default="vitl",
                    choices=["vits", "vitb", "vitl", "vitg"])
parser.add_argument(
    "--dataset",
    default="hypersim",
    choices=[
        "hypersim", "vkitti", "kitti", "c3vd", "simcol", "endomapper",
        "combined"
    ],
)
parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--min-depth", default=0.001, type=float)
parser.add_argument("--max-depth", default=20, type=float)
parser.add_argument("--load-from",
                    type=str,
                    required=True,
                    help="Path to checkpoint")
parser.add_argument("--save-path",
                    type=str,
                    default=None,
                    help="Path to save visualizations")

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
parser.add_argument("--lora_bias",
                    default="none",
                    choices=["none", "lora_only", "all"])


def main():
    args = parser.parse_args()

    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

    logger.info("\n{}\n".format(pprint.pformat(vars(args))))

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)

    # Create dataset based on selection
    if args.dataset == "hypersim":
        val_split_path = os.path.join(project_root,
                                      "dataset/splits/hypersim/val.txt")
        valset = Hypersim(val_split_path, "val", size=size)
    elif args.dataset == "vkitti":
        val_split_path = os.path.join(project_root,
                                      "dataset/splits/vkitti2/val.txt")
        valset = VKITTI2(val_split_path, "val", size=size)
    elif args.dataset == "kitti":
        val_split_path = os.path.join(project_root,
                                      "dataset/splits/kitti/val.txt")
        valset = KITTI(val_split_path, "val", size=size)
    elif args.dataset == "c3vd":
        val_split_path = os.path.join(
            project_root, "metric_depth/dataset/splits/c3vd/val.txt")
        valset = C3VD(val_split_path, "val", size=size)
    elif args.dataset == "simcol":
        val_split_path = os.path.join(
            project_root, "metric_depth/dataset/splits/simcol/val.txt")
        valset = Simcol(val_split_path, "val", size=size)
    elif args.dataset == "endomapper":
        val_split_path = os.path.join(
            project_root, "metric_depth/dataset/splits/endomapper/val.txt")
        valset = Endomapper(val_split_path, "val", size=size)
    elif args.dataset == "combined":
        # Combine datasets for evaluation
        c3vd_val_path = os.path.join(
            project_root, "metric_depth",
            "metric_depth/dataset/splits/c3vd/val.txt")
        simcol_val_path = os.path.join(
            project_root, "metric_depth",
            "metric_depth/dataset/splits/simcol/val.txt")

        valset = torch.utils.data.ConcatDataset([
            C3VD(c3vd_val_path, "val", size=size),
            # Simcol(simcol_val_path, "val", size=size),
        ])
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        shuffle=False,
    )

    # Model Definition
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

    checkpoint_path = args.load_from
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 加载整个检查点
    # weights_only=False 是必需的，因为 train_lora.py 保存了 'args' (argparse.Namespace)
    logger.info(f"正在从 {checkpoint_path} 加载检查点...")
    checkpoint_content = torch.load(checkpoint_path,
                                    map_location="cpu",
                                    weights_only=False)

    # 提取模型的 state_dict
    if isinstance(checkpoint_content, dict) and "model" in checkpoint_content:
        raw_model_state_dict = checkpoint_content["model"]
        if "epoch" in checkpoint_content:
            logger.info(f"检查点 Epoch: {checkpoint_content['epoch']}")
        if "args" in checkpoint_content:  # 打印部分训练参数以供参考
            train_args = checkpoint_content['args']
            logger.info(
                f"训练时 LoRA r: {getattr(train_args, 'lora_r', 'N/A')}, alpha: {getattr(train_args, 'lora_alpha', 'N/A')}"
            )
    elif isinstance(checkpoint_content, OrderedDict):  # 直接是 state_dict
        raw_model_state_dict = checkpoint_content
        logger.info("加载的检查点似乎直接是模型的 state_dict。")
    else:
        logger.error(
            f"来自 {checkpoint_path} 的检查点格式不符合预期（应为包含 'model' 键的字典或直接的 state_dict）。"
        )
        raise ValueError("无效的检查点格式。")

    processed_model_state_dict = strip_module_prefix(raw_model_state_dict)

    # 在尝试加载前验证权重
    # 注意：此时模型在 CPU 上（如果尚未移动）。
    # 为了验证，最好模型和 state_dict 都在 CPU 上。
    verify_model_weights_loaded(model, processed_model_state_dict, logger)

    # 将状态字典加载到模型中
    model.load_state_dict(processed_model_state_dict, strict=True)
    logger.info("模型权重已通过 strict=True 成功加载。")

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize results dictionary
    results = {
        "d1": 0.0,
        "d2": 0.0,
        "d3": 0.0,
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "rmse": 0.0,
        "rmse_log": 0.0,
        "log10": 0.0,
        "silog": 0.0,
    }
    nsamples = 0

    with torch.no_grad():
        for i, sample in enumerate(valloader):
            img = sample["image"].to(device)
            depth = sample["depth"].to(device)[0]
            valid_mask = sample["valid_mask"].to(device)[0]

            # Forward pass
            pred = model(img)

            # Interpolate prediction to match ground truth resolution
            pred_interpolated = F.interpolate(pred[:, None],
                                              depth.shape[-2:],
                                              mode="bilinear",
                                              align_corners=True)[0, 0]

            # 直接处理单通道深度值
            pred_single = pred_interpolated
            depth_single = depth
            valid_mask_single = valid_mask

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

            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(valloader)} images")

            if args.save_path and i < 10:
                filename = sample["image_path"][0].split("/")[-2:]
                os.makedirs(args.save_path, exist_ok=True)
                os.makedirs(os.path.join(args.save_path, filename[-2]),
                            exist_ok=True)
                output_path = os.path.join(args.save_path, filename[-2],
                                           filename[-1])
                print(output_path)
                # 修改保存逻辑
                pred_single = pred_interpolated.squeeze().cpu().numpy()

                # 归一化到 0-255 范围
                pred_normalized = ((pred_single - pred_single.min()) /
                                   (pred_single.max() - pred_single.min()) *
                                   65535)
                pred_16bit = pred_normalized.astype(np.uint16)

                # 使用 OpenCV 保存为 PNG
                cv2.imwrite(output_path, pred_16bit)

    # Calculate final results
    if nsamples > 0:
        final_results = {k: v / nsamples for k, v in results.items()}
        logger.info("================ Evaluation Results ================")
        logger.info(
            "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
                *final_results.keys()))
        logger.info(
            "{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}"
            .format(*final_results.values()))
        logger.info("===================================================")
    else:
        logger.warning("No validation samples were evaluated.")


if __name__ == "__main__":
    main()
