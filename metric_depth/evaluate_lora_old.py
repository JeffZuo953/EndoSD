import argparse
import logging
import os
import pprint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.c3vd import C3VD
from dataset.simcol import Simcol
from dataset.endomapper import Endomapper
from depth_anything_v2.dpt_lora import DepthAnythingV2_LoRA
from util.metric import eval_depth
from util.utils import init_log



parser = argparse.ArgumentParser(
    description="Depth Anything V2 LoRA Evaluation for Metric Depth Estimation"
)

parser.add_argument(
    "--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
)
parser.add_argument(
    "--dataset",
    default="hypersim",
    choices=["hypersim", "vkitti", "kitti", "c3vd", "simcol", "endomapper", "combined"]
)
parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--min-depth", default=0.001, type=float)
parser.add_argument("--max-depth", default=20, type=float)
parser.add_argument("--load-from", type=str, required=True, help="Path to LoRA checkpoint")
parser.add_argument("--save-path", type=str, default=None, help="Path to save visualizations")

parser.add_argument("--lora_r", default=0, type=int, help="LoRA rank (0 means no LoRA)")
parser.add_argument("--lora_alpha", default=1, type=int, help="LoRA alpha scaling factor")
parser.add_argument("--lora_dropout", default=0.0, type=float, help="LoRA dropout probability")
parser.add_argument("--lora_bias", default="none", choices=["none", "lora_only", "all"])
parser.add_argument("--dino_weights", type=str, default=None, help="Path to DINO pretrained weights")

def merge_state_dicts(lora_weights, backbone_weights, logger):
    """
    合并两个状态字典，以backbone为基础，lora权重优先
    
    Args:
        lora_weights (dict): LoRA权重字典（高优先级）
        backbone_weights (dict): 骨干网络权重字典（基础）
        logger (logging.Logger): 日志记录器
    
    Returns:
        dict: 合并后的状态字典
    """
    # 以backbone_weights为基础创建新字典
    merged_dict = backbone_weights.copy()
    
    logger.info("===== Merging State Dictionaries =====")
    for key, value in lora_weights.items():
        if key in merged_dict:
            # 检查形状是否匹配
            if merged_dict[key].shape == value.shape:
                logger.info(f"Updating existing key with LoRA weights: {key}")
                merged_dict[key] = value
            else:
                logger.warning(f"Shape mismatch for key {key}:")
                logger.warning(f"  Backbone Shape: {merged_dict[key].shape}")
                logger.warning(f"  LoRA Shape: {value.shape}")
                logger.warning(f"  Using LoRA weights despite shape mismatch.")
                merged_dict[key] = value
        else:
            logger.info(f"Adding new key from LoRA weights: {key}")
            merged_dict[key] = value
    
    return merged_dict

def load_weights_from_checkpoints(model, checkpoint_path, dino_weights_path, args, script_dir, logger):
    """
    从多个检查点加载权重，并进行权重合并和验证
    
    Args:
        model: 要加载权重的模型
        checkpoint_path (str): LoRA检查点路径
        dino_weights_path (str): DINO预训练权重路径
        args: 命令行参数
        script_dir (str): 脚本所在目录
        logger (logging.Logger): 日志记录器
    
    Returns:
        dict: 合并后的权重字典
    """
    # 加载 LoRA 检查点
    logger.info(f"Loading LoRA checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 检查 LoRA 检查点的一致性
    if "model_trainable_parts" not in checkpoint:
        logger.error("Checkpoint does not contain 'model_trainable_parts' key.")
        raise ValueError("Invalid checkpoint format")
    
    # 提取 LoRA 权重
    lora_weights = {k.replace("pretrained.", "", 1): v 
                    for k, v in checkpoint["model_trainable_parts"].items() 
                    if k.startswith("pretrained.")}
    head_weights = {k.replace("depth_head.", "", 1): v 
                    for k, v in checkpoint["model_trainable_parts"].items() 
                    if k.startswith("depth_head.")}
    
    # 加载 DINO 权重（如果提供）
    backbone_weights = {}
    if dino_weights_path and os.path.exists(dino_weights_path):
        logger.info(f"Loading DINO pretrained weights from: {dino_weights_path}")
        dino_state_dict = torch.load(dino_weights_path, map_location='cpu')
        
        # 处理不同的权重格式
        if 'state_dict' in dino_state_dict:
            dino_state_dict = dino_state_dict['state_dict']
        elif 'model' in dino_state_dict:
            dino_state_dict = dino_state_dict['model']
        
        # 提取骨干网络权重
        for k, v in dino_state_dict.items():
            if k.startswith('pretrained.') :
                new_key = k.replace('pretrained.', '')
                backbone_weights[new_key] = v

            if k.startswith('depth_head.'):
                new_key = k.replace('depth_head.', '')
                backbone_weights[new_key] = v
        

    # 合并 LoRA 和 DINO 权重
    merged_lora_weights = merge_state_dicts(lora_weights, backbone_weights, logger)
    
    # 获取模型的实际权重
    model_lora_state_dict = model.pretrained.state_dict()
    model_head_state_dict = model.depth_head.state_dict()
    
    # 详细检查权重合并后的一致性
    logger.info("\n===== Merged Weights Detailed Validation =====")
    
    # 验证 LoRA 权重
    lora_consistent = True
    for key in model_lora_state_dict.keys():
        if key in merged_lora_weights:
            # 检查形状是否匹配
            if model_lora_state_dict[key].shape != merged_lora_weights[key].shape:
                logger.warning(f"LoRA Weight Shape Mismatch for key '{key}':")
                logger.warning(f"  Model Shape: {model_lora_state_dict[key].shape}")
                logger.warning(f"  Merged Weights Shape: {merged_lora_weights[key].shape}")
                lora_consistent = False
        else:
            logger.warning(f"LoRA Weight Key Missing in Merged Weights: '{key}'")
            lora_consistent = False
    
    # 验证深度头权重
    head_consistent = True
    for key in model_head_state_dict.keys():
        if key in head_weights:
            # 检查形状是否匹配
            if model_head_state_dict[key].shape != head_weights[key].shape:
                logger.warning(f"Depth Head Weight Shape Mismatch for key '{key}':")
                logger.warning(f"  Model Shape: {model_head_state_dict[key].shape}")
                logger.warning(f"  Checkpoint Weights Shape: {head_weights[key].shape}")
                head_consistent = False
        else:
            logger.warning(f"Depth Head Weight Key Missing in Checkpoint: '{key}'")
            head_consistent = False
    
    # 总体一致性
    is_consistent = lora_consistent and head_consistent
    logger.info(f"\nMerged Weights Consistency: {'CONSISTENT' if is_consistent else 'INCONSISTENT'}")
    
    # 根据一致性选择加载策略
    if is_consistent:
        # 如果完全一致，使用 strict=True
        model.pretrained.load_state_dict(merged_lora_weights, strict=True)
        model.depth_head.load_state_dict(head_weights, strict=True)
        logger.info("Weights loaded with strict matching.")
    else:
        # 如果不完全一致，使用 strict=False 并记录警告
        logger.warning("Merged weight keys or shapes are not fully consistent. Using non-strict loading.")
        model.pretrained.load_state_dict(merged_lora_weights, strict=False)
        model.depth_head.load_state_dict(head_weights, strict=False)
    
    return checkpoint

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
        val_split_path = os.path.join(project_root, "dataset/splits/hypersim/val.txt")
        valset = Hypersim(val_split_path, "val", size=size)
    elif args.dataset == "vkitti":
        val_split_path = os.path.join(project_root, "dataset/splits/vkitti2/val.txt")
        valset = VKITTI2(val_split_path, "val", size=size)
    elif args.dataset == "kitti":
        val_split_path = os.path.join(project_root, "dataset/splits/kitti/val.txt")
        valset = KITTI(val_split_path, "val", size=size)
    elif args.dataset == "c3vd":
        val_split_path = os.path.join(project_root, "dataset/splits/c3vd/val.txt")
        valset = C3VD(val_split_path, "val", size=size)
    elif args.dataset == "simcol":
        val_split_path = os.path.join(project_root, "dataset/splits/simcol/val.txt")
        valset = Simcol(val_split_path, "val", size=size)
    elif args.dataset == "endomapper":
        val_split_path = os.path.join(project_root, "dataset/splits/endomapper/val.txt")
        valset = Endomapper(val_split_path, "val", size=size)
    elif args.dataset == "combined":
        # Combine datasets for evaluation
        c3vd_val_path = os.path.join(project_root, 'metric_depth', "dataset/splits/c3vd/val.txt")
        simcol_val_path = os.path.join(project_root, 'metric_depth', "dataset/splits/simcol/val.txt")
        
        valset = torch.utils.data.ConcatDataset([
            C3VD(c3vd_val_path, "val", size=size),
            Simcol(simcol_val_path, "val", size=size),
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
        "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
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

    # 解析检查点和 DINO 权重路径
    if not os.path.isabs(args.load_from):
        checkpoint_path = os.path.abspath(os.path.join(script_dir, args.load_from))
    else:
        checkpoint_path = args.load_from

    if args.dino_weights:
        if not os.path.isabs(args.dino_weights):
            dino_weights_path = os.path.abspath(os.path.join(script_dir, args.dino_weights))
        else:
            dino_weights_path = args.dino_weights
    else:
        dino_weights_path = None

    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if dino_weights_path and not os.path.exists(dino_weights_path):
        logger.error(f"DINO weights not found: {dino_weights_path}")
        raise FileNotFoundError(f"DINO weights not found: {dino_weights_path}")

    # 使用新的权重加载函数
    checkpoint = load_weights_from_checkpoints(
        model, 
        checkpoint_path, 
        dino_weights_path, 
        args, 
        script_dir, 
        logger
    )

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
            pred_interpolated = F.interpolate(
                pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
            )[0, 0]
            
            # 直接处理单通道深度值
            pred_single = pred_interpolated
            depth_single = depth
            valid_mask_single = valid_mask
            
            eval_mask = (
                (valid_mask_single == 1) & 
                (depth_single >= args.min_depth) & 
                (depth_single <= args.max_depth)
            )
            
            if eval_mask.sum() < 10:
                continue
            
            cur_results = eval_depth(pred_single[eval_mask], depth_single[eval_mask])
            
            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1
                
            if i % 100 == 0:
                logger.info(f'Processed {i}/{len(valloader)} images')
                
            if args.save_path and i < 10:
                filename = sample["image_path"][0].split("/")[-2:]
                os.makedirs(args.save_path, exist_ok=True)
                os.makedirs(os.path.join(args.save_path, filename[-2]), exist_ok=True)
                output_path = os.path.join(args.save_path, filename[-2], filename[-1])
                print(output_path)
                # 修改保存逻辑
                pred_single = pred_interpolated.squeeze().cpu().numpy()
                
                # 归一化到 0-255 范围
                pred_normalized = (pred_single - pred_single.min()) / (pred_single.max() - pred_single.min()) * 65535
                pred_16bit = pred_normalized.astype(np.uint16)
                
                # 使用 OpenCV 保存为 PNG
                cv2.imwrite(output_path, pred_16bit)
                

    # Calculate final results
    if nsamples > 0:
        final_results = {k: v / nsamples for k, v in results.items()}
        logger.info("================ Evaluation Results ================")
        logger.info(
            "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
                *final_results.keys()
            )
        )
        logger.info(
            "{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
                *final_results.values()
            )
        )
        logger.info("===================================================")
    else:
        logger.warning("No validation samples were evaluated.")

if __name__ == "__main__":
    main()
