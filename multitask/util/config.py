#!/usr/bin/env python3
"""
配置管理模块
处理命令行参数解析、配置验证和默认值设置
"""

import argparse
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    # 模型参数
    encoder: str = "vits"
    features: int = 64
    num_classes: int = 3
    min_depth: float = 1e-6
    max_depth: float = 0.2
    frozen_backbone: bool = False
    seg_input_type: str = "last_four"

    # 训练参数
    epochs: int = 50
    bs: int = 4
    seg_bs: Optional[int] = None
    val_bs: int = 1
    lr: float = 1e-5
    lr_depth: Optional[float] = None  # 深度任务学习率
    lr_seg: Optional[float] = None    # 分割任务学习率
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    val_interval: int = 1
    save_interval: int = 1

    # 数据参数
    img_size: int = 518

    # 其他参数
    pretrained_from: str = ""
    resume_from: str = ""
    save_path: str = ""
    local_rank: int = 0
    port: Optional[int] = None
    mixed_precision: bool = False
    dinov3_repo_path: str = "/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3"


def create_parser() -> argparse.ArgumentParser:
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description="DepthAnythingV2 MultiTask Training (Depth + Segmentation)")

    # 模型参数
    parser.add_argument("--encoder",
                        default="vits",
                        choices=["vits", "vitb", "vitl", "vitg", "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16"])
    parser.add_argument("--features", default=64, type=int, help="Feature dimension")
    parser.add_argument("--num-classes", default=3, type=int, help="Number of segmentation classes")
    parser.add_argument("--min-depth", default=1e-6, type=float, help="Minimum depth value")
    parser.add_argument("--max-depth", default=0.2, type=float, help="Maximum depth value for model")
    parser.add_argument("--frozen-backbone", action="store_true", help="Freeze the backbone and only train the heads")
    parser.add_argument("--seg-input-type",
                        default="last_four",
                        choices=["last", "last_four", "from_depth"],
                        help="Input type for segmentation head ('last', 'last_four', 'from_depth')")

    # 训练参数
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--bs", default=4, type=int, help="Batch size")
    parser.add_argument("--seg-bs", default=None, type=int, help="Batch size for segmentation training (default: same as --bs)")
    parser.add_argument("--val-bs", default=1, type=int, help="Batch size for validation")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate (deprecated, use --lr-depth and --lr-seg for separate control)")
    parser.add_argument("--lr-depth", default=None, type=float, help="Learning rate for depth task (overrides --lr for depth)")
    parser.add_argument("--lr-seg", default=None, type=float, help="Learning rate for segmentation task (overrides --lr * 10 for seg)")
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int, help="Number of gradient accumulation steps")
    parser.add_argument("--val-interval", default=1, type=int, help="Run validation every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="Save checkpoint every N epochs")
 
    # 数据参数
    parser.add_argument("--img-size", default=518, type=int)

    # 其他参数
    parser.add_argument("--pretrained-from", type=str, required=False, help="Path to pretrained weights (optional for evaluation)")
    parser.add_argument("--resume-from", type=str, required=False, help="Path to checkpoint to resume training from")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--dinov3-repo-path", type=str, default="/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3", help="Path to the local dinov3 repository")

    return parser


def args_to_config(args: argparse.Namespace) -> TrainingConfig:
    """将argparse.Namespace转换为TrainingConfig"""
    return TrainingConfig(encoder=args.encoder,
                          features=args.features,
                          num_classes=getattr(args, 'num_classes', 3),
                          min_depth=getattr(args, 'min_depth', 1e-6),
                          max_depth=getattr(args, 'max_depth', 0.2),
                          frozen_backbone=getattr(args, 'frozen_backbone', False),
                          seg_input_type=getattr(args, 'seg_input_type', "last_four"),
                          epochs=args.epochs,
                          bs=args.bs,
                          seg_bs=getattr(args, 'seg_bs', None),
                          val_bs=getattr(args, 'val_bs', 1),
                          lr=args.lr,
                          lr_depth=getattr(args, 'lr_depth', None),
                          lr_seg=getattr(args, 'lr_seg', None),
                          weight_decay=getattr(args, 'weight_decay', 0.01),
                          gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
                          val_interval=getattr(args, 'val_interval', 1),
                          save_interval=getattr(args, 'save_interval', 1),
                          img_size=getattr(args, 'img_size', 518),
                          pretrained_from=getattr(args, 'pretrained_from', ""),
                          resume_from=getattr(args, 'resume_from', ""),
                          save_path=getattr(args, 'save_path', ""),
                          local_rank=getattr(args, 'local_rank', 0),
                          port=getattr(args, 'port', None),
                          mixed_precision=getattr(args, 'mixed_precision', False),
                          dinov3_repo_path=getattr(args, 'dinov3_repo_path', "/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3"))


def validate_config(config: TrainingConfig) -> List[str]:
    """验证配置参数的合理性"""
    errors = []

    # 检查必需的路径参数
    if not config.save_path:
        errors.append("save_path is required")

    # 检查 --resume-from 和 --pretrained-from 是否冲突
    if config.resume_from and config.pretrained_from:
        errors.append("Cannot use both --resume-from and --pretrained-from simultaneously")

    # 检查数值参数的合理性
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.bs <= 0:
        errors.append("batch size must be positive")
    if config.val_bs <= 0:
        errors.append("validation batch size must be positive")
    if config.lr <= 0:
        errors.append("learning rate must be positive")
    if config.num_classes <= 0:
        errors.append("num_classes must be positive")
    if config.min_depth <= 0:
        errors.append("min_depth must be positive")
    if config.max_depth <= config.min_depth:
        errors.append("max_depth must be greater than min_depth")

    # 检查seg_bs的合理性
    if config.seg_bs is not None and config.seg_bs <= 0:
        errors.append("seg_bs must be positive if specified")

    # 检查encoder和seg_input_type的兼容性
    if config.seg_input_type == "from_depth" and config.frozen_backbone:
        errors.append("from_depth seg_input_type may not work well with frozen_backbone")

    return errors


def parse_and_validate_config() -> TrainingConfig:
    """解析并验证配置"""
    parser = create_parser()
    args = parser.parse_args()
    config = args_to_config(args)

    # 验证配置
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration")

    return config
