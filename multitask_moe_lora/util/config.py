#!/usr/bin/env python3
"""
配置管理模块
处理命令行参数解析、配置验证和默认值设置
"""

import argparse
import os
import math
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
    seg_head_type: str = "linear"
    camera_head_mode: str = "none"
    camera_loss_type: str = "l1"
    camera_backbone_loss_scale: float = 1.0
    camera_head_loss_scale: float = 1.0

    # 模式选择参数
    mode: str = "original"  # 可选择: "original", "lora-only", "legacy-lora", "endo-unid"

    # LoRA 参数
    use_lora: bool = False  # 由mode参数自动设置
    lora_r: int = 0
    lora_alpha: int = 1

    # EndoUniD 专用参数
    endo_unid_shared_shards: int = 1
    endo_unid_shared_r: int = 4
    endo_unid_shared_alpha: int = 8
    endo_unid_depth_r: int = 8
    endo_unid_depth_alpha: int = 16
    endo_unid_seg_r: int = 8
    endo_unid_seg_alpha: int = 16
    endo_unid_camera_r: int = 4
    endo_unid_camera_alpha: int = 8
    endo_unid_dropout: float = 0.0

    # MoE 参数
    use_moe: bool = False  # 由mode参数自动设置
    num_experts: int = 8
    top_k: int = 2

    # 训练参数
    epochs: int = 50
    bs: int = 4
    seg_bs: Optional[int] = None
    val_bs: int = 1
    lr: float = 1e-5
    lr_depth: Optional[float] = None  # 深度任务学习率
    lr_seg: Optional[float] = None    # 分割任务学习率
    lr_camera: Optional[float] = None  # 相机头学习率
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float = 1.0
    val_interval: int = 1
    save_interval: int = 1
    massive_checkpoint: bool = False
    
    # 损失加权
    depth_loss_weight: float = 1.0
    seg_loss_weight: float = 1.0
    camera_loss_weight: float = 1.0
    disable_seg_head: bool = False
    tolerate_validation_errors: bool = False

    # 数据参数
    img_size: int = 518
    dataset_config_name: str = "server_hk_01"
    path_transform_name: Optional[str] = "sz_to_hk"
    max_samples_per_dataset: Optional[int] = None
    train_sample_step: int = 1
    val_sample_step: int = 1
    val_min_samples_per_dataset: int = 0
    dataset_modality: str = "mt"
    train_dataset_include: Optional[List[str]] = None
    val_dataset_include: Optional[List[str]] = None
    local_cache_dir: Optional[str] = None

    # Demo / visualization outputs
    demo_output_root: Optional[str] = None

    # 其他参数
    resume_from: str = ""
    resume_full_state: bool = False
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
    parser.add_argument("--seg-head-type",
                        default="linear",
                        choices=["linear", "sf", "none"],
                        help="Segmentation head architecture (linear BN head, SegFormer-style head, or 'none' to disable)")
    parser.add_argument("--camera-head-mode",
                        default="none",
                        choices=["none", "simple", "prolike", "vggtlike", "vggt-like"],
                        help="Camera head variant: none, simple transformer, Pro-like, or VGGT-like head")
    parser.add_argument("--camera-loss-type",
                        default="l1",
                        choices=["l1", "l2"],
                        help="Camera-head loss type (relative L1 or relative L2)")
    parser.add_argument("--camera-backbone-loss-scale",
                        default=1.0,
                        type=float,
                        help="Multiplier applied to camera loss when updating backbone/depth parameters")
    parser.add_argument("--camera-head-loss-scale",
                        default=1.0,
                        type=float,
                        help="Multiplier applied to camera loss for the camera head itself")

    # 模式选择和PEFT参数
    parser.add_argument("--mode",
                        default="original",
                        choices=["original", "lora-only", "legacy-lora", "endo-unid"],
                        help="Training mode: original, lora-only, legacy-lora (attention-only LoRA), or endo-unid (task-split LoRA)")

    parser.add_argument("--lora-r", default=4, type=int, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", default=8, type=int, help="LoRA alpha")

    # EndoUniD specific knobs
    parser.add_argument("--endo-unid-shared-shards", default=2, type=int,
                        help="Number of shards for shared adapters in EndoUniD mode")
    parser.add_argument("--endo-unid-shared-r", default=4, type=int,
                        help="Shared adapter rank for EndoUniD")
    parser.add_argument("--endo-unid-shared-alpha", default=8, type=int,
                        help="Shared adapter alpha for EndoUniD")
    parser.add_argument("--endo-unid-depth-r", default=8, type=int,
                        help="Depth-only adapter rank for EndoUniD")
    parser.add_argument("--endo-unid-depth-alpha", default=16, type=int,
                        help="Depth-only adapter alpha for EndoUniD")
    parser.add_argument("--endo-unid-seg-r", default=8, type=int,
                        help="Seg-only adapter rank for EndoUniD")
    parser.add_argument("--endo-unid-seg-alpha", default=16, type=int,
                        help="Seg-only adapter alpha for EndoUniD")
    parser.add_argument("--endo-unid-camera-r", default=4, type=int,
                        help="Camera-head adapter rank for EndoUniD")
    parser.add_argument("--endo-unid-camera-alpha", default=8, type=int,
                        help="Camera-head adapter alpha for EndoUniD")
    parser.add_argument("--endo-unid-dropout", default=0.0, type=float,
                        help="Adapter dropout for EndoUniD LoRA")

    parser.add_argument("--num-experts", default=8, type=int, help="Number of experts in MoE")
    parser.add_argument("--top-k", default=2, type=int, help="Number of experts to use for each token")

    # 训练参数
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--bs", default=4, type=int, help="Batch size")
    parser.add_argument("--seg-bs", default=None, type=int, help="Batch size for segmentation training (default: same as --bs)")
    parser.add_argument("--val-bs", default=1, type=int, help="Batch size for validation")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate (deprecated, use --lr-depth and --lr-seg for separate control)")
    parser.add_argument("--lr-depth", default=None, type=float, help="Learning rate for depth task (overrides --lr for depth)")
    parser.add_argument("--lr-seg", default=None, type=float, help="Learning rate for segmentation task (overrides --lr * 10 for seg)")
    parser.add_argument("--lr-camera", default=None, type=float, help="Learning rate for camera head (overrides auto scaling)")
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int, help="Number of gradient accumulation steps")
    parser.add_argument("--clip-grad-norm", default=1.0, type=float,
                        help="Max gradient norm for clip_grad_norm_ (set <=0 to disable clipping)")
    parser.add_argument("--val-interval", default=1, type=int, help="Run validation every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="Save checkpoint every N epochs")
    parser.add_argument("--massive-checkpoint", action="store_true", help="Save checkpoint for every single epoch")
    
    # 损失加权设置
    parser.add_argument("--depth-loss-weight", default=1.0, type=float,
                        help="Base depth loss multiplier used when computing UWL weights")
    parser.add_argument("--seg-loss-weight", default=1.0, type=float,
                        help="Base segmentation loss multiplier used when computing UWL weights")
    parser.add_argument("--camera-loss-weight", default=1.0, type=float,
                        help="Weight applied to camera intrinsics L1 loss when camera head is enabled.")
    parser.add_argument("--disable-seg-head", action="store_true",
                        help="Disable segmentation head entirely (depth-only training/inference)")
    
    # 数据参数
    parser.add_argument("--img-size", default=518, type=int)
    parser.add_argument("--dataset-config-name", type=str, default="server_hk_01",
                        choices=['server_sz', 'server_hk_01', 'no_ls_v1', 'no_ls_local', 'no_only_v1', 'ls_only_v1', 'ls_only_v1_filelist', 'fd_depth_fm_v1'],
                        help="Name of the dataset configuration to use (e.g., 'server_sz', 'server_hk_01')")
    parser.add_argument("--path-transform-name", type=str, default="sz_to_hk",
                        choices=['sz_to_hk', 'no_ls_default', 'no_only_default', 'ls_default', 'none'],
                        help="Name of the path transformation config to use (e.g., 'sz_to_hk', or 'none' to disable)")
    parser.add_argument("--max-samples-per-dataset", type=int, default=None,
                        help="Limit each individual dataset to at most N samples (useful for debugging)")
    parser.add_argument("--train-sample-step", type=int, default=1,
                        help="Stride for epoch-cycling sampling. Values >1 will iterate indices [offset, offset+step, ...] per epoch.")
    parser.add_argument("--val-sample-step", type=int, default=1,
                        help="Stride for validation sampling (e.g., 20 keeps every 20th sample).")
    parser.add_argument("--val-min-samples-per-dataset", type=int, default=0,
                        help="Minimum number of samples to keep per validation dataset. When set and step is too large or -1, samples are evenly spaced.")
    parser.add_argument("--dataset-modality",
                        default="mt",
                        choices=["mt", "fd"],
                        help="Dataset modality subset: mt (multi-task) or fd (depth-focused)")
    parser.add_argument("--dataset-include",
                        type=str,
                        default=None,
                        help="(Deprecated) same as --train-dataset-include; kept for backward compatibility.")
    parser.add_argument("--train-dataset-include",
                        type=str,
                        default=None,
                        help="Comma-separated dataset names to keep in training (e.g., 'EndoSynth').")
    parser.add_argument("--val-dataset-include",
                        type=str,
                        default=None,
                        help="Comma-separated dataset names to keep in validation.")
    parser.add_argument("--local-cache-dir",
                        type=str,
                        default=os.environ.get("LOCAL_CACHE_DIR") or os.environ.get("LOCAL_CACHE_PATH"),
                        help="Optional local directory for mirroring remote caches / samples (.pt). Falls back to LOCAL_CACHE_DIR or LOCAL_CACHE_PATH env vars.")

    # 其他参数
    parser.add_argument("--resume-from", type=str, required=False, help="Path to checkpoint to resume training from or load pretrained weights")
    parser.add_argument("--resume-full-state", action="store_true", help="If set, resumes full training state (optimizer, scheduler, epoch) from checkpoint")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--tolerate-validation-errors", action="store_true",
                        help="Skip over validation batches that raise errors instead of aborting training")
    parser.add_argument("--dinov3-repo-path", type=str, default="/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3", help="Path to the local dinov3 repository")
    parser.add_argument("--demo-output-root", type=str, default=None, help="Optional root directory to store demo / sample visualizations")

    return parser


def args_to_config(args: argparse.Namespace) -> TrainingConfig:
    """将argparse.Namespace转换为TrainingConfig"""

    # 根据mode参数设置use_lora（MoE 模式已移除）
    mode = getattr(args, 'mode', 'original')
    if mode in ('endo-unid', 'lora-only', 'legacy-lora'):
        use_lora = True
    else:
        use_lora = False
    use_moe = False

    def _parse_list(value: Optional[str]) -> Optional[List[str]]:
        if not value:
            return None
        return [item.strip() for item in value.split(',') if item.strip()]

    legacy_include = _parse_list(getattr(args, 'dataset_include', None))
    train_include_list = _parse_list(getattr(args, 'train_dataset_include', None)) or legacy_include
    val_include_list = _parse_list(getattr(args, 'val_dataset_include', None)) or legacy_include

    raw_seg_head_type = getattr(args, 'seg_head_type', "linear") or "linear"
    seg_head_type = raw_seg_head_type.lower()
    disable_seg_head = getattr(args, 'disable_seg_head', False) or seg_head_type == "none"

    return TrainingConfig(encoder=args.encoder,
                          features=args.features,
                          num_classes=getattr(args, 'num_classes', 3),
                          min_depth=getattr(args, 'min_depth', 1e-6),
                          max_depth=getattr(args, 'max_depth', 0.2),
                          frozen_backbone=getattr(args, 'frozen_backbone', False),
                          seg_input_type=getattr(args, 'seg_input_type', "last_four"),
                          seg_head_type=seg_head_type,
                          camera_head_mode=getattr(args, 'camera_head_mode', "none").lower(),
                          camera_loss_type=getattr(args, 'camera_loss_type', "l1").lower(),
                          camera_backbone_loss_scale=getattr(args, 'camera_backbone_loss_scale', 1.0),
                          camera_head_loss_scale=getattr(args, 'camera_head_loss_scale', 1.0),
                          mode=mode,
                          use_lora=use_lora,
                          use_moe=use_moe,
                          num_experts=getattr(args, 'num_experts', 8),
                          top_k=getattr(args, 'top_k', 2),
                          lora_r=getattr(args, 'lora_r', 4),
                          lora_alpha=getattr(args, 'lora_alpha', 8),
                          epochs=args.epochs,
                          bs=args.bs,
                          seg_bs=getattr(args, 'seg_bs', None),
                          val_bs=getattr(args, 'val_bs', 1),
                          lr=args.lr,
                          lr_depth=getattr(args, 'lr_depth', None),
                          lr_seg=getattr(args, 'lr_seg', None),
                          lr_camera=getattr(args, 'lr_camera', None),
                          weight_decay=getattr(args, 'weight_decay', 0.01),
                          gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
                          clip_grad_norm=getattr(args, 'clip_grad_norm', 1.0),
                          val_interval=getattr(args, 'val_interval', 1),
                          save_interval=getattr(args, 'save_interval', 1),
                          massive_checkpoint=getattr(args, 'massive_checkpoint', False),
                          depth_loss_weight=getattr(args, 'depth_loss_weight', 1.0),
                          seg_loss_weight=getattr(args, 'seg_loss_weight', 1.0),
                          camera_loss_weight=getattr(args, 'camera_loss_weight', 1.0),
                          disable_seg_head=disable_seg_head,
                          img_size=getattr(args, 'img_size', 518),
                          dataset_config_name=getattr(args, 'dataset_config_name', 'server_hk_01'),
                          path_transform_name=getattr(args, 'path_transform_name', 'sz_to_hk'),
                          max_samples_per_dataset=getattr(args, 'max_samples_per_dataset', None),
                          train_sample_step=getattr(args, 'train_sample_step', 1),
                          val_sample_step=getattr(args, 'val_sample_step', 1),
                          val_min_samples_per_dataset=getattr(args, 'val_min_samples_per_dataset', 0),
                          dataset_modality=getattr(args, 'dataset_modality', 'mt'),
                          train_dataset_include=train_include_list,
                          val_dataset_include=val_include_list,
                          local_cache_dir=getattr(args, 'local_cache_dir', None),
                          demo_output_root=getattr(args, 'demo_output_root', None),
                          tolerate_validation_errors=getattr(args, 'tolerate_validation_errors', False),
                          resume_from=getattr(args, 'resume_from', ""),
                          resume_full_state=getattr(args, 'resume_full_state', False),
                          save_path=getattr(args, 'save_path', ""),
                          local_rank=getattr(args, 'local_rank', 0),
                          port=getattr(args, 'port', None),
                          mixed_precision=getattr(args, 'mixed_precision', False),
                          dinov3_repo_path=getattr(args, 'dinov3_repo_path', "/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3"),
                          endo_unid_shared_shards=getattr(args, 'endo_unid_shared_shards', 1),
                          endo_unid_shared_r=getattr(args, 'endo_unid_shared_r', 4),
                          endo_unid_shared_alpha=getattr(args, 'endo_unid_shared_alpha', 8),
                          endo_unid_depth_r=getattr(args, 'endo_unid_depth_r', 8),
                          endo_unid_depth_alpha=getattr(args, 'endo_unid_depth_alpha', 16),
                          endo_unid_seg_r=getattr(args, 'endo_unid_seg_r', 8),
                          endo_unid_seg_alpha=getattr(args, 'endo_unid_seg_alpha', 16),
                          endo_unid_camera_r=getattr(args, 'endo_unid_camera_r', 4),
                          endo_unid_camera_alpha=getattr(args, 'endo_unid_camera_alpha', 8),
                          endo_unid_dropout=getattr(args, 'endo_unid_dropout', 0.0))


def validate_config(config: TrainingConfig) -> List[str]:
    """验证配置参数的合理性"""
    errors = []

    # 检查必需的路径参数
    if not config.save_path:
        errors.append("save_path is required")

    # 检查 --resume-from 和 --pretrained-from 是否冲突

    # 检查数值参数的合理性
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.bs <= 0:
        errors.append("batch size must be positive")
    if config.val_bs <= 0:
        errors.append("validation batch size must be positive")
    if config.lr <= 0:
        errors.append("learning rate must be positive")
    if config.lr_camera is not None and config.lr_camera <= 0:
        errors.append("lr_camera must be positive if specified")
    if config.num_classes <= 0:
        errors.append("num_classes must be positive")
    if config.min_depth <= 0:
        errors.append("min_depth must be positive")
    if config.max_depth <= config.min_depth:
        errors.append("max_depth must be greater than min_depth")
    if config.val_min_samples_per_dataset < 0:
        errors.append("val_min_samples_per_dataset must be >= 0")
    if not math.isfinite(config.clip_grad_norm):
        errors.append("clip_grad_norm must be a finite float")

    # 检查seg_bs的合理性
    if config.seg_bs is not None and config.seg_bs <= 0:
        errors.append("seg_bs must be positive if specified")

    if config.max_samples_per_dataset is not None and config.max_samples_per_dataset <= 0:
        errors.append("max_samples_per_dataset must be positive if specified")
    if config.train_sample_step <= 0:
        errors.append("train_sample_step must be positive")
    if config.val_sample_step == 0:
        errors.append("val_sample_step must be positive or -1")
    if config.val_sample_step < -1:
        errors.append("val_sample_step must be >= -1")
    if config.camera_loss_weight < 0:
        errors.append("camera_loss_weight must be non-negative")
    if config.camera_backbone_loss_scale < 0:
        errors.append("camera_backbone_loss_scale must be non-negative")
    if config.camera_head_loss_scale < 0:
        errors.append("camera_head_loss_scale must be non-negative")

    # 检查encoder和seg_input_type的兼容性
    if config.seg_input_type == "from_depth" and config.frozen_backbone:
        errors.append("from_depth seg_input_type may not work well with frozen_backbone")

    valid_camera_modes = {"none", "simple", "vggtlike", "vggt-like"}
    if config.camera_head_mode.lower() not in valid_camera_modes:
        errors.append(f"camera_head_mode must be one of {sorted(valid_camera_modes)}")
    if config.camera_loss_type.lower() not in {"l1", "l2"}:
        errors.append("camera_loss_type must be either 'l1' or 'l2'")

    # EndoUniD checks
    if config.mode == "endo-unid":
        if config.encoder not in {"vits", "vitb"}:
            errors.append("EndoUniD mode currently only supports vits or vitb encoders")
        if config.endo_unid_shared_shards <= 0:
            errors.append("endo_unid_shared_shards must be positive")
        for label, rank in [("shared", config.endo_unid_shared_r),
                            ("depth", config.endo_unid_depth_r),
                            ("seg", config.endo_unid_seg_r),
                            ("camera", config.endo_unid_camera_r)]:
            if rank < 0:
                errors.append(f"EndoUniD {label} rank must be non-negative")
        if config.endo_unid_dropout < 0 or config.endo_unid_dropout > 1:
            errors.append("endo_unid_dropout must be within [0, 1]")

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

    # 如果 transform name 是 'none'，则将其设为 None
    if config.path_transform_name and config.path_transform_name.lower() == 'none':
        config.path_transform_name = None
        
    return config
