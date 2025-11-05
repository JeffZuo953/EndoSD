import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .dinov2 import DINOv2
from .dinov3 import DINOv3
from .dpt import DPTHead, IntrinsicsHead

# 使用绝对导入避免相对导入问题
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from seg_heads.linear_head.linear_head import LinearBNHead


class DepthAnythingV2_MultiTask(nn.Module):
    """
    多任务模型：基于 DepthAnythingV2 同时进行深度估计和语义分割
    共享 DINOv2 backbone，使用独立的深度头和分割头
    """

    def __init__(self,
                 encoder='vits',
                 num_classes=3,
                 features=256,
                 out_channels=None,
                 use_bn=False,
                 use_clstoken=False,
                 max_depth=20.0,
                 seg_input_type='last_four',
                 dinov3_repo_path='/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3',
                 pretrained_weights_path='',
                 mode='original',  # 可选择: "original", "lora-only", "moe-only", "lora-moe"
                 num_experts: int = 8,
                 top_k: int = 2,
                 lora_r: int = 4,
                 lora_alpha: int = 8):
        super(DepthAnythingV2_MultiTask, self).__init__()

        # 模型配置
        self.encoder = encoder
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.seg_input_type = seg_input_type
        self.mode = mode

        # 根据mode参数解耦为内部参数
        self.use_lora = mode in ['lora-only', 'lora-moe']
        self.use_moe = mode in ['moe-only', 'lora-moe']

        # 深度任务使用指定的layers (适配各模型的实际层数)
        self.depth_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39],
            'dinov3_vits16': [2, 5, 8, 11],
            'dinov3_vits16plus': [2, 5, 8, 11],
            'dinov3_vitb16': [2, 5, 8, 11],
            'dinov3_vitl16': [4, 11, 17, 23],
            'dinov3_vitl16plus': [4, 11, 17, 23],
            'dinov3_vith16plus': [4, 11, 17, 23],
            'dinov3_vit7b16': [4, 11, 17, 23]
        }
        self.num_layers = {
            'vits': 12,
            'vitb': 12,
            'vitl': 24,
            'vitg': 40,
            'dinov3_vits16': 12,
            'dinov3_vits16plus': 12,
            'dinov3_vitb16': 12,
            'dinov3_vitl16': 24,
            'dinov3_vitl16plus': 24,
            'dinov3_vith16plus': 32,
            'dinov3_vit7b16': 40
        }

        # Patch size 配置
        self.patch_sizes = {
            'vits': 14,
            'vitb': 14,
            'vitl': 14,
            'vitg': 14,
            'dinov3_vits16': 16,
            'dinov3_vits16plus': 16,
            'dinov3_vitb16': 16,
            'dinov3_vitl16': 16,
            'dinov3_vitl16plus': 16,
            'dinov3_vith16plus': 16,
            'dinov3_vit7b16': 16
        }

        # 默认输出通道配置
        if out_channels is None:
            out_channels_config = {
                'vits': [48, 96, 192, 384],
                'vitb': [96, 192, 384, 768],
                'vitl': [256, 512, 1024, 1024],
                'vitg': [1536, 1536, 1536, 1536],
                'dinov3_vits16': [48, 96, 192, 384],
                'dinov3_vits16plus': [48, 96, 192, 384],
                'dinov3_vitb16': [96, 192, 384, 768],
                'dinov3_vitl16': [256, 512, 1024, 1024],
                'dinov3_vitl16plus': [256, 512, 1024, 1024],
                'dinov3_vith16plus': [256, 512, 1024, 1024],
                'dinov3_vit7b16': [256, 512, 1024, 1024]
            }
            out_channels = out_channels_config[encoder]

        # 共享的 backbone
        import logging
        logger = logging.getLogger("dinov2_dpt")
        logger.info(f"[dpt_multitask.py] Mode: {mode}, use_lora: {self.use_lora}, use_moe: {self.use_moe}, num_experts={num_experts}, top_k={top_k}, lora_r={lora_r}, lora_alpha={lora_alpha}")

        if 'dinov3' in encoder:
            if mode == 'original':
                logger.info("Using original DINOv3 backbone for 'original' mode.")
                self.backbone = DINOv3(model_name=encoder)
            else:
                # 使用新的DINOv3_LoRA包装器
                from .dinov3_lora import DINOv3_LoRA
                logger.info(f"Using DINOv3_LoRA backbone with mode: {mode}")
                self.backbone = DINOv3_LoRA(
                    model_name=encoder,
                    mode=mode,
                    num_experts=num_experts,
                    top_k=top_k,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha
                )
        else:
            # 从导入正确的DINOv2模块
            if mode == 'original':
                logger.info("Using original DINOv2 backbone for 'original' mode.")
                self.backbone = DINOv2(model_name=encoder)
            else:
                from .dinov2_lora import DINOv2_LoRA
                self.backbone = DINOv2_LoRA(
                    model_name=encoder,
                    mode=mode,  # 传递mode而不是单独的flag
                    num_experts=num_experts,
                    top_k=top_k,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                )
        # 深度估计头
        patch_size = self.get_patch_size()
        self.depth_head = DPTHead(self.backbone.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, patch_size=patch_size)
        self.intrinsics_head = IntrinsicsHead(self.backbone.embed_dim, hidden_dim=max(256, features * 2), use_clstoken=use_clstoken)

        # 语义分割头 - 使用从 linear_head.py 导入的LinearBNHead
        # 为分割任务定义独立的层索引
        self.seg_layer_idx_map = {
            'vits': [8, 9, 10, 11],
            'vitb': [8, 9, 10, 11],
            'vitl': [20, 21, 22, 23],
            'vitg': [36, 37, 38, 39],
            'dinov3_vits16': [8, 9, 10, 11],
            'dinov3_vits16plus': [8, 9, 10, 11],
            'dinov3_vitb16': [8, 9, 10, 11],
            'dinov3_vitl16': [20, 21, 22, 23],
            'dinov3_vitl16plus': [20, 21, 22, 23],
            'dinov3_vith16plus': [28, 29, 30, 31],
            'dinov3_vit7b16': [36, 37, 38, 39]
        }

        total_layers = self.num_layers[self.encoder]
        if self.seg_input_type == 'last':
            self.seg_layer_idx = [total_layers - 1]
        elif self.seg_input_type == 'last_four':
            # 使用精确的层索引以匹配'思路.md'
            self.seg_layer_idx = self.seg_layer_idx_map.get(self.encoder, list(range(total_layers - 4, total_layers)))
        elif self.seg_input_type == 'from_depth':
            # 'from_depth' 模式保持不变，使用深度头的索引
            self.seg_layer_idx = self.depth_layer_idx[self.encoder]
        else:
            raise ValueError(f"Unknown seg_input_type: {self.seg_input_type}")

        num_seg_layers = len(self.seg_layer_idx)
        seg_in_channels = [self.backbone.embed_dim] * num_seg_layers

        self.seg_head = LinearBNHead(in_channels=seg_in_channels, channels=features, num_classes=num_classes, in_index=list(range(num_seg_layers)))

    def get_patch_size(self):
        """获取当前编码器对应的 patch size"""
        return self.patch_sizes.get(self.encoder, 14)  # 默认为14
        self.seg_head = LinearBNHead(in_channels=seg_in_channels, channels=features, num_classes=num_classes, in_index=list(range(num_seg_layers)))

    def forward_features(self, x, task='both'):
        """提取任务特定特征"""
        h, w = x.shape[-2:]

        # 对于 ViT patch embed，需要输入尺寸是 patch_size 的整数倍；否则进行零填充
        patch_size = self.get_patch_size()
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        results = {}

        # 根据任务提取不同的层特征
        if task in ['depth', 'both']:
            # 深度任务使用指定的layers
            depth_features = self.backbone.get_intermediate_layers(x, n=self.depth_layer_idx[self.encoder], return_class_token=True)
            depth_features = [(patch, cls) for patch, cls in depth_features]
            results['depth_features'] = depth_features

        if task in ['seg', 'both']:
            # 根据配置获取分割任务的特征
            seg_features = self.backbone.get_intermediate_layers(x, n=self.seg_layer_idx, return_class_token=False)
            results['seg_features'] = seg_features

        return results, h, w

    def forward_depth(self, features, h, w):
        """深度估计前向传播"""
        patch_size = self.get_patch_size()
        # 使用向上取整以匹配前面可能的padding后产生的token网格
        patch_h, patch_w = math.ceil(h / patch_size), math.ceil(w / patch_size)
        depth_pred = self.depth_head(features, patch_h, patch_w)
        return depth_pred

    def forward_segmentation(self, features, h, w):
        """
        语义分割前向传播 (for non 'from_depth' modes)
        """
        patch_size = self.get_patch_size()
        # 使用向上取整以匹配前面可能的padding后产生的token网格
        patch_h, patch_w = math.ceil(h / patch_size), math.ceil(w / patch_size)

        # Reshape features from (B, N, C) to (B, C, H, W)
        reshaped_features = []
        for x in features:
            if len(x.shape) == 3:
                b, _, c = x.shape
                x = x.permute(0, 2, 1).reshape(b, c, patch_h, patch_w)
            # 数值稳健性：清理中间特征中的 NaN/Inf
            x = torch.nan_to_num(x, nan=0.0)
            x = x.clamp_(-100.0, 100.0)
            reshaped_features.append(x)

        seg_pred = self.seg_head(reshaped_features)
        return seg_pred

    def forward(self, x, task='both', return_features=False):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]
            task: 任务类型 'depth', 'seg', 'both'
            return_features: 是否返回中间特征用于对齐损失

        Returns:
            dict: 包含预测结果的字典
        """
        # 提取任务特定特征
        feature_results, h, w = self.forward_features(x, task)

        results = {}

        if task in ['depth', 'both']:
            # 深度预测 - 使用depth专用features
            depth_features = feature_results['depth_features']
            patch_size = self.get_patch_size()
            patch_h, patch_w = h // patch_size, w // patch_size
            depth_raw = self.depth_head(depth_features, patch_h, patch_w) * self.max_depth
            depth_pred = depth_raw.squeeze(1)
            # 上采样到原始输入尺寸
            depth_pred = F.interpolate(depth_pred[:, None], (h, w), mode="bilinear", align_corners=True)[:, 0]
            results['depth'] = depth_pred
            fx_fy_pred = self.intrinsics_head(depth_features)
            results['intrinsics'] = fx_fy_pred
            results['fx'] = fx_fy_pred[:, 0]
            results['fy'] = fx_fy_pred[:, 1]


            # 返回深度任务的中间特征（用于特征对齐）
            if return_features:
                results['depth_features'] = F.adaptive_avg_pool2d(depth_raw, (8, 8))

        if task in ['seg', 'both']:
            # 分割预测 - 使用seg专用features
            seg_features = feature_results['seg_features']
            seg_raw_features = self.forward_segmentation(seg_features, h, w)
            seg_pred = F.interpolate(seg_raw_features, size=(h, w), mode='bilinear', align_corners=True)
            results['seg'] = seg_pred

            # 返回分割任务的中间特征（用于特征对齐）
            if return_features:
                results['seg_features'] = F.adaptive_avg_pool2d(seg_raw_features, (8, 8))

        # 返回共享backbone特征（用于特征对齐）
        if return_features:
            # 根据task选择合适的features来提取backbone特征
            if 'depth_features' in feature_results:
                backbone_features = feature_results['depth_features'][-1]  # Get patch feature from (patch, cls) tuple
            elif 'seg_features' in feature_results:
                # seg_features from backbone are already just patch features
                backbone_features = feature_results['seg_features'][-1]
            else:
                return results

            # 重新整理特征维度
            patch_size = self.get_patch_size()
            patch_h, patch_w = h // patch_size, w // patch_size
            b, _, c = backbone_features.shape
            backbone_features = backbone_features.permute(0, 2, 1).reshape(b, c, patch_h, patch_w).contiguous()
            results['features'] = F.adaptive_avg_pool2d(backbone_features, (8, 8))

        return results

    def get_depth_prediction(self, x):
        """仅获取深度预测"""
        return self.forward(x, task='depth')['depth']

    def get_segmentation_prediction(self, x):
        """仅获取分割预测"""
        return self.forward(x, task='seg')['seg']

    def get_both_predictions(self, x):
        """同时获取深度和分割预测"""
        return self.forward(x, task='both')


class DepthAnythingV2_MultiTask_Frozen(DepthAnythingV2_MultiTask):
    """
    冻结backbone的多任务模型版本
    只训练任务特定的头部
    """

    def __init__(self,
                 encoder='vits',
                 num_classes=3,
                 features=256,
                 out_channels=None,
                 use_bn=False,
                 use_clstoken=False,
                 max_depth=20.0,
                 seg_input_type='last_four',
                 dinov3_repo_path='',
                 pretrained_weights_path='',
                 mode='original',
                 num_experts=8,
                 top_k=2,
                 lora_r=4,
                 lora_alpha=8):
        # 直接传递所有参数给父类
        super(DepthAnythingV2_MultiTask_Frozen, self).__init__(
            encoder=encoder,
            num_classes=num_classes,
            features=features,
            out_channels=out_channels,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            max_depth=max_depth,
            seg_input_type=seg_input_type,
            dinov3_repo_path=dinov3_repo_path,
            pretrained_weights_path=pretrained_weights_path,
            mode=mode,
            num_experts=num_experts,
            top_k=top_k,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        # 冻结backbone参数
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_trainable_parameters(self):
        """获取可训练参数（仅头部）"""
        trainable_params = []

        # 深度头参数
        for param in self.depth_head.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # 内参头参数
        for param in self.intrinsics_head.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # 分割头参数
        for param in self.seg_head.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params


def create_multitask_model(encoder='vits',
                           num_classes=3,
                           features=None,
                           max_depth=20.0,
                           frozen_backbone=False,
                           seg_input_type='last_four',
                           dinov3_repo_path='',
                           pretrained_weights_path='',
                           mode='original',
                           num_experts: int = 8,
                           top_k: int = 2,
                           lora_r: int = 4,
                           lora_alpha: int = 8):
    """
    创建多任务模型的便捷函数
    
    Args:
        encoder: 编码器类型
        num_classes: 分割类别数
        features: 特征维度 (将被忽略，使用config中的值)
        max_depth: 最大深度值
        frozen_backbone: 是否冻结backbone
        seg_input_type: 分割头输入类型 ('last', 'last_four', 'from_depth')
        
    Returns:
        多任务模型实例
    """
    # 根据编码器类型设置特征配置
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

    dinov3_configs = {
        "dinov3_vits16": {
            "features": 64,
            "out_channels": [48, 96, 192, 384]
        },
        "dinov3_vits16plus": {
            "features": 64,
            "out_channels": [48, 96, 192, 384]
        },
        "dinov3_vitb16": {
            "features": 128,
            "out_channels": [96, 192, 384, 768]
        },
        "dinov3_vitl16": {
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        },
        "dinov3_vitl16plus": {
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        },
        "dinov3_vith16plus": {
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        },
        "dinov3_vit7b16": {
            "features": 256,
            "out_channels": [256, 512, 1024, 1024]
        },
    }

    model_configs.update(dinov3_configs)

    config = model_configs.get(encoder, model_configs["vits"])

    model_args = {
        'encoder': encoder,
        'num_classes': num_classes,
        'features': config["features"],
        'out_channels': config["out_channels"],
        'max_depth': max_depth,
        'seg_input_type': seg_input_type,
        'dinov3_repo_path': dinov3_repo_path,
        'pretrained_weights_path': pretrained_weights_path,
        'mode': mode,
        'num_experts': num_experts,
        'top_k': top_k,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
    }

    if frozen_backbone:
        model = DepthAnythingV2_MultiTask_Frozen(**model_args)
    else:
        model = DepthAnythingV2_MultiTask(**model_args)

    return model
