from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .dinov2 import DINOv2
from .dinov3 import DINOv3
from .dpt import DPTHead
from .camera_head import ProLikeCameraHead, SimpleCameraHead, VGGTLiteCameraHead

# 使用绝对导入避免相对导入问题
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from seg_heads.linear_head.linear_head import LinearBNHead
from seg_heads.segformer_head.segformer_head import SegFormerHead


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
                 seg_head_type='linear',
                 dinov3_repo_path='/media/ExtHDD1/jianfu/depth/DepthAnythingV2/dinov3',
                 pretrained_weights_path='',
                 mode='original',  # 可选择: "original", "lora-only", "legacy-lora"
                 num_experts: int = 8,
                 top_k: int = 2,
                 lora_r: int = 4,
                 lora_alpha: int = 8,
                 camera_head_mode: str = "none",
                 adapter_scope_params: Optional[dict] = None,
                 use_semantic_tokens: bool = False,
                 semantic_token_count: int = 0):
        super(DepthAnythingV2_MultiTask, self).__init__()

        # 模型配置
        self.encoder = encoder
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.seg_input_type = seg_input_type
        self.seg_head_type = (seg_head_type or "linear").lower()
        self.mode = mode
        self.camera_head_mode = camera_head_mode.lower()
        self.adapter_scope_params = adapter_scope_params or {}
        scoped_modes = {"mtlora", "mtlga", "mtoat", "endounid"}
        self.adapter_scopes_enabled = self.mode in scoped_modes
        if self.mode in {"mtlora", "mtlga"} and "dinov3" in self.encoder:
            raise ValueError(f"Mode '{self.mode}' currently supports DINOv2 backbones only.")
        if self.adapter_scopes_enabled and self.encoder not in {"vits", "vitb"}:
            raise ValueError("Adapter-scoped modes currently only support vits or vitb encoders")

        # 根据mode参数解耦为内部参数
        self.use_lora = mode in ['lora-only', 'legacy-lora', 'mtlora', 'mtlga', 'mtoat', 'endounid']
        self.use_moe = False
        self.attention_only_lora = mode == 'legacy-lora'
        self.freeze_camera_backbone_grad = self.mode in {'mtlora', 'mtlga', 'mtoat', 'endounid'}
        self.use_semantic_tokens = bool(use_semantic_tokens and semantic_token_count > 0)
        self.semantic_token_count = semantic_token_count if self.use_semantic_tokens else 0

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

        self.seg_layer_idx = []
        if self.seg_head_type != "none":
            # 语义分割头层索引预计算
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
                self.seg_layer_idx = self.seg_layer_idx_map.get(self.encoder, list(range(total_layers - 4, total_layers)))
            elif self.seg_input_type == 'from_depth':
                self.seg_layer_idx = self.depth_layer_idx[self.encoder]
            else:
                raise ValueError(f"Unknown seg_input_type: {self.seg_input_type}")

        allowed_seg_heads = {"linear", "sf", "none"}
        if self.seg_head_type not in allowed_seg_heads:
            raise ValueError(f"Unsupported seg_head_type: {self.seg_head_type}")
        if self.seg_head_type != "none":
            # 新的语义头共享与深度头相同的层索引，便于特征对齐
            self.seg_layer_idx = self.depth_layer_idx[self.encoder]

        if self.use_semantic_tokens:
            desired_token_count = max(len(self.seg_layer_idx) - 1, 1)
            current = self.semantic_token_count if self.semantic_token_count and self.semantic_token_count > 0 else desired_token_count
            self.semantic_token_count = max(current, desired_token_count)

        if self.adapter_scopes_enabled:
            self.adapter_scope_cfg = self._build_adapter_scope_cfg()
        else:
            self.adapter_scope_cfg = None

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
        logger.info(
            f"[dpt_multitask.py] Mode: {mode}, use_lora: {self.use_lora},"
            f" attention_only_lora: {self.attention_only_lora}, lora_r={lora_r}, lora_alpha={lora_alpha}"
        )

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
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    adapter_scope_cfg=self.adapter_scope_cfg,
                )

        self.scenario_token: Optional[torch.nn.Parameter] = None
        self.semantic_token_bank: Optional[torch.nn.Parameter] = None
        if self.use_semantic_tokens:
            if 'dinov3' in self.encoder:
                raise ValueError("Semantic tokens currently only support DINOv2 backbones.")
            embed_dim = self.backbone.embed_dim
            self.scenario_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.scenario_token, std=1e-2)
            self.semantic_token_bank = nn.Parameter(torch.zeros(self.semantic_token_count, embed_dim))
            nn.init.normal_(self.semantic_token_bank, std=1e-2)
        # 深度估计头
        patch_size = self.get_patch_size()
        self.depth_head = DPTHead(self.backbone.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, patch_size=patch_size)

        self.seg_head = None
        if self.seg_head_type != "none":
            num_seg_layers = len(self.seg_layer_idx)
            seg_in_channels = [self.backbone.embed_dim] * num_seg_layers

            if self.seg_head_type == "linear":
                self.seg_head = LinearBNHead(in_channels=seg_in_channels, channels=features, num_classes=num_classes, in_index=list(range(num_seg_layers)))
            else:
                self.seg_head = SegFormerHead(in_channels=seg_in_channels, embedding_dim=features, num_classes=num_classes, align_corners=False)

        scoped_camera_enabled = self.adapter_scopes_enabled
        camera_adapter_rank = self.adapter_scope_params.get('camera_r', 0) if scoped_camera_enabled else 0
        camera_adapter_alpha = self.adapter_scope_params.get('camera_alpha', 1) if scoped_camera_enabled else 1
        camera_adapter_dropout = self.adapter_scope_params.get('dropout', 0.0) if scoped_camera_enabled else 0.0

        # Optional camera head
        if self.camera_head_mode == "simple":
            self.camera_head = SimpleCameraHead(
                self.backbone.embed_dim,
                adapter_rank=camera_adapter_rank,
                adapter_alpha=camera_adapter_alpha,
                adapter_dropout=camera_adapter_dropout,
            )
        elif self.camera_head_mode == "prolike":
            self.camera_head = ProLikeCameraHead(
                self.backbone.embed_dim,
                adapter_rank=camera_adapter_rank,
                adapter_alpha=camera_adapter_alpha,
                adapter_dropout=camera_adapter_dropout,
            )
        elif self.camera_head_mode in {"vggtlike", "vggt-like"}:
            self.camera_head = VGGTLiteCameraHead(
                self.backbone.embed_dim,
                adapter_rank=camera_adapter_rank,
                adapter_alpha=camera_adapter_alpha,
                adapter_dropout=camera_adapter_dropout,
            )
            self.camera_head_mode = "vggtlike"
        elif self.camera_head_mode == "none":
            self.camera_head = None
        else:
            raise ValueError(f"Unknown camera_head_mode: {camera_head_mode}")

    def get_patch_size(self):
        """获取当前编码器对应的 patch size"""
        return self.patch_sizes.get(self.encoder, 14)  # 默认为14

    def _build_extra_tokens(self,
                            batch_size: int,
                            semantic_token_ids: Optional[torch.Tensor],
                            device: torch.device) -> Optional[torch.Tensor]:
        self._current_extra_token_count = 0
        if (not self.use_semantic_tokens or
                self.semantic_token_bank is None or
                self.scenario_token is None or
                self.semantic_token_count <= 0):
            return None

        token_device = device or self.scenario_token.device
        scenario = self.scenario_token.to(token_device).expand(batch_size, -1, -1)
        semantic = self._compute_semantic_token_embedding(batch_size, semantic_token_ids, token_device)
        if semantic is None:
            extra_tokens = scenario
        else:
            extra_tokens = torch.cat([scenario, semantic], dim=1)
        self._current_extra_token_count = extra_tokens.shape[1]
        return extra_tokens

    def forward_features(self, x, task='both', return_features=False, semantic_token_ids: Optional[torch.Tensor] = None):
        """提取任务特定特征"""
        orig_h, orig_w = x.shape[-2:]
        h, w = orig_h, orig_w

        # 对于 ViT patch embed，需要输入尺寸是 patch_size 的整数倍；否则进行零填充
        patch_size = self.get_patch_size()
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        effective_h = h + pad_h
        effective_w = w + pad_w
        self._current_hw = (orig_h, orig_w)
        self._current_effective_hw = (effective_h, effective_w)

        results = {}

        extra_tokens = self._build_extra_tokens(x.shape[0], semantic_token_ids, x.device) if self.use_semantic_tokens else None
        if extra_tokens is None:
            self._current_extra_token_count = 0

        # 根据任务提取不同的层特征
        if task in ['depth', 'both']:
            # 深度任务使用指定的layers
            self._set_adapter_scopes(['shared', 'depth'])
            raw_depth_features = self.backbone.get_intermediate_layers(
                x,
                n=self.depth_layer_idx[self.encoder],
                return_class_token=True,
                extra_tokens=extra_tokens,
            )
            depth_features = []
            for patch, cls in raw_depth_features:
                depth_features.append((self._strip_extra_tokens(patch), cls))
            results['depth_features'] = depth_features
            if return_features and depth_features:
                last_patch = depth_features[-1][0]
                results['ga_depth_tokens'] = last_patch

        if task in ['seg', 'both'] and self.seg_head is not None:
            # 根据配置获取分割任务的特征
            self._set_adapter_scopes(['shared', 'seg'])
            raw_seg_features = self.backbone.get_intermediate_layers(
                x,
                n=self.seg_layer_idx,
                return_class_token=False,
                extra_tokens=extra_tokens,
            )
            seg_features = [self._strip_extra_tokens(feat) for feat in raw_seg_features]
            results['seg_features'] = seg_features
            if return_features and seg_features:
                results['ga_seg_tokens'] = seg_features[-1]

        self._set_adapter_scopes(['shared'])
        return results, orig_h, orig_w

    def forward_depth(self, features, h, w):
        """深度估计前向传播"""
        patch_size = self.get_patch_size()
        eff_h, eff_w = getattr(self, "_current_effective_hw", (h, w))
        # 使用向上取整以匹配前面可能的padding后产生的token网格
        patch_h = math.ceil(eff_h / patch_size)
        patch_w = math.ceil(eff_w / patch_size)
        depth_pred = self.depth_head(features, patch_h, patch_w)
        return depth_pred

    def _reshape_seg_inputs(self, features, patch_h: int, patch_w: int) -> List[torch.Tensor]:
        reshaped = []
        for feat in features:
            if isinstance(feat, tuple):
                feat = feat[0]
            if feat.dim() != 3:
                reshaped.append(torch.nan_to_num(feat, nan=0.0).clamp_(-100.0, 100.0))
                continue
            b, _, c = feat.shape
            tensor = feat.permute(0, 2, 1).reshape(b, c, patch_h, patch_w)
            tensor = torch.nan_to_num(tensor, nan=0.0)
            tensor = tensor.clamp_(-100.0, 100.0)
            reshaped.append(tensor)
        return reshaped

    def forward_segmentation(self, features, h, w):
        """
        语义分割前向传播
        """
        if self.seg_head is None:
            raise RuntimeError("Segmentation head is disabled but forward_segmentation was called.")
        patch_size = self.get_patch_size()
        eff_h, eff_w = getattr(self, "_current_effective_hw", (h, w))
        patch_h = math.ceil(eff_h / patch_size)
        patch_w = math.ceil(eff_w / patch_size)
        reshaped_features = self._reshape_seg_inputs(features, patch_h, patch_w)
        seg_pred = self.seg_head(reshaped_features)
        return seg_pred

    def forward(
        self,
        x,
        task: str = 'both',
        return_features: bool = False,
        semantic_token_ids: Optional[torch.Tensor] = None,
        skip_depth_head: bool = False,
        skip_seg_head: bool = False,
    ):
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
        feature_results, h, w = self.forward_features(
            x,
            task,
            return_features=return_features,
            semantic_token_ids=semantic_token_ids,
        )

        results = {}

        run_depth = task in ['depth', 'both']
        run_seg = task in ['seg', 'both'] and self.seg_head is not None

        if run_depth and not skip_depth_head:
            # 深度预测 - 使用depth专用features
            depth_features = feature_results['depth_features']
            patch_size = self.get_patch_size()
            patch_h, patch_w = math.ceil(h / patch_size), math.ceil(w / patch_size)
            depth_raw = self.depth_head(depth_features, patch_h, patch_w) * self.max_depth
            depth_pred = depth_raw.squeeze(1)
            # 上采样到原始输入尺寸
            depth_pred = F.interpolate(depth_pred[:, None], (h, w), mode="bilinear", align_corners=True)[:, 0]
            results['depth'] = depth_pred

            # 返回深度任务的中间特征（用于特征对齐）
            if return_features:
                results['depth_features'] = F.adaptive_avg_pool2d(depth_raw, (8, 8))

            if self.camera_head is not None:
                # Use the last depth layer tokens as camera input
                last_depth_tokens, _ = depth_features[-1]
                camera_tokens = last_depth_tokens.detach() if self.freeze_camera_backbone_grad else last_depth_tokens
                camera_norm = self.camera_head(camera_tokens, patch_h, patch_w)
                results['camera_intrinsics_norm'] = camera_norm
                fx = camera_norm[:, 0] * w
                fy = camera_norm[:, 1] * h
                cx = camera_norm[:, 2] * w
                cy = camera_norm[:, 3] * h
                intrinsics = camera_norm.new_zeros((camera_norm.size(0), 3, 3))
                intrinsics[:, 0, 0] = fx
                intrinsics[:, 1, 1] = fy
                intrinsics[:, 0, 2] = cx
                intrinsics[:, 1, 2] = cy
                intrinsics[:, 2, 2] = 1.0
                results['camera_intrinsics'] = intrinsics

        if run_seg and not skip_seg_head:
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
                backbone_features = feature_results['depth_features'][-1]
            elif 'seg_features' in feature_results:
                # seg_features from backbone are already just patch features
                backbone_features = feature_results['seg_features'][-1]
            else:
                return results

            # 重新整理特征维度
            patch_size = self.get_patch_size()
            patch_h = math.ceil(h / patch_size)
            patch_w = math.ceil(w / patch_size)

            if isinstance(backbone_features, tuple):
                backbone_features = backbone_features[0]

            b, _, c = backbone_features.shape
            backbone_features = backbone_features.permute(0, 2, 1).reshape(b, c, patch_h, patch_w).contiguous()
            results['features'] = F.adaptive_avg_pool2d(backbone_features, (8, 8))

        return results

    def _build_adapter_scope_cfg(self) -> Dict[str, dict]:
        depth_layers = set(self.depth_layer_idx[self.encoder])
        seg_layers = set(self.seg_layer_idx)
        total_layers = self.num_layers[self.encoder]
        block_scopes: Dict[int, List[str]] = {}
        for idx in range(total_layers):
            scopes: List[str] = []
            if idx in depth_layers:
                scopes.append('depth')
            if idx in seg_layers:
                scopes.append('seg')
            if not scopes:
                scopes.append('shared')
            block_scopes[idx] = scopes
        ranks = {
            'shared': self.adapter_scope_params.get('shared_r', 4),
            'depth': self.adapter_scope_params.get('depth_r', 8),
            'seg': self.adapter_scope_params.get('seg_r', 8),
        }
        alphas = {
            'shared': self.adapter_scope_params.get('shared_alpha', 8),
            'depth': self.adapter_scope_params.get('depth_alpha', 16),
            'seg': self.adapter_scope_params.get('seg_alpha', 16),
        }
        cfg = {
            'block_scopes': block_scopes,
            'ranks': ranks,
            'alphas': alphas,
            'shared_shards': self.adapter_scope_params.get('shared_shards', 1),
            'dropout': self.adapter_scope_params.get('dropout', 0.0),
            'default_scopes': ['shared'],
        }
        return cfg

    def _set_adapter_scopes(self, scopes: List[str]) -> None:
        if hasattr(self.backbone, "set_active_adapter_scopes"):
            self.backbone.set_active_adapter_scopes(scopes)

    def _strip_extra_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens is None or tokens.dim() != 3:
            return tokens
        extra = getattr(self, "_current_extra_token_count", 0)
        if extra <= 0 or tokens.shape[1] <= extra:
            return tokens
        return tokens[:, extra:, :]

    def _compute_semantic_token_embedding(self,
                                          batch_size: int,
                                          semantic_token_ids: Optional[torch.Tensor],
                                          device: torch.device) -> Optional[torch.Tensor]:
        if self.semantic_token_bank is None or self.semantic_token_count <= 0:
            return None
        ids: torch.Tensor
        mask: torch.Tensor
        if semantic_token_ids is None:
            ids = torch.arange(self.semantic_token_count, device=device, dtype=torch.long)
            ids = ids.unsqueeze(0).expand(batch_size, -1)
            mask = torch.ones_like(ids, dtype=torch.bool)
        else:
            ids = semantic_token_ids.to(device, dtype=torch.long)
            if ids.dim() == 1:
                ids = ids.unsqueeze(1)
            if ids.size(0) != batch_size:
                if ids.size(0) == 1:
                    ids = ids.expand(batch_size, -1)
                else:
                    ids = ids[:batch_size]
                    if ids.size(0) < batch_size:
                        pad = torch.full((batch_size - ids.size(0), ids.size(1)), 0, dtype=torch.long, device=device)
                        ids = torch.cat([ids, pad], dim=0)
            mask = ids > 0
            ids = ids.clamp(min=1, max=self.semantic_token_count) - 1

        embeddings = F.embedding(ids, self.semantic_token_bank.to(device))
        mask_f = mask.unsqueeze(-1).float()
        masked_embeddings = embeddings * mask_f
        counts = mask_f.sum(dim=1, keepdim=True)
        counts = counts.clamp_min(1.0)
        mean_embeddings = masked_embeddings.sum(dim=1, keepdim=True) / counts

        zero_mask = (mask.sum(dim=1, keepdim=True) == 0)
        if zero_mask.any():
            if self.scenario_token is not None:
                scenario_embed = self.scenario_token.to(device)
                scenario_embed = scenario_embed.expand(batch_size, -1, -1)
                mean_embeddings[zero_mask.squeeze(-1), 0, :] = scenario_embed[zero_mask.squeeze(-1), 0, :]
            else:
                fallback = self.semantic_token_bank.mean(dim=0, keepdim=True)
                mean_embeddings[zero_mask.squeeze(-1), 0, :] = fallback

        return mean_embeddings

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
                 seg_head_type='linear',
                 dinov3_repo_path='',
                 pretrained_weights_path='',
                 mode='original',
                 num_experts=8,
                 top_k=2,
                 lora_r=4,
                 lora_alpha=8,
                 camera_head_mode: str = "none",
                 adapter_scope_params: Optional[dict] = None,
                 use_semantic_tokens: bool = False,
                 semantic_token_count: int = 0):
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
            seg_head_type=seg_head_type,
            dinov3_repo_path=dinov3_repo_path,
            pretrained_weights_path=pretrained_weights_path,
            mode=mode,
            num_experts=num_experts,
            top_k=top_k,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            camera_head_mode=camera_head_mode,
            adapter_scope_params=adapter_scope_params,
            use_semantic_tokens=use_semantic_tokens,
            semantic_token_count=semantic_token_count,
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

        # 分割头参数
        for param in self.seg_head.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        if getattr(self, "camera_head", None) is not None:
            for param in self.camera_head.parameters():
                if param.requires_grad:
                    trainable_params.append(param)

        return trainable_params


def create_multitask_model(encoder='vits',
                           num_classes=3,
                           features=None,
                           max_depth=20.0,
                           frozen_backbone=False,
                           seg_input_type='last_four',
                           seg_head_type='linear',
                           dinov3_repo_path='',
                           pretrained_weights_path='',
                           mode='original',
                           num_experts: int = 8,
                           top_k: int = 2,
                           lora_r: int = 4,
                           lora_alpha: int = 8,
                           camera_head_mode: str = "none",
                           adapter_scope_params: Optional[dict] = None,
                           use_semantic_tokens: bool = False,
                           semantic_token_count: int = 0):
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
        'seg_head_type': seg_head_type,
        'num_experts': num_experts,
        'top_k': top_k,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'camera_head_mode': camera_head_mode,
        'adapter_scope_params': adapter_scope_params,
        'use_semantic_tokens': use_semantic_tokens,
        'semantic_token_count': semantic_token_count,
    }

    if frozen_backbone:
        model = DepthAnythingV2_MultiTask_Frozen(**model_args)
    else:
        model = DepthAnythingV2_MultiTask(**model_args)

    return model
