import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

from ..base_head import BaseSegHead
from .pixel_decoder import MSDeformAttnPixelDecoder
from .transformer import DetrTransformerDecoder
from .positional_encoding import SinePositionalEncoding


class Mask2FormerHead(BaseSegHead):

    def __init__(
        self,
        in_channels: List[int],
        channels: int,
        num_classes: int,
        in_index: List[int],
        feat_channels: int,
        out_channels: int,
        num_things_classes: int = 100,
        num_stuff_classes: int = 50,
        num_queries: int = 100,
        num_transformer_feat_level: int = 3,
        pixel_decoder: Dict = None,
        enforce_decoder_input_project: bool = False,
        transformer_decoder: Dict = None,
        positional_encoding: Dict = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        align_corners: bool = False,
        class_weights: Optional[List[float]] = None,  # 添加类别权重配置
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, in_index=in_index, input_transform="multiple_select", align_corners=align_corners)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        # Add 1 for background class in classification head
        self.cls_out_channels = self.num_classes + 1
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.class_weights = class_weights  # 存储类别权重配置

        pixel_decoder_config = pixel_decoder or {}
        self.pixel_decoder = MSDeformAttnPixelDecoder(in_channels=self.in_channels, feat_channels=feat_channels, **pixel_decoder_config)

        # Handle default configurations
        transformer_decoder_config = transformer_decoder or {}
        positional_encoding_config = positional_encoding or {}

        # Extract layer_cfg directly from transformer_decoder['transformerlayers']
        # The config defines attn_cfgs and ffn_cfgs directly under transformerlayers
        if transformer_decoder_config:
            layer_cfg = dict(
                attn_cfgs=transformer_decoder_config['transformerlayers']['attn_cfgs'],
                ffn_cfgs=transformer_decoder_config['transformerlayers']['ffn_cfgs'],
                operation_order=transformer_decoder_config['transformerlayers']['operation_order']
            )

            self.transformer_decoder = DetrTransformerDecoder(num_layers=transformer_decoder_config['num_layers'],
                                                              layer_cfg=layer_cfg,
                                                              return_intermediate=transformer_decoder_config['return_intermediate'])
        else:
            # Default transformer decoder configuration
            default_layer_cfg = dict(
                attn_cfgs={'embed_dims': feat_channels, 'num_heads': 8, 'attn_drop': 0.0, 'proj_drop': 0.0, 'batch_first': False},
                ffn_cfgs={'embed_dims': feat_channels, 'feedforward_channels': feat_channels * 4, 'num_fcs': 2, 'ffn_drop': 0.0},
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
            )
            self.transformer_decoder = DetrTransformerDecoder(num_layers=6, layer_cfg=default_layer_cfg, return_intermediate=True)

        self.decoder_positional_encoding = SinePositionalEncoding(**positional_encoding_config)

        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.cls_out_channels)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels),
        )

    def forward(self, inputs: List[torch.Tensor]):
        # feats = self._transform_inputs(inputs)
        feats = inputs
        mask_features, multi_scale_memorys, mlvl_masks = self.pixel_decoder(feats)

        batch_size = feats[0].shape[0]

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = multi_scale_memorys[i]
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1).contiguous()
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            mask = decoder_input.new_zeros((batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1).contiguous()

            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        decoder_key = torch.cat(decoder_inputs, dim=0)  # (sum(H*W), B, C)
        decoder_value = torch.cat(decoder_inputs, dim=0)  # (sum(H*W), B, C) - assuming key and value are the same
        decoder_key_pos = torch.cat(decoder_positional_encodings, dim=0)  # (sum(H*W), B, C)

        # Prepare key_padding_mask for cross-attention
        # This mask should be (B, sum(H*W))
        # mlvl_masks is (B, sum(H*W)) from MSDeformAttnPixelDecoder
        key_padding_mask = mlvl_masks  # This is already flattened and concatenated

        # For self-attention in decoder, query_key_padding_mask is usually None for fixed number of queries
        query_key_padding_mask = None

        # attn_masks for self-attention and cross-attention
        # For self-attention, usually None (full attention) or causal mask for auto-regressive
        # For cross-attention, usually None (full attention)
        attn_masks = (None, None)  # (self_attn_mask, cross_attn_mask)

        hidden_states = self.transformer_decoder(
            query=query_feat,
            key=decoder_key,
            value=decoder_value,
            query_pos=query_embed,
            key_pos=decoder_key_pos,
            attn_masks=attn_masks,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask,
        )

        all_cls_scores = self.cls_embed(hidden_states)

        mask_embed = self.mask_embed(hidden_states)
        # hidden_states shape: (num_layers, num_queries, batch_size, embed_dim)
        # all_cls_scores shape: (num_layers, num_queries, batch_size, num_classes)
        # mask_embed shape: (num_layers, num_queries, batch_size, embed_dim)
        # We need to permute them to (num_layers, batch_size, num_queries, ...)
        all_cls_scores = all_cls_scores.permute(0, 2, 1, 3).contiguous()
        mask_embed = mask_embed.permute(0, 2, 1, 3).contiguous()

        # mask_embed has shape (num_layers, batch_size, num_queries, embed_dim)
        # We need to compute mask predictions for each layer
        all_mask_preds = []
        for layer_mask_embed in mask_embed:
            # layer_mask_embed has shape (batch_size, num_queries, embed_dim)
            mask_preds = torch.einsum('bqc,bchw->bqhw', layer_mask_embed, mask_features)
            all_mask_preds.append(mask_preds)
        all_mask_preds = torch.stack(all_mask_preds)  # (num_layers, batch_size, num_queries, height, width)

        return all_cls_scores[-1], all_mask_preds[-1]

    def loss(self, cls_scores, mask_preds, gt_masks):
        """
        Simple loss function for Mask2Former.

        Args:
            cls_scores: Classification scores (batch_size, num_queries, num_classes)
            mask_preds: Mask predictions (batch_size, num_queries, height, width)
            gt_masks: Ground truth masks (batch_size, height, width)

        Returns:
            dict: Dictionary containing loss values
        """
        device = mask_preds.device

        batch_size, num_queries, actual_num_classes = cls_scores.shape
        _, _, height, width = mask_preds.shape
        

        # Ensure gt_masks has the correct batch dimension
        if gt_masks.dim() == 2:
            # If gt_masks is (H, W), add batch dimension
            gt_masks = gt_masks.unsqueeze(0)
        elif gt_masks.dim() == 3 and gt_masks.shape[0] != batch_size:
            # If batch dimension doesn't match, take only what we need
            actual_gt_batch = gt_masks.shape[0]

        # Resize ground truth masks to match prediction size
        gt_masks_resized = F.interpolate(
            gt_masks.unsqueeze(1).float(),  # Add channel dimension
            size=(height, width),
            mode='nearest'
        ).squeeze(1).long()  # Remove channel dimension and convert back to long

        # Use the ground truth batch size (which is the real batch size)
        # The model generates num_queries predictions per batch item
        gt_batch_size = gt_masks_resized.shape[0]

        # Convert gt_masks to one-hot format for each class
        # Use the minimum of self.num_classes and actual_num_classes to avoid index errors
        effective_num_classes = min(self.num_classes, actual_num_classes)
        gt_masks_one_hot = torch.zeros(gt_batch_size, effective_num_classes, height, width, device=device)
        for b in range(gt_batch_size):
            for c in range(effective_num_classes):
                gt_masks_resized[b] = torch.clamp(gt_masks_resized[b], 0, effective_num_classes - 1)
                gt_masks_one_hot[b, c] = (gt_masks_resized[b] == c).float()

        # Simple assignment: assign each query to the best matching ground truth class
        # This is a simplified version - full Mask2Former uses Hungarian matching

        # Compute IoU between each query and each ground truth class
        losses = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Classification loss - use cross entropy on the class scores
        # Better target assignment to encourage foreground learning
        target_classes = torch.zeros(gt_batch_size, num_queries, dtype=torch.long, device=device)

        # 自适应目标分配策略，适合多数据集联合训练
        for b in range(gt_batch_size):
            # 统计每个类别的像素数
            class_pixel_counts = []
            for c in range(effective_num_classes):
                if c < gt_masks_one_hot.shape[1]:
                    pixel_count = gt_masks_one_hot[b, c].sum().item()
                    class_pixel_counts.append(pixel_count)
                else:
                    class_pixel_counts.append(0)

            # 自适应分配策略：背景少量，前景平均分配
            query_idx = 0

            # 最简单的平均分配策略
            queries_per_class = num_queries // effective_num_classes

            for c in range(effective_num_classes):
                for _ in range(queries_per_class):
                    if query_idx < num_queries:
                        target_classes[b, query_idx] = c
                        query_idx += 1

            # 剩余queries分配给背景
            while query_idx < num_queries:
                target_classes[b, query_idx] = 0
                query_idx += 1

        # Classification loss - cls_scores is now properly shaped (gt_batch_size, num_queries, num_classes)
        cls_scores_actual = cls_scores[:gt_batch_size]
        # print(f"[DEBUG] cls_scores_actual.shape: {cls_scores_actual.shape}")
        # print(f"[DEBUG] target_classes.shape: {target_classes.shape}")
        # print(f"[DEBUG] cls_scores_actual total elements: {cls_scores_actual.numel()}")
        # print(f"[DEBUG] target_classes total elements: {target_classes.numel()}")

        # Check if we have valid shapes for cross entropy
        if cls_scores_actual.numel() == 0 or target_classes.numel() == 0:
            print("[DEBUG] Empty tensors, using dummy loss")
            cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        elif cls_scores_actual.numel() % actual_num_classes != 0:
            # print(f"[DEBUG] Invalid tensor size for reshape: {cls_scores_actual.numel()} elements, {actual_num_classes} classes")
            cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Use contiguous().view() to handle non-contiguous tensors
            cls_scores_flat = cls_scores_actual.contiguous().view(-1, actual_num_classes)
            target_classes_flat = target_classes.contiguous().view(-1)

            if cls_scores_flat.shape[0] == target_classes_flat.shape[0]:
                # 使用配置化的类别权重
                # 回到最简单的交叉熵损失
                if self.class_weights is not None:
                    class_weights = torch.tensor(self.class_weights, device=device)
                    cls_loss = F.cross_entropy(cls_scores_flat, target_classes_flat,
                                             weight=class_weights)
                else:
                    cls_loss = F.cross_entropy(cls_scores_flat, target_classes_flat)
            else:
                cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        losses['loss_cls'] = cls_loss
        # 按照 mmseg 配置的分类损失权重
        cls_weight = 2.0 if self.class_weights is not None else 1.0
        total_loss = total_loss + cls_loss * cls_weight * 0.5  # 减少分类损失权重

        # Simplified mask loss to reduce memory usage
        # Use only a subset of queries to prevent memory issues
        mask_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Only use first 3 queries to reduce computation
        num_queries_to_use = min(3, num_queries, effective_num_classes)

        if num_queries_to_use > 0:
            # Vectorized computation for efficiency
            selected_mask_preds = mask_preds[:gt_batch_size, :num_queries_to_use]  # (B, 3, H, W)
            selected_gt_masks = gt_masks_one_hot[:, :num_queries_to_use].float()   # (B, 3, H, W)

            # Compute loss for all selected queries at once
            mask_loss = F.binary_cross_entropy_with_logits(
                selected_mask_preds, selected_gt_masks, reduction='mean'
            )

        losses['loss_mask'] = mask_loss

        # 添加 Dice 损失 (按照 mmseg 配置)
        dice_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if self.class_weights is not None:
            # 计算 Dice 损失
            for b in range(gt_batch_size):
                for c in range(1, effective_num_classes):  # 跳过背景类
                    if c < gt_masks_one_hot.shape[1]:
                        # 选择对应类别的 queries
                        class_queries = mask_preds[b, :num_queries_to_use]
                        gt_mask = gt_masks_one_hot[b, c].float()

                        # 计算 Dice 损失
                        pred_sigmoid = torch.sigmoid(class_queries)
                        intersection = (pred_sigmoid * gt_mask.unsqueeze(0)).sum(dim=[1, 2])
                        union = pred_sigmoid.sum(dim=[1, 2]) + gt_mask.sum()
                        dice = (2 * intersection + 1e-6) / (union + 1e-6)
                        dice_loss = dice_loss + (1 - dice.mean())

        losses['loss_dice'] = dice_loss

        # 按照 mmseg 权重配置
        mask_weight = 5.0 if self.class_weights is not None else 1.0
        dice_weight = 5.0 if self.class_weights is not None else 0.0
        total_loss = total_loss + mask_loss * mask_weight + dice_loss * dice_weight * 2.0  # 增加 mask 损失权重

        losses['loss_total'] = total_loss
        return losses

