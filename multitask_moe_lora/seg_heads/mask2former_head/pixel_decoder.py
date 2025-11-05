import torch
import torch.nn as nn
from typing import List

from .transformer import TransformerEncoderLayer, DetrTransformerEncoder
from .positional_encoding import SinePositionalEncoding
from .utils import get_reference_points


class MSDeformAttnPixelDecoder(nn.Module):

    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            strides=[4, 8, 16, 32],
            feat_channels=256,
            out_channels=256,
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention",
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.0,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_levels = encoder['transformerlayers']['attn_cfgs']['num_levels']

        self.input_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.input_convs.append(nn.Conv2d(in_channels[i], feat_channels, kernel_size=1))

        encoder_layer = TransformerEncoderLayer(
            d_model=feat_channels,
            nhead=encoder['transformerlayers']['attn_cfgs']['num_heads'],
            dim_feedforward=encoder['transformerlayers']['ffn_cfgs']['feedforward_channels'],
            dropout=encoder['transformerlayers']['ffn_cfgs']['ffn_drop'],
            activation='relu',
            normalize_before=False,
            num_levels=encoder['transformerlayers']['attn_cfgs']['num_levels'],
            num_points=encoder['transformerlayers']['attn_cfgs']['num_points'],
        )
        self.encoder = DetrTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=encoder['num_layers'],
            norm=nn.LayerNorm(feat_channels)  # Final normalization layer
        )

        self.pos_encoding = SinePositionalEncoding(**positional_encoding)

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for _ in range(num_outs):
            self.lateral_convs.append(nn.Conv2d(feat_channels, feat_channels, kernel_size=1))
            self.output_convs.append(
                nn.Sequential(nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False), nn.GroupNorm(norm_cfg['num_groups'], feat_channels),
                              nn.ReLU(inplace=True)))

    def forward(self, feats: List[torch.Tensor]):
        if len(feats) > self.num_levels:
            feats = feats[-self.num_levels:]
            input_convs = self.input_convs[-self.num_levels:]
        else:
            input_convs = self.input_convs
        
        # Apply input convolutions
        mlvl_feats = [input_conv(feat) for input_conv, feat in zip(input_convs, feats)]

        # Prepare for transformer encoder
        mlvl_masks = []
        mlvl_pos_embeds = []
        mlvl_flatten_feats = []
        mlvl_spatial_shapes = []
        mlvl_level_start_index = [0]

        for i, feat in enumerate(mlvl_feats):
            h, w = feat.shape[-2:]
            mask = feat.new_zeros((feat.shape[0], h, w), dtype=torch.bool)
            pos_embed = self.pos_encoding(mask)

            mlvl_masks.append(mask)
            mlvl_pos_embeds.append(pos_embed)
            mlvl_flatten_feats.append(feat.flatten(2).permute(0, 2, 1))  # (B, H*W, C)
            mlvl_spatial_shapes.append([h, w])
            if i > 0:
                mlvl_level_start_index.append(mlvl_level_start_index[-1] + h * w)

        mlvl_flatten_feats = torch.cat(mlvl_flatten_feats, 1)
        mlvl_spatial_shapes = torch.tensor(mlvl_spatial_shapes, dtype=torch.long, device=mlvl_flatten_feats.device)
        mlvl_level_start_index = torch.tensor(mlvl_level_start_index, dtype=torch.long, device=mlvl_flatten_feats.device)
        mlvl_masks = torch.cat([mask.flatten(1) for mask in mlvl_masks], 1)
        mlvl_pos_embeds = torch.cat([pos_embed.flatten(2).permute(0, 2, 1) for pos_embed in mlvl_pos_embeds], 1)

        # Calculate valid_ratios and reference_points
        reference_points = get_reference_points(mlvl_spatial_shapes, mlvl_flatten_feats.device)

        # Transformer Encoder
        encoder_output = self.encoder(
            src=mlvl_flatten_feats.permute(1, 0, 2),  # (S, N, E)
            mask=mlvl_masks,  # (N, S)
            src_key_padding_mask=mlvl_masks,  # (N, S)
            pos=mlvl_pos_embeds.permute(1, 0, 2),  # (N, S, E)
            reference_points=reference_points.permute(1, 0, 2, 3),  # (S, N, num_levels, 2)
            spatial_shapes=mlvl_spatial_shapes,
            level_start_index=mlvl_level_start_index).permute(1, 0, 2)  # (N, S, E) back to (B, H*W, C)

        # Reshape encoder output back to feature maps
        multi_scale_memorys = []
        _cur = 0
        for h, w in mlvl_spatial_shapes:
            multi_scale_memorys.append(encoder_output[:, _cur:_cur + h * w].permute(0, 2, 1).reshape(encoder_output.shape[0], -1, h, w))
            _cur += h * w

        # Apply lateral and output convolutions
        mask_features = self.lateral_convs[0](multi_scale_memorys[0])  # Assuming the first level is used for mask_features
        for i in range(self.num_outs):
            mask_features = self.output_convs[i](mask_features)

        return mask_features, multi_scale_memorys, mlvl_masks
