#!/usr/bin/env python3
"""
轻量 SegFormer 风格的语义分割头：
1x1 conv 将每个尺度映射到统一 embedding 维度，
双线性插值到最高分辨率后拼接、融合并输出类别 logits。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHead(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 embedding_dim: int,
                 num_classes: int,
                 align_corners: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.align_corners = align_corners
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embedding_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True),
            ) for ch in in_channels
        ])
        fuse_in = embedding_dim * len(in_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    @staticmethod
    def _sanitize(tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor.clamp_(-100.0, 100.0)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if not inputs:
            raise ValueError("SegFormerHead requires a non-empty feature list.")
        ref_h, ref_w = inputs[0].shape[-2:]
        fused = []
        for feat, proj in zip(inputs, self.projections):
            feat = self._sanitize(feat)
            if feat.shape[-2:] != (ref_h, ref_w):
                feat = F.interpolate(feat, size=(ref_h, ref_w), mode="bilinear", align_corners=self.align_corners)
            fused.append(proj(feat))
        x = torch.cat(fused, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return self._sanitize(x)
