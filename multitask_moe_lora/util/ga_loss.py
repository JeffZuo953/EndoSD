#!/usr/bin/env python3
"""
Gram alignment loss (GA loss) adapted from dinov3/loss/gram_loss.py.

Used to encourage the depth and segmentation branches to share similar
feature correlations during multi-task training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramLoss(nn.Module):
    """Computes MSE between Gram matrices of two feature tensors."""

    def __init__(
        self,
        apply_norm: bool = True,
        remove_neg: bool = True,
    ) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg

    def forward(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
        img_level: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_feats: Tensor shaped (B, N, C) or (B*N, C)
            teacher_feats: Tensor shaped (B, N, C) or (B*N, C)
            img_level: whether to compute Gram per-image (expects rank-3 tensors)
        """

        if img_level:
            assert len(student_feats.shape) == 3 and len(teacher_feats.shape) == 3, \
                "GramLoss expects (B, N, C) tensors for img-level mode."

        student_feats = student_feats.float()
        teacher_feats = teacher_feats.float()

        if self.apply_norm:
            teacher_feats = F.normalize(teacher_feats, dim=-1)
        if not img_level and teacher_feats.dim() == 3:
            teacher_feats = teacher_feats.flatten(0, 1)
        teacher_gram = torch.matmul(teacher_feats, teacher_feats.transpose(-1, -2))

        if self.apply_norm:
            student_feats = F.normalize(student_feats, dim=-1)
        if not img_level and student_feats.dim() == 3:
            student_feats = student_feats.flatten(0, 1)
        student_gram = torch.matmul(student_feats, student_feats.transpose(-1, -2))

        if self.remove_neg:
            teacher_gram = torch.where(teacher_gram < 0, torch.zeros_like(teacher_gram), teacher_gram)
            student_gram = torch.where(student_gram < 0, torch.zeros_like(student_gram), student_gram)

        return self.mse_loss(student_gram, teacher_gram)
