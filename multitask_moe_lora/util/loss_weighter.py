#!/usr/bin/env python3
"""
动态损失加权模块（仅 UWL）
"""

import torch
from .config import TrainingConfig


class LossWeighter:
    """
    仅实现 Uncertainty to Weigh Losses for Multi-Task Learning (https://arxiv.org/abs/1705.07115)
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.seg_enabled = not getattr(config, "disable_seg_head", False)
        # UWL: log_vars 是可学习的参数，代表每个任务的不确定性。
        # 公式: L_total = sum(exp(-log_var_i) * L_i + log_var_i)
        if self.seg_enabled:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_vars = torch.nn.Parameter(torch.zeros(2, device=device))
        else:
            self.log_vars = None
        # 限制log_vars的搜索范围，避免UWL权重在AMP下指数爆炸。
        self.log_var_bounds = (-6.0, 6.0)

    def get_loss(self, depth_loss: torch.Tensor, seg_loss: torch.Tensor, task: str = None) -> torch.Tensor:
        """
        计算基于UWL的损失。

        Args:
            depth_loss: 深度任务的原始损失
            seg_loss: 分割任务的原始损失
            task: 指定单独任务 ('depth' or 'seg')，否则返回两者之和
        """
        if not self.seg_enabled or self.log_vars is None:
            if task == 'depth':
                return depth_loss
            if task == 'seg':
                return seg_loss
            return depth_loss + seg_loss

        log_vars_clamped = torch.clamp(self.log_vars,
                                       min=self.log_var_bounds[0],
                                       max=self.log_var_bounds[1])

        depth_weight = torch.exp(-log_vars_clamped[0]) * self.config.depth_loss_weight
        seg_weight = torch.exp(-log_vars_clamped[1]) * self.config.seg_loss_weight

        depth_term = depth_loss * depth_weight + log_vars_clamped[0]
        seg_term = seg_loss * seg_weight + log_vars_clamped[1]

        if task == 'depth':
            return depth_term
        if task == 'seg':
            return seg_term
        return depth_term + seg_term

    def sanitize_parameters(self) -> None:
        """在优化器step之后对log_vars做投影，清理NaN/Inf并限制在安全范围内。"""
        if self.log_vars is None:
            return
        with torch.no_grad():
            nan_free = torch.nan_to_num(self.log_vars,
                                        nan=0.0,
                                        posinf=self.log_var_bounds[1],
                                        neginf=self.log_var_bounds[0])
            self.log_vars.copy_(nan_free.clamp_(self.log_var_bounds[0], self.log_var_bounds[1]))
