#!/usr/bin/env python3
"""
动态损失加权模块
"""

import torch
from .config import TrainingConfig

class LossWeighter:
    """
    处理不同损失加权策略的类
    - 'fixed': 使用固定的权重
    - 'uwl': Uncertainty to Weigh Losses for Multi-Task Learning (https://arxiv.org/abs/1705.07115)
    - 'dwa': Dynamic Weight Averaging (https://arxiv.org/abs/1803.10704)
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.strategy = config.loss_weighting_strategy

        if self.strategy == 'uwl':
            # UWL: log_vars是可学习的参数，代表每个任务的不确定性。
            # 公式: L_total = sum(1 / (2 * sigma_i^2) * L_i + log(sigma_i))
            # 通过设置 log_var = 2 * log(sigma)，我们可以优化 log_var。
            # L_total = sum(exp(-log_var_i) * L_i + 0.5 * log_var_i)
            # 这里的实现省略了0.5的常数项，因为它不影响优化。
            self.log_vars = torch.nn.Parameter(torch.zeros(2, device='cuda'))
            # 限制log_vars的搜索范围，避免UWL权重在AMP下指数爆炸。
            self.log_var_bounds = (-6.0, 6.0)
        elif self.strategy == 'dwa':
            self.dwa_losses = {'depth': [], 'seg': []}
            # 为移动平均初始化权重
            self.config.depth_loss_weight = 1.0
            self.config.seg_loss_weight = 1.0

    def get_loss(self, depth_loss: torch.Tensor, seg_loss: torch.Tensor, task: str = None) -> torch.Tensor:
        """
        根据策略组合损失。
        增加了一个'task'参数以支持梯度累积修复DDP问题，允许单独计算每个任务的加权损失。
        
        Args:
            depth_loss: 深度任务的原始损失
            seg_loss: 分割任务的原始损失
            task (str, optional): 指定计算哪个任务的损失 ('depth' or 'seg').
                                  如果为None，则计算总损失. Defaults to None.
            
        Returns:
            加权后的损失
        """
        if self.strategy == 'uwl':
            # 数值安全：限制log_vars范围，防止出现exp溢出导致的NaN/Inf梯度
            log_vars_clamped = torch.clamp(self.log_vars,
                                           min=self.log_var_bounds[0],
                                           max=self.log_var_bounds[1])

            depth_weight = torch.exp(-log_vars_clamped[0]) * self.config.depth_loss_weight
            seg_weight = torch.exp(-log_vars_clamped[1]) * self.config.seg_loss_weight
            
            depth_term = depth_loss * depth_weight + log_vars_clamped[0]
            seg_term = seg_loss * seg_weight + log_vars_clamped[1]
            
            if task == 'depth':
                return depth_term
            elif task == 'seg':
                return seg_term
            else:
                return depth_term + seg_term
        
        # 对于 'fixed' 和 'dwa' 策略
        # DWA的权重已经通过update_weights更新到config的depth_loss_weight/seg_loss_weight中了
        # fixed策略直接使用这两个值
        # 所以这里的逻辑对两者都适用
        depth_weight = self.config.depth_loss_weight
        seg_weight = self.config.seg_loss_weight
        
        if task == 'depth':
            return depth_loss * depth_weight
        elif task == 'seg':
            return seg_loss * seg_weight
        else:
            return depth_loss * depth_weight + seg_loss * seg_weight

    def update_weights(self, avg_depth_loss: float, avg_seg_loss: float):
        """
        在每个epoch后更新权重（仅DWA需要）
        
        Args:
            avg_depth_loss: 平均深度损失
            avg_seg_loss: 平均分割损失
        """
        if self.strategy == 'dwa':
            if avg_depth_loss > 0 and avg_seg_loss > 0:
                self._update_dwa_weights(avg_depth_loss, avg_seg_loss)

    def _update_dwa_weights(self, depth_loss: float, seg_loss: float):
        """
        更新DWA权重
        公式: w_i(t) = exp(l_i(t-1) / (l_i(t-2) * T))
        lambda_k(t) = K * w_k(t) / sum_i(w_i(t))
        """
        self.dwa_losses['depth'].append(depth_loss)
        self.dwa_losses['seg'].append(seg_loss)

        if len(self.dwa_losses['depth']) > 2:
            self.dwa_losses['depth'].pop(0)
            self.dwa_losses['seg'].pop(0)

        if len(self.dwa_losses['depth']) == 2:
            # DWA比率: loss_t-1 / loss_t-2（加入极小值避免除0）
            w_depth = float(self.dwa_losses['depth'][0]) / (float(self.dwa_losses['depth'][1]) + 1e-8)
            w_seg = float(self.dwa_losses['seg'][0]) / (float(self.dwa_losses['seg'][1]) + 1e-8)

            # 增加数值稳定性：对损失比率进行截断，防止因损失骤降导致比率过大
            # 一个任务的损失从一个正常值降到一个极小值，会导致比率爆炸，从而使softmax计算溢出
            w_depth = torch.tensor(w_depth).clamp(min=0.1, max=10.0).item()
            w_seg = torch.tensor(w_seg).clamp(min=0.1, max=10.0).item()

            # 使用数值稳定的softmax来避免exp溢出导致的inf/NaN
            t = float(self.config.dwa_temperature) if getattr(self.config, 'dwa_temperature', None) is not None else 2.0
            logits = torch.tensor([w_depth / t, w_seg / t], device='cuda', dtype=torch.float32)
            # softmax内部会做max-shift保证稳定
            probs = torch.softmax(logits, dim=0)

            # 防御式编程：如果出现非有限值，退回到均匀权重
            if not torch.isfinite(probs).all():
                probs = torch.tensor([0.5, 0.5], device='cuda', dtype=torch.float32)

            # K=2，因为有两个任务；总和保持为2
            depth_w = (2.0 * probs[0]).item()
            seg_w = (2.0 * probs[1]).item()

            # 进一步保障：若出现非有限或非正权重，回退到1.0/1.0
            if not (torch.isfinite(torch.tensor(depth_w)) and depth_w > 0):
                depth_w = 1.0
            if not (torch.isfinite(torch.tensor(seg_w)) and seg_w > 0):
                seg_w = 1.0

            # 使用移动平均来平滑权重的剧烈变化，防止震荡
            # alpha 越大，权重变化越平滑
            alpha = 0.9
            new_depth_w = alpha * self.config.depth_loss_weight + (1 - alpha) * depth_w
            new_seg_w = alpha * self.config.seg_loss_weight + (1 - alpha) * seg_w

            self.config.depth_loss_weight = new_depth_w
            self.config.seg_loss_weight = new_seg_w

    def sanitize_parameters(self) -> None:
        """在优化器step之后对log_vars做投影，清理NaN/Inf并限制在安全范围内。"""
        if self.strategy != 'uwl':
            return
        with torch.no_grad():
            nan_free = torch.nan_to_num(self.log_vars,
                                        nan=0.0,
                                        posinf=self.log_var_bounds[1],
                                        neginf=self.log_var_bounds[0])
            self.log_vars.copy_(nan_free.clamp_(self.log_var_bounds[0], self.log_var_bounds[1]))
