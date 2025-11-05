import torch
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask=None):
        """
        Scale-invariant logarithmic loss
        Args:
            pred: predicted depth [B, 1, H, W] or [B, H, W]
            target: ground truth depth [B, 1, H, W] or [B, H, W]
            valid_mask: optional valid mask
        """
        # Cast to float32 to prevent instability in mixed precision
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        # 确保pred和target有相同的形状
        if pred.dim() != target.dim():
            if pred.dim() == 4 and target.dim() == 3:
                target = target.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            elif pred.dim() == 3 and target.dim() == 4:
                pred = pred.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        if valid_mask is None:
            # 创建默认的有效掩码（深度值大于0）
            # 确保掩码与target形状完全一致
            target_mask = target > 0
            pred_mask = pred > 0

            # 如果pred和target维度不同，需要调整pred_mask
            if pred_mask.shape != target_mask.shape:
                if pred_mask.dim() == 4 and target_mask.dim() == 3:
                    pred_mask = pred_mask.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
                elif pred_mask.dim() == 3 and target_mask.dim() == 4:
                    pred_mask = pred_mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

            valid_mask = target_mask & pred_mask

        valid_mask = valid_mask.detach()

        # 确保有有效像素
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # 将张量展平以便索引
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        valid_mask_flat = valid_mask.view(-1)

        # 计算对数差
        valid_pred = pred_flat[valid_mask_flat]
        valid_target = target_flat[valid_mask_flat]

        # 确保没有零值或负值
        valid_pred = torch.clamp(valid_pred, min=1e-6)
        valid_target = torch.clamp(valid_target, min=1e-6)

        diff_log = torch.log(valid_target) - torch.log(valid_pred)

        # SiLog损失
        # 避免使用torch.sqrt导致的in-place操作问题，改用torch.pow(..., 0.5)
        variance_term = torch.pow(diff_log, 2).mean()
        bias_term = self.lambd * torch.pow(diff_log.mean(), 2)
        loss_squared = variance_term - bias_term
        loss = torch.pow(torch.clamp(loss_squared, min=1e-8), 0.5)

        return loss


class FeatureAlignmentLoss(nn.Module):
    """特征对齐损失：拉近不同任务的特征表示"""

    def __init__(self, distance_type='l2'):
        super().__init__()
        self.distance_type = distance_type

    def forward(self, features_1, features_2):
        """
        计算两个特征之间的对齐损失
        Args:
            features_1: 任务1的特征 [B, C, H, W] 或 [B, C]
            features_2: 任务2的特征 [B, C, H, W] 或 [B, C]
        """
        if features_1.shape != features_2.shape:
            # 如果形状不同，进行自适应池化对齐
            if len(features_1.shape) == 4 and len(features_2.shape) == 4:
                # 都是4D特征图，对齐到相同尺寸
                h, w = min(features_1.shape[2], features_2.shape[2]), min(features_1.shape[3], features_2.shape[3])
                features_1 = F.adaptive_avg_pool2d(features_1, (h, w))
                features_2 = F.adaptive_avg_pool2d(features_2, (h, w))
            elif len(features_1.shape) == 4:
                # features_1是4D，features_2是2D，对features_1进行全局池化
                features_1 = F.adaptive_avg_pool2d(features_1, (1, 1)).flatten(1)
            elif len(features_2.shape) == 4:
                # features_2是4D，features_1是2D，对features_2进行全局池化
                features_2 = F.adaptive_avg_pool2d(features_2, (1, 1)).flatten(1)

        if self.distance_type == 'l2':
            return F.mse_loss(features_1, features_2)
        elif self.distance_type == 'l1':
            return F.l1_loss(features_1, features_2)
        elif self.distance_type == 'cosine':
            # 余弦距离
            features_1_flat = features_1.contiguous().view(features_1.size(0), -1)
            features_2_flat = features_2.contiguous().view(features_2.size(0), -1)
            cosine_sim = F.cosine_similarity(features_1_flat, features_2_flat, dim=1)
            return 1 - cosine_sim.mean()
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")


class TaskConsistencyLoss(nn.Module):
    """任务一致性损失：通过KL散度促进任务间知识转移"""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_1, pred_2, task_1_type='depth', task_2_type='seg'):
        """
        计算两个任务预测之间的一致性损失
        Args:
            pred_1: 任务1的预测 [B, C1, H, W]
            pred_2: 任务2的预测 [B, C2, H, W]
            task_1_type: 任务1类型 ('depth' 或 'seg')
            task_2_type: 任务2类型 ('depth' 或 'seg')
        """
        # 将预测转换为概率分布
        if task_1_type == 'depth':
            # 深度预测转换为软分布
            pred_1_flat = pred_1.contiguous().view(pred_1.size(0), -1)
            pred_1_norm = F.softmax(pred_1_flat / self.temperature, dim=1)
        else:
            # 分割预测已经是logits
            pred_1_flat = pred_1.contiguous().view(pred_1.size(0), -1)
            pred_1_norm = F.softmax(pred_1_flat / self.temperature, dim=1)

        if task_2_type == 'depth':
            # 深度预测转换为软分布
            pred_2_flat = pred_2.contiguous().view(pred_2.size(0), -1)
            pred_2_norm = F.softmax(pred_2_flat / self.temperature, dim=1)
        else:
            # 分割预测已经是logits
            pred_2_flat = pred_2.contiguous().view(pred_2.size(0), -1)
            pred_2_norm = F.softmax(pred_2_flat / self.temperature, dim=1)

        # 如果维度不同，需要对齐
        if pred_1_norm.shape[1] != pred_2_norm.shape[1]:
            # 使用较小的维度
            min_dim = min(pred_1_norm.shape[1], pred_2_norm.shape[1])
            pred_1_norm = pred_1_norm[:, :min_dim]
            pred_2_norm = pred_2_norm[:, :min_dim]

        # 计算双向KL散度（对称）
        kl_1_2 = F.kl_div(pred_1_norm.log(), pred_2_norm, reduction='batchmean')
        kl_2_1 = F.kl_div(pred_2_norm.log(), pred_1_norm, reduction='batchmean')

        return (kl_1_2 + kl_2_1) / 2


class MultiTaskLossWithAlignment(nn.Module):
    """多任务损失，包含特征对齐和任务一致性正则项"""

    def __init__(self,
                 depth_criterion,
                 seg_criterion,
                 depth_weight=1.0,
                 seg_weight=1.0,
                 align_weight=0.1,
                 consistency_weight=0.05,
                 feature_align_type='l2',
                 consistency_temperature=1.0,
                 num_classes=None):
        super().__init__()

        self.depth_criterion = depth_criterion
        self.seg_criterion = seg_criterion
        self.num_classes = num_classes

        # 任务损失权重
        self.depth_weight = depth_weight
        self.seg_weight = seg_weight

        # 正则项权重
        self.align_weight = align_weight
        self.consistency_weight = consistency_weight

        # 正则项损失函数
        self.feature_align_loss = FeatureAlignmentLoss(distance_type=feature_align_type)
        self.consistency_loss = TaskConsistencyLoss(temperature=consistency_temperature)

    def forward(self,
                depth_pred=None, depth_gt=None, depth_features=None,
                seg_pred=None, seg_gt=None, seg_features=None,
                use_dummy_depth=False, use_dummy_seg=False, mask=None):
        """
        计算多任务损失
        Args:
            depth_pred: 深度预测 [B, 1, H, W]
            depth_gt: 深度真值 [B, 1, H, W]
            depth_features: 深度任务的中间特征
            seg_pred: 分割预测 [B, C, H, W]
            seg_gt: 分割真值 [B, H, W]
            seg_features: 分割任务的中间特征
            use_dummy_depth: 是否使用虚拟深度数据（损失权重为0）
            use_dummy_seg: 是否使用虚拟分割数据（损失权重为0）
        """
        total_loss = 0
        loss_dict = {}

        # 任务特定损失
        if depth_pred is not None and depth_gt is not None:
            depth_loss = self.depth_criterion(depth_pred, depth_gt, valid_mask=mask)
            # 如果是虚拟数据，损失权重为0
            depth_weight = 0.0 if use_dummy_depth else self.depth_weight
            total_loss += depth_weight * depth_loss
            loss_dict['depth_loss'] = depth_loss.item()

        if seg_pred is not None and seg_gt is not None:
            # 确保 seg_gt 是 long 类型
            seg_gt_long = seg_gt.long()

            # 检查并修复标签值，确保在有效范围内
            if self.num_classes is not None:
                # 获取忽略索引（默认为255）
                ignore_idx = getattr(self.seg_criterion, 'ignore_index', 255)
                
                # 打印调试信息：检查标签范围
                if torch.isfinite(seg_gt_long).all():
                    unique_labels = torch.unique(seg_gt_long)
                    if len(unique_labels) > 0:
                        min_label = unique_labels.min().item()
                        max_label = unique_labels.max().item()
                        if min_label < 0 or (max_label >= self.num_classes and max_label != ignore_idx):
                            print(f"Warning: Found labels outside valid range. Min: {min_label}, Max: {max_label}, "
                                f"Expected: [0, {self.num_classes-1}] or {ignore_idx}")
                else:
                    print("Warning: Found non-finite values in segmentation labels")
                
                # 有效标签：在 [0, num_classes-1] 范围内，或者等于 ignore_idx
                valid_class_mask = (seg_gt_long >= 0) & (seg_gt_long < self.num_classes)
                ignore_mask = (seg_gt_long == ignore_idx)
                valid_mask = valid_class_mask | ignore_mask
                
                # 将无效标签设置为 ignore_idx
                seg_gt_long = torch.where(valid_mask, seg_gt_long, torch.tensor(ignore_idx, device=seg_gt_long.device))

            # 确保标签值在有效范围内
            valid_labels = (seg_gt_long >= 0) & (seg_gt_long < self.num_classes)
            ignore_idx = getattr(self.seg_criterion, 'ignore_index', 255)
            valid_labels = valid_labels | (seg_gt_long == ignore_idx)
            if not valid_labels.all():
                invalid_indices = torch.where(~valid_labels)
                seg_gt_long[~valid_labels] = ignore_idx  # 将无效标签设置为忽略索引

            seg_loss = self.seg_criterion(seg_pred, seg_gt_long)
            # 如果是虚拟数据，损失权重为0
            seg_weight = 0.0 if use_dummy_seg else self.seg_weight
            total_loss += seg_weight * seg_loss
            loss_dict['seg_loss'] = seg_loss.item()

        # 特征对齐损失（只有在两个任务都有数据时才计算）
        if (depth_features is not None and seg_features is not None and
            self.align_weight > 0):
            align_loss = self.feature_align_loss(depth_features, seg_features)
            total_loss += self.align_weight * align_loss
            loss_dict['align_loss'] = align_loss.item()
        else:
            loss_dict['align_loss'] = 0.0

        # 任务一致性损失（只有在两个任务都有数据时才计算）
        if (depth_pred is not None and seg_pred is not None and
            self.consistency_weight > 0):
            consistency_loss = self.consistency_loss(depth_pred, seg_pred, 'depth', 'seg')
            total_loss += self.consistency_weight * consistency_loss
            loss_dict['consistency_loss'] = consistency_loss.item()
        else:
            loss_dict['consistency_loss'] = 0.0

        return total_loss, loss_dict