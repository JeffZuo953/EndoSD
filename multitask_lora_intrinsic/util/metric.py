import torch
import numpy as np
from typing import Dict


class SegMetric:
    """
    Comprehensive segmentation metrics calculator using CUDA acceleration
    Compatible with SegmentationMetrics format
    """

    def __init__(self, num_classes: int, device: str = 'cuda'):
        """
        Initialize metrics calculator

        Args:
            num_classes (int): Number of segmentation classes
            device (str): Device for computation ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self):
        """
        Reset all accumulated metrics
        """
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device
        )
        self.total_loss = 0.0
        self.number_of_batches = 0
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, label_trues, label_preds, batch_loss: float = 0.0):
        """
        Update metrics with new batch results

        Args:
            label_trues: Ground truth segmentation masks (numpy or torch tensor)
            label_preds: Predicted segmentation masks (numpy or torch tensor)
            batch_loss (float): Batch loss value
        """
        # Convert to torch tensors if needed
        if isinstance(label_trues, np.ndarray):
            label_trues = torch.from_numpy(label_trues)
        if isinstance(label_preds, np.ndarray):
            label_preds = torch.from_numpy(label_preds)

        # Move to device
        label_trues = label_trues.to(self.device)
        label_preds = label_preds.to(self.device)

        # Process each sample in the batch
        for true_mask, pred_mask in zip(label_trues, label_preds):
            # Flatten masks
            true_flat = true_mask.flatten()
            pred_flat = pred_mask.flatten()

            # Create mask for valid pixels (ignore negative values and out-of-range predictions)
            valid_true_mask = (true_flat >= 0) & (true_flat < self.num_classes)
            valid_pred_mask = (pred_flat >= 0) & (pred_flat < self.num_classes)
            # 同时满足真实标签和预测标签都在有效范围内
            valid_mask = valid_true_mask & valid_pred_mask
            
            # Apply the same mask to both true and pred to ensure they have the same length
            true_valid = true_flat[valid_mask]
            pred_valid = pred_flat[valid_mask]

            # Update confusion matrix using CUDA
            # Ensure only valid indices are used for bincount
            if true_valid.numel() > 0:
                # 确保所有值都在有效范围内，防止reshape错误
                indices = true_valid * self.num_classes + pred_valid
                bincount_result = torch.bincount(
                    indices,
                    minlength=self.num_classes ** 2
                )
                # 确保bincount结果的大小正确
                if bincount_result.numel() == self.num_classes ** 2:
                    self.confusion_matrix += bincount_result.reshape(self.num_classes, self.num_classes)
                else:
                    # 如果大小不匹配，截取或填充到正确大小
                    expected_size = self.num_classes ** 2
                    if bincount_result.numel() > expected_size:
                        bincount_result = bincount_result[:expected_size]
                    else:
                        # 填充零到正确大小
                        padding = torch.zeros(expected_size - bincount_result.numel(),
                                            dtype=bincount_result.dtype,
                                            device=bincount_result.device)
                        bincount_result = torch.cat([bincount_result, padding])
                    self.confusion_matrix += bincount_result.reshape(self.num_classes, self.num_classes)

            # Update pixel accuracy
            batch_total_pixels = valid_mask.sum().item()
            batch_correct_pixels = (true_valid == pred_valid).sum().item()

            self.total_pixels += batch_total_pixels
            self.correct_pixels += batch_correct_pixels

        # Accumulate loss
        self.total_loss += batch_loss
        self.number_of_batches += 1

    def get_scores(self) -> Dict[str, float]:
        """
        计算简化的、无重复的分割指标。

        Returns:
            Dict[str, float]: 计算后的指标字典。
        """
        # Convert confusion matrix to float for calculations
        confusion_matrix_float = self.confusion_matrix.float()

        class_ious = []
        class_dices = []
        class_precisions = []
        class_recalls = []
        class_pixel_accuracies = []
        
        metrics = {}

        for class_index in range(self.num_classes):
            true_positives = confusion_matrix_float[class_index, class_index]
            false_positives = confusion_matrix_float[:, class_index].sum() - true_positives
            false_negatives = confusion_matrix_float[class_index, :].sum() - true_positives

            # Per-class Pixel Accuracy
            class_total_pixels = confusion_matrix_float[class_index, :].sum()
            class_pa = (true_positives / class_total_pixels) if class_total_pixels > 0 else torch.tensor(0.0, device=self.device)
            class_pixel_accuracies.append(class_pa)
            metrics[f'acc_class_{class_index}'] = class_pa.item()

            # IoU
            iou_denominator = true_positives + false_positives + false_negatives
            iou = (true_positives / iou_denominator) if iou_denominator > 0 else torch.tensor(0.0, device=self.device)
            class_ious.append(iou)
            metrics[f'iou_class_{class_index}'] = iou.item()

            # Dice
            dice_denominator = 2 * true_positives + false_positives + false_negatives
            dice = (2 * true_positives / dice_denominator) if dice_denominator > 0 else torch.tensor(0.0, device=self.device)
            class_dices.append(dice)
            metrics[f'dice_class_{class_index}'] = dice.item()
            
            # Precision and Recall
            precision_denominator = true_positives + false_positives
            precision = (true_positives / precision_denominator) if precision_denominator > 0 else torch.tensor(0.0, device=self.device)
            class_precisions.append(precision)
            metrics[f'precision_class_{class_index}'] = precision.item()

            recall_denominator = true_positives + false_negatives
            recall = (true_positives / recall_denominator) if recall_denominator > 0 else torch.tensor(0.0, device=self.device)
            class_recalls.append(recall)
            metrics[f'recall_class_{class_index}'] = recall.item()

        # 全局和平均指标
        total_correct = torch.diag(confusion_matrix_float).sum()
        total_pixels = confusion_matrix_float.sum()
        
        metrics['acc_overall'] = (total_correct / total_pixels).item() if total_pixels > 0 else 0.0
        metrics['total_pixels_N'] = total_pixels.item()
        
        # mIoU and mDice
        if class_ious:
            metrics['miou'] = torch.stack(class_ious).mean().item()
        if class_dices:
            metrics['mdice'] = torch.stack(class_dices).mean().item()
        
        # mAP (mean average precision) 和 mAR (mean average recall)
        if class_precisions:
            metrics['map'] = torch.stack(class_precisions).mean().item()
        if class_recalls:
            metrics['mar'] = torch.stack(class_recalls).mean().item()
            
        # 平均损失
        metrics['avg_loss'] = (self.total_loss / self.number_of_batches) if self.number_of_batches > 0 else 0.0

        return metrics

    def _compute_freq_weighted_acc(self) -> float:
        """
        Compute frequency weighted accuracy using CUDA
        """
        confusion_matrix_float = self.confusion_matrix.float()

        # Compute IoU for each class
        iu = torch.diag(confusion_matrix_float) / (
            confusion_matrix_float.sum(axis=1) +
            confusion_matrix_float.sum(axis=0) -
            torch.diag(confusion_matrix_float)
        )

        # Compute frequency for each class
        freq = confusion_matrix_float.sum(axis=1) / confusion_matrix_float.sum()

        # Compute frequency weighted accuracy
        valid_mask = freq > 0
        if valid_mask.any():
            fwavacc = (freq[valid_mask] * iu[valid_mask]).sum().item()
        else:
            fwavacc = 0.0

        return fwavacc


def eval_depth(pred, target):
    """
    评估深度预测的指标
    Args:
        pred: 预测深度 numpy array
        target: 真实深度 numpy array
    Returns:
        dict: 包含各种深度指标的字典
    """
    import torch
    import numpy as np

    # 转换为torch tensor如果是numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    assert pred.shape == target.shape

    # 确保是正值
    pred = torch.clamp(pred, min=1e-6)
    target = torch.clamp(target, min=1e-6)

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {
        'd1': d1.item(),
        'd2': d2.item(),
        'd3': d3.item(),
        'absrel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'log10': log10.item(),
        'silog': silog.item()
    }
