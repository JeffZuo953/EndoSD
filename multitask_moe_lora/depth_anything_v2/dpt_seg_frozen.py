import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2
from seg_heads.linear_head.linear_head import LinearBNHead

class DepthAnythingV2_Seg_Frozen(nn.Module):

    def __init__(self, encoder='vitl', num_classes=2, features=256, seg_head_type='BNHead', seg_input_type='last_four', **kwargs):
        super(DepthAnythingV2_Seg_Frozen, self).__init__()

        # 模型配置
        self.encoder = encoder
        self.num_classes = num_classes
        self.seg_head_type = seg_head_type
        self.seg_input_type = seg_input_type

        # 层配置
        self.depth_layer_idx = {'vits': [2, 5, 8, 11], 'vitb': [2, 5, 8, 11], 'vitl': [4, 11, 17, 23], 'vitg': [9, 19, 29, 39]}
        self.num_layers = {'vits': 12, 'vitb': 12, 'vitl': 24, 'vitg': 40}

        self.backbone = DINOv2(model_name=encoder)
        for param in self.backbone.parameters():
            param.requires_grad = False

        if seg_head_type == 'BNHead':
            total_layers = self.num_layers[self.encoder]
            if self.seg_input_type == 'last':
                self.seg_layer_idx = [total_layers - 1]
            elif self.seg_input_type == 'last_four':
                self.seg_layer_idx = list(range(total_layers - 4, total_layers))
            elif self.seg_input_type == 'from_depth':
                self.seg_layer_idx = self.depth_layer_idx[self.encoder]
            else:
                raise ValueError(f"Unknown seg_input_type: {self.seg_input_type}")

            num_seg_layers = len(self.seg_layer_idx)
            seg_in_channels = [self.backbone.embed_dim] * num_seg_layers
            
            self.seg_head = LinearBNHead(
                in_channels=seg_in_channels,
                channels=features,
                num_classes=num_classes,
                in_index=list(range(num_seg_layers))
            )
        else:
            raise ValueError(f"Unsupported seg_head_type: {seg_head_type}")

    def forward_segmentation(self, features, h, w):
        """语义分割前向传播"""
        # 处理最后4层特征，不包含cls_token
        out = []
        for x in features:
            b, _, c = x.shape
            patch_h, patch_w = h // 14, w // 14
            x = x.permute(0, 2, 1).reshape(b, c, patch_h, patch_w)
            out.append(x)

        # 确保 seg_head 接收的特征与 linear_head 中的定义一致
        seg_pred = self.seg_head(out)
        return seg_pred

    def forward(self, x, masks=None, task='seg', return_features=False):
        """
        前向传播 - 与dpt_multitask.py保持一致的接口
        """
        h, w = x.shape[-2:]

        # BNHead - 根据 seg_layer_idx 提取特征
        features = self.backbone.get_intermediate_layers(x, self.seg_layer_idx, return_class_token=False)
        
        if return_features:
            # 如果需要返回特征，使用与dpt_multitask.py一致的格式
            results = {}
            
            # 分割预测 - 直接使用features
            seg_raw_features = self.forward_segmentation(features, h, w)
            seg_pred = F.interpolate(seg_raw_features, size=(h, w), mode='bilinear', align_corners=True)
            results['seg'] = seg_pred
            
            # 返回分割任务的中间特征（用于特征对齐）
            results['seg_features'] = F.adaptive_avg_pool2d(seg_raw_features, (8, 8))
            
            # 返回共享backbone特征（用于特征对齐）
            backbone_features = features[-1] if isinstance(features[-1], torch.Tensor) else features[-1]
            # 重新整理特征维度
            patch_h, patch_w = h // 14, w // 14
            backbone_features = backbone_features.permute(0, 2, 1).reshape(backbone_features.shape[0], backbone_features.shape[-1], patch_h, patch_w).contiguous()
            results['features'] = F.adaptive_avg_pool2d(backbone_features, (8, 8))
            
            return results
        else:
            # 标准的分割预测 - 直接使用features
            seg_raw_features = self.forward_segmentation(features, h, w)
            seg_pred = F.interpolate(seg_raw_features, size=(h, w), mode='bilinear', align_corners=True)
            return seg_pred
                
    def get_segmentation_prediction(self, x):
        """仅获取分割预测 - 与dpt_multitask.py保持一致"""
        result = self.forward(x, task='seg', return_features=False)
        if isinstance(result, dict):
            return result['seg']
        return result
