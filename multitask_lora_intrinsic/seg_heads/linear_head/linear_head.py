import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from ..base_head import BaseSegHead


class LinearBNHead(BaseSegHead):

    def __init__(self, in_channels: List[int], channels: int, num_classes: int, in_index: List[int], resize_factors: Optional[List[float]] = None, align_corners: bool = False, **kwargs):
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, in_index=in_index, align_corners=align_corners, input_transform="resize_concat")

        self.resize_factors = resize_factors

        # 计算实际的输入通道数
        if isinstance(in_channels, list):
            bn_in_channels = sum([in_channels[i] for i in in_index])
        else:
            bn_in_channels = in_channels

        self.bn = nn.SyncBatchNorm(bn_in_channels)
        self.cls_seg = nn.Conv2d(bn_in_channels, self.num_classes, kernel_size=1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # inputs are already reshaped to (B, C, H, W)
        inputs = [inputs[i] for i in self.in_index]

        def _sanitize(tensor: torch.Tensor) -> torch.Tensor:
            tensor = torch.nan_to_num(tensor, nan=0.0)
            return torch.clamp(tensor, -100.0, 100.0)

        inputs = [_sanitize(x) for x in inputs]

        if self.resize_factors is not None:
            assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
            inputs = [
                _sanitize(F.interpolate(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area"))
                for x, f in zip(inputs, self.resize_factors)
            ]

        if len(inputs) > 1:
            upsampled_inputs = [
                _sanitize(F.interpolate(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners))
                for x in inputs
            ]
            x = torch.cat(upsampled_inputs, dim=1)
        else:
            x = inputs[0]

        x = _sanitize(x)
        feats = _sanitize(self.bn(x))
        output = self.cls_seg(feats)
        output = _sanitize(output)
        return output
