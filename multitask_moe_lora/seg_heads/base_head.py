import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class BaseSegHead(nn.Module):

    def __init__(self, in_channels: List[int], channels: int, num_classes: int, in_index: List[int], input_transform: str, align_corners: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners

    def _transform_inputs(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Select features from specified indices
        inputs = [inputs[i] for i in self.in_index]

        if self.input_transform == "resize_concat":
            # Upsample and concatenate features
            upsampled_inputs = [F.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            ) for x in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            # This is handled by the specific head implementation (e.g., Mask2Former)
            pass

        return inputs

    def forward(self, inputs: List[torch.Tensor]):
        raise NotImplementedError
