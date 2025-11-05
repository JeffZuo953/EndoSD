# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# (LoRA modifications added)

from typing import Callable, Optional

from torch import Tensor, nn

# Import LoRA layer
from .lora import LoRALinear

# Renamed to Mlp_LoRA
class Mlp_LoRA(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        # --- LoRA Args --- #
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        # merge_weights: bool = True # Handled within LoRALinear
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Use LoRALinear
        self.fc1 = LoRALinear(
            in_features,
            hidden_features,
            bias=bias,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # merge_weights=merge_weights
        )
        self.act = act_layer()
        # Use LoRALinear
        self.fc2 = LoRALinear(
            hidden_features,
            out_features,
            bias=bias,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # merge_weights=merge_weights
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        # Forward logic uses LoRALinear implicitly
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x 