# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# (LoRA modifications added)

from typing import Callable, Optional

from torch import Tensor, nn
import torch.nn.functional as F

# Import LoRA layer
from .lora import LoRALinear

# Renamed to SwiGLUFFN_LoRA
class SwiGLUFFN_LoRA(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
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
        hidden_features = hidden_features or int(in_features * 4 / 3 * 2) # SwiGLU FFN has *2 intermediate dim
        # Use LoRALinear
        self.w12 = LoRALinear(
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
        self.w3 = LoRALinear(
            hidden_features // 2, # SwiGLU FFN takes half of the hidden features
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
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.act(x1) * x2
        x = self.w3(hidden)
        x = self.drop(x)
        return x

try:
    # Keep attempt to import xformers SwiGLU for fused kernel if available
    from xformers.ops import SwiGLU as SwiGLU_xformers
    XFORMERS_AVAILABLE = True
except ImportError:
    # Fallback: Use the non-fused LoRA version if xformers not available
    SwiGLU_xformers = SwiGLUFFN_LoRA # This name choice might be confusing, maybe rename variable
    XFORMERS_AVAILABLE = False

# Renamed to SwiGLUFFNFused_LoRA
# Note: This implementation still relies on the separate LoRALinear layers
# A truly fused LoRA implementation might require a custom kernel or different approach.
class SwiGLUFFNFused_LoRA(SwiGLUFFN_LoRA):
    # Inherits the modified __init__ from SwiGLUFFN_LoRA
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        drop: float = 0.0,
        bias: bool = True,
        # --- LoRA Args --- # (Propagate these)
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        # merge_weights: bool = True # Handled within LoRALinear
    ) -> None:
        # Ensure hidden_features calculation matches expectations for fused layers if different
        hidden_features = hidden_features or int(in_features * 4 / 3 * 2)
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            drop=drop,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # merge_weights=merge_weights
        )

    # Forward pass uses the inherited method from SwiGLUFFN_LoRA
    # If a specific fused implementation (like using SwiGLU_xformers) is desired
    # AND compatible with the LoRA structure, it would need modification here.
    # Sticking to the inherited non-fused forward for now.
    # def forward(self, x: Tensor) -> Tensor:
    #     x = self._forward_fused(x) # Requires _forward_fused method adapted for LoRA
    #     return x 