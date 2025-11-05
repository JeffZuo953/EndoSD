# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# (LoRA modifications added)

import logging
from typing import Optional

from torch import Tensor
from torch import nn

# Import original Attention components if needed (e.g., for inheritance)
# from .attention import Attention as Attention_Original # If inheriting
from .lora import LoRALinear

logger = logging.getLogger("dinov2")

try:
    from xformers.ops import memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available for LoRA attention")
    XFORMERS_AVAILABLE = False

# Renamed to Attention_LoRA
class Attention_LoRA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # --- LoRA Args --- #
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        # merge_weights: bool = True # Handled within LoRALinear
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Use LoRALinear
        self.qkv = LoRALinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # merge_weights=merge_weights # Pass if needed, else defaults
        )
        self.attn_drop = nn.Dropout(attn_drop)
        # Use LoRALinear for projection
        self.proj = LoRALinear(
            dim,
            dim,
            bias=proj_bias,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=True, # Common for projection layers
            # merge_weights=merge_weights # Pass if needed, else defaults
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # LoRALinear handles the forward pass
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # LoRALinear handles the forward pass
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Renamed to MemEffAttention_LoRA
class MemEffAttention_LoRA(Attention_LoRA):
    # Inherits modified __init__ from Attention_LoRA
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage with LoRA attention"
            # Fall back to parent LoRA implementation
            return super().forward(x)

        B, N, C = x.shape
        # self.qkv is already LoRALinear
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        # memory_efficient_attention should work directly with the output of LoRALinear
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        # self.proj is already LoRALinear
        x = self.proj(x)
        x = self.proj_drop(x)
        return x 