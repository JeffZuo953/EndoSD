# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# (LoRA modifications added)

import logging
from typing import Callable, List, Any, Tuple, Dict

import torch
from torch import nn, Tensor

# Import LoRA versions of components
from .attention_lora import Attention_LoRA, MemEffAttention_LoRA
from .mlp_lora import Mlp_LoRA
# Import other necessary original components
from .drop_path import DropPath
from .layer_scale import LayerScale

# Import LoRA utilities if needed (e.g., for type hints, though not strictly necessary here)
# from .lora import LoRALinear

logger = logging.getLogger("dinov2")

try:
    from xformers.ops import fmha
    from xformers.ops import scaled_index_add, index_select_cat
    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available for LoRA block")
    XFORMERS_AVAILABLE = False

# Original utility functions (drop_add_residual_stochastic_depth*, get_branges_scales, add_residual, etc.)
# are assumed to be compatible or need slight adjustments if they directly access weights.
# For now, assuming they work on the tensor level.

# --- Copy original utility functions START --- #
def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)

def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor

def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual

attn_bias_cache: Dict[Tuple, Any] = {}

def get_attn_bias_and_cat(x_list, branges=None):
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    cache_key = (*all_shapes, x_list[0].dtype, x_list[0].device)
    if cache_key not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens, B=len(seqlens))
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[cache_key] = attn_bias

    if branges is not None:
        # Ensure indices are on the same device as the tensors
        branges_on_device = [b.to(x_list[0].device) for b in branges]
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges_on_device).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[cache_key], cat_tensors

def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> List[Tensor]:
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    residual_cat = residual_func(x_cat, attn_bias=attn_bias)
    residual_list = attn_bias.split(residual_cat)

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange.to(x.device), residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs
# --- Copy original utility functions END --- #


# Renamed Block to Block_LoRA
class Block_LoRA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        # Use LoRA versions by default
        attn_class: Callable[..., nn.Module] = Attention_LoRA,
        ffn_layer: Callable[..., nn.Module] = Mlp_LoRA,
        # --- LoRA Args --- #
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        # merge_weights: bool = True # Handled within LoRALinear
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Instantiate LoRA Attention layer
        self.attn = attn_class( # type: ignore
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # merge_weights=merge_weights
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Instantiate LoRA FFN layer
        # Need to handle different ffn_layer types (Mlp_LoRA, SwiGLUFFN_LoRA, etc.)
        self.mlp = ffn_layer( # type: ignore
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            # merge_weights=merge_weights
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        # This forward logic should work with LoRA layers implicitly
        # Need to ensure drop_add_residual_stochastic_depth is compatible
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # Ensure the residual funcs work with LoRA layers
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            # Original had drop_path1 here too, assume typo and use drop_path2
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x

# Renamed NestedTensorBlock to NestedTensorBlock_LoRA
class NestedTensorBlock_LoRA(Block_LoRA):
    # Inherits LoRA-modified __init__
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        assert isinstance(self.attn, MemEffAttention_LoRA), "NestedTensor requires MemEffAttention_LoRA"
        assert XFORMERS_AVAILABLE, "Please install xFormers for nested tensors usage (LoRA version)"

        if self.training and self.sample_drop_ratio > 0.0:
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                # Pass attn_bias to LoRA attention layer
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                # FFN doesn't typically use attn_bias
                return self.mlp(self.norm2(x))

            # Use stochastic depth list function (ensure compatibility)
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls1.gamma if isinstance(self.ls1, LayerScale) else None,
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=self.ls2.gamma if isinstance(self.ls2, LayerScale) else None,
            )
            return x_list
        else:
            # Non-stochastic depth path
            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            # Apply LoRA-enabled layers
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            # Use standard Block_LoRA forward for single tensor
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            # Use nested forward for list of tensors
            return self.forward_nested(x_or_x_list)
        else:
            raise TypeError(f"Unsupported input type: {type(x_or_x_list)}") 