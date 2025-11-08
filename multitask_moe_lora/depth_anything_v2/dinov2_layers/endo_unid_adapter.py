#!/usr/bin/env python3
"""
Scoped LoRA adapters for EndoUniD mode.
Implements task-aware adapter routing so depth/seg/shared/camera branches can
own disjoint low-rank updates while sharing the same frozen backbone weights.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .drop_path import DropPath
from .layer_scale import LayerScale


class AdapterScopeController:
    """Tracks which adapter scopes should be active for the current forward."""

    def __init__(self, default: Optional[Iterable[str]] = None):
        self._stack: List[List[str]] = [list(default or ["shared"])]

    @contextmanager
    def activate(self, scopes: Iterable[str]):
        scopes = list(scopes)
        self._stack.append(scopes)
        try:
            yield
        finally:
            self._stack.pop()

    def set_active(self, scopes: Iterable[str]) -> None:
        self._stack[-1] = list(scopes)

    def current(self) -> List[str]:
        return self._stack[-1]


class _BaseAdapter(nn.Module):
    """Common logic for low-rank adapters."""

    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = (alpha / r) if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_a = nn.Parameter(torch.zeros(r, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward_from_projection(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DenseAdapter(_BaseAdapter):
    """Standard LoRA adapter that consumes the full feature dimension."""

    def forward_from_projection(self, x: torch.Tensor) -> torch.Tensor:
        if self.r == 0:
            return x.new_zeros(*x.shape[:-1], self.out_features)
        projected = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return projected * self.scaling


class ShardedAdapter(nn.Module):
    """Splits the input dimension into shards and applies a lightweight adapter per shard."""

    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float, num_shards: int):
        super().__init__()
        self.num_shards = max(1, num_shards)
        self.out_features = out_features
        self.shards = nn.ModuleList()
        base = in_features // self.num_shards
        remainder = in_features % self.num_shards
        start = 0
        for shard_idx in range(self.num_shards):
            width = base + (1 if shard_idx < remainder else 0)
            end = start + width
            adapter = DenseAdapter(width, out_features, r, alpha, dropout)
            adapter.start = start
            adapter.end = end
            self.shards.append(adapter)
            start = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.shards:
            return x.new_zeros(*x.shape[:-1], self.out_features)
        output = None
        for shard in self.shards:
            sub_x = x[..., shard.start:shard.end]
            shard_out = shard.forward_from_projection(sub_x)
            if output is None:
                output = shard_out
            else:
                output = output + shard_out
        return output


class ScopedLoRALinear(nn.Module):
    """
    Frozen linear projection + a bank of low-rank adapters.
    Only adapters whose scopes are active will contribute during forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        scope_controller: AdapterScopeController,
        scope_specs: Dict[str, Dict[str, int]],
        shared_shards: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.scope_controller = scope_controller
        self.adapters = nn.ModuleDict()
        self.dropout = dropout

        for scope_name, spec in scope_specs.items():
            r = spec.get("r", 0)
            alpha = spec.get("alpha", 1)
            if r <= 0:
                continue
            if scope_name == "shared" and shared_shards > 1:
                adapter = ShardedAdapter(
                    in_features=in_features,
                    out_features=out_features,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    num_shards=shared_shards,
                )
            else:
                adapter = DenseAdapter(in_features, out_features, r, alpha, dropout)
            self.adapters[scope_name] = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        active_scopes = self.scope_controller.current()
        for scope in active_scopes:
            if scope not in self.adapters:
                continue
            adapter = self.adapters[scope]
            if isinstance(adapter, DenseAdapter):
                result = result + adapter.forward_from_projection(x)
            else:
                result = result + adapter(x)
        return result


class EndoUniDAttention(nn.Module):
    """Attention module powered by scoped adapters."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        scope_controller: AdapterScopeController,
        scope_specs: Dict[str, Dict[str, int]],
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        shared_shards: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = ScopedLoRALinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            scope_controller=scope_controller,
            scope_specs=scope_specs,
            shared_shards=shared_shards,
            dropout=dropout,
        )
        self.proj = ScopedLoRALinear(
            dim,
            dim,
            bias=proj_bias,
            scope_controller=scope_controller,
            scope_specs=scope_specs,
            shared_shards=shared_shards,
            dropout=dropout,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EndoUniDMlp(nn.Module):
    """Scoped adapter MLP (two-layer)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int],
        scope_controller: AdapterScopeController,
        scope_specs: Dict[str, Dict[str, int]],
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        shared_shards: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = ScopedLoRALinear(
            in_features,
            hidden_features,
            bias=bias,
            scope_controller=scope_controller,
            scope_specs=scope_specs,
            shared_shards=shared_shards,
            dropout=dropout,
        )
        self.act = act_layer()
        self.fc2 = ScopedLoRALinear(
            hidden_features,
            in_features,
            bias=bias,
            scope_controller=scope_controller,
            scope_specs=scope_specs,
            shared_shards=shared_shards,
            dropout=dropout,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EndoUniDBlock(nn.Module):
    """Transformer block that routes adapters based on EndoUniD scopes."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        scope_controller: AdapterScopeController,
        scope_specs: Dict[str, Dict[str, int]],
        shared_shards: int = 1,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EndoUniDAttention(
            dim=dim,
            num_heads=num_heads,
            scope_controller=scope_controller,
            scope_specs=scope_specs,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            shared_shards=shared_shards,
            dropout=dropout,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = EndoUniDMlp(
            in_features=dim,
            hidden_features=hidden_dim,
            scope_controller=scope_controller,
            scope_specs=scope_specs,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            shared_shards=shared_shards,
            dropout=dropout,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
