#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

# 导入dinov3原始层
from .dinov3_layers.ffn_layers import SwiGLUFFN
from .dinov3_layers.attention import SelfAttention
from .dinov3_layers.block import Block as OriginalBlock
from .dinov3_layers.patch_embed import PatchEmbed

logger = logging.getLogger("dinov3")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=OriginalBlock,
        ffn_layer="swiglu",
        block_chunks=1,
        **kwargs
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # DINOv3 vits16plus does not use pos_embed

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "swiglu":
            ffn_layer_class = SwiGLUFFN
        else:
            raise NotImplementedError
            
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer_class,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        # trunc_normal_(self.pos_embed, std=0.02) # DINOv3 vits16plus does not use pos_embed
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    # This method is not needed if pos_embed is removed.
    # def interpolate_pos_encoding(self, x, w, h):
    #     ...

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        # x = x + self.interpolate_pos_encoding(x, w, h) # DINOv3 vits16plus does not use pos_embed
        return x

    def forward_features(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
        }
        
    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        return output

    def get_intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence] = 1, reshape: bool = False, return_class_token: bool = False, norm=True):
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        
        if norm:
            outputs = [self.norm(out) for out in outputs]
            
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
            
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def dinov3_vits16(**kwargs):
    model = DinoVisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs
    )
    return model

def dinov3_vits16plus(**kwargs):
    model = DinoVisionTransformer(
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=6, **kwargs
    )
    return model

def dinov3_vitb16(**kwargs):
    model = DinoVisionTransformer(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, **kwargs
    )
    return model

def dinov3_vitl16(**kwargs):
    model = DinoVisionTransformer(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, **kwargs
    )
    return model


def DINOv3(model_name, **kwargs):
    model_zoo = {
        "dinov3_vits16": dinov3_vits16,
        "dinov3_vits16plus": dinov3_vits16plus,
        "dinov3_vitb16": dinov3_vitb16,
        "dinov3_vitl16": dinov3_vitl16,
    }
    
    if model_name not in model_zoo:
        raise ValueError(f"Unknown DINOv3 model name: {model_name}")

    base_config = {
        'img_size': 512,
        'patch_size': 16,
        'init_values': 1.0,
        'ffn_layer': "swiglu",
        'block_chunks': 0,
    }
    config = {**base_config, **kwargs}

    # Filter out LoRA/MoE specific arguments that the original model does not accept
    known_args = [
        'img_size', 'patch_size', 'in_chans', 'embed_dim', 'depth', 'num_heads',
        'mlp_ratio', 'qkv_bias', 'ffn_bias', 'proj_bias', 'drop_path_rate',
        'drop_path_uniform', 'init_values', 'embed_layer', 'act_layer',
        'block_fn', 'ffn_layer', 'block_chunks'
    ]
    
    # Create a new config dictionary with only the known arguments
    filtered_config = {k: v for k, v in config.items() if k in known_args}

    model = model_zoo[model_name](**filtered_config)
    return model