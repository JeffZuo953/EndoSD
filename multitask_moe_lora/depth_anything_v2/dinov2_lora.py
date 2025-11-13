# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
from contextlib import contextmanager
import math
import logging
from typing import Sequence, Tuple, Union, Callable, Optional, Dict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

# Import LoRA layers and utilities
from .dinov2_layers.lora import mark_only_lora_as_trainable, lora_state_dict
from .dinov2_layers.mlp_lora import Mlp_LoRA
from .dinov2_layers.swiglu_ffn_lora import SwiGLUFFNFused_LoRA
from .dinov2_layers.attention_lora import MemEffAttention_LoRA
from .dinov2_layers.block_lora import NestedTensorBlock_LoRA as Block_LoRA # Use LoRA block
from .dinov2_layers.endo_unid_adapter import AdapterScopeController, EndoUniDBlock
# Import original layers needed
from .dinov2_layers import PatchEmbed # Assuming PatchEmbed doesn't need LoRA
from .dinov2_layers.mlp import Mlp
from .dinov2_layers.swiglu_ffn import SwiGLUFFNFused

logger = logging.getLogger("dinov2")


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


class DinoVisionTransformer_LoRA(nn.Module):
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
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block_LoRA,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        # --- Mode Control --- #
        mode='original',
        # --- LoRA Args (for backward compatibility) --- #
        lora_r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
        lora_bias: str = 'none',
        endo_unid_cfg: Optional[dict] = None,
        **kwargs
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
            lora_r: (int) LoRA rank.
            lora_alpha: (int) LoRA alpha scaling factor.
            lora_dropout: (float) LoRA dropout probability.
            lora_bias: (str) Controls which bias parameters are trainable ('none', 'lora_only', 'all').
        """
        # 根据mode参数解码配置项
        attention_only_lora = mode == 'legacy-lora'

        if mode in ('endo-unid', 'mtlora', 'mtlga'):
            use_lora = True
            lora_r = 0  # ranks handled per scope
            lora_alpha = 1
        elif mode == 'lora-only' or attention_only_lora:
            use_lora = True
            lora_r = lora_r if lora_r is not None else 4
            lora_alpha = lora_alpha if lora_alpha is not None else 8
        else:  # original
            use_lora = False
            lora_r = 0
            lora_alpha = 1

        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mode = mode
        self.is_endo_unid = mode in {'endo-unid', 'mtlora', 'mtlga'}
        self.use_lora = use_lora
        self.attention_only_lora = attention_only_lora
        self.endo_unid_cfg = endo_unid_cfg if self.is_endo_unid else None
        if self.is_endo_unid:
            if self.endo_unid_cfg is None:
                raise ValueError("EndoUniD mode requires endo_unid_cfg dictionary")
            self.adapter_controller = AdapterScopeController(default=self.endo_unid_cfg.get('default_scopes', ['shared']))

        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # 根据LoRA模式决定FFN类型
        if use_lora:
            if attention_only_lora:
                logger.info("using original MLP FFN with attention-only LoRA (legacy-lora mode)")
                _ffn_layer_class = Mlp
            elif ffn_layer == "mlp":
                logger.info("using MLP_LoRA layer as FFN")
                _ffn_layer_class = Mlp_LoRA
            elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
                logger.info("using SwiGLUFFNFused_LoRA layer as FFN")
                _ffn_layer_class = SwiGLUFFNFused_LoRA
            elif ffn_layer == "identity":
                logger.info("using Identity layer as FFN")
                def _identity_ffn(*args, **kwargs):
                    # Identity layer doesn't have LoRA params
                    return nn.Identity()
                _ffn_layer_class = _identity_ffn
            else:
                raise NotImplementedError(f"FFN layer {ffn_layer} not implemented for LoRA")
        else:
            # 非LoRA模式，使用原始组件
            if ffn_layer == "mlp":
                logger.info("using MLP layer as FFN")
                _ffn_layer_class = Mlp
            elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
                logger.info("using SwiGLUFFN layer as FFN")
                _ffn_layer_class = SwiGLUFFNFused
            elif ffn_layer == "identity":
                logger.info("using Identity layer as FFN")
                def _identity_ffn(*args, **kwargs):
                    return nn.Identity()
                _ffn_layer_class = _identity_ffn
            else:
                raise NotImplementedError(f"FFN layer {ffn_layer} not implemented")

        # Pass LoRA params to block_fn constructor，扩展为支持MoE参数
        blocks_list = []
        for i in range(depth):
            if self.is_endo_unid:
                block_scopes = self.endo_unid_cfg['block_scopes'].get(i, ['shared'])
                scope_specs = self._build_endo_scope_specs(block_scopes)
                block_kwargs = {
                    'dim': embed_dim,
                    'num_heads': num_heads,
                    'mlp_ratio': mlp_ratio,
                    'qkv_bias': qkv_bias,
                    'proj_bias': proj_bias,
                    'ffn_bias': ffn_bias,
                    'drop_path': dpr[i],
                    'norm_layer': norm_layer,
                    'act_layer': act_layer,
                    'scope_controller': self.adapter_controller,
                    'scope_specs': scope_specs,
                    'shared_shards': self.endo_unid_cfg.get('shared_shards', 1),
                    'dropout': self.endo_unid_cfg.get('dropout', 0.0),
                    'init_values': init_values,
                }
                blocks_list.append(EndoUniDBlock(**block_kwargs))
                continue

            block_kwargs = {
                'dim': embed_dim,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': qkv_bias,
                'proj_bias': proj_bias,
                'ffn_bias': ffn_bias,
                'drop_path': dpr[i],
                'norm_layer': norm_layer,
                'act_layer': act_layer,
                'ffn_layer': _ffn_layer_class,
                'init_values': init_values,
            }
            if self.use_lora:
                block_kwargs['lora_r'] = lora_r
                block_kwargs['lora_alpha'] = lora_alpha
                block_kwargs['lora_dropout'] = lora_dropout

            blocks_list.append(block_fn(**block_kwargs))
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

        # Apply LoRA parameter freezing logic
        if lora_r > 0 or self.is_endo_unid:
            mark_only_lora_as_trainable(self, bias=lora_bias)

    def _build_endo_scope_specs(self, scopes: Sequence[str]) -> Dict[str, Dict[str, int]]:
        cfg = self.endo_unid_cfg or {}
        rank_table = cfg.get('ranks', {})
        alpha_table = cfg.get('alphas', {})
        specs = {}
        for scope in scopes:
            specs[scope] = {
                'r': rank_table.get(scope, 0),
                'alpha': alpha_table.get(scope, 1),
            }
        return specs

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias
        )
        
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None, extra_tokens=None):
        B, nc, w, h = x.shape
        tokens = self.patch_embed(x)
        if masks is not None:
            tokens = torch.where(masks.unsqueeze(-1), self.mask_token.to(tokens.dtype).unsqueeze(0), tokens)

        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        base_seq = torch.cat((cls_tokens, tokens), dim=1)
        pos_embed = self.interpolate_pos_encoding(base_seq, w, h)
        cls_tokens = cls_tokens + pos_embed[:, :1]
        patch_tokens = tokens + pos_embed[:, 1:]

        parts = [cls_tokens]
        if extra_tokens is not None:
            parts.append(extra_tokens.to(dtype=patch_tokens.dtype))
        parts.append(patch_tokens)
        x = torch.cat(parts, dim=1)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list, extra_tokens=None):
        if extra_tokens is None or torch.is_tensor(extra_tokens):
            token_list = [extra_tokens] * len(x_list)
        else:
            token_list = extra_tokens
        x = [
            self.prepare_tokens_with_masks(x, masks, extra_tokens=tokens)
            for x, masks, tokens in zip(x_list, masks_list, token_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, extra_tokens=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks, extra_tokens=extra_tokens)

        x = self.prepare_tokens_with_masks(x, masks, extra_tokens=extra_tokens)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1, extra_tokens=None):
        x = self.prepare_tokens_with_masks(x, extra_tokens=extra_tokens)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1, extra_tokens=None):
        x = self.prepare_tokens_with_masks(x, extra_tokens=extra_tokens)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
        extra_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n, extra_tokens=extra_tokens)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n, extra_tokens=extra_tokens)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
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

    # --- Add LoRA Helper Methods --- #
    def mark_only_lora_as_trainable(self, bias: str = 'none'):
        mark_only_lora_as_trainable(self, bias=bias)

    def lora_state_dict(self, bias: str = 'none'):
        return lora_state_dict(self, bias=bias)

    def set_active_adapter_scopes(self, scopes):
        if hasattr(self, "adapter_controller"):
            self.adapter_controller.set_active(scopes)

    def adapter_scope(self, scopes):
        if not hasattr(self, "adapter_controller"):
            @contextmanager
            def _noop_scope():
                yield
            return _noop_scope()
        return self.adapter_controller.activate(scopes)
    # ----------------------------- #


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small_lora(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer_LoRA(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block_LoRA, attn_class=MemEffAttention_LoRA),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base_lora(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer_LoRA(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block_LoRA, attn_class=MemEffAttention_LoRA),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large_lora(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer_LoRA(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block_LoRA, attn_class=MemEffAttention_LoRA),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2_lora(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer_LoRA(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block_LoRA, attn_class=MemEffAttention_LoRA),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def DINOv2_LoRA(model_name, mode='original', lora_r=None, lora_alpha=None, lora_dropout=0.0, lora_bias='none', **kwargs):
    model_zoo_lora = {
        "vits": vit_small_lora,
        "vitb": vit_base_lora,
        "vitl": vit_large_lora,
        "vitg": vit_giant2_lora
    }
    
    if model_name not in model_zoo_lora:
        raise ValueError(f"Unknown DINOv2 LoRA model name: {model_name}")

    # Define base config and override with kwargs
    base_config = {
        'img_size': 518,
        'patch_size': 14,
        'init_values': 1.0,
        'ffn_layer': "mlp" if model_name != "vitg" else "swiglufused",
        'block_chunks': 0,
        'num_register_tokens': 0,
        'interpolate_antialias': False,
        'interpolate_offset': 0.1,
        'mode': mode,
        # LoRA args - 由mode决定，但可以被kwargs覆盖
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'lora_bias': lora_bias,
    }
    config = {**base_config, **kwargs} # kwargs override base_config

    # Call the specific LoRA model factory
    model = model_zoo_lora[model_name](**config)
    return model
