import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import copy

from .ms_deform_attn import MSDeformAttn


def _get_clones(module, N):
    """Helper function to clone a module N times."""
    return nn.ModuleList([_get_deepcopy_module(module) for i in range(N)])


def _get_deepcopy_module(module):
    """Helper function to deepcopy a module."""


    # This is a simplified deepcopy. For actual DETR, you might need
    # to ensure proper weight initialization or shared weights if applicable.
    # For a generic transformer, this is usually sufficient.
    # return type(module)(**{k: v for k, v in module.__dict__.items() if not k.startswith('_') and k != 'training'})
def _get_deepcopy_module(module):
    """Helper function to deepcopy a module."""
    return copy.deepcopy(module)


def _get_activation_fn(activation: str):
    """Helper function to get activation function."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder, as used in DETR.
    This layer consists of a multi-head self-attention mechanism,
    followed by a feed-forward network. Layer normalization and dropout
    are applied at appropriate steps.
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 normalize_before: bool = False,
                 num_levels: int = 4,
                 num_points: int = 4):
        """
        Args:
            d_model (int): The number of expected features in the input (e.g., 256 for DETR).
            nhead (int): The number of heads in the multiheadattention models.
            dim_feedforward (int): The dimension of the feedforward network model (e.g., 2048).
            dropout (float): The dropout value.
            activation (str): The activation function of the intermediate layer, e.g., "relu" or "gelu".
            normalize_before (bool): If True, apply layer norm before attention/FFN, else after.
                                     DETR typically uses post-norm (False).
            num_levels (int): The number of feature map used in Attention.
            num_points (int): The number of sampling points for each query in each head.
        """
        super().__init__()
        self.self_attn = MSDeformAttn(d_model=d_model, n_heads=nhead, n_levels=num_levels, n_points=num_points, ratio=1.0)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: torch.Tensor | None):
        """
        Adds positional embeddings to the input tensor.
        Args:
            tensor (torch.Tensor): The input tensor.
            pos (torch.Tensor | None): Positional embeddings.
        Returns:
            torch.Tensor: Tensor with positional embeddings added.
        """
        return tensor if pos is None else tensor + pos

    def forward_post_norm(self,
                          src: torch.Tensor,
                          src_mask: torch.Tensor | None = None,
                          src_key_padding_mask: torch.Tensor | None = None,
                          pos: torch.Tensor | None = None,
                          reference_points: torch.Tensor = None,
                          spatial_shapes: torch.Tensor = None,
                          level_start_index: torch.Tensor = None):
        """
        Forward pass with post-normalization (norm after attention/FFN).
        This is the typical setup for DETR.
        Args:
            src (torch.Tensor): The sequence to the encoder layer (S, N, E).
                                S is the sequence length, N is the batch size, E is the embedding dimension.
            src_mask (torch.Tensor | None): The mask for the src sequence.
            src_key_padding_mask (torch.Tensor | None): The mask for the src keys per batch.
            pos (torch.Tensor | None): Positional embeddings to be added to src.
        Returns:
            torch.Tensor: The output of the encoder layer.
        """
        # Self-attention block
        # For MSDeformAttn, query, key, value are src. query_pos is pos.
        # reference_points, spatial_shapes, level_start_index are needed.
        # The arguments to self_attn are passed positionally, so we need to
        # make sure they are in the correct order.
        # The expected signature is:
        # query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask
        # MSDeformAttn expects batch-first format (N, S, E), but src is in (S, N, E)
        # So we need to transpose for the attention call
        query_batch_first = self.with_pos_embed(src, pos).transpose(0, 1)  # (S, N, E) -> (N, S, E)
        src_batch_first = src.transpose(0, 1)  # (S, N, E) -> (N, S, E)

        # reference_points is in (S, N, num_levels, 2) format, need to transpose to (N, S, num_levels, 2)
        reference_points_batch_first = reference_points.transpose(0, 1)  # (S, N, num_levels, 2) -> (N, S, num_levels, 2)

        src2 = self.self_attn(query=query_batch_first,
                              reference_points=reference_points_batch_first,
                              input_flatten=src_batch_first,
                              input_spatial_shapes=spatial_shapes,
                              input_level_start_index=level_start_index,
                              input_padding_mask=src_key_padding_mask)

        # Convert back to (S, N, E) format
        src2 = src2.transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed-forward network block
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre_norm(self,
                         src: torch.Tensor,
                         src_mask: torch.Tensor | None = None,
                         src_key_padding_mask: torch.Tensor | None = None,
                         pos: torch.Tensor | None = None,
                         reference_points: torch.Tensor = None,
                         spatial_shapes: torch.Tensor = None,
                         level_start_index: torch.Tensor = None):
        """
        Forward pass with pre-normalization (norm before attention/FFN).
        Args:
            src (torch.Tensor): The sequence to the encoder layer (S, N, E).
            src_mask (torch.Tensor | None): The mask for the src sequence.
            src_key_padding_mask (torch.Tensor | None): The mask for the src keys per batch.
            pos (torch.Tensor | None): Positional embeddings to be added to src.
        Returns:
            torch.Tensor: The output of the encoder layer.
        """
        # Self-attention block
        src2 = self.norm1(src)
        # MSDeformAttn expects batch-first format (N, S, E), but src is in (S, N, E)
        query_batch_first = self.with_pos_embed(src2, pos).transpose(0, 1)  # (S, N, E) -> (N, S, E)
        src_batch_first = src.transpose(0, 1)  # (S, N, E) -> (N, S, E)

        # reference_points is in (S, N, num_levels, 2) format, need to transpose to (N, S, num_levels, 2)
        reference_points_batch_first = reference_points.transpose(0, 1)  # (S, N, num_levels, 2) -> (N, S, num_levels, 2)

        src2 = self.self_attn(query=query_batch_first,
                              reference_points=reference_points_batch_first,
                              input_flatten=src_batch_first,
                              input_spatial_shapes=spatial_shapes,
                              input_level_start_index=level_start_index,
                              input_padding_mask=src_key_padding_mask)

        # Convert back to (S, N, E) format
        src2 = src2.transpose(0, 1)
        src = src + self.dropout1(src2)

        # Feed-forward network block
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                src_key_padding_mask: torch.Tensor | None = None,
                pos: torch.Tensor | None = None,
                reference_points: torch.Tensor = None,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None):
        """
        Forward pass of the TransformerEncoderLayer.
        Args:
            src (torch.Tensor): The sequence to the encoder layer (S, N, E).
                                S is the sequence length, N is the batch size, E is the embedding dimension.
            src_mask (torch.Tensor | None): The mask for the src sequence.
            src_key_padding_mask (torch.Tensor | None): The mask for the src keys per batch.
            pos (torch.Tensor | None): Positional embeddings to be added to src.
        Returns:
            torch.Tensor: The output of the encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre_norm(src, src_mask, src_key_padding_mask, pos, reference_points, spatial_shapes, level_start_index)
        return self.forward_post_norm(src, src_mask, src_key_padding_mask, pos, reference_points, spatial_shapes, level_start_index)


class DetrTransformerEncoder(nn.Module):
    """
    The Transformer encoder module, composed of multiple TransformerEncoderLayer instances.
    This module processes the input sequence (e.g., flattened image features from a CNN backbone)
    and outputs context-rich features.
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: nn.Module | None = None):
        """
        Args:
            encoder_layer (nn.Module): An instance of TransformerEncoderLayer.
            num_layers (int): The number of encoder layers to stack.
            norm (nn.Module | None): An optional normalization layer to apply to the output
                                     of the last encoder layer.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: torch.Tensor,
                mask: torch.Tensor | None = None,
                src_key_padding_mask: torch.Tensor | None = None,
                pos: torch.Tensor | None = None,
                reference_points: torch.Tensor = None,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None):
        """
        Forward pass of the TransformerEncoder.
        Args:
            src (torch.Tensor): The sequence to the encoder (S, N, E).
                                S is the sequence length, N is the batch size, E is the embedding dimension.
            mask (torch.Tensor | None): The mask for the src sequence.
            src_key_padding_mask (torch.Tensor | None): The mask for the src keys per batch.
            pos (torch.Tensor | None): Positional embeddings to be added to src at each layer.
        Returns:
            torch.Tensor: The output of the encoder.
        """
        output = src

        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos,
                           reference_points=reference_points,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DetrTransformerDecoderLayer(nn.Module):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | `mmcv.ConfigDict`):
            Configs for `self_attn` and `cross_attn`
            If it is a dict, it would be used as the configdict of the self_attn
            module, and the cross_attn module would be named as `self_attn`
            also. If it is a list, the first element would be used as the
            configdict of the self_attn module, and the second element would
            be used as the configdict of the cross_attn module.
        ffn_cfgs (list[`mmcv.ConfigDict`] | `mmcv.ConfigDict`):
            Configs for FFN.
            If it is a dict, it would be used as the configdict of the ffn
            module. If it is a list, the elements would be used as the
            configdict of multiple ffn modules.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Store in `self.operation_order`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `dict(type='LN')`.
        act_cfg (dict): Config dict for activation layer.
            Default: `dict(type='ReLU', inplace=True)`.
        ffn_dropout (float): The dropout rate for ffn. Default: 0.0.
        post_norm (bool): Whether to add a post normalization layer.
            Default: `False`.
        add_identity (bool): Whether to add an identity connection.
            Default: `False`.
        batch_first (bool): Whether the input tensor is batch first.
            Default: `False`.
    """

    def __init__(self,
                 attn_cfgs: Dict,
                 ffn_cfgs: Dict,
                 operation_order: Tuple[str],
                 norm_cfg: Dict = dict(type='LN'),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 ffn_dropout: float = 0.0,
                 post_norm: bool = False,
                 add_identity: bool = False,
                 batch_first: bool = False):
        super().__init__()
        self.operation_order = operation_order
        self.post_norm = post_norm
        self.add_identity = add_identity
        self.batch_first = batch_first

        d_model = attn_cfgs['embed_dims']
        self.norms = nn.ModuleList()
        # Create a normalization layer for each operation that needs one
        # Based on the forward method, every operation uses a norm layer
        for layer_name in operation_order:
            self.norms.append(nn.LayerNorm(d_model))

        new_attn_cfgs = {
            'embed_dim': attn_cfgs['embed_dims'],  # Rename 'embed_dims' to 'embed_dim'
            'num_heads': attn_cfgs['num_heads'],
            'dropout': attn_cfgs['attn_drop'],  # Use 'attn_drop' value for 'dropout'
            'batch_first': attn_cfgs['batch_first'],
        }
        self.self_attn = nn.MultiheadAttention(**new_attn_cfgs)
        self.cross_attn = nn.MultiheadAttention(**new_attn_cfgs)

        self.ffn = nn.Sequential(nn.Linear(ffn_cfgs['embed_dims'], ffn_cfgs['feedforward_channels']), nn.ReLU(inplace=True), nn.Dropout(ffn_cfgs['ffn_drop']),
                                 nn.Linear(ffn_cfgs['feedforward_channels'], ffn_cfgs['embed_dims']))

    def with_pos_embed(self, tensor, pos: torch.Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                query_pos: torch.Tensor,
                key_pos: torch.Tensor,
                attn_masks: Tuple[torch.Tensor] = (None, None),
                query_key_padding_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward function for `DetrTransformerDecoderLayer`."""
        
        # self attention
        temp_query = query
        q = k = self.with_pos_embed(temp_query, query_pos)
        v = temp_query
        query2 = self.self_attn(q, k, value=v, attn_mask=attn_masks[0], key_padding_mask=query_key_padding_mask)[0]
        query = query + query2
        query = self.norms[0](query)

        # cross attention
        temp_query = query
        q = self.with_pos_embed(temp_query, query_pos)
        k = self.with_pos_embed(key, key_pos)
        v = value
        query2 = self.cross_attn(q, k, value=v, attn_mask=attn_masks[1], key_padding_mask=key_padding_mask)[0]
        query = query + query2
        query = self.norms[1](query)

        # ffn
        temp_query = query
        query2 = self.ffn(temp_query)
        query = query + query2
        query = self.norms[2](query)

        return query


class DetrTransformerDecoder(nn.Module):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, num_layers: int, layer_cfg: Dict, return_intermediate: bool = False, post_norm_cfg=dict(type="LN")):
        super().__init__()
        self.layers = nn.ModuleList([DetrTransformerDecoderLayer(**layer_cfg) for _ in range(num_layers)])
        self.return_intermediate = return_intermediate

        if post_norm_cfg is not None:
            # print(layer_cfg)
            embed_dims = layer_cfg['ffn_cfgs']['embed_dims']
            self.post_norm = nn.LayerNorm(embed_dims)
        else:
            self.post_norm = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                query_pos: torch.Tensor,
                key_pos: torch.Tensor,
                attn_masks: Tuple[torch.Tensor] = (None, None),
                query_key_padding_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            output = query
            for layer in self.layers:
                output = layer(output, key, value, query_pos, key_pos, attn_masks, query_key_padding_mask, key_padding_mask, **kwargs)

            if self.post_norm:
                output = self.post_norm(output)
            return output.unsqueeze(0)

        intermediate = []
        output = query
        for layer in self.layers:
            output = layer(output, key, value, query_pos, key_pos, attn_masks, query_key_padding_mask, key_padding_mask, **kwargs)
            if self.post_norm is not None:
                intermediate.append(self.post_norm(output))
            else:
                intermediate.append(output)

        return torch.stack(intermediate)
