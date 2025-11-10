import math
from typing import Optional

import torch
import torch.nn as nn


class TokenLoRAAdapter(nn.Module):
    """Light-weight adapter that perturbs tokens before the camera head."""

    def __init__(self, dim: int, rank: int = 0, alpha: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        if rank <= 0:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
            self.dropout = nn.Identity()
            self.scaling = 0.0
            return
        self.lora_a = nn.Parameter(torch.zeros(rank, dim))
        self.lora_b = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scaling = alpha / rank

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return tokens
        delta = (self.dropout(tokens) @ self.lora_a.T @ self.lora_b.T) * self.scaling
        return tokens + delta


class _BaseCameraHead(nn.Module):
    """Common utilities shared by camera heads."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        adapter_rank: int = 0,
        adapter_alpha: int = 1,
        adapter_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.input_adapter = TokenLoRAAdapter(embed_dim, adapter_rank, adapter_alpha, adapter_dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def _normalize_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Map unconstrained logits to [0, 1] range for fx, fy, cx, cy."""
        return torch.sigmoid(logits)


class SimpleCameraHead(_BaseCameraHead):
    """
    Transformer encoder that pools ViT patch tokens to predict normalized intrinsics.

    This version does not introduce additional tokens; it simply refines the patch tokens
    and averages them to obtain a global representation.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        adapter_rank: int = 0,
        adapter_alpha: int = 1,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_dim, hidden_dim, adapter_rank, adapter_alpha, adapter_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, patch_tokens: torch.Tensor, patch_h: Optional[int] = None, patch_w: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            patch_tokens: Tensor with shape [B, N, C], containing patch embeddings.

        Returns:
            Tensor [B, 4] with normalized fx, fy, cx, cy.
        """
        tokens = self.input_adapter(patch_tokens)
        encoded = self.encoder(tokens)
        global_feat = encoded.mean(dim=1)
        logits = self.output_head(global_feat)
        return self._normalize_output(logits)


class VGGTLiteCameraHead(_BaseCameraHead):
    """
    VGGT-inspired camera head.

    Adds learnable camera/register tokens in front of patch tokens and applies a
    Transformer encoder. The prediction is taken from the dedicated camera token.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        adapter_rank: int = 0,
        adapter_alpha: int = 1,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_dim, hidden_dim, adapter_rank, adapter_alpha, adapter_dropout)
        self.camera_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, patch_tokens: torch.Tensor, patch_h: Optional[int] = None, patch_w: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            patch_tokens: Tensor with shape [B, N, C], containing patch embeddings.

        Returns:
            Tensor [B, 4] with normalized fx, fy, cx, cy.
        """
        bsz = patch_tokens.size(0)
        camera_tok = self.camera_token.expand(bsz, -1, -1)
        register_tok = self.register_token.expand(bsz, -1, -1)
        tokens = torch.cat([camera_tok, register_tok, self.input_adapter(patch_tokens)], dim=1)
        encoded = self.encoder(tokens)
        camera_feat = encoded[:, 0]  # camera token position
        logits = self.output_head(camera_feat)
        return self._normalize_output(logits)


class ProLikeCameraHead(_BaseCameraHead):
    """
    FOV encoder inspired camera head that operates on spatial feature maps.

    It reshapes the final transformer block tokens into a 2D map and applies a
    stack of stride-2 convolutions followed by a 1x1 prediction of fx/fy/cx/cy.
    """

    def __init__(
        self,
        embed_dim: int,
        adapter_rank: int = 0,
        adapter_alpha: int = 1,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__(embed_dim, hidden_dim=embed_dim, adapter_rank=adapter_rank, adapter_alpha=adapter_alpha, adapter_dropout=adapter_dropout)
        num_features = embed_dim
        self.conv_head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features // 2, num_features // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features // 4, num_features // 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_features // 8, 4, kernel_size=6, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, patch_tokens: torch.Tensor, patch_h: Optional[int] = None, patch_w: Optional[int] = None) -> torch.Tensor:
        bsz, num_patches, channels = patch_tokens.shape
        if patch_h is None or patch_w is None:
            patch_h = patch_w = int(math.sqrt(num_patches))
        if patch_h * patch_w != num_patches:
            raise ValueError(f"Unable to reshape tokens of length {num_patches} into ({patch_h}, {patch_w}) grid.")

        tokens = self.input_adapter(patch_tokens)
        feature_map = tokens.transpose(1, 2).reshape(bsz, channels, patch_h, patch_w).contiguous()
        logits = self.conv_head(feature_map).flatten(1)
        return self._normalize_output(logits)
