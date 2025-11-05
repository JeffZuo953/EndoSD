import torch
import torch.nn as nn


class _BaseCameraHead(nn.Module):
    """Common utilities shared by camera heads."""

    def __init__(self, embed_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
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
    ) -> None:
        super().__init__(embed_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: Tensor with shape [B, N, C], containing patch embeddings.

        Returns:
            Tensor [B, 4] with normalized fx, fy, cx, cy.
        """
        encoded = self.encoder(patch_tokens)
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
    ) -> None:
        super().__init__(embed_dim, hidden_dim)
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

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: Tensor with shape [B, N, C], containing patch embeddings.

        Returns:
            Tensor [B, 4] with normalized fx, fy, cx, cy.
        """
        bsz = patch_tokens.size(0)
        camera_tok = self.camera_token.expand(bsz, -1, -1)
        register_tok = self.register_token.expand(bsz, -1, -1)
        tokens = torch.cat([camera_tok, register_tok, patch_tokens], dim=1)
        encoded = self.encoder(tokens)
        camera_feat = encoded[:, 0]  # camera token position
        logits = self.output_head(camera_feat)
        return self._normalize_output(logits)
