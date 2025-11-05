import torch
import torch.nn as nn
import math


class SinePositionalEncoding(nn.Module):
    """Position encoding module."""

    def __init__(self, num_feats, temperature=10000, normalize=False, scale=2 * math.pi, eps=1e-6):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps

    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
