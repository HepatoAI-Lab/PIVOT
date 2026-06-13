from __future__ import annotations

import torch
from torch import nn


class ResidualMLPAdapter(nn.Module):
    """Lightweight sequence adapter applied after pooled MRI sequence embedding."""

    def __init__(self, dim: int, bottleneck_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class GatedAttentionAggregator(nn.Module):
    """Variable-length attention pooling for slide- or patient-level bags."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.attn_v = nn.Linear(input_dim, hidden_dim)
        self.attn_u = nn.Linear(input_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool a variable-length bag.

        Args:
            x: Tensor with shape [batch, items, dim].
            mask: Optional boolean tensor [batch, items], True for valid items.
        """
        scores = self.attn_w(torch.tanh(self.attn_v(x)) * torch.sigmoid(self.attn_u(x))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        pooled = self.norm(self.proj(pooled))
        return pooled, weights
