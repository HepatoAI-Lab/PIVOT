from __future__ import annotations

import torch
from torch import nn

from .mri_triad import PIVOTMRIEncoder


class PIVOTModel(nn.Module):
    """MRI model with H&E and CD34 reference-alignment heads."""

    def __init__(
        self,
        mri_encoder: PIVOTMRIEncoder,
        model_dim: int = 768,
        reference_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mri_encoder = mri_encoder
        self.vetc_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        self.morphology_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, reference_dim),
        )
        self.vascular_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, reference_dim),
        )

    def forward(
        self,
        mri_volumes: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mri_outputs = self.mri_encoder(mri_volumes, sequence_mask)
        vetc_embedding = mri_outputs["vetc_embedding"]
        return {
            **mri_outputs,
            "logit": self.vetc_head(vetc_embedding).squeeze(-1),
            "morphology_embedding": self.morphology_head(vetc_embedding),
            "vascular_embedding": self.vascular_head(vetc_embedding),
        }
