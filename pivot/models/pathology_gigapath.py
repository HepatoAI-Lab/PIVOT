from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .layers import GatedAttentionAggregator


class PathologyReferenceModel(nn.Module):
    """Patient-level H&E or CD34 reference model from slide-level embeddings."""

    def __init__(
        self,
        slide_embedding_dim: int = 768,
        reference_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patient_aggregator = GatedAttentionAggregator(
            input_dim=slide_embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=reference_dim,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(reference_dim, 1),
        )

    def forward(
        self,
        slide_embeddings: torch.Tensor,
        slide_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reference_embedding, attention = self.patient_aggregator(slide_embeddings, slide_mask)
        logit = self.classifier(reference_embedding).squeeze(-1)
        return {
            "reference_embedding": reference_embedding,
            "logit": logit,
            "slide_attention": attention,
        }


class GigaPathSlideEmbedder(nn.Module):
    """Thin wrapper around local Prov-GigaPath tile and slide encoders."""

    def __init__(
        self,
        gigapath_repo: str | Path = "/home/rj/Claude_space/Project/prov-gigapath",
        tile_encoder_path: str | Path | None = "/home/rj/Claude_space/Project/prov-gigapath/hf_weights/pytorch_model.bin",
        slide_encoder_path: str | Path | None = "/home/rj/Claude_space/Project/prov-gigapath/hf_weights/slide_encoder.pth",
        global_pool: bool = False,
    ) -> None:
        super().__init__()
        gigapath_repo = Path(gigapath_repo)
        if str(gigapath_repo) not in sys.path:
            sys.path.insert(0, str(gigapath_repo))
        try:
            from gigapath.pipeline import load_tile_slide_encoder
        except Exception as exc:  # pragma: no cover - depends on local GigaPath deps
            raise ImportError(
                f"Could not import Prov-GigaPath from {gigapath_repo}. "
                "Install prov-gigapath dependencies or check config.paths.gigapath_repo."
            ) from exc
        tile_path = str(tile_encoder_path) if tile_encoder_path else ""
        slide_path = str(slide_encoder_path) if slide_encoder_path else ""
        tile_encoder, slide_encoder = load_tile_slide_encoder(tile_path, slide_path, global_pool=global_pool)
        self.tile_encoder = tile_encoder
        self.slide_encoder = slide_encoder

    @torch.no_grad()
    def encode_slide_from_tile_embeddings(
        self,
        tile_embeddings: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if tile_embeddings.dim() == 2:
            tile_embeddings = tile_embeddings.unsqueeze(0)
            coords = coords.unsqueeze(0)
        outputs = self.slide_encoder(tile_embeddings, coords, all_layer_embed=False)
        if isinstance(outputs, list):
            outputs = outputs[-1]
        return outputs
