from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .layers import ResidualMLPAdapter


DEFAULT_SEQUENCES = ("T1WI", "T2WI", "DWI", "ADC", "AP", "PVP", "DP")


class TriadSwinBBackbone(nn.Module):
    """SwinUNETR-style Triad-SwinB feature encoder.

    This mirrors the Swin component used in the local Triad QuickStart while
    avoiding imports from the PlainConvUNet side of that repository.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        feature_size: int = 48,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        try:
            from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
            from monai.utils import ensure_tuple_rep
        except Exception as exc:  # pragma: no cover - depends on MONAI install
            raise ImportError(
                "Triad-SwinB requires MONAI. Install requirements.txt in the active environment."
            ) from exc
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            use_v2=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.swinViT(x)
        pooled = [F.adaptive_avg_pool3d(state, (1, 1, 1)).flatten(1) for state in hidden_states]
        return torch.cat(pooled, dim=1)


def _normalise_triad_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalised = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalised[new_key] = value
    return normalised


def _load_triad_backbone(
    triad_repo: str | Path,
    checkpoint_path: str | Path | None,
    feature_size: int = 48,
    use_checkpoint: bool = True,
) -> nn.Module:
    """Load Triad-SwinB as a single-volume 3D MRI backbone."""
    triad_repo = Path(triad_repo)
    if str(triad_repo) not in sys.path:
        sys.path.insert(0, str(triad_repo))
    try:
        model = TriadSwinBBackbone(feature_size=feature_size, use_checkpoint=use_checkpoint)
    except Exception as exc:  # pragma: no cover - depends on MONAI deps
        raise ImportError(
            "Could not construct Triad-SwinB backbone. Install MONAI and check requirements.txt."
        ) from exc
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _normalise_triad_state_dict(state.get("state_dict", state))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[PIVOT] Triad unexpected keys: {len(unexpected)}")
        if missing:
            print(f"[PIVOT] Triad missing keys: {len(missing)}")
    return model


class PIVOTMRIEncoder(nn.Module):
    """Shared Triad-SwinB encoder with sequence tokens for multiparametric liver MRI."""

    def __init__(
        self,
        triad_repo: str | Path = "../Triad",
        triad_checkpoint: str | Path | None = "../Triad/weights/Triad-SwinB-SimMIM.pth",
        sequences: tuple[str, ...] = DEFAULT_SEQUENCES,
        model_dim: int = 768,
        adapter_bottleneck_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.sequences = tuple(sequences)
        self.backbone = _load_triad_backbone(triad_repo, triad_checkpoint, use_checkpoint=use_checkpoint)
        self.sequence_projector = nn.LazyLinear(model_dim)
        self.adapters = nn.ModuleDict(
            {
                seq: ResidualMLPAdapter(model_dim, adapter_bottleneck_dim, dropout)
                for seq in self.sequences
            }
        )
        self.sequence_type_embeddings = nn.Parameter(torch.zeros(len(self.sequences), model_dim))
        self.vetc_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=transformer_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(model_dim)
        self._init_tokens()

    def _init_tokens(self) -> None:
        nn.init.trunc_normal_(self.sequence_type_embeddings, std=0.02)
        nn.init.trunc_normal_(self.vetc_token, std=0.02)

    def set_triad_trainable(self, final_stage_only: bool = False) -> None:
        """Control Triad fine-tuning according to the staged PIVOT protocol."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        if not final_stage_only:
            return
        for name, param in self.backbone.named_parameters():
            if any(key in name for key in ("layers4", "layers4c", "proj_out")):
                param.requires_grad = True

    def forward(
        self,
        volumes: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode seven registered MRI sequences.

        Args:
            volumes: Tensor shaped [B, S, C, D, H, W] or [B, S, D, H, W].
            sequence_mask: Optional boolean tensor [B, S], True for available sequences.
        """
        if volumes.dim() == 5:
            volumes = volumes.unsqueeze(2)
        if volumes.dim() != 6:
            raise ValueError("volumes must have shape [B,S,C,D,H,W] or [B,S,D,H,W]")
        batch_size, n_seq = volumes.shape[:2]
        if n_seq != len(self.sequences):
            raise ValueError(f"expected {len(self.sequences)} sequences, got {n_seq}")
        if sequence_mask is None:
            sequence_mask = torch.ones(batch_size, n_seq, device=volumes.device, dtype=torch.bool)
        else:
            sequence_mask = sequence_mask.to(device=volumes.device, dtype=torch.bool)

        tokens = []
        for idx, seq in enumerate(self.sequences):
            sequence_volume = volumes[:, idx]
            sequence_embedding = self.backbone(sequence_volume)
            sequence_embedding = self.sequence_projector(sequence_embedding)
            sequence_embedding = self.adapters[seq](sequence_embedding)
            sequence_embedding = sequence_embedding + self.sequence_type_embeddings[idx].unsqueeze(0)
            tokens.append(sequence_embedding.unsqueeze(1))
        sequence_tokens = torch.cat(tokens, dim=1)

        cls_token = self.vetc_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([cls_token, sequence_tokens], dim=1)
        cls_mask = torch.ones(batch_size, 1, device=volumes.device, dtype=torch.bool)
        token_mask = torch.cat([cls_mask, sequence_mask], dim=1)
        encoded = self.sequence_transformer(
            transformer_input,
            src_key_padding_mask=~token_mask,
        )
        vetc_embedding = self.norm(encoded[:, 0])
        return {
            "vetc_embedding": vetc_embedding,
            "sequence_tokens": encoded[:, 1:],
            "sequence_mask": sequence_mask,
        }
