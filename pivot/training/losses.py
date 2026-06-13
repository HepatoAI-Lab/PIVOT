from __future__ import annotations

import torch
import torch.nn.functional as F


def scaled_cosine_error_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
    gamma: float = 3.0,
) -> torch.Tensor:
    cosine_error = 1.0 - F.cosine_similarity(prediction, reference, dim=-1)
    losses = torch.clamp(cosine_error, min=0.0).pow(gamma)
    if mask is not None:
        mask = mask.to(device=losses.device, dtype=torch.bool)
        if not mask.any():
            return losses.new_tensor(0.0)
        losses = losses[mask]
    return losses.mean()


def cosine_distance_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible unscaled cosine-distance loss."""

    return scaled_cosine_error_loss(prediction, reference, mask=mask, gamma=1.0)


def pivot_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor | None = None,
    he_reference: torch.Tensor | None = None,
    cd34_reference: torch.Tensor | None = None,
    he_mask: torch.Tensor | None = None,
    cd34_mask: torch.Tensor | None = None,
    lambda_morph: float = 0.05,
    lambda_vasc: float = 0.10,
    alignment_gamma: float = 3.0,
    classification_weight: float = 1.0,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = outputs["logit"].device
    total = torch.zeros((), device=device)
    parts: dict[str, torch.Tensor] = {}

    if labels is not None and classification_weight > 0:
        labels = labels.to(device=device, dtype=torch.float32)
        cls = F.binary_cross_entropy_with_logits(outputs["logit"], labels, pos_weight=pos_weight)
        parts["loss_cls"] = cls
        total = total + classification_weight * cls

    if he_reference is not None and lambda_morph > 0:
        morph = scaled_cosine_error_loss(
            outputs["morphology_embedding"],
            he_reference.to(device),
            he_mask,
            gamma=alignment_gamma,
        )
        parts["loss_morph"] = morph
        total = total + lambda_morph * morph

    if cd34_reference is not None and lambda_vasc > 0:
        vasc = scaled_cosine_error_loss(
            outputs["vascular_embedding"],
            cd34_reference.to(device),
            cd34_mask,
            gamma=alignment_gamma,
        )
        parts["loss_vasc"] = vasc
        total = total + lambda_vasc * vasc

    parts["loss_total"] = total
    return parts
