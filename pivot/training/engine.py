from __future__ import annotations

import torch
from torch import nn

from .losses import pivot_loss
from .metrics import binary_auc


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


@torch.no_grad()
def build_pathology_references(
    he_model: nn.Module,
    cd34_model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    he_out = he_model(batch["he_slides"].to(device), batch["he_slide_mask"].to(device))
    cd34_out = cd34_model(batch["cd34_slides"].to(device), batch["cd34_slide_mask"].to(device))
    return {
        "he_reference": he_out["reference_embedding"].detach(),
        "cd34_reference": cd34_out["reference_embedding"].detach(),
        "he_mask": batch["he_slide_mask"].any(dim=1).to(device),
        "cd34_mask": batch["cd34_slide_mask"].any(dim=1).to(device),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    he_model: nn.Module | None = None,
    cd34_model: nn.Module | None = None,
    classification_weight: float = 1.0,
    lambda_morph: float = 0.05,
    lambda_vasc: float = 0.10,
    alignment_gamma: float = 3.0,
) -> dict[str, float]:
    model.train()
    if he_model is not None:
        freeze_module(he_model)
    if cd34_model is not None:
        freeze_module(cd34_model)

    running = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["mri"].to(device), batch["sequence_mask"].to(device))
        references = {}
        if he_model is not None and cd34_model is not None:
            references = build_pathology_references(he_model, cd34_model, batch, device)
        losses = pivot_loss(
            outputs,
            labels=batch["label"].to(device),
            classification_weight=classification_weight,
            lambda_morph=lambda_morph,
            lambda_vasc=lambda_vasc,
            alignment_gamma=alignment_gamma,
            **references,
        )
        losses["loss_total"].backward()
        optimizer.step()
        running.append(float(losses["loss_total"].detach().cpu()))
    return {"loss": float(sum(running) / max(1, len(running)))}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, scores = [], []
    for batch in loader:
        outputs = model(batch["mri"].to(device), batch["sequence_mask"].to(device))
        prob = torch.sigmoid(outputs["logit"]).detach().cpu().tolist()
        scores.extend(prob)
        labels.extend(batch["label"].cpu().tolist())
    return {"auc": binary_auc(labels, scores)}
