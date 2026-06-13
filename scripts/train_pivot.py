#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pivot.data import PIVOTCaseDataset, pivot_collate
from pivot.models import PIVOTMRIEncoder, PIVOTModel, PathologyReferenceModel
from pivot.training import evaluate, train_one_epoch
from pivot.utils.config import load_config, resolve_config_path


def load_reference_model(path: str, cfg: dict, device: torch.device) -> PathologyReferenceModel:
    model = PathologyReferenceModel(
        slide_embedding_dim=cfg["pathology"].get("slide_embedding_dim", 768),
        reference_dim=cfg["model"].get("reference_dim", 768),
        hidden_dim=cfg["pathology"].get("patient_hidden_dim", 256),
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def make_loaders(cfg: dict):
    train_ds = PIVOTCaseDataset(cfg["data"]["manifest_csv"], split=cfg["data"].get("train_split", "train"))
    val_ds = PIVOTCaseDataset(cfg["data"]["manifest_csv"], split=cfg["data"].get("val_split", "val"))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"].get("mri_batch_size", 2),
        shuffle=True,
        collate_fn=pivot_collate,
        num_workers=cfg["training"].get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"].get("mri_batch_size", 2),
        shuffle=False,
        collate_fn=pivot_collate,
        num_workers=cfg["training"].get("num_workers", 4),
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--he-checkpoint", required=True)
    parser.add_argument("--cd34-checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg["output_dir"]) / "pivot_mri"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(cfg)
    he_model = load_reference_model(args.he_checkpoint, cfg, device)
    cd34_model = load_reference_model(args.cd34_checkpoint, cfg, device)

    mri_encoder = PIVOTMRIEncoder(
        triad_repo=resolve_config_path(cfg, cfg["paths"]["triad_repo"]),
        triad_checkpoint=resolve_config_path(cfg, cfg["paths"]["triad_checkpoint"]),
        model_dim=cfg["model"].get("model_dim", 768),
        adapter_bottleneck_dim=cfg["model"].get("adapter_bottleneck_dim", 128),
        transformer_layers=cfg["model"].get("sequence_transformer_layers", 2),
        transformer_heads=cfg["model"].get("sequence_transformer_heads", 8),
    )
    model = PIVOTModel(
        mri_encoder=mri_encoder,
        model_dim=cfg["model"].get("model_dim", 768),
        reference_dim=cfg["model"].get("reference_dim", 768),
    ).to(device)

    # Stage 1 for the MRI pathway: alignment-only training with frozen Triad.
    model.mri_encoder.set_triad_trainable(final_stage_only=False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"].get("new_module_lr", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )
    for epoch in range(1, cfg["training"].get("alignment_epochs", 20) + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            he_model=he_model,
            cd34_model=cd34_model,
            classification_weight=0.0,
            lambda_morph=cfg["training"].get("lambda_morph", 0.5),
            lambda_vasc=cfg["training"].get("lambda_vasc", 0.5),
        )
        print(f"alignment_epoch={epoch} {metrics}")

    # Stage 2 for the MRI pathway: VETC prediction plus pathology-guided alignment.
    model.mri_encoder.set_triad_trainable(final_stage_only=True)
    backbone_params, new_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mri_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            new_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": new_params, "lr": cfg["training"].get("new_module_lr", 1e-4)},
            {"params": backbone_params, "lr": cfg["training"].get("backbone_lr", 1e-5)},
        ],
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )

    best_auc = -1.0
    for epoch in range(1, cfg["training"].get("joint_epochs", 80) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            he_model=he_model,
            cd34_model=cd34_model,
            classification_weight=1.0,
            lambda_morph=cfg["training"].get("lambda_morph", 0.5),
            lambda_vasc=cfg["training"].get("lambda_vasc", 0.5),
        )
        val_metrics = evaluate(model, val_loader, device)
        print(f"joint_epoch={epoch} train={train_metrics} val={val_metrics}")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({"model": model.state_dict(), "config": cfg}, output_dir / "best.pt")


if __name__ == "__main__":
    main()
