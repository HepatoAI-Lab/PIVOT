#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_epoch(model, loader, optimizer, device, stain: str, train: bool = True):
    import torch
    import torch.nn.functional as F

    from pivot.training.metrics import binary_auc

    model.train(train)
    labels, scores, losses = [], [], []
    for batch in loader:
        slides_key = f"{stain}_slides"
        mask_key = f"{stain}_slide_mask"
        slides = batch[slides_key].to(device)
        mask = batch[mask_key].to(device)
        label = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = model(slides, mask)
        loss = F.binary_cross_entropy_with_logits(out["logit"], label)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        labels.extend(label.detach().cpu().tolist())
        scores.extend(torch.sigmoid(out["logit"]).detach().cpu().tolist())
    return {"loss": sum(losses) / max(1, len(losses)), "auc": binary_auc(labels, scores)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stain", choices=["he", "cd34"], required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from pivot.data import PIVOTCaseDataset, pivot_collate
    from pivot.models import PathologyReferenceModel
    from pivot.utils.config import load_config

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg["output_dir"]) / f"pathology_{args.stain}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PIVOTCaseDataset(cfg["data"]["manifest_csv"], split=cfg["data"].get("train_split", "train"))
    val_ds = PIVOTCaseDataset(cfg["data"]["manifest_csv"], split=cfg["data"].get("val_split", "val"))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"].get("pathology_batch_size", 16),
        shuffle=True,
        collate_fn=pivot_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"].get("pathology_batch_size", 16),
        shuffle=False,
        collate_fn=pivot_collate,
    )

    model = PathologyReferenceModel(
        slide_embedding_dim=cfg["pathology"].get("slide_embedding_dim", 768),
        reference_dim=cfg["model"].get("reference_dim", 768),
        hidden_dim=cfg["pathology"].get("patient_hidden_dim", 256),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"].get("pathology_lr", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )

    best_auc = -1.0
    epochs = cfg["training"].get("pathology_epochs", 50)
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.stain, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, args.stain, train=False)
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({"model": model.state_dict(), "config": cfg, "stain": args.stain}, output_dir / "best.pt")


if __name__ == "__main__":
    main()
