#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pivot.data import PIVOTCaseDataset, pivot_collate
from pivot.models import PIVOTMRIEncoder, PIVOTModel
from pivot.utils.config import load_config, resolve_config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    ds = PIVOTCaseDataset(cfg["data"]["manifest_csv"], split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pivot_collate)

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
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["mri"].to(device), batch["sequence_mask"].to(device))
            score = torch.sigmoid(out["logit"]).detach().cpu().item()
            rows.append({"patient_id": batch["patient_id"][0], "pivot_score": score})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "pivot_score"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
