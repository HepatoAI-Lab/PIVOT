#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slides-csv", required=True, help="CSV with slide_id, patient_id, stain, slide_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    import pandas as pd
    import torch

    from pivot.utils.config import load_config, resolve_config_path

    cfg = load_config(args.config)
    gigapath_repo = Path(resolve_config_path(cfg, cfg["paths"]["gigapath_repo"]))
    if str(gigapath_repo) not in sys.path:
        sys.path.insert(0, str(gigapath_repo))

    from gigapath.pipeline import (
        load_tile_slide_encoder,
        run_inference_with_slide_encoder,
        run_inference_with_tile_encoder,
        tile_one_slide,
    )

    output_dir = Path(args.output_dir)
    tile_root = output_dir / "tiles"
    embed_root = output_dir / "slide_embeddings"
    tile_root.mkdir(parents=True, exist_ok=True)
    embed_root.mkdir(parents=True, exist_ok=True)

    tile_encoder, slide_encoder = load_tile_slide_encoder(
        local_tile_encoder_path=resolve_config_path(cfg, cfg["paths"].get("gigapath_tile_checkpoint", "")) or "",
        local_slide_encoder_path=resolve_config_path(cfg, cfg["paths"].get("gigapath_slide_checkpoint", "")) or "",
        global_pool=False,
    )
    slides = pd.read_csv(args.slides_csv)
    records = []
    for row in slides.itertuples(index=False):
        slide_id = str(row.slide_id)
        patient_id = str(row.patient_id)
        stain = str(row.stain).lower()
        slide_path = str(row.slide_path)
        slide_tile_dir = tile_root / slide_id
        tile_one_slide(slide_file=slide_path, save_dir=str(slide_tile_dir), level=args.level, tile_size=args.tile_size)

        dataset_csv = next(slide_tile_dir.glob("output/**/dataset.csv"))
        tile_df = pd.read_csv(dataset_csv)
        tile_paths = tile_df["image"].tolist() if "image" in tile_df.columns else tile_df.iloc[:, 0].tolist()
        tile_outputs = run_inference_with_tile_encoder(tile_paths, tile_encoder, batch_size=args.batch_size)
        slide_outputs = run_inference_with_slide_encoder(
            tile_outputs["tile_embeds"],
            tile_outputs["coords"],
            slide_encoder,
        )
        slide_embedding = slide_outputs["last_layer_embed"].squeeze(0).float().cpu()
        save_path = embed_root / f"{patient_id}_{stain}_{slide_id}.pt"
        torch.save(
            {
                "patient_id": patient_id,
                "stain": stain,
                "slide_id": slide_id,
                "embedding": slide_embedding,
            },
            save_path,
        )
        records.append(
            {
                "patient_id": patient_id,
                "stain": stain,
                "slide_id": slide_id,
                "embedding_path": str(save_path),
            }
        )
    pd.DataFrame(records).to_csv(output_dir / "slide_embedding_manifest.csv", index=False)


if __name__ == "__main__":
    main()
