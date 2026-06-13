#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile WSIs and extract Prov-GigaPath slide embeddings.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--slides-csv", required=True, help="CSV with slide_id, patient_id, stain, slide_path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    import pandas as pd

    from pivot.preprocessing.wsi import build_slide_embedding_manifest, extract_slide_embedding_with_gigapath, tile_slide_with_gigapath
    from pivot.utils.config import load_config, resolve_config_path

    cfg = load_config(args.config)
    gigapath_repo = resolve_config_path(cfg, cfg["paths"]["gigapath_repo"])
    tile_checkpoint = resolve_config_path(cfg, cfg["paths"].get("gigapath_tile_checkpoint", ""))
    slide_checkpoint = resolve_config_path(cfg, cfg["paths"].get("gigapath_slide_checkpoint", ""))

    slides = pd.read_csv(args.slides_csv)
    required = {"slide_id", "patient_id", "stain", "slide_path"}
    missing = required.difference(slides.columns)
    if missing:
        raise ValueError(f"slides CSV is missing columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    records = []
    for row in slides.itertuples(index=False):
        slide_id = str(row.slide_id)
        patient_id = str(row.patient_id)
        stain = str(row.stain).lower()
        tile_dir = output_dir / "tiles" / slide_id
        dataset_csv = tile_slide_with_gigapath(
            row.slide_path,
            tile_dir,
            gigapath_repo=gigapath_repo,
            level=args.level,
            tile_size=args.tile_size,
        )
        embedding_path = output_dir / "slide_embeddings" / f"{patient_id}_{stain}_{slide_id}.pt"
        extract_slide_embedding_with_gigapath(
            dataset_csv,
            embedding_path,
            gigapath_repo=gigapath_repo,
            tile_encoder_path=tile_checkpoint,
            slide_encoder_path=slide_checkpoint,
            batch_size=args.batch_size,
        )
        records.append(
            {
                "patient_id": patient_id,
                "stain": stain,
                "slide_id": slide_id,
                "embedding_path": str(embedding_path),
            }
        )
    build_slide_embedding_manifest(records, output_dir / "slide_embedding_manifest.csv")


if __name__ == "__main__":
    main()
