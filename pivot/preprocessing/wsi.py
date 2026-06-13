from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


def _add_repo_to_path(repo: str | Path) -> Path:
    repo = Path(repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def tile_slide_with_gigapath(
    slide_path: str | Path,
    output_dir: str | Path,
    gigapath_repo: str | Path = "../prov-gigapath",
    level: int = 0,
    tile_size: int = 256,
) -> Path:
    """Tile one WSI using the Prov-GigaPath preprocessing implementation."""

    _add_repo_to_path(gigapath_repo)
    try:
        from gigapath.pipeline import tile_one_slide
    except Exception as exc:  # pragma: no cover - depends on local GigaPath deps
        raise ImportError("Could not import Prov-GigaPath tiling utilities.") from exc
    output_dir = Path(output_dir)
    tile_one_slide(slide_file=str(slide_path), save_dir=str(output_dir), level=level, tile_size=tile_size)
    candidates = list(output_dir.glob("output/**/dataset.csv"))
    if not candidates:
        raise FileNotFoundError(f"No tile dataset.csv was generated under {output_dir}.")
    return candidates[0]


def extract_slide_embedding_with_gigapath(
    tile_dataset_csv: str | Path,
    output_path: str | Path,
    gigapath_repo: str | Path = "../prov-gigapath",
    tile_encoder_path: str | Path | None = "../prov-gigapath/hf_weights/pytorch_model.bin",
    slide_encoder_path: str | Path | None = "../prov-gigapath/hf_weights/slide_encoder.pth",
    batch_size: int = 128,
) -> Path:
    """Extract a Prov-GigaPath slide embedding from a generated tile dataset."""

    _add_repo_to_path(gigapath_repo)
    try:
        from gigapath.pipeline import (
            load_tile_slide_encoder,
            run_inference_with_slide_encoder,
            run_inference_with_tile_encoder,
        )
    except Exception as exc:  # pragma: no cover - depends on local GigaPath deps
        raise ImportError("Could not import Prov-GigaPath encoding utilities.") from exc

    tile_dataset_csv = Path(tile_dataset_csv)
    tile_df = pd.read_csv(tile_dataset_csv)
    if "image" not in tile_df.columns:
        raise ValueError(f"{tile_dataset_csv} must contain an image column.")
    tile_root = tile_dataset_csv.parent.parent
    tile_paths = [
        str((tile_root / path).resolve()) if not Path(str(path)).is_absolute() else str(path)
        for path in tile_df["image"].tolist()
    ]
    tile_encoder, slide_encoder = load_tile_slide_encoder(
        local_tile_encoder_path=str(tile_encoder_path or ""),
        local_slide_encoder_path=str(slide_encoder_path or ""),
        global_pool=False,
    )
    tile_outputs = run_inference_with_tile_encoder(tile_paths, tile_encoder, batch_size=batch_size)
    slide_outputs = run_inference_with_slide_encoder(
        tile_outputs["tile_embeds"],
        tile_outputs["coords"],
        slide_encoder,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slide_outputs, output_path)
    return output_path


def build_slide_embedding_manifest(
    records: list[dict[str, str]],
    output_csv: str | Path,
) -> Path:
    """Write a slide-level embedding manifest with patient, stain, and embedding path."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return output_csv
