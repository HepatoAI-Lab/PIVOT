from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


MRI_SEQUENCES = ("T1WI", "T2WI", "DWI", "ADC", "AP", "PVP", "DP")


def _split_paths(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        return [str(v) for v in json.loads(text)]
    return [item for item in text.split(";") if item]


def load_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("tensor", "array", "volume", "embedding", "embeddings", "last_layer_embed"):
                if key in obj:
                    obj = obj[key]
                    break
        return torch.as_tensor(obj).float()
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if path.suffix == ".npz":
        data = np.load(path)
        key = "volume" if "volume" in data else list(data.keys())[0]
        return torch.from_numpy(data[key]).float()
    raise ValueError(f"Unsupported tensor file: {path}")


def load_slide_embeddings(paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    if not paths:
        return torch.empty(0, 768), torch.empty(0, dtype=torch.bool)
    tensors = [load_tensor(path).view(-1) for path in paths]
    max_dim = max(t.numel() for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.numel() < max_dim:
            tensor = torch.nn.functional.pad(tensor, (0, max_dim - tensor.numel()))
        padded.append(tensor)
    return torch.stack(padded, dim=0), torch.ones(len(padded), dtype=torch.bool)


class PIVOTCaseDataset(Dataset):
    """Patient-level PIVOT dataset backed by a CSV manifest.

    Required columns:
        patient_id, label, split, T1WI, T2WI, DWI, ADC, AP, PVP, DP

    Optional columns:
        he_slide_embeddings, cd34_slide_embeddings

    Slide embedding columns may be JSON lists or semicolon-separated file paths.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        sequences: tuple[str, ...] = MRI_SEQUENCES,
        require_all_sequences: bool = True,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)
        self.df = self.df[self.df["split"].astype(str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split={split!r} in {manifest_csv}")
        self.sequences = tuple(sequences)
        self.require_all_sequences = require_all_sequences

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        volumes = []
        sequence_mask = []
        for seq in self.sequences:
            value = row.get(seq, "")
            paths = _split_paths(value)
            if not paths:
                if self.require_all_sequences:
                    raise ValueError(f"Missing required sequence {seq} for patient {row['patient_id']}")
                volumes.append(None)
                sequence_mask.append(False)
                continue
            volume = load_tensor(paths[0])
            if volume.dim() == 3:
                volume = volume.unsqueeze(0)
            volumes.append(volume)
            sequence_mask.append(True)

        valid_volumes = [v for v in volumes if v is not None]
        if not valid_volumes:
            raise ValueError(f"No MRI volumes available for patient {row['patient_id']}")
        template = valid_volumes[0]
        volumes = [v if v is not None else torch.zeros_like(template) for v in volumes]
        mri = torch.stack(volumes, dim=0)

        he_paths = _split_paths(row.get("he_slide_embeddings", ""))
        cd34_paths = _split_paths(row.get("cd34_slide_embeddings", ""))
        he_slides, he_mask = load_slide_embeddings(he_paths)
        cd34_slides, cd34_mask = load_slide_embeddings(cd34_paths)

        return {
            "patient_id": str(row["patient_id"]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "mri": mri,
            "sequence_mask": torch.tensor(sequence_mask, dtype=torch.bool),
            "he_slides": he_slides,
            "he_slide_mask": he_mask,
            "cd34_slides": cd34_slides,
            "cd34_slide_mask": cd34_mask,
        }


def _pad_bag(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    dim = max((t.shape[-1] if t.numel() else 768) for t in tensors)
    max_items = max(t.shape[0] for t in tensors)
    out = torch.zeros(len(tensors), max_items, dim)
    mask = torch.zeros(len(tensors), max_items, dtype=torch.bool)
    for i, tensor in enumerate(tensors):
        if tensor.numel() == 0:
            continue
        if tensor.shape[-1] < dim:
            tensor = torch.nn.functional.pad(tensor, (0, dim - tensor.shape[-1]))
        out[i, : tensor.shape[0]] = tensor
        mask[i, : tensor.shape[0]] = True
    return out, mask


def pivot_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    he_slides, he_mask = _pad_bag([item["he_slides"] for item in batch])
    cd34_slides, cd34_mask = _pad_bag([item["cd34_slides"] for item in batch])
    return {
        "patient_id": [item["patient_id"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "mri": torch.stack([item["mri"] for item in batch]),
        "sequence_mask": torch.stack([item["sequence_mask"] for item in batch]),
        "he_slides": he_slides,
        "he_slide_mask": he_mask,
        "cd34_slides": cd34_slides,
        "cd34_slide_mask": cd34_mask,
    }
