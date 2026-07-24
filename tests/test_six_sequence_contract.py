from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from pivot.data import MRI_SEQUENCES, PIVOTCaseDataset
from pivot.models import mri_triad
from pivot.preprocessing.manifest import (
    DEFAULT_SEQUENCES as MANIFEST_SEQUENCES,
    build_patient_manifest,
)


EXPECTED_SEQUENCES = ("T1WI", "T2WI", "DWI", "AP", "PVP", "DP")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_and_python_defaults_use_six_sequences() -> None:
    with open(REPOSITORY_ROOT / "configs" / "pivot_default.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    assert tuple(config["data"]["sequences"]) == EXPECTED_SEQUENCES
    assert MRI_SEQUENCES == EXPECTED_SEQUENCES
    assert mri_triad.DEFAULT_SEQUENCES == EXPECTED_SEQUENCES
    assert MANIFEST_SEQUENCES == EXPECTED_SEQUENCES


def test_example_csv_headers_use_six_sequences() -> None:
    with open(REPOSITORY_ROOT / "data" / "mri_cases.example.csv", newline="", encoding="utf-8") as stream:
        mri_header = tuple(next(csv.reader(stream)))
    with open(
        REPOSITORY_ROOT / "data" / "pivot_manifest.example.csv",
        newline="",
        encoding="utf-8",
    ) as stream:
        manifest_header = tuple(next(csv.reader(stream)))

    assert mri_header == ("patient_id", "mask_path", *EXPECTED_SEQUENCES)
    assert manifest_header == (
        "patient_id",
        "split",
        "label",
        *EXPECTED_SEQUENCES,
        "he_slide_embeddings",
        "cd34_slide_embeddings",
    )


def test_dataset_stacks_six_single_channel_volumes(tmp_path: Path) -> None:
    row: dict[str, object] = {"patient_id": "P001", "split": "test", "label": 1}
    for index, sequence in enumerate(EXPECTED_SEQUENCES):
        volume_path = tmp_path / f"{sequence}.npy"
        np.save(volume_path, np.full((2, 3, 4), index, dtype=np.float32))
        row[sequence] = str(volume_path)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([row]).to_csv(manifest_path, index=False)

    item = PIVOTCaseDataset(manifest_path, split="test")[0]

    assert item["mri"].shape == (len(EXPECTED_SEQUENCES), 1, 2, 3, 4)
    assert item["sequence_mask"].shape == (len(EXPECTED_SEQUENCES),)
    assert item["sequence_mask"].all()


def test_manifest_builder_requires_and_preserves_six_sequence_columns(tmp_path: Path) -> None:
    mri_path = tmp_path / "mri.csv"
    labels_path = tmp_path / "labels.csv"
    output_path = tmp_path / "pivot_manifest.csv"
    pd.DataFrame(
        [
            {
                "patient_id": "P001",
                **{sequence: f"/data/P001/{sequence}.pt" for sequence in EXPECTED_SEQUENCES},
            }
        ]
    ).to_csv(mri_path, index=False)
    pd.DataFrame([{"patient_id": "P001", "split": "train", "label": 1}]).to_csv(
        labels_path,
        index=False,
    )

    build_patient_manifest(mri_path, labels_path, None, output_path)
    built = pd.read_csv(output_path)

    assert tuple(column for column in built.columns if column in EXPECTED_SEQUENCES) == EXPECTED_SEQUENCES
    assert built.loc[0, list(EXPECTED_SEQUENCES)].notna().all()


class _MeanBackbone(nn.Module):
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return volume.mean(dim=(2, 3, 4))


def test_encoder_emits_one_token_per_configured_sequence(monkeypatch) -> None:
    monkeypatch.setattr(
        mri_triad,
        "_load_triad_backbone",
        lambda *_args, **_kwargs: _MeanBackbone(),
    )
    encoder = mri_triad.PIVOTMRIEncoder(
        triad_checkpoint=None,
        sequences=EXPECTED_SEQUENCES,
        model_dim=8,
        adapter_bottleneck_dim=4,
        transformer_layers=1,
        transformer_heads=2,
        dropout=0.0,
        use_checkpoint=False,
    )
    volumes = torch.randn(2, len(EXPECTED_SEQUENCES), 1, 2, 3, 4)

    outputs = encoder(volumes)

    assert encoder.sequence_type_embeddings.shape == (len(EXPECTED_SEQUENCES), 8)
    assert outputs["sequence_tokens"].shape == (2, len(EXPECTED_SEQUENCES), 8)
    assert outputs["sequence_mask"].shape == (2, len(EXPECTED_SEQUENCES))
