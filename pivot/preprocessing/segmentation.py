from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_nnunet_segmentation(
    image_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path = "../LiverTumorSegmentor",
    dataset_id: str = "666",
    configuration: str = "3d_fullres",
    fold: str = "0",
    device: str = "cuda",
    disable_tta: bool = True,
) -> Path:
    """Run the local LiverTumorSegmentor nnU-Net model on one NIfTI image."""

    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    case_name = image_path.name.replace(".nii.gz", "").replace(".nii", "")

    env = os.environ.copy()
    env["nnUNet_raw"] = str(model_dir / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(model_dir / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(model_dir / "nnUNet_results")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        nnunet_input = tmp_dir / f"{case_name}_0000.nii.gz"
        shutil.copy2(image_path, nnunet_input)
        cmd = [
            "nnUNetv2_predict",
            "-i",
            str(tmp_dir),
            "-o",
            str(output_dir),
            "-d",
            str(dataset_id),
            "-c",
            configuration,
            "-f",
            str(fold),
            "-device",
            device,
        ]
        if disable_tta:
            cmd.append("--disable_tta")
        subprocess.run(cmd, env=env, check=True)
    return output_dir / f"{case_name}.nii.gz"
