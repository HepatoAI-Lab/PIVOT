from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MRIPreprocessConfig:
    """Configuration for patient-level multiparametric MRI preparation."""

    target_spacing: tuple[float, float, float] = (1.5, 1.5, 3.0)
    crop_size: tuple[int, int, int] = (96, 160, 160)
    intensity_percentiles: tuple[float, float] = (1.0, 99.0)
    reference_sequence: str = "PVP"
    output_dtype: str = "float32"


def _require_nibabel():
    try:
        import nibabel as nib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("MRI preprocessing requires nibabel. Install it with `pip install nibabel`.") from exc
    return nib


def _require_simpleitk():
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("MRI registration/resampling requires SimpleITK. Install it with `pip install SimpleITK`.") from exc
    return sitk


def load_nifti(path: str | Path) -> tuple[np.ndarray, object]:
    """Load a NIfTI volume as channel-free XYZ data and retain its image header object."""

    nib = _require_nibabel()
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.float32)
    return data, image


def save_numpy_tensor(path: str | Path, array: np.ndarray, key: str = "volume") -> None:
    """Save an MRI tensor in the compact NPZ format expected by PIVOT datasets."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: array.astype(np.float32)})


def normalize_intensity(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    percentiles: tuple[float, float] = (1.0, 99.0),
    eps: float = 1e-6,
) -> np.ndarray:
    """Clip and z-score normalize MRI intensities using foreground or lesion context."""

    data = np.asarray(volume, dtype=np.float32)
    if mask is not None and np.any(mask):
        values = data[np.asarray(mask) > 0]
    else:
        values = data[np.isfinite(data)]
    if values.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo, hi = np.percentile(values, percentiles)
    clipped = np.clip(data, lo, hi)
    mean = float(np.mean(clipped[np.isfinite(clipped)]))
    std = float(np.std(clipped[np.isfinite(clipped)]))
    return ((clipped - mean) / max(std, eps)).astype(np.float32)


def resample_to_spacing(
    image_path: str | Path,
    output_path: str | Path,
    target_spacing: tuple[float, float, float],
    interpolator: str = "linear",
) -> Path:
    """Resample a NIfTI image to target voxel spacing using SimpleITK."""

    sitk = _require_simpleitk()
    image = sitk.ReadImage(str(image_path))
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear if interpolator == "linear" else sitk.sitkNearestNeighbor)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampler.Execute(image), str(output_path))
    return output_path


def register_to_reference(
    moving_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    transform_path: str | Path | None = None,
) -> Path:
    """Rigidly register a moving MRI sequence to the reference phase."""

    sitk = _require_simpleitk()
    fixed = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration.Execute(fixed, moving)
    resampled = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output_path))
    if transform_path:
        transform_path = Path(transform_path)
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(final_transform, str(transform_path))
    return output_path


def crop_around_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    crop_size: tuple[int, int, int],
) -> np.ndarray:
    """Crop a fixed 3D box around the lesion mask centroid with zero padding."""

    data = np.asarray(volume, dtype=np.float32)
    lesion = np.asarray(mask) > 0
    if lesion.shape != data.shape:
        raise ValueError(f"Mask shape {lesion.shape} does not match volume shape {data.shape}.")
    if np.any(lesion):
        center = np.round(np.argwhere(lesion).mean(axis=0)).astype(int)
    else:
        center = np.asarray(data.shape) // 2

    crop = np.zeros(crop_size, dtype=np.float32)
    source_slices = []
    target_slices = []
    for dim, size in enumerate(crop_size):
        start = int(center[dim] - size // 2)
        end = start + size
        src_start = max(start, 0)
        src_end = min(end, data.shape[dim])
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        source_slices.append(slice(src_start, src_end))
        target_slices.append(slice(dst_start, dst_end))
    crop[tuple(target_slices)] = data[tuple(source_slices)]
    return crop


def preprocess_mri_case(
    patient_id: str,
    sequence_paths: Mapping[str, str | Path],
    mask_path: str | Path,
    output_dir: str | Path,
    config: MRIPreprocessConfig = MRIPreprocessConfig(),
) -> dict[str, str]:
    """Prepare registered tumor-centered sequence tensors for one patient."""

    output_dir = Path(output_dir) / str(patient_id)
    resampled_dir = output_dir / "resampled"
    registered_dir = output_dir / "registered"
    tensor_dir = output_dir / "tensors"
    reference = config.reference_sequence
    if reference not in sequence_paths:
        raise ValueError(f"Reference sequence {reference} is missing for patient {patient_id}.")

    resampled_sequences: dict[str, Path] = {}
    for seq, path in sequence_paths.items():
        resampled_sequences[seq] = resample_to_spacing(
            path,
            resampled_dir / f"{seq}.nii.gz",
            config.target_spacing,
            interpolator="linear",
        )
    resampled_mask = resample_to_spacing(
        mask_path,
        resampled_dir / "tumor_mask.nii.gz",
        config.target_spacing,
        interpolator="nearest",
    )

    registered_sequences: dict[str, Path] = {reference: resampled_sequences[reference]}
    for seq, path in resampled_sequences.items():
        if seq == reference:
            continue
        registered_sequences[seq] = register_to_reference(
            path,
            resampled_sequences[reference],
            registered_dir / f"{seq}_to_{reference}.nii.gz",
            transform_path=registered_dir / f"{seq}_to_{reference}.tfm",
        )

    mask, _ = load_nifti(resampled_mask)
    mask = (mask > 0).astype(np.uint8)
    outputs: dict[str, str] = {"patient_id": str(patient_id)}
    for seq, path in registered_sequences.items():
        volume, _ = load_nifti(path)
        volume = normalize_intensity(volume, mask=mask, percentiles=config.intensity_percentiles)
        cropped = crop_around_mask(volume, mask, config.crop_size)
        save_path = tensor_dir / f"{seq}.npz"
        save_numpy_tensor(save_path, cropped.astype(config.output_dtype))
        outputs[seq] = str(save_path)
    return outputs
