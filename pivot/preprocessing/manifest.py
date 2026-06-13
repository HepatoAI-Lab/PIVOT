from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SEQUENCES = ("T1WI", "T2WI", "DWI", "ADC", "AP", "PVP", "DP")


def _join_paths(values: pd.Series) -> str:
    items = [str(v) for v in values.dropna().tolist() if str(v).strip()]
    return ";".join(items)


def build_patient_manifest(
    mri_csv: str | Path,
    labels_csv: str | Path,
    slide_embedding_csv: str | Path | None,
    output_csv: str | Path,
    sequences: tuple[str, ...] = DEFAULT_SEQUENCES,
) -> Path:
    """Build the patient-level CSV consumed by PIVOT training and inference."""

    mri = pd.read_csv(mri_csv)
    labels = pd.read_csv(labels_csv)
    if "patient_id" not in mri.columns or "patient_id" not in labels.columns:
        raise ValueError("mri_csv and labels_csv must contain patient_id.")
    out = labels.merge(mri, on="patient_id", how="left")
    missing_sequence_cols = [seq for seq in sequences if seq not in out.columns]
    if missing_sequence_cols:
        raise ValueError(f"mri_csv is missing sequence columns: {missing_sequence_cols}")

    if slide_embedding_csv:
        slides = pd.read_csv(slide_embedding_csv)
        required = {"patient_id", "stain", "embedding_path"}
        missing = required.difference(slides.columns)
        if missing:
            raise ValueError(f"slide_embedding_csv is missing columns: {sorted(missing)}")
        slides["stain"] = slides["stain"].astype(str).str.lower()
        he = (
            slides[slides["stain"].isin(["he", "h&e"])]
            .groupby("patient_id")["embedding_path"]
            .apply(_join_paths)
            .rename("he_slide_embeddings")
        )
        cd34 = (
            slides[slides["stain"].eq("cd34")]
            .groupby("patient_id")["embedding_path"]
            .apply(_join_paths)
            .rename("cd34_slide_embeddings")
        )
        out = out.merge(he, on="patient_id", how="left").merge(cd34, on="patient_id", how="left")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return output_csv
