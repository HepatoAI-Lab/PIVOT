# PIVOT Pipeline

This document summarizes the reproducible workflow implemented in this repository.

## 1. MRI Segmentation

`scripts/run_tumor_segmentation.py` wraps a local nnU-Net liver tumor segmentation model. The script accepts a CSV containing `patient_id` and the reference phase image path, and writes one tumor mask path per patient.

## 2. MRI Preprocessing

`scripts/preprocess_mri.py` resamples the seven MRI sequences, registers them to the PVP phase, normalizes intensity, and exports tumor-centered 3D tensors. The output is an MRI manifest with one column per sequence.

## 3. WSI Processing

`scripts/tile_wsi.py` uses the local Prov-GigaPath preprocessing and encoder implementation. H&E and CD34 slides are tiled, encoded, and summarized as slide-level embeddings.

## 4. Patient-Level Manifest

`scripts/build_pivot_manifest.py` combines MRI tensors, labels, cohort splits, and slide embeddings into `data/pivot_manifest.csv`.

## 5. Model Training

`scripts/train_pathology.py` trains patient-level H&E and CD34 histopathologic reference models from slide embeddings.

`scripts/train_pivot.py` trains the MRI PIVOT model using staged pathology-guided alignment and VETC classification.

## 6. Inference and Evaluation

`scripts/infer_pivot.py` writes patient-level PIVOT scores from pretreatment MRI.

`scripts/evaluate_predictions.py` computes diagnostic summaries without manuscript-specific plotting code.
