# PIVOT

Official implementation of **PIVOT**, a pathology-informed foundation model framework for preoperative prediction of vessels encapsulating tumor clusters (VETC) in hepatocellular carcinoma using multiparametric MRI.

PIVOT uses paired postoperative H&E whole-slide images and CD34 immunohistochemistry during model development to guide MRI representation learning. The clinical prediction pathway uses pretreatment MRI only.

## 🔎 Overview

PIVOT is designed for noninvasive assessment of VETC status before surgery. The framework contains three components.

1. **Histopathologic reference learning** from H&E and CD34 whole-slide images using Prov-GigaPath slide representations.
2. **Multiparametric MRI encoding** using a shared Triad-SwinB 3D MRI foundation backbone with sequence-specific adapters.
3. **MRI-pathology alignment** between the MRI-derived VETC representation and fixed H&E-derived morphologic and CD34-derived vascular reference embeddings.

The final PIVOT score is generated from pretreatment MRI and can be used for VETC risk estimation, model comparison, calibration analysis, decision-curve analysis, and downstream recurrence-risk assessment.

## 🧠 Method

For each patient, seven registered MRI sequences are used as model input.

```text
T1WI, T2WI, DWI, ADC, AP, PVP, DP
```

Each sequence is encoded independently with the same Triad-SwinB encoder. Sequence-level embeddings are adapted with lightweight sequence-specific adapters, combined with sequence-type embeddings, and processed by a sequence-token transformer. A learnable VETC classification token is used for MRI-based prediction and for alignment with histopathologic reference embeddings during training.

For pathology, slide-level embeddings from all available H&E or CD34 slides of the same patient are aggregated with a patient-level attention module. H&E provides a morphologic reference embedding, whereas CD34 provides a vascular reference embedding and defines the VETC endpoint.

The MRI model is optimized with VETC classification loss and two cosine-distance alignment losses.

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{cls}}
+
\lambda_{\mathrm{morph}}\mathcal{L}_{\mathrm{morph}}
+
\lambda_{\mathrm{vasc}}\mathcal{L}_{\mathrm{vasc}}
$$

## ⚙️ Installation

Create an environment with PyTorch, MONAI, Triad dependencies, and Prov-GigaPath dependencies. Then install PIVOT in editable mode.

```bash
cd PIVOT
pip install -r requirements.txt
pip install -e .
```

The default configuration expects the following local foundation-model repositories and weights.

```text
../Triad
../Triad/weights/Triad-SwinB-SimMIM.pth
../prov-gigapath
../prov-gigapath/hf_weights/
```

These paths are relative to [configs/pivot_default.yaml](configs/pivot_default.yaml) and can be changed for a different local layout.

## 🧾 Data Preparation

PIVOT uses a patient-level CSV manifest.

Required columns:

```text
patient_id,split,label,T1WI,T2WI,DWI,ADC,AP,PVP,DP,he_slide_embeddings,cd34_slide_embeddings
```

MRI columns should point to tumor-centered, registered 3D MRI tensors saved as `.pt`, `.npy`, or `.npz`. Pathology columns should point to precomputed slide-level embeddings from Prov-GigaPath. Multiple slide embeddings can be provided as semicolon-separated paths.

Example files are provided in:

```text
data/pivot_manifest.example.csv
data/slides.example.csv
```

To extract slide-level embeddings from raw WSIs:

```bash
python scripts/extract_gigapath_slide_embeddings.py \
  --config configs/pivot_default.yaml \
  --slides-csv data/slides.csv \
  --output-dir outputs/gigapath_embeddings
```

The slide CSV should contain:

```text
slide_id,patient_id,stain,slide_path
```

## 🚀 Training

Train H&E and CD34 histopathologic reference models.

```bash
python scripts/train_pathology.py \
  --config configs/pivot_default.yaml \
  --stain he

python scripts/train_pathology.py \
  --config configs/pivot_default.yaml \
  --stain cd34
```

Train the MRI PIVOT model with fixed histopathologic references.

```bash
python scripts/train_pivot.py \
  --config configs/pivot_default.yaml \
  --he-checkpoint outputs/pathology_he/best.pt \
  --cd34-checkpoint outputs/pathology_cd34/best.pt
```

The MRI training script follows a staged procedure.

1. H&E and CD34 reference models are fixed.
2. Newly introduced MRI adaptation and alignment modules are trained with alignment losses.
3. VETC classification loss is added, and the final Triad-SwinB stage is fine-tuned with a lower learning rate.

## 🧪 Inference

Run MRI-only inference with a trained PIVOT checkpoint.

```bash
python scripts/infer_pivot.py \
  --config configs/pivot_default.yaml \
  --checkpoint outputs/pivot_mri/best.pt \
  --split test \
  --output outputs/pivot_scores_test.csv
```

The output file contains patient identifiers and PIVOT scores.

```text
patient_id,pivot_score
```

## 📁 Repository Structure

```text
configs/
  pivot_default.yaml              Default experiment configuration
data/
  pivot_manifest.example.csv      Example patient-level manifest
  slides.example.csv              Example WSI slide manifest
pivot/
  data/                           Dataset and collate functions
  models/                         MRI encoder, pathology reference model, PIVOT model
  training/                       Losses, metrics, and training utilities
  utils/                          Configuration utilities
scripts/
  extract_gigapath_slide_embeddings.py
  train_pathology.py
  train_pivot.py
  infer_pivot.py
```

## 📚 Citation

If you use this code, please cite the PIVOT manuscript.

```bibtex
@article{pivot_vetc,
  title   = {Pathology-Informed Foundation Model for Preoperative MRI Prediction of Vessels Encapsulating Tumor Clusters in Hepatocellular Carcinoma},
  author  = {PIVOT Investigators},
  journal = {Manuscript in preparation},
  year    = {2026}
}
```

## 🙏 Acknowledgements

PIVOT builds on publicly available foundation-model resources for 3D MRI and whole-slide pathology representation learning, including Triad and Prov-GigaPath. Model weights are not redistributed in this repository and should be obtained from their original sources or local institutional mirrors.

## 📄 License

The PIVOT source code is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Pretrained weights, third-party foundation models, and datasets are not redistributed in this repository and remain subject to their original licenses and access terms.
