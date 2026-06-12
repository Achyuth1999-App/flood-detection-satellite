

# Flood Detection from Satellite Images 

**Student:** Madhavapeddy Achyuth Reddy
**Programme:** MSc Data Science — University of Europe for Applied Sciences
**Supervisor:** Shan Faiz
**Course:** Machine Learning
**Phase:** Phase 2 — Proposal and Code Implementation

## Overview

Binary CNN-based image classification pipeline to detect **flooded vs
non-flooded** regions in satellite imagery. Two models are trained and
compared:

1. **Custom CNN** — 4-block convolutional network built from scratch
2. **MobileNetV2** — ImageNet pre-trained transfer learning model
   (frozen-base feature extraction + fine-tuning of top 30 layers)

## Dataset

- **Name:** Flood Area Segmentation
- **Source:** Kaggle — https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation
- **Classes:** `flooded`, `non_flooded`

If no dataset is found at the standard Kaggle input path, the script
automatically generates a realistic synthetic satellite-image dataset
(150 flooded + 150 non-flooded) so the full pipeline still runs end to end.

## Repository Structure

```
.
├── flood_detection_cnn.py   # complete pipeline (single script)
├── requirements.txt
├── figures/                 # output figures generated on run
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python flood_detection_cnn.py
```

To use the real dataset, download it from Kaggle and place it so that:

```
/kaggle/input/flood-area-segmentation/Image/flooded/
/kaggle/input/flood-area-segmentation/Image/non_flooded/
```

exist (or edit `download_or_generate_dataset()` to point at your local path).

## Pipeline Stages

1. Dataset loading (real or auto-generated)
2. EDA — class distribution, sample images
3. Preprocessing — normalisation, stratified 70/15/15 train/val/test split
4. Data augmentation (rotation, flip, zoom, brightness, shear)
5. Custom CNN — build, train, evaluate
6. MobileNetV2 transfer learning — feature extraction + fine-tuning, evaluate
7. Model comparison (accuracy, precision, recall, F1, ROC AUC)
8. Grad-CAM interpretability visualisation
9. Error analysis on misclassified test images
10. Model saving (`.keras`) and figure export to `figures/`

## Output Figures

| File | Description |
|---|---|
| fig01_class_distribution.png | Class balance bar/pie chart |
| fig02_sample_images.png | Sample flooded/non-flooded images |
| fig03_augmentation.png | Augmentation examples |
| fig04_cnn_curves.png | Custom CNN accuracy/loss curves |
| fig05_mobilenet_curves.png | MobileNetV2 accuracy/loss curves |
| fig06_model_comparison.png | Metric comparison bar chart |
| fig07_roc_comparison.png | ROC curve comparison |
| fig08_gradcam.png | Grad-CAM visualisations |
| fig09_error_analysis.png | Misclassified test samples |
| fig_cm_*.png | Confusion matrices |
| fig_roc_*.png | Individual ROC curves |

## References

1. Sandler et al. (2018). MobileNetV2. CVPR.
2. Selvaraju et al. (2017). Grad-CAM. ICCV.
3. LeCun, Bengio & Hinton (2015). Deep learning. Nature.
4. Goodfellow et al. (2016). Deep Learning. MIT Press.
