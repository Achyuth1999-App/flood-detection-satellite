# Instructions to Run the Code

## Option 1: Run on Kaggle (Recommended)

1. Go to **kaggle.com** -> click **Create -> New Notebook**
2. Copy the entire contents of `flood_detection_cnn.py` and paste it into a code cell
3. Enable **GPU**: click **Settings -> Accelerator -> GPU T4 x2**
4. Enable **Internet**: click **Settings -> Internet -> ON** (needed to download MobileNetV2 weights)
5. Click **Run All**
6. Wait ~10-15 minutes
7. All figures will be saved automatically to the `figures/` folder, visible in the **Output** panel on the right

## Option 2: Run on Google Colab

1. Go to **colab.research.google.com** -> **New Notebook**
2. Click **Runtime -> Change runtime type -> T4 GPU -> Save**
3. In the first cell, run:
   ```python
   !pip install -q opencv-python-headless
   ```
4. In a new cell, paste the entire contents of `flood_detection_cnn.py`
5. Click **Runtime -> Run all**
6. Figures will be saved to `figures/` in the Colab file system (visible in the left sidebar **Files** tab)

## Option 3: Run Locally

```bash
git clone <your-repo-url>
cd flood-detection-cnn
pip install -r requirements.txt
python flood_detection_cnn.py
```

**Requirements:** Python 3.10+, ~2GB free disk space, GPU optional (CPU works but slower, ~20-30 min)

## Notes

- **No dataset attachment needed** - the script auto-detects a Kaggle dataset if present, otherwise generates a synthetic 300-image satellite dataset automatically
- To use the **real dataset**, download from `kaggle.com/datasets/faizalkarim/flood-area-segmentation` and place it at `/kaggle/input/flood-area-segmentation/Image/` with `flooded/` and `non_flooded/` subfolders
- Output: trained models (`.keras` files) + 9 figures (class distribution, training curves, confusion matrices, ROC curves, Grad-CAM, error analysis) saved to `figures/`
