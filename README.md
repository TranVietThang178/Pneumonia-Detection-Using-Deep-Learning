# Deep Learning-Based Pneumonia Classification from Chest Radiographs with Grad-CAM Interpretability

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC--BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)

**Author:** Viet Thang Tran  
**Institution:** Atlantic Technological University, MSc Data Science  
**Module:** Deep Learning  

---

## Overview

This project develops and evaluates three deep learning architectures - **ResNet-50**, **DenseNet-121**, and **EfficientNet-B0** - for automated pneumonia detection from chest X-ray (CXR) images. Transfer learning is applied to all three models, and **Grad-CAM** visualisations are used to interpret model decisions by highlighting the image regions most influential to each prediction.

All three models achieve strong diagnostic performance with AUC-ROC exceeding 99%, with **DenseNet-121 achieving the best Recall at 97.0%**, making it the most clinically reliable model for this task.

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| ResNet-50 | 96.4% | 99.0% | 95.9% | 97.5% | **99.7%** |
| **DenseNet-121** | **96.9%** | 98.7% | **97.0%** | **97.9%** | 99.6% |
| EfficientNet-B0 | 95.8% | **99.2%** | 95.0% | 97.1% | 99.6% |

---

## Repository Structure

```
Pneumonia-Detection-Using-Deep-Learning/
│
├── data/
│   ├── raw/                    # Original Kaggle dataset (not included)
│   └── processed/              # Re-split dataset 70/15/15 (not included)
│
├── notebooks/
│   ├── 00_data.ipynb           # Kaggle dataset download
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb  # Merge and stratified re-split
│   ├── 03_training.ipynb       # Model training (ResNet-50, DenseNet-121, EfficientNet-B0)
│   └── 04_evaluation_gradcam.ipynb  # Evaluation metrics and Grad-CAM visualisations
│
├── src/
│   ├── dataset.py              # Custom dataset class and dataloaders
│   ├── models.py               # Model definitions and transfer learning setup
│   ├── train.py                # Training loop with early stopping
│   ├── evaluate.py             # Evaluation metrics (Accuracy, Precision, Recall, F1, AUC)
│   └── gradcam.py              # GradCAM implementation
│
├── outputs/
│   ├── models/                 # Saved model checkpoints (.pth files)
│   └── figures/                # Generated plots and visualisations
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Dataset

The dataset used is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle, originally collected by Kermany et al. (2018).

- **Total images:** 5,856 chest X-rays (JPEG format)
- **Classes:** Normal (1,583) | Pneumonia (4,273)
- **License:** CC-BY 4.0

> The dataset is not included in this repository. Follow the setup instructions below to download it.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/TranVietThang178/Pneumonia-Detection-Using-Deep-Learning.git
cd Pneumonia-Detection-Using-Deep-Learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** This project was developed with CUDA 13.2. If you are running on CPU only, install the standard PyTorch release instead:
> ```bash
> pip install torch torchvision
> ```

### 3. Download the dataset

Run `00_data.ipynb` to automatically download the dataset via the Kaggle library.

If the download fails, try upgrading the Kaggle library first:
```bash
pip install --upgrade kaggle
```

Alternatively, download the dataset manually from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract the zip directly into the `data/raw/` directory.

---

## How to Run

Run the notebooks **in order**:

| Step | Notebook | Description |
|---|---|---|
| 1 | `00_data.ipynb` | Downloads the dataset from Kaggle |
| 2 | `01_EDA.ipynb` | Explores class distribution, image sizes, pixel statistics |
| 3 | `02_preprocessing.ipynb` | Merges original splits and re-splits 70/15/15 stratified |
| 4 | `03_training.ipynb` | Trains ResNet-50, DenseNet-121, and EfficientNet-B0 |
| 5 | `04_evaluation_gradcam.ipynb` | Evaluates all models and generates Grad-CAM visualisations |

> Trained model checkpoints will be saved to `outputs/models/` and all figures to `outputs/figures/`.

---

## Key Implementation Details

- **Loss function:** BCEWithLogitsLoss with class weights to handle class imbalance
- **Optimiser:** AdamW with weight decay of 1e-4
- **Learning rates:** Differential - 1e-4 for pretrained layers, 1e-3 for classifier heads
- **Scheduler:** ReduceLROnPlateau (factor=0.1, patience=3)
- **Early stopping:** Patience of 7 epochs
- **Batch size:** 32
- **Augmentation:** Brightness/contrast jitter, random rotation (±10°)
- **Grad-CAM target layers:** `layer4` (ResNet-50), `denseblock4` (DenseNet-121), `features[8]` (EfficientNet-B0)

---

## Dependencies

Key packages (see `requirements.txt` for full list):

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0 (CUDA 13.2) |
| torchvision | 0.27.0 |
| numpy | 2.4.4 |
| matplotlib | 3.10.8 |
| scikit-learn | 1.8.0 |
| seaborn | 0.13.2 |
| Pillow | 12.1.1 |
| kaggle | 2.0.0 |

---

## References

- Kermany, D. S., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122–1131.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Huang, G., et al. (2017). Densely connected convolutional networks. *CVPR*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Rajpurkar, P., et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv:1711.05225*.
