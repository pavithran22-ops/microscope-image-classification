# Microscope Image Dataset Guide

## 1. Overview
This directory contains the dataset used to evaluate and compare the **Fully Connected Network (FCN)** and the **Convolutional Neural Network (CNN)** for cellular microscope image classification.

Microscope images have unique physical characteristics:
- **Circular or elliptical cellular boundaries** (cell membrane)
- **Distinct internal organelles** (nuclei, vacuoles, chromatin texture)
- **Pathological indicators** (intracellular parasite inclusions, hyper-segmented nuclei, or granules)
- **Staining artifacts and background illumination gradients**

---

## 2. Dataset Classes (Binary Classification)

The dataset is structured around two distinct cellular classes:

| Class ID | Class Label | Description | Visual Characteristics |
| :--- | :--- | :--- | :--- |
| **0** | `Normal_Cell` | Healthy cell (e.g., Uninfected red blood cell or normal lymphocyte) | Smooth outer membrane, uniform central pallor, homogenous cytoplasm, no intracellular abnormalities. |
| **1** | `Infected_Cell` | Pathological / Infected cell (e.g., *Plasmodium*-infected erythrocyte / abnormal blast) | Irregular boundary, dark nuclear chromatin clumps, punctate parasite ring-form inclusions, or cytoplasmic granularity. |

---

## 3. Dataset Options

### Option A: Built-in Synthetic Cell Microscope Generator (Default & Recommended)
To ensure the project works **instantly out-of-the-box** without downloading multi-gigabyte archives or requiring API tokens, `src/data_loader.py` includes a deterministic synthetic cell microscopy image generator.

- **Image Dimensions**: $64 \times 64$ pixels (grayscale, shape: `(64, 64, 1)`).
- **Physical Modeling**:
  - Cell membranes generated using parametric ellipses with slight morphological deformations.
  - Normal cells feature smooth circular membranes with realistic central pallor and subtle intracellular texture.
  - Infected cells contain distinct high-contrast localized inclusions (representing malaria ring-forms or dense nuclear granules).
  - Background includes Gaussian optical blur and microscopic field illumination gradients.
- **Reproducibility**: Uses a fixed random seed (`seed=42`) ensuring identical train/val/test splits across runs.

### Option B: Using Public Real-World Microscopy Datasets
If you wish to test with real laboratory images, you can use either of the following publicly available benchmarks:

1. **NIH Malaria Cell Images Dataset**:
   - **Source**: National Institutes of Health (NIH) / National Library of Medicine
   - **URL**: [https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
   - **Description**: 27,558 Giemsa-stained thin blood smear images categorized into *Parasitized* and *Uninfected*.
   - **Usage**: Download the zip file and extract into:
     ```
     data/
     ├── Parasitized/
     └── Uninfected/
     ```

2. **BloodMNIST / PathMNIST (MedMNIST v2)**:
   - **Source**: MedMNIST v2 Benchmark (`pip install medmnist`)
   - **URL**: [https://medmnist.com/](https://medmnist.com/)
   - **Description**: Standardized $28 \times 28$ and $64 \times 64$ biomedical images for educational and research comparisons.

---

## 4. Preprocessing Pipeline
Both FCN and CNN ingest images processed identically:
1. **Grayscale Conversion**: Eliminates color bias while preserving structural morphology.
2. **Resizing**: Standardized to $64 \times 64$ pixels.
3. **Normalization**: Pixel intensities scaled from $[0, 255]$ to $[0.0, 1.0]$.
4. **Data Splitting**: Stratified split into 70% Train, 15% Validation, and 15% Test.
