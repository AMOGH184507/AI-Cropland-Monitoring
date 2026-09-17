# Stage 6 AI Model Architecture: Field Boundary Deduction Using Boundary Guidance

This directory contains the PyTorch implementation of the **Cropland Boundary Model**, combining CNN feature extraction, Spatial Transformer global context modelling, and a Boundary Guidance Module for precise agricultural field extent segmentation and boundary prediction.

---

## 🏗️ Model Architecture Overview

```text
Input (Sentinel-2 4-Band Image: B4, B3, B2, B8)
                   ↓
         [ CNN Encoder ]
                   ↓
         Multi-Scale Local Features
                   ↓
      [ Spatial Transformer ]
                   ↓
       Global Spatial Context
                   ↓
  ┌──────────────────────────────────┐
  │                                  │
[ Boundary Decoder ]      [ Boundary Guidance Module ]
  │                                  │
  ↓                                  ↓
Boundary Prediction       Boundary-Guided Features
  │                                  │
  ↓                                  ↓
256×256 Boundary Logits     [ Field Extent Decoder ]
                                     │
                                     ↓
                          256×256 Field Mask Logits
```

### Why Each Component is Used:
1. **CNN Encoder (`models/baseline_cnn.py` & `models/model.py`)**: Convolutional layers extract high-frequency local spatial textures, edges, and multi-spectral band interactions from 4-channel Sentinel-2 imagery `[Red, Green, Blue, NIR]`.
2. **Spatial Transformer (`models/transformer.py`)**: Multi-Head Self-Attention (MHSA) bottleneck models long-range spatial context, neighboring field interactions, and distant field edges without excessive computational complexity.
3. **Boundary Guidance Module (`models/boundary_guidance.py`)**: Uses boundary feature maps to compute a spatial attention gating map, refining feature activations near field perimeters and sharpening cropland segmentation.
4. **Dual Decoders**: Simultaneously predicts **Cropland Field Extent Mask** `(256, 256)` and **Field Boundary Map** `(256, 256)`.

---

## ⚠️ Empirical Boundary Label Note

- **Mask Inspection Finding**: Unique mask values across dataset samples are strictly **`[0.0, 1.0]`** (Binary Field Extent).
- **Technical Limitation Note**: Morphological boundary extraction (`dilation - erosion`) produces outer perimeter edges of connected cropland regions. Where adjacent fields touch with no non-cropland gap between them, binary gradient extracts the perimeter boundary of the merged region.

---

## 📉 Loss Functions & Training Configuration

### Loss Functions (`losses/boundary_loss.py`)
- **Segmentation Loss**: Combined `BCEWithLogitsLoss + Soft Dice Loss`
- **Boundary Loss**: Combined `BCEWithLogitsLoss + Soft Dice Loss`
- **Total Loss**:
  $$\text{Total Loss} = \text{Segmentation Loss} + \lambda \times \text{Boundary Loss}$$
  *(where $\lambda = 0.5$ is configured in `training/config.py`)*

### Training Parameters (`training/config.py`)
- **Optimizer**: AdamW (`lr=1e-3`, `weight_decay=1e-4`)
- **Batch Size**: `4`
- **Device**: Automated detection (`cuda` if GPU available, else `cpu`)
- **Random Seed**: `42`

---

## 📂 Output Files Directory Structure

All generated output files are saved under `outputs/`:

```text
outputs/
├── visualizations/
│   ├── boundaries/          # Sample RGB, mask, boundary & overlay PNGs
│   │   ├── sample_001_AT_1433_rgb.png
│   │   ├── sample_001_AT_1433_mask.png
│   │   ├── sample_001_AT_1433_boundary.png
│   │   └── sample_001_AT_1433_comparison.png
│   │
│   └── predictions/         # Model prediction comparison PNGs
│       ├── test_001_AT_245_comparison.png
│       └── test_002_FR_21056_comparison.png
│
├── metrics/
│   ├── validation_metrics.json
│   ├── test_metrics.json
│   └── training_history.csv
│
├── checkpoints/
│   └── best_model.pth        # Saved PyTorch model weights
│
└── model_summary.txt         # Parameter breakdown & experiment specs
```

---

## 🚀 Execution Commands

### 1. Run Boundary Generator Test & Visualizations
```bash
python -m models.boundary_generator
```

### 2. Run Model Training Pipeline
```bash
python -m training.train
```

### 3. Run Evaluation & Generate Prediction Visualizations
```bash
python -m evaluation.evaluate
```
