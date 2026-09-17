# Preprocessing Package for AI-Based Intelligent Cropland Monitoring

This package provides a self-contained, reproducible data engineering pipeline that parses Sentinel-2 NetCDF tiles and GeoTIFF field boundary masks, extracts patches, manages reproducible field-level splits, applies data augmentations, and exposes a clean PyTorch `CroplandDataset` interface.

---

## 📁 Package Folder Structure

```
preprocessing/
├── __init__.py          # Package exports (CroplandDataset, parse_tile, extract_patches)
├── config.py            # Central configuration & magic numbers (patch size, seed, bands)
├── parse_netcdf.py      # NetCDF & GeoTIFF parser with spatial alignment assertions
├── patchify.py          # Patch extraction & field-level split generator
├── augment.py          # Aligned spatial transforms & image brightness jitter
├── dataset.py          # Self-contained PyTorch CroplandDataset class
├── splits.json          # Reproducible split manifest (patch_id -> split -> field_id)
├── README.md            # Technical documentation
└── data/                # Pre-patchified data directory
    ├── train/           # *.npy image and mask patch files
    ├── val/             # *.npy image and mask patch files
    └── test/            # *.npy image and mask patch files
```

---

## 🛰️ Band Order in Image Tensor

Every `image_tensor` returned by `CroplandDataset` follows the canonical 4-channel arrangement:

| Channel Index | Band Code | Wavelength / Description | Scale Range |
| :--- | :--- | :--- | :--- |
| **0** | `B4` | Red (665 nm) | `[0.0, 1.0]` |
| **1** | `B3` | Green (560 nm) | `[0.0, 1.0]` |
| **2** | `B2` | Blue (490 nm) | `[0.0, 1.0]` |
| **3** | `B8` | Near-Infrared / NIR (842 nm) | `[0.0, 1.0]` |

*(Optional 5th channel `NDVI` can be enabled by setting `INCLUDE_NDVI = True` in `preprocessing/config.py`)*.

---

## 🌾 Mask Semantics

Mask tensors are binary `float32` arrays:
- **`1.0`**: **Cropland Field Extent** (pixels inside agricultural field boundaries).
- **`0.0`**: **Non-Field / Background** (roads, forests, water, urban areas).

---

## ✂️ Patch Size & How to Change It

- Default patch size is `(256, 256)` pixels matching source Sentinel-2 tile dimensions (10m resolution).
- To change patch size or stride, edit `PATCH_SIZE` and `STRIDE` in `preprocessing/config.py`:
  ```python
  PATCH_SIZE = (128, 128)
  STRIDE = (128, 128)
  ```

---

## 📊 Dataset Splits & Statistics

Splits are constructed at the **FIELD / TILE level** (grouping by `field_id`) using `RANDOM_SEED = 42` to ensure patches from the same agricultural field never leak across splits:

- **Random Seed**: `42`
- **Split Ratios**: 70% Train / 15% Validation / 15% Test
- **Saved Patch Counts**:
  - `train`: **463** patches
  - `val`: **102** patches
  - `test`: **97** patches
  - **Total**: **662** patches saved under `preprocessing/data/`

---

## 🚀 Teammate Import & Instantiation Example

Teammates building segmentation models or analysis pipelines can import `CroplandDataset` directly without handling NetCDF or GeoTIFF files:

```python
from preprocessing.dataset import CroplandDataset
from torch.utils.data import DataLoader

# Instantiate training dataset (applies augmentations automatically)
train_ds = CroplandDataset(split="train")

# Instantiate validation/test dataset (no augmentation applied)
val_ds = CroplandDataset(split="val")
test_ds = CroplandDataset(split="test")

# Access single item
image_tensor, mask_tensor, field_id = train_ds[0]

print("Image shape:", image_tensor.shape)  # torch.Size([4, 256, 256])
print("Mask shape :", mask_tensor.shape)   # torch.Size([256, 256])
print("Field ID   :", field_id)            # e.g., 'AT_1433'

# Create PyTorch DataLoader
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
```

---

## 🔄 Regeneration Command

To regenerate all patches and update `preprocessing/splits.json` from raw NetCDF tiles:
```bash
python -m preprocessing.patchify
```
