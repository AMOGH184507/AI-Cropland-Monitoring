# Google Colab Setup & Training Guide for CTHBNet

This guide provides step-by-step instructions to fix Google Drive FUSE NetCDF reading issues (`OSError: [Errno -101] NetCDF: HDF error`), verify dataset integrity, and run **CTHBNet (CNN-Transformer Hybrid with Boundary Guidance)** training on **Google Colab GPU**.

---

## 🛠️ Root Cause & Solution: NetCDF HDF Error

### Problem
Reading `.nc` files directly from Google Drive (`/content/drive/MyDrive/...`) fails with `OSError: [Errno -101] NetCDF: HDF error`. This occurs because Google Drive's FUSE filesystem does not support POSIX low-level file locking or concurrent random `pread`/`fseek` reads required by `xarray` and `netCDF4`.

### Solution
1. **Mount Google Drive** to access the stored project and dataset.
2. **Copy the complete project and dataset** to Google Colab's fast local ephemeral disk storage (`/content/AI_Cropland_Monitoring`).
3. **Set `DATASET_DIR`** so `dataset_loader.py` uses local fast NVMe/SSD paths (`/content/AI_Cropland_Monitoring/dataset/sentinel2`).
4. **Run `verify_dataset.py`** to safely verify all 500 train, 100 validation, and 100 test NetCDF `.nc` files and matching `.tif` masks before training.

---

## 🚀 Exact Step-by-Step Google Colab Commands

### Step 1: Mount Google Drive & Copy Project to Local Colab Disk

In your Google Colab notebook cell:

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# 2. Create local directory on fast Colab storage
!mkdir -p /content/AI_Cropland_Monitoring

# 3. Copy project and dataset from Google Drive to local disk
# (Replace '/content/drive/MyDrive/AI_Cropland_Monitoring' with your Drive folder path if different)
!cp -r /content/drive/MyDrive/AI_Cropland_Monitoring/* /content/AI_Cropland_Monitoring/

# 4. Navigate into local project directory
%cd /content/AI_Cropland_Monitoring
```

Set environment variable so `dataset_loader.py` uses the local path:

```python
import os
os.environ["DATASET_DIR"] = "/content/AI_Cropland_Monitoring/dataset/sentinel2"
print("Local dataset path set to:", os.environ["DATASET_DIR"])
```

---

### Step 2: Install Dependencies & Verify GPU Hardware

```bash
# Install requirements
!pip install -r requirements.txt
```

Verify GPU acceleration (Tesla T4 / V100 / A100):

```python
import torch

print("=" * 60)
print("GPU HARDWARE VERIFICATION")
print("=" * 60)
print("CUDA Available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Device Name :", torch.cuda.get_device_name(0))
    print("Device Count    :", torch.cuda.device_count())
    print("CUDA Version    :", torch.version.cuda)
    print("PyTorch Version :", torch.__version__)
else:
    print("⚠️ WARNING: GPU not detected! Switch runtime to GPU in Colab menu.")
print("=" * 60)
```

Command line GPU status check:
```bash
!nvidia-smi
```

---

### Step 3: Run Safe NetCDF Dataset Verification

Run the non-modifying dataset verification step. This script:
- Checks all 500 train `.nc` files.
- Checks all 100 validation `.nc` files.
- Checks all 100 test `.nc` files.
- Verifies that all 5 bands (`B2`, `B3`, `B4`, `B8`, `NDVI`) can be opened cleanly via `xarray`.
- Confirms matching `.tif` masks exist and match image dimensions.
- Reports any corrupted or unreadable files explicitly.
- **Does NOT redownload or modify any files.**
- **Does NOT modify the CTHBNet architecture.**

```bash
!python verify_dataset.py
```

---

### Step 4: Test DataLoader Batch Reading from Local Storage

Verify dataset loading with PyTorch DataLoader:

```python
from dataset_loader import CroplandDataset, DataLoader

train_dataset = CroplandDataset(split="train")
val_dataset   = CroplandDataset(split="val")
test_dataset  = CroplandDataset(split="test")

print(f"Train Dataset Size : {len(train_dataset)} samples")
print(f"Val Dataset Size   : {len(val_dataset)} samples")
print(f"Test Dataset Size  : {len(test_dataset)} samples")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
images, masks = next(iter(train_loader))

print("\nDataLoader Verification:")
print(f"  Input Batch Tensor Shape : {list(images.shape)} (Expected: [2, 5, 6, 256, 256])")
print(f"  Target Mask Tensor Shape : {list(masks.shape)}  (Expected: [2, 256, 256])")
```

---

### Step 5: Run Small GPU Training Test (1-Epoch Dry Run)

Before starting the full training run, execute a 1-epoch GPU test run to verify forward pass, multi-task loss calculation, backward pass, optimizer step, validation, and checkpoint saving on GPU:

```python
from train import run_training

print("Starting Small GPU Training Test (1 Epoch)...")
run_training(
    batch_size=2,           # Test batch size
    num_epochs=1,           # 1 epoch dry-run
    lr=1e-4,
    checkpoint_dir="checkpoints"
)
print("Small GPU Training Test Passed Successfully!")
```

*Or via terminal command:*
```bash
!python -c "from train import run_training; run_training(batch_size=2, num_epochs=1, lr=1e-4)"
```

---

### Step 6: Run Full CTHBNet GPU Training

Once the 1-epoch test passes, start the full CTHBNet training:

```python
from train import run_training

# Start Full Training
run_training(
    batch_size=2,           # Keep batch_size=2 (or increase to 4/8 if GPU memory permits)
    num_epochs=30,          # Full training epochs
    lr=1e-4,
    checkpoint_dir="checkpoints"
)
```

*Or via terminal command:*
```bash
!python -c "from train import run_training; run_training(batch_size=2, num_epochs=30, lr=1e-4)"
```

---

### Step 7: Backup Best Model Checkpoint to Google Drive

After training finishes, save `best_model.pth` back to Google Drive so it persists after the Colab session closes:

```python
import shutil
import os

drive_backup_dir = "/content/drive/MyDrive/AI_Cropland_Monitoring/checkpoints"
os.makedirs(drive_backup_dir, exist_ok=True)

shutil.copy("checkpoints/best_model.pth", os.path.join(drive_backup_dir, "best_model.pth"))
print(f"Saved best_model.pth to Google Drive at: {drive_backup_dir}/best_model.pth")
```

---

### Step 8: Run Final Test Evaluation & Generate Visualizations

Evaluate `best_model.pth` on the test set and generate multi-panel visual prediction figures:

```bash
!python evaluate.py --checkpoint checkpoints/best_model.pth --split test --visualize --num_samples 5 --output_dir predictions
```

Display prediction figures inside Colab notebook cells:

```python
from IPython.display import Image, display
import glob

prediction_files = sorted(glob.glob("predictions/*.png"))
for img_path in prediction_files:
    print(f"Displaying: {img_path}")
    display(Image(filename=img_path))
```

---

## 🔒 Verification Checklist

- [x] Model architecture (`model/`) and loss functions (`loss.py`) left 100% unchanged.
- [x] Dataset files and structure left unmodified.
- [x] Project and dataset copied to Colab local fast disk (`/content/AI_Cropland_Monitoring`).
- [x] `verify_dataset.py` checks all 500 train, 100 val, and 100 test NetCDF & mask files.
- [x] DataLoader tested from local fast disk path.
- [x] Small 1-epoch GPU test run verified before full training.
- [x] Full GPU training and Drive checkpoint backup configured.
