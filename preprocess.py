import xarray as xr
import rasterio
import numpy as np
import torch
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2"
)

TRAIN_IMAGE_DIR = BASE_DIR / "train" / "images"
TRAIN_MASK_DIR = BASE_DIR / "train" / "masks"


# ============================================================
# FIND ONE TRAINING SAMPLE
# ============================================================

image_files = list(TRAIN_IMAGE_DIR.rglob("*.nc"))

if len(image_files) == 0:
    raise FileNotFoundError("No .nc files found.")

image_path = image_files[0]

country = image_path.parent.name

mask_name = image_path.name.replace(
    "_S2_10m_256.nc",
    "_S2label_10m_256.tif"
)

mask_path = TRAIN_MASK_DIR / country / mask_name


print("=" * 60)
print("PREPROCESSING TEST")
print("=" * 60)

print("Image :", image_path)
print("Mask  :", mask_path)


# ============================================================
# LOAD SENTINEL-2 DATA
# ============================================================

ds = xr.open_dataset(image_path)


# ============================================================
# LOAD ALL 6 TIME STEPS
# ============================================================

bands = ["B2", "B3", "B4", "B8", "NDVI"]

data = []

for band in bands:

    band_data = ds[band].values

    print(
        band,
        "shape:",
        band_data.shape
    )

    data.append(band_data)


# ============================================================
# STACK DATA
# ============================================================

# Current shape:
# (5, 6, 256, 256)

data = np.stack(data, axis=0)

print("\nStacked NumPy shape:")
print(data.shape)


# ============================================================
# HANDLE NaN / INFINITE VALUES
# ============================================================

data = np.nan_to_num(
    data,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# NORMALIZATION
# ============================================================

normalized_data = np.zeros_like(
    data,
    dtype=np.float32
)

for band_index in range(data.shape[0]):

    band = data[band_index]

    low = np.percentile(
        band,
        2
    )

    high = np.percentile(
        band,
        98
    )

    normalized_band = (
        band - low
    ) / (
        high - low + 1e-8
    )

    normalized_band = np.clip(
        normalized_band,
        0,
        1
    )

    normalized_data[band_index] = normalized_band


# ============================================================
# LOAD MASK
# ============================================================

with rasterio.open(mask_path) as src:

    mask = src.read(1)


print("\nMask shape:")
print(mask.shape)

print(
    "Mask values:",
    np.unique(mask)
)


# ============================================================
# CONVERT TO PYTORCH TENSORS
# ============================================================

image_tensor = torch.tensor(
    normalized_data,
    dtype=torch.float32
)

mask_tensor = torch.tensor(
    mask,
    dtype=torch.float32
)


# ============================================================
# PRINT FINAL SHAPES
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL INPUT")
print("=" * 60)

print(
    "Image tensor:",
    image_tensor.shape
)

print(
    "Mask tensor:",
    mask_tensor.shape
)


# ============================================================
# EXPECTED SHAPE
# ============================================================

print("\nExpected image shape:")
print("(5, 6, 256, 256)")

print("\nExpected mask shape:")
print("(256, 256)")


# ============================================================
# CLOSE DATASET
# ============================================================

ds.close()


print("\n" + "=" * 60)
print("PREPROCESSING TEST COMPLETED")
print("=" * 60)