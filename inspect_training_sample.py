import xarray as xr
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2"
)

TRAIN_IMAGE_DIR = BASE_DIR / "train" / "images"
TRAIN_MASK_DIR = BASE_DIR / "train" / "masks"


# ============================================================
# FIND ONE IMAGE
# ============================================================

image_files = list(TRAIN_IMAGE_DIR.rglob("*.nc"))

if len(image_files) == 0:
    print("❌ No training .nc files found.")
    exit()

image_path = image_files[0]

country = image_path.parent.name
image_name = image_path.name

mask_name = image_name.replace(
    "_S2_10m_256.nc",
    "_S2label_10m_256.tif"
)

mask_path = TRAIN_MASK_DIR / country / mask_name


print("=" * 60)
print("TRAINING SAMPLE")
print("=" * 60)

print("Country :", country)
print("Image   :", image_path)
print("Mask    :", mask_path)


# ============================================================
# CHECK MASK
# ============================================================

if not mask_path.exists():

    print("❌ Matching mask not found!")
    exit()

print("✅ Matching mask found")


# ============================================================
# OPEN SENTINEL-2 IMAGE
# ============================================================

print("\nOpening Sentinel-2 image...")

ds = xr.open_dataset(image_path)

print("\nDataset:")
print(ds)

print("\nVariables:")
print(list(ds.data_vars))

print("\nDimensions:")
print(ds.sizes)


# ============================================================
# READ BANDS
# ============================================================

B2 = ds["B2"].isel(time=0).values
B3 = ds["B3"].isel(time=0).values
B4 = ds["B4"].isel(time=0).values
B8 = ds["B8"].isel(time=0).values
NDVI = ds["NDVI"].isel(time=0).values


print("\nBand shapes:")

print("B2   :", B2.shape)
print("B3   :", B3.shape)
print("B4   :", B4.shape)
print("B8   :", B8.shape)
print("NDVI :", NDVI.shape)


# ============================================================
# CREATE RGB
# ============================================================

RGB = (
    __import__("numpy").stack(
        [B4, B3, B2],
        axis=2
    )
)

print("\nRGB shape:", RGB.shape)


# ============================================================
# NORMALIZE RGB FOR DISPLAY
# ============================================================

import numpy as np

RGB_display = np.zeros_like(RGB, dtype=np.float32)

for i in range(3):

    band = RGB[:, :, i]

    low = np.percentile(band, 2)
    high = np.percentile(band, 98)

    band = (band - low) / (high - low + 1e-8)

    band = np.clip(
        band,
        0,
        1
    )

    RGB_display[:, :, i] = band


# ============================================================
# OPEN MASK
# ============================================================

print("\nOpening mask...")

with rasterio.open(mask_path) as src:

    mask = src.read(1)

    mask_height = src.height
    mask_width = src.width


print("Mask shape:", mask.shape)
print("Mask dimensions:", mask_height, "x", mask_width)

print(
    "Unique mask values:",
    np.unique(mask)
)


# ============================================================
# CHECK DIMENSIONS
# ============================================================

if RGB.shape[:2] == mask.shape:

    print("\n✅ IMAGE AND MASK DIMENSIONS MATCH")

else:

    print("\n❌ IMAGE AND MASK DIMENSIONS DO NOT MATCH")


# ============================================================
# DISPLAY RESULTS
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(RGB_display)

plt.title(
    f"Sentinel-2 RGB\n{image_name}"
)

plt.axis("off")

plt.show()


plt.figure(figsize=(6, 6))

plt.imshow(mask)

plt.title(
    "Ground Truth Mask"
)

plt.axis("off")

plt.show()


# ============================================================
# NDVI
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(NDVI)

plt.title("NDVI")

plt.axis("off")

plt.colorbar()

plt.show()


# ============================================================
# CLOSE DATASET
# ============================================================

ds.close()

print("\n============================================================")
print("SAMPLE INSPECTION COMPLETED")
print("============================================================")