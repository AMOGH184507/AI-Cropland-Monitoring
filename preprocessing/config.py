import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "sentinel2"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
PROCESSED_DATA_DIR = PREPROCESSING_DIR / "data"
SPLITS_MANIFEST_PATH = PREPROCESSING_DIR / "splits.json"

# Sentinel-2 Band Definitions & Channel Ordering
# Canonical Band Order: Red (B4), Green (B3), Blue (B2), Near-Infrared (B8)
# Sentinel-2 Band Definitions & Channel Ordering
# Red, Green, Blue, NIR + NDVI
CANONICAL_BAND_ORDER = ["B4", "B3", "B2", "B8"]
INCLUDE_NDVI = True
# Normalization Constants (Sentinel-2 L2A Reflectance scale factor: 10,000)
NORM_SCALE = 10000.0

# Patching & Tiling Parameters
PATCH_SIZE = (256, 256)
STRIDE = (256, 256)
EMPTY_PATCH_THRESHOLD = 0.95  # Max fraction of nodata/empty background mask allowed

# Train / Val / Test Split Parameters
RANDOM_SEED = 42
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

# Augmentation Parameters (Applied strictly to 'train' split)
AUGMENT_PROBS = {
    "hflip": 0.5,
    "vflip": 0.5,
    "rot90": 0.5,
    "brightness": 0.5
}
BRIGHTNESS_JITTER_MAX = 0.2

# Mask Semantics Documentation
MASK_SEMANTICS = {
    0.0: "Non-Field / Background (roads, forests, water, urban)",
    1.0: "Cropland Field Extent (pixels within agricultural boundaries)"
}
