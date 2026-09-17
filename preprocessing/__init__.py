"""
AI Cropland Monitoring Preprocessing Package
=============================================

Exports:
- CroplandDataset: Self-contained PyTorch Dataset returning (image_tensor, mask_tensor, field_id)
- parse_tile: NetCDF and GeoTIFF tile parser
- extract_patches: Sliding window patch extraction function
- apply_augmentations: Aligned spatial & brightness transformations
"""

from preprocessing.dataset import CroplandDataset
from preprocessing.parse_netcdf import parse_tile
from preprocessing.patchify import extract_patches
from preprocessing.augment import apply_augmentations
from preprocessing.config import CANONICAL_BAND_ORDER, PATCH_SIZE, RANDOM_SEED, SPLIT_RATIOS

__all__ = [
    "CroplandDataset",
    "parse_tile",
    "extract_patches",
    "apply_augmentations",
    "CANONICAL_BAND_ORDER",
    "PATCH_SIZE",
    "RANDOM_SEED",
    "SPLIT_RATIOS"
]
