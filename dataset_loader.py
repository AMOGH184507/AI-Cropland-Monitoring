import xarray as xr
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


import os

# Dynamic base directory determination: env var > relative project path > local fallback
DEFAULT_BASE_DIR = Path(__file__).resolve().parent / "dataset" / "sentinel2"
ENV_BASE_DIR = os.environ.get("DATASET_DIR")
if ENV_BASE_DIR:
    BASE_DIR = Path(ENV_BASE_DIR)
elif DEFAULT_BASE_DIR.exists():
    BASE_DIR = DEFAULT_BASE_DIR
else:
    BASE_DIR = Path(
        r"C:\Users\Amogh roy s\OneDrive\Desktop\AI_Cropland_Monitoring\dataset\sentinel2"
    )


# ============================================================
# DATASET CLASS
# ============================================================

class CroplandDataset(Dataset):

    def __init__(self, split, base_dir=None):

        self.split = split
        self.base_dir = Path(base_dir) if base_dir is not None else BASE_DIR

        self.image_dir = self.base_dir / split / "images"
        self.mask_dir = self.base_dir / split / "masks"

        # Find all .nc images
        self.image_files = sorted(
            self.image_dir.rglob("*.nc")
        )

        print(
            f"{split.upper()} dataset: "
            f"{len(self.image_files)} images"
        )


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, index):

        # ----------------------------------------------------
        # IMAGE PATH
        # ----------------------------------------------------

        image_path = self.image_files[index]

        country = image_path.parent.name

        image_name = image_path.name


        # ----------------------------------------------------
        # MATCHING MASK
        # ----------------------------------------------------

        mask_name = image_name.replace(
            "_S2_10m_256.nc",
            "_S2label_10m_256.tif"
        )

        mask_path = (
            self.mask_dir /
            country /
            mask_name
        )


        # ----------------------------------------------------
        # LOAD IMAGE & MASK WITH CORRUPTION SAFETY FALLBACK
        # ----------------------------------------------------
        try:
            ds = xr.open_dataset(image_path)

            bands = [
                "B2",
                "B3",
                "B4",
                "B8",
                "NDVI"
            ]

            data = []

            for band in bands:
                band_data = ds[band].values
                data.append(band_data)

            ds.close()

            # ----------------------------------------------------
            # STACK & HANDLE NaN / INFINITE VALUES
            # Shape: (5, 6, 256, 256)
            # ----------------------------------------------------
            data = np.stack(data, axis=0)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # ----------------------------------------------------
            # NORMALIZATION
            # ----------------------------------------------------
            normalized = np.zeros_like(data, dtype=np.float32)

            for band_index in range(data.shape[0]):
                band = data[band_index]
                low = np.percentile(band, 2)
                high = np.percentile(band, 98)
                band = (band - low) / (high - low + 1e-8)
                band = np.clip(band, 0, 1)
                normalized[band_index] = band

            # ----------------------------------------------------
            # LOAD MASK
            # ----------------------------------------------------
            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            # ----------------------------------------------------
            # CONVERT TO TENSORS
            # ----------------------------------------------------
            image_tensor = torch.tensor(normalized, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)

            return image_tensor, mask_tensor

        except Exception as e:
            print(f" [WARNING] Skipping corrupted sample #{index} ({image_name}): {e}")
            # Fallback to next sample to prevent training crashes
            next_index = (index + 1) % len(self.image_files)
            return self.__getitem__(next_index)


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING CROPLAND DATASET")
    print("=" * 60)


    # --------------------------------------------------------
    # Create training dataset
    # --------------------------------------------------------

    train_dataset = CroplandDataset(
        "train"
    )


    # --------------------------------------------------------
    # Create validation dataset
    # --------------------------------------------------------

    val_dataset = CroplandDataset(
        "val"
    )


    # --------------------------------------------------------
    # Create test dataset
    # --------------------------------------------------------

    test_dataset = CroplandDataset(
        "test"
    )


    # --------------------------------------------------------
    # Check dataset sizes
    # --------------------------------------------------------

    print()
    print("Dataset sizes:")
    print("Train:", len(train_dataset))
    print("Val  :", len(val_dataset))
    print("Test :", len(test_dataset))


    # --------------------------------------------------------
    # Load ONE sample
    # --------------------------------------------------------

    image, mask = train_dataset[0]


    print()
    print("=" * 60)
    print("SINGLE SAMPLE TEST")
    print("=" * 60)

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Mask shape:",
        mask.shape
    )

    print(
        "Image dtype:",
        image.dtype
    )

    print(
        "Mask dtype:",
        mask.dtype
    )


    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )


    # --------------------------------------------------------
    # Load ONE BATCH
    # --------------------------------------------------------

    images, masks = next(
        iter(train_loader)
    )


    print()
    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print(
        "Batch image shape:",
        images.shape
    )

    print(
        "Batch mask shape:",
        masks.shape
    )


    print()
    print("=" * 60)
    print("DATASET + DATALOADER TEST COMPLETED")
    print("=" * 60)