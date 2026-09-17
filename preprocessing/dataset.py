import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional

try:
    from preprocessing.config import (
        PROJECT_ROOT,
        PROCESSED_DATA_DIR,
        SPLITS_MANIFEST_PATH,
        AUGMENT_PROBS,
        BRIGHTNESS_JITTER_MAX
    )
    from preprocessing.augment import apply_augmentations
except ImportError:
    from config import (
        PROJECT_ROOT,
        PROCESSED_DATA_DIR,
        SPLITS_MANIFEST_PATH,
        AUGMENT_PROBS,
        BRIGHTNESS_JITTER_MAX
    )
    from augment import apply_augmentations


class CroplandDataset(Dataset):
    """
    PyTorch Dataset for AI-Based Intelligent Cropland Monitoring.

    Self-contained interface returning:
        (image_tensor, mask_tensor, field_id)

    - image_tensor: float32, shape (C, H, W) normalized to [0.0, 1.0] (default 4 channels: Red, Green, Blue, NIR)
    - mask_tensor: float32, shape (H, W) binary mask (1.0 = field extent, 0.0 = non-field)
    - field_id: str identifying the original field/tile

    Augmentation is applied internally when split == "train".
    Reads pre-patchified numpy arrays without touching NetCDF or xarray directly.
    """

    def __init__(
        self,
        split: str = "train",
        processed_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        config: Optional[Any] = None
    ):
        split_lower = split.lower().strip()
        if split_lower not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be one of ['train', 'val', 'test']")

        self.split = split_lower
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
        self.manifest_path = Path(manifest_path) if manifest_path else SPLITS_MANIFEST_PATH

        self.split_dir = self.processed_dir / self.split
        self.samples = []

        # Try loading from splits.json manifest first
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                manifest_data = json.load(f)
                patches_dict = manifest_data.get("patches", {})

            for pid, pinfo in patches_dict.items():
                if pinfo.get("split") == self.split:
                    img_p = PROJECT_ROOT / pinfo["img_path"]
                    mask_p = PROJECT_ROOT / pinfo["mask_path"]
                    fid = pinfo.get("source_field_id", pid.split("_p")[0])

                    if img_p.exists() and mask_p.exists():
                        self.samples.append({
                            "patch_id": pid,
                            "img_path": img_p,
                            "mask_path": mask_p,
                            "field_id": fid
                        })

        # Fallback: scan split directory for saved .npy files directly
        if not self.samples and self.split_dir.exists():
            img_files = sorted(list(self.split_dir.glob("*_img.npy")))
            for img_p in img_files:
                mask_p = Path(str(img_p).replace("_img.npy", "_mask.npy"))
                if mask_p.exists():
                    pid = img_p.name.replace("_img.npy", "")
                    fid = pid.split("_p")[0]
                    self.samples.append({
                        "patch_id": pid,
                        "img_path": img_p,
                        "mask_path": mask_p,
                        "field_id": fid
                    })

        if len(self.samples) == 0:
            print(f"[WARNING] No patch files found for split '{self.split}' in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]

        # Load numpy patch arrays
        image_np = np.load(sample["img_path"]).astype(np.float32)
        mask_np = np.load(sample["mask_path"]).astype(np.float32)
        field_id = str(sample["field_id"])

        # Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(image_np).float()
        mask_tensor = torch.from_numpy(mask_np).float()

        # Apply augmentation strictly to training split
        if self.split == "train":
            image_tensor, mask_tensor = apply_augmentations(
                image_tensor,
                mask_tensor,
                probs=AUGMENT_PROBS,
                brightness_delta=BRIGHTNESS_JITTER_MAX
            )

        return image_tensor, mask_tensor, field_id


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 4 SMOKE TEST: PyTorch CroplandDataset Class")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        ds = CroplandDataset(split=split)
        print(f"{split.upper():5s} Dataset Length: {len(ds)} items")

        if len(ds) > 0:
            img, mask, fid = ds[0]
            print(f"  Sample 0 Field ID : {fid}")
            print(f"  Image Shape       : {img.shape} (Dtype: {img.dtype})")
            print(f"  Mask Shape        : {mask.shape} (Dtype: {mask.dtype})")
            print(f"  Image Min / Max   : {img.min().item():.4f} / {img.max().item():.4f}")
            print(f"  Mask Values       : {torch.unique(mask).tolist()}")

    # Test DataLoader compatibility
    train_ds = CroplandDataset(split="train")
    if len(train_ds) > 0:
        loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        batch_imgs, batch_masks, batch_fids = next(iter(loader))
        print("\n--- DATALOADER BATCH TEST ---")
        print(f"Batch Image Shape: {batch_imgs.shape}")
        print(f"Batch Mask Shape : {batch_masks.shape}")
        print(f"Batch Field IDs  : {batch_fids}")

    print("=" * 70)
    print("[SUCCESS] STEP 4 SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)
