import torch
import random
from typing import Tuple, Dict, Any, Optional

try:
    from preprocessing.config import (
        AUGMENT_PROBS,
        BRIGHTNESS_JITTER_MAX
    )
except ImportError:
    AUGMENT_PROBS = {
        "hflip": 0.5,
        "vflip": 0.5,
        "rot90": 0.5,
        "brightness": 0.5
    }
    BRIGHTNESS_JITTER_MAX = 0.2


def apply_augmentations(
    image: torch.Tensor,
    mask: torch.Tensor,
    probs: Dict[str, float] = AUGMENT_PROBS,
    brightness_delta: float = BRIGHTNESS_JITTER_MAX
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies aligned spatial and image-only augmentations to image and mask tensors.

    Args:
        image: torch.Tensor of shape (C, H, W), float32
        mask: torch.Tensor of shape (H, W) or (1, H, W), float32
        probs: dict of probabilities for each augmentation type
        brightness_delta: maximum percentage delta for brightness jitter (e.g. 0.2 for +/-20%)

    Returns:
        (augmented_image, augmented_mask): torch.Tensor pair with identical spatial dimensions
    """
    # Ensure inputs are torch.Tensors
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32)

    # 1. Random Horizontal Flip (aligned for image and mask)
    if random.random() < probs.get("hflip", 0.5):
        # Flip along spatial width axis (dim -1)
        image = torch.flip(image, dims=[-1])
        mask = torch.flip(mask, dims=[-1])

    # 2. Random Vertical Flip (aligned for image and mask)
    if random.random() < probs.get("vflip", 0.5):
        # Flip along spatial height axis (dim -2)
        image = torch.flip(image, dims=[-2])
        mask = torch.flip(mask, dims=[-2])

    # 3. Random 90-degree Rotations (aligned for image and mask)
    if random.random() < probs.get("rot90", 0.5):
        k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        image = torch.rot90(image, k=k, dims=[-2, -1])
        mask = torch.rot90(mask, k=k, dims=[-2, -1])

    # 4. Brightness Jitter (IMAGE ONLY - never mask!)
    if random.random() < probs.get("brightness", 0.5):
        # Sample factor in [1 - delta, 1 + delta]
        factor = random.uniform(1.0 - brightness_delta, 1.0 + brightness_delta)
        image = image * factor
        image = torch.clamp(image, 0.0, 1.0)

    return image, mask


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 3 SMOKE TEST: Data Augmentation & Spatial Alignment")
    print("=" * 70)

    # Create synthetic test tensors
    # Image: 4 channels, 256x256
    # Mask: 256x256 with an asymmetric pattern to verify exact rotation/flip alignment
    dummy_img = torch.ones((4, 256, 256), dtype=torch.float32)
    dummy_mask = torch.zeros((256, 256), dtype=torch.float32)
    dummy_mask[50:100, 100:200] = 1.0  # Asymmetric rectangular region

    print(f"Original Image Shape : {dummy_img.shape}")
    print(f"Original Mask Shape  : {dummy_mask.shape}")
    print(f"Original Mask Sum    : {dummy_mask.sum().item()} pixels")

    # Apply augmentations (forcing probs to 1.0 for test)
    test_probs = {"hflip": 1.0, "vflip": 1.0, "rot90": 1.0, "brightness": 1.0}
    aug_img, aug_mask = apply_augmentations(dummy_img, dummy_mask, probs=test_probs)

    print("\n--- AUGMENTED RESULTS ---")
    print(f"Augmented Image Shape: {aug_img.shape}")
    print(f"Augmented Mask Shape : {aug_mask.shape}")
    print(f"Augmented Image Min  : {aug_img.min().item():.4f}")
    print(f"Augmented Image Max  : {aug_img.max().item():.4f}")
    print(f"Augmented Mask Sum   : {aug_mask.sum().item()} pixels (Must match original sum!)")
    print(f"Augmented Mask Values: {torch.unique(aug_mask).tolist()}")

    assert aug_img.shape == dummy_img.shape, "Shape mismatch in image!"
    assert aug_mask.shape == dummy_mask.shape, "Shape mismatch in mask!"
    assert aug_mask.sum().item() == dummy_mask.sum().item(), "Pixel count changed after spatial transform!"

    print("=" * 70)
    print("[SUCCESS] STEP 3 SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)
