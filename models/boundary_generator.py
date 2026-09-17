import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BOUNDARIES_VIS_DIR = OUTPUTS_DIR / "visualizations" / "boundaries"


def generate_boundary_map_numpy(
    mask_np: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Generates a binary field boundary map from a 2D binary field mask using morphological gradient.

    EMPIRICAL NOTE:
    The dataset masks are strictly binary [0.0, 1.0] (Cropland Field Extent).
    This function computes the outer/inner perimeter boundary of connected cropland regions.
    If two fields touch with no non-cropland gap between them, binary gradient extracts the
    perimeter of the merged region.

    Args:
        mask_np: np.ndarray of shape (H, W), values in {0.0, 1.0}
        kernel_size: odd integer size of morphological dilation/erosion structuring element

    Returns:
        boundary_np: np.ndarray of shape (H, W), float32 values in {0.0, 1.0}
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    mask_bool = mask_np > 0.5
    struct = np.ones((kernel_size, kernel_size), dtype=bool)

    dilated = binary_dilation(mask_bool, structure=struct)
    eroded = binary_erosion(mask_bool, structure=struct)

    # Morphological gradient: boundary is pixels present in dilation but not erosion
    boundary_bool = dilated & (~eroded)

    return boundary_bool.astype(np.float32)


def generate_boundary_map_torch(
    mask_tensor: torch.Tensor,
    kernel_size: int = 3
) -> torch.Tensor:
    """
    PyTorch GPU/CPU compatible morphological boundary extractor for batch tensors.

    Args:
        mask_tensor: torch.Tensor of shape (B, 1, H, W) or (H, W), float32

    Returns:
        boundary_tensor: torch.Tensor of same shape and device as input, float32 {0.0, 1.0}
    """
    original_shape = mask_tensor.shape
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(1)

    device = mask_tensor.device
    mask_bin = (mask_tensor > 0.5).float()

    # Max pool 2d acts as morphological dilation
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask_bin, kernel_size=kernel_size, stride=1, padding=padding)

    # -Max pool(-mask) acts as morphological erosion
    eroded = -F.max_pool2d(-mask_bin, kernel_size=kernel_size, stride=1, padding=padding)

    boundary = (dilated - eroded) > 0.5
    boundary = boundary.float()

    if len(original_shape) == 2:
        return boundary.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        return boundary.squeeze(1)

    return boundary


def create_boundary_visualization(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    boundary_np: np.ndarray,
    sample_name: str = "sample_001",
    save_dir: Path = BOUNDARIES_VIS_DIR
) -> None:
    """
    Generates and saves visual comparisons as .png files:
    - RGB image
    - Ground-truth field mask
    - Generated boundary map
    - RGB + mask overlay
    - RGB + boundary overlay
    - Combined 5-panel figure
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare RGB array (channels 0=B4(R), 1=B3(G), 2=B2(B))
    if image_np.shape[0] >= 3:
        rgb = np.stack([image_np[0], image_np[1], image_np[2]], axis=-1)
    else:
        rgb = np.repeat(image_np[0, :, :, None], 3, axis=-1)

    # Normalize RGB to [0, 1] for display
    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)
    rgb_display = np.clip((rgb - low) / (high - low + 1e-8), 0.0, 1.0)

    # 1. Save individual PNGs
    plt.imsave(save_dir / f"{sample_name}_rgb.png", rgb_display)
    plt.imsave(save_dir / f"{sample_name}_mask.png", mask_np, cmap="gray")
    plt.imsave(save_dir / f"{sample_name}_boundary.png", boundary_np, cmap="magma")

    # RGB + Mask Overlay
    overlay_mask = rgb_display.copy()
    overlay_mask[..., 1] = np.clip(overlay_mask[..., 1] + 0.4 * mask_np, 0.0, 1.0)  # Green tint for fields
    plt.imsave(save_dir / f"{sample_name}_mask_overlay.png", overlay_mask)

    # RGB + Boundary Overlay
    overlay_boundary = rgb_display.copy()
    overlay_boundary[..., 0] = np.clip(overlay_boundary[..., 0] + 0.8 * boundary_np, 0.0, 1.0)  # Red tint for boundaries
    plt.imsave(save_dir / f"{sample_name}_boundary_overlay.png", overlay_boundary)

    # 2. Combined Panel Figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(rgb_display)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Field Mask")
    axes[1].axis("off")

    axes[2].imshow(boundary_np, cmap="magma")
    axes[2].set_title("Boundary Map")
    axes[2].axis("off")

    axes[3].imshow(overlay_mask)
    axes[3].set_title("RGB + Mask Overlay")
    axes[3].axis("off")

    axes[4].imshow(overlay_boundary)
    axes[4].set_title("RGB + Boundary Overlay")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / f"{sample_name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 6: Testing Boundary Generator & Visualization Generation")
    print("=" * 70)

    try:
        from preprocessing.dataset import CroplandDataset
        dataset = CroplandDataset(split="train")
        print(f"Loaded CroplandDataset: {len(dataset)} samples found.")

        if len(dataset) > 0:
            for idx in range(min(3, len(dataset))):
                img_t, mask_t, fid = dataset[idx]
                img_np = img_t.numpy()
                mask_np = mask_t.numpy()

                boundary_np = generate_boundary_map_numpy(mask_np)
                sample_name = f"sample_{idx+1:03d}_{fid}"

                create_boundary_visualization(
                    img_np,
                    mask_np,
                    boundary_np,
                    sample_name=sample_name
                )
                print(f"  [SAVED] Visualizations for {sample_name} -> {BOUNDARIES_VIS_DIR}")

        print("=" * 70)
        print("[SUCCESS] Boundary Generator Test & PNG Generation Completed!")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error during boundary generator execution: {e}")
