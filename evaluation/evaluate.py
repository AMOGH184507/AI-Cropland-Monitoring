import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

from preprocessing.dataset import CroplandDataset
from models.model import CroplandBoundaryModel
from models.boundary_generator import generate_boundary_map_torch, generate_boundary_map_numpy
from evaluation.metrics import compute_binary_metrics
from training.config import (
    DEVICE,
    BEST_MODEL_PATH,
    VAL_METRICS_JSON,
    TEST_METRICS_JSON,
    PRED_VIS_DIR
)


def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = DEVICE
) -> Dict[str, Dict[str, float]]:
    """Evaluates model over DataLoader for both Field Extent and Boundary prediction."""
    model.eval()

    field_metrics_list = []
    boundary_metrics_list = []

    with torch.no_grad():
        for imgs, masks, fids in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            gt_boundaries = generate_boundary_map_torch(masks)

            field_logits, bound_logits = model(imgs)

            field_probs = torch.sigmoid(field_logits).squeeze(1)
            bound_probs = torch.sigmoid(bound_logits).squeeze(1)

            # Compute batch metrics
            f_m = compute_binary_metrics(field_probs, masks)
            b_m = compute_binary_metrics(bound_probs, gt_boundaries)

            field_metrics_list.append(f_m)
            boundary_metrics_list.append(b_m)

    # Average metrics
    avg_field = {k: float(np.mean([m[k] for m in field_metrics_list])) for k in field_metrics_list[0].keys()}
    avg_bound = {k: float(np.mean([m[k] for m in boundary_metrics_list])) for k in boundary_metrics_list[0].keys()}

    return {
        "field_extent_metrics": avg_field,
        "boundary_metrics": avg_bound
    }


def generate_prediction_visualizations(
    model: torch.nn.Module,
    dataset: CroplandDataset,
    num_samples: int = 5,
    out_dir: Path = PRED_VIS_DIR,
    device: torch.device = DEVICE
) -> None:
    """
    Generates and saves multi-panel prediction visualization PNG files under outputs/visualizations/predictions/

    Panels:
    1. Input RGB Image
    2. Ground Truth Field Mask
    3. Predicted Field Mask
    4. Ground Truth Boundary Map
    5. Predicted Boundary Map
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    sample_indices = list(range(min(num_samples, len(dataset))))

    with torch.no_grad():
        for idx in sample_indices:
            img_t, mask_t, fid = dataset[idx]
            img_batch = img_t.unsqueeze(0).to(device)

            field_logits, bound_logits = model(img_batch)
            pred_field = torch.sigmoid(field_logits).squeeze().cpu().numpy()
            pred_bound = torch.sigmoid(bound_logits).squeeze().cpu().numpy()

            img_np = img_t.numpy()
            mask_np = mask_t.numpy()
            gt_bound_np = generate_boundary_map_numpy(mask_np)

            # Build RGB image
            if img_np.shape[0] >= 3:
                rgb = np.stack([img_np[0], img_np[1], img_np[2]], axis=-1)
            else:
                rgb = np.repeat(img_np[0, :, :, None], 3, axis=-1)

            low = np.percentile(rgb, 2)
            high = np.percentile(rgb, 98)
            rgb_display = np.clip((rgb - low) / (high - low + 1e-8), 0.0, 1.0)

            bin_pred_field = (pred_field > 0.5).astype(np.float32)
            bin_pred_bound = (pred_bound > 0.5).astype(np.float32)

            # Generate 5-panel figure
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            axes[0].imshow(rgb_display)
            axes[0].set_title("Input RGB")
            axes[0].axis("off")

            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("GT Field Mask")
            axes[1].axis("off")

            axes[2].imshow(bin_pred_field, cmap="gray")
            axes[2].set_title("Pred Field Mask")
            axes[2].axis("off")

            axes[3].imshow(gt_bound_np, cmap="magma")
            axes[3].set_title("GT Boundary")
            axes[3].axis("off")

            axes[4].imshow(bin_pred_bound, cmap="magma")
            axes[4].set_title("Pred Boundary")
            axes[4].axis("off")

            plt.suptitle(f"Prediction Comparison — Field ID: {fid}", fontsize=14)
            plt.tight_layout()

            filename = f"test_{idx+1:03d}_{fid}_comparison.png"
            plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  [SAVED] Prediction Visualization -> {out_dir / filename}")


def run_evaluation() -> None:
    """Runs complete model evaluation on validation and test datasets."""
    print("=" * 70)
    print("STAGE 6: Running Model Evaluation & Visualization Generation")
    print("=" * 70)

    val_ds = CroplandDataset(split="val")
    test_ds = CroplandDataset(split="test")

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)

    model = CroplandBoundaryModel(in_channels=4, base_features=32).to(DEVICE)

    if BEST_MODEL_PATH.exists():
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained model checkpoint from: {BEST_MODEL_PATH}")
    else:
        print("⚠️ No trained checkpoint found! Running evaluation on randomly initialized weights.")

    # 1. Validation Evaluation
    val_results = evaluate_loader(model, val_loader)
    VAL_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(VAL_METRICS_JSON, "w") as f:
        json.dump(val_results, f, indent=2)
    print(f"Validation Metrics saved to: {VAL_METRICS_JSON}")

    # 2. Test Evaluation
    test_results = evaluate_loader(model, test_loader)
    TEST_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_METRICS_JSON, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"Test Metrics saved to: {TEST_METRICS_JSON}")

    # 3. Generate Prediction Comparison PNGs
    print("\nGenerating prediction comparison visualizations...")
    generate_prediction_visualizations(model, test_ds, num_samples=5)

    print("=" * 70)
    print("[SUCCESS] Evaluation & Prediction Visualization Completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
