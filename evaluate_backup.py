import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from model import CTHBNet
from loss import MultiTaskLoss, generate_boundary_target
from dataset_loader import CroplandDataset
from train import calculate_metrics


# ==============================================================================
# EVALUATE MODEL ON SPECIFIED SPLIT
# ==============================================================================
def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates CTHBNet on a full dataset split.
    
    Returns:
        dict: Mean loss values and segmentation metrics across all samples
    """
    model.eval()
    total_loss = 0.0
    total_extent_loss = 0.0
    total_boundary_loss = 0.0
    num_batches = len(dataloader)

    iou_list, dice_list, prec_list, recall_list = [], [], [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            extent_logits, boundary_logits = model(images)
            loss, loss_dict = criterion(extent_logits, boundary_logits, masks)

            total_loss += loss.item()
            total_extent_loss += loss_dict["loss_extent"]
            total_boundary_loss += loss_dict["loss_boundary"]

            metrics = calculate_metrics(extent_logits, masks, threshold=0.6)
            iou_list.append(metrics["iou"])
            dice_list.append(metrics["dice"])
            prec_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])

    return {
        "loss": total_loss / num_batches,
        "extent_loss": total_extent_loss / num_batches,
        "boundary_loss": total_boundary_loss / num_batches,
        "iou": np.mean(iou_list),
        "dice": np.mean(dice_list),
        "precision": np.mean(prec_list),
        "recall": np.mean(recall_list)
    }


# ==============================================================================
# GENERATE AND SAVE PREDICTION VISUALIZATIONS
# ==============================================================================
def visualize_predictions(model, dataloader, device, output_dir="predictions", num_samples=5):
    """
    Generates multi-panel visual comparisons for prediction results:
    1. Sentinel-2 Image (False-Color Composite: NIR-Red-Green)
    2. Ground Truth Parcel Extent Mask
    3. Ground Truth Field Boundary Mask
    4. Predicted Parcel Extent Mask (Probabilities & Binary Threshold)
    5. Predicted Field Boundary Mask (Probabilities)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    sample_count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            extent_logits, boundary_logits = model(images)
            extent_probs = torch.sigmoid(extent_logits)
            boundary_probs = torch.sigmoid(boundary_logits)

            boundary_gt = generate_boundary_target(masks)

            batch_size = images.size(0)
            for b in range(batch_size):
                if sample_count >= num_samples:
                    break

                # Extract single sample [5, 6, 256, 256] -> pick middle timestep (t=3)
                img = images[b].cpu().numpy()  # [5, timesteps, H, W]
                t_idx = img.shape[1] // 2  # timestep index

                # Create RGB / False Color Composite for background visualization
                # Band mapping: 0=B2(Blue), 1=B3(Green), 2=B4(Red), 3=B8(NIR), 4=NDVI
                nir = img[3, t_idx]
                red = img[2, t_idx]
                green = img[1, t_idx]

                # False color composite (NIR, Red, Green) normalized [0, 1]
                fc_img = np.stack([nir, red, green], axis=-1)
                fc_img = np.clip(fc_img, 0, 1)

                gt_extent = masks[b].cpu().numpy()
                gt_boundary = boundary_gt[b, 0].cpu().numpy()

                pred_extent_prob = extent_probs[b, 0].cpu().numpy()
                pred_extent_bin = (pred_extent_prob > 0.6).astype(np.float32)
                pred_bound_prob = boundary_probs[b, 0].cpu().numpy()

                # Plot multi-panel comparison figure
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                axes[0].imshow(fc_img)
                axes[0].set_title(f"Sentinel-2 (NIR-R-G t={t_idx+1})")
                axes[0].axis("off")

                axes[1].imshow(gt_extent, cmap="Greens", vmin=0, vmax=1)
                axes[1].set_title("GT Parcel Extent")
                axes[1].axis("off")

                axes[2].imshow(gt_boundary, cmap="Reds", vmin=0, vmax=1)
                axes[2].set_title("GT Field Boundaries")
                axes[2].axis("off")

                axes[3].imshow(pred_extent_bin, cmap="Greens", vmin=0, vmax=1)
                axes[3].set_title("Predicted Extent (Binary)")
                axes[3].axis("off")

                axes[4].imshow(pred_bound_prob, cmap="Reds", vmin=0, vmax=1)
                axes[4].set_title("Predicted Boundary Prob")
                axes[4].axis("off")

                plt.tight_layout()
                save_path = os.path.join(output_dir, f"sample_{sample_count + 1:02d}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                print(f" Saved visualization sample {sample_count + 1} to {save_path}")
                sample_count += 1

            if sample_count >= num_samples:
                break


# ==============================================================================
# MAIN EVALUATION SCRIPT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate CTHBNet Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate prediction visualization images")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of visualization samples to generate")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save visual predictions")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"CTHBNET EVALUATION ON '{args.split.upper()}' SPLIT")
    print("=" * 70)
    print(f"   Device     : {device}")
    print(f"   Checkpoint : {args.checkpoint}")

    # 1. Dataset & DataLoader
    dataset = CroplandDataset(split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Model & Loss Function
    model = CTHBNet(in_bands=5, timesteps=6).to(device)
    criterion = MultiTaskLoss().to(device)

    # 3. Load Checkpoint Weights
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False
)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"   Loaded model weights from checkpoint epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print("   Loaded raw model state dictionary")
    else:
        print(f"  WARNING: Checkpoint path '{args.checkpoint}' not found! Evaluating randomly initialized model weights for testing.")

    # 4. Run Evaluation
    results = evaluate_model(model, dataloader, criterion, device)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"   Split             : {args.split.upper()}")
    print(f"   Total Loss        : {results['loss']:.4f}")
    print(f"   Extent Loss       : {results['extent_loss']:.4f}")
    print(f"   Boundary Loss     : {results['boundary_loss']:.4f}")
    print(f"   Parcel Extent mIoU: {results['iou']:.4f}")
    print(f"   Parcel Extent Dice: {results['dice']:.4f}")
    print(f"   Precision         : {results['precision']:.4f}")
    print(f"   Recall            : {results['recall']:.4f}")
    print("=" * 70)

    # 5. Generate Visualizations
    if args.visualize:
        print(f"\nGenerating {args.num_samples} prediction visualizations in '{args.output_dir}'...")
        visualize_predictions(model, dataloader, device, output_dir=args.output_dir, num_samples=args.num_samples)
        print(f" Visualizations generated successfully in directory '{args.output_dir}'!")


if __name__ == "__main__":
    main()
