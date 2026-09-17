import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from pathlib import Path

from model import CTHBNet
from loss import MultiTaskLoss, generate_boundary_target
from dataset_loader import CroplandDataset


# ==============================================================================
# EVALUATION METRICS (IoU, Dice/F1, Precision, Recall)
# ==============================================================================
def calculate_metrics(pred_logits: torch.Tensor, target_masks: torch.Tensor, threshold: float = 0.5):
    """
    Calculates binary segmentation metrics: IoU, Dice/F1, Precision, Recall.
    
    Args:
        pred_logits: Tensor [B, 1, H, W]
        target_masks: Tensor [B, H, W] or [B, 1, H, W]
        threshold: Classification probability threshold
    Returns:
        dict: {'iou': float, 'dice': float, 'precision': float, 'recall': float}
    """
    if target_masks.dim() == 3:
        target_masks = target_masks.unsqueeze(1)
        
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()
    targets = (target_masks > 0.5).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall
    }


# ==============================================================================
# TRAIN ONE EPOCH
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_extent_loss = 0.0
    total_boundary_loss = 0.0
    num_batches = len(dataloader)

    for step, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        extent_logits, boundary_logits = model(images)
        loss, loss_dict = criterion(extent_logits, boundary_logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_extent_loss += loss_dict["loss_extent"]
        total_boundary_loss += loss_dict["loss_boundary"]

    return {
        "loss": total_loss / num_batches,
        "extent_loss": total_extent_loss / num_batches,
        "boundary_loss": total_boundary_loss / num_batches
    }


# ==============================================================================
# VALIDATE ONE EPOCH
# ==============================================================================
def validate(model, dataloader, criterion, device):
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

            metrics = calculate_metrics(extent_logits, masks)
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
# MAIN TRAINING PIPELINE FUNCTION
# ==============================================================================
def run_training(batch_size=2, num_epochs=2, lr=1e-4, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("STARTING CTHBNET FIRST TRAINING RUN")
    print("=" * 70)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Datasets & Loaders
    train_dataset = CroplandDataset(split="train")
    val_dataset = CroplandDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model, Loss, Optimizer
    model = CTHBNet(in_bands=5, timesteps=6).to(device)
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 3. Print Pre-Training Diagnostics
    total_params = sum(p.numel() for p in model.parameters())
    first_images, first_masks = next(iter(train_loader))

    print("\nPRE-TRAINING DIAGNOSTICS:")
    print(f"   Train Dataset Size : {len(train_dataset)} samples")
    print(f"   Val Dataset Size   : {len(val_dataset)} samples")
    print(f"   Device             : {device}")
    print(f"   Batch Size         : {batch_size}")
    print(f"   Total Parameters   : {total_params:,}")
    print(f"   First Image Batch  : {list(first_images.shape)}")
    print(f"   First Mask Batch   : {list(first_masks.shape)}")
    print("=" * 70)

    best_iou = 0.0
    start_total_time = time.time()
    epoch_times = []

    print("\nTRAINING PROGRESS:")
    for epoch in range(1, num_epochs + 1):
        start_epoch_t = time.time()

        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed_epoch = time.time() - start_epoch_t
        epoch_times.append(elapsed_epoch)

        print("-" * 70)
        print(f"EPOCH [{epoch:02d}/{num_epochs:02d}] (Time: {elapsed_epoch:.2f}s | {elapsed_epoch/60:.2f} min)")
        print(f"  Train -> Total Loss: {train_res['loss']:.4f} | Extent Loss: {train_res['extent_loss']:.4f} | Boundary Loss: {train_res['boundary_loss']:.4f}")
        print(f"  Val   -> Total Loss: {val_res['loss']:.4f} | Extent Loss: {val_res['extent_loss']:.4f} | Boundary Loss: {val_res['boundary_loss']:.4f}")
        print(f"  Val Metrics -> mIoU: {val_res['iou']:.4f} | F1/Dice: {val_res['dice']:.4f} | Precision: {val_res['precision']:.4f} | Recall: {val_res['recall']:.4f}")

        # Checkpoints
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": val_res["iou"],
            "val_dice": val_res["dice"],
            "val_loss": val_res["loss"]
        }

        # Save Latest Checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_model.pth"))

        # Save Best Model Checkpoint
        if val_res["iou"] > best_iou:
            best_iou = val_res["iou"]
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"  [*] BEST MODEL SAVED -> mIoU: {best_iou:.4f}")

    # Save Final Checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, "final_model.pth"))
    print("\n  [*] FINAL MODEL SAVED to checkpoints/final_model.pth")

    total_training_time = time.time() - start_total_time
    avg_epoch_time = np.mean(epoch_times)

    print("\n" * 70)
    print("=" * 70)
    print("FIRST REAL TRAINING RUN COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Total Training Time         : {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Average Time per Epoch      : {avg_epoch_time:.2f}s ({avg_epoch_time/60:.2f} minutes)")
    print(f"Estimated 20 Epochs Time    : {(avg_epoch_time * 20)/60:.2f} minutes ({(avg_epoch_time * 20)/3600:.2f} hours)")
    print(f"Estimated 30 Epochs Time    : {(avg_epoch_time * 30)/60:.2f} minutes ({(avg_epoch_time * 30)/3600:.2f} hours)")
    print(f"Estimated 50 Epochs Time    : {(avg_epoch_time * 50)/60:.2f} minutes ({(avg_epoch_time * 50)/3600:.2f} hours)")
    print("=" * 70)


if __name__ == "__main__":
    run_training(batch_size=2, num_epochs=20, lr=1e-4)
