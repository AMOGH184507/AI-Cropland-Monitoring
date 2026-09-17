import torch
import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import CTHBNet
from loss import MultiTaskLoss
from dataset_loader import CroplandDataset, DataLoader
from train import calculate_metrics


def test_single_batch_training_pipeline():
    print("=" * 70)
    print("SINGLE BATCH TRAINING PIPELINE VERIFICATION")
    print("=" * 70)

    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device               : {device}")
    print(f"Batch Size           : {batch_size}")

    # 1. Model Instantiation
    model = CTHBNet(in_bands=5, timesteps=6).to(device)
    model.train()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters     : {total_params:,}")

    # 2. Data Loaders
    train_dataset = CroplandDataset(split="train")
    val_dataset = CroplandDataset(split="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    images, masks = next(iter(train_loader))
    images, masks = images.to(device), masks.to(device)
    
    print(f"Train Batch Input    : {list(images.shape)}")
    print(f"Train Mask Target    : {list(masks.shape)}")

    # 3. Forward Pass Timing
    start_t = time.time()
    extent_logits, boundary_logits = model(images)
    fwd_time_ms = (time.time() - start_t) * 1000
    print(f"Forward Pass Time    : {fwd_time_ms:.2f} ms")

    print(f"Extent Logits Shape  : {list(extent_logits.shape)}")
    print(f"Boundary Logits Shape: {list(boundary_logits.shape)}")

    assert list(extent_logits.shape) == [batch_size, 1, 256, 256]
    assert list(boundary_logits.shape) == [batch_size, 1, 256, 256]

    # 4. Multi-Task Loss Calculation
    criterion = MultiTaskLoss().to(device)
    loss, loss_dict = criterion(extent_logits, boundary_logits, masks)
    
    print(f"Extent Loss Value    : {loss_dict['loss_extent']:.6f}")
    print(f"Boundary Loss Value  : {loss_dict['loss_boundary']:.6f}")
    print(f"Total Loss Value     : {loss_dict['loss_total']:.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    # 5. Backward Pass & Optimizer Step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer.zero_grad()
    loss.backward()
    
    param_has_grad = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in model.parameters())
    assert param_has_grad, "No parameter gradients calculated!"
    print("Backward Pass        : SUCCESSFUL")

    optimizer.step()
    print("Optimizer Step       : SUCCESSFUL")

    # 6. Single Validation Step
    model.eval()
    val_images, val_masks = next(iter(val_loader))
    val_images, val_masks = val_images.to(device), val_masks.to(device)

    with torch.no_grad():
        val_extent, val_boundary = model(val_images)
        val_loss, val_loss_dict = criterion(val_extent, val_boundary, val_masks)
        metrics = calculate_metrics(val_extent, val_masks)

    print(f"Validation Total Loss: {val_loss_dict['loss_total']:.6f}")
    print(f"Validation mIoU      : {metrics['iou']:.4f}")
    print(f"Validation F1/Dice   : {metrics['dice']:.4f}")
    print(f"Validation Precision : {metrics['precision']:.4f}")
    print(f"Validation Recall    : {metrics['recall']:.4f}")

    print("\n" + "=" * 40)
    print("TRAINING PIPELINE TEST PASSED")
    print("=" * 40)


if __name__ == "__main__":
    test_single_batch_training_pipeline()
