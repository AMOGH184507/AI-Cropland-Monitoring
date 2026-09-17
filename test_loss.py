import torch
import sys
import math
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loss import generate_boundary_target, MultiTaskLoss
from model import CTHBNet
from dataset_loader import CroplandDataset, DataLoader


def test_loss_function():
    print("=" * 70)
    print("TESTING MULTI-TASK LOSS & BOUNDARY TARGET GENERATOR")
    print("=" * 70)

    batch_size = 2
    height, width = 256, 256

    # --------------------------------------------------------------------------
    # 1. TEST DUMMY DATA & BOUNDARY GENERATOR
    # --------------------------------------------------------------------------
    print("\n1. Testing Boundary Target Generator on Dummy Mask...")
    dummy_mask = torch.zeros(batch_size, height, width, dtype=torch.float32)
    # Add a square cropland parcel in the center
    dummy_mask[:, 50:200, 50:200] = 1.0

    boundary_target = generate_boundary_target(dummy_mask)

    print(f"   Input Mask Shape           : {list(dummy_mask.shape)}")
    print(f"   Generated Boundary Shape   : {list(boundary_target.shape)}")
    
    unique_vals = torch.unique(boundary_target).tolist()
    print(f"   Boundary Unique Values     : {unique_vals}")

    # Assertions
    assert list(boundary_target.shape) == [batch_size, 1, height, width], \
        f"Boundary target shape mismatch: {boundary_target.shape}"
    assert all(val in [0.0, 1.0] for val in unique_vals), \
        f"Boundary target contains non-binary values: {unique_vals}"
    print("   ✅ Boundary Target Generation Successful!")

    # --------------------------------------------------------------------------
    # 2. TEST MULTI-TASK LOSS ON DUMMY PREDICTIONS
    # --------------------------------------------------------------------------
    print("\n2. Testing MultiTaskLoss on Dummy Predictions...")
    criterion = MultiTaskLoss()
    
    dummy_extent_logits = torch.randn(batch_size, 1, height, width, requires_grad=True)
    dummy_boundary_logits = torch.randn(batch_size, 1, height, width, requires_grad=True)

    loss, loss_dict = criterion(dummy_extent_logits, dummy_boundary_logits, dummy_mask)

    print(f"   Extent Loss   : {loss_dict['loss_extent']:.6f}")
    print(f"   Boundary Loss : {loss_dict['loss_boundary']:.6f}")
    print(f"   Total Loss    : {loss_dict['loss_total']:.6f}")

    # Check finite
    assert not math.isnan(loss_dict['loss_extent']), "Extent loss is NaN"
    assert not math.isinf(loss_dict['loss_extent']), "Extent loss is Inf"
    assert not math.isnan(loss_dict['loss_boundary']), "Boundary loss is NaN"
    assert not math.isinf(loss_dict['loss_boundary']), "Boundary loss is Inf"
    assert not math.isnan(loss_dict['loss_total']), "Total loss is NaN"
    assert not math.isinf(loss_dict['loss_total']), "Total loss is Inf"
    print("   ✅ All Dummy Loss Values Are Finite!")

    # Test Backpropagation
    loss.backward()
    assert dummy_extent_logits.grad is not None, "Extent logits gradient is None"
    assert dummy_boundary_logits.grad is not None, "Boundary logits gradient is None"
    print("   ✅ Backpropagation Successful on Dummy Tensors!")

    # --------------------------------------------------------------------------
    # 3. TEST LOSS ON REAL DATASET BATCH & CTHBNET MODEL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TESTING LOSS WITH REAL DATASET BATCH & CTHBNET FORWARD/BACKWARD")
    print("=" * 70)

    train_dataset = CroplandDataset(split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    real_images, real_masks = next(iter(train_loader))

    print(f"\n   Real Images Shape : {list(real_images.shape)}")
    print(f"   Real Masks Shape  : {list(real_masks.shape)}")

    model = CTHBNet(in_bands=5, timesteps=6)
    model.train()

    extent_logits, boundary_logits = model(real_images)

    real_loss, real_loss_dict = criterion(extent_logits, boundary_logits, real_masks)

    print(f"   Real Extent Loss   : {real_loss_dict['loss_extent']:.6f}")
    print(f"   Real Boundary Loss : {real_loss_dict['loss_boundary']:.6f}")
    print(f"   Real Total Loss    : {real_loss_dict['loss_total']:.6f}")

    assert not math.isnan(real_loss_dict['loss_total']), "Real total loss is NaN"
    assert not math.isinf(real_loss_dict['loss_total']), "Real total loss is Inf"

    # Backward pass on real batch
    real_loss.backward()
    param_grad_check = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
    assert param_grad_check, "Model parameters received no non-zero gradients!"
    print("   ✅ Model Parameter Gradients Calculated Successfully!")

    print("\n" + "=" * 70)
    print("LOSS IMPLEMENTATION TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_loss_function()
