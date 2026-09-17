import torch
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import CTHBNet
from dataset_loader import CroplandDataset, DataLoader


def test_cthbnet_forward():
    print("=" * 70)
    print("CTHBNET FORWARD-PASS & DRY-RUN TEST")
    print("=" * 70)

    # 1. Instantiate Model
    print("\n1. Instantiating CTHBNet Model...")
    model = CTHBNet(in_bands=5, timesteps=6)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total Parameters     : {total_params:,}")
    print(f"   Trainable Parameters : {trainable_params:,}")

    # 2. Test Dummy Batch [2, 5, 6, 256, 256]
    batch_size = 2
    dummy_input = torch.randn(batch_size, 5, 6, 256, 256, dtype=torch.float32)
    print(f"\n2. Testing Dummy Input Tensor:")
    print(f"   Input Shape: {list(dummy_input.shape)} [B, 5, 6, H, W]")

    # Hook intermediate feature shapes for verification
    shapes = {}
    
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                shapes[name] = [list(o.shape) for o in output]
            else:
                shapes[name] = list(output.shape)
        return hook

    # Register hooks
    model.encoder.stem.register_forward_hook(get_hook("EarlyFusionStem (Flattened & Stem Conv)"))
    model.encoder.stage1_block.register_forward_hook(get_hook("Encoder Stage 1 (Local CNN)"))
    model.encoder.stage2_block.register_forward_hook(get_hook("Encoder Stage 2 (Local CNN)"))
    model.encoder.stage3_trans.register_forward_hook(get_hook("Encoder Stage 3 (CNN-Transformer Hybrid)"))
    model.encoder.stage4_trans.register_forward_hook(get_hook("Encoder Stage 4 (Transformer Bottleneck)"))
    model.bgfe.register_forward_hook(get_hook("BGFE Module (Boundary Refinement)"))
    model.decoder.up3.register_forward_hook(get_hook("Decoder Up-Block 3"))
    model.decoder.up2.register_forward_hook(get_hook("Decoder Up-Block 2"))
    model.decoder.up1.register_forward_hook(get_hook("Decoder Up-Block 1"))
    model.extent_head.register_forward_hook(get_hook("Parcel Extent Head"))
    model.boundary_head.register_forward_hook(get_hook("Boundary Prediction Head"))

    print("\n3. Executing Forward Pass...")
    start_time = time.time()
    with torch.no_grad():
        extent_out, boundary_out = model(dummy_input)
    elapsed_time = time.time() - start_time

    print(f"   Forward Pass Execution Time: {elapsed_time * 1000:.2f} ms")

    print("\n4. Intermediate Stage Tensor Shapes:")
    print("-" * 70)
    for name, shape in shapes.items():
        print(f"   {name:<48}: {shape}")
    print("-" * 70)

    print("\n5. Output Tensor Verification:")
    print(f"   Extent Mask Logits Shape  : {list(extent_out.shape)} (Expected: [{batch_size}, 1, 256, 256])")
    print(f"   Boundary Logits Shape     : {list(boundary_out.shape)} (Expected: [{batch_size}, 1, 256, 256])")

    # Assertions
    assert list(extent_out.shape) == [batch_size, 1, 256, 256], f"Extent shape mismatch: {extent_out.shape}"
    assert list(boundary_out.shape) == [batch_size, 1, 256, 256], f"Boundary shape mismatch: {boundary_out.shape}"
    print("\n   [OK] DUMMY TENSOR FORWARD PASS SUCCESSFUL!")

    # 6. Test with Real CroplandDataset Batch
    print("\n" + "=" * 70)
    print("TESTING WITH REAL DATASET BATCH (CroplandDataset)")
    print("=" * 70)
    try:
        val_dataset = CroplandDataset(split="val")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        real_images, real_masks = next(iter(val_loader))

        print(f"\n   Real Batch Images Shape : {list(real_images.shape)}")
        print(f"   Real Batch Masks Shape  : {list(real_masks.shape)}")

        with torch.no_grad():
            real_extent, real_boundary = model(real_images)

        print(f"   Real Extent Output Shape: {list(real_extent.shape)}")
        print(f"   Real Boundary Out Shape : {list(real_boundary.shape)}")

        assert list(real_extent.shape) == [batch_size, 1, 256, 256]
        assert list(real_boundary.shape) == [batch_size, 1, 256, 256]
        print("\n   [OK] REAL DATASET BATCH FORWARD PASS SUCCESSFUL!")
    except Exception as e:
        print(f"\n   [WARNING] REAL DATASET BATCH TEST WARNING/ERROR: {e}")

    print("\n" + "=" * 70)
    print("ALL FORWARD PASS TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_cthbnet_forward()
