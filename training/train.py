import csv
import random
import torch
import numpy as np

from pathlib import Path

from preprocessing.dataset import CroplandDataset
from model.cthbnet import CTHBNet
from loss import MultiTaskLoss

from training.config import (
    RANDOM_SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    DEVICE,
    BEST_MODEL_PATH,
    MODEL_SUMMARY_PATH,
    TRAINING_HISTORY_CSV,
    NUM_WORKERS
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODEL SUMMARY
# ============================================================

def generate_model_summary(
    model: torch.nn.Module,
    train_count: int,
    val_count: int,
    test_count: int
) -> None:

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    summary_text = f"""
======================================================================
AI CROPLAND MONITORING - MODEL & EXPERIMENT SUMMARY
======================================================================

Architecture Name     : CTHBNet
Architecture          : CNN + Transformer + Boundary Guidance

Input Shape           : [Batch, 5, 6, 256, 256]
                        5 channels:
                        B2, B3, B4, B8, NDVI
                        6 temporal observations

Field Output Shape    : [Batch, 1, 256, 256]
Boundary Output Shape : [Batch, 1, 256, 256]

Encoder:
- Early Fusion Stem
- CNN Residual Blocks
- Spatial Transformer Blocks
- Multi-scale feature extraction

Boundary Guidance:
- Boundary-Guided Feature Enhancement (BGFE)

Decoder:
- Hierarchical Information Fusion Decoder

Prediction Heads:
- Parcel Extent Head
- Boundary Prediction Head

Parameter Breakdown:
- Total Parameters     : {total_params:,}
- Trainable Parameters : {trainable_params:,}

Training Configuration:
- Device Used          : {DEVICE}
- Random Seed          : {RANDOM_SEED}
- Batch Size           : {BATCH_SIZE}
- Learning Rate        : {LEARNING_RATE}
- Optimizer            : AdamW
- Weight Decay         : {WEIGHT_DECAY}
- Epochs               : {EPOCHS}

Dataset Counts:
- Train Split          : {train_count}
- Validation Split     : {val_count}
- Test Split           : {test_count}

Loss Functions:
- Extent Loss          : BCE + Dice
- Boundary Loss        : BCE + Focal
- Total Loss           : Extent Loss + Boundary Loss

Boundary Target:
- 3x3 Morphological Gradient

======================================================================
"""

    MODEL_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(MODEL_SUMMARY_PATH, "w") as f:
        f.write(summary_text)

    print(
        f"Model summary written to: "
        f"{MODEL_SUMMARY_PATH}"
    )


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_pipeline(
    epochs: int = EPOCHS
) -> torch.nn.Module:

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("STAGE 6: CTHBNet MODEL TRAINING")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Batch Size : {BATCH_SIZE}")
    print(f"Epochs     : {epochs}")

    # ========================================================
    # DATASETS
    # ========================================================

    train_ds = CroplandDataset(
        split="train"
    )

    val_ds = CroplandDataset(
        split="val"
    )

    test_ds = CroplandDataset(
        split="test"
    )

    print(
        f"Dataset Counts -> "
        f"Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}, "
        f"Test: {len(test_ds)}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = CTHBNet(
        in_bands=5,
        timesteps=6
    ).to(DEVICE)

    print()
    print("Model: CTHBNet")
    print("Input: [B, 5, 6, 256, 256]")

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    generate_model_summary(
        model,
        len(train_ds),
        len(val_ds),
        len(test_ds)
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = MultiTaskLoss(
        w_extent=1.0,
        w_boundary=1.0
    ).to(DEVICE)

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_val_loss = float("inf")

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        train_total_loss = 0.0
        train_extent_loss = 0.0
        train_boundary_loss = 0.0

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for batch_idx, (
            imgs,
            masks,
            fids
        ) in enumerate(train_loader):

            imgs = imgs.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # Forward pass
            extent_logits, boundary_logits = model(
                imgs
            )

            # Multi-task loss
            total_loss, loss_dict = criterion(
                extent_logits,
                boundary_logits,
                masks
            )

            # Backpropagation
            total_loss.backward()

            optimizer.step()

            train_total_loss += loss_dict[
                "loss_total"
            ]

            train_extent_loss += loss_dict[
                "loss_extent"
            ]

            train_boundary_loss += loss_dict[
                "loss_boundary"
            ]

        # ----------------------------------------------------
        # AVERAGE TRAINING LOSSES
        # ----------------------------------------------------

        avg_train_total = (
            train_total_loss /
            max(len(train_loader), 1)
        )

        avg_train_extent = (
            train_extent_loss /
            max(len(train_loader), 1)
        )

        avg_train_boundary = (
            train_boundary_loss /
            max(len(train_loader), 1)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_total_loss = 0.0
        val_extent_loss = 0.0
        val_boundary_loss = 0.0

        with torch.no_grad():

            for (
                imgs,
                masks,
                fids
            ) in val_loader:

                imgs = imgs.to(
                    DEVICE,
                    non_blocking=True
                )

                masks = masks.to(
                    DEVICE,
                    non_blocking=True
                )

                # Forward pass
                extent_logits, boundary_logits = model(
                    imgs
                )

                # Loss
                total_loss, loss_dict = criterion(
                    extent_logits,
                    boundary_logits,
                    masks
                )

                val_total_loss += loss_dict[
                    "loss_total"
                ]

                val_extent_loss += loss_dict[
                    "loss_extent"
                ]

                val_boundary_loss += loss_dict[
                    "loss_boundary"
                ]

        # ----------------------------------------------------
        # AVERAGE VALIDATION LOSSES
        # ----------------------------------------------------

        avg_val_total = (
            val_total_loss /
            max(len(val_loader), 1)
        )

        avg_val_extent = (
            val_extent_loss /
            max(len(val_loader), 1)
        )

        avg_val_boundary = (
            val_boundary_loss /
            max(len(val_loader), 1)
        )

        # ====================================================
        # PRINT EPOCH RESULTS
        # ====================================================

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_total:.4f} "
            f"(Extent: {avg_train_extent:.4f}, "
            f"Boundary: {avg_train_boundary:.4f}) | "
            f"Val Loss: {avg_val_total:.4f} "
            f"(Extent: {avg_val_extent:.4f}, "
            f"Boundary: {avg_val_boundary:.4f})"
        )

        # ====================================================
        # HISTORY
        # ====================================================

        history.append({
            "epoch": epoch,
            "train_total_loss": avg_train_total,
            "train_extent_loss": avg_train_extent,
            "train_boundary_loss": avg_train_boundary,
            "val_total_loss": avg_val_total,
            "val_extent_loss": avg_val_extent,
            "val_boundary_loss": avg_val_boundary
        })

        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if avg_val_total < best_val_loss:

            best_val_loss = avg_val_total

            BEST_MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss
                },
                BEST_MODEL_PATH
            )

            print(
                f"  [CHECKPOINT] "
                f"Saved best model to "
                f"{BEST_MODEL_PATH}"
            )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    TRAINING_HISTORY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        TRAINING_HISTORY_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(history)

    # ========================================================
    # COMPLETE
    # ========================================================

    print("=" * 70)

    print(
        f"[SUCCESS] CTHBNet Training Completed!"
    )

    print(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}"
    )

    print("=" * 70)

    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_pipeline(
        epochs=EPOCHS
    )