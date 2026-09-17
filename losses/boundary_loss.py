import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combines Binary Cross-Entropy with Logits (BCE) and Dice Loss.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class JointCroplandLoss(nn.Module):
    """
    Joint Loss function for Cropland Extent Segmentation & Boundary Prediction.

    Total Loss = Segmentation_Loss + λ * Boundary_Loss

    where:
    - Segmentation_Loss: BCE + Dice Loss between field extent prediction & ground-truth mask
    - Boundary_Loss: BCE + Dice Loss between boundary prediction & generated boundary map
    - λ (boundary_weight): Weight hyperparameter configured in training/config.py
    """
    def __init__(self, boundary_weight: float = 0.5):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.seg_loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        self.boundary_loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    def forward(
        self,
        field_logits: torch.Tensor,
        field_targets: torch.Tensor,
        boundary_logits: torch.Tensor,
        boundary_targets: torch.Tensor
    ) -> torch.Dict[str, torch.Tensor]:
        # Reshape targets to match (B, 1, H, W)
        if field_targets.ndim == 3:
            field_targets = field_targets.unsqueeze(1)
        if boundary_targets.ndim == 3:
            boundary_targets = boundary_targets.unsqueeze(1)

        seg_loss = self.seg_loss_fn(field_logits, field_targets)
        bound_loss = self.boundary_loss_fn(boundary_logits, boundary_targets)

        total_loss = seg_loss + self.boundary_weight * bound_loss

        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "boundary_loss": bound_loss
        }


if __name__ == "__main__":
    print("Testing JointCroplandLoss...")
    dummy_field_logits = torch.randn(2, 1, 256, 256, requires_grad=True)
    dummy_field_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()
    dummy_bound_logits = torch.randn(2, 1, 256, 256, requires_grad=True)
    dummy_bound_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()

    loss_fn = JointCroplandLoss(boundary_weight=0.5)
    loss_dict = loss_fn(dummy_field_logits, dummy_field_targets, dummy_bound_logits, dummy_bound_targets)

    print(f"Total Loss   : {loss_dict['total_loss'].item():.4f}")
    print(f"Seg Loss     : {loss_dict['seg_loss'].item():.4f}")
    print(f"Boundary Loss: {loss_dict['boundary_loss'].item():.4f}")

    loss_dict["total_loss"].backward()
    print("Backward pass successful!")
