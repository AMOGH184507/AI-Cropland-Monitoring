import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# BOUNDARY GROUND-TRUTH GENERATOR (3x3 Morphological Gradient)
# ==============================================================================
def generate_boundary_target(mask: torch.Tensor) -> torch.Tensor:
    """
    Generates binary field parcel boundary target from ground truth mask
    using a 3x3 morphological gradient: boundary = dilation(mask) - erosion(mask).
    
    Args:
        mask: Tensor of shape [B, H, W] or [B, 1, H, W]
    Returns:
        boundary: Tensor of shape [B, 1, H, W] with values in {0.0, 1.0}
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B, 1, H, W]
        
    mask_float = (mask > 0.5).float()
    
    # Dilation using 3x3 Max Pooling
    dilation = F.max_pool2d(mask_float, kernel_size=3, stride=1, padding=1)
    
    # Erosion using Negative Max Pooling
    erosion = -F.max_pool2d(-mask_float, kernel_size=3, stride=1, padding=1)
    
    # Morphological Gradient
    boundary = dilation - erosion
    boundary = (boundary > 0.5).float()
    
    return boundary


# ==============================================================================
# DICE LOSS
# ==============================================================================
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


# ==============================================================================
# FOCAL LOSS FOR BOUNDARY PREDICTION
# ==============================================================================
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for handling boundary pixel class imbalance.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_factor * ((1.0 - p_t) ** self.gamma)
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ==============================================================================
# EXTENT LOSS (BCE + DICE)
# ==============================================================================
class ExtentLoss(nn.Module):
    """
    Combined BCE + Dice Loss for Parcel Extent segmentation.
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ==============================================================================
# BOUNDARY LOSS (BCE + FOCAL)
# ==============================================================================
class BoundaryLoss(nn.Module):
    """
    Combined BCE + Focal Loss for Field Boundary edge detection.
    """
    def __init__(self, bce_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = BinaryFocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss


# ==============================================================================
# MULTI-TASK LOSS (COMBINED EXTENT + BOUNDARY)
# ==============================================================================
class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss for CTHBNet combining Parcel Extent and Boundary Loss:
    L_total = 1.0 * L_extent + 1.0 * L_boundary
    """
    def __init__(self, w_extent=1.0, w_boundary=1.0):
        super().__init__()
        self.w_extent = w_extent
        self.w_boundary = w_boundary
        
        self.extent_loss_fn = ExtentLoss()
        self.boundary_loss_fn = BoundaryLoss()

    def forward(
        self, 
        extent_logits: torch.Tensor, 
        boundary_logits: torch.Tensor, 
        target_mask: torch.Tensor
    ):
        """
        Args:
            extent_logits: Tensor [B, 1, H, W]
            boundary_logits: Tensor [B, 1, H, W]
            target_mask: Tensor [B, H, W] or [B, 1, H, W]
        Returns:
            total_loss: Scalar PyTorch tensor loss
            loss_dict: Dictionary containing detailed breakdown
        """
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(1)  # [B, 1, H, W]
            
        target_mask = (target_mask > 0.5).float()
        
        # 1. Generate Boundary Target dynamically via 3x3 Morphological Gradient
        boundary_target = generate_boundary_target(target_mask)
        
        # 2. Compute Individual Losses
        l_extent = self.extent_loss_fn(extent_logits, target_mask)
        l_boundary = self.boundary_loss_fn(boundary_logits, boundary_target)
        
        # 3. Combined Multi-Task Loss
        l_total = self.w_extent * l_extent + self.w_boundary * l_boundary
        
        loss_dict = {
            "loss_extent": l_extent.item(),
            "loss_boundary": l_boundary.item(),
            "loss_total": l_total.item()
        }
        
        return l_total, loss_dict
