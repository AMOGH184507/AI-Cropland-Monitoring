import torch
import numpy as np
from typing import Dict


def compute_binary_metrics(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes standard segmentation evaluation metrics:
    - IoU (Jaccard Index)
    - Dice (F1 Score)
    - Precision
    - Recall
    - Accuracy

    Args:
        preds: Raw logits or probabilities array/tensor
        targets: Ground truth binary targets array/tensor {0, 1}
        threshold: Decision threshold (default 0.5)

    Returns:
        dict with calculated metrics
    """
    if isinstance(preds, torch.Tensor):
        if preds.requires_grad:
            preds = preds.detach()
        if (preds < 0).any() or (preds > 1).any():
            preds = torch.sigmoid(preds)
        preds_bin = (preds > threshold).byte()
        targets_bin = (targets > threshold).byte()

        intersection = (preds_bin & targets_bin).float().sum().item()
        union = (preds_bin | targets_bin).float().sum().item()
        tp = intersection
        fp = (preds_bin & (~targets_bin)).float().sum().item()
        fn = ((~preds_bin) & targets_bin).float().sum().item()
        tn = ((~preds_bin) & (~targets_bin)).float().sum().item()
        total = targets_bin.numel()
    else:
        preds_arr = np.asarray(preds)
        targets_arr = np.asarray(targets)

        if np.min(preds_arr) < 0 or np.max(preds_arr) > 1:
            preds_arr = 1.0 / (1.0 + np.exp(-preds_arr))

        preds_bin = (preds_arr > threshold).astype(bool)
        targets_bin = (targets_arr > threshold).astype(bool)

        tp = np.logical_and(preds_bin, targets_bin).sum()
        fp = np.logical_and(preds_bin, ~targets_bin).sum()
        fn = np.logical_and(~preds_bin, targets_bin).sum()
        tn = np.logical_and(~preds_bin, ~targets_bin).sum()
        union = tp + fp + fn
        total = targets_bin.size

    iou = (tp + eps) / (union + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / max(total, 1)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy)
    }


from typing import Union

if __name__ == "__main__":
    print("Testing Evaluation Metrics...")
    dummy_pred = torch.randn(4, 1, 256, 256)
    dummy_target = torch.randint(0, 2, (4, 1, 256, 256)).float()

    m = compute_binary_metrics(dummy_pred, dummy_target)
    for k, v in m.items():
        print(f"{k.capitalize():10s}: {v:.4f}")
