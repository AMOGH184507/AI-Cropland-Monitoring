import torch
import numpy as np

from dataset_loader import CroplandDataset
from model.cthbnet import CTHBNet

# --------------------------------------------------
# Setup
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("THRESHOLD ANALYSIS ON TEST DATASET")
print("=" * 70)
print("Device:", device)

dataset = CroplandDataset("test")

model = CTHBNet().to(device)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

results = {
    t: {"intersection": 0, "union": 0, "pred": 0, "gt": 0}
    for t in thresholds
}

# --------------------------------------------------
# Evaluate entire test set
# --------------------------------------------------
with torch.no_grad():

    for i in range(len(dataset)):

        image, mask = dataset[i]

        x = image.unsqueeze(0).to(device)

        output = model(x)

        if isinstance(output, dict):
            logits = output["extent_logits"]
        else:
            logits = output[0]

        probability = torch.sigmoid(logits)

        probability = probability.squeeze().cpu().numpy()
        gt = mask.numpy() > 0.5

        for threshold in thresholds:

            pred = probability > threshold

            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()

            results[threshold]["intersection"] += intersection
            results[threshold]["union"] += union
            results[threshold]["pred"] += pred.sum()
            results[threshold]["gt"] += gt.sum()

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

# --------------------------------------------------
# Print results
# --------------------------------------------------
print()
print("=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

print(
    f"{'Threshold':<12}"
    f"{'IoU':<12}"
    f"{'Dice':<12}"
    f"{'Pred %':<12}"
    f"{'GT %':<12}"
)

total_pixels = len(dataset) * 256 * 256

for threshold in thresholds:

    r = results[threshold]

    intersection = r["intersection"]
    union = r["union"]

    iou = intersection / (union + 1e-8)

    dice = (
        2 * intersection /
        (r["pred"] + r["gt"] + 1e-8)
    )

    pred_percent = r["pred"] / total_pixels * 100
    gt_percent = r["gt"] / total_pixels * 100

    print(
        f"{threshold:<12}"
        f"{iou:<12.4f}"
        f"{dice:<12.4f}"
        f"{pred_percent:<12.2f}"
        f"{gt_percent:<12.2f}"
    )

print("=" * 70)