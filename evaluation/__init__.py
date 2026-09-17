"""
AI Cropland Monitoring Evaluation Package
==========================================
Exports:
- compute_binary_metrics: Function computing IoU, Dice, Precision, Recall, Accuracy
- evaluate_loader: Evaluation over PyTorch DataLoader
- generate_prediction_visualizations: PNG comparison visualizer
- run_evaluation: Main evaluation entrypoint
"""

from evaluation.metrics import compute_binary_metrics
from evaluation.evaluate import evaluate_loader, generate_prediction_visualizations, run_evaluation

__all__ = [
    "compute_binary_metrics",
    "evaluate_loader",
    "generate_prediction_visualizations",
    "run_evaluation"
]
