"""
AI Cropland Monitoring Training Package
=======================================
Exports:
- train_pipeline: Function running model training and saving checkpoints
- BATCH_SIZE, LEARNING_RATE, EPOCHS, BOUNDARY_LOSS_WEIGHT, DEVICE
"""

from training.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    BOUNDARY_LOSS_WEIGHT,
    DEVICE,
    BEST_MODEL_PATH
)
from training.train import train_pipeline, set_seed

__all__ = [
    "train_pipeline",
    "set_seed",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "EPOCHS",
    "BOUNDARY_LOSS_WEIGHT",
    "DEVICE",
    "BEST_MODEL_PATH"
]
