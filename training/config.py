import torch
from pathlib import Path


# ============================================================
# PROJECT ROOT & OUTPUT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
METRICS_DIR = OUTPUTS_DIR / "metrics"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

PRED_VIS_DIR = VISUALIZATIONS_DIR / "predictions"
BOUNDARIES_VIS_DIR = VISUALIZATIONS_DIR / "boundaries"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PRED_VIS_DIR.mkdir(parents=True, exist_ok=True)
BOUNDARIES_VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42


# ============================================================
# MODEL INPUT
# ============================================================

# Sentinel-2:
# 5 channels = B4, B3, B2, B8, NDVI
# 6 temporal observations

IN_BANDS = 5
TIMESTEPS = 6


# Training Hyperparameters
RANDOM_SEED = 42

BATCH_SIZE = 2

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

EPOCHS = 20

NUM_WORKERS = 0

# Kept for compatibility/documentation.
# MultiTaskLoss currently uses w_boundary=1.0.
BOUNDARY_LOSS_WEIGHT = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECKPOINTS
# ============================================================

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"

BASELINE_MODEL_PATH = CHECKPOINTS_DIR / "baseline_model.pth"


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_SUMMARY_PATH = OUTPUTS_DIR / "model_summary.txt"

TRAINING_HISTORY_CSV = (
    METRICS_DIR / "training_history.csv"
)

VAL_METRICS_JSON = (
    METRICS_DIR / "validation_metrics.json"
)

TEST_METRICS_JSON = (
    METRICS_DIR / "test_metrics.json"
)