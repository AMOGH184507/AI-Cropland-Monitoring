"""
AI Cropland Monitoring Models Package
=====================================
Exports:
- CroplandBoundaryModel: Complete CNN-Transformer Boundary Guidance architecture
- BaselineUNet: Baseline 4-channel UNet model
- SpatialTransformer: Global spatial context multi-head self-attention module
- BoundaryGuidanceModule: Spatial attention feature gating module
- generate_boundary_map_numpy, generate_boundary_map_torch, create_boundary_visualization
"""

from models.model import CroplandBoundaryModel, CNNEncoder
from models.baseline_cnn import BaselineUNet
from models.transformer import SpatialTransformer
from models.boundary_guidance import BoundaryGuidanceModule
from models.boundary_generator import (
    generate_boundary_map_numpy,
    generate_boundary_map_torch,
    create_boundary_visualization
)

__all__ = [
    "CroplandBoundaryModel",
    "CNNEncoder",
    "BaselineUNet",
    "SpatialTransformer",
    "BoundaryGuidanceModule",
    "generate_boundary_map_numpy",
    "generate_boundary_map_torch",
    "create_boundary_visualization"
]
