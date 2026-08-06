"""DriveWorld: Modular World Model Framework for Autonomous Driving.

DriveWorld provides a unified training and evaluation framework for
autonomous driving world models, supporting multiple paradigms:

- **OccWorld**: 3D occupancy prediction via autoregressive transformers
- **DriveDiffuser**: Diffusion-based future state generation
- **Hybrid**: Combined occupancy + diffusion for robust predictions

Key Features:
    - Pluggable encoder/decoder architectures (BEV, Voxel, Transformer)
    - Multi-paradigm world model training with unified config system
    - Rich evaluation: IoU, Video Prediction metrics, 3D rendering
    - Production-ready: CI/CD, Docker, type hints, tests
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from driveworld.utils.config import Config, load_config
from driveworld.models import OccWorld, DriveDiffuser, build_encoder, build_decoder
from driveworld.data import NuScenesWorldModelDataset, create_dataloader
from driveworld.training import WorldModelTrainer

__all__ = [
    "Config",
    "load_config",
    "OccWorld",
    "DriveDiffuser",
    "build_encoder",
    "build_decoder",
    "NuScenesWorldModelDataset",
    "create_dataloader",
    "WorldModelTrainer",
]
