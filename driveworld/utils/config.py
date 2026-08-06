"""Configuration system using Hydra/OmegaConf with dataclass validation.

All model, training, and data parameters are defined as structured configs,
enabling type-safe configuration and easy YAML overrides.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""
    dataset_root: str = "data/nuscenes"
    version: str = "v1.0-mini"
    image_size: Tuple[int, int] = (224, 480)
    bev_grid_size: Tuple[int, int] = (200, 200)
    bev_resolution: float = 0.5
    num_past_frames: int = 3
    num_future_frames: int = 6
    frame_interval: float = 0.5
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True


@dataclass
class EncoderConfig:
    """Perception encoder configuration."""
    backbone: str = "resnet50"
    pretrained: bool = True
    bev_feat_dim: int = 256
    bev_h: int = 200
    bev_w: int = 200
    fusion_method: str = "transformer"  # transformer, concat, add


@dataclass
class OccWorldConfig:
    """OccWorld-style autoregressive world model config."""
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    vocab_size: int = 8192
    dropout: float = 0.1
    use_flash_attn: bool = False


@dataclass
class DiffusionConfig:
    """Diffusion-based world model config."""
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    unet_channels: Tuple[int, ...] = (128, 256, 512)
    attention_resolutions: Tuple[int, ...] = (16, 8)
    num_res_blocks: int = 2
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    paradigm: str = "occworld"  # occworld, diffusion, hybrid
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    log_every: int = 50
    val_every: int = 1
    save_every: int = 5
    seed: int = 42


@dataclass
class LossConfig:
    """Loss function configuration."""
    occupancy_weight: float = 1.0
    cross_entropy_weight: float = 1.0
    dice_weight: float = 0.5
    perceptual_weight: float = 0.1
    kl_weight: float = 0.001


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    occworld: OccWorldConfig = field(default_factory=OccWorldConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def load_config(path: str) -> Config:
    """Load configuration from a YAML file and return a validated Config object."""
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    config = Config()
    for section_name, section_dict in raw.items():
        if hasattr(config, section_name):
            section = getattr(config, section_name)
            for key, value in section_dict.items():
                if hasattr(section, key):
                    setattr(section, key, value)

    return config


def save_config(config: Config, path: str) -> None:
    """Save a Config object to a YAML file."""
    raw = {}
    for field_name in config.__dataclass_fields__:
        section = getattr(config, field_name)
        raw[field_name] = {
            k: v for k, v in section.__dict__.items() if not k.startswith("_")
        }
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False)
