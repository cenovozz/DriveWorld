"""Data loading utilities."""

import random
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def _seed_worker(worker_id: int, base_seed: int) -> None:
    random.seed(base_seed + worker_id)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn: Optional[callable] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for world model training."""
    if collate_fn is None:
        collate_fn = collate_world_model

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = lambda worker_id: _seed_worker(worker_id, seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


def collate_world_model(batch) -> Tuple[torch.Tensor, ...]:
    """Custom collate function for world model batches.

    Each sample is a dict with keys:
        past_images: (T_past, C, H, W)
        past_ego_pose: (T_past, 3)  -- x, y, yaw
        future_occupancy: (T_future, Z, H, W)  -- 3D occupancy grid
        future_ego_pose: (T_future, 3)
        token: str
    """
    past_images = torch.stack([s["past_images"] for s in batch])
    past_ego_pose = torch.stack([s["past_ego_pose"] for s in batch])
    future_occupancy = torch.stack([s["future_occupancy"] for s in batch])
    future_ego_pose = torch.stack([s["future_ego_pose"] for s in batch])
    tokens = [s["token"] for s in batch]
    return past_images, past_ego_pose, future_occupancy, future_ego_pose, tokens
