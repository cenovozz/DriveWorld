"""Data augmentation and preprocessing transforms for autonomous driving data."""

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        for t in self.transforms:
            data = t(data)
        return data


class ResizeImages:
    """Resize all camera images to a fixed size."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, data: dict) -> dict:
        images = data["past_images"]  # (T, C, H, W)
        resized = []
        for t in range(images.shape[0]):
            img = F.resize(images[t], self.size, antialias=True)
            resized.append(img)
        data["past_images"] = torch.stack(resized)
        return data


class NormalizeImages:
    """Normalize images with given mean and std."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, data: dict) -> dict:
        images = data["past_images"]
        normalized = []
        for t in range(images.shape[0]):
            img = F.normalize(images[t], self.mean, self.std)
            normalized.append(img)
        data["past_images"] = torch.stack(normalized)
        return data


class RandomHorizontalFlip:
    """Randomly flip all images and corresponding targets horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data: dict) -> dict:
        if random.random() < self.p:
            images = data["past_images"]
            data["past_images"] = torch.flip(images, dims=[-1])
            if "future_occupancy" in data:
                data["future_occupancy"] = torch.flip(
                    data["future_occupancy"], dims=[-1]
                )
            if "past_ego_pose" in data:
                pose = data["past_ego_pose"]
                pose[..., 1] *= -1  # flip y coordinate
                data["past_ego_pose"] = pose
            if "future_ego_pose" in data:
                pose = data["future_ego_pose"]
                pose[..., 1] *= -1
                data["future_ego_pose"] = pose
        return data


def build_transforms(config) -> Compose:
    """Build transform pipeline from config."""
    transforms = [
        ResizeImages(config.data.image_size),
        NormalizeImages(),
    ]
    if config.data.augment:
        transforms.insert(0, RandomHorizontalFlip(p=0.5))
    return Compose(transforms)
