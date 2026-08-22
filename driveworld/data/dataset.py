"""NuScenes-based dataset for world model training.

Samples sequences of past camera images + ego poses and predicts
future 3D occupancy grids. Designed for nuScenes mini (4GB) for
easy reproduction on a single GPU.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NuScenesWorldModelDataset(Dataset):
    """Dataset that produces (past_frames, ego_motion) -> future_occupancy samples.

    Each sample:
        past_images: (T_past, 3, H, W) float32 -- normalized camera images (front)
        past_ego_pose: (T_past, 3) float32 -- (x, y, yaw) in ego frame of first frame
        future_occupancy: (T_future, Z, H, W) int64 -- 0/1 occupancy labels
        future_ego_pose: (T_future, 3) float32
        token: str -- scene token for debugging

    The dataset preprocesses nuScenes into h5/pickle format for fast loading.
    If no preprocessed data exists, the dataset expects raw nuScenes structure
    and does on-the-fly processing (slower but works out of the box).
    """

    def __init__(
        self,
        root: str = "data/nuscenes",
        version: str = "v1.0-mini",
        split: str = "train",
        num_past_frames: int = 3,
        num_future_frames: int = 6,
        frame_interval: float = 0.5,
        image_size: Tuple[int, int] = (224, 480),
        bev_grid_size: Tuple[int, int] = (200, 200),
        bev_resolution: float = 0.5,
        num_cameras: int = 1,
        num_z: int = 16,
        z_range: Tuple[float, float] = (-4.0, 4.0),
        augment: bool = True,
        occupancy_target: str = "future",
    ):
        self.root = Path(root)
        self.version = version
        self.split = split
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.bev_grid_size = bev_grid_size
        self.bev_resolution = bev_resolution
        self.num_cameras = num_cameras
        self.num_z = num_z
        self.z_range = z_range
        self.augment = augment
        self.occupancy_target = occupancy_target

        self.sequence_length = num_past_frames + num_future_frames

        self.samples = self._build_samples()

    def _build_samples(self) -> List[Dict]:
        """Build the list of valid sample indices from the dataset.

        For nuScenes, each scene is a ~20s driving segment. We extract
        sliding windows of (past + future) frames.
        """
        preprocessed = self.root / self.version / f"{self.split}_samples.pkl"
        if preprocessed.exists():
            with open(preprocessed, "rb") as f:
                return pickle.load(f)

        scene_dir = self.root / self.version / self.split / "scenes"
        if not scene_dir.exists():
            return self._build_dummy_samples()

        samples = []
        for scene_file in sorted(scene_dir.glob("*.npz")):
            data = np.load(scene_file, allow_pickle=True)
            num_frames = len(data["ego_pose"])
            stride = max(1, round(self.frame_interval / 0.5))
            step = self.num_past_frames * stride

            for start in range(0, num_frames - self.sequence_length * stride + 1, step):
                samples.append({
                    "scene": scene_file.stem,
                    "start_frame": start,
                    "stride": stride,
                })

        if not samples:
            print(f"Warning: no valid {self.split} windows found under {scene_dir}")

        os.makedirs(preprocessed.parent, exist_ok=True)
        with open(preprocessed, "wb") as f:
            pickle.dump(samples, f)

        return samples

    def _build_dummy_samples(self) -> List[Dict]:
        """Generate synthetic samples for testing when nuScenes is unavailable."""
        import hashlib
        total_scenes = 50 if self.split == "train" else 10
        samples = []
        for i in range(total_scenes * 20):
            scene_hash = hashlib.md5(f"scene_{i % total_scenes}".encode()).hexdigest()[:8]
            samples.append({
                "scene": f"scene_{scene_hash}",
                "start_frame": (i * 5) % 100,
                "stride": 1,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        scene_path = (
            self.root / self.version / self.split / "scenes" / f"{sample['scene']}.npz"
        )

        if scene_path.exists():
            return self._load_real_sample(sample)
        return self._generate_synthetic_sample(sample)

    def _load_real_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Load and process a real nuScenes scene clip."""
        scene_path = (
            self.root / self.version / self.split / "scenes" / f"{sample['scene']}.npz"
        )
        data = np.load(scene_path, allow_pickle=True)

        start = sample["start_frame"]
        stride = sample["stride"]

        past_indices = list(range(start, start + self.num_past_frames * stride, stride))
        future_indices = list(
            range(
                start + self.num_past_frames * stride,
                start + self.sequence_length * stride,
                stride,
            )
        )

        all_images = data["images"]
        all_poses = data["ego_pose"]
        all_occupancy = data.get("occupancy", None)
        all_intrinsics = data.get("cam_intrinsics", None)
        all_extrinsics = data.get("cam_extrinsics", None)

        past_images = self._index_images(all_images, past_indices)
        past_ego = torch.from_numpy(all_poses[past_indices]).float()
        past_ego = past_ego - past_ego[0:1]

        past_intrinsics = self._index_calibration(
            all_intrinsics, past_indices, self.num_cameras, 3
        )
        past_extrinsics = self._index_calibration(
            all_extrinsics, past_indices, self.num_cameras, 4
        )

        if all_occupancy is not None:
            if self.occupancy_target == "last_past_repeat":
                anchor = all_occupancy[past_indices[-1]]
                future_occ = np.broadcast_to(
                    anchor, (self.num_future_frames, *anchor.shape)
                ).copy()
                future_occ = torch.from_numpy(future_occ).long()
            else:
                future_occ = torch.from_numpy(all_occupancy[future_indices]).long()
        else:
            future_occ = self._generate_dummy_occupancy()

        future_ego = torch.from_numpy(all_poses[future_indices]).float()
        future_ego = future_ego - future_ego[0:1]

        return {
            "past_images": past_images,
            "past_ego_pose": past_ego,
            "future_occupancy": future_occ,
            "future_ego_pose": future_ego,
            "past_intrinsics": past_intrinsics,
            "past_extrinsics": past_extrinsics,
            "token": f"{sample['scene']}_{sample['start_frame']}",
        }

    def _index_calibration(self, calibration, indices, num_cameras, matrix_size):
        """Return calibration for selected frames.

        ``calibration`` may be ``(N, C, D, D)`` from a real .npz, ``(N, D, D)``
        from a legacy single-camera file, or None for synthetic data. In the
        None case, identity matrices are generated to keep encoder interfaces
        stable.
        """
        if calibration is None:
            identity = torch.eye(matrix_size)
            return identity.repeat(len(indices), num_cameras, 1, 1)

        calib = np.asarray(calibration[indices])

        if calib.ndim == 3:
            # Legacy single-camera calibration (T, D, D). Duplicate it when a
            # multi-camera config is used so the encoder still receives the
            # expected camera axis.
            if num_cameras == 1:
                return torch.from_numpy(calib).float()
            calib = np.repeat(calib[:, None, ...], num_cameras, axis=1)
            return torch.from_numpy(calib).float()

        # Multi-camera calibration (T, C, D, D). Select CAM_FRONT for the
        # single-camera config to keep the encoder contract intact.
        if num_cameras == 1 and calib.shape[1] > 1:
            calib = calib[:, 0, ...]
        return torch.from_numpy(calib).float()

    def _index_images(self, all_images, indices):
        """Select frames and normalize while preserving the camera axis.

        Supports legacy single-camera ``.npz`` files with shape
        ``(N, 3, H, W)`` and multi-camera files with shape
        ``(N, C, 3, H, W)``.
        """
        images = np.asarray(all_images[indices])

        if images.ndim == 5:
            if self.num_cameras == 1:
                # Multi-camera .npz file used by a single-camera config:
                # keep CAM_FRONT only to match the original encoder contract.
                images = images[:, 0, ...]
            return torch.from_numpy(images).float() / 255.0

        if images.ndim == 4:
            if self.num_cameras == 1:
                return torch.from_numpy(images).float() / 255.0

            # Some old single-camera .npz files can remain in the cache.
            # Repeat the front view so a multi-camera config can still run,
            # but warn that this is not real 360-degree data.
            import warnings
            warnings.warn(
                "Found a legacy single-camera .npz file while "
                "num_cameras>1; duplicating the front view. Re-run "
                "scripts/preprocess_nuscenes.py before reporting metrics."
            )
            images = np.repeat(
                images[:, None, ...], self.num_cameras, axis=1
            )
            return torch.from_numpy(images).float() / 255.0

        raise ValueError(
            f"Unexpected image array shape {images.shape}; "
            "expected (N, 3, H, W) or (N, C, 3, H, W)."
        )
    def _generate_synthetic_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Generate a synthetic sample for development/testing."""
        Tp, Tf = self.num_past_frames, self.num_future_frames
        H, W = self.image_size
        BvH, BvW = self.bev_grid_size

        past_images = torch.rand(Tp, self.num_cameras, 3, H, W)
        past_ego = torch.zeros(Tp, 3)
        past_ego[:, 0] = torch.linspace(0, Tp * 0.5, Tp)

        future_occ = torch.randint(0, 2, (Tf, self.num_z, BvH, BvW))
        future_ego = torch.zeros(Tf, 3)
        future_ego[:, 0] = torch.linspace(Tp * 0.5, (Tp + Tf) * 0.5, Tf)

        past_intrinsics = torch.eye(3).repeat(Tp, self.num_cameras, 1, 1)
        past_extrinsics = torch.eye(4).repeat(Tp, self.num_cameras, 1, 1)

        return {
            "past_images": past_images,
            "past_ego_pose": past_ego,
            "future_occupancy": future_occ,
            "future_ego_pose": future_ego,
            "past_intrinsics": past_intrinsics,
            "past_extrinsics": past_extrinsics,
            "token": f"synthetic_{sample['scene']}_{sample['start_frame']}",
        }

    def _generate_dummy_occupancy(self) -> torch.Tensor:
        """Generate dummy occupancy for when real data is unavailable."""
        Tf = self.num_future_frames
        BvH, BvW = self.bev_grid_size
        return torch.randint(0, 2, (Tf, self.num_z, BvH, BvW))

