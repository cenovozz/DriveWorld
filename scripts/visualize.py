#!/usr/bin/env python3
"""Standalone visualization script for DriveWorld predictions.

Usage:
    python scripts/visualize.py --checkpoint checkpoints/best.pt --config configs/occworld.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from driveworld.utils.config import load_config
from driveworld.training import WorldModelTrainer
from driveworld.data import NuScenesWorldModelDataset
from driveworld.eval.visualize import (
    visualize_occupancy_comparison,
    create_prediction_gif,
    render_bev_feature_map,
)


def main():
    parser = argparse.ArgumentParser(description="DriveWorld Visualization")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="configs/occworld.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/viz",
    )
    parser.add_argument(
        "--num-samples", type=int, default=3,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = WorldModelTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    model = trainer.model
    model.eval()

    common = dict(
        root=config.data.dataset_root,
        version=config.data.version,
        num_past_frames=config.data.num_past_frames,
        num_future_frames=config.data.num_future_frames,
        image_size=config.data.image_size,
        bev_grid_size=config.data.bev_grid_size,
        bev_resolution=config.data.bev_resolution,
        num_cameras=config.data.num_cameras,
        occupancy_target=getattr(config.data, "occupancy_target", "future"),
    )
    val_ds = NuScenesWorldModelDataset(split="val", augment=False, **common)

    for i in range(min(args.num_samples, len(val_ds))):
        sample = val_ds[i]

        past_images = sample["past_images"].unsqueeze(0).to(device)
        past_ego = sample["past_ego_pose"].unsqueeze(0).to(device)
        future_ego = sample["future_ego_pose"].unsqueeze(0).to(device)
        future_occ_gt = sample["future_occupancy"].numpy()
        past_intrinsics = sample["past_intrinsics"].unsqueeze(0).to(device)
        past_extrinsics = sample["past_extrinsics"].unsqueeze(0).to(device)

        with torch.no_grad():
            if config.training.paradigm == "occworld":
                output = model(
                    past_images,
                    past_ego,
                    future_ego,
                    past_intrinsics=past_intrinsics,
                    past_extrinsics=past_extrinsics,
                )
                pred = output["occupancy_pred"].argmax(dim=2)[0].cpu().numpy()
            else:
                pred = model.sample(
                    past_images,
                    past_ego,
                    future_ego,
                    num_inference_steps=50,
                    past_intrinsics=past_intrinsics,
                    past_extrinsics=past_extrinsics,
                )
                pred = (pred[0] > 0.5).cpu().numpy()

        visualize_occupancy_comparison(
            pred, future_occ_gt,
            save_path=str(output_dir / f"comparison_{i}.png"),
        )

        create_prediction_gif(
            pred, future_occ_gt,
            save_path=str(output_dir / f"prediction_{i}.gif"),
        )

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
