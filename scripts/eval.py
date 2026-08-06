#!/usr/bin/env python3
"""Evaluation entry point for DriveWorld.

Usage:
    python scripts/eval.py --checkpoint checkpoints/best.pt --config configs/occworld.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from driveworld.utils.config import load_config, Config
from driveworld.utils.logging import setup_logger
from driveworld.training import WorldModelTrainer
from driveworld.data import NuScenesWorldModelDataset, create_dataloader
from driveworld.eval import WorldModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="DriveWorld Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default="configs/occworld.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval",
        help="Directory for evaluation outputs"
    )
    parser.add_argument(
        "--num-vis", type=int, default=5,
        help="Number of visualization samples"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir, "eval")

    logger.info(f"Loading checkpoint: {args.checkpoint}")

    trainer = WorldModelTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    model = trainer.model

    common = dict(
        root=config.data.dataset_root,
        version=config.data.version,
        num_past_frames=config.data.num_past_frames,
        num_future_frames=config.data.num_future_frames,
        image_size=config.data.image_size,
        bev_grid_size=config.data.bev_grid_size,
        bev_resolution=config.data.bev_resolution,
    )
    val_ds = NuScenesWorldModelDataset(split="val", augment=False, **common)
    val_loader = create_dataloader(
        val_ds, batch_size=config.data.batch_size, shuffle=False,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
    )

    evaluator = WorldModelEvaluator(
        model=model, config=config, device=device,
        logger=logger, output_dir=args.output_dir,
    )
    results = evaluator.evaluate(val_loader, num_vis_samples=args.num_vis)

    logger.info("=" * 40)
    logger.info("Final Results:")
    for k, v in sorted(results.items()):
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
