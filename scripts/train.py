#!/usr/bin/env python3
"""Training entry point for DriveWorld.

Usage:
    python scripts/train.py --config configs/occworld.yaml
    python scripts/train.py --config configs/diffusion.yaml --resume checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from driveworld.utils.config import load_config
from driveworld.utils.logging import setup_logger
from driveworld.training import WorldModelTrainer


def main():
    parser = argparse.ArgumentParser(description="DriveWorld Training")
    parser.add_argument(
        "--config", type=str, default="configs/occworld.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--paradigm", type=str, default=None,
        choices=["occworld", "diffusion", "hybrid"],
        help="Override paradigm in config"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.paradigm:
        config.training.paradigm = args.paradigm

    logger = setup_logger(config.training.log_dir)

    logger.info("=" * 60)
    logger.info("DriveWorld Training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Paradigm: {config.training.paradigm}")
    logger.info("=" * 60)

    trainer = WorldModelTrainer(config, logger)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
