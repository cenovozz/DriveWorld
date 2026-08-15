#!/usr/bin/env python3
"""Run a tracked DriveWorld experiment end to end.

Usage:
    python scripts/run_experiment.py --config configs/occworld.yaml --name occworld_focal_tw_20260815 --seed 42 --eval
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from driveworld.data import NuScenesWorldModelDataset, create_dataloader
from driveworld.eval import WorldModelEvaluator
from driveworld.training import WorldModelTrainer
from driveworld.utils.config import load_config, save_config
from driveworld.utils.logging import setup_logger
from driveworld.utils.seed import set_seed


def build_val_loader(config):
    common = dict(
        root=config.data.dataset_root,
        version=config.data.version,
        num_past_frames=config.data.num_past_frames,
        num_future_frames=config.data.num_future_frames,
        image_size=config.data.image_size,
        bev_grid_size=config.data.bev_grid_size,
        bev_resolution=config.data.bev_resolution,
        num_cameras=config.data.num_cameras,
    )
    val_ds = NuScenesWorldModelDataset(split="val", augment=False, **common)
    return create_dataloader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.training.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="DriveWorld tracked experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.training.seed = args.seed
    set_seed(config.training.seed)

    exp_dir = Path("outputs/experiments") / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(args.config, exp_dir / "config.yaml")
    save_config(config, str(exp_dir / "effective_config.yaml"))
    (exp_dir / "run_cmd.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    config.training.checkpoint_dir = str(exp_dir / "checkpoints")
    config.training.log_dir = str(exp_dir / "logs")

    logger = setup_logger(config.training.log_dir)
    logger.info(f"Experiment: {args.name}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {config.training.seed}")
    logger.info(f"Output dir: {exp_dir}")

    trainer = WorldModelTrainer(config, logger)
    trainer.train()

    if args.eval:
        checkpoint = Path(config.training.checkpoint_dir) / "best.pt"
        if not checkpoint.exists():
            checkpoint = Path(config.training.checkpoint_dir) / "final.pt"
        if not checkpoint.exists():
            logger.info("No checkpoint found; skipping evaluation.")
            logger.close()
            return

        trainer.load_checkpoint(str(checkpoint))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_loader = build_val_loader(config)
        evaluator = WorldModelEvaluator(
            model=trainer.model,
            config=config,
            device=device,
            logger=None,
            output_dir=str(exp_dir / "eval"),
        )
        results = evaluator.evaluate(val_loader, num_vis_samples=5)
        (exp_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Metrics saved to {exp_dir / 'metrics.json'}")

    logger.close()


if __name__ == "__main__":
    main()
