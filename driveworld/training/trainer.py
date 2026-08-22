"""World Model Trainer with mixed precision, checkpointing, and logging.

Supports:
- OccWorld autoregressive training
- DriveDiffuser diffusion training
- Mixed precision (AMP) for memory efficiency
- Gradient accumulation and clipping
- Exponential moving average (EMA)
- TensorBoard logging
- Periodic validation and checkpointing
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from driveworld.data import NuScenesWorldModelDataset, create_dataloader
from driveworld.models import OccWorld, DriveDiffuser, build_encoder, build_decoder
from driveworld.training.losses import WorldModelLoss
from driveworld.training.metrics import AverageMeter, compute_iou, compute_video_metrics
from driveworld.utils.config import Config
from driveworld.utils.logging import Logger
from driveworld.utils.seed import set_seed


class WorldModelTrainer:
    """Unified trainer for all world model paradigms.

    Usage:
        config = load_config("configs/occworld.yaml")
        trainer = WorldModelTrainer(config)
        trainer.train()
    """

    def __init__(self, config: Config, logger: Optional[Logger] = None):
        self.config = config
        set_seed(config.training.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if logger is None:
            logger = Logger(config.training.log_dir)
        self.logger = logger

        self.model = self._build_model()
        self.model = self.model.to(self.device)

        self.loss_fn = WorldModelLoss(config)
        self.loss_fn = self.loss_fn.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.scaler = GradScaler(enabled=config.training.mixed_precision)

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.training.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=config.training.max_epochs - config.training.warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.training.warmup_epochs],
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        self.logger.info(f"Trainer initialized. Device: {self.device}")
        self.logger.info(f"Paradigm: {config.training.paradigm}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    def _build_model(self) -> nn.Module:
        encoder = build_encoder(self.config)
        decoder = build_decoder(self.config)
        paradigm = self.config.training.paradigm

        if paradigm == "occworld":
            from driveworld.models.occworld import OccWorld as Model
            return Model(
                encoder=encoder,
                decoder=decoder,
                hidden_dim=self.config.occworld.hidden_dim,
                num_layers=self.config.occworld.num_layers,
                num_heads=self.config.occworld.num_heads,
                vocab_size=self.config.occworld.vocab_size,
                dropout=self.config.occworld.dropout,
            )
        elif paradigm == "diffusion":
            from driveworld.models.diffusion import UNet2D, DriveDiffuser as Model
            num_z = 16
            num_future = self.config.data.num_future_frames
            unet = UNet2D(
                in_channels=num_future * num_z,
                out_channels=num_future * num_z,
                cond_channels=64,
                base_channels=self.config.diffusion.unet_channels[0],
                channel_mult=(1, 2, 4),
                num_res_blocks=self.config.diffusion.num_res_blocks,
                dropout=self.config.diffusion.dropout,
            )
            return Model(
                encoder=encoder,
                unet=unet,
                num_timesteps=self.config.diffusion.num_timesteps,
                beta_schedule=self.config.diffusion.beta_schedule,
                num_future_frames=num_future,
                num_z=num_z,
            )
        
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")

    def _build_dataloaders(self):
        cfg = self.config.data
        common = dict(
            root=cfg.dataset_root,
            version=cfg.version,
            num_past_frames=cfg.num_past_frames,
            num_future_frames=cfg.num_future_frames,
            image_size=cfg.image_size,
            bev_grid_size=cfg.bev_grid_size,
            bev_resolution=cfg.bev_resolution,
            num_cameras=cfg.num_cameras,
            occupancy_target=getattr(cfg, "occupancy_target", "future"),
        )

        train_ds = NuScenesWorldModelDataset(split="train", augment=True, **common)
        val_ds = NuScenesWorldModelDataset(split="val", augment=False, **common)

        self.train_loader = create_dataloader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            seed=self.config.training.seed,
        )
        self.val_loader = create_dataloader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            seed=self.config.training.seed,
        )

        self.logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    def train(self) -> None:
        """Main training loop."""
        self._build_dataloaders()
        cfg = self.config.training

        for epoch in range(self.current_epoch, cfg.max_epochs):
            self.current_epoch = epoch
            self.logger.info(f"--- Epoch {epoch + 1}/{cfg.max_epochs} ---")

            train_loss = self._train_epoch()
            self.logger.log_scalar("train/loss_epoch", train_loss, epoch)
            self.logger.info(f"Train Loss: {train_loss:.4f}")

            if (epoch + 1) % cfg.val_every == 0:
                val_metrics = self._validate_epoch()
                val_loss = val_metrics.get("loss", float("inf"))
                self.logger.log_scalar("val/loss", val_loss, epoch)
                for key, val in val_metrics.items():
                    if key != "loss":
                        self.logger.log_scalar(f"val/{key}", val, epoch)
                self.logger.info(
                    f"Val Loss: {val_loss:.4f} | "
                    + " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items() if k != "loss")
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    self._save_checkpoint("best")
                else:
                    self.early_stop_counter += 1

            if (epoch + 1) % cfg.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}")

            self.scheduler.step()

        self._save_checkpoint("final")
        self.logger.info("Training complete!")
        self.logger.close()

    def _train_epoch(self) -> float:
        self.model.train()
        loss_meter = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            past_images, past_ego, future_occ, future_ego, intrinsics, extrinsics, _ = batch

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.training.mixed_precision):
                if self.config.training.paradigm == "occworld":
                    output = self.model(
                        past_images,
                        past_ego,
                        future_ego,
                        future_occ,
                        intrinsics,
                        extrinsics,
                    )
                else:
                    output = self.model(
                        past_images,
                        past_ego,
                        future_ego,
                        future_occ,
                        intrinsics,
                        extrinsics,
                    )

                losses = self.loss_fn(output, batch)
                loss = losses["total"]

            self.scaler.scale(loss).backward()

            if self.config.training.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), past_images.size(0))

            if batch_idx % self.config.training.log_every == 0:
                self.logger.log_scalar("train/loss_step", loss.item(), self.global_step)
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_scalar("train/lr", lr, self.global_step)

            self.global_step += 1

        return loss_meter.avg

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()

        for batch in self.val_loader:
            batch = self._to_device(batch)
            past_images, past_ego, future_occ, future_ego, intrinsics, extrinsics, _ = batch

            if self.config.training.paradigm == "occworld":
                output = self.model(
                    past_images,
                    past_ego,
                    future_ego,
                    future_occ,
                    intrinsics,
                    extrinsics,
                )
                if "occupancy_pred" in output:
                    ious = compute_iou(output["occupancy_pred"], future_occ)
                    iou_meter.update(ious["miou"], past_images.size(0))
            else:
                output = self.model(
                    past_images,
                    past_ego,
                    future_ego,
                    future_occ,
                    intrinsics,
                    extrinsics,
                )

            losses = self.loss_fn(output, batch)
            loss_meter.update(losses["total"].item(), past_images.size(0))

        metrics = {"loss": loss_meter.avg}
        if iou_meter.count > 0:
            metrics["miou"] = iou_meter.avg
        return metrics

    def _to_device(self, batch) -> tuple:
        (
            past_images,
            past_ego,
            future_occ,
            future_ego,
            intrinsics,
            extrinsics,
            tokens,
        ) = batch
        return (
            past_images.to(self.device),
            past_ego.to(self.device),
            future_occ.to(self.device),
            future_ego.to(self.device),
            intrinsics.to(self.device),
            extrinsics.to(self.device),
            tokens,
        )

    def _save_checkpoint(self, name: str) -> None:
        ckpt_dir = Path(self.config.training.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{name}.pt"

        # Optimizer/scheduler/scaler state is only needed for full resume.
        # Periodic epoch checkpoints keep the model weights only, which avoids
        # filling the data disk with multi-hundred-MB optimizer snapshots.
        payload = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if name in ("best", "final"):
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
            payload["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(payload, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.logger.info(f"Checkpoint loaded from {path} (epoch {ckpt['epoch']})")
