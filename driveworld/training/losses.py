"""Loss functions for world model training.

Supports multiple paradigms:
- OccupancyLoss: Cross-entropy + Dice for OccWorld
- DiffusionLoss: Simple (noise prediction) MSE for DriveDiffuser
- WorldModelLoss: Combined loss with optional KL regularization
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Differentiable Dice loss for occupancy prediction."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)[:, 1]
        target = target.float()

        if target.dim() == pred.dim() + 1 and target.shape[2] == 2:
            target = target[:, :, 1]

        intersection = (pred * target).sum(dim=[-3, -2, -1])
        union = pred.sum(dim=[-3, -2, -1]) + target.sum(dim=[-3, -2, -1])
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class OccupancyLoss(nn.Module):
    """Combined cross-entropy + Dice loss for 3D occupancy prediction."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = SoftDiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute occupancy losses.

        Args:
            pred: (B, T, num_classes, Z, H, W) logits
            target: (B, T, Z, H, W) integer labels

        Returns:
            Dict with "total", "ce", "dice" losses
        """
        B, T, C, Z, H, W = pred.shape
        pred_flat = pred.view(B * T, C, Z, H, W)
        target_flat = target.view(B * T, Z, H, W)

        ce = self.ce_loss(pred_flat, target_flat)
        dice = self.dice_loss(pred.view(B * T, C, Z, H, W), target_flat)

        total = self.ce_weight * ce + self.dice_weight * dice
        return {"total": total, "ce": ce, "dice": dice}


class DiffusionLoss(nn.Module):
    """Simple MSE loss for diffusion noise prediction."""

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mse = F.mse_loss(noise_pred, noise)
        return {"total": mse, "mse": mse}


class WorldModelLoss(nn.Module):
    """Unified loss for multi-paradigm world model training.

    Automatically selects the appropriate loss based on model output keys.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.occ_loss = OccupancyLoss(
            ce_weight=config.loss.cross_entropy_weight,
            dice_weight=config.loss.dice_weight,
        )
        self.diff_loss = DiffusionLoss()

    def forward(self, model_output: Dict, batch) -> Dict[str, torch.Tensor]:
        """Compute losses based on output type.

        Args:
            model_output: Model forward output dict
            batch: (past_images, past_ego, future_occ, future_ego, tokens)

        Returns:
            Dict with "total" loss and component losses
        """
        _, _, future_occupancy, _, _ = batch

        losses = {}

        if "occupancy_pred" in model_output:
            occ_losses = self.occ_loss(
                model_output["occupancy_pred"], future_occupancy
            )
            losses["occ_total"] = occ_losses["total"]
            losses["occ_ce"] = occ_losses["ce"]
            losses["occ_dice"] = occ_losses["dice"]
            losses["total"] = losses.get("total", 0) + occ_losses["total"]

        if "noise_pred" in model_output:
            diff_losses = self.diff_loss(
                model_output["noise_pred"], model_output["noise"]
            )
            losses["diff_total"] = diff_losses["total"]
            losses["diff_mse"] = diff_losses["mse"]
            losses["total"] = losses.get("total", 0) + diff_losses["total"]

        return losses
