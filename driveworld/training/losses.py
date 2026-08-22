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


class FocalLoss(nn.Module):
    """Focal loss for heavily imbalanced occupancy labels."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(pred, dim=1)
        log_prob_t = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        prob_t = log_prob_t.exp()
        alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
        return -(alpha_t * (1.0 - prob_t) ** self.gamma * log_prob_t).mean()


class OccupancyLoss(nn.Module):
    """Combined cross-entropy (or focal) + Dice loss for 3D occupancy prediction."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        temporal_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if use_focal:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = SoftDiceLoss()

        if temporal_weights is None:
            temporal_weights = torch.ones(1)
        self.register_buffer("temporal_weights", temporal_weights)

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
        weights = self.temporal_weights.to(pred.device)
        if weights.numel() == 1:
            weights = weights.expand(T)
        if weights.numel() != T:
            raise ValueError(f"Expected temporal_weights length {T}, got {weights.numel()}")

        weight_sum = weights.sum().clamp_min(1e-8)
        ce = torch.zeros((), device=pred.device)
        dice = torch.zeros((), device=pred.device)
        for t in range(T):
            ce = ce + weights[t] * self.ce_loss(pred[:, t], target[:, t])
            dice = dice + weights[t] * self.dice_loss(pred[:, t], target[:, t])

        ce = ce / weight_sum
        dice = dice / weight_sum
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

        temporal_weights = None
        if config.loss.temporal_weighting:
            temporal_weights = torch.linspace(
                config.loss.temporal_weight_start,
                config.loss.temporal_weight_end,
                config.data.num_future_frames,
            )

        self.occ_loss = OccupancyLoss(
            ce_weight=config.loss.cross_entropy_weight,
            dice_weight=config.loss.dice_weight,
            use_focal=config.loss.use_focal,
            focal_alpha=config.loss.focal_alpha,
            focal_gamma=config.loss.focal_gamma,
            temporal_weights=temporal_weights,
        )
        self.diff_loss = DiffusionLoss()

    def forward(self, model_output: Dict, batch) -> Dict[str, torch.Tensor]:
        """Compute losses based on output type.

        Args:
            model_output: Model forward output dict
            batch: (past_images, past_ego, future_occ, future_ego,
                   past_intrinsics, past_extrinsics, tokens)

        Returns:
            Dict with "total" loss and component losses
        """
        future_occupancy = batch[2]

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
