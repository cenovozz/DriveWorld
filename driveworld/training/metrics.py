"""Evaluation metrics for world model predictions."""

from typing import Dict

import torch
import torch.nn.functional as F


class AverageMeter:
    """Tracks running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    """Compute per-class and mean IoU for occupancy prediction.

    Args:
        pred: (B, T, num_classes, Z, H, W) logits
        target: (B, T, Z, H, W) integer labels

    Returns:
        Dict with "iou_class_0", "iou_class_1", "miou"
    """
    pred_labels = pred.argmax(dim=2)
    ious = {}

    for cls in range(num_classes):
        pred_cls = (pred_labels == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        iou = (intersection + 1e-8) / (union + 1e-8)
        ious[f"iou_class_{cls}"] = iou.item()

    ious["miou"] = sum(v for k, v in ious.items() if k.startswith("iou_class")) / num_classes
    return ious


def compute_video_metrics(
    pred_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
) -> Dict[str, float]:
    """Compute video prediction quality metrics.

    Args:
        pred_sequence: (B, T, Z, H, W) predicted occupancy (0/1 after argmax)
        target_sequence: (B, T, Z, H, W) ground truth

    Returns:
        Dict with PSNR, SSIM-like metrics, and per-timestep IoU
    """
    pred = pred_sequence.float()
    target = target_sequence.float()

    mse = F.mse_loss(pred, target)

    max_val = 1.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))

    per_step_iou = []
    for t in range(pred.shape[1]):
        intersection = (pred[:, t] * target[:, t]).sum()
        union = ((pred[:, t] + target[:, t]) > 0).float().sum()
        per_step_iou.append((intersection / (union + 1e-8)).item())

    return {
        "psnr": psnr.item(),
        "mse": mse.item(),
        "iou_t0": per_step_iou[0] if per_step_iou else 0.0,
        "iou_tmid": per_step_iou[len(per_step_iou) // 2] if per_step_iou else 0.0,
        "iou_tfinal": per_step_iou[-1] if per_step_iou else 0.0,
        "iou_avg": sum(per_step_iou) / len(per_step_iou) if per_step_iou else 0.0,
    }
