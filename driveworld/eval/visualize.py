"""Visualization utilities for world model predictions.

Produces publication-quality comparison figures, GIFs, and
BEV feature map visualizations for project demos and papers.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_occupancy_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    timesteps: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """Side-by-side comparison of predicted vs ground truth occupancy.

    Creates a figure showing ground truth (top row) and prediction (bottom row)
    for multiple future timesteps. Green = correct, red = false positive,
    blue = false negative.

    Args:
        pred: (T, Z, H, W) predicted binary occupancy
        target: (T, Z, H, W) ground truth occupancy
        timesteps: which timesteps to display (default: first, mid, last)
        save_path: if provided, save figure to this path
        figsize: figure size in inches
    """
    T = pred.shape[0]

    if timesteps is None:
        timesteps = [0, T // 2, T - 1]
    timesteps = [min(t, T - 1) for t in timesteps]

    fig, axes = plt.subplots(2, len(timesteps), figsize=figsize)

    for col, t in enumerate(timesteps):
        pred_bev = pred[t].max(axis=0)
        target_bev = target[t].max(axis=0)

        axes[0, col].imshow(target_bev, cmap="Blues", vmin=0, vmax=1)
        axes[0, col].set_title(f"GT t={t + 1}")
        axes[0, col].axis("off")

        diff_map = np.zeros((*pred_bev.shape, 3))
        diff_map[..., 0] = pred_bev * (1 - target_bev)
        diff_map[..., 1] = pred_bev * target_bev
        diff_map[..., 2] = target_bev * (1 - pred_bev)

        axes[1, col].imshow(diff_map)
        axes[1, col].set_title(f"Pred t={t + 1} (R:FP G:OK B:FN)")
        axes[1, col].axis("off")

    plt.suptitle("Occupancy Prediction: Ground Truth vs Prediction", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def create_prediction_gif(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: str = "prediction.gif",
    duration: float = 0.3,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """Create a side-by-side GIF showing prediction over time.

    Args:
        pred: (T, Z, H, W) predicted binary occupancy
        target: (T, Z, H, W) ground truth occupancy
        save_path: output GIF path
        duration: frame duration in seconds
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Install with: pip install imageio")
        return

    frames = []
    T = pred.shape[0]

    for t in range(T):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        pred_bev = pred[t].max(axis=0)
        target_bev = target[t].max(axis=0)

        ax1.imshow(target_bev, cmap="Blues", vmin=0, vmax=1)
        ax1.set_title(f"Ground Truth (t={t + 1}/{T})")
        ax1.axis("off")

        ax2.imshow(pred_bev, cmap="Oranges", vmin=0, vmax=1)
        ax2.set_title(f"Prediction (t={t + 1}/{T})")
        ax2.axis("off")

        plt.tight_layout()
        fig.canvas.draw()

        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(img[..., :3])

        plt.close()

    imageio.mimsave(save_path, frames, duration=duration, loop=0)
    print(f"GIF saved to {save_path}")


def render_bev_feature_map(
    bev_feat: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Visualize BEV feature map as a heatmap.

    Args:
        bev_feat: (C, H, W) or (1, C, H, W) BEV feature tensor
        save_path: optional save path
        figsize: figure size
    """
    if bev_feat.dim() == 4:
        bev_feat = bev_feat[0]

    feat_np = bev_feat.detach().cpu().numpy()
    feat_mean = feat_np.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(feat_mean, cmap="viridis")
    ax.set_title("BEV Feature Map (channel mean)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def plot_training_curves(
    log_dir: str,
    save_path: str = "training_curves.png",
) -> None:
    """Plot training and validation curves from TensorBoard logs.

    Args:
        log_dir: TensorBoard log directory
        save_path: output image path
    """
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_path = Path(log_dir)
    event_files = sorted(log_path.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return

    ea = EventAccumulator(str(event_files[0].parent))
    ea.Reload()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "train/loss_epoch" in ea.Tags()["scalars"]:
        train_loss = ea.Scalars("train/loss_epoch")
        steps = [s.step for s in train_loss]
        vals = [s.value for s in train_loss]
        axes[0].plot(steps, vals, label="Train Loss", color="blue")

    if "val/loss" in ea.Tags()["scalars"]:
        val_loss = ea.Scalars("val/loss")
        steps = [s.step for s in val_loss]
        vals = [s.value for s in val_loss]
        axes[0].plot(steps, vals, label="Val Loss", color="red", marker="o")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "val/miou" in ea.Tags().get("scalars", []):
        miou = ea.Scalars("val/miou")
        steps = [s.step for s in miou]
        vals = [s.value for s in miou]
        axes[1].plot(steps, vals, label="mIoU", color="green", marker="o")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")
