"""Comprehensive evaluator for world model predictions."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from driveworld.training.metrics import compute_iou, compute_video_metrics
from driveworld.utils.config import Config
from driveworld.utils.logging import Logger


class WorldModelEvaluator:
    """Evaluates trained world models on validation/test splits.

    Computes:
        - Per-class and mean IoU
        - Video prediction quality (PSNR, per-step IoU)
        - Temporal consistency metrics
        - Generates visualization artifacts
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        device: torch.device,
        logger: Optional[Logger] = None,
        output_dir: str = "outputs/eval",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        num_vis_samples: int = 5,
    ) -> Dict[str, float]:
        """Run full evaluation."""
        self.model.eval()

        all_ious = []
        all_video_metrics = []
        vis_count = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = self._to_device(batch)
            past_images, past_ego, future_occ, future_ego, intrinsics, extrinsics, tokens = batch

            if self.config.training.paradigm == "occworld":
                output = self.model(
                    past_images,
                    past_ego,
                    future_ego,
                    future_occ,
                    intrinsics,
                    extrinsics,
                )
                pred = output["occupancy_pred"]
            else:
                pred_logits = self.model.sample(
                    past_images,
                    past_ego,
                    future_ego,
                    num_inference_steps=50,
                    past_intrinsics=intrinsics,
                    past_extrinsics=extrinsics,
                )
                pred = pred_logits.unsqueeze(2).expand(-1, -1, 2, -1, -1, -1)

            ious = compute_iou(pred, future_occ)
            all_ious.append(ious)

            pred_binary = pred.argmax(dim=2)
            video_m = compute_video_metrics(pred_binary, future_occ)
            all_video_metrics.append(video_m)

            if vis_count < num_vis_samples:
                from driveworld.eval.visualize import visualize_occupancy_comparison
                visualize_occupancy_comparison(
                    pred_binary[0].cpu().numpy(),
                    future_occ[0].cpu().numpy(),
                    save_path=self.output_dir / f"sample_{batch_idx}.png",
                )
                vis_count += 1

        results = {}
        for metric_name in all_ious[0].keys():
            results[metric_name] = np.mean([m[metric_name] for m in all_ious])
        for metric_name in all_video_metrics[0].keys():
            results[metric_name] = np.mean([m[metric_name] for m in all_video_metrics])

        if self.logger:
            self.logger.info("Evaluation Results:")
            for k, v in results.items():
                self.logger.info(f"  {k}: {v:.4f}")

        return results

    def _to_device(self, batch):
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
