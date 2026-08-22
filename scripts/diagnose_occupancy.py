#!/usr/bin/env python3
"""Diagnose occupancy prediction: GT sparsity vs predicted sparsity.

Useful when mIoU looks flat because the empty class dominates. This prints
occupied-class recall/precision/IoU/Dice and per-timestep IoU so you can tell
whether the model collapses to "all empty" or finds some occupied voxels.

Usage:
    python scripts/diagnose_occupancy.py \
        --checkpoint outputs/experiments/occworld_lss_bev32_20260822/checkpoints/best.pt \
        --config configs/lss_occworld.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from driveworld.utils.config import load_config
from driveworld.training import WorldModelTrainer
from driveworld.data import NuScenesWorldModelDataset


def build_val_loader(config):
    from driveworld.data import create_dataloader

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
    parser = argparse.ArgumentParser(description="Diagnose occupancy predictions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/occworld.yaml")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = WorldModelTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    model = trainer.model.to(device).eval()

    loader = build_val_loader(config)

    gt_total = 0
    pred_total = 0
    inter_total = 0
    union_total = 0
    voxel_total = 0
    t_inter = {}
    t_union = {}
    t_gt = {}
    t_pred = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Diagnosing")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            (
                past_images,
                past_ego,
                future_occ,
                future_ego,
                intrinsics,
                extrinsics,
                _,
            ) = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            if config.training.paradigm == "occworld":
                output = model(
                    past_images, past_ego, future_ego, future_occ,
                    intrinsics, extrinsics,
                )
                pred = output["occupancy_pred"]
            else:
                pred_logits = model.sample(
                    past_images, past_ego, future_ego,
                    num_inference_steps=50,
                    past_intrinsics=intrinsics,
                    past_extrinsics=extrinsics,
                )
                pred = pred_logits.unsqueeze(2).expand(-1, -1, 2, -1, -1, -1)

            pred_binary = pred.argmax(dim=2)
            gt_binary = future_occ

            B, T = gt_binary.shape[:2]
            gt = gt_binary
            pr = pred_binary

            gt_total += gt.sum().item()
            pred_total += pr.sum().item()
            inter_total += (gt * pr).sum().item()
            union_total += ((gt + pr) > 0).sum().item()
            voxel_total += gt.numel()

            for t in range(T):
                gt_t = gt[:, t]
                pr_t = pr[:, t]
                t_gt[t] = t_gt.get(t, 0) + gt_t.sum().item()
                t_pred[t] = t_pred.get(t, 0) + pr_t.sum().item()
                t_inter[t] = t_inter.get(t, 0) + (gt_t * pr_t).sum().item()
                t_union[t] = t_union.get(t, 0) + ((gt_t + pr_t) > 0).sum().item()

    if voxel_total == 0:
        print("No val samples found.")
        return

    print("\n================ Occupancy Diagnosis ================")
    print(f"Voxels evaluated  : {voxel_total:,}")
    print(f"GT occupied ratio : {gt_total / voxel_total:.5f}")
    print(f"Pred occupied ratio: {pred_total / voxel_total:.5f}")
    eps = 1e-8
    recall = inter_total / (gt_total + eps)
    precision = inter_total / (pred_total + eps)
    iou = inter_total / (union_total + eps)
    dice = 2 * inter_total / (gt_total + pred_total + eps)
    print(f"Occupied recall   : {recall:.5f}")
    print(f"Occupied precision: {precision:.5f}")
    print(f"Occupied IoU      : {iou:.5f}")
    print(f"Occupied Dice     : {dice:.5f}")
    print("\nPer-timestep occupied IoU:")
    for t in sorted(t_union.keys()):
        iou_t = t_inter[t] / (t_union[t] + eps)
        print(f"  t={t}: IoU={iou_t:.5f}  GT={t_gt[t]:,}  Pred={t_pred[t]:,}")
    print("=====================================================")


if __name__ == "__main__":
    main()
