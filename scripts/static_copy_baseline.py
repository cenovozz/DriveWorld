#!/usr/bin/env python3
"""Static-copy oracle baseline for occupancy prediction.

For every (past, future) window, repeat the last past occupancy frame for all
future frames and report occupied-class IoU/Dice. This is the "nothing moves"
upper bound that any temporal model should at least beat on slow scenes.

Usage:
    python scripts/static_copy_baseline.py --root data/nuscenes/v1.0-mini --split val
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Static-copy occupancy baseline")
    parser.add_argument("--root", default="data/nuscenes/v1.0-mini")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_past_frames", type=int, default=3)
    parser.add_argument("--num_future_frames", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    scene_dir = Path(args.root) / args.split / "scenes"
    files = sorted(scene_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found under {scene_dir}")
        return

    Tp, Tf, stride = args.num_past_frames, args.num_future_frames, args.stride
    seq = Tp + Tf
    step = Tp * stride

    inter = 0
    union = 0
    gt_sum = 0
    pr_sum = 0
    t_inter = {}
    t_union = {}
    t_gt = {}

    for f in files:
        occ = np.load(f)["occupancy"].astype(np.uint8)
        n_frames = occ.shape[0]
        for start in range(0, n_frames - seq * stride + 1, step):
            anchor_idx = start + (Tp - 1) * stride
            anchor = occ[anchor_idx]
            future_idx = range(start + Tp * stride, start + seq * stride, stride)
            for i, fi in enumerate(future_idx):
                gt = occ[fi]
                pr = anchor
                gt_b = gt > 0
                pr_b = pr > 0
                inter += int((gt_b & pr_b).sum())
                union += int((gt_b | pr_b).sum())
                gt_sum += int(gt_b.sum())
                pr_sum += int(pr_b.sum())
                t_inter[i] = t_inter.get(i, 0) + int((gt_b & pr_b).sum())
                t_union[i] = t_union.get(i, 0) + int((gt_b | pr_b).sum())
                t_gt[i] = t_gt.get(i, 0) + int(gt_b.sum())

    eps = 1e-8
    print("\n================ Static-Copy Baseline ================")
    print("Strategy: repeat the last past frame for all future frames")
    print(f"Occupied IoU  : {inter / (union + eps):.5f}")
    print(f"Occupied Dice : {2 * inter / (gt_sum + pr_sum + eps):.5f}")
    print("Per-timestep occupied IoU:")
    for i in sorted(t_union):
        print(f"  t={i}: IoU={t_inter[i] / (t_union[i] + eps):.5f}  GT={t_gt[i]:,}")
    print("=====================================================")


if __name__ == "__main__":
    main()
