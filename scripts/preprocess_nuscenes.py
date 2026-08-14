"""将 nuScenes 原始数据预处理为模型训练用的 .npz 格式。

输入: data/nuscenes/v1.0-mini/ 原始 nuScenes 目录
输出: data/nuscenes/v1.0-mini/train/scenes/*.npz  (每个 scene 一个文件)

每个 .npz 包含:
  images:    (N_frames, 3, H, W)      uint8   前视相机图像
  ego_pose:  (N_frames, 3)             float32 自车位姿 (x, y, yaw)
  occupancy: (N_frames, 16, 200, 200)  bool    3D 占用网格(LiDAR生成)
  timestamps: (N_frames,)              int64   帧序号

用法:
  python scripts/preprocess_nuscenes.py
  python scripts/preprocess_nuscenes.py --dataroot data/nuscenes --version v1.0-mini
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ========== 配置 ==========
DEFAULT_DATAROOT = "data/nuscenes"
DEFAULT_VERSION = "v1.0-mini"
IMAGE_SIZE = (224, 480)
BEV_GRID = (200, 200)
BEV_RES = 0.5           # 米/像素
Z_BINS = 16
Z_RANGE = (-4.0, 4.0)   # 垂直范围（米）
XY_RANGE = 50.0         # BEV 半范围（米）
FRAME_INTERVAL = 2      # 采样间隔（nuScenes 关键帧 @2Hz）


def lidar_to_occupancy(lidar_points):
    """将 LiDAR 点云 (N,3) 转换为 (Z_BINS, H, W) 的 3D 占用网格。

    坐标系:
      - LiDAR 坐标: x 向前, y 向左, z 向上
      - BEV 坐标: 以自车为中心, 范围 [-XY_RANGE, XY_RANGE] 米
      - 垂直: Z_RANGE[0] ~ Z_RANGE[1]，地面以下点滤除

    Args:
        lidar_points: (N, 3) float32 LiDAR 点云

    Returns:
        occupancy: (Z_BINS, BEV_H, BEV_W) bool 数组
    """
    if len(lidar_points) == 0:
        return np.zeros((Z_BINS, *BEV_GRID), dtype=bool)

    x, y, z = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2]

    # 过滤 BEV 范围外
    mask = (np.abs(x) < XY_RANGE) & (np.abs(y) < XY_RANGE)
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) == 0:
        return np.zeros((Z_BINS, *BEV_GRID), dtype=bool)

    # 滤除地面以下点（z < Z_RANGE[0]）
    mask = z >= Z_RANGE[0]
    x, y, z = x[mask], y[mask], z[mask]

    if len(x) == 0:
        return np.zeros((Z_BINS, *BEV_GRID), dtype=bool)

    # 离散化到网格
    h_idx = ((y + XY_RANGE) / BEV_RES).astype(int)
    w_idx = ((x + XY_RANGE) / BEV_RES).astype(int)
    z_idx = ((z - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0]) * Z_BINS).astype(int)

    # 裁剪到合法范围
    valid = (
        (0 <= h_idx) & (h_idx < BEV_GRID[0]) &
        (0 <= w_idx) & (w_idx < BEV_GRID[1]) &
        (0 <= z_idx) & (z_idx < Z_BINS)
    )
    h_idx, w_idx, z_idx = h_idx[valid], w_idx[valid], z_idx[valid]

    # 填充占用网格
    occupancy = np.zeros((Z_BINS, *BEV_GRID), dtype=bool)
    occupancy[z_idx, h_idx, w_idx] = True

    return occupancy


def load_front_image(nusc, sample_data_token):
    """加载前视相机图片为 numpy 数组 (3, H, W) uint8。

    Args:
        nusc: NuScenes 实例
        sample_data_token: CAM_FRONT 的 sample_data token

    Returns:
        image: (3, H, W) uint8 RGB 图像
    """
    import cv2

    sd = nusc.get('sample_data', sample_data_token)
    img_path = os.path.join(nusc.dataroot, sd['filename'])

    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((3, *IMAGE_SIZE), dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    return img.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def process_scene(nusc, scene, split, dataroot):
    """处理单个 nuScenes scene，保存为 .npz 文件。

    Args:
        nusc: NuScenes 实例
        scene: nuScenes scene 对象
        split: 'train' 或 'val'
        dataroot: nuScenes 数据根目录

    Returns:
        num_frames: 该 scene 的帧数
    """
    scene_name = scene['name']
    first_token = scene['first_sample_token']

    # 收集该 scene 的所有 sample（按时间顺序）
    sample = nusc.get('sample', first_token)
    samples = []
    while True:
        samples.append(sample)
        if sample['next'] == '':
            break
        sample = nusc.get('sample', sample['next'])

    # 提取每帧的数据
    images_list = []
    poses_list = []
    occs_list = []

    for sample in samples:
        # ---- 前视相机 ----
        cam_front_token = sample['data'].get('CAM_FRONT')
        if cam_front_token is None:
            continue

        img = load_front_image(nusc, cam_front_token)
        images_list.append(img)

        # ---- 自车位姿 ----
        cam_data = nusc.get('sample_data', cam_front_token)
        ego_pose_data = nusc.get('ego_pose', cam_data['ego_pose_token'])

        x = ego_pose_data['translation'][0]
        y = ego_pose_data['translation'][1]
        # yaw from quaternion: yaw = 2 * arctan2(wz, w)
        qx, qy, qz, qw = ego_pose_data['rotation']
        yaw = 2 * np.arctan2(qz, qw)

        poses_list.append([x, y, yaw])

        # ---- LiDAR 转占用网格 ----
        lidar_token = sample['data'].get('LIDAR_TOP')
        if lidar_token:
            lidar_data = nusc.get('sample_data', lidar_token)
            lidar_path = os.path.join(dataroot, lidar_data['filename'])
            if os.path.exists(lidar_path):
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
                occ = lidar_to_occupancy(points)
            else:
                occ = np.zeros((Z_BINS, *BEV_GRID), dtype=bool)
        else:
            occ = np.zeros((Z_BINS, *BEV_GRID), dtype=bool)
        occs_list.append(occ)

    # 保存
    save_dir = Path(dataroot) / 'v1.0-mini' / split / 'scenes'
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_dir / f'{scene_name}.npz',
        images=np.stack(images_list, axis=0).astype(np.uint8),
        ego_pose=np.stack(poses_list, axis=0).astype(np.float32),
        occupancy=np.stack(occs_list, axis=0).astype(bool),
        timestamps=np.arange(len(images_list)).astype(np.int64),
    )

    return len(images_list)


def main():
    parser = argparse.ArgumentParser(description="Preprocess nuScenes for DriveWorld")
    parser.add_argument("--dataroot", type=str, default=DEFAULT_DATAROOT,
                        help="nuScenes data root directory")
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION,
                        help="nuScenes version (v1.0-mini or v1.0-trainval)")
    parser.add_argument("--scenes", type=int, default=None,
                        help="Process only first N scenes (for debugging)")
    args = parser.parse_args()

    dataroot = args.dataroot
    version = args.version

    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("Error: nuscenes-devkit not installed.")
        print("Install with: pip install nuscenes-devkit")
        print("Or download nuScenes and place in data/nuscenes/v1.0-mini/")
        sys.exit(1)

    print(f"Initializing nuScenes: {dataroot}/{version}")
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    except Exception as e:
        print(f"Error loading nuScenes: {e}")
        print("Make sure nuScenes data is downloaded and extracted correctly.")
        print("Expected structure: {dataroot}/{version}/samples/, maps/, sweeps/")
        sys.exit(1)

    scenes = nusc.scene
    if args.scenes:
        scenes = scenes[:args.scenes]

    print(f"Processing {len(scenes)} scenes...")
    total_frames = 0

    num_val_scenes = max(1, len(scenes) // 5)
    for idx, scene in enumerate(tqdm(scenes, desc="Scenes")):
        split = "val" if idx >= len(scenes) - num_val_scenes else "train"
        frames = process_scene(nusc, scene, split, dataroot)
        total_frames += frames

    print(f"\nDone! Processed {total_frames} frames across {len(scenes)} scenes.")
    print(f"Output saved to: {dataroot}/{version}/train/scenes/ and val/scenes/")
    print("\nNow run: python scripts/train.py --config configs/occworld.yaml")


if __name__ == '__main__':
    main()
