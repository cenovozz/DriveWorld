# DriveWorld 服务器部署完全指南

本指南覆盖从零开始将 DriveWorld 部署到云端 GPU 服务器的完整流程，面向第一次接触服务器的新手。

---

## 目录

- [1. 服务器选型与租用](#1-服务器选型与租用)
- [2. 连接到服务器](#2-连接到服务器)
- [3. 环境配置](#3-环境配置)
- [4. 上传项目代码](#4-上传项目代码)
- [5. 安装依赖](#5-安装依赖)
- [6. 数据集准备](#6-数据集准备)
- [7. 验证环境](#7-验证环境)
- [8. 启动训练](#8-启动训练)
- [9. 远程监控训练进度](#9-远程监控训练进度)
- [10. 训练完成后操作](#10-训练完成后的操作)
- [11. 常见问题排查](#11-常见问题排查)
- [12. 费用控制与关机](#12-费用控制与关机)

---

## 1. 服务器选型与租用

### 推荐平台：AutoDL（学生首选）

| 项目 | 说明 |
|------|------|
| 网址 | https://www.autodl.com |
| 注册 | 手机号注册 |
| 充值 | 建议首次充 30 元 |
| 推荐 GPU | **RTX 3090 (24GB 显存)** |
| 单价 | 约 1.5 ~ 2 元/小时 |
| 预估总费用 | 完整训练约 15-35 元 |

### 租用步骤

```
1. 登录 autodl.com
2. 点击「容器实例」->「租用新实例」
3. 选择配置：
   - GPU: RTX 3090（数量选 1）
   - 镜像: 社区镜像 -> 搜索「PyTorch 2.1.0」
   - 选「PyTorch 2.1.0 + Python 3.10(ubuntu22.04) + Cuda 12.1」
   - 数据盘: 默认 50GB（建议扩容到 100GB）
4. 点击「立即创建」
5. 等待 1-2 分钟，状态变为「运行中」
```

---

## 2. 连接到服务器

### 方式一：AutoDL 网页终端（推荐新手）

```
AutoDL 控制台 -> 你的实例 -> 点击「JupyterLab」
-> 点击顶部菜单「Terminal」-> 打开终端窗口
```

### 方式二：SSH 连接

```powershell
# 在 AutoDL 控制台复制 SSH 命令
ssh -p 12345 root@region-1.autodl.com
```

---

## 3. 环境配置

```bash
# 3.1 确认基础环境
python --version       # 应该是 Python 3.10.x
nvidia-smi             # 查看 GPU 型号和显存

# 3.2 确认 PyTorch（AutoDL 镜像已预装）
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# 期望输出: PyTorch 2.1.x / CUDA: True / GPU: NVIDIA GeForce RTX 3090

# 3.3 安装基础工具
apt-get update && apt-get install -y git tmux htop

# 3.4 进入数据盘目录（重启不丢失）
cd /root/autodl-tmp
```

---

## 4. 上传项目代码

### 方式 A：Git Clone（推送过代码后用）

```bash
cd /root/autodl-tmp
git clone https://github.com/cenovozz/DriveWorld.git
cd DriveWorld
```

### 方式 B：本地打包上传

**本地 PowerShell：**

```powershell
cd C:\Users\12742\Documents\Codex\2026-08-05\wep
tar --exclude='venv' --exclude='checkpoints' --exclude='logs' --exclude='outputs' --exclude='__pycache__' -czf driveworld.tar.gz DriveWorld/
```

上传：AutoDL 控制台 ->「文件管理」-> 上传 `driveworld.tar.gz`

**服务器端解压：**

```bash
cd /root/autodl-tmp
tar -xzf driveworld.tar.gz
cd DriveWorld
```

---

## 5. 安装依赖

```bash
cd /root/autodl-tmp/DriveWorld

# 安装核心依赖
pip install -e .

# 安装辅助工具
pip install tensorboard imageio nuscenes-devkit

# 验证
python -c "from driveworld import Config; print('DriveWorld OK')"
```

---

## 6. 数据集准备

### 6.1 下载 nuScenes mini

1. 访问 https://www.nuscenes.org/nuscenes#download
2. 注册账号 -> 登录 -> 同意条款
3. 复制 v1.0-mini 下载链接（带 token）

```bash
cd /root/autodl-tmp/DriveWorld
mkdir -p data/nuscenes

# 用你的 token 替换 YOUR_TOKEN
wget -O v1.0-mini.tgz "下载链接"
tar -xzf v1.0-mini.tgz -C data/nuscenes/
rm v1.0-mini.tgz

# 验证
ls data/nuscenes/v1.0-mini/
# 应该看到: samples/  maps/  sweeps/  v1.0-mini/
```

### 6.2 运行预处理

```bash
pip install nuscenes-devkit
python scripts/preprocess_nuscenes.py
# 将 LiDAR 点云转为 16x200x200 的 3D 占用网格
# 耗时约 10-20 分钟
```

---

## 7. 验证环境

```bash
cd /root/autodl-tmp/DriveWorld

# 验证数据加载
python -c "
from driveworld.data import NuScenesWorldModelDataset
ds = NuScenesWorldModelDataset(root='data/nuscenes', num_past_frames=3, num_future_frames=4)
print(f'Dataset: {len(ds)} samples')
sample = ds[0]
print(f'Image shape: {sample[\"past_images\"].shape}')
print(f'Occupancy shape: {sample[\"future_occupancy\"].shape}')
"

# 验证模型前向
python -c "
import torch
from driveworld.models import BEVEncoder, OccupancyDecoder, OccWorld
encoder = BEVEncoder(backbone='resnet18', pretrained=False, bev_feat_dim=128, bev_h=50, bev_w=50)
decoder = OccupancyDecoder(bev_feat_dim=128, num_classes=2, num_z=16, bev_h=50, bev_w=50)
model = OccWorld(encoder=encoder, decoder=decoder, hidden_dim=256, num_layers=2, num_heads=4)
out = model(torch.randn(1,3,3,224,480), torch.randn(1,3,3), torch.randn(1,4,3))
print(f'Model output: {out[\"occupancy_pred\"].shape}')
print('All checks passed!')
"
```

---

## 8. 启动训练

### 使用 tmux 保持训练不中断

```bash
# 创建 tmux 会话
tmux new -s train

# 进入环境和项目
conda activate base   # AutoDL 通常用 base 环境
cd /root/autodl-tmp/DriveWorld

# 启动训练
python scripts/train.py --config configs/occworld.yaml
```

```
tmux 快捷键:
  Ctrl+B, D          = 安全退出（训练继续跑）
  tmux attach -t train = 重新进入查看
  Ctrl+C              = 停止训练
```

### 同时训练两种模型

```bash
# 终端 1: OccWorld
tmux new -s occworld
cd /root/autodl-tmp/DriveWorld
python scripts/train.py --config configs/occworld.yaml

# 终端 2: DriveDiffuser
tmux new -s diffusion
cd /root/autodl-tmp/DriveWorld
python scripts/train.py --config configs/diffusion.yaml
```

---

## 9. 远程监控训练进度

```bash
# 启动 TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
# AutoDL 控制台 ->「自定义服务」-> 添加端口 6006 -> 点链接查看

# 查看 GPU 使用
watch -n 2 nvidia-smi

# 查看 checkpoint
ls -lh checkpoints/occworld/
```

---

## 10. 训练完成后的操作

```bash
cd /root/autodl-tmp/DriveWorld

# 评估
python scripts/eval.py \
    --checkpoint checkpoints/occworld/best.pt \
    --config configs/occworld.yaml \
    --output-dir outputs/eval_occworld \
    --num-vis 10

# 生成可视化 GIF
python scripts/visualize.py \
    --checkpoint checkpoints/occworld/best.pt \
    --config configs/occworld.yaml \
    --output-dir outputs/viz \
    --num-samples 5

# 打包下载
tar -czf results.tar.gz outputs/ checkpoints/occworld/best.pt logs/
# AutoDL 网页「文件管理」中下载 results.tar.gz
```

---

## 11. 常见问题

### Q: CUDA not available

```bash
pip uninstall torch torchvision -y
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### Q: 显存不足 (OOM)

```bash
# 编辑 configs/occworld.yaml
# batch_size: 2     （从 4 减小）
# bev_grid_size: [100, 100]  （从 200 减小）
```

### Q: 训练速度太慢

```bash
# 增加数据加载线程
# configs/occworld.yaml -> num_workers: 8
```

### Q: SSH 断开训练中断

```bash
# 永远在 tmux 里启动训练！
tmux new -s train
python scripts/train.py ...
# Ctrl+B, D 退出（不要直接关终端）
```

### Q: ImportError: nuscenes

```bash
pip install nuscenes-devkit
# 如果没有数据，dataset.py 会自动切合成数据模式，不影响调试
```

---

## 12. 费用控制

### 关机

```
AutoDL 控制台 -> 实例 ->「关机」
数据盘内容不会丢失
```

### 省钱技巧

1. 配环境时用「无卡模式」（免费）
2. 先跑 10 个 epoch 验证 Pipeline
3. 夜里训练可能有折扣

### 费用预估

| 阶段 | 时长 | 费用 |
|------|------|------|
| 环境配置 | 1h | ~2元 |
| 数据预处理 | 0.5h | ~1元 |
| OccWorld 100 epoch | 6-8h | ~12-16元 |
| DriveDiffuser 100 epoch | 8-10h | ~16-20元 |
| 评估/可视化 | 0.5h | ~1元 |
| **总计** | **~18h** | **~35元** |

---

## 附录：完整命令速查

```bash
# 一键环境
apt-get update && apt-get install -y git tmux
cd /root/autodl-tmp
git clone https://github.com/cenovozz/DriveWorld.git
cd DriveWorld && pip install -e .
pip install tensorboard imageio nuscenes-devkit

# 数据
mkdir -p data/nuscenes
# 下载 v1.0-mini -> 解压 -> python scripts/preprocess_nuscenes.py

# 训练
tmux new -s train
python scripts/train.py --config configs/occworld.yaml

# 监控
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# 完成
python scripts/eval.py --checkpoint checkpoints/occworld/best.pt --config configs/occworld.yaml
tar -czf results.tar.gz outputs/ checkpoints/
```
