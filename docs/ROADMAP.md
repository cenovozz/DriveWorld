# DriveWorld 后续工作路线图

> 目标：把当前 MVP 推进到「可复现、可展示、可面试」的完整项目。
> 本文档与代码现状对齐，更新于 2026-08-15。

## 如何使用本文档

- 状态标记：`[ ]` 未开始，`[~]` 进行中，`[x]` 已完成。
- 每个阶段都包含：目标、输入、具体任务、验收标准、交付物、风险。
- 建议先读「当前基线」和「关键路径」，再按阶段顺序推进。
- 与服务器部署相关的手把手操作见 `docs/DEPLOY.md`，本文只写决策和验收标准。

---

## 当前基线（先对齐现实）

下表是截至 2026-08-15 的实际代码状态，后续计划都基于它展开。

| 模块 | 现状 | 关键文件 | 状态 |
|------|------|---------|------|
| 配置系统 | YAML + dataclass，支持 occworld/diffusion | `configs/*.yaml`, `driveworld/utils/config.py` | ✅ 可用 |
| 数据加载 | `NuScenesWorldModelDataset`；支持单相机 `(T,3,H,W)` 与多相机 `(T,6,3,H,W)`，无 `.npz` 时自动合成随机数据 | `driveworld/data/dataset.py` | ✅ 可跑，待真实验证 |
| 预处理 | `preprocess_nuscenes.py` 读取 6 相机并保存内参/外参 + LIDAR_TOP | `scripts/preprocess_nuscenes.py` | ⚠️ 未在真实数据上验证 |
| OccWorld | 单次 Transformer 解码，非严格自回归 | `driveworld/models/occworld.py` | ✅ 可训练，约 36M |
| DriveDiffuser | 2D UNet + DDPM/DDIM，50 步采样 | `driveworld/models/diffusion.py` | ✅ 可训练，约 49M |
| 编码器 | `ConvBEVEncoder` 已支持 6 相机 mean/sum 融合；`BEVEncoder` 已实现 LSS depth net + frustum splat | `driveworld/models/encoder.py` | ⚠️ LSS 已实现，待真实数据验证 |
| 训练 | AMP、warmup+cosine、checkpoint、TensorBoard | `driveworld/training/trainer.py` | ✅ 可用；缺 EMA/早停 |
| 评估 | mIoU、Dice、PSNR、per-step IoU、可视化 | `driveworld/eval/evaluator.py` | ✅ 可用；缺报告生成 |
| 工程 | CI、Docker、pre-commit、CLI、部署文档 | `.github/workflows/ci.yml`, `Dockerfile` | ✅ 基础完整 |
| 论文/展示 | 无技术报告、无 Gradio、无 CARLA 闭环 | — | ❌ 未开始 |

### 需要先修正的认知偏差

1. `build_encoder()` 已支持 `cnn/mean` 多相机 ConvBEV 融合，`BEVEncoder` 已接入 `cam_intrinsics/cam_extrinsics` 实现 LSS 投影；但 LSS 路径尚未在真实数据上跑出指标，先标注「已实现、未验证」，不要宣称有提升。
2. 当前 `OccWorld` 的 `forward` 是「BEV tokens + future ego tokens 一次性读未来 token」，不是真正逐帧自回归 rollout；简历和文档要描述成 one-shot/teacher-forcing 解码，避免面试被追问。
3. 扩散评估里用 `pred_logits.unsqueeze(2).expand(...)` 再 `argmax(dim=2)`，对概率图不严谨；应改为 `(pred > threshold)` 或显式定义二类 logits。
4. mini 数据集只有 10 个 scene，预处理按 8:2 分 train/val；完整 trainval 才有 ~700 train / ~150 val。原路线图里「~700 个 .npz」只适用于完整版。
5. `scripts/preprocess_nuscenes.py` 中的 `FRAME_INTERVAL = 2` 实际未被使用，需要清理或真正实现跳帧。

### Definition of Done（总体验收）

- [ ] 用 nuScenes mini 完整跑通 OccWorld 和 DriveDiffuser，得到可复现指标。
- [ ] 至少完成 3 个有对比价值的消融实验，并用表格呈现。
- [ ] 有 GT vs 预测对比图、GIF、训练曲线。
- [ ] 有错误分析（不是只报 mIoU）。
- [ ] README 和 `docs/ROADMAP.md` 与实际代码一致。
- [ ] 有在线 Demo 或导出模型，能演示端到端效果。
- [ ] 面试清单里的每道题都有基于真实实验的回答。

---

## 关键路径与优先级

| 优先级 | 工作 | 理由 |
|--------|------|------|
| P0 | 真实数据闭环 + 两个基线 | 没有真实指标，后续都无意义 |
| P0 | 多相机融合 | 收益最大、代码预留最多、故事性最强 |
| P0 | 实验报告/错误分析 | 面试最需要「我知道模型哪里不行」 |
| P1 | Focal/时间加权 Loss | 低成本、高收益，适合作为第一个消融 |
| P1 | 时间建模增强 | 论文级贡献，但实现成本更高 |
| P1 | Gradio Demo | 展示价值高，工程成本低 |
| P2 | CARLA 闭环 | 最有说服力，但环境成本高，放后期 |
| P2 | 扩散加速/CFG/Hybrid | 加分项，依赖前面基线稳定 |

---

## 阶段一：数据闭环（第 1-2 周）

> 目标：用真实 nuScenes 数据完整训练一轮，拿到可展示的量化结果。
> 退出标准：OccWorld 和 DriveDiffuser 各有一份 `best.pt`，且评估脚本能输出完整指标表和对比图。

### 1.0 阶段验收清单

- [ ] `python scripts/preprocess_nuscenes.py` 在真实 mini 数据上成功结束。
- [ ] 每个 scene 的 `.npz` 都能被 `np.load` 正常打开，且 keys 包含 `images/ego_pose/occupancy/timestamps`。
- [ ] 抽样 3 个 scene 做占位网格可视化，确认道路结构在 BEV 下合理。
- [ ] `python scripts/train.py --config configs/occworld.yaml` 能跑通 5 个 epoch 不报错。
- [ ] `python scripts/eval.py --checkpoint checkpoints/occworld/best.pt --config configs/occworld.yaml` 能输出 mIoU/PSNR。
- [ ] 两个范式的基线结果写进 `outputs/baseline_summary.md`。

### 1.1 nuScenes 完整数据预处理

**现状**：`driveworld/data/dataset.py` 在没有 `.npz` 文件时自动使用合成随机数据；`scripts/preprocess_nuscenes.py` 已写好，但只在代码层面完成，未跑真实数据 QA。

**需要做**：

- [x] 编写 `scripts/preprocess_nuscenes.py`（已完成）。
- [ ] 在服务器上运行预处理，生成每个 scene 的 `.npz` 文件。
- [ ] 修正/确认预处理参数表，与模型输入保持一致：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `IMAGE_SIZE` | `(224, 480)` | 与 `configs/*.yaml` 的 `image_size` 一致 |
| `BEV_GRID` | `(200, 200)` | 与 `bev_grid_size` 一致 |
| `BEV_RES` | `0.5` | 每像素 0.5m，覆盖 x/y ±50m |
| `Z_BINS` | `16` | 垂直方向 z ∈ [-4, 4]m |
| `XY_RANGE` | `50.0` | BEV 半宽 |
| 相机 | `CAM_FRONT` | 当前只读前视 |
| 点云 | `LIDAR_TOP` | 转换为二值占据网格 |
| 位姿 | `ego_pose` 的 `(x, y, yaw)` | 未来会扩展 6DoF |

- [ ] 清理 `FRAME_INTERVAL = 2` 未使用变量，改为显式说明采样率为 nuScenes keyframe 的 2Hz。
- [ ] 验证预处理后的占用标签质量：
  - 每个 `.npz` 的 `occupancy` 应该是 `(N_frames, 16, 200, 200)` 的 `bool`。
  - 抽 3 个 scene 画 BEV 最大投影图，和相机前视图对照。
  - 统计平均占据率，期望落在 3%-15% 之间（大部分是空的，少量被占据）。

**交付物**：

- `data/nuscenes/v1.0-mini/train/scenes/` 下生成约 8 个 train `.npz`（mini）。
- `data/nuscenes/v1.0-mini/val/scenes/` 下生成约 2 个 val `.npz`（mini）。
- 如果切到完整 `v1.0-trainval`：约 700 train / 150 val 个 `.npz`。

### 1.2 训练数据加载验证

```bash
# 验证数据 Pipeline 完整可用
python -c '
from driveworld.data import NuScenesWorldModelDataset
ds = NuScenesWorldModelDataset(root="data/nuscenes", num_past_frames=3, num_future_frames=6)
print(f"{len(ds)} training samples")
for i in range(3):
    s = ds[i]
    occ = s["future_occupancy"]
    print(f"Sample {i}: occupancy rate = {occ.float().mean():.3f}")
'
```

- 期望：`len(ds) > 0`，占据率在 3%-15% 之间。
- 额外检查：`past_images` 为 `(3, 3, 224, 480)`，`future_occupancy` 为 `(6, 16, 200, 200)`。
- 如果仍然合成数据，说明 `.npz` 路径或 split 目录不匹配，优先检查 `root/version/split/scenes`。

### 1.3 首次完整训练（OccWorld 基线）

```bash
# 先跑 5 epoch smoke test，确认无 OOM/NaN
python scripts/train.py --config configs/occworld.yaml

# 完整 100 epoch
python scripts/train.py --config configs/occworld.yaml
```

- 训练配置：batch size 4、lr 1e-4、warmup 5 epoch、cosine 衰减、AMP。
- 监控：`tensorboard --logdir logs/occworld`，重点看 `train/loss_step`、`val/loss`、`val/miou`。
- 资源预估：3090 24GB 约 6-8 小时；OOM 时先把 `batch_size` 降到 2，再考虑 `bev_grid_size` 降到 `[100, 100]`。

**预期结果**：

- mIoU: 0.35 - 0.45。
- IoU@t0: ~0.50，IoU@t_final: ~0.30（随时间衰减）。
- 远未来帧质量明显下降是正常现象，也是后续优化的方向。

### 1.4 训练第二个模型：DriveDiffuser 基线

```bash
python scripts/train.py --config configs/diffusion.yaml
```

- 训练配置：batch size 2、lr 2e-4、warmup 10 epoch、150 epoch。
- 评估采样：默认 DDIM 50 步；先把 `num_inference_steps` 固定为 50，后续再做加速消融。
- 修复扩散评估口径：将 `scripts/eval.py` 中扩散分支改成对采样概率图做 `(pred > 0.5)`，并记录 `threshold=0.5` 到结果文件。

### 1.5 基线结果归档

**做一张对比表**，同时放进 README 和简历：

| 指标 | OccWorld | DriveDiffuser | 说明 |
|------|----------|---------------|------|
| mIoU | 待填 | 待填 | 越高越好 |
| PSNR | 待填 | 待填 | 越高越好 |
| IoU@t0 | 待填 | 待填 | 短期预测质量 |
| IoU@t_final | 待填 | 待填 | 长期预测质量 |
| 参数量 | ~36M | ~49M | 部署成本 |
| 推理方式 | 单次前向 | DDIM 50 步 | 实时性差异 |
| 训练时长 | 待填 | 待填 | 成本 |

**交付物**：`outputs/baseline_summary.md`，附 3 个样例的 GT vs 预测对比图。

---

## 阶段二：模型增强（第 2-4 周）

> 目标：在基线基础上做可论证的改进，每个改进都有消融实验支撑。
> 退出标准：至少完成 2.1 多相机融合和 2.3 损失函数优化，且各有一个对比实验。

### 2.0 实验管理（先做，否则后期会乱）

> 状态：代码已落地；多相机 OccWorld 已在 AutoDL 完成 100 epochs 训练。

- [x] 固定随机种子：`training.seed=42`，`set_seed()` 已设置 Python/NumPy/PyTorch 种子。
- [x] DataLoader worker seed 可控：`create_dataloader(seed=...)` 已设置 `generator` 与 `worker_init_fn`。
- [x] 实验命名：`{paradigm}_{改动}_{关键超参}_{日期}`，统一由 `scripts/run_experiment.py --name` 传入。
- [x] 每个实验归档：`outputs/experiments/{name}/` 保存 `config.yaml`、`effective_config.yaml`、`run_cmd.txt`、`checkpoints/`、`logs/`、`eval/`、`metrics.json`。
- [x] `.gitignore` 已放行实验元数据，仍忽略 checkpoint、log、评估图大文件。
- [ ] 接 wandb（可选），或每次实验结束后备份 `logs/` 的 TensorBoard event。

**具体操作**

```powershell
# 统一用实验 runner，不再直接跑 scripts/train.py
python scripts/run_experiment.py --config configs/occworld.yaml --name occworld_baseline_tpast3_20260815 --seed 42 --eval

# 训练结束后检查归档
Get-ChildItem outputs/experiments/occworld_baseline_tpast3_20260815
```

### 2.1 多相机融合（最重要）

**状态：数据层 + ConvBEV 融合路径已落地；LSS 投影路径已实现，尚未在真实数据上验证。**

**已改动**

- `scripts/preprocess_nuscenes.py`：读取 6 相机，保存 `images (N,6,3,H,W)`、`cam_intrinsics (N,6,3,3)`、`cam_extrinsics (N,6,4,4)`；内参已按 resize 后的 `224x480` 缩放，保证 LSS 几何正确。
- `driveworld/data/dataset.py`：`past_images` 变为 `(T,6,3,H,W)`，并输出 `past_intrinsics/past_extrinsics`；保留单相机旧 `.npz` 兼容。
- `driveworld/models/encoder.py`：`ConvBEVEncoder` 支持 6 相机 `mean/sum` 融合；`BEVEncoder` 新增 `_forward_multicam_lss` + `_splat_camera`（depth net + frustum 投影 + BEV splat）。
- `driveworld/models/occworld.py`、`diffusion.py`、`training/trainer.py`、`eval/evaluator.py`、`data/utils.py`：标定参数已全链路透传。
- `configs/lss_occworld.yaml`、`configs/lss_diffusion.yaml` 已新增（`fusion_method: lss`）。

**剩余操作（按顺序）**

1. 在真实数据上重新预处理（内参缩放需要重跑）：
   ```powershell
   python scripts/preprocess_nuscenes.py --dataroot data/nuscenes --version v1.0-mini
   ```

2. 抽样验证 `.npz` 的 shape 和 keys：
   ```powershell
   python -c "import numpy as np; d=np.load('data/nuscenes/v1.0-mini/train/scenes/<scene>.npz'); print(d['images'].shape, d['cam_intrinsics'].shape, d['cam_extrinsics'].shape)"
   ```
   期望输出：`images (N,6,3,224,480)`、`cam_intrinsics (N,6,3,3)`、`cam_extrinsics (N,6,4,4)`。

3. 先跑 LSS 1 epoch smoke train：
   ```powershell
   python scripts/run_experiment.py --config configs/lss_occworld.yaml --name occworld_lss_smoke_20260815 --seed 42
   ```
   显存不足时把 `batch_size` 从 2 降到 1；torchvision 无法下载 ImageNet 权重时把 `encoder.pretrained` 改为 `false`。

4. 跑「ConvBEV mean vs LSS」同口径消融：同一 seed、同一 loss、同一 `bev_h/bev_w=16`，只改 `encoder.fusion_method`，比较 `metrics.json` 的 `miou`/`dice`/`PSNR`。

**LSS 实现要点与已知瓶颈**

- `_splat_camera` 用内参把 feature-grid 像素反投影到 camera 坐标，再经 camera-to-ego 外参 splat 到共享 BEV，6 相机在 BEV 网格上按深度概率加权累加。
- 兼容性规则：`num_cameras=1` 走原单相机路径，旧 checkpoint 不受影响。
- 已知瓶颈：当前 BEV 只有 `16x16`、`bev_range=50`，每格 `6.25m`，几何信息被高度量化；若 LSS 提升不明显，优先把 `encoder.bev_h/bev_w` 提到 `32x32` 或 `50x50` 再做对比，而不是否定 LSS 本身。

**实测结果（AutoDL, 2026-08-15）**

| 指标 | 单相机 baseline | 6 相机 multi-camera | 6 相机 LSS | 6 相机 LSS (32x32 BEV) |
|------|----------------|---------------------|-------------|------------------------|
| 实验 | `occworld_baseline_tpast3_20260815_v2` | `occworld_multicam_tpast3_20260815` | `occworld_lss_tpast3_20260816_v2` | `occworld_lss_bev32_20260822` |
| 配置 | 1 camera, batch 4 | 6 cameras, ConvBEV mean, batch 2 | 6 cameras, LSS splat, batch 2 | 6 cameras, LSS splat, 32x32 BEV, batch 1 |
| `val/mIoU` | 0.5613 | 0.5607 | 0.5607 | 0.5608 |
| `occupied IoU` | 0.1347 | 0.1320 | 0.1322 | 0.1324 |
| `IoU avg` | 0.1347 | 0.1320 | 0.1322 | 0.1324 |
| `PSNR` | 19.164 | 19.734 | 19.700 | 19.710 |
| `MSE` | 0.01213 | 0.01065 | 0.01073 | 0.010709 |
| `IoU@t0` | 0.1326 | 0.1298 | 0.1298 | 0.1299 |
| `IoU@tmid` | 0.1343 | 0.1303 | 0.1305 | 0.1307 |
| `IoU@tfinal` | 0.1375 | 0.1355 | 0.1358 | 0.1357 |

结论：`mean`、LSS 16x16、LSS 32x32 三条多相机路径的 mIoU 都和单相机 baseline 基本持平，PSNR/MSE 略有改善；把 BEV 从 16x16 提到 32x32 也没有改变 occupied IoU，说明瓶颈不只是 BEV 量化。真正该看的是 `occupied IoU`（约 0.13），下一步要排查标签稀疏/类别不平衡，以及 depth head 是否学到了有效几何。

**预期提升**：mIoU +3~5 个百分点；目前所有路径都未达到，且提高 BEV 分辨率已排除单纯量化瓶颈。
**风险**：LSS 训练不稳定，先固定 backbone 只训 depth head；结论已同步到 README。下一轮做标签/损失诊断：统计 occupied 体素占比、对比 focal alpha/gamma、可视化 depth map 与 BEV 特征。
### 2.2 时间建模增强

**现状**：`ConvBEVEncoder` 把过去 3 帧特征做均值池化，丢失时序信息；`OccWorld` 也只是把 ego token 拼进序列，不是真正时序演化。

**改进方案（按推荐顺序）**：

1. **时序 Transformer**：保留 T_past 帧的 BEV 特征，用 temporal self-attention 融合。
   - 输入 `(B, T_past, C, H, W)` 重排为 `(B, H*W, T_past, C)` 后做跨帧 attention。
   - 替换 `ConvBEVEncoder` 的 `mean(dim=1)`。

2. **ConvGRU / ConvLSTM**：用循环网络建模时序演化。
   ```python
   class TemporalFusion(nn.Module):
       def __init__(self, input_dim=128, hidden_dim=128):
           super().__init__()
           self.conv_gru = ConvGRU(input_dim, hidden_dim)
       def forward(self, bev_seq):  # (B, T, C, H, W)
           return self.conv_gru(bev_seq)  # (B, C, H, W)
   ```

3. **真自回归 rollout（进阶）**：训练时 teacher forcing，推理时把上一帧预测作为下一帧输入。
   - 先在 `OccWorld.generate` 实现逐帧 loop，再对比 one-shot 和 rollout 的误差累积曲线。

### 2.3 损失函数优化

**状态：Focal Loss、时间加权 Loss 已实现并接入配置；消融实验尚未跑。**

**已改动**

- `driveworld/training/losses.py`：新增 `FocalLoss`；`OccupancyLoss` 支持 `use_focal/focal_alpha/focal_gamma`、`class_weights`、`temporal_weights`。
- `driveworld/utils/config.py`：`LossConfig` 已暴露上述开关。
- `configs/occworld.yaml`：当前启用 `use_focal: true` 和 `temporal_weighting: true`。
- `configs/default.yaml`、`configs/diffusion.yaml`：保留 CE 基线开关，便于消融。

**具体操作**

1. 建三个对照配置并逐一跑实验，只改 loss：
   ```powershell
   python scripts/run_experiment.py --config configs/occworld.yaml --name occworld_focal_tw_20260815 --seed 42 --eval
   # 将 configs/occworld.yaml 的 use_focal/temporal_weighting 改为 false 后：
   python scripts/run_experiment.py --config configs/occworld_ce.yaml --name occworld_ce_20260815 --seed 42 --eval
   ```

2. 验收项：比较 `outputs/experiments/*/metrics.json` 中的 `miou`、`dice`，并记录每个实验的 `train/loss` 曲线是否稳定。

**未实现（后续）**

- 类别权重：数据统计得到 `class_weights` 后再传入 `OccupancyLoss`。
- 辅助一致性损失：对同一输入两次前向约束输出一致，尚未实现。

**验收**：每个 Loss 变体单独跑一个消融，记录 mIoU、Dice、训练稳定性。
### 2.4 扩散模型加速

**现状**：`DriveDiffuser.sample` 用 DDIM 50 步。

**改进方案（先做便宜的，再做难的）**：

1. **DDIM 步数扫描**：把 `num_inference_steps` 设为 5/10/20/50，画「步数 vs 指标 vs 耗时」曲线。
2. **Classifier-free guidance**：训练时随机 drop 条件，采样时 `eps = eps_uncond + w * (eps_cond - eps_uncond)`。
3. **Progressive Distillation**：用 50-step teacher 蒸馏成 4-step student。
4. **Consistency Models**：直接从噪声映射到数据，1-2 步采样。

**预期效果**：推理从 50 步降到 4-8 步，质量下降控制在 1-2 个点。

### 2.5 混合模型（可选，进阶）

**思路**：用 OccWorld 提供低成本初值，DriveDiffuser 对初值做 refinement，或反之用扩散生成多样化候选、OccWorld 做一致性筛选。

- 先定义接口：两个模型都能接收 `(past_images, past_ego, future_ego)` 并输出 occupancy。
- 再做 ensemble：加权平均或 logit 融合，跑一个对比实验即可。
- 不急于写新架构，ensemble 的收益已经能讲清楚「范式互补」。

### 2.6 不确定性估计（可选）

- OccWorld：用 MC Dropout 或 ensemble 方差估计每个体素的不确定性。
- DriveDiffuser：多次采样的方差天然是不确定性。
- 输出 `uncertainty_map`，可视化「模型对哪里不确定」，是很好的面试素材。

---

## 阶段三：实验与评估（第 4-6 周）

> 目标：完成系统性的消融实验，产出可用于简历和面试的量化结果。
> 退出标准：有 7 组实验的结果表、报告生成脚本、至少 5 个失败案例。

### 3.0 评估协议（先统一口径）

- 指标定义：
  - mIoU：`compute_iou` 的 `miou`（当前对 class 0/1 求平均）。
  - PSNR：`compute_video_metrics` 的 `psnr`，在二值占据图上计算。
  - IoU@t0 / IoU@tmid / IoU@tfinal：per-step IoU。
- 固定 val split：mini 用预处理产生的 2 个 val scene；完整版用官方 val。
- 固定推理超参：OccWorld 单次前向；DriveDiffuser 默认 DDIM 50 步、`eta=0`。
- 固定 threshold：扩散输出概率图用 0.5 二值化。
- 所有实验至少跑 3 个不同 seed 或报告单 seed 并说明局限。

### 3.1 消融实验矩阵

| 实验编号 | 变量 | 配置 | 预期影响 | 状态 |
|---------|------|------|---------|------|
| EXP-00 | 基线 | occworld.yaml / diffusion.yaml | 参考值 | [ ] |
| EXP-01 | 过去帧数 | T_past = 1, 3, 5 | 3 帧最佳，再多边际递减 | [ ] |
| EXP-02 | 预测帧数 | T_future = 3, 6, 10 | 6 帧后 mIoU 快速下降 | [ ] |
| EXP-03 | BEV 分辨率 | 100x100, 200x200, 400x400 | 200 性价比最高 | [ ] |
| EXP-04 | 编码器类型 | ConvBEV vs LSS vs BEVFormer | BEVFormer 更准但更慢 | [ ] |
| EXP-05 | Loss 类型 | CE only, CE+Dice, Focal+Dice | Focal 可能提升 2-3 点 | [ ] |
| EXP-06 | 多相机 | 单前视 vs 6 相机 | +3~5 点 mIoU | [ ] |
| EXP-07 | 范式对比 | OccWorld vs Diffusion | 验证论文核心假设 | [ ] |
| EXP-08 | 时间建模 | mean-pool vs temporal attn vs ConvGRU | 远未来提升更明显 | [ ] |
| EXP-09 | 采样步数 | DDIM 5/10/20/50 | 4-8 步性价比高 | [ ] |

### 3.2 评估脚本增强

新增 `scripts/generate_report.py`：

```python
# 输入：checkpoint + config + val split
# 输出：
# 1. metrics.json 与 markdown 指标表
# 2. GT vs 预测对比图（每个场景选 3 个时间步）
# 3. IoU 随时间衰减曲线
# 4. 失败案例 Top-K（mIoU 最低的 5 个场景）
# 5. uncertainty_map（若模型输出支持）
```

- 复用 `driveworld/eval/visualize.py` 的 `visualize_occupancy_comparison`、`create_prediction_gif`、`plot_training_curves`。
- 输出目录统一为 `outputs/experiments/{exp_name}/`。

### 3.3 错误分析

不只报指标，还要分析模型在哪类场景犯错：

- 动态物体多（十字路口、拥堵）。
- 远距离（>30m）。
- 复杂几何（弯道、坡道）。
- 夜间 / 雨天（nuScenes 有这些场景）。
- 占据率极低或极高的场景。

**方法**：

- 按 `future_occupancy` 的占据率分桶统计 mIoU。
- 按场景 token 聚类，把 Top-K 失败案例可视化。
- 记录错误模式：FP（预测有障碍实际没有）还是 FN（实际有障碍没预测到）。

**产出**：`outputs/error_analysis.md`，这是面试时最有深度的讨论点。

### 3.4 指标口径与统计显著性

- 对关键对比（例如多相机 vs 单相机）跑 3 次不同 seed，报告 mean ± std。
- 如果无法多次跑，至少说明训练时间限制，并把该点写进 limitations。
- 区分「验证集指标」和「测试集指标」，不要在 val 上反复调参后宣称 test。

### 3.5 实验追踪

- 每个实验生成 `config.yaml` 快照 + `metrics.json` + `report.png`。
- 用 git 记录实验脚本变化，不要用未提交的临时改动训练。
- 建议在 `outputs/experiments/README.md` 维护一张「实验索引表」。

---

## 阶段四：闭环评估（第 6-8 周）

> 目标：在世界模型的实际应用场景中验证——能帮助规划器做更好的决策吗？
> 退出标准：CARLA 在线对比实验至少输出一组碰撞率/急刹/平均速度指标，或完成离线替代评估。

### 4.0 目标与前置条件

- 前置：已有一个能稳定推理的 OccWorld 或 DriveDiffuser checkpoint。
- 前置：已有多相机版本更好，否则先用前视简化闭环。
- 建议先做离线替代评估，再做 CARLA，降低环境成本。

### 4.1 CARLA 集成

**步骤**：

1. 安装 CARLA 0.9.x（建议 Linux 服务器 + Docker 镜像）。
2. 用 CARLA Python API 控制车辆和传感器：
   - RGB 相机：尺寸对齐 `224x480`，或在线 resize。
   - LiDAR：采集点云并转成 `16x200x200` 占据网格。
   - 位姿：记录 ego `(x, y, yaw)`。
3. 每 0.5 秒记录一帧，累积 3 帧作为 `past_images`，预测未来 6 帧 3D 占用。
4. 与真实发生的占用对比，计算在线 mIoU / per-step IoU。
5. 关键演示：如果预测「前面会有障碍物」，而当前规划器没反应，说明世界模型提前发现了危险。

**技术要点**：

- 模型用 PyTorch 在 GPU 上推理，CARLA 场景进程与模型推理进程解耦。
- 记录完整日志，便于离线复现和回放。
- 注意 CARLA 的相机外参和 nuScenes 不同，需要做坐标对齐。

### 4.2 规划器对比实验

```
实验组 A：只用当前感知
对照组 B：当前感知 + 世界模型预测的未来占用
评估：碰撞率、急刹次数、平均速度、规划成功时长
```

- 规划器可以先从规则式（如 PID 避障）开始，不必先上学习型规划器。
- 世界模型的未来占用可以转成 cost map，例如前方某体素占据概率 > 阈值就减速/绕行。
- 这是论文级贡献——「验证世界模型能提升规划安全性」。

### 4.3 评估指标与统计

| 指标 | 定义 | 目标 |
|------|------|------|
| collision_rate | 发生碰撞的 episode 比例 | B 组低于 A 组 |
| hard_brake_rate | 急刹次数/公里 | B 组低于 A 组 |
| avg_speed | 平均速度 km/h | B 组不低于 A 组太多 |
| route_completion | 完成路线比例 | B 组高于 A 组 |
| online_iou | 预测占用 vs 真实占用 | 越高越好 |

### 4.4 离线替代方案（如果 CARLA 成本高）

- 用 nuScenes 的 val scene 做「回放式闭环」：预测未来 6 帧后，把第 1 帧当作新观测的一部分，逐步 rollout。
- 计算 rollout 误差累积曲线，替代 CARLA 的一部分证明力。
- 优点：环境稳定、可复现；缺点：不能真正和规划器交互。

### 4.5 风险管理

- CARLA 环境搭建和版本兼容是最耗时部分，预留 1 周。
- 如果 GPU 服务器不便跑 GUI，用 `CARLA_HEADLESS` 或 Docker headless 模式。
- 如果闭环实验长期不收敛，先交付离线 rollout + 定性视频，闭环作为后续工作。

---

## 阶段五：工程完善（持续）

> 目标：让仓库达到「可开源、可复现、可协作」水平。
> 注意：本阶段大量工作已经完成，先打勾，再补缺口。

### 5.0 现状与缺口

| 项目 | 现状 | 还需做 |
|------|------|--------|
| docstring | 多数模块有 docstring | 补齐 `trainer.py`/`evaluator.py` 中缺漏部分 |
| CONTRIBUTING.md | 无 | [ ] 新增 |
| Issue/PR 模板 | 无 | [ ] 新增 `.github/ISSUE_TEMPLATE/` |
| API 文档 | 无 | [ ] mkdocs 或 Sphinx |
| Dependabot | 无 | [ ] 新增 `.github/dependabot.yml` |
| CI CPU | 已配置 ruff/mypy/pytest/codecov/black/isort | 保持 |
| CI GPU | 无 | [ ] self-hosted runner 或 GitHub Actions GPU |
| Docker | 已有 | [ ] 补充 docker-compose + volume 说明 |
| Demo | 无 | [ ] Gradio + HuggingFace Spaces |
| 导出 | 无 | [ ] ONNX / TorchScript |
| Profiling | 无 | [ ] torch.profiler 记录显存/耗时 |

### 5.1 代码仓库完善

- [ ] 补充所有公共函数的 docstring（Google style）。
- [ ] 添加 `CONTRIBUTING.md`。
- [ ] 添加 GitHub Issue / PR 模板。
- [ ] 生成 API 文档（推荐 mkdocs-material，简单且免费）。
- [ ] 配置 Dependabot 自动更新依赖。

### 5.2 CI/CD 完善

- [ ] GitHub Actions 加入 GPU 测试（self-hosted runner 或免费 GPU）。
- [ ] 加入 Docker build 检查，防止 Dockerfile 失效。
- [ ] pre-commit 加入 mypy（当前 CI 已跑 mypy，但用 `|| true` 放行；逐步消除错误后改为阻断）。
- [ ] 加入 benchmark 脚本，CI 跑一个小 batch 的性能 smoke test。

### 5.3 Demo 应用

- [ ] Gradio Web Demo：上传视频或选择 nuScenes sample → 展示 GT vs 预测 GIF。
- [ ] 部署到 HuggingFace Spaces（免费 CPU 可先做推理，慢一点没关系）。
- [ ] README 放「在线体验」链接。
- [ ] Demo 输入降级方案：如果 GPU 不够，用预计算好的对比 GIF 展示，避免 Spaces 超时。

### 5.4 模型导出

- [ ] ONNX 导出：`torch.onnx.export(model, dummy_input, 'model.onnx')`。
- [ ] TorchScript 导出：`traced = torch.jit.trace(model, dummy_input)`。
- [ ] 记录导出前后输出误差，确保 `onnxruntime` 结果和 PyTorch 一致。
- [ ] 面试时可以聊部署和推理优化：动态 shape、fp16、batch=1 latency。

### 5.5 性能与显存优化

- [ ] 用 `torch.profiler` 找瓶颈（数据加载 vs encoder vs transformer/UNet vs decoder）。
- [ ] 记录 OccWorld 与 DriveDiffuser 的参数量、FLOPs、推理延迟、峰值显存。
- [ ] 生成一张「速度 vs 质量」对比表，作为工程能力证据。

---

## 阶段六：论文与展示（可选，进阶）

> 目标：把项目整理成可被审阅和传播的材料，而不是只留代码。
> 退出标准：4-6 页技术报告 PDF + 一张 A0 海报 + 5 分钟讲解材料。

### 6.1 技术报告

推荐用 LaTeX（arXiv 风格），章节：

```
1. Introduction（为什么自动驾驶需要世界模型）
2. Related Work（OccWorld, DriveDreamer, GAIA-1, UniSim...）
3. Method（OccWorld + DriveDiffuser 统一框架设计）
4. Experiments（消融实验 + 对比分析 + 错误分析）
5. Conclusion（局限性与未来工作）
```

- 不一定要投稿，但生成 PDF 放在 GitHub 上，简历里引用，含金量翻倍。
- 图片规范：统一分辨率、统一 colormap、图注说明「红=FP，绿=TP，蓝=FN」。

### 6.2 可视化海报

用 matplotlib/Plotly 生成一张信息图：

- 左上：模型架构示意图。
- 右上：GT vs 预测对比 GIF。
- 左下：IoU 随时间衰减曲线。
- 右下：失败案例（模型预测错了什么）。
- 格式化为 A0 海报，面试时打印出来。

### 6.3 讲解材料

- 准备 5 分钟口头讲解：问题 → 方法 → 实验 → 教训。
- 录制一段 screen demo，作为 README 的 media。
- 把技术报告、海报、Demo 链接统一放在 README 顶部。

---

## 面试准备清单

### 能回答的核心问题

1. **为什么选世界模型这个方向？**
   > 自动驾驶的传统 Pipeline（感知→预测→规划）是独立模块。世界模型把预测提升到 3D 场景级，能直接输出「未来路况」，对规划的价值比单纯检测 2D 框大得多。

2. **你的项目与 OccWorld 论文有什么区别？**
   > 论文只做了自回归方案。我的贡献是把自回归和扩散两个流派统一在一个框架里直接对比，并加入多相机融合、时间建模、不确定性估计等实用组件。

3. **训练中遇到的最大困难？**
   > 3D 占用的稀疏性（90%+ 是空的）。普通 CE 会让模型「偷懒」全输出空。解决方法是 Dice Loss + Focal Loss。扩散模型里还有 3D UNet 显存优化——把 `(T, Z)` 展平成通道，用 2D UNet 代替 3D UNet。

4. **如果我让你加一个新功能，你打算怎么做？**
   > 举多相机融合：数据层加 5 个相机读取 → 改造 `build_encoder` 支持 `num_cameras=6` → splat 阶段融合到同一 BEV → 写新 YAML → 跑消融对比。

5. **如何评估你的模型不是过拟合？**
   > 固定 val split、固定 seed、报告 per-step IoU 而非只报平均；关键实验跑 3 个 seed；错误分析证明模型学到的是场景结构而非记忆。

6. **模型部署时考虑什么？**
   > ONNX/TorchScript 导出、fp16、batch=1 延迟、DDIM 步数 vs 质量权衡、CPU/GPU 资源差异。

### 简历亮点提炼

```
DriveWorld | 自动驾驶世界模型框架 | PyTorch

- 独立实现 OccWorld (ECCV'24) 和 DDPM/DDIM 双范式世界模型，统一框架直接对比
- 用 nuScenes 真实数据训练，评估 3D 占用预测（mIoU, PSNR, 时间衰减 IoU）
- 多相机 BEV 融合 + 时序建模增强 + Focal Loss + CARLA 闭环评估
- 生产级工程：混合精度训练、CI/CD、Docker、Type Hints 全覆盖
```

### STAR 叙述模板（面试时套用）

- **Situation**：单前视模型在远未来和侧向场景 mIoU 快速下降。
- **Task**：在不显著增加训练成本的前提下提升 360 度场景预测。
- **Action**：扩展数据层到 6 相机，改造 BEV 编码器做多相机 splat 融合，并跑单相机 vs 多相机消融。
- **Result**：mIoU +3~5 点，远未来 IoU 衰减变缓；同步输出对比图与错误分析。

---

## 时间线与里程碑

```
第 1 周  [==== 数据预处理 + OccWorld 基线 ====]
第 2 周  [==== DriveDiffuser 基线 + 基线报告 ====]
第 3 周  [==== 多相机融合 ====]
第 4 周  [==== 时序增强 + Focal Loss ====]
第 5 周  [==== 消融实验矩阵（7-9 组）====]
第 6 周  [==== 错误分析 + 技术报告初稿 ====]
第 7 周  [==== CARLA 闭环评估 ====]
第 8 周  [==== Gradio Demo + 导出模型 ====]
第 9 周  [==== README/海报/简历 polish ====]
第 10 周 [==== 面试复盘与补漏 ====]
```

### 里程碑退出条件

| 里程碑 | 退出条件 |
|--------|---------|
| M1 数据闭环 | 两个 baseline 都有 `best.pt` 和指标表 |
| M2 模型增强 | 多相机和 Loss 优化各有对比结果 |
| M3 实验报告 | 消融矩阵 + 错误分析 + 训练曲线齐全 |
| M4 闭环验证 | CARLA 或离线 rollout 有一组规划对比指标 |
| M5 展示 | Demo 在线可用，README 能 5 分钟讲完 |

---

## 风险登记表

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 服务器 OOM | 中 | 高 | batch 降 2，BEV 降到 100，用 AMP |
| 真实数据标签质量差 | 中 | 高 | 预处理 QA，抽 3 个 scene 可视化 |
| 多相机实现复杂 | 高 | 中 | 先只做 2-3 相机，逐步扩到 6 |
| LSS 训练不稳定 | 中 | 中 | 固定 backbone，只训 depth head |
| 扩散采样慢 | 高 | 中 | 先做 DDIM 步数扫描，再蒸馏 |
| CARLA 环境失败 | 高 | 中 | 用离线 rollout 替代，闭环降级 |
| 时间不够 | 高 | 高 | 按 P0→P1→P2 执行，砍掉混合模型 |

## 每周检查表

- [ ] 第 1 周：mini 数据预处理跑通，OccWorld 5 epoch smoke test 通过。
- [ ] 第 2 周：两个基线 `best.pt` 生成，`baseline_summary.md` 完成。
- [ ] 第 3 周：多相机数据加载和 `build_encoder` 改造完成。
- [ ] 第 4 周：Focal Loss 与时间加权 Loss 各完成一次实验。
- [ ] 第 5 周：消融矩阵至少 5 组实验归档。
- [ ] 第 6 周：错误分析 Top-K 案例与技术报告初稿完成。
- [ ] 第 7 周：CARLA 或离线 rollout 指标产出。
- [ ] 第 8 周：Gradio Demo 上线，ONNX 导出验证。
- [ ] 第 9 周：README、海报、简历同步更新。
- [ ] 第 10 周：按面试清单做一次完整 mock。

## 决策记录与命名规范

- 实验命名：`{paradigm}_{改动}_{关键超参}_{日期}`。
- checkpoint 保存：`checkpoints/{paradigm}/{exp_name}/best.pt`。
- 报告输出：`outputs/experiments/{exp_name}/report.md`。
- 可视化输出：`outputs/experiments/{exp_name}/figs/`。
- 任何实验改动先提交到 git，再开始训练。
- 指标只来自 `scripts/eval.py` 或 `scripts/generate_report.py`，不手工拼数字。

## 附录：相关论文阅读清单

| 论文 | 会议 | 与项目关系 | 必读程度 |
|------|------|-----------|---------|
| OccWorld | ECCV 2024 | 自回归世界模型基础 | ⭐⭐⭐⭐⭐ |
| DDPM | NeurIPS 2020 | 扩散模型理论基础 | ⭐⭐⭐⭐⭐ |
| DDIM | ICLR 2021 | 推理加速 | ⭐⭐⭐⭐ |
| LSS | ECCV 2020 | BEV 编码 | ⭐⭐⭐⭐ |
| BEVFormer | ECCV 2022 | 增强 BEV 编码 | ⭐⭐⭐ |
| DriveDreamer | arXiv 2023 | 扩散世界模型参考 | ⭐⭐⭐ |
| GAIA-1 | arXiv 2023 | 视频生成式世界模型 | ⭐⭐⭐ |
| UniSim | CoRL 2023 | 通用世界模拟器 | ⭐⭐⭐ |
| Focal Loss | ICCV 2017 | 改进损失函数 | ⭐⭐ |
| Progressive Distillation | ICLR 2022 | 扩散加速 | ⭐⭐ |
| Consistency Models | ICML 2023 | 扩散加速（替代 DDIM） | ⭐⭐ |
| SurroundOcc | ICCV 2023 | 多相机 3D 占用标签 | ⭐⭐⭐ |

## 备注：与 README 的同步

完成每个里程碑后，同步更新 README 的 Roadmap 小节和 Results 小节，避免简历引用过期指标。
