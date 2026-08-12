# DriveWorld 后续工作路线图

本文档规划了 DriveWorld 项目从当前 MVP 到可写入简历的完整版之间的所有工作内容，按优先级排序。

---

## 阶段一：数据闭环（第 1-2 周）

> 目标：用真实 nuScenes 数据完整训练一轮，拿到可展示的量化结果。

### 1.1 nuScenes 完整数据预处理

**现状**：`dataset.py` 在没有 `.npz` 文件时自动使用合成随机数据。

**需要做**：

- [x] 编写 `scripts/preprocess_nuscenes.py`（已完成）
- [ ] 在服务器上运行预处理，生成每个 scene 的 `.npz` 文件
- [ ] 验证预处理后的占用标签质量（可视化几个 scene，确认占据网格与真实场景对应）

**交付物**：`data/nuscenes/v1.0-mini/train/scenes/` 下生成 ~700 个 `.npz` 文件

### 1.2 训练数据加载验证

```bash
# 验证数据 Pipeline 完整可用
python -c "
from driveworld.data import NuScenesWorldModelDataset
ds = NuScenesWorldModelDataset(root='data/nuscenes', num_past_frames=3, num_future_frames=6)
print(f'{len(ds)} training samples')
for i in range(3):
    s = ds[i]
    occ = s['future_occupancy']
    print(f'Sample {i}: occupancy occupancy rate = {occ.float().mean():.3f}')
"
# 期望：occupancy rate 在 3%-15% 之间（大部分是空的，少量被占据）
```

### 1.3 首次完整训练

```bash
# OccWorld 100 epoch，记录基线指标
python scripts/train.py --config configs/occworld.yaml
```

**预期结果**：
- mIoU: 0.35 - 0.45
- IoU@t0: ~0.50, IoU@t_final: ~0.30（随时间衰减）
- 远未来帧质量明显下降（这是正常现象，也是后续优化的方向）

### 1.4 训练第二个模型：DriveDiffuser

```bash
python scripts/train.py --config configs/diffusion.yaml
```

**对比两个模型**：生成一张对比表格，放到 README 和简历里。

---

## 阶段二：模型增强（第 2-4 周）

> 目标：在基线基础上做可论证的改进，每个改进都有消融实验支撑。

### 2.1 多相机融合（最重要）

**现状**：只用前视相机 CAM_FRONT。nuScenes 有 6 个相机。

**改动点**：

1. **数据层**：修改 `preprocess_nuscenes.py`，同时读取 6 个相机
   ```python
   # 从只读 CAM_FRONT 改为：
   cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
              'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
   for cam in cameras:
       img = load_image(nusc, sample['data'][cam])
   # images shape: (N, 6, 3, 224, 480)
   ```

2. **模型层**：`BEVEncoder` 已经有 `num_cameras` 参数
   ```python
   encoder = BEVEncoder(num_cameras=6, ...)
   # 6 个相机的特征会在 splat 阶段融合到同一个 BEV
   ```

3. **配置文件**：新增 `configs/multicam_occworld.yaml`

**预期提升**：mIoU +3~5 个百分点（360 度视野比单前视信息量大得多）

### 2.2 时间建模增强

**现状**：OccWorld 把过去 3 帧的 BEV 特征池化为一个向量，丢失了时序信息。

**改进方案**：

1. **时序 Transformer**：3 帧独立的 BEV 特征，用 temporal self-attention 融合
   ```python
   # 在 CausalTransformerBlock 之前加一层 temporal attention
   # 输入: (B, T_past, bev_dim, H, W) -> (B, H*W, T_past, bev_dim)
   # 每帧的 BEV tokens 和相邻帧做 cross-attention
   ```

2. **ConvLSTM / ConvGRU**：用循环网络建模时序演化
   ```python
   # 替代当前简单的均值池化
   class TemporalFusion(nn.Module):
       def __init__(self):
           self.conv_gru = ConvGRU(input_dim=256, hidden_dim=256)
       def forward(self, bev_seq):  # (B, T, C, H, W)
           return self.conv_gru(bev_seq)  # (B, C, H, W) with temporal context
   ```

### 2.3 损失函数优化

**现状**：CE + Dice，权重固定。

**改进方案**：

1. **Focal Loss**（代替 CE）：解决类别极度不平衡（空白占 90%+）
   ```python
   # Focal Loss 自动降低"容易分类"的空白区域的权重
   alpha = 0.25  # 正样本权重
   gamma = 2.0   # 难样本聚焦参数
   ```

2. **时间加权 Loss**：近未来权重大，远未来权重小
   ```python
   # t=0 权重 1.0, t=5 权重 0.5
   temporal_weights = torch.linspace(1.0, 0.5, T_future)
   loss = (temporal_weights * per_frame_loss).mean()
   ```

3. **对比学习辅助 Loss**：让预测的占据特征和真实占据特征在嵌入空间中对齐
   ```python
   # SimCLR-style: 同一帧的预测和 GT 拉近，不同帧的推开
   ```

### 2.4 扩散模型加速

**现状**：DriveDiffuser 需要 50 步 DDIM 采样。

**改进方案**：

1. **Distillation**：用训练好的扩散模型教一个单步预测器
   ```python
   # Progressive Distillation (Salimans & Ho, 2022)
   # 把 50-step teacher 蒸馏成 4-step student
   ```

2. **Consistency Models**：直接从噪声映射到数据，只需 1-2 步
   ```python
   # Consistency Training (Song et al., 2023)
   ```

**预期效果**：推理速度从 50 步降到 4 步，质量仅下降 1-2 个点。

---

## 阶段三：实验与评估（第 4-6 周）

> 目标：完成系统性的消融实验，产出可用于简历和面试的量化结果。

### 3.1 消融实验矩阵

| 实验编号 | 变量 | 配置 | 预期影响 |
|---------|------|------|---------|
| EXP-01 | 过去帧数 | T_past = 1, 3, 5 | 3 帧最佳，再多边际递减 |
| EXP-02 | 预测帧数 | T_future = 3, 6, 10 | 6 帧后 mIoU 快速下降 |
| EXP-03 | BEV 分辨率 | 100x100, 200x200, 400x400 | 200 性价比最高 |
| EXP-04 | 编码器类型 | LSS vs BEVFormer | BEVFormer 更准但更慢 |
| EXP-05 | Loss 类型 | CE only, CE+Dice, Focal+Dice | Focal 可能提升 2-3 点 |
| EXP-06 | 多相机 | 单前视 vs 6 相机 | +3~5 点 mIoU |
| EXP-07 | OccWorld vs Diffusion | 两种范式对比 | 验证论文核心假设 |

### 3.2 评估脚本增强

为每个实验自动生成报告：

```python
# scripts/generate_report.py
# 输入: 实验的 checkpoint + config
# 输出:
#   1. 指标表格 (markdown)
#   2. GT vs 预测对比图 (每个场景选 3 个时间步)
#   3. 随时间衰减的 IoU 曲线图
#   4. 失败案例分析 (选取 mIoU 最低的 5 个场景)
```

### 3.3 错误分析

不只报指标，还要分析模型在哪类场景犯错：

- 动态物体多（十字路口、拥堵）
- 远距离（>30m）
- 复杂几何（弯道、坡道）
- 夜间 / 雨天（nuScenes 有这些场景）

这会成为面试时最有深度的讨论点："我知道模型的弱点在哪"。

---

## 阶段四：闭环评估（第 6-8 周）

> 目标：在世界模型的实际应用场景中验证——能帮助规划器做更好的决策吗？

### 4.1 CARLA 集成

```python
# 思路：
# 1. 在 CARLA 里跑自动驾驶，每 0.5 秒记录一次相机图像 + LiDAR + 位姿
# 2. 每 3 帧喂给训练好的世界模型
# 3. 预测未来 6 帧的 3D 占用
# 4. 与真实发生的占用对比
# 5. 如果预测"前面会有障碍物"，而规划器没反应 → 世界模型发现了危险

# 技术要点：
# - CARLA Python API 控制车辆和传感器
# - 模型用 PyTorch 在 GPU 上做推理
# - 在线评估（不是离线），真实衡量"预测未来"的准确度
```

### 4.2 规划器对比实验

```
实验组: 规划器 A（只用当前感知）
对照组: 规划器 B（当前感知 + 世界模型预测的未来占用）
评估: 碰撞率、急刹次数、平均速度
```

这是论文级别的贡献——"验证世界模型能提升规划安全性"。

---

## 阶段五：工程完善（持续）

### 5.1 代码仓库完善

- [ ] 补充所有函数的 docstring（Google style）
- [ ] 添加 `CONTRIBUTING.md`
- [ ] 添加 GitHub Issue / PR 模板
- [ ] 生成 API 文档（Sphinx / mkdocs）
- [ ] 配置 Dependabot 自动更新依赖

### 5.2 CI/CD 完善

- [ ] GitHub Actions 加入 GPU 测试（用 self-hosted runner 或免费 GPU）
- [ ] 加入代码覆盖率 badge（Codecov）
- [ ] pre-commit 加入 mypy 类型检查

### 5.3 Demo 应用

- [ ] Gradio Web Demo（上传视频 → 实时展示预测 GIF）
- [ ] 部署到 HuggingFace Spaces（免费）
- [ ] README 里放一个「在线体验」链接

### 5.4 模型导出

- [ ] ONNX 导出：`torch.onnx.export(model, dummy_input, "model.onnx")`
- [ ] TorchScript 导出：`traced = torch.jit.trace(model, dummy_input)`
- [ ] 面试时可以聊部署和推理优化

---

## 阶段六：论文与展示（可选，进阶）

### 6.1 技术报告

把整个项目整理成一篇 4-6 页的 arXiv 风格论文：

```
1. Introduction（为什么自动驾驶需要世界模型）
2. Related Work（OccWorld, DriveDreamer, GAIA-1, UniSim...）
3. Method（OccWorld + DriveDiffuser 统一框架设计）
4. Experiments（消融实验 + 对比分析 + 错误分析）
5. Conclusion（局限性与未来工作）
```

不一定要投稿，但生成 PDF 放在 GitHub 上，简历里引用，含金量直接翻倍。

### 6.2 可视化海报

用 matplotlib/Plotly 生成一张信息图：

- 左上：模型架构示意图
- 右上：GT vs 预测对比 GIF
- 左下：IoU 随时间衰减曲线
- 右下：失败案例（模型预测错了什么）

格式化为 A0 海报，面试时打印出来。

---

## 面试准备清单

### 能回答的核心问题

1. **"为什么选世界模型这个方向？"**
   > 自动驾驶的传统 Pipeline（感知→预测→规划）是独立模块。世界模型把预测提升到 3D 场景级，能直接输出"未来路况"，对规划的价值比单纯检测 2D 框大得多。

2. **"你的项目与 OccWorld 论文有什么区别？"**
   > 论文只做了自回归方案。我的贡献是把自回归和扩散两个流派统一在一个框架里，直接对比。还加了不确定性估计、多尺度解码等实用组件。

3. **"训练中遇到的最大困难？"**
   > 3D 占用的稀疏性（90%+ 是空的）。普通 CE 会让模型"偷懒"全输出空。解决方法是 Dice Loss + 可能换 Focal Loss。扩散模型里还有 3D UNet 显存优化——channel_mult 缩减 + 条件向量注入替代 concat。

4. **"如果我让你加一个新功能，你打算怎么做？"**
   > 举一个例子，比如多相机融合：数据层加 5 个相机读取 → 编码器 `num_cameras=6` 已预留 → splat 阶段自然融合 → 写新的 YAML 配置 → 跑消融实验对比。

### 简历亮点提炼

```
DriveWorld | 自动驾驶世界模型框架 | PyTorch

- 独立实现 OccWorld (ECCV'24) 和 DDPM/DDIM 双范式世界模型，统一框架直接对比
- 用 nuScenes 真实数据训练，评估 3D 占用预测（mIoU, PSNR, 时间衰减 IoU）
- 生产级工程：混合精度训练、CI/CD、Docker、Type Hints 全覆盖
- 多相机 BEV 融合 + 时序建模增强 + Focal Loss + CARLA 闭环评估
```

---

## 时间线与里程碑

```
第 1 周  [==== 数据预处理 + 首次训练 ====]
第 2 周  [==== OccWorld 完整训练 + 基线指标 ====]
第 3 周  [==== 多相机融合 + 时序增强 ====]
第 4 周  [==== DriveDiffuser 训练 + 范式对比 ====]
第 5 周  [==== 消融实验矩阵（7 组实验）====]
第 6 周  [==== CARLA 闭环评估 ====]
第 7 周  [==== Gradio Demo + 技术报告 ====]
第 8 周  [==== 面试准备 + 简历 polish ====]
```

---

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
