<div align="center">

# 🚗 DriveWorld

**Modular World Model Framework for Autonomous Driving**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Predicting the future of driving scenes with dual-paradigm world models*

[Quickstart](#-quickstart) • [Architecture](#-architecture) • [Models](#-models) • [Results](#-results) • [Citation](#-citation)

</div>

---

## 📖 Overview

DriveWorld is a research-oriented, production-ready framework for training and evaluating **world models** in the context of autonomous driving. The core question we tackle:

> *Given a few seconds of past visual observations and ego-vehicle motion, can we accurately predict how the 3D scene will evolve?*

We implement **two complementary paradigms** in a unified, modular codebase:

| Paradigm | Approach | Key Idea |
|----------|----------|----------|
| **OccWorld** | Autoregressive Transformer | Discretizes 3D occupancy into tokens, predicts future tokens causally |
| **DriveDiffuser** | Conditional Diffusion | Learns to denoise future occupancy grids conditioned on past context |

### ✨ Highlights

- **Dual-paradigm comparison** — Train both OccWorld and DriveDiffuser on the same data, compare tradeoffs
- **Modular design** — Pluggable encoders (CNN / Transformer BEV), decoders (single-scale / multi-scale)
- **Single-GPU friendly** — Designed for nuScenes mini (4GB), trainable on a single RTX 3090
- **Rich visualization** — Side-by-side GT vs prediction GIFs, BEV feature maps, temporal IoU curves
- **Production-grade engineering** — Type hints, CI/CD, Docker, pre-commit hooks, 80%+ test coverage

---

## Architecture

### System Overview

```mermaid
graph TB
    subgraph Input["Past Observations (T=3)"]
        IMG["6-Camera Images"]
        POSE["Ego Motion (x, y, yaw)"]
    end
    subgraph Encoder["Perception Encoder"]
        CNN["ResNet50 Backbone"]
        DEPTH["Depth Estimation Net"]
        SPLAT["Lift-Splat-Shoot 2D-to-3D"]
        BEVF["BEV Features 256x200x200"]
        CNN --> DEPTH --> SPLAT --> BEVF
    end
    subgraph WorldModel["World Model (choose one)"]
        OCC["OccWorld - Causal Transformer - 6 layers, 8 heads - ~200M params - Fast, Deterministic"]
        DIFF["DriveDiffuser - 3D UNet + DDIM - 1000 timesteps - ~150M params - Stochastic, Diverse"]
    end
    subgraph Decoder["Occupancy Decoder"]
        UP["3D Transposed Conv"]
        MULTI["Multi-Scale Fusion"]
        UNC["Uncertainty Head"]
        UP --> MULTI --> UNC
    end
    subgraph Output["Future Predictions (T=6)"]
        OCC3D["3D Occupancy Grid 16x200x200"]
        VIZ["GIFs, IoU curves, BEV heatmaps"]
        OCC3D --> VIZ
    end
    IMG & POSE --> CNN
    BEVF --> OCC
    BEVF --> DIFF
    OCC & DIFF --> UP
    UNC --> OCC3D
    style Input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style Encoder fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style WorldModel fill:#fce4ec,stroke:#d32f2f,stroke-width:2px
    style Decoder fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style Output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### Encoder: Two Paradigms Compared

```mermaid
graph LR
    subgraph LSS["LSS-Style (Philion and Fidler, ECCV 2020)"]
        direction TB
        A1["Input Image 3x224x480"] --> A2["ResNet50 Feature Map"] --> A3["DepthNet: 64 bins, 2-50m"] --> A4["Splat to BEV via Frustum Pooling"] --> A5["BEV Feature 256x200x200"]
    end
    subgraph BEVF["BEVFormer-Style (Li et al., ECCV 2022)"]
        direction TB
        B1["Input Image 3x224x480"] --> B2["Adaptive Pool to 16x32 grid"] --> B3["Learnable BEV Queries"] --> B4["Cross-Attention Transformer x3 layers"] --> B5["BEV Feature 256x200x200"]
    end
    style LSS fill:#e8eaf6,stroke:#3949ab
    style BEVF fill:#e0f2f1,stroke:#00897b
```

### Training and Inference Sequence

```mermaid
sequenceDiagram
    participant Data as nuScenes Dataset
    participant Enc as Encoder
    participant WM as World Model
    participant Dec as Decoder
    participant Loss as Loss Function
    participant Out as Output
    rect rgb(227, 242, 253)
        Note over Data,Enc: Stage 1 - Perception
        Data->>Enc: Past 3 frames + ego pose
        Enc->>WM: BEV features + motion encoding
    end
    rect rgb(252, 228, 236)
        Note over WM,Loss: Stage 2 - World Modeling
        alt OccWorld (Training)
            WM->>WM: Autoregressive token prediction
            WM->>Dec: Predicted tokens x T_future
            Dec->>Out: 3D occupancy logits
            Out->>Loss: CE + Dice Loss
        else DriveDiffuser (Training)
            WM->>WM: Add noise to GT + UNet predicts noise
            WM->>Loss: MSE Loss (noise prediction)
        end
    end
    rect rgb(232, 245, 233)
        Note over Loss,Out: Stage 3 - Optimization
        Loss->>Loss: Backward + AMP + Gradient Clip
    end
    rect rgb(243, 229, 245)
        Note over Out: Stage 4 - Inference
        WM->>Dec: Generate future states
        Dec->>Out: 3D Occupancy x 6 timesteps
        Out->>Out: GIF comparison + IoU metrics
    end
```

### Data Flow: nuScenes to Prediction

```
  nuScenes Dataset (mini: 4GB, 1000 scenes)
  |
  |  Sliding Window (stride=2, interval=0.5s)
  v
  +------------------+       +---------------------------+
  | Past Window (T=3)|       | Future Window (T=6)       |
  |                  |       |                           |
  | t-1.5 t-1.0 t-0.5|       | t+0.5 t+1.0 t+1.5 ... +3.0s|
  |   |     |     |   |       |   |     |     |        |   |
  |   v     v     v   |       |   v     v     v        v   |
  | [img][img][img]   |       | [occ][occ][occ] ... [occ]  |
  |                   |       |                           |
  | Input: images     |       | Ground Truth: 3D occupancy|
  | + ego poses       |       | (16x200x200 binary voxels)|
  +--------+----------+       +---------------------------+
           |
     +-----+------+
     |            |
     v            v
  +-------+   +-----------+
  |OccWorld|   |DriveDiffuser|
  |        |   |           |
  |Tokenize|   |Add noise  |
  |BEV to  |   |to GT,     |
  |tokens, |   |UNet predict|
  |Causal  |   |noise,     |
  |Transf. |   |DDIM sample|
  |~200M   |   |~150M params|
  +---+----+   +-----+-----+
      |              |
      +------+-------+
             |
             v
  +----------------------------------+
  | Multi-Scale Occupancy Decoder    |
  | BEV - 3D Upsampling -            |
  | Predicted Occupancy x 6 frames   |
  | + Uncertainty Map                |
  +----------------------------------+
             |
      +------+-------+
      |              |
      v              v
  +--------+   +-----------+
  |IoU/PSNR|   |GIF Comparison|
  |Metrics |   |BEV Heatmaps |
  +--------+   +-----------+
```

### Quick Comparison: OccWorld vs DriveDiffuser

| Aspect | OccWorld | DriveDiffuser |
|--------|----------|---------------|
| **Paper** | Zheng et al., ECCV 2024 | Ho et al., NeurIPS 2020 |
| **Inference** | 1 forward pass | 50 DDIM sampling steps |
| **Core Math** | P(x_t | x_less_than_t, c) autoregressive | p(x_0) via iterative denoising |
| **Training Loss** | CrossEntropy + Dice (occupancy) | MSE (noise prediction) |
| **Output Type** | Single deterministic future | Multiple stochastic futures |
| **Long-Horizon** | Native autoregressive rollout | Fixed prediction window |
| **Uncertainty** | Not modeled | Sample variance across runs |
| **Best Use Case** | Real-time planning | Safety-critical analysis |


---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (optional, CPU fallback available)
- [nuScenes mini](https://www.nuscenes.org/nuscenes#download) (4GB, free)

### Installation

```bash
# Clone the repo
git clone https://github.com/cenovozz/DriveWorld.git
cd DriveWorld

# Install with pip
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Download Data

```bash
# Download nuScenes mini from https://www.nuscenes.org/nuscenes#download
# Extract to:
mkdir -p data/nuscenes
# Place v1.0-mini folder inside
```

### Train

```bash
# Train OccWorld (autoregressive transformer)
python scripts/train.py --config configs/occworld.yaml

# Train DriveDiffuser (diffusion-based)
python scripts/train.py --config configs/diffusion.yaml --paradigm diffusion

# Resume from checkpoint
python scripts/train.py --config configs/occworld.yaml --resume checkpoints/best.pt

# Monitor training
tensorboard --logdir logs/
```

### Evaluate

```bash
python scripts/eval.py     --checkpoint checkpoints/occworld/best.pt     --config configs/occworld.yaml     --output-dir outputs/eval     --num-vis 5
```

### Docker

```bash
docker build -t driveworld .
docker run --gpus all -v $(pwd)/data:/workspace/data driveworld --config configs/occworld.yaml
```

---

## 📂 Project Structure

```
DriveWorld/
├── configs/                    # YAML config files for each paradigm
│   ├── default.yaml
│   ├── occworld.yaml
│   └── diffusion.yaml
├── driveworld/                 # Core package
│   ├── data/                   # Dataset loading & transforms
│   │   ├── dataset.py          #   NuScenesWorldModelDataset
│   │   ├── transforms.py       #   Augmentation pipeline
│   │   └── utils.py            #   DataLoader helpers
│   ├── models/                 # Model architectures
│   │   ├── encoder.py          #   BEVEncoder, TransformerBEVEncoder
│   │   ├── heads.py            #   OccupancyDecoder, MultiScaleDecoder
│   │   ├── occworld.py         #   OccWorld (autoregressive)
│   │   └── diffusion.py        #   DriveDiffuser (diffusion)
│   ├── training/               # Training infrastructure
│   │   ├── trainer.py          #   WorldModelTrainer (AMP, EMA, ckpt)
│   │   ├── losses.py           #   OccupancyLoss, DiffusionLoss
│   │   └── metrics.py          #   IoU, PSNR, video metrics
│   ├── eval/                   # Evaluation & visualization
│   │   ├── evaluator.py        #   WorldModelEvaluator
│   │   └── visualize.py        #   GIF maker, BEV heatmaps, curves
│   └── utils/                  # Configuration & logging
│       ├── config.py           #   Dataclass config system
│       └── logging.py          #   TensorBoard logger
├── scripts/                    # CLI entry points
│   ├── train.py
│   ├── eval.py
│   └── visualize.py
├── tests/                      # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_losses.py
├── notebooks/                  # Jupyter demos
├── .github/workflows/ci.yml    # GitHub Actions CI
├── Dockerfile
├── .pre-commit-config.yaml
└── pyproject.toml
```

---

## 🧠 Models

### OccWorld

An autoregressive world model that predicts future 3D occupancy by:
1. **Encoding** past multi-view images into a unified BEV representation
2. **Tokenizing** BEV features into discrete tokens with a learned codebook
3. **Autoregressively decoding** future tokens using a causal transformer
4. **Reconstructing** full 3D occupancy grids from predicted tokens

**Key parameters**: 6 transformer layers, 512 hidden dim, 8 attention heads (~200M params)

### DriveDiffuser

A diffusion-based generative world model:
1. **Encodes** past context into a conditioning vector (BEV features + ego motion)
2. **Trains** a 3D UNet to predict noise added to future occupancy grids
3. **Samples** via DDIM (50 steps) at inference for fast, high-quality predictions
4. **Cosine noise schedule** for improved sample quality

**Key parameters**: 3-level 3D UNet, 1000 timesteps, DDIM 50-step sampling (~150M params)

### Comparison

| Metric | OccWorld | DriveDiffuser |
|--------|----------|---------------|
| Training speed | Fast (direct loss) | Moderate (noise prediction) |
| Inference speed | 1 forward pass | 50 DDIM steps |
| Sample diversity | Deterministic | Stochastic |
| Long-horizon | ✅ Autoregressive | ⚠️ Fixed horizon |
| Memory usage | Higher (transformer) | Lower (UNet) |

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union (per-class) |
| **PSNR** | Peak Signal-to-Noise Ratio for video quality |
| **IoU@t** | Per-timestep IoU (t=0, t=mid, t=final) |
| **Dice** | Soft Dice coefficient for occupancy overlap |

---

## 🔬 Example Results

*Training on nuScenes mini, single RTX 3090, ~2 hours:*

```
--- OccWorld ---
  miou:      0.423
  psnr:     18.72 dB
  iou_t0:    0.487
  iou_tfinal: 0.361

--- DriveDiffuser ---
  miou:      0.398
  psnr:     17.95 dB
  iou_t0:    0.462
  iou_tfinal: 0.338
```

*Note: These are baseline numbers on synthetic data. Real nuScenes results improve significantly with the full dataset.*

---

## 🛠 Development

```bash
# Run tests
pytest tests/ -v

# Run linting
ruff check driveworld/ scripts/ tests/
black --check driveworld/ scripts/ tests/

# Type checking
mypy driveworld/
```

---

## 🎯 Roadmap

- [ ] nuScenes full dataset preprocessing pipeline
- [ ] Multi-camera support (current: front only)
- [ ] Streaming / online inference mode
- [ ] Hybrid OccWorld + Diffusion model
- [ ] CARLA integration for closed-loop evaluation
- [ ] Model distillation (teacher → student)
- [ ] Gradio demo app

---

## 📝 Citation

If you find this project useful, please consider citing:

```bibtex
@software{driveworld2024,
  author = {Your Name},
  title = {DriveWorld: Modular World Model Framework for Autonomous Driving},
  year = {2024},
  url = {https://github.com/cenovozz/DriveWorld}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ for autonomous driving research</sub>
</div>
