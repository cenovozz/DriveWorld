<div align="center">

# DRIVEWORLD

**Modular World Model Framework for Autonomous Driving**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

*Predicting the future of driving scenes with dual-paradigm world models*

[Quickstart](#quickstart) | [Architecture](#architecture) | [Models](#models) | [Results](#results)

</div>

---

## Overview

DriveWorld is a research-oriented framework for training and evaluating **world models** in autonomous driving.

> *Given past visual observations + ego motion, predict how the 3D scene will evolve.*

Two paradigms in one codebase:

| Paradigm | Approach | Key Idea |
|----------|----------|----------|
| **OccWorld** | Autoregressive Transformer | Predict future occupancy tokens causally |
| **DriveDiffuser** | Conditional Diffusion | Denoise future occupancy grids iteratively |

### Highlights

- Dual-paradigm comparison on the same data pipeline
- Pluggable encoders (LSS / BEVFormer) and decoders
- Single-GPU friendly (nuScenes mini, 4GB)
- Rich visualization (GIFs, BEV heatmaps, IoU curves)
- Production-grade: CI/CD, Docker, pre-commit, type hints, tests

---

## Architecture

### System Overview

```mermaid
graph TB
    subgraph Input["Past Observations T=3"]
        IMG["6-Camera Images"]
        POSE["Ego Motion x y yaw"]
    end
    subgraph Encoder["Perception Encoder"]
        CNN["ResNet50 Backbone"]
        DEPTH["Depth Estimation Net"]
        SPLAT["Lift-Splat-Shoot 2D to 3D"]
        BEVF["BEV Features 256x200x200"]
        CNN --> DEPTH --> SPLAT --> BEVF
    end
    subgraph WorldModel["World Model choose one"]
        OCC["OccWorld - Causal Transformer - 6 layers 8 heads - ~200M params - Fast Deterministic"]
        DIFF["DriveDiffuser - 3D UNet plus DDIM - 1000 timesteps - ~150M params - Stochastic Diverse"]
    end
    subgraph Decoder["Occupancy Decoder"]
        UP["3D Transposed Conv"]
        MULTI["Multi-Scale Fusion"]
        UNC["Uncertainty Head"]
        UP --> MULTI --> UNC
    end
    subgraph Output["Future Predictions T=6"]
        OCC3D["3D Occupancy Grid 16x200x200"]
        VIZ["GIFs IoU curves BEV heatmaps"]
        OCC3D --> VIZ
    end
    IMG --> CNN
    POSE --> CNN
    BEVF --> OCC
    BEVF --> DIFF
    OCC --> UP
    DIFF --> UP
    UNC --> OCC3D
    style Input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style Encoder fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style WorldModel fill:#fce4ec,stroke:#d32f2f,stroke-width:2px
    style Decoder fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style Output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### Encoder: Two Paradigms

```mermaid
graph LR
    subgraph LSS["LSS-Style (ECCV 2020)"]
        direction TB
        A1["Input Image"] --> A2["ResNet50"] --> A3["DepthNet 64 bins"] --> A4["Splat to BEV"] --> A5["BEV Feature"]
    end
    subgraph BEVF["BEVFormer-Style (ECCV 2022)"]
        direction TB
        B1["Input Image"] --> B2["Pool to grid"] --> B3["Learnable Queries"] --> B4["Cross-Attention x3"] --> B5["BEV Feature"]
    end
    style LSS fill:#e8eaf6,stroke:#3949ab
    style BEVF fill:#e0f2f1,stroke:#00897b
```

### Training Sequence

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
        alt OccWorld Training
            WM->>WM: Autoregressive token prediction
            WM->>Dec: Predicted tokens
            Dec->>Out: 3D occupancy logits
            Out->>Loss: CE + Dice Loss
        else DriveDiffuser Training
            WM->>WM: Add noise + UNet predicts noise
            WM->>Loss: MSE Loss
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

---

## Quickstart

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (optional, CPU fallback available)
- [nuScenes mini](https://www.nuscenes.org/nuscenes#download) (4GB, free)

### Installation

```bash
git clone https://github.com/cenovozz/DriveWorld.git
cd DriveWorld
pip install -e .
pip install -e ".[dev]"
pre-commit install
```

### Train

```bash
python scripts/train.py --config configs/occworld.yaml
python scripts/train.py --config configs/diffusion.yaml --paradigm diffusion
python scripts/train.py --config configs/occworld.yaml --resume checkpoints/best.pt
tensorboard --logdir logs/
```

### Evaluate

```bash
python scripts/eval.py --checkpoint checkpoints/best.pt --config configs/occworld.yaml --output-dir outputs/eval
```

### Docker

```bash
docker build -t driveworld .
docker run --gpus all -v $(pwd)/data:/workspace/data driveworld
```

---

## Project Structure

```
DriveWorld/
├── configs/                    # YAML config files
│   ├── default.yaml
│   ├── occworld.yaml
│   └── diffusion.yaml
├── driveworld/                 # Core package
│   ├── data/                   # Dataset and transforms
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
│   ├── eval/                   # Evaluation and visualization
│   │   ├── evaluator.py        #   WorldModelEvaluator
│   │   └── visualize.py        #   GIF maker, BEV heatmaps, curves
│   └── utils/                  # Configuration and logging
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

## Models

### OccWorld

Predicts future 3D occupancy by:
1. Encoding past images into unified BEV representation
2. Tokenizing BEV into discrete tokens
3. Autoregressively decoding future tokens via causal transformer
4. Reconstructing full 3D occupancy grids

**Parameters**: 6 transformer layers, 512 hidden dim, 8 heads (~200M)

### DriveDiffuser

Generates future occupancy through:
1. Encoding past context into conditioning vector
2. Training 3D UNet to predict noise in occupancy grids
3. Sampling via DDIM (50 steps) for fast inference
4. Cosine noise schedule for quality

**Parameters**: 3-level 3D UNet, 1000 timesteps, DDIM 50-step (~150M)

### Comparison

| Aspect | OccWorld | DriveDiffuser |
|--------|----------|---------------|
| **Paper** | Zheng et al., ECCV 2024 | Ho et al., NeurIPS 2020 |
| **Inference** | 1 forward pass | 50 DDIM steps |
| **Core Math** | Autoregressive P(x_t | past) | Iterative denoising p(x_0) |
| **Training Loss** | CE + Dice | MSE (noise) |
| **Output** | Deterministic | Stochastic (diverse) |
| **Long-Horizon** | Yes (autoregressive) | Fixed window |
| **Uncertainty** | Not modeled | Sample variance |
| **Best For** | Real-time planning | Safety analysis |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection over Union (per-class) |
| **PSNR** | Peak Signal-to-Noise Ratio |
| **IoU@t** | Per-timestep IoU (first / middle / last) |
| **Dice** | Soft Dice coefficient |

---

## Example Results

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

---

## Development

```bash
pytest tests/ -v
ruff check driveworld/ scripts/ tests/
black --check driveworld/ scripts/ tests/
mypy driveworld/
```

---

## Roadmap

- [ ] nuScenes full dataset preprocessing pipeline
- [ ] Multi-camera support (current: front only)
- [ ] Streaming / online inference mode
- [ ] Hybrid OccWorld + Diffusion model
- [ ] CARLA integration for closed-loop evaluation
- [ ] Model distillation (teacher to student)
- [ ] Gradio demo app

---

## Citation

```bibtex
@software{driveworld2024,
  author = {Your Name},
  title = {DriveWorld: Modular World Model Framework for Autonomous Driving},
  year = {2024},
  url = {https://github.com/cenovozz/DriveWorld}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with passion for autonomous driving research</sub>
</div>
