FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="your@email.com"
LABEL description="DriveWorld: Modular World Model Framework for Autonomous Driving"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml .
COPY driveworld/ driveworld/
COPY scripts/ scripts/
COPY configs/ configs/

RUN pip install --no-cache-dir -e .

RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs /workspace/outputs

VOLUME ["/workspace/data", "/workspace/checkpoints", "/workspace/logs"]

ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/occworld.yaml"]
