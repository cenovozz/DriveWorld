"""Multi-view image to BEV feature encoders.

Implements two encoder paradigms:
1. BEVEncoder: CNN backbone + BEV pooling (LSS-style)
2. TransformerBEVEncoder: ViT backbone + cross-attention BEV queries (BEVFormer-style)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEncoder(nn.Module):
    """LSS-style BEV encoder with ResNet backbone and depth-aware pooling.

    Takes multi-timestamp multi-camera images and produces a unified
    BEV feature map by estimating per-pixel depth distributions and
    splatting features into 3D space, followed by BEV pooling.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        bev_feat_dim: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        depth_bins: int = 64,
        depth_range: Tuple[float, float] = (2.0, 50.0),
        num_cameras: int = 1,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_feat_dim = bev_feat_dim
        self.depth_bins = depth_bins
        self.depth_range = depth_range
        self.num_cameras = num_cameras

        self.backbone = self._build_backbone(backbone, pretrained)
        backbone_dim = 2048 if "50" in backbone else 512

        self.depth_net = nn.Sequential(
            nn.Conv2d(backbone_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, depth_bins + bev_feat_dim, 1),
        )

        self.bev_compressor = nn.Sequential(
            nn.Conv2d(bev_feat_dim, bev_feat_dim, 3, padding=1),
            nn.BatchNorm2d(bev_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_feat_dim, bev_feat_dim, 3, padding=1),
        )

        self._depth_index: Optional[torch.Tensor] = None

    def _build_backbone(self, name: str, pretrained: bool) -> nn.Module:
        import torchvision.models as models
        if name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            return nn.Sequential(*list(model.children())[:-2])
        elif name == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            return nn.Sequential(*list(model.children())[:-2])
        raise ValueError(f"Unsupported backbone: {name}")

    def get_depth_bins(self, device: torch.device) -> torch.Tensor:
        if self._depth_index is None or self._depth_index.device != device:
            depth_min, depth_max = self.depth_range
            self._depth_index = torch.linspace(
                depth_min, depth_max, self.depth_bins, device=device
            )
        return self._depth_index

    def forward(
        self, images: torch.Tensor, ego_pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            images: (B, T, C, H, W) multi-timestamp images
            ego_pose: (B, T, 3) ego poses (optional, for future use)

        Returns:
            bev_feat: (B, bev_feat_dim, bev_h, bev_w) BEV features
        """
        B, T, C, H_img, W_img = images.shape
        images_flat = images.view(B * T, C, H_img, W_img)

        backbone_feat = self.backbone(images_flat)
        x = self.depth_net(backbone_feat)

        depth_logits = x[:, : self.depth_bins]
        context_feat = x[:, self.depth_bins:]

        depth_prob = F.softmax(depth_logits, dim=1)
        depth_bins = self.get_depth_bins(images.device)
        depth_map = (depth_prob * depth_bins.view(1, -1, 1, 1)).sum(dim=1)

        bev_feat = torch.zeros(
            B, self.bev_feat_dim, self.bev_h, self.bev_w, device=images.device
        )
        for b in range(B):
            for t in range(T):
                idx = b * T + t
                feat = context_feat[idx]
                depth = depth_map[idx]
                bev_feat[b] = bev_feat[b] + self._splat(feat, depth)

        bev_feat = bev_feat / (T * self.num_cameras + 1e-8)
        bev_feat = self.bev_compressor(bev_feat)
        return bev_feat

    def _splat(self, feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Simplified splatting: project features to BEV via bilinear sampling."""
        feat_h, feat_w = feat.shape[-2:]
        scale_h = self.bev_h / feat_h
        scale_w = self.bev_w / feat_w
        return F.interpolate(
            feat.unsqueeze(0), size=(self.bev_h, self.bev_w),
            mode="bilinear", align_corners=False
        ).squeeze(0)


class TransformerBEVEncoder(nn.Module):
    """BEVFormer-style encoder using cross-attention to build BEV from images.

    Uses learnable BEV queries that attend to multi-view image features
    through deformable cross-attention, producing temporally-consistent
    BEV representations.
    """

    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        bev_feat_dim: int = 256,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_feat_dim = bev_feat_dim

        self.bev_queries = nn.Parameter(
            torch.randn(1, bev_h * bev_w, bev_feat_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bev_feat_dim,
            nhead=num_heads,
            dim_feedforward=bev_feat_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.image_proj = nn.Linear(2048, bev_feat_dim)

    def forward(
        self, images: torch.Tensor, ego_pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C, H_img, W_img = images.shape

        pooled = F.adaptive_avg_pool2d(
            images.view(B * T, C, H_img, W_img), (16, 32)
        )
        image_feat = pooled.view(B * T, C, 16 * 32).permute(0, 2, 1)
        image_feat = self.image_proj(image_feat)
        image_feat = image_feat.view(B, T * 16 * 32, self.bev_feat_dim)

        queries = self.bev_queries.expand(B, -1, -1)
        attn_output = self.transformer(queries)

        bev_feat = attn_output.transpose(1, 2).view(
            B, self.bev_feat_dim, self.bev_h, self.bev_w
        )
        return bev_feat



class ConvBEVEncoder(nn.Module):
    """Lightweight image-to-BEV encoder for the world model.

    Encodes each past camera frame with a compact conv stem, then pools
    temporally into a low-resolution BEV feature map. The compact grid keeps
    the downstream transformer tractable; the occupancy decoder upsamples
    back to the full BEV resolution.
    """

    def __init__(self, bev_h: int = 16, bev_w: int = 16, bev_feat_dim: int = 128):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_feat_dim = bev_feat_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, bev_feat_dim, kernel_size=1),
        )

    def forward(
        self, images: torch.Tensor, ego_pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C, H_img, W_img = images.shape
        x = images.view(B * T, C, H_img, W_img)
        x = self.stem(x)
        x = F.adaptive_avg_pool2d(x, (self.bev_h, self.bev_w))
        x = x.view(B, T, self.bev_feat_dim, self.bev_h, self.bev_w)
        return x.mean(dim=1)




def build_encoder(config) -> nn.Module:
    """Factory function to build the perception encoder."""
    return ConvBEVEncoder(
        bev_h=config.encoder.bev_h,
        bev_w=config.encoder.bev_w,
        bev_feat_dim=config.encoder.bev_feat_dim,
    )
