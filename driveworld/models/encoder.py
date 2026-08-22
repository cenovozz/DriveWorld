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
        bev_range: float = 50.0,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_feat_dim = bev_feat_dim
        self.depth_bins = depth_bins
        self.depth_range = depth_range
        self.num_cameras = num_cameras
        self.bev_range = bev_range

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

        builders = {
            "resnet50": models.resnet50,
            "resnet18": models.resnet18,
        }
        if name not in builders:
            raise ValueError(f"Unsupported backbone: {name}")

        if pretrained:
            try:
                model = builders[name](weights="IMAGENET1K_V1")
            except Exception as exc:  # noqa: BLE001 - network/cache failures
                import warnings

                warnings.warn(
                    f"Failed to load pretrained {name} weights ({exc}); "
                    "falling back to random initialization."
                )
                model = builders[name](weights=None)
        else:
            model = builders[name](weights=None)

        return nn.Sequential(*list(model.children())[:-2])

    def get_depth_bins(self, device: torch.device) -> torch.Tensor:
        if self._depth_index is None or self._depth_index.device != device:
            depth_min, depth_max = self.depth_range
            self._depth_index = torch.linspace(
                depth_min, depth_max, self.depth_bins, device=device
            )
        return self._depth_index

    def forward(
        self,
        images: torch.Tensor,
        ego_pose: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            images: (B, T, C, H, W) or (B, T, N, C, H, W)
            ego_pose: (B, T, 3) ego poses (optional, for future use)
            intrinsics: optional (B, T, N, 3, 3) camera intrinsics
            extrinsics: optional (B, T, N, 4, 4) camera-to-ego transforms

        Returns:
            bev_feat: (B, bev_feat_dim, bev_h, bev_w) BEV features
        """
        if images.dim() == 4:
            images = images.unsqueeze(2)

        if images.dim() == 6:
            return self._forward_multicam_lss(images, intrinsics, extrinsics)

        B, T, C, H_img, W_img = images.shape
        images_flat = images.view(B * T, C, H_img, W_img)

        backbone_feat = self.backbone(images_flat)
        x = self.depth_net(backbone_feat)

        depth_logits = x[:, : self.depth_bins]
        context_feat = x[:, self.depth_bins :]

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

    def _forward_multicam_lss(
        self,
        images: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """LSS-style multi-camera BEV construction.

        Projects each depth candidate from camera coordinates to ego
        coordinates, then splats context features into a shared BEV grid.
        """
        B, T, N, C, H_img, W_img = images.shape
        if intrinsics is None or extrinsics is None:
            raise ValueError(
                "Multi-camera LSS requires camera intrinsics and extrinsics. "
                "Re-run scripts/preprocess_nuscenes.py to generate them."
            )

        images_flat = images.view(B * T * N, C, H_img, W_img)
        backbone_feat = self.backbone(images_flat)
        x = self.depth_net(backbone_feat)

        depth_logits = x[:, : self.depth_bins]
        context_feat = x[:, self.depth_bins :]

        depth_prob = F.softmax(depth_logits, dim=1)
        feat_h, feat_w = depth_prob.shape[-2:]
        depth_prob = depth_prob.view(B, T, N, self.depth_bins, feat_h, feat_w)
        context_feat = context_feat.view(
            B, T, N, self.bev_feat_dim, feat_h, feat_w
        )

        bev_feat = torch.zeros(
            B, self.bev_feat_dim, self.bev_h, self.bev_w,
            device=images.device, dtype=context_feat.dtype,
        )
        bev_weight = torch.zeros(
            B, 1, self.bev_h, self.bev_w,
            device=images.device, dtype=context_feat.dtype,
        )

        for b in range(B):
            for t in range(T):
                for n in range(N):
                    self._splat_camera(
                        bev_feat[b],
                        bev_weight[b, 0],
                        context_feat[b, t, n],
                        depth_prob[b, t, n],
                        intrinsics[b, t, n],
                        extrinsics[b, t, n],
                        H_img,
                        W_img,
                    )

        # Normalize in fp32: the 1e-8 epsilon underflows to zero in fp16,
        # which turns empty BEV cells into 0/0 -> NaN. Casting the buffers up
        # keeps the normalization stable and leaves bev_compressor under AMP
        # control for the following conv/BN.
        bev_feat = bev_feat.float()
        bev_weight = bev_weight.float()
        bev_feat = bev_feat / (bev_weight + 1e-8)
        return self.bev_compressor(bev_feat)

    def _splat_camera(
        self,
        bev_feat: torch.Tensor,
        bev_weight: torch.Tensor,
        context_feat: torch.Tensor,
        depth_prob: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        image_h: int,
        image_w: int,
    ) -> None:
        """Accumulate one camera's LSS splat into BEV buffers in-place."""
        device = context_feat.device
        feat_h, feat_w = context_feat.shape[-2:]
        depth_bins = self.get_depth_bins(device)

        u = (torch.arange(feat_w, device=device, dtype=torch.float32) + 0.5) * (
            image_w / feat_w
        )
        v = (torch.arange(feat_h, device=device, dtype=torch.float32) + 0.5) * (
            image_h / feat_h
        )
        grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        x_norm = (grid_u - cx) / fx
        y_norm = (grid_v - cy) / fy
        ones = torch.ones_like(x_norm)

        cam_pts = torch.stack([x_norm, y_norm, ones], dim=-1).unsqueeze(2)
        cam_pts = cam_pts * depth_bins.view(1, 1, -1, 1)
        cam_pts = torch.cat(
            [cam_pts, torch.ones_like(cam_pts[..., :1])], dim=-1
        )
        cam_pts = cam_pts.view(-1, 4)

        ego_pts = torch.einsum("ij,pj->pi", extrinsics, cam_pts)
        ego_x = ego_pts[:, 0]
        ego_y = ego_pts[:, 1]

        half_range = self.bev_range
        cell_x = (2.0 * half_range) / self.bev_w
        cell_y = (2.0 * half_range) / self.bev_h
        gx = ((ego_x + half_range) / cell_x).long()
        gy = ((ego_y + half_range) / cell_y).long()

        valid = (
            (gx >= 0) & (gx < self.bev_w) & (gy >= 0) & (gy < self.bev_h)
        )
        if not valid.any():
            return

        gx = gx[valid]
        gy = gy[valid]
        idx = gy * self.bev_w + gx

        weights = depth_prob.permute(1, 2, 0).reshape(-1)[valid]
        context_flat = context_feat.reshape(self.bev_feat_dim, -1)
        context_flat = (
            context_flat.unsqueeze(-1)
            .expand(-1, feat_h * feat_w, self.depth_bins)
            .reshape(self.bev_feat_dim, -1)[:, valid]
        )

        weighted_feats = context_flat * weights.unsqueeze(0)
        # Under AMP, conv outputs are fp16 while softmax(depth logits) is
        # computed in fp32. Cast both sides to the BEV accumulator dtype so
        # index_add_ never mixes Half and Float tensors.
        weighted_feats = weighted_feats.to(bev_feat.dtype)
        weights = weights.to(bev_weight.dtype)

        bev_flat = bev_feat.view(self.bev_feat_dim, -1)
        bev_flat.index_add_(1, idx, weighted_feats)
        bev_weight.view(-1).index_add_(0, idx, weights)

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

    def __init__(
        self,
        bev_h: int = 16,
        bev_w: int = 16,
        bev_feat_dim: int = 128,
        num_cameras: int = 1,
        fusion_method: str = "mean",
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_feat_dim = bev_feat_dim
        self.num_cameras = num_cameras
        self.fusion_method = fusion_method

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
        self,
        images: torch.Tensor,
        ego_pose: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if images.dim() == 4:
            images = images.unsqueeze(2)

        if images.dim() == 6:
            return self._forward_multicam(images)

        if images.dim() != 5:
            raise ValueError(
                f"Expected images with shape (B, T, 3, H, W) or "
                f"(B, T, C, 3, H, W), got {tuple(images.shape)}"
            )

        B, T, C, H_img, W_img = images.shape
        x = images.view(B * T, C, H_img, W_img)
        x = self.stem(x)
        x = F.adaptive_avg_pool2d(x, (self.bev_h, self.bev_w))
        x = x.view(B, T, self.bev_feat_dim, self.bev_h, self.bev_w)
        return x.mean(dim=1)

    def _forward_multicam(self, images: torch.Tensor) -> torch.Tensor:
        """Fuse a ``(B, T, N, 3, H, W)`` multi-camera tensor into one BEV."""
        B, T, N, C, H_img, W_img = images.shape
        if C != 3:
            raise ValueError(
                f"Expected 3 image channels for multi-camera input, got {C}"
            )

        x = images.view(B * T * N, C, H_img, W_img)
        x = self.stem(x)
        x = F.adaptive_avg_pool2d(x, (self.bev_h, self.bev_w))
        x = x.view(B, T, N, self.bev_feat_dim, self.bev_h, self.bev_w)

        if self.fusion_method == "mean":
            x = x.mean(dim=2)
        elif self.fusion_method == "sum":
            x = x.sum(dim=2)
        else:
            raise ValueError(
                f"Unsupported camera fusion method: {self.fusion_method}"
            )

        return x.mean(dim=1)




def build_encoder(config) -> nn.Module:
    """Factory function to build the perception encoder.

    The ``fusion_method`` from ``config.encoder`` selects the encoder. The
    default ``cnn``/``mean`` paths preserve the original single-camera
    ConvBEVEncoder behavior. For multi-camera inputs, set
    ``data.num_cameras`` and use ``fusion_method: mean`` (the LSS path is
    kept as a research scaffold until camera extrinsics are wired in).
    """
    num_cameras = getattr(config.data, "num_cameras", 1)
    fusion_method = config.encoder.fusion_method

    if fusion_method == "lss":
        return BEVEncoder(
            backbone=config.encoder.backbone,
            pretrained=config.encoder.pretrained,
            bev_feat_dim=config.encoder.bev_feat_dim,
            bev_h=config.encoder.bev_h,
            bev_w=config.encoder.bev_w,
            num_cameras=num_cameras,
        )

    if fusion_method == "transformer":
        return TransformerBEVEncoder(
            bev_h=config.encoder.bev_h,
            bev_w=config.encoder.bev_w,
            bev_feat_dim=config.encoder.bev_feat_dim,
        )

    return ConvBEVEncoder(
        bev_h=config.encoder.bev_h,
        bev_w=config.encoder.bev_w,
        bev_feat_dim=config.encoder.bev_feat_dim,
        num_cameras=num_cameras,
        fusion_method=fusion_method,
    )
