"""Prediction heads for world model outputs.

Decodes compact BEV features into full-resolution 3D occupancy grids.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupancyDecoder(nn.Module):
    """Decodes a compact BEV feature map into 3D occupancy logits.

    Uses 2D transposed convolutions to upsample the BEV grid, then produces
    num_classes * num_z output channels and reshapes to
    (B, num_classes, num_z, H, W). Keeping the upsampling in 2D avoids the
    memory blow-up of full-resolution 3D convolutions.
    """

    def __init__(
        self,
        bev_feat_dim: int = 128,
        num_classes: int = 2,
        num_z: int = 16,
        bev_h: int = 200,
        bev_w: int = 200,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_z = num_z
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                bev_feat_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes * num_z, 1),
        )

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """Args: bev_feat (B, bev_feat_dim, H, W). Returns (B, num_classes, num_z, bev_h, bev_w)."""
        x = self.up(bev_feat)
        x = F.interpolate(
            x, size=(self.bev_h, self.bev_w), mode="bilinear", align_corners=False
        )
        B, _, H, W = x.shape
        return x.view(B, self.num_classes, self.num_z, H, W)


class MultiScaleOccupancyDecoder(OccupancyDecoder):
    """Alias kept for API compatibility."""


def build_decoder(config) -> nn.Module:
    """Factory function to build the occupancy decoder."""
    return OccupancyDecoder(
        bev_feat_dim=config.encoder.bev_feat_dim,
        num_classes=2,
        num_z=16,
        bev_h=config.data.bev_grid_size[0],
        bev_w=config.data.bev_grid_size[1],
    )