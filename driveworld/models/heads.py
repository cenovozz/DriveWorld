"""Prediction heads for world model outputs.

Decodes BEV features into future 3D occupancy grids at multiple
temporal horizons, with support for uncertainty estimation.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OccupancyDecoder(nn.Module):
    """Decodes BEV features into 3D occupancy grids.

    Uses a series of 3D transposed convolutions to upsample from
    compressed BEV representation to full (Z, H, W) occupancy.
    """

    def __init__(
        self,
        bev_feat_dim: int = 256,
        num_classes: int = 2,
        num_z: int = 16,
        bev_h: int = 200,
        bev_w: int = 200,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_z = num_z
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.z_expand = nn.Linear(bev_feat_dim, num_z * hidden_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            bev_feat: (B, bev_feat_dim, bev_h, bev_w) BEV features

        Returns:
            occupancy: (B, num_classes, num_z, bev_h, bev_w) logits
        """
        B, C, H, W = bev_feat.shape
        x = bev_feat.mean(dim=[-1, -2])
        x = self.z_expand(x)
        x = x.view(B, -1, self.num_z, 1, 1)
        x = x.expand(-1, -1, -1, H, W)

        occupancy = self.decoder(x)
        return occupancy


class MultiScaleOccupancyDecoder(nn.Module):
    """Multi-scale occupancy decoder with feature pyramid.

    Produces occupancy predictions at multiple resolutions and fuses them
    for improved detail preservation. Outputs both final occupancy and
    intermediate features for auxiliary losses.
    """

    def __init__(
        self,
        bev_feat_dim: int = 256,
        num_classes: int = 2,
        num_z: int = 16,
        bev_h: int = 200,
        bev_w: int = 200,
        scales: List[int] = [1, 2],
    ):
        super().__init__()
        self.num_z = num_z
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.scales = scales

        self.z_proj = nn.Sequential(
            nn.Linear(bev_feat_dim, num_z * 128),
            nn.LayerNorm(num_z * 128),
        )

        self.decoder_blocks = nn.ModuleList()
        in_ch = 128
        for scale in scales:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, in_ch * 2, 3, padding=1),
                    nn.BatchNorm3d(in_ch * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_ch * 2, in_ch, 3, padding=1),
                    nn.BatchNorm3d(in_ch),
                    nn.ReLU(inplace=True),
                )
            )

        self.final_conv = nn.Conv3d(in_ch, num_classes, 1)
        self.uncertainty_head = nn.Conv3d(in_ch, 1, 1)

    def forward(
        self, bev_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Returns:
            occupancy: (B, num_classes, num_z, H, W) final logits
            uncertainty: (B, 1, num_z, H, W) epistemic uncertainty
            intermediates: list of intermediate feature maps
        """
        B, C, H, W = bev_feat.shape
        x = bev_feat.mean(dim=[-1, -2])
        x = self.z_proj(x)
        x = x.view(B, 128, self.num_z, 1, 1)
        x = x.expand(-1, -1, -1, H, W)

        intermediates = []
        for block in self.decoder_blocks:
            x = block(x)
            intermediates.append(x.clone())

        occupancy = self.final_conv(x)
        uncertainty = self.uncertainty_head(x).sigmoid()

        return occupancy, uncertainty, intermediates


def build_decoder(config) -> nn.Module:
    """Factory function to build the appropriate decoder."""
    return MultiScaleOccupancyDecoder(
        bev_feat_dim=config.encoder.bev_feat_dim,
        num_classes=2,
        num_z=16,
        bev_h=config.data.bev_grid_size[0],
        bev_w=config.data.bev_grid_size[1],
    )
