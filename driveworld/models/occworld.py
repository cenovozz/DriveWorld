"""OccWorld: Autoregressive 3D Occupancy World Model.

Core idea (inspired by OccWorld, ECCV 2024):
- Encode past observations (images + poses) into a compact BEV token sequence
- Autoregressively predict future 3D occupancy tokens with a transformer
- Decode predicted tokens back to 3D occupancy grids

This implementation is lightweight and modular, designed for single-GPU
training on nuScenes mini.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class CausalTransformerBlock(nn.Module):
    """Single transformer block with causal self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class OccWorld(nn.Module):
    """Autoregressive 3D Occupancy World Model.

    Architecture:
        1. Encoder: compact image-to-BEV encoder
        2. World model: transformer over BEV tokens + ego motion tokens
        3. Decoder: 2D upsampling head producing 3D occupancy logits

    Args:
        encoder: BEV feature encoder module
        decoder: occupancy reconstruction decoder
        hidden_dim: transformer hidden dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        vocab_size: reserved for VQ tokenization (kept for API compatibility)
        dropout: dropout rate
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        vocab_size: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = hidden_dim

        self.bev_h = getattr(encoder, "bev_h", 16)
        self.bev_w = getattr(encoder, "bev_w", 16)
        self.bev_feat_dim = getattr(encoder, "bev_feat_dim", 128)
        num_tokens = self.bev_h * self.bev_w

        self.bev_proj = nn.Linear(self.bev_feat_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.bev_feat_dim * num_tokens)

        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            CausalTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ego_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.bev_proj, self.output_proj, *self.ego_encoder]:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        past_images: torch.Tensor,
        past_ego_pose: torch.Tensor,
        future_ego_pose: torch.Tensor,
        future_occupancy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            past_images: (B, T_past, 3, H, W)
            past_ego_pose: (B, T_past, 3)
            future_ego_pose: (B, T_future, 3)
            future_occupancy: optional, unused in this forward

        Returns:
            Dict with occupancy_pred: (B, T_future, 2, Z, H, W)
        """
        B, T_past = past_images.shape[:2]
        T_future = future_ego_pose.shape[1]

        bev_feat = self.encoder(past_images, past_ego_pose)
        B, C_bev, H_bev, W_bev = bev_feat.shape
        num_tokens = H_bev * W_bev

        bev_tokens = bev_feat.view(B, C_bev, -1).permute(0, 2, 1)
        bev_tokens = self.bev_proj(bev_tokens)

        ego_past = self.ego_encoder(past_ego_pose)
        ego_future = self.ego_encoder(future_ego_pose)
        ego_tokens = torch.cat([ego_past, ego_future], dim=1)

        input_tokens = torch.cat([bev_tokens, ego_tokens], dim=1)
        x = self.pos_encoding(input_tokens)
        hidden_states = []
        for block in self.transformer_blocks:
            x = block(x)
            hidden_states.append(x.clone())

        future_tokens = x[:, num_tokens + T_past : num_tokens + T_past + T_future]

        occupancy_preds = []
        for t in range(T_future):
            token_t = future_tokens[:, t]
            feat_t = self.output_proj(token_t)
            feat_t = feat_t.view(B, C_bev, H_bev, W_bev)
            occ_t = self.decoder(feat_t)
            if isinstance(occ_t, tuple):
                occ_t = occ_t[0]
            occupancy_preds.append(occ_t)

        occupancy_pred = torch.stack(occupancy_preds, dim=1)

        return {
            "occupancy_pred": occupancy_pred,
            "bev_features": bev_feat,
            "hidden_states": hidden_states,
        }

    def generate(
        self,
        past_images: torch.Tensor,
        past_ego_pose: torch.Tensor,
        future_ego_pose: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generation mode for inference."""
        self.eval()
        with torch.no_grad():
            output = self.forward(past_images, past_ego_pose, future_ego_pose)
        return output["occupancy_pred"]
