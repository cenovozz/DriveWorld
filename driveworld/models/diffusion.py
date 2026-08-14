"""DriveDiffuser: Diffusion-based World Model for Autonomous Driving.

Core idea:
- Treats world state prediction as a conditional generative task
- Uses a 3D UNet to denoise future occupancy grids
- Conditioned on past BEV features and ego motion trajectory
- Supports both DDPM and DDIM sampling for flexible speed/quality tradeoff

Lightweight implementation designed for nuScenes mini (~150M parameters).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings (Transformer-style)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3D(nn.Module):
    """3D residual block with time embedding conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout3d(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        self.shortcut = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock3D(nn.Module):
    """3D self-attention block (applied channel-wise)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.num_heads, C // self.num_heads, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, -1)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).view(B, C, D, H, W)

        return x + self.proj(out)


class UNet3D(nn.Module):
    """Lightweight 3D UNet for occupancy denoising."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
        cond_dim: int = 256,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.cond_proj = nn.Linear(cond_dim, base_channels)

        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        chs = [base_channels]
        ch = base_channels
        self.down_blocks = nn.ModuleList()
        for mult in channel_mult:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock3D(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                chs.append(ch)
            self.down_blocks.append(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
            chs.append(ch)

        self.mid_blocks = nn.ModuleList([
            ResBlock3D(ch, ch, time_emb_dim, dropout),
            AttentionBlock3D(ch),
            ResBlock3D(ch, ch, time_emb_dim, dropout),
        ])

        self.up_blocks = nn.ModuleList()
        channel_mult_rev = list(reversed(channel_mult))
        for i, mult in enumerate(channel_mult_rev):
            out_ch = base_channels * mult
            for j in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                self.up_blocks.append(ResBlock3D(ch + skip_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
            if i < len(channel_mult_rev) - 1:
                self.up_blocks.append(
                    nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1)
                )

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv3d(ch, out_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        cond = self.cond_proj(cond)
        cond = cond[:, :, None, None, None]
        x = self.input_conv(x) + cond

        hs = []
        for block in self.down_blocks:
            if isinstance(block, ResBlock3D):
                x = block(x, t_emb)
            else:
                x = block(x)
            hs.append(x)

        for block in self.mid_blocks:
            if isinstance(block, ResBlock3D):
                x = block(x, t_emb)
            else:
                x = block(x)

        for block in self.up_blocks:
            if isinstance(block, ResBlock3D):
                skip = hs.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
            else:
                x = block(x)

        x = self.out_norm(x)
        x = F.silu(x)
        return self.out_conv(x)


class DriveDiffuser(nn.Module):
    """Diffusion-based world model for driving scene prediction.

    Predicts future 3D occupancy grids by iteratively denoising
    random noise conditioned on past BEV features and planned trajectory.

    Supports DDPM (1000 steps) for training and DDIM (50 steps) for
    fast inference. The noise schedule uses a cosine beta schedule
    for improved sample quality.

    Architecture:
        1. Encoder: Shared BEV encoder (same as OccWorld)
        2. UNet3D: Denoising network operating on (Z, H, W) occupancy
        3. Conditioner: Projects BEV features + ego motion to UNet condition
    """

    def __init__(
        self,
        encoder: nn.Module,
        unet: UNet3D,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
    ):
        super().__init__()
        self.encoder = encoder
        self.unet = unet
        self.num_timesteps = num_timesteps

        if beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(200 * 200 + 3 * 6, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def forward(
        self,
        past_images: torch.Tensor,
        past_ego_pose: torch.Tensor,
        future_ego_pose: torch.Tensor,
        future_occupancy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training (with noise injection)."""
        B, Tf, Z, H, W = future_occupancy.shape
        device = past_images.device

        bev_feat = self.encoder(past_images, past_ego_pose)

        ego_all = torch.cat([past_ego_pose, future_ego_pose], dim=1)
        ego_cond = ego_all.mean(dim=1)

        bev_pooled = bev_feat.view(B, -1)
        condition = self.cond_mlp(torch.cat([bev_pooled, ego_cond], dim=-1))

        x0 = future_occupancy[:, :, 0:1].float().expand(-1, -1, self.unet.input_conv.in_channels, -1, -1)
        x0 = x0.permute(0, 2, 1, 3, 4)

        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        noise_pred = self.unet(xt, t, condition)

        return {
            "noise_pred": noise_pred,
            "noise": noise,
            "bev_features": bev_feat,
        }

    @torch.no_grad()
    def sample(
        self,
        past_images: torch.Tensor,
        past_ego_pose: torch.Tensor,
        future_ego_pose: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """DDIM sampling for inference.

        Args:
            past_images: (B, T_past, 3, H, W)
            past_ego_pose: (B, T_past, 3)
            future_ego_pose: (B, T_future, 3)
            num_inference_steps: DDIM steps (fewer = faster)
            eta: 0 = DDIM, 1 = DDPM
            shape: (T_future, Z, H, W) override

        Returns:
            occupancy: (B, T_future, Z, H, W) predicted occupancy (0-1)
        """
        self.eval()
        B = past_images.shape[0]
        device = past_images.device
        T_future = future_ego_pose.shape[1]

        if shape is None:
            Z = 16
            H_bv, W_bv = 200, 200
        else:
            T_future, Z, H_bv, W_bv = shape

        bev_feat = self.encoder(past_images, past_ego_pose)
        ego_all = torch.cat([past_ego_pose, future_ego_pose], dim=1)
        ego_cond = ego_all.mean(dim=1)
        bev_pooled = bev_feat.view(B, -1)
        condition = self.cond_mlp(torch.cat([bev_pooled, ego_cond], dim=-1))

        channels = self.unet.input_conv.in_channels
        x = torch.randn(B, channels, T_future, H_bv, W_bv, device=device)

        timesteps = list(range(0, self.num_timesteps, self.num_timesteps // num_inference_steps))

        for i, t_step in enumerate(reversed(timesteps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            noise_pred = self.unet(x, t, condition)

            alpha = self.alphas[t_step]
            alpha_cumprod = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            if i < len(timesteps) - 1:
                t_next = timesteps[len(timesteps) - 2 - i]
                alpha_cumprod_next = self.alphas_cumprod[t_next]
            else:
                alpha_cumprod_next = torch.tensor(1.0, device=device)

            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            pred_x0 = pred_x0.clamp(-1, 1)

            dir_xt = torch.sqrt(1 - alpha_cumprod_next - eta * eta * beta) * noise_pred
            x = torch.sqrt(alpha_cumprod_next) * pred_x0 + dir_xt

            if eta > 0 and i < len(timesteps) - 1:
                x = x + eta * torch.sqrt(beta) * torch.randn_like(x)

        occupancy = x.mean(dim=1).sigmoid()
        occupancy = occupancy.permute(0, 1, 3, 4, 2)
        occupancy = occupancy.unsqueeze(2).expand(-1, -1, Z // T_future if T_future <= Z else 1, -1, -1)

        return occupancy
