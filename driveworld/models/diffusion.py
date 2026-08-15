"""DriveDiffuser: Diffusion-based World Model for Autonomous Driving.

Core idea:
- Represent future 3D occupancy as (B, T_future, num_z, H, W)
- Flatten (T_future, num_z) into channels and denoise a 2D map of shape
  (B, T_future * num_z, H, W) with a 2D UNet
- Condition on past BEV features (spatial map) and ego motion (global vector)

The 2D formulation is memory-friendly on a single 24GB GPU and avoids the
skip-connection resolution bugs of a naive 3D UNet on a short time axis.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings (Transformer-style)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock2D(nn.Module):
    """2D residual block with FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.film(emb).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock2D(nn.Module):
    """Lightweight 2D self-attention used at the UNet bottleneck."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet2D(nn.Module):
    """2D UNet for denoising future occupancy maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 64,
        cond_vec_dim: int = 256,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.cond_vec_proj = nn.Linear(cond_vec_dim, time_emb_dim)

        self.input_conv = nn.Conv2d(
            in_channels + cond_channels, base_channels, 3, padding=1
        )

        self.down_res = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            level = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = ch if j == 0 else out_ch
                level.append(ResBlock2D(in_ch, out_ch, time_emb_dim, dropout))
            self.down_res.append(level)
            ch = out_ch
            if i < len(channel_mult) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))

        self.mid_blocks = nn.ModuleList([
            ResBlock2D(ch, ch, time_emb_dim, dropout),
            AttentionBlock2D(ch),
            ResBlock2D(ch, ch, time_emb_dim, dropout),
        ])

        self.up_res = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = base_channels * mult
            skip_ch = base_channels * channel_mult[len(channel_mult) - 1 - i]
            level = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = (ch + skip_ch) if j == 0 else out_ch
                level.append(ResBlock2D(in_ch, out_ch, time_emb_dim, dropout))
            self.up_res.append(level)
            ch = out_ch
            if i < len(channel_mult) - 1:
                self.upsamples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_map: Optional[torch.Tensor] = None,
        cond_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(get_timestep_embedding(t, self.time_emb_dim))
        if cond_vec is not None:
            t_emb = t_emb + self.cond_vec_proj(cond_vec)

        if cond_map is not None:
            x = torch.cat([x, cond_map], dim=1)

        h = self.input_conv(x)
        skips = []
        for i, level in enumerate(self.down_res):
            for block in level:
                h = block(h, t_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        for block in self.mid_blocks:
            if isinstance(block, ResBlock2D):
                h = block(h, t_emb)
            else:
                h = block(h)

        for i, level in enumerate(self.up_res):
            if i > 0:
                h = self.upsamples[i - 1](h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in level:
                h = block(h, t_emb)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class DriveDiffuser(nn.Module):
    """Diffusion-based world model for future 3D occupancy prediction."""

    def __init__(
        self,
        encoder: nn.Module,
        unet: UNet2D,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        num_future_frames: int = 6,
        num_z: int = 16,
        bev_h_out: int = 200,
        bev_w_out: int = 200,
    ):
        super().__init__()
        self.encoder = encoder
        self.unet = unet
        self.num_timesteps = num_timesteps
        self.num_future_frames = num_future_frames
        self.num_z = num_z
        self.bev_h_out = bev_h_out
        self.bev_w_out = bev_w_out

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

        bev_dim = getattr(encoder, "bev_feat_dim", 128)
        cond_channels = unet.input_conv.in_channels - num_future_frames * num_z

        self.cond_proj = nn.Sequential(
            nn.Conv2d(bev_dim, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, cond_channels, 1),
        )
        self.cond_vec_ego = nn.Sequential(
            nn.Linear(num_future_frames * 3, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.cond_vec_bev = nn.Sequential(
            nn.Linear(bev_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _conditioning(
        self,
        past_images,
        past_ego_pose,
        future_ego_pose,
        past_intrinsics=None,
        past_extrinsics=None,
    ):
        bev_feat = self.encoder(
            past_images,
            past_ego_pose,
            past_intrinsics,
            past_extrinsics,
        )

        cond_map = self.cond_proj(bev_feat)
        cond_map = F.interpolate(
            cond_map, size=(self.bev_h_out, self.bev_w_out),
            mode="bilinear", align_corners=False,
        )

        ego_vec = future_ego_pose.reshape(past_images.shape[0], -1)
        bev_vec = bev_feat.mean(dim=[2, 3])
        cond_vec = torch.cat(
            [self.cond_vec_ego(ego_vec), self.cond_vec_bev(bev_vec)], dim=-1
        )
        return cond_map, cond_vec

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def forward(
        self,
        past_images: torch.Tensor,
        past_ego_pose: torch.Tensor,
        future_ego_pose: torch.Tensor,
        future_occupancy: Optional[torch.Tensor] = None,
        past_intrinsics: Optional[torch.Tensor] = None,
        past_extrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, Tf, Z, H, W = future_occupancy.shape
        device = past_images.device

        cond_map, cond_vec = self._conditioning(
            past_images,
            past_ego_pose,
            future_ego_pose,
            past_intrinsics,
            past_extrinsics,
        )

        x0 = future_occupancy.float().reshape(B, Tf * Z, H, W)
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        noise_pred = self.unet(xt, t, cond_map=cond_map, cond_vec=cond_vec)

        return {
            "noise_pred": noise_pred,
            "noise": noise,
            "bev_features": cond_map,
        }

    @torch.no_grad()
    def sample(
        self,
        past_images: torch.Tensor,
        past_ego_pose: torch.Tensor,
        future_ego_pose: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        past_intrinsics: Optional[torch.Tensor] = None,
        past_extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM-style sampling.

        Returns:
            occupancy: (B, T_future, num_z, H, W) in [0, 1]
        """
        self.eval()
        B = past_images.shape[0]
        device = past_images.device

        cond_map, cond_vec = self._conditioning(
            past_images,
            past_ego_pose,
            future_ego_pose,
            past_intrinsics,
            past_extrinsics,
        )

        channels = self.num_future_frames * self.num_z
        x = torch.randn(B, channels, self.bev_h_out, self.bev_w_out, device=device)

        timesteps = list(
            range(0, self.num_timesteps, self.num_timesteps // num_inference_steps)
        )

        for i, t_step in enumerate(reversed(timesteps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            noise_pred = self.unet(x, t, cond_map=cond_map, cond_vec=cond_vec)

            alpha_cumprod = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            if i < len(timesteps) - 1:
                t_next = timesteps[len(timesteps) - 2 - i]
                alpha_cumprod_next = self.alphas_cumprod[t_next]
            else:
                alpha_cumprod_next = torch.tensor(1.0, device=device)

            pred_x0 = (
                x - torch.sqrt(1.0 - alpha_cumprod) * noise_pred
            ) / torch.sqrt(alpha_cumprod)
            pred_x0 = pred_x0.clamp(0.0, 1.0)

            dir_xt = torch.sqrt(
                torch.clamp(1.0 - alpha_cumprod_next - eta * eta * beta, min=0.0)
            ) * noise_pred
            x = torch.sqrt(alpha_cumprod_next) * pred_x0 + dir_xt

            if eta > 0 and i < len(timesteps) - 1:
                x = x + eta * torch.sqrt(beta) * torch.randn_like(x)

        occupancy = x.clamp(0.0, 1.0)
        return occupancy.reshape(
            B, self.num_future_frames, self.num_z, self.bev_h_out, self.bev_w_out
        )