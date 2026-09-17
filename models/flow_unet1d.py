"""
FlowPolicy-compatible ConditionalUnet1D for 1D action sequences.

Adapted from:
  FlowPolicy/FlowPolicy/flow_policy_3d/model/flow/conditional_unet1d.py

Key changes vs models/unet.py:
  - Timestep should be scaled OUTSIDE (caller does t * 99) for proper
    sinusoidal embedding utilisation
  - Removed var_est (variance estimation) module — not used in our pipeline
  - Added mlp_film conditioning option (default: film)
  - kernel_size default = 5 (matches FlowPolicy configs)

Input/output convention (same as original unet.py):
  sample: (B, action_horizon, action_dim)
  output: (B, action_horizon, action_dim)  — velocity field
"""

import math
from typing import Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding.  Input should be in *large* integer
    range (e.g. t * 99) for good frequency coverage."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with optional FiLM / add / mlp_film conditioning."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 cond_dim: int,
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 condition_type: str = 'film'):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels,
                        kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels,
                        kernel_size, n_groups=n_groups),
        ])

        self.condition_type = condition_type
        self.out_channels = out_channels

        if condition_type == 'film':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels * 2),
            )
        elif condition_type == 'add':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
            )
        elif condition_type == 'mlp_film':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, out_channels * 2),
            )
        else:
            raise NotImplementedError(
                f"condition_type '{condition_type}' not supported. "
                f"Choose from: film, add, mlp_film")

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        """
        x   : (B, in_channels, T)
        cond: (B, cond_dim)
        """
        out = self.blocks[0](x)

        if cond is not None:
            embed = self.cond_encoder(cond)              # (B, C*2) or (B, C)
            if self.condition_type in ('film', 'mlp_film'):
                embed = embed.reshape(embed.shape[0], 2, self.out_channels)
                scale = embed[:, 0, :, None]             # (B, C, 1)
                bias  = embed[:, 1, :, None]             # (B, C, 1)
                out = scale * out + bias
            else:  # 'add'
                out = out + embed[:, :, None]

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# ---------------------------------------------------------------------------
# Main UNet
# ---------------------------------------------------------------------------

class FlowConditionalUnet1D(nn.Module):
    """
    Conditional 1-D UNet for Flow Matching velocity prediction.

    Parameters
    ----------
    input_dim : int
        Dimension of each action token (e.g. 4 for pd_ee_delta_pos).
    global_cond_dim : int | None
        Dimension of the global conditioning vector (e.g. obs_dim).
        When None, no global conditioning is applied.
    diffusion_step_embed_dim : int
        Size of the timestep embedding (default 256).
    down_dims : list[int]
        Channel sizes for each down-sampling level (default [256,512,1024]).
    kernel_size : int
        Convolution kernel size (default 5, matching FlowPolicy default).
    n_groups : int
        Number of groups for GroupNorm.
    condition_type : str
        How the conditioning vector is injected: 'film' (default), 'add',
        or 'mlp_film'.

    Calling convention
    ------------------
    The caller is responsible for **scaling the timestep**:
        velocity = model(sample, timestep=t * 99, global_cond=obs)
    where t ∈ [0, 1].  This gives the SinusoidalPosEmb a proper dynamic
    range (matching FlowPolicy practice).
    """

    def __init__(self,
                 input_dim: int,
                 global_cond_dim: int = None,
                 diffusion_step_embed_dim: int = 256,
                 down_dims=None,
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 condition_type: str = 'film'):
        super().__init__()

        if down_dims is None:
            down_dims = [256, 512, 1024]

        self.condition_type = condition_type

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        # ---- timestep encoder ----
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        # ---- mid ----
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type),
        ])

        # ---- down ----
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # ---- up ----
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = (ind >= len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        # ---- final ----
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("FlowConditionalUnet1D: %e parameters", n_params)

    # ------------------------------------------------------------------
    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """
        Parameters
        ----------
        sample     : (B, action_horizon, action_dim)  —  x_t
        timestep   : (B,) or scalar  — **already scaled** (e.g. t * 99)
        global_cond: (B, global_cond_dim)

        Returns
        -------
        velocity   : (B, action_horizon, action_dim)
        """
        # (B, T, C) → (B, C, T)
        x = sample.permute(0, 2, 1).contiguous()

        # ---- timestep embedding ----
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(
                [timestep], dtype=torch.float32, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).to(sample.device)

        timestep = timestep.expand(x.shape[0]).to(
            device=sample.device, dtype=torch.float32)
        timestep_embed = self.diffusion_step_encoder(timestep)  # (B, dsed)

        # ---- global condition ----
        if global_cond is not None:
            global_feature = torch.cat([timestep_embed, global_cond], dim=-1)
        else:
            global_feature = timestep_embed

        # ---- down ----
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # ---- mid ----
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # ---- up ----
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # ---- final ----
        x = self.final_conv(x)

        # (B, C, T) → (B, T, C)
        velocity = x.permute(0, 2, 1).contiguous()
        return velocity
