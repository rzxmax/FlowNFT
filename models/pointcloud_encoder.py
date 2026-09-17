"""
独立点云观测编码器，不依赖 FlowPolicy 路径。

PointCloudObsEncoder:
    输入: point_cloud (B, T, N, 3)  +  agent_pos (B, T, A)
    输出: global_cond (B, T * (pcd_feat_dim + state_feat_dim))

可直接作为 ConditionalUnet1D 的 global_cond 使用。
"""

import torch
import torch.nn as nn
from typing import List, Type


# ---------------------------------------------------------------------------
# PointNet-style per-point MLP + global max-pool encoder (XYZ only)
# ---------------------------------------------------------------------------

class PointNetEncoderXYZ(nn.Module):
    """
    轻量 PointNet 编码器（仅 XYZ，无 T-Net）。

    网络结构：
        Linear(3→64) → [LayerNorm] → ReLU
        Linear(64→128) → [LayerNorm] → ReLU
        Linear(128→256) → [LayerNorm] → ReLU
        GlobalMaxPool
        Linear(256→out_channels) → [LayerNorm / none]

    输入: (B, N, 3)
    输出: (B, out_channels)
    """

    def __init__(self,
                 out_channels: int = 256,
                 use_layernorm: bool = True,
                 final_norm: str = 'layernorm',
                 ):
        super().__init__()
        hidden = [64, 128, 256]

        def _block(in_c, out_c):
            layers = [nn.Linear(in_c, out_c)]
            if use_layernorm:
                layers.append(nn.LayerNorm(out_c))
            layers.append(nn.ReLU())
            return layers

        self.mlp = nn.Sequential(
            *_block(3, hidden[0]),
            *_block(hidden[0], hidden[1]),
            *_block(hidden[1], hidden[2]),
        )

        if final_norm == 'layernorm':
            self.projection = nn.Sequential(
                nn.Linear(hidden[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        else:
            self.projection = nn.Linear(hidden[-1], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 3) → (B, out_channels)"""
        x = self.mlp(x)                 # (B, N, 256)
        x = torch.max(x, dim=1)[0]     # global max pool → (B, 256)
        return self.projection(x)       # (B, out_channels)


# ---------------------------------------------------------------------------
# State MLP
# ---------------------------------------------------------------------------

def _build_state_mlp(in_dim: int,
                     hidden_sizes: List[int],
                     activation: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Combined encoder
# ---------------------------------------------------------------------------

class PointCloudObsEncoder(nn.Module):
    """
    将多帧点云 + 机器人状态编码为 global_cond 向量。

    参数
    ----
    n_obs_steps       : 历史帧数 T（取前 T 帧）
    pcd_out_dim       : PointNet 输出维度（默认 256）
    agent_pos_dim     : agent_pos 向量维度（默认 19）
    state_mlp_hidden  : state MLP 隐层大小列表（默认 [64, 64]）
    use_layernorm     : PointNet 中间层是否用 LayerNorm
    final_norm        : PointNet 最终 projection 的 norm 类型（'layernorm'/'none'）

    输出维度
    --------
    output_dim = (pcd_out_dim + state_mlp_hidden[-1]) * n_obs_steps
    """

    def __init__(self,
                 n_obs_steps: int = 2,
                 pcd_out_dim: int = 256,
                 agent_pos_dim: int = 19,
                 state_mlp_hidden: List[int] = None,
                 use_layernorm: bool = True,
                 final_norm: str = 'layernorm',
                 ):
        super().__init__()

        if state_mlp_hidden is None:
            state_mlp_hidden = [64, 64]

        self.n_obs_steps = n_obs_steps
        self.pcd_out_dim = pcd_out_dim
        self.state_out_dim = state_mlp_hidden[-1]

        self.pcd_encoder = PointNetEncoderXYZ(
            out_channels=pcd_out_dim,
            use_layernorm=use_layernorm,
            final_norm=final_norm,
        )
        self.state_mlp = _build_state_mlp(agent_pos_dim, state_mlp_hidden)

    @property
    def output_dim(self) -> int:
        return (self.pcd_out_dim + self.state_out_dim) * self.n_obs_steps

    def forward(self,
                point_cloud: torch.Tensor,
                agent_pos: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        point_cloud : (B, T, N, 3) — 多帧点云
        agent_pos   : (B, T, A)    — 多帧机器人状态

        返回
        ----
        global_cond : (B, output_dim)
        """
        B, T, N, _ = point_cloud.shape

        # 将 T 帧合并到 batch 维度统一处理
        pcd_flat  = point_cloud.reshape(B * T, N, 3)      # (B*T, N, 3)
        apos_flat = agent_pos.reshape(B * T, -1)           # (B*T, A)

        pcd_feat   = self.pcd_encoder(pcd_flat)            # (B*T, pcd_out_dim)
        state_feat = self.state_mlp(apos_flat)             # (B*T, state_out_dim)

        combined = torch.cat([pcd_feat, state_feat], dim=-1)   # (B*T, feat_dim)
        global_cond = combined.reshape(B, T * combined.shape[-1])  # (B, T*feat_dim)

        return global_cond
