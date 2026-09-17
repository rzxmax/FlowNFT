"""
图像观测编码器（参考 examples/flow_pusht.py 的 ResNet18 路径），与 PointCloudObsEncoder 接口对齐。

ImageObsEncoder:
    输入: image (B, T, C, H, W)  C ∈ {3, 4}  +  agent_pos (B, T, A)
    输出: global_cond (B, T * (img_feat_dim + state_feat_dim))

C=4 时第一个 conv 改为 in_channels=4（接受 RGBD），并将所有 BatchNorm 替换为 GroupNorm
（GroupNorm 更适合小 batch / RL 场景，与原 flow_pusht 一致）。

可直接作为 ConditionalUnet1D 的 global_cond 使用。
"""

import torch
import torch.nn as nn
from typing import List, Type

from models.resnet import get_resnet, replace_bn_with_gn
from models.pointcloud_encoder import _build_state_mlp


class ImageEncoderResNet(nn.Module):
    """
    ResNet 主干 + 可选 4 通道 RGBD 输入（替换 conv1）+ BN→GN。

    输入: (B, C, H, W),  C ∈ {3, 4}
    输出: (B, feat_dim)，feat_dim 与 ResNet 主干一致（resnet18=512）
    """

    def __init__(self,
                 resnet_name: str = 'resnet18',
                 in_channels: int = 4,
                 weights=None,
                 use_groupnorm: bool = True):
        super().__init__()

        resnet = get_resnet(resnet_name, weights=weights)

        if in_channels != 3:
            orig_conv1 = resnet.conv1
            new_conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv1.out_channels,
                kernel_size=orig_conv1.kernel_size,
                stride=orig_conv1.stride,
                padding=orig_conv1.padding,
                bias=(orig_conv1.bias is not None),
            )
            nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
            if new_conv1.bias is not None:
                nn.init.zeros_(new_conv1.bias)
            resnet.conv1 = new_conv1

        if use_groupnorm:
            resnet = replace_bn_with_gn(resnet)

        self.backbone = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ImageObsEncoder(nn.Module):
    """
    将多帧 RGB(D) 图像 + 机器人状态编码为 global_cond 向量。

    参数
    ----
    n_obs_steps     : 历史帧数 T（取前 T 帧）
    in_channels     : 图像输入通道数（3=RGB / 4=RGBD）
    resnet_name     : 主干网络名（默认 resnet18）
    img_feat_dim    : 主干输出维度（resnet18=512；与 resnet 选型保持一致）
    agent_pos_dim   : agent_pos 向量维度（默认 19）
    state_mlp_hidden: state MLP 隐层（默认 [64, 64]）
    use_groupnorm   : 是否将 BN 替换为 GN（默认 True，对小 batch 更友好）

    输出维度
    --------
    output_dim = (img_feat_dim + state_mlp_hidden[-1]) * n_obs_steps
    """

    def __init__(self,
                 n_obs_steps: int = 2,
                 in_channels: int = 4,
                 resnet_name: str = 'resnet18',
                 img_feat_dim: int = 512,
                 agent_pos_dim: int = 19,
                 state_mlp_hidden: List[int] = None,
                 use_groupnorm: bool = True,
                 ):
        super().__init__()

        if state_mlp_hidden is None:
            state_mlp_hidden = [64, 64]

        self.n_obs_steps = n_obs_steps
        self.in_channels = in_channels
        self.img_feat_dim = img_feat_dim
        self.state_out_dim = state_mlp_hidden[-1]

        self.img_encoder = ImageEncoderResNet(
            resnet_name=resnet_name,
            in_channels=in_channels,
            use_groupnorm=use_groupnorm,
        )
        self.state_mlp = _build_state_mlp(agent_pos_dim, state_mlp_hidden)

    @property
    def output_dim(self) -> int:
        return (self.img_feat_dim + self.state_out_dim) * self.n_obs_steps

    def forward(self,
                image: torch.Tensor,
                agent_pos: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        image     : (B, T, C, H, W) — 多帧 RGB(D) 图像，float32，已归一到 [0,1] 量级
        agent_pos : (B, T, A)       — 多帧机器人状态

        返回
        ----
        global_cond : (B, output_dim)
        """
        assert image.dim() == 5, (
            f"ImageObsEncoder expects image of shape (B, T, C, H, W), got {tuple(image.shape)}"
        )
        B, T, C, H, W = image.shape
        assert C == self.in_channels, (
            f"ImageObsEncoder configured with in_channels={self.in_channels}, "
            f"but received image with C={C}"
        )
        assert agent_pos.shape[:2] == (B, T), (
            f"agent_pos shape {tuple(agent_pos.shape)} 与 image 时间维 (B={B}, T={T}) 不匹配"
        )

        img_flat = image.reshape(B * T, C, H, W)
        apos_flat = agent_pos.reshape(B * T, -1)

        img_feat = self.img_encoder(img_flat)                       # (B*T, img_feat_dim)
        state_feat = self.state_mlp(apos_flat)                      # (B*T, state_out_dim)

        combined = torch.cat([img_feat, state_feat], dim=-1)        # (B*T, feat_dim)
        global_cond = combined.reshape(B, T * combined.shape[-1])   # (B, T*feat_dim)

        return global_cond
