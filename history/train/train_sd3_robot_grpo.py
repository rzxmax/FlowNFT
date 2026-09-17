#!/usr/bin/env python
"""
机器人动作生成的 Flow-GRPO 训练脚本
"""

import sys
import os

# !!!!! 重要：在导入任何 CUDA 库之前设置 GPU !!!!!
# 提前解析 --gpu 参数并设置 CUDA_VISIBLE_DEVICES
gpu_id = 0  # 默认值
for i, arg in enumerate(sys.argv):
    if arg.startswith('--gpu='):
        gpu_id = int(arg.split('=')[1])
        break
    elif arg == '--gpu' and i + 1 < len(sys.argv):
        gpu_id = int(sys.argv[i + 1])
        break

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print(f"\n{'='*70}")
print(f"[早期设置] 使用物理 GPU {gpu_id}, 设置 CUDA_VISIBLE_DEVICES={gpu_id}")
print(f"[早期设置] 后续 PyTorch 将把物理 GPU {gpu_id} 映射为 cuda:0")
print(f"[早期设置] 当前命令行参数: {' '.join(sys.argv)}")
print(f"{'='*70}\n")


# 现在才导入 CUDA 相关库
sys.dont_write_bytecode = True
sys.path.insert(0, './flow_grpo')  # 改为 insert(0, ...)
sys.path.insert(0, './flow_grpo/diffusers_patch')  # 改为 insert(0, ...)
sys.path.insert(0, './external/models')  # 改为 insert(0, ...)
sys.path.insert(0, './external')  # 改为 insert(0, ...)

from collections import defaultdict
import contextlib
import datetime
import math
from concurrent import futures
import time
import json
import logging
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FlowMatchEulerDiscreteScheduler
import numpy as np
import torch
import wandb
from functools import partial
import tqdm
import random
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker

# 导入机器人相关模块
from unet import ConditionalUnet1D
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import RobotActionPipeline, ActionDecoder, pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import ode_step_with_logprob

# 导入 ManiSkill 环境
import gymnasium as gym
import mani_skill.envs

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/robot_base.py", "Training configuration.")
flags.DEFINE_bool("train", True, "Run training")
flags.DEFINE_bool("test", False, "Run testing")
flags.DEFINE_bool("pretrain", False, "Run pretraining on demonstration data")
flags.DEFINE_bool("pretrain_auto", False, "Enable auto mode for pretraining (auto stop when success)")
flags.DEFINE_string("checkpoint", "", "Path to checkpoint for testing")
flags.DEFINE_string("pretrain_checkpoint", "", "Path to pretrain checkpoint to load for GRPO training")
flags.DEFINE_integer("num_test_episodes", 5, "Number of episodes to test")
flags.DEFINE_integer("gpu", 0, "GPU device to use (0 or 1)")
flags.DEFINE_bool("sampletest", False, "Run sample test to generate trajectory videos")
# 预训练 loss 类型（互斥，覆盖 config.pretrain.loss_type）
flags.DEFINE_bool("action_loss", False, "Pretrain with original MSE(pred_actions, target_actions)")
flags.DEFINE_bool("velocity_loss", False, "Pretrain with flow matching velocity MSE(v_theta, x1-x0)")

# 配置简洁的日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # 只显示消息内容
    force=True
)
logger = get_logger(__name__)
logger.logger.setLevel(logging.INFO)
# 移除默认的handler，使用我们自定义的格式
if logger.logger.handlers:
    for handler in logger.logger.handlers[:]:
        logger.logger.removeHandler(handler)
# 添加简洁格式的handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.logger.addHandler(handler)
logger.logger.propagate = False


def _obs_to_tensor(obs, device):
    """辅助函数：转换观测为tensor"""
    if isinstance(obs, torch.Tensor):
        obs_tensor = obs.to(device, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
    else:
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    return obs_tensor


def _prepare_obs_tensor(obs, device):
    """统一的obs转tensor逻辑（用于筛选和采样）"""
    if isinstance(obs, torch.Tensor):
        if obs.device != device:
            obs_tensor = obs.to(device, dtype=torch.float32)
        else:
            obs_tensor = obs.to(dtype=torch.float32)
    else:
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
    
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    
    return obs_tensor


# ===========================================================================
# 点云观测辅助函数
# 与 convert_pointcloud_hdf5_to_zarr.py 的过滤 / 采样逻辑完全对齐，
# 保证在线 rollout 的 obs 格式与训练数据一致。
# ===========================================================================

# PickCube-v1 分割ID（Panda 机械臂 link 1-15 + 物块 18，排除桌面 16/地面 17）
_PCD_ROBOT_SEG_IDS = set(range(1, 16))
_PCD_CUBE_SEG_ID   = 18
_PCD_KEEP_SEG_LIST = list(_PCD_ROBOT_SEG_IDS | {_PCD_CUBE_SEG_ID})

try:
    import pytorch3d.ops as _pcd_torch3d_ops
    _PCD_HAS_PYTORCH3D = True
except ImportError:
    _PCD_HAS_PYTORCH3D = False

_pcd_rng = np.random.default_rng(42)


def _to_numpy_pcd(x) -> np.ndarray:
    """Tensor 或 ndarray → float32 ndarray"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32)


def sample_pcd(xyzw: np.ndarray, n_points: int, seg=None) -> np.ndarray:
    """
    从单帧点云 xyzw (N,4) 中采样 n_points 个点。
    与 convert_pointcloud_hdf5_to_zarr.py 处理逻辑完全一致：
      1. w > 0 基础过滤
      2. 语义过滤：只保留机械臂(1-15) + 物块(18)，排除桌面/地面背景
      3. 有效点 < n_points 时补零；否则 FPS（有 pytorch3d）或随机均匀采样

    Args:
        xyzw:     (N,4) 或 (1,N,4) float32
        n_points: 目标采样点数
        seg:      (N,1) 或 (1,N,1) int，None 时跳过语义过滤

    Returns:
        (n_points, 3) float32
    """
    if xyzw.ndim == 3:
        xyzw = xyzw[0]
    if seg is not None and seg.ndim == 3:
        seg = seg[0]

    valid_mask = xyzw[:, 3] > 0

    if seg is not None:
        seg_flat = np.array(seg, dtype=np.int32).squeeze(-1)
        valid_mask = valid_mask & np.isin(seg_flat, _PCD_KEEP_SEG_LIST)

    valid_xyz = xyzw[valid_mask, :3].astype(np.float32)
    n_valid   = len(valid_xyz)

    if n_valid == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    if n_valid < n_points:
        pad = np.zeros((n_points - n_valid, 3), dtype=np.float32)
        return np.concatenate([valid_xyz, pad], axis=0)

    if _PCD_HAS_PYTORCH3D and torch.cuda.is_available():
        pts = torch.from_numpy(valid_xyz).unsqueeze(0).cuda()   # (1, N, 3)
        k   = torch.tensor([n_points]).cuda()
        _, idx = _pcd_torch3d_ops.sample_farthest_points(points=pts, K=k)
        return valid_xyz[idx.squeeze(0).cpu().numpy()]
    else:
        idx = _pcd_rng.choice(n_valid, n_points, replace=False)
        return valid_xyz[idx]


def batched_fps_sample(
    padded_pts: torch.Tensor,
    valid_count: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """对一个 batch 的"过滤后但未采样"点云做 GPU 批量 FPS。

    每帧 padded_pts[b] 的前 valid_count[b] 个点是有效点，后面是补零 padding。
    使用 pytorch3d.sample_farthest_points 的 lengths 参数让 FPS 只在有效点中选取。

    采样规则（与 sample_pcd / convert 脚本对齐）：
      - valid == 0          → 全零 (n_points, 3)
      - 0 < valid < n_points → 前 valid 个保留，余下补零（不做 FPS）
      - valid >= n_points    → GPU FPS 采样 n_points 个（pytorch3d.sample_farthest_points
                              默认 random_start_point=True，每次起点随机，得到不同子集）

    Args:
        padded_pts:  (B, max_pts, 3)  float32, 在 GPU 上
        valid_count: (B,)             long, 每帧有效点数
        n_points:    目标采样点数（如 512）

    Returns:
        (B, n_points, 3) float32
    """
    assert padded_pts.dim() == 3 and padded_pts.shape[-1] == 3, (
        f"padded_pts shape={padded_pts.shape}, expected (B, max_pts, 3)"
    )
    assert valid_count.dim() == 1 and valid_count.shape[0] == padded_pts.shape[0], (
        f"valid_count shape={valid_count.shape}, expected ({padded_pts.shape[0]},)"
    )
    B, max_pts, _ = padded_pts.shape
    device = padded_pts.device
    out = torch.zeros(B, n_points, 3, device=device, dtype=padded_pts.dtype)

    if not (_PCD_HAS_PYTORCH3D and padded_pts.is_cuda):
        # 退化路径：纯 numpy 随机均匀采样
        pts_np = padded_pts.detach().cpu().numpy()
        vc_np  = valid_count.detach().cpu().numpy()
        rng = _pcd_rng
        out_np = np.zeros((B, n_points, 3), dtype=np.float32)
        for b in range(B):
            vc = int(vc_np[b])
            if vc == 0:
                continue
            if vc <= n_points:
                out_np[b, :vc] = pts_np[b, :vc]
                continue
            idx = rng.choice(vc, n_points, replace=False)
            out_np[b] = pts_np[b, idx]
        return torch.from_numpy(out_np).to(device)

    # ── GPU 批量 FPS（pytorch3d）─────────────────────────────────────
    # sample_farthest_points 的 lengths 参数会让每行只在前 lengths[i] 个点中采样
    # 但当 lengths[i] < K 时会出错，需要先按 valid 分两组处理
    valid_count = valid_count.to(device=device, dtype=torch.long)

    # 第一组：valid >= n_points → GPU FPS
    fps_mask = valid_count >= n_points
    if fps_mask.any():
        sub_pts  = padded_pts[fps_mask]                       # (B', max_pts, 3)
        sub_vc   = valid_count[fps_mask].clamp(max=max_pts)   # (B',)
        # pytorch3d 要求 lengths 是 int64
        sampled, _idx = _pcd_torch3d_ops.sample_farthest_points(
            points=sub_pts,
            lengths=sub_vc,
            K=n_points,
            random_start_point=True,
        )
        out[fps_mask] = sampled

    # 第二组：0 < valid < n_points → 前 vc 个保留 + 补零
    small_mask = (valid_count > 0) & (valid_count < n_points)
    if small_mask.any():
        small_idx_list = small_mask.nonzero(as_tuple=False).flatten().tolist()
        for b in small_idx_list:
            vc = int(valid_count[b].item())
            out[b, :vc] = padded_pts[b, :vc]

    return out


def extract_pointcloud_obs(raw_obs: dict, n_points: int) -> dict:
    """
    将 ManiSkill obs_mode='pointcloud' 的单步 obs 转为策略输入格式：
        {'point_cloud': (n_points, 3), 'agent_pos': (19,)}

    state = [qpos(9) + goal_pos(3) + tcp_pose(7)] = 19维，
    与训练 zarr 数据集中 data/state 的拼接顺序完全一致。

    Args:
        raw_obs:  ManiSkill 环境返回的 obs dict（obs_mode='pointcloud'）
        n_points: 目标点数（通常 512）
    """
    xyzw     = _to_numpy_pcd(raw_obs['pointcloud']['xyzw'])
    qpos     = _to_numpy_pcd(raw_obs['agent']['qpos'])
    goal_pos = _to_numpy_pcd(raw_obs['extra']['goal_pos'])
    tcp_pose = _to_numpy_pcd(raw_obs['extra']['tcp_pose'])

    seg = None
    if 'segmentation' in raw_obs['pointcloud']:
        seg = _to_numpy_pcd(raw_obs['pointcloud']['segmentation'])

    # GPU sim 并行 env 可能有 batch 维 (1, D)
    if qpos.ndim == 2:     qpos     = qpos[0]
    if goal_pos.ndim == 2: goal_pos = goal_pos[0]
    if tcp_pose.ndim == 2: tcp_pose = tcp_pose[0]

    agent_pos = np.concatenate([qpos, goal_pos, tcp_pose], axis=0)  # (19,)

    return {
        'point_cloud': sample_pcd(xyzw, n_points, seg),   # (n_points, 3)
        'agent_pos':   agent_pos,                           # (19,)
    }


def stack_pointcloud_obs(obs_deque, n_obs_steps: int, device) -> dict:
    """
    将最近 n_obs_steps 帧堆叠为 PointCloudObsEncoder 所需的输入格式。

    Returns:
        {
            'point_cloud': Tensor (1, n_obs_steps, n_points, 3),
            'agent_pos':   Tensor (1, n_obs_steps, 19),
        }
    帧数不足时复制最早帧填充。
    """
    obs_list = list(obs_deque)[-n_obs_steps:]
    while len(obs_list) < n_obs_steps:
        obs_list.insert(0, obs_list[0])

    pcd_stack  = np.stack([o['point_cloud'] for o in obs_list])   # (T, n_pts, 3)
    apos_stack = np.stack([o['agent_pos']   for o in obs_list])   # (T, 19)

    return {
        'point_cloud': torch.from_numpy(pcd_stack).unsqueeze(0).to(device),   # (1, T, n_pts, 3)
        'agent_pos':   torch.from_numpy(apos_stack).unsqueeze(0).to(device),  # (1, T, 19)
    }


# ===========================================================================
# Image (RGB/RGBD) obs 抽取辅助 —— 与训练用 ManiSkillZarrImageDataset 严格对齐
# ===========================================================================
# ManiSkill 在 obs_mode='rgbd' 下的 raw_obs 结构（与 HDF5 字段一致）：
#   raw_obs['sensor_data'][camera]['rgb']    (1, H, W, 3) uint8
#   raw_obs['sensor_data'][camera]['depth']  (1, H, W, 1) int16  (单位: mm)
#   raw_obs['agent']['qpos']                 (1, 9)
#   raw_obs['extra']['goal_pos']             (1, 3)
#   raw_obs['extra']['tcp_pose']             (1, 7)
#
# 归一化方式：
#   RGB   : uint8 / 255.0       → float32 ∈ [0, 1]
#   Depth : int16 / 1000.0 (mm→m) → clip [0, depth_max] / depth_max → ∈ [0, 1]
# 与 ManiSkillZarrImageDataset 中的预处理逻辑完全一致，保证训练-推理一致。

_DEPTH_MM_SCALE = 1000.0   # ManiSkill 原始深度 int16 单位 mm，/1000 → 米


def _to_numpy_img(x):
    """统一把 torch.Tensor / np.ndarray 转 numpy。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_image_obs(raw_obs: dict,
                      camera: str = 'base_camera',
                      use_depth: bool = True,
                      depth_max: float = 5.0) -> dict:
    """
    将 ManiSkill obs_mode='rgbd' 的单步 obs 转为策略输入格式：
        {
            'image':     (C, H, W) float32,  C=4 (RGBD) 或 C=3 (RGB only)
            'agent_pos': (19,)     float32
        }

    state = [qpos(9) + goal_pos(3) + tcp_pose(7)] = 19 维，与训练 zarr 中
    data/state 的拼接顺序完全一致。
    """
    sensor = raw_obs['sensor_data'][camera]

    rgb = _to_numpy_img(sensor['rgb'])         # (1, H, W, 3) 或 (H, W, 3)
    if rgb.ndim == 4:
        rgb = rgb[0]                            # → (H, W, 3) uint8
    rgb_chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)   # (3, H, W)

    if use_depth:
        depth_raw = _to_numpy_img(sensor['depth'])  # (1, H, W, 1) 或 (H, W, 1)
        if depth_raw.ndim == 4:
            depth_raw = depth_raw[0]                # → (H, W, 1)
        depth_m = depth_raw.astype(np.float32) / _DEPTH_MM_SCALE   # mm → m
        depth_m = np.clip(depth_m, 0.0, depth_max) / max(depth_max, 1e-6)
        depth_chw = depth_m.transpose(2, 0, 1)      # (1, H, W)
        image_chw = np.concatenate([rgb_chw, depth_chw], axis=0)   # (4, H, W)
    else:
        image_chw = rgb_chw                                         # (3, H, W)

    qpos     = _to_numpy_img(raw_obs['agent']['qpos'])
    goal_pos = _to_numpy_img(raw_obs['extra']['goal_pos'])
    tcp_pose = _to_numpy_img(raw_obs['extra']['tcp_pose'])

    if qpos.ndim == 2:     qpos     = qpos[0]
    if goal_pos.ndim == 2: goal_pos = goal_pos[0]
    if tcp_pose.ndim == 2: tcp_pose = tcp_pose[0]

    agent_pos = np.concatenate([qpos, goal_pos, tcp_pose], axis=0).astype(np.float32)  # (19,)

    return {
        'image':     image_chw.astype(np.float32),
        'agent_pos': agent_pos,
    }


def stack_image_obs(obs_deque, n_obs_steps: int, device) -> dict:
    """
    将最近 n_obs_steps 帧堆叠为 ImageObsEncoder 所需的输入格式。

    Returns:
        {
            'image':     Tensor (1, n_obs_steps, C, H, W),
            'agent_pos': Tensor (1, n_obs_steps, 19),
        }
    帧数不足时复制最早帧填充。
    """
    obs_list = list(obs_deque)[-n_obs_steps:]
    while len(obs_list) < n_obs_steps:
        obs_list.insert(0, obs_list[0])

    img_stack  = np.stack([o['image']     for o in obs_list])   # (T, C, H, W)
    apos_stack = np.stack([o['agent_pos'] for o in obs_list])   # (T, 19)

    return {
        'image':     torch.from_numpy(img_stack).unsqueeze(0).to(device),    # (1, T, C, H, W)
        'agent_pos': torch.from_numpy(apos_stack).unsqueeze(0).to(device),   # (1, T, 19)
    }


# ===========================================================================
def sample_complete_trajectories(
    env, pipeline, action_unet, action_decoder,
    num_state_groups, samples_per_group,
    config, accelerator, autocast, seed_offset,
    epoch=0  # 新增：用于噪声退火
):
    """
    完整轨迹采样：从初始状态执行到任务完成或最大步数
    
    实现：
    - GRPO分组：num_state_groups个不同的初始状态（通过episode_id）
    - 每个初始状态重复采样samples_per_group次（利用策略随机性）
    - 同时设置seed和episode_id，确保同组内初始状态完全相同（包括目标位置）
    - Valid Action Mask：跟踪每条轨迹的首次done步数
    - 轨迹长度：记录每条轨迹的总step数和Flow调用次数
    
    返回：
    - 样本列表，每个样本包含：
      - obs_list: 每次Flow调用的obs
      - latents_list: 每次Flow调用的latent序列
      - log_probs_list: 每次Flow调用的log_prob序列
      - timesteps: scheduler timesteps
      - reward: 累积总奖励
      - num_steps: 轨迹总步数
      - num_flow_calls: Flow调用次数
      - first_done_step: 首次done的步数（-1表示未done）
      - terminated: 是否成功
      - group_id: GRPO组ID
      - episode_id: 初始状态的episode_id
    """
    action_unet.eval()
    action_decoder.eval()
    
    num_envs = num_state_groups * samples_per_group
    
    # 修改：使用episode_id控制初始状态
    # 生成 num_state_groups 个不同的episode_id（每个对应一个初始状态）
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed_offset)
    
    episode_ids = torch.randint(
        low=0,
        high=10000,  # ManiSkill的episode空间
        size=(num_state_groups,),
        generator=generator,
        dtype=torch.int64
    )
    
    # 每个episode_id重复 samples_per_group 次（同组使用相同初始状态）
    all_episode_ids = episode_ids.repeat_interleave(samples_per_group)
    
    # 转换为列表以传递给ManiSkill（GPU环境需要tensor或list）
    if isinstance(all_episode_ids, torch.Tensor):
        all_episode_ids_list = all_episode_ids.tolist()
    else:
        all_episode_ids_list = all_episode_ids
    
    # ✅ 关键修改：同时设置seed和episode_id（参考RLinf）
    # 将episode_id也作为seed，确保完全固定初始状态（包括目标位置）
    all_seeds = all_episode_ids_list
    
    # 使用seed和episode_id重置环境（确保同组轨迹有完全相同的初始状态）
    obs, info = env.reset(seed=all_seeds, options={'episode_id': all_episode_ids_list})
    
    # 跟踪每个环境的轨迹数据
    env_obs_list = [[] for _ in range(num_envs)]
    env_clean_actions_list = [[] for _ in range(num_envs)]  # ✅ DiffusionNFT: 保存clean actions
    env_step_rewards = [[] for _ in range(num_envs)]  # 改为记录每一步的奖励
    env_step_dones = [[] for _ in range(num_envs)]    # 记录每一步的done状态
    
    # ✅ 滑动窗口：保存每一步的obs和action历史
    env_obs_history = [[] for _ in range(num_envs)]      # 每步obs
    env_action_history = [[] for _ in range(num_envs)]   # 每步action
    
    # StepNFT 数据：每个 flow_call 保存一组 (x_t, v_t, x_next, step_idx, std_t)
    # 仅当 config.train.loss_type == "stepnft" 时填充
    use_stepnft = getattr(config.train, 'loss_type', 'rollout') == 'stepnft'
    env_nft_traces = [[] for _ in range(num_envs)] if use_stepnft else None
    
    env_done = torch.zeros(num_envs, dtype=torch.bool, device=accelerator.device)
    env_steps = torch.zeros(num_envs, dtype=torch.long, device=accelerator.device)
    env_first_done_step = torch.full((num_envs,), -1, dtype=torch.long, device=accelerator.device)
    
    # ✅ 添加：相对奖励计算（参考RLinf）
    # 跟踪每个环境的上一步奖励，用于计算奖励差值
    env_prev_reward = torch.zeros(num_envs, dtype=torch.float32, device=accelerator.device)
    
    max_steps = config.max_episode_steps
    flow_call_count = 0
    
    # 持续执行直到所有环境完成或达到最大步数
    while not env_done.all() and flow_call_count * config.action_horizon < max_steps:
        active_mask = ~env_done
        
        if active_mask.any():
            obs_tensor = _prepare_obs_tensor(obs, accelerator.device)
            
            # Flow生成动作（ODE确定性采样）
            with autocast():
                with torch.no_grad():
                    # ✅ DiffusionNFT风格：为每个样本创建独立的generator
                    # 关键设计：
                    # 1. 相同group内：相同episode_id（初始状态相同）+ 不同sample_id（初始动作噪声不同）
                    # 2. 不同Flow调用：flow_call_count递增（轨迹时序变化）
                    # ✅ 修复：使用obs_tensor的实际batch_size而不是假定的num_envs
                    actual_batch_size = obs_tensor.shape[0]
                    generators = []
                    for env_idx in range(num_envs):
                        # 计算group_id和sample_id
                        group_id = env_idx // samples_per_group
                        sample_id = env_idx % samples_per_group
                        
                        # 基础seed：每个group不同
                        base_seed = episode_ids[group_id].item()
                        
                        # 组合seed：
                        # - base_seed: 区分不同group
                        # - sample_id * 1000: 区分同group内的不同样本（DiffusionNFT风格）
                        # - flow_call_count * 100000: 区分同一轨迹的不同时刻
                        flow_seed = base_seed + sample_id * 1000 + flow_call_count * 100000
                        
                        gen = torch.Generator(device=accelerator.device)
                        gen.manual_seed(flow_seed)
                        generators.append(gen)
                    
                    if use_stepnft:
                        # StepNFT 路径：SDE 多步采样，每个样本随机选一步保存 nft 数据
                        # 与 pipeline_with_logprob 保持一致：使用 pipeline.normalize_obs 做 obs 归一化
                        obs_normalizer = getattr(pipeline, 'normalize_obs', None)
                        # action_dim 从 pipeline 或 action_decoder 上拿（pipeline.action_dim 由初始化设置）
                        sde_action_dim = getattr(pipeline, 'action_dim', None)
                        if sde_action_dim is None:
                            sde_action_dim = getattr(action_decoder, 'action_dim', None)
                        if sde_action_dim is None:
                            raise ValueError("无法从 pipeline 或 action_decoder 推导 action_dim")
                        actions_batch, nft_traces_batch = denoise_with_sde_snap(
                            obs_tensor=obs_tensor,
                            action_unet=action_unet,
                            action_decoder=action_decoder,
                            num_steps=config.sample.num_steps,
                            noise_level=config.train.stepnft_noise_level,
                            action_horizon=config.action_horizon,
                            action_dim=sde_action_dim,
                            generators=generators,
                            device=accelerator.device,
                            obs_normalizer=obs_normalizer,
                        )
                    else:
                        # 现有 ODE rollout 路径，行为不变
                        result = pipeline_with_logprob(
                            pipeline, obs=obs_tensor,
                            num_inference_steps=config.sample.num_steps,
                            output_type="clipped", return_dict=True,
                            action_decoder=action_decoder,
                            generator=generators,
                        )
                        actions_batch = result["actions"]
                        nft_traces_batch = None
            
            # ✅ DiffusionNFT: 保存clean actions (ODE采样的确定性输出)
            for i in range(num_envs):
                if active_mask[i]:
                    clean_actions = actions_batch[i].clone()  # [action_horizon, action_dim]
                    env_clean_actions_list[i].append(clean_actions)
                    env_obs_list[i].append(obs_tensor[i:i+1].clone())
                    
                    # StepNFT：保存当前 flow_call 的 nft 数据
                    if use_stepnft and nft_traces_batch is not None:
                        env_nft_traces[i].append({
                            "nft_xt": nft_traces_batch["nft_xt"][i].clone(),
                            "nft_v": nft_traces_batch["nft_v"][i].clone(),
                            "nft_xnext": nft_traces_batch["nft_xnext"][i].clone(),
                            "nft_step_index": nft_traces_batch["nft_step_index"][i].item(),
                            "nft_std": nft_traces_batch["nft_std"][i].item(),
                        })
            
            # 执行action_horizon步
            for j in range(config.action_horizon):
                if env_done.all():
                    break
                
                # 每步动态计算active_mask
                step_active_mask = ~env_done
                
                action_step = actions_batch[:, j].contiguous()
                
                # ✅ 滑动窗口：在执行前保存当前obs和即将执行的action
                obs_tensor_before = _prepare_obs_tensor(obs, accelerator.device)
                for env_idx in range(num_envs):
                    if step_active_mask[env_idx]:
                        env_obs_history[env_idx].append(obs_tensor_before[env_idx:env_idx+1].clone())
                        env_action_history[env_idx].append(action_step[env_idx:env_idx+1].clone())
                
                obs, reward, terminated, truncated, info = env.step(action_step)
                
                # 转换为tensor
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, device=accelerator.device)
                if not isinstance(terminated, torch.Tensor):
                    terminated = torch.tensor(terminated, device=accelerator.device)
                
                # ✅ 相对奖励计算（参考RLinf）
                # 计算奖励差值而非绝对值，避免累积奖励导致的长度偏差
                reward_diff = reward - env_prev_reward
                env_prev_reward = reward.clone()  # 更新上一步奖励
                
                # ✅ 修复：只记录第一个episode的数据（在done之前）
                # 确保每条轨迹只包含从reset到首次done的完整episode
                for env_idx in range(num_envs):
                    if step_active_mask[env_idx]:  # 只在环境未done时记录
                        env_step_rewards[env_idx].append(reward_diff[env_idx].item())  # 使用相对奖励
                        env_step_dones[env_idx].append(terminated[env_idx].item())
                
                # 记录首次done的步数（Valid Action Mask）
                newly_done = terminated & (env_first_done_step == -1)
                env_first_done_step[newly_done] = env_steps[newly_done]
                
                # ✅ 关键修复：累积done状态（而非覆盖）
                # 这确保一旦环境done，它会保持done状态
                # 防止auto_reset导致的状态重置（虽然我们已禁用auto_reset）
                env_done = env_done | terminated  # 累积而非覆盖
                env_steps[step_active_mask] += 1
        
        flow_call_count += 1
    
    # 构建样本
    timesteps = pipeline.scheduler.timesteps.unsqueeze(0).to(accelerator.device)
    all_samples = []
    
    for env_idx in range(num_envs):
        if len(env_obs_list[env_idx]) == 0:
            continue
        
        group_id = env_idx // samples_per_group
        
        # 倒序计算累积奖励（参考RLinf的calculate_scores实现）
        # 从后往前累积，遇到done则清零
        # 核心逻辑：if dones[step]: scores = 0; scores += rewards[step]
        rewards = env_step_rewards[env_idx]
        dones = env_step_dones[env_idx]
        
        cumulative_reward = 0.0
        for step in reversed(range(len(rewards))):
            # 如果当前步执行后done了，清零之前（时间上更早）累积的奖励
            # 但当前步的reward仍然要加上
            if dones[step]:
                cumulative_reward = 0.0
            cumulative_reward += rewards[step]
        
        # 获取该环境的episode_id
        env_episode_id = all_episode_ids_list[env_idx] if isinstance(all_episode_ids_list, list) else all_episode_ids[env_idx].item()
        
        all_samples.append({
            "obs_list": env_obs_list[env_idx],
            "clean_actions_list": env_clean_actions_list[env_idx],  # ✅ DiffusionNFT
            "timesteps": timesteps,  # scheduler的timesteps（需要保留）
            "reward": cumulative_reward,
            "num_steps": env_steps[env_idx].item(),
            "num_flow_calls": len(env_obs_list[env_idx]),
            "terminated": env_done[env_idx].item(),
            "group_id": group_id,
            "episode_id": env_episode_id,
            # ✅ 滑动窗口：添加逐步历史
            "obs_history": env_obs_history[env_idx],      # 每一步的obs
            "action_history": env_action_history[env_idx], # 每一步的action
            # StepNFT：每个 flow_call 一组 (x_t, v_t, x_next, step_idx, std_t)
            # 仅当 config.train.loss_type == "stepnft" 时为非空列表
            "nft_traces": env_nft_traces[env_idx] if use_stepnft else [],
        })
    
    return all_samples


def compute_episode_level_advantages(samples, num_train_timesteps, config):
    """
    基于完整轨迹的GRPO advantage计算
    
    支持两种 advantage 计算模式（由 config.stage.advantage_type 切换）：
    
    1. "group"（默认）：组内相对优势
       - 按GRPO组分组（组内相同初始状态）
       - 使用相对奖励的总和（参考RLinf）
       - Success Rate Filter（参考RLinf）：过滤全成功或全失败的组
       - 组内均值/方差归一化
    
    2. "single"：单条轨迹绝对优势
       - 不依赖组内对比，每条轨迹独立判定
       - 成功（reward >= config.stage.max_reward）→ +1
       - 失败 → -1
       - 不做 Success Rate Filter（绝对值 advantage 本身有学习信号）
    """
    advantage_type = getattr(config.stage, 'advantage_type', 'group')
    
    # ========== Single 模式：每条轨迹独立 ±1 ==========
    if advantage_type == 'single':
        max_reward = config.stage.max_reward
        advantage_scale = getattr(config.stage, 'advantage_scale', 1.0)
        
        success_count = 0
        for sample in samples:
            is_success = sample["reward"] >= max_reward
            sample["filtered"] = False
            sample["advantage"] = (1.0 if is_success else -1.0) * advantage_scale
            success_count += int(is_success)
        
        filter_stats = {
            "filtered_groups": 0,
            "filtered_samples": 0,
            "total_groups": config.episode.num_state_groups,
            "total_samples": len(samples),
        }
        return samples, filter_stats
    
    # ========== Group 模式（原有逻辑）==========
    if advantage_type != 'group':
        raise ValueError(
            f"Unknown advantage_type: {advantage_type!r}. "
            f"Expected 'group' or 'single'."
        )
    
    num_state_groups = config.episode.num_state_groups
    samples_per_group = config.episode.samples_per_group
    
    # 按组分配样本
    group_samples = [[] for _ in range(num_state_groups)]
    for sample in samples:
        group_id = sample["group_id"]
        group_samples[group_id].append(sample)
    
    # 统计过滤信息
    filtered_groups = 0
    filtered_samples = 0
    
    # 为每个样本计算advantage
    for group_id, group in enumerate(group_samples):
        if len(group) == 0:
            continue
        
        # ✅ Success Rate Filter（参考RLinf）
        # 检查组内是否全部成功或全部失败
        success_count = sum(1 for s in group if s["terminated"])
        all_success = (success_count == len(group))
        all_fail = (success_count == 0)
        
        # 如果全成功或全失败，过滤掉这个组
        if all_success or all_fail:
            filtered_groups += 1
            filtered_samples += len(group)
            # 标记这些样本为过滤状态
            for sample in group:
                sample["filtered"] = True
                sample["advantage"] = 0.0  # ✅ DiffusionNFT: 标量
            continue
        
        # ✅ 使用相对奖励的总和（已经消除累积效应）
        # 相对奖励 = 当前累积奖励 - 上一步累积奖励
        # 总和 = 最终累积奖励 - 初始累积奖励 = 任务完成度
        group_rewards = np.array([s["reward"] for s in group])
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-8
        
        # ✅ DiffusionNFT: 标量advantage（整条轨迹一个值）
        for i, sample in enumerate(group):
            sample["filtered"] = False
            total_reward = sample["reward"]
            advantage = (total_reward - group_mean) / group_std
            advantage *= config.stage.advantage_scale
            
            # ✅ 整条轨迹一个标量advantage（不再是列表）
            sample["advantage"] = advantage  # scalar float
    
    # ✅ 移除全局标准化（对齐ReinFlow）
    # ReinFlow只在组内进行标准化，不做全局标准化
    # 全局标准化会过度压缩advantages，削弱组间差异的学习信号
    
    # 返回样本和过滤统计信息
    filter_stats = {
        "filtered_groups": filtered_groups,
        "filtered_samples": filtered_samples,
        "total_groups": num_state_groups,
        "total_samples": len(samples),
    }
    
    return samples, filter_stats


# ❌ 原GRPO训练函数已删除，使用train_on_episode_samples_diffusionnft替代


def return_decay(step, decay_type):
    """
    Old model的EMA decay系数（DiffusionNFT Line 169-189）
    
    decay_type:
      0: 始终为0（不使用old model）
      1: 线性增长到0.5
      2: 前75步为0，之后增长到0.999（推荐）
    """
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.001, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    
    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def rollout_from_xt(
    xt,
    initial_velocity,
    current_timestep_idx,
    num_train_timesteps,
    action_unet,
    action_decoder,
    batch_obs,
    accelerator
):
    """
    从xt使用给定的初始velocity进行多步rollout到x1（t=1.0）
    
    核心思想：
    1. 第一步使用传入的initial_velocity（保留梯度用于优化）
    2. 后续步骤使用current模型预测velocity（detach以节省显存）
    3. 使用ode_step_with_logprob保持与预训练一致的ODE求解方式
    
    参数:
        xt: 当前加噪状态 [batch, action_horizon, action_dim]
        initial_velocity: 初始速度场预测（positive或negative） [batch, action_horizon, action_dim]
        current_timestep_idx: 当前timestep索引（1-9对应t=0.1-0.9）
        num_train_timesteps: 总timestep数（10）
        action_unet: 当前模型的UNet
        action_decoder: 当前模型的Decoder
        batch_obs: 观测条件 [batch, obs_dim]
        accelerator: Accelerator对象
        
    返回:
        x1_prediction: rollout到t=1.0的最终预测 [batch, action_horizon, action_dim]
    """
    from flow_grpo.diffusers_patch.sd3_sde_with_logprob import ode_step_with_logprob
    
    # ========== 第一步：使用initial_velocity（保留梯度） ==========
    current_sample = xt
    next_sample, _, _, _ = ode_step_with_logprob(
        model_output=initial_velocity.float(),
        current_timestep_idx=current_timestep_idx,
        sample=current_sample.float(),
        total_timesteps=num_train_timesteps,
    )
    
    current_sample = next_sample
    
    # ========== 后续步骤：逐步rollout到t=1.0 ==========
    for step_idx in range(current_timestep_idx + 1, num_train_timesteps):
        # 计算当前timestep（归一化到[0,1]，仅用于此处下一步 ODE 推进的位置约束；
        # 喂给 UNet 时按 pretrain 的 SinusoidalPosEmb 频率范围缩放）
        t_normalized = torch.full(
            (current_sample.shape[0],),
            _unet_timestep_input(step_idx / num_train_timesteps),
            device=current_sample.device,
            dtype=torch.float32
        )
        
        # 使用current模型预测velocity（detach以节省显存）
        # 注意：这里detach是为了只优化初始velocity，不优化后续步骤
        # with torch.no_grad():
        pred_raw = action_unet(
            sample=current_sample,
            timestep=t_normalized,
            global_cond=batch_obs
        )
        velocity = action_decoder(pred_raw)
        
        # ODE步骤
        next_sample, _, _, _ = ode_step_with_logprob(
            model_output=velocity.float(),
            current_timestep_idx=step_idx,
            sample=current_sample.float(),
            total_timesteps=num_train_timesteps,
        )
        
        current_sample = next_sample
    
    return current_sample


def _unet_timestep_input(t):
    """把 Flow Matching 的归一化时间 t ∈ [0, 1] 缩放到 pretrain stage 使用的
    SinusoidalPosEmb 输入尺度，使 RL/test 与 pretrain 共用同一套 UNet 时间编码。

    pretrain (pipeline_with_logprob)：
        t_normalized = (i / N) * 0.99 + 0.01
        t_scaled     = t_normalized * 99   →  在 i=0..N-1 时落在 [0.99, ≈89.2]

    对任意 t ∈ [0, 1]，该函数返回与 pretrain 同源的 timestep：
        t_unet = (t * 0.99 + 0.01) * 99    = t * 98.01 + 0.99

    所有 RL 训练 / 推理 / 测试都必须使用本函数把 t 喂给 UNet，否则会与 pretrain
    的 SinusoidalPosEmb 频率覆盖范围不一致，导致 UNet 输出毫无意义的 velocity。
    """
    return t * 98.01 + 0.99


def denoise_with_sde_snap(
    obs_tensor,
    action_unet,
    action_decoder,
    num_steps,
    noise_level,
    action_horizon,
    action_dim,
    generators,
    device,
    obs_normalizer=None,
):
    """
    StepNFT 风格的 SDE 多步去噪采样，每个样本独立随机选一步保存 (x_t, v_t, x_next, step_idx, std_t)。
    
    设计要点：
    1. 与训练时 timestep 编码完全一致：t = step_idx / num_steps（[0, (N-1)/N]），不使用 pipeline 的 t_scaled 缩放
    2. 时间方向：t=0 是噪声，t=1 是数据（与 train_on_episode_samples_diffusionnft 一致）
    3. SDE 单步：x_next = x_t + v_t·delta + std_t·sqrt(delta)·eps，eps ~ N(0, I)
       std 公式参考 pi-StepNFT，但按当前方向重写：sigma_base(t) = sqrt((1-t)/t_safe)
    4. v_t 在 action_decoder 后空间（与 ODE pipeline 保持一致）
    
    参数:
        obs_tensor: [batch, obs_dim] 观测条件
        action_unet: 当前模型的 UNet
        action_decoder: 当前模型的 Decoder
        num_steps: 去噪步数（与 ODE 推理一致，如 10）
        noise_level: SDE 噪声系数（>0），控制随机性强度
        action_horizon: 动作序列长度
        action_dim: 单步动作维度
        generators: 长度为 batch 的 torch.Generator 列表（与 pipeline 行为一致）
        device: 计算设备
        obs_normalizer: 可选的 obs 归一化函数（默认 None）
    
    返回: (actions, nft_traces)
        actions: [batch, action_horizon, action_dim] 最终动作
        nft_traces: dict
            - "nft_xt": [batch, action_horizon, action_dim]，被采样步的 x_t
            - "nft_v":  [batch, action_horizon, action_dim]，被采样步的 v_t（v_old）
            - "nft_xnext": [batch, action_horizon, action_dim]，SDE 推进后的 x_next
            - "nft_step_index": [batch] long，被采样步的索引（[0, num_steps-1]）
            - "nft_std": [batch] float，被采样步的 std_t（标量按 sample 复制）
    """
    batch_size = obs_tensor.shape[0]
    
    # 1. 生成初始噪声（与 pipeline 行为一致：每个样本独立 generator）
    if isinstance(generators, list):
        if len(generators) != batch_size:
            raise ValueError(
                f"generators 列表长度 ({len(generators)}) 必须等于 batch_size ({batch_size})"
            )
        latents_list = []
        for i in range(batch_size):
            latent_i = torch.randn(
                1, action_horizon, action_dim,
                generator=generators[i],
                device=device,
                dtype=torch.float32,
            )
            latents_list.append(latent_i)
        x_t = torch.cat(latents_list, dim=0)
    else:
        x_t = torch.randn(
            batch_size, action_horizon, action_dim,
            generator=generators,
            device=device,
            dtype=torch.float32,
        )
    
    # 2. obs 编码与归一化（与 pipeline 对齐）
    obs_embeds = obs_tensor
    if obs_normalizer is not None:
        obs_embeds = obs_normalizer(obs_embeds)
    
    # 3. 每个样本独立随机选一步保存 nft 数据
    snap_idx = torch.randint(
        0, num_steps, (batch_size,), device=device, dtype=torch.long
    )
    
    snap_xt = torch.empty_like(x_t)
    snap_v = torch.empty_like(x_t)
    snap_xnext = torch.empty_like(x_t)
    snap_std = torch.zeros(batch_size, device=device, dtype=torch.float32)
    
    # 4. SDE 多步循环（不需要梯度，仅用于数据收集）
    with torch.no_grad():
        for i in range(num_steps):
            # Flow Matching 归一化时间 t ∈ [0, (N-1)/N]，用于 SDE/ODE 推进的数学
            t_value = i / num_steps
            # UNet 输入需要按 pretrain stage 的 SinusoidalPosEmb 频率范围缩放，
            # 否则与 pipeline_with_logprob 训练出的速度场频率不一致 → 速度场失效
            t_input = torch.full(
                (batch_size,), _unet_timestep_input(t_value),
                device=device, dtype=torch.float32,
            )
            
            # UNet + Decoder forward（v_t 在 decoder 后空间）
            pred_raw = action_unet(
                sample=x_t,
                timestep=t_input,
                global_cond=obs_embeds,
            )
            v_t = action_decoder(pred_raw)
            
            # SDE 系数推导（当前方向 t=0 noise → t=1 data）
            # delta = 1/N（向 data 方向推进）
            # sigma_base(t) = sqrt((1-t)/t_safe)，t_safe=max(t, 1/N) 避开 t=0 奇异
            # std_t = noise_level * sigma_base * sqrt(delta)
            delta = 1.0 / num_steps
            t_safe = max(t_value, 1.0 / num_steps)
            sigma_base = math.sqrt((1.0 - t_safe) / t_safe)
            std_t = noise_level * sigma_base * math.sqrt(delta)
            
            # 推进
            x_next_mean = x_t + v_t * delta
            
            # 注入 SDE 噪声（用每个样本独立的 generator 保持采样可复现）
            if isinstance(generators, list):
                noise_list = []
                for k in range(batch_size):
                    n_k = torch.randn(
                        1, action_horizon, action_dim,
                        generator=generators[k],
                        device=device,
                        dtype=torch.float32,
                    )
                    noise_list.append(n_k)
                eps = torch.cat(noise_list, dim=0)
            else:
                eps = torch.randn(
                    batch_size, action_horizon, action_dim,
                    generator=generators,
                    device=device,
                    dtype=torch.float32,
                )
            
            x_next = x_next_mean + std_t * eps
            
            # 在被选中的样本上保存 (x_t, v_t, x_next, step_idx, std_t)
            mask = snap_idx == i  # [batch] bool
            if mask.any():
                snap_xt[mask] = x_t.detach()[mask]
                snap_v[mask] = v_t.detach()[mask]
                snap_xnext[mask] = x_next.detach()[mask]
                snap_std[mask] = std_t  # 标量广播
            
            x_t = x_next
    
    # 5. 输出动作裁剪（对齐 pipeline_with_logprob 的 "clipped" 模式）
    actions = torch.clamp(x_t, -1.0, 1.0)
    
    nft_traces = {
        "nft_xt": snap_xt,
        "nft_v": snap_v,
        "nft_xnext": snap_xnext,
        "nft_step_index": snap_idx,
        "nft_std": snap_std,
    }
    
    return actions, nft_traces


def train_on_episode_samples_diffusionnft(
    samples, 
    action_unet, action_decoder,
    action_unet_old, action_decoder_old,
    action_unet_ref, action_decoder_ref,
    optimizer, accelerator, config, ema, trainable_parameters,
    global_step, epoch
):
    """
    DiffusionNFT风格训练函数（完全替换原GRPO训练）
    
    核心特性：
    1. 遍历所有timesteps（完全对齐DiffusionNFT Line 854）
    2. 三模型velocity prediction（current/old/ref）
    3. Clip+归一化+clamp advantage（DiffusionNFT Line 910-929）
    4. 标准 MSE loss（移除 Adaptive Weighting 以适应机器人控制）
    5. Policy loss + KL loss（移除 adv_clip_max 缩放）
    """
    action_unet.train()
    action_decoder.train()
    action_unet_old.eval() # 这里是 eval 因为函数外后面会更新
    action_decoder_old.eval()
    action_unet_ref.eval()
    action_decoder_ref.eval()
    
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction) # 10
    
    # 阶段1：展平所有flows为训练样本
    batch_obs = []
    batch_clean_actions = []
    batch_advantages = []
    # ✅ 方案1B：不再需要 batch_timesteps，训练时随机采样 timestep
    
    # ✅ 滑动窗口参数
    action_horizon = config.action_horizon  # 16
    sliding_window_stride = getattr(config.train, 'sliding_window_stride', 1)  # 默认步长1
    
    for sample in samples:
        if sample.get("filtered", False):
            continue
        
        trajectory_advantage = sample["advantage"]  # scalar
        
        # ✅ 滑动窗口：使用obs_history和action_history
        obs_history = sample.get("obs_history", [])
        action_history = sample.get("action_history", [])
        
        # 如果没有history数据，回退到旧方法（向后兼容）
        if len(obs_history) == 0 or len(action_history) == 0:
            num_flow_calls = sample["num_flow_calls"]
            for flow_idx in range(num_flow_calls):
                batch_obs.append(sample["obs_list"][flow_idx])
                batch_clean_actions.append(sample["clean_actions_list"][flow_idx])
                batch_advantages.append(trajectory_advantage)
            continue
        
        # ✅ 滑动窗口采样
        total_steps = len(obs_history)
        
        # 遍历所有可能的窗口起点
        for start_idx in range(0, total_steps - action_horizon + 1, sliding_window_stride):
            end_idx = start_idx + action_horizon
            
            # 提取窗口内的obs和actions
            window_obs = obs_history[start_idx]  # 窗口起点的obs
            window_actions = torch.cat(action_history[start_idx:end_idx], dim=0)  # [action_horizon, action_dim]
            
            # 添加到批次
            batch_obs.append(window_obs)
            batch_clean_actions.append(window_actions)
            batch_advantages.append(trajectory_advantage)
    
    if len(batch_obs) == 0:
        logger.warning(f"Epoch {epoch}: 没有有效样本，跳过训练")
        return global_step, 0.0  # 返回loss=0.0表示未训练
    
    # 阶段2：转换为tensors
    obs_cond = torch.cat(batch_obs, dim=0)  # [N_flows, obs_dim]
    x0 = torch.stack(batch_clean_actions)  # [N_flows, action_horizon, action_dim]
    advantages = torch.tensor(batch_advantages, device=accelerator.device, dtype=torch.float32)
    
    total_batch_size = obs_cond.shape[0]
    logger.info(f"Epoch {epoch}: 训练 {total_batch_size} 个flows")
    
    # 阶段3：打乱数据
    perm = torch.randperm(total_batch_size, device=accelerator.device)
    obs_cond = obs_cond[perm]
    x0 = x0[perm]
    advantages = advantages[perm]
    
    # 阶段4：分批训练
    num_batches = max(1, total_batch_size // config.train.batch_size)
    
    # ✅ 多步Rollout训练：遍历所有batch，每个batch遍历所有timesteps（t=0.1到0.9）
    # 策略：在每个timestep处加噪，然后通过多步去噪rollout到最终数据
    # 优势：充分利用预训练模型的多步去噪能力，避免过拟合单个timestep
    # 循环顺序：与DiffusionNFT对齐（batch在外，timestep在内）
    effective_grad_accum_steps = config.train.gradient_accumulation_steps * (num_train_timesteps - 1)
    current_accumulated_steps = 0
    info = defaultdict(list)
    
    # ✅ 用于记录整个epoch的平均loss
    epoch_total_losses = []
    
    # ✅ 外层循环：遍历batch（与DiffusionNFT对齐）
    for batch_idx in tqdm(
        range(num_batches),
        desc=f"Epoch {epoch}: training",
        position=0,
        disable=not accelerator.is_local_main_process
    ):
        start_idx = batch_idx * config.train.batch_size
        end_idx = min(start_idx + config.train.batch_size, total_batch_size)
        
        batch_obs = obs_cond[start_idx:end_idx]
        batch_x0 = x0[start_idx:end_idx]
        batch_advantages = advantages[start_idx:end_idx]
        
        # ✅ 内层循环：遍历所有timesteps（从j_idx=1开始，对应t=0.1到0.9）
        for j_idx in tqdm(
            range(1, num_train_timesteps),
            desc="Timestep",
            position=1,
            leave=False,
            disable=not accelerator.is_local_main_process
        ):
            t_value = j_idx / num_train_timesteps  # 计算实际的t值
            
            # ========== 核心：多步Rollout训练 ==========
            with accelerator.autocast():
                # 1. ✅ 在当前timestep t处加噪
                # 理念：从xt开始rollout到x1，利用预训练模型的多步去噪能力
                batch_size_actual = batch_x0.shape[0]
                t = torch.full(
                    (batch_size_actual,), 
                    t_value,
                    device=accelerator.device,
                    dtype=torch.float32
                )  # [batch], 所有样本使用相同的 timestep
                t_expanded = t.view(-1, 1, 1)  # [batch, 1, 1]
                
                # 2. 加噪：xt = (1-t)*x0 + t*x1，其中 x0=noise, x1=data
                # ✅ 反向Flow Matching: 从噪声(t=0)流向数据(t=1)
                noise = torch.randn_like(batch_x0.float())  # x0 = noise
                batch_x1 = batch_x0  # x1 = data (clean actions)
                xt = (1 - t_expanded) * noise + t_expanded * batch_x1
                
                # ========== DEBUG: 第一个epoch的第一个batch，第一个timestep ==========
                if epoch == 0 and batch_idx == 0 and j_idx == 1:
                    logger.info(f"\n{'='*70}")
                    logger.info(f"【DEBUG】Epoch 0, Batch 0, Timestep {j_idx} (t={t_value:.1f}): 多步Rollout训练")
                    logger.info(f"{'='*70}")
                    logger.info(f"batch_x0 (clean actions, 即x1/data) shape: {batch_x0.shape}")
                    logger.info(f"batch_x1 (数据) 统计: min={batch_x1.min().item():.6f}, max={batch_x1.max().item():.6f}, mean={batch_x1.mean().item():.6f}, std={batch_x1.std().item():.6f}")
                    logger.info(f"noise (x0) 统计: min={noise.min().item():.6f}, max={noise.max().item():.6f}, mean={noise.mean().item():.6f}, std={noise.std().item():.6f}")
                    logger.info(f"t (当前timestep): {t[0].item():.6f} (timestep_idx={j_idx})")
                    logger.info(f"从t={t_value:.1f}开始，将rollout {num_train_timesteps - j_idx} 步到t=1.0")
                    logger.info(f"batch_advantages 统计: min={batch_advantages.min().item():.6f}, max={batch_advantages.max().item():.6f}, mean={batch_advantages.mean().item():.6f}")
                    
                    # ✅ 检查三个模型的参数是否相同
                    logger.info(f"\n【模型参数对比】")
                    current_unet_params = list(unwrap_model(action_unet, accelerator).parameters())
                    old_unet_params = list(action_unet_old.parameters())
                    ref_unet_params = list(action_unet_ref.parameters())
                    
                    # 检查第一层参数
                    param_diff_old = (current_unet_params[0] - old_unet_params[0]).abs().max().item()
                    param_diff_ref = (current_unet_params[0] - ref_unet_params[0]).abs().max().item()
                    logger.info(f"UNet第一层参数差异: current-old={param_diff_old:.10f}, current-ref={param_diff_ref:.10f}")
                    
                    # 检查模型dtype
                    logger.info(f"current_unet第一层参数dtype: {current_unet_params[0].dtype}")
                    logger.info(f"old_unet第一层参数dtype: {old_unet_params[0].dtype}")
                    
                    logger.info(f"\nxt (加噪后) 统计: min={xt.min().item():.6f}, max={xt.max().item():.6f}, mean={xt.mean().item():.6f}, std={xt.std().item():.6f}")
                
                # 3. 三模型预测 velocity
                # 数学层面的 t（[0, (N-1)/N]）继续供 _flow_mean / ode_step 使用，
                # 而喂给 UNet 的 timestep 需缩放到 pretrain 的 SinusoidalPosEmb 频率范围
                # （(t*0.99+0.01)*99），保证与 pretrain 学到的速度场用同一时间编码。
                t_unet = _unet_timestep_input(t)

                # Old model（no grad，DiffusionNFT Line 874-882）
                with torch.no_grad():
                    old_pred_raw = action_unet_old(
                        sample=xt, 
                        timestep=t_unet,
                        global_cond=batch_obs
                    )
                    old_prediction = action_decoder_old(old_pred_raw)
                
                # Current model（with grad，DiffusionNFT Line 886-892）
                current_pred_raw = action_unet(
                    sample=xt,
                    timestep=t_unet,
                    global_cond=batch_obs
                )
                current_prediction = action_decoder(current_pred_raw)
                
                # Reference model（no grad，DiffusionNFT Line 894-907）
                with torch.no_grad():
                    ref_pred_raw = action_unet_ref(
                        sample=xt,
                        timestep=t_unet,
                        global_cond=batch_obs
                    )
                    ref_prediction = action_decoder_ref(ref_pred_raw)
                
                # # ========== DEBUG: 三模型预测值 ==========
                # if epoch == 0 and batch_idx == 0:
                #     logger.info(f"\n【三模型预测值】")
                #     logger.info(f"old_pred_raw (UNet输出) 统计: min={old_pred_raw.min().item():.6f}, max={old_pred_raw.max().item():.6f}, mean={old_pred_raw.mean().item():.6f}, std={old_pred_raw.std().item():.6f}")
                #     logger.info(f"current_pred_raw (UNet输出) 统计: min={current_pred_raw.min().item():.6f}, max={current_pred_raw.max().item():.6f}, mean={current_pred_raw.mean().item():.6f}, std={current_pred_raw.std().item():.6f}")
                #     logger.info(f"ref_pred_raw (UNet输出) 统计: min={ref_pred_raw.min().item():.6f}, max={ref_pred_raw.max().item():.6f}, mean={ref_pred_raw.mean().item():.6f}, std={ref_pred_raw.std().item():.6f}")
                #     logger.info(f"\nDecoder输出后:")
                #     logger.info(f"old_prediction 统计: min={old_prediction.min().item():.6f}, max={old_prediction.max().item():.6f}, mean={old_prediction.mean().item():.6f}, std={old_prediction.std().item():.6f}")
                #     logger.info(f"current_prediction 统计: min={current_prediction.min().item():.6f}, max={current_prediction.max().item():.6f}, mean={current_prediction.mean().item():.6f}, std={current_prediction.std().item():.6f}")
                #     logger.info(f"ref_prediction 统计: min={ref_prediction.min().item():.6f}, max={ref_prediction.max().item():.6f}, mean={ref_prediction.mean().item():.6f}, std={ref_prediction.std().item():.6f}")
                #     logger.info(f"\npred_raw dtype: current={current_pred_raw.dtype}, old={old_pred_raw.dtype}, ref={ref_pred_raw.dtype}")
                #     logger.info(f"prediction dtype: current={current_prediction.dtype}, old={old_prediction.dtype}, ref={ref_prediction.dtype}")
                
                # ========== 4. Advantage处理（DiffusionNFT Line 910-929） ==========
                # Step 1: Clip到[-adv_clip_max, adv_clip_max]
                advantages_clip = torch.clamp(
                    batch_advantages,
                    -config.train.adv_clip_max,
                    config.train.adv_clip_max,
                )
            
            # Step 2: 可选adv_mode
            if hasattr(config.train, "adv_mode") and config.train.adv_mode:
                if config.train.adv_mode == "positive_only":
                    advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                elif config.train.adv_mode == "negative_only":
                    advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                elif config.train.adv_mode == "one_only":
                    advantages_clip = torch.where(
                        advantages_clip > 0, 
                        torch.ones_like(advantages_clip), 
                        torch.zeros_like(advantages_clip)
                    )
                elif config.train.adv_mode == "binary":
                    advantages_clip = torch.sign(advantages_clip)
            
            # Step 3: 线性归一化到[0, 1]
            normalized_advantages = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
            
            # Step 4: Clamp确保在[0, 1]
            r = torch.clamp(normalized_advantages, 0, 1)
            
            # ========== 5. Positive/Negative prediction（DiffusionNFT Line 934-937） ==========
            beta = config.train.beta
            positive_prediction = beta * current_prediction + (1 - beta) * old_prediction.detach()
            implicit_negative_prediction = (1.0 + beta) * old_prediction.detach() - beta * current_prediction
            
            # ========== DEBUG: Advantage处理和r值 ==========
            if epoch == 0 and batch_idx == 0 and j_idx == 1:
                logger.info(f"\n【Advantage处理】")
                logger.info(f"advantages_clip 统计: min={advantages_clip.min().item():.6f}, max={advantages_clip.max().item():.6f}, mean={advantages_clip.mean().item():.6f}")
                logger.info(f"normalized_advantages 统计: min={normalized_advantages.min().item():.6f}, max={normalized_advantages.max().item():.6f}, mean={normalized_advantages.mean().item():.6f}")
                logger.info(f"r (权重) 统计: min={r.min().item():.6f}, max={r.max().item():.6f}, mean={r.mean().item():.6f}")
                logger.info(f"beta = {beta}")
                logger.info(f"\npositive_prediction 统计: min={positive_prediction.min().item():.6f}, max={positive_prediction.max().item():.6f}, mean={positive_prediction.mean().item():.6f}")
                logger.info(f"implicit_negative_prediction 统计: min={implicit_negative_prediction.min().item():.6f}, max={implicit_negative_prediction.max().item():.6f}, mean={implicit_negative_prediction.mean().item():.6f}")
                
            # ========== 6. x1估计：多步Rollout（从xt到x1） ==========
            # ✅ 使用rollout函数，从xt开始沿着positive和negative两条路径去噪到x1
            # 正向路径：使用positive_prediction作为初始velocity
            x1_prediction_pos = rollout_from_xt(
                xt=xt,
                initial_velocity=positive_prediction,
                current_timestep_idx=j_idx,
                num_train_timesteps=num_train_timesteps,
                action_unet=action_unet,
                action_decoder=action_decoder,
                batch_obs=batch_obs,
                accelerator=accelerator
            )
            
            # 负向路径：使用implicit_negative_prediction作为初始velocity
            x1_prediction_neg = rollout_from_xt(
                xt=xt,
                initial_velocity=implicit_negative_prediction,
                current_timestep_idx=j_idx,
                num_train_timesteps=num_train_timesteps,
                action_unet=action_unet,
                action_decoder=action_decoder,
                batch_obs=batch_obs,
                accelerator=accelerator
            )
            
            # ========== DEBUG: x1预测值（Rollout结果） ==========
            if epoch == 0 and batch_idx == 0 and j_idx == 1:
                logger.info(f"\n【x1预测（多步Rollout结果）】")
                logger.info(f"从t={t_value:.1f}经过{num_train_timesteps - j_idx}步rollout到t=1.0")
                logger.info(f"x1_prediction_pos 统计: min={x1_prediction_pos.min().item():.6f}, max={x1_prediction_pos.max().item():.6f}, mean={x1_prediction_pos.mean().item():.6f}")
                logger.info(f"x1_prediction_neg 统计: min={x1_prediction_neg.min().item():.6f}, max={x1_prediction_neg.max().item():.6f}, mean={x1_prediction_neg.mean().item():.6f}")
                logger.info(f"x1预测误差 (pos): mean={(x1_prediction_pos - batch_x1).abs().mean().item():.6f}, max={(x1_prediction_pos - batch_x1).abs().max().item():.6f}")
                logger.info(f"x1预测误差 (neg): mean={(x1_prediction_neg - batch_x1).abs().mean().item():.6f}, max={(x1_prediction_neg - batch_x1).abs().max().item():.6f}")
            
            # ========== 7. Policy loss（标准 MSE，移除 Adaptive Weighting） ==========
            # ✅ 使用标准 MSE loss，适合机器人控制任务
            # 移除 Adaptive Weighting 以避免逆向激励和训练不稳定
            positive_loss = ((x1_prediction_pos - batch_x1) ** 2).mean(dim=(1, 2))
            negative_loss = ((x1_prediction_neg - batch_x1) ** 2).mean(dim=(1, 2))
            
            # ✅ 使用 advantage 加权，但移除 adv_clip_max 缩放
            # adv_clip_max 用于 clip advantage，不应该作为 loss 的放大系数
            ori_policy_loss = r * positive_loss / beta + (1.0 - r) * negative_loss / beta
            policy_loss = ori_policy_loss.mean()
            
            # ========== DEBUG: Loss计算 ==========
            if epoch == 0 and batch_idx == 0 and j_idx == 1:
                logger.info(f"\n【Loss计算】")
                logger.info(f"positive_loss (标准MSE) 统计: min={positive_loss.min().item():.6f}, max={positive_loss.max().item():.6f}, mean={positive_loss.mean().item():.6f}")
                logger.info(f"negative_loss (标准MSE) 统计: min={negative_loss.min().item():.6f}, max={negative_loss.max().item():.6f}, mean={negative_loss.mean().item():.6f}")
                logger.info(f"ori_policy_loss (advantage加权) 统计: min={ori_policy_loss.min().item():.6f}, max={ori_policy_loss.max().item():.6f}, mean={ori_policy_loss.mean().item():.6f}")
                logger.info(f"policy_loss (最终): {policy_loss.item():.6f}")
            
            # ========== 8. KL loss（DiffusionNFT Line 966-970） ==========
            kl_div_loss = ((current_prediction - ref_prediction) ** 2).mean(dim=(1, 2))
            kl_loss = config.train.kl_weight * kl_div_loss.mean()
            
            # ========== DEBUG: KL loss ==========
            if epoch == 0 and batch_idx == 0 and j_idx == 1:
                logger.info(f"\n【KL Loss】")
                logger.info(f"kl_div_loss 统计: min={kl_div_loss.min().item():.6f}, max={kl_div_loss.max().item():.6f}, mean={kl_div_loss.mean().item():.6f}")
                logger.info(f"kl_loss (乘以kl_weight={config.train.kl_weight}后): {kl_loss.item():.6f}")
            
            # ========== 9. Total loss ==========
            loss = policy_loss + kl_loss
            
            # ========== DEBUG: Total loss ==========
            if epoch == 0 and batch_idx == 0 and j_idx == 1:
                logger.info(f"\n【Total Loss】")
                logger.info(f"total_loss = policy_loss + kl_loss = {policy_loss.item():.6f} + {kl_loss.item():.6f} = {loss.item():.6f}")
                logger.info(f"effective_grad_accum_steps = {effective_grad_accum_steps}")
                logger.info(f"scaled_loss = loss / {effective_grad_accum_steps} = {(loss / effective_grad_accum_steps).item():.6f}")
                logger.info(f"{'='*70}\n")
            
            # ========== 10. 缩放并累积梯度（DiffusionNFT Line 983） ==========
            scaled_loss = loss / effective_grad_accum_steps
            accelerator.backward(scaled_loss)
            current_accumulated_steps += 1
            
            # 记录
            info["policy_loss"].append(policy_loss.detach())
            info["kl_div_loss"].append(kl_loss.detach())
            info["total_loss"].append(loss.detach())
            
            # ===== 监控指标3: Loss组成和模型预测 =====
            info["positive_loss"].append(positive_loss.mean().detach())
            info["negative_loss"].append(negative_loss.mean().detach())
            info["x1_error_pos"].append((x1_prediction_pos - batch_x1).abs().mean().detach())
            info["x1_error_neg"].append((x1_prediction_neg - batch_x1).abs().mean().detach())
            info["prediction_std"].append(current_prediction.std().detach())
            info["kl_div_per_sample"].append(kl_div_loss.mean().detach())
            
            # ✅ 收集loss用于计算epoch平均值
            epoch_total_losses.append(loss.detach().item())
            
            # ========== 12. 梯度累积完成后更新（多步Rollout训练） ==========
            if current_accumulated_steps % effective_grad_accum_steps == 0:
                # 梯度范数监控：在裁剪前计算
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, 
                    float('inf')  # 不裁剪，只计算范数
                ).item()
                
                # 梯度裁剪
                grad_norm_after = accelerator.clip_grad_norm_(
                    trainable_parameters, 
                    config.train.max_grad_norm
                ).item()
                
                # ===== 监控指标4: 梯度裁剪率 =====
                grad_clipped = 1.0 if grad_norm_before > config.train.max_grad_norm else 0.0
                
                # ===== 监控指标5: 参数变化前的快照 =====
                # 保存参数用于计算更新幅度
                with torch.no_grad():
                    param_before = torch.cat([p.data.flatten() for p in trainable_parameters])
                
                optimizer.step()
                optimizer.zero_grad()
                
                # ===== 监控指标6: 参数更新幅度 =====
                with torch.no_grad():
                    param_after = torch.cat([p.data.flatten() for p in trainable_parameters])
                    param_update_norm = (param_after - param_before).norm().item()
                    param_norm = param_after.norm().item()
                    param_update_ratio = param_update_norm / (param_norm + 1e-8)
                
                # Logging
                info_aggregated = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                
                if global_step % 10 == 0 and accelerator.is_local_main_process:
                    logger.info(
                        f"  Step {global_step}: "
                        f"loss={info_aggregated['total_loss']:.4f}, "
                        f"grad_norm={grad_norm_before:.4f}"
                    )
                
                # 记录到 wandb（使用 global_step 作为横轴）
                if accelerator.is_main_process:
                    import wandb
                    if wandb.run is not None:
                        # 基础loss指标
                        log_dict = {
                            "train/policy_loss": info_aggregated['policy_loss'].item(),
                            "train/kl_div_loss": info_aggregated['kl_div_loss'].item(),
                            "train/total_loss": info_aggregated['total_loss'].item(),
                            "train/grad_norm_before_clip": grad_norm_before,
                            "train/grad_norm_after_clip": grad_norm_after,
                            "train/epoch": epoch,
                        }
                        
                        # Loss组成指标
                        if "positive_loss" in info_aggregated:
                            log_dict.update({
                                "loss/positive_loss": info_aggregated['positive_loss'].item(),
                                "loss/negative_loss": info_aggregated['negative_loss'].item(),
                                "loss/pos_neg_ratio": (info_aggregated['positive_loss'] / (info_aggregated['negative_loss'] + 1e-8)).item(),
                                "loss/policy_kl_ratio": (info_aggregated['policy_loss'] / (info_aggregated['kl_div_loss'] + 1e-8)).item(),
                            })
                        
                        # 模型预测指标
                        if "x1_error_pos" in info_aggregated:
                            log_dict.update({
                                "prediction/x1_error_pos": info_aggregated['x1_error_pos'].item(),
                                "prediction/x1_error_neg": info_aggregated['x1_error_neg'].item(),
                                "prediction/output_std": info_aggregated['prediction_std'].item(),
                                "prediction/kl_div_per_sample": info_aggregated['kl_div_per_sample'].item(),
                            })
                        
                        # 优化过程指标
                        log_dict.update({
                            "optim/grad_clipped": grad_clipped,
                            "optim/param_update_norm": param_update_norm,
                            "optim/param_norm": param_norm,
                            "optim/param_update_ratio": param_update_ratio,
                            "optim/learning_rate": optimizer.param_groups[0]['lr'],
                        })
                        
                        wandb.log(log_dict, step=global_step)
                
                global_step += 1
                current_accumulated_steps = 0
                info = defaultdict(list)
                
                # EMA更新
                if config.train.ema:
                    ema.step(trainable_parameters, global_step)
                
                # ========== Old model 更新（每20个global_step更新一次） ==========
                if global_step % 20 == 0:
                    with torch.no_grad():
                        decay = return_decay(global_step, config.train.decay_type)
                        
                        # ===== 监控指标7: 模型间差异（更新前） =====
                        # 计算current vs old的差异
                        current_old_distance = 0.0
                        current_ref_distance = 0.0
                        old_ref_distance = 0.0
                        
                        for cur_p, old_p, ref_p in zip(
                            unwrap_model(action_unet, accelerator).parameters(),
                            action_unet_old.parameters(),
                            action_unet_ref.parameters()
                        ):
                            current_old_distance += (cur_p - old_p).pow(2).sum().item()
                            current_ref_distance += (cur_p - ref_p).pow(2).sum().item()
                            old_ref_distance += (old_p - ref_p).pow(2).sum().item()
                        
                        # 加上decoder的差异
                        for cur_p, old_p, ref_p in zip(
                            unwrap_model(action_decoder, accelerator).parameters(),
                            action_decoder_old.parameters(),
                            action_decoder_ref.parameters()
                        ):
                            current_old_distance += (cur_p - old_p).pow(2).sum().item()
                            current_ref_distance += (cur_p - ref_p).pow(2).sum().item()
                            old_ref_distance += (old_p - ref_p).pow(2).sum().item()
                        
                        # L2距离（开方）
                        current_old_distance = np.sqrt(current_old_distance)
                        current_ref_distance = np.sqrt(current_ref_distance)
                        old_ref_distance = np.sqrt(old_ref_distance)
                        
                        # 更新UNet
                        for src_param, tgt_param in zip(
                            unwrap_model(action_unet, accelerator).parameters(),
                            action_unet_old.parameters()
                        ):
                            tgt_param.data.copy_(
                                tgt_param.data * decay + src_param.data * (1.0 - decay)
                            )
                        
                        # 更新Decoder
                        for src_param, tgt_param in zip(
                            unwrap_model(action_decoder, accelerator).parameters(),
                            action_decoder_old.parameters()
                        ):
                            tgt_param.data.copy_(
                                tgt_param.data * decay + src_param.data * (1.0 - decay)
                            )
                        
                        if accelerator.is_local_main_process:
                            logger.info(f"  [Old model 更新] Step {global_step}: decay={decay:.4f}")
                        
                        # ===== 记录模型差异到wandb =====
                        if accelerator.is_main_process:
                            import wandb
                            if wandb.run is not None:
                                wandb.log({
                                    "model/current_old_distance": current_old_distance,
                                    "model/current_ref_distance": current_ref_distance,
                                    "model/old_ref_distance": old_ref_distance,
                                    "model/old_decay": decay,
                                }, step=global_step)
    
    # 计算整个epoch的平均loss
    avg_epoch_loss = np.mean(epoch_total_losses) if len(epoch_total_losses) > 0 else 0.0
    
    return global_step, avg_epoch_loss


def train_on_episode_samples_stepnft(
    samples,
    action_unet, action_decoder,
    action_unet_old, action_decoder_old,
    action_unet_ref, action_decoder_ref,
    optimizer, accelerator, config, ema, trainable_parameters,
    global_step, epoch
):
    """
    StepNFT 风格训练函数：单步 forward + DPO/weighted NLL loss
    
    数据来源：sample_complete_trajectories 在 SDE 采样时保存的 nft_traces：
        每个 flow_call 一组 (x_t, v_t, x_next, step_idx, std_t)
    
    关键差异（vs train_on_episode_samples_diffusionnft）：
    1. 仅在采样时随机选中的 step 上做一次 UNet forward（无多步 rollout 反传）
    2. v_old 直接用采样时保存的 nft_v（不再调 action_unet_old）
    3. loss 形式：
       - "weighted": L = r·E_pos + (1-r)·E_neg
       - "dpo":     L = softplus((β/2)·y·(E_pos - E_neg))
    4. 时间方向：当前代码 t∈[0,1]，t=0 是 noise，t=1 是 data；_flow_mean 系数按此推导
    
    保留与 diffusionnft 一致的部分：
    - EMA、old/ref 模型同步、grad clip、wandb log、optimizer.step 节奏
    """
    # ========== 入口校验 ==========
    noise_level = float(config.train.stepnft_noise_level)
    if noise_level <= 0:
        raise ValueError(
            f"StepNFT 在 ODE 采样下数学退化（advantage 信号丢失，梯度恒为 0 或与 r 无关）。"
            f"必须 config.train.stepnft_noise_level > 0，当前为 {noise_level}。"
        )
    
    action_unet.train()
    action_decoder.train()
    action_unet_old.eval()
    action_decoder_old.eval()
    action_unet_ref.eval()
    action_decoder_ref.eval()
    
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    # StepNFT 超参数
    beta = float(config.train.stepnft_beta)
    max_drift = float(config.train.stepnft_max_drift)
    loss_form = str(config.train.stepnft_loss_form)
    dpo_beta = float(config.train.stepnft_dpo_beta)
    kl_beta = float(config.train.stepnft_kl_beta)
    std_epsilon = float(config.train.stepnft_std_epsilon)
    adv_clip_max = float(config.train.adv_clip_max)
    
    # ========== 阶段1：展平所有 flow_call 的 nft_data 为训练批 ==========
    batch_obs = []
    batch_xt = []
    batch_v_old = []
    batch_xnext = []
    batch_step_index = []
    batch_std = []
    batch_advantages = []
    
    for sample in samples:
        # 跳过被过滤的样本（与 diffusionnft 对齐）
        if sample.get("filtered", False):
            continue
        
        nft_traces = sample.get("nft_traces", [])
        if len(nft_traces) == 0:
            # 没有 nft 数据（可能是 ODE 采样时收集的 sample），跳过
            continue
        
        trajectory_advantage = sample["advantage"]
        obs_list = sample["obs_list"]
        num_flow_calls = len(nft_traces)
        
        for flow_idx in range(num_flow_calls):
            trace = nft_traces[flow_idx]
            batch_obs.append(obs_list[flow_idx])
            batch_xt.append(trace["nft_xt"])
            batch_v_old.append(trace["nft_v"])
            batch_xnext.append(trace["nft_xnext"])
            batch_step_index.append(int(trace["nft_step_index"]))
            batch_std.append(float(trace["nft_std"]))
            batch_advantages.append(trajectory_advantage)
    
    if len(batch_obs) == 0:
        logger.warning(f"Epoch {epoch}: 没有有效 StepNFT 样本，跳过训练")
        return global_step, 0.0
    
    # ========== 阶段2：组装 tensors ==========
    obs_cond = torch.cat(batch_obs, dim=0)  # [N, obs_dim]
    xt_all = torch.stack(batch_xt, dim=0)   # [N, action_horizon, action_dim]
    v_old_all = torch.stack(batch_v_old, dim=0)
    xnext_all = torch.stack(batch_xnext, dim=0)
    step_idx_all = torch.tensor(batch_step_index, device=accelerator.device, dtype=torch.long)
    std_all = torch.tensor(batch_std, device=accelerator.device, dtype=torch.float32)
    adv_all = torch.tensor(batch_advantages, device=accelerator.device, dtype=torch.float32)
    
    total_batch_size = obs_cond.shape[0]
    logger.info(f"Epoch {epoch} [StepNFT]: 训练 {total_batch_size} 个 nft 样本（每个仅单步 forward）")
    
    # ========== 阶段3：打乱数据 ==========
    perm = torch.randperm(total_batch_size, device=accelerator.device)
    obs_cond = obs_cond[perm]
    xt_all = xt_all[perm]
    v_old_all = v_old_all[perm]
    xnext_all = xnext_all[perm]
    step_idx_all = step_idx_all[perm]
    std_all = std_all[perm]
    adv_all = adv_all[perm]
    
    # ========== 阶段4：分批训练 ==========
    num_batches = max(1, total_batch_size // config.train.batch_size)
    effective_grad_accum_steps = config.train.gradient_accumulation_steps
    current_accumulated_steps = 0
    info = defaultdict(list)
    epoch_total_losses = []
    
    for batch_idx in tqdm(
        range(num_batches),
        desc=f"Epoch {epoch} [StepNFT]: training",
        disable=not accelerator.is_local_main_process
    ):
        start_idx = batch_idx * config.train.batch_size
        end_idx = min(start_idx + config.train.batch_size, total_batch_size)
        
        b_obs = obs_cond[start_idx:end_idx]
        b_xt = xt_all[start_idx:end_idx]
        b_v_old = v_old_all[start_idx:end_idx]
        b_xnext = xnext_all[start_idx:end_idx]
        b_step_idx = step_idx_all[start_idx:end_idx]
        b_std = std_all[start_idx:end_idx]
        b_adv = adv_all[start_idx:end_idx]
        
        bsz = b_obs.shape[0]
        
        with accelerator.autocast():
            # ========== 单步 forward：v_theta = current_unet(x_t, t, obs) ==========
            # 数学使用归一化 timestep（t = step_idx / num_steps，与 _flow_mean 一致），
            # 喂给 UNet 时再用 _unet_timestep_input 缩放到 pretrain 的 SinusoidalPosEmb
            # 频率范围 [0.99, ≈89.2]，否则与 pretrain 学到的速度场不在同一时间编码下。
            t_norm = b_step_idx.to(torch.float32) / float(num_train_timesteps)  # [bsz]
            t_unet = _unet_timestep_input(t_norm)
            
            current_pred_raw = action_unet(
                sample=b_xt,
                timestep=t_unet,
                global_cond=b_obs,
            )
            v_theta = action_decoder(current_pred_raw)
            
            # Reference model（KL 项用，no grad）
            with torch.no_grad():
                ref_pred_raw = action_unet_ref(
                    sample=b_xt,
                    timestep=t_unet,
                    global_cond=b_obs,
                )
                v_ref = action_decoder_ref(ref_pred_raw)
            
            # ========== StepNFT loss 主体（参考 pi-StepNFT/losses.py compute_nft_actor_loss）==========
            # 1) advantage → r/y
            advantages_clip = torch.clamp(b_adv, -adv_clip_max, adv_clip_max)
            r = torch.clamp((advantages_clip / adv_clip_max) / 2.0 + 0.5, 0.0, 1.0)
            y = 2.0 * r - 1.0  # [bsz]
            
            # 2) Δv 与 max_drift clip（防止单步漂移过大）
            #    v_old 来自采样时保存（detach），不再调 old 模型
            v_old_detached = b_v_old.detach()
            delta_v = v_theta - v_old_detached
            
            # delta_v.norm 沿 action 维度（与 pi-StepNFT 同：对每个 batch 项的 (action_horizon, action_dim) 取 norm）
            dims_action = tuple(range(1, delta_v.ndim))  # (1, 2) for [bsz, action_horizon, action_dim]
            delta_norm = delta_v.norm(dim=dims_action, keepdim=True) + 1e-8
            clip_coef = (max_drift / delta_norm).clamp(max=1.0)
            delta_v_clipped = delta_v * clip_coef
            
            v_pos = v_old_detached + beta * delta_v_clipped
            v_neg = v_old_detached - beta * delta_v_clipped
            
            # 3) _flow_mean（按当前方向 t∈[0,1], t=0 noise → t=1 data 重新推导）
            #    Flow Matching: x_t = (1-t)*x0 + t*x1, v = x1 - x0
            #    => x0_pred (noise 端) = x_t - v * t
            #       x1_pred (data 端)  = x_t + v * (1-t)
            #
            #    SDE 转移分布的 mean（对应 pi-StepNFT 的方向变换后系数）：
            #       mean = x1_pred * t_next + x0_pred * ((1 - t_next) - sigma_i^2 * delta / (2 * (1 - t_safe)))
            #    其中 sigma_i = noise_level * sigma_base = noise_level * sqrt((1-t)/t_safe)
            t_cur = t_norm  # [bsz]
            t_next = (b_step_idx.to(torch.float32) + 1.0) / float(num_train_timesteps)  # [bsz]
            delta = t_next - t_cur  # [bsz]，恒为 1/N
            t_safe = torch.clamp(t_cur, min=1.0 / float(num_train_timesteps))  # 避开 t=0 奇异
            
            # sigma_i^2（与采样保存的 std 对齐：std = noise_level * sigma_base * sqrt(delta)）
            # sigma_base^2 = (1 - t_safe) / t_safe
            # sigma_i^2   = noise_level^2 * sigma_base^2
            sigma_i_sq = (noise_level ** 2) * ((1.0 - t_safe) / t_safe)  # [bsz]
            
            # 广播到 action 形状
            t_cur_bc = t_cur.view(-1, *([1] * (b_xt.ndim - 1)))
            t_next_bc = t_next.view(-1, *([1] * (b_xt.ndim - 1)))
            delta_bc = delta.view(-1, *([1] * (b_xt.ndim - 1)))
            t_safe_bc = t_safe.view(-1, *([1] * (b_xt.ndim - 1)))
            sigma_i_sq_bc = sigma_i_sq.view(-1, *([1] * (b_xt.ndim - 1)))
            
            def _flow_mean(x_cur, velocity):
                x0_pred = x_cur - velocity * t_cur_bc       # noise 端
                x1_pred = x_cur + velocity * (1.0 - t_cur_bc)  # data 端
                x1_weight = t_next_bc  # data 端权重
                x0_weight = (1.0 - t_next_bc) - sigma_i_sq_bc * delta_bc / (2.0 * (1.0 - t_safe_bc))
                return x0_pred * x0_weight + x1_pred * x1_weight
            
            mean_pos = _flow_mean(b_xt, v_pos)
            mean_neg = _flow_mean(b_xt, v_neg)
            
            # 4) std / var（用采样时保存的 std；nft_std 已经是 noise_level*sigma_base*sqrt(delta)）
            std_t_bc = b_std.view(-1, *([1] * (b_xt.ndim - 1))).detach()
            var = std_t_bc.pow(2) + std_epsilon
            
            # 5) E_pos / E_neg / Δ_E
            E_pos = ((b_xnext - mean_pos).pow(2) / var).sum(dim=dims_action)  # [bsz]
            E_neg = ((b_xnext - mean_neg).pow(2) / var).sum(dim=dims_action)
            delta_E = E_pos - E_neg
            
            # 6) Loss 主体（weighted 或 dpo）
            if loss_form == "weighted":
                loss_per_sample = r * E_pos + (1.0 - r) * E_neg
            elif loss_form == "dpo":
                logit = (dpo_beta / 2.0) * y * delta_E
                loss_per_sample = torch.nn.functional.softplus(logit)
            else:
                raise ValueError(
                    f"Unknown stepnft_loss_form: {loss_form!r}. Expected 'weighted' or 'dpo'."
                )
            
            policy_loss = loss_per_sample.mean()
            
            # 7) KL 正则（v_theta 与 v_ref 的 MSE）
            kl_per_sample = ((v_theta - v_ref) ** 2).mean(dim=dims_action)
            kl_loss = kl_beta * kl_per_sample.mean()
            
            # 8) Total loss
            loss = policy_loss + kl_loss
        
        # ========== 反传 + 梯度累积 ==========
        scaled_loss = loss / effective_grad_accum_steps
        accelerator.backward(scaled_loss)
        current_accumulated_steps += 1
        
        # 收集监控指标
        info["policy_loss"].append(policy_loss.detach())
        info["kl_div_loss"].append(kl_loss.detach())
        info["total_loss"].append(loss.detach())
        info["E_pos"].append(E_pos.mean().detach())
        info["E_neg"].append(E_neg.mean().detach())
        info["delta_E"].append(delta_E.mean().detach())
        info["clip_frac"].append((clip_coef < 1.0).float().mean().detach())
        info["std_mean"].append(std_t_bc.mean().detach())
        info["r_mean"].append(r.mean().detach())
        info["y_abs_mean"].append(y.abs().mean().detach())
        info["prediction_std"].append(v_theta.std().detach())
        
        epoch_total_losses.append(loss.detach().item())
        
        # ========== 梯度累积完成后更新 ==========
        if current_accumulated_steps % effective_grad_accum_steps == 0:
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float('inf')
            ).item()
            grad_norm_after = accelerator.clip_grad_norm_(
                trainable_parameters, config.train.max_grad_norm
            ).item()
            grad_clipped = 1.0 if grad_norm_before > config.train.max_grad_norm else 0.0
            
            with torch.no_grad():
                param_before = torch.cat([p.data.flatten() for p in trainable_parameters])
            
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                param_after = torch.cat([p.data.flatten() for p in trainable_parameters])
                param_update_norm = (param_after - param_before).norm().item()
                param_norm = param_after.norm().item()
                param_update_ratio = param_update_norm / (param_norm + 1e-8)
            
            info_aggregated = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            
            if global_step % 10 == 0 and accelerator.is_local_main_process:
                logger.info(
                    f"  [StepNFT] Step {global_step}: "
                    f"loss={info_aggregated['total_loss']:.4f}, "
                    f"ΔE={info_aggregated['delta_E']:.4f}, "
                    f"grad_norm={grad_norm_before:.4f}"
                )
            
            # ========== wandb 日志 ==========
            if accelerator.is_main_process:
                if wandb.run is not None:
                    log_dict = {
                        "train/policy_loss": info_aggregated['policy_loss'].item(),
                        "train/kl_div_loss": info_aggregated['kl_div_loss'].item(),
                        "train/total_loss": info_aggregated['total_loss'].item(),
                        "train/grad_norm_before_clip": grad_norm_before,
                        "train/grad_norm_after_clip": grad_norm_after,
                        "train/epoch": epoch,
                        # StepNFT 专属指标
                        "stepnft/E_pos_mean": info_aggregated['E_pos'].item(),
                        "stepnft/E_neg_mean": info_aggregated['E_neg'].item(),
                        "stepnft/delta_E_mean": info_aggregated['delta_E'].item(),
                        "stepnft/clip_frac": info_aggregated['clip_frac'].item(),
                        "stepnft/std_mean": info_aggregated['std_mean'].item(),
                        "stepnft/r_mean": info_aggregated['r_mean'].item(),
                        "stepnft/y_abs_mean": info_aggregated['y_abs_mean'].item(),
                        "stepnft/loss": info_aggregated['policy_loss'].item(),
                        "stepnft/kl_loss": info_aggregated['kl_div_loss'].item(),
                        "stepnft/loss_form": 0.0 if loss_form == "weighted" else 1.0,
                        # 模型预测指标
                        "prediction/output_std": info_aggregated['prediction_std'].item(),
                        # 优化过程指标
                        "optim/grad_clipped": grad_clipped,
                        "optim/param_update_norm": param_update_norm,
                        "optim/param_norm": param_norm,
                        "optim/param_update_ratio": param_update_ratio,
                        "optim/learning_rate": optimizer.param_groups[0]['lr'],
                    }
                    wandb.log(log_dict, step=global_step)
            
            global_step += 1
            current_accumulated_steps = 0
            info = defaultdict(list)
            
            # EMA 更新（与 diffusionnft 对齐）
            if config.train.ema:
                ema.step(trainable_parameters, global_step)
            
            # ========== Old/Ref 模型同步（每 20 个 global_step；与 diffusionnft 对齐）==========
            if global_step % 20 == 0:
                with torch.no_grad():
                    decay = return_decay(global_step, config.train.decay_type)
                    
                    current_old_distance = 0.0
                    current_ref_distance = 0.0
                    old_ref_distance = 0.0
                    
                    for cur_p, old_p, ref_p in zip(
                        unwrap_model(action_unet, accelerator).parameters(),
                        action_unet_old.parameters(),
                        action_unet_ref.parameters()
                    ):
                        current_old_distance += (cur_p - old_p).pow(2).sum().item()
                        current_ref_distance += (cur_p - ref_p).pow(2).sum().item()
                        old_ref_distance += (old_p - ref_p).pow(2).sum().item()
                    
                    for cur_p, old_p, ref_p in zip(
                        unwrap_model(action_decoder, accelerator).parameters(),
                        action_decoder_old.parameters(),
                        action_decoder_ref.parameters()
                    ):
                        current_old_distance += (cur_p - old_p).pow(2).sum().item()
                        current_ref_distance += (cur_p - ref_p).pow(2).sum().item()
                        old_ref_distance += (old_p - ref_p).pow(2).sum().item()
                    
                    current_old_distance = np.sqrt(current_old_distance)
                    current_ref_distance = np.sqrt(current_ref_distance)
                    old_ref_distance = np.sqrt(old_ref_distance)
                    
                    for src_param, tgt_param in zip(
                        unwrap_model(action_unet, accelerator).parameters(),
                        action_unet_old.parameters()
                    ):
                        tgt_param.data.copy_(
                            tgt_param.data * decay + src_param.data * (1.0 - decay)
                        )
                    
                    for src_param, tgt_param in zip(
                        unwrap_model(action_decoder, accelerator).parameters(),
                        action_decoder_old.parameters()
                    ):
                        tgt_param.data.copy_(
                            tgt_param.data * decay + src_param.data * (1.0 - decay)
                        )
                    
                    if accelerator.is_local_main_process:
                        logger.info(f"  [StepNFT] [Old model 更新] Step {global_step}: decay={decay:.4f}")
                    
                    if accelerator.is_main_process:
                        if wandb.run is not None:
                            wandb.log({
                                "model/current_old_distance": current_old_distance,
                                "model/current_ref_distance": current_ref_distance,
                                "model/old_ref_distance": old_ref_distance,
                                "model/old_decay": decay,
                            }, step=global_step)
    
    avg_epoch_loss = np.mean(epoch_total_losses) if len(epoch_total_losses) > 0 else 0.0
    
    return global_step, avg_epoch_loss


def eval_in_env(pipeline, action_unet, action_decoder, env, config, accelerator, global_step, num_test_episodes=3):
    """
    在环境中评估策略
    
    参数:
        pipeline: RobotActionPipeline
        action_unet: ConditionalUnet1D模型
        action_decoder: ActionDecoder
        env: ManiSkill环境
        config: 配置
        accelerator: Accelerator
        global_step: 全局步数
        num_test_episodes: 测试episode数量
    """
    action_unet.eval()
    action_decoder.eval()
    
    all_episode_rewards = []
    all_episode_lengths = []
    
    max_steps = config.max_episode_steps if hasattr(config, 'max_episode_steps') else 500
    
    for test_idx in range(num_test_episodes):
        obs, info = env.reset(seed=10000 + test_idx + accelerator.process_index * 100)
        
        # 转换观测为tensor
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(accelerator.device, dtype=torch.float32)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            obs_tensor = torch.tensor(obs, device=accelerator.device, dtype=torch.float32).unsqueeze(0)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            with torch.no_grad():
                # 生成动作序列（ODE确定性模式）
                # 评估时也使用generator确保可复现
                eval_gen = torch.Generator(device=accelerator.device)
                eval_gen.manual_seed(test_idx + 10000)  # 使用固定seed
                
                result = pipeline_with_logprob(
                    pipeline,
                    obs=obs_tensor,
                    num_inference_steps=config.sample.eval_num_steps if hasattr(config.sample, 'eval_num_steps') else 10,
                    output_type="clipped",
                    return_dict=True,
                    action_decoder=action_decoder,
                    generator=eval_gen,  # ✅ 传入generator
                )
                
                # GPU优化：保持actions在GPU上
                actions = result["actions"][0]
            
            # 执行完整的 action_horizon 步动作
            for j in range(config.action_horizon):
                if done or episode_length >= max_steps:
                    break
                
                # 确保动作是连续内存
                action = actions[j].contiguous() if isinstance(actions, torch.Tensor) else actions[j]
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                episode_length += 1
                done = terminated
            
            # 更新观测（用于下一次生成）
            if not done and episode_length < max_steps:
                if isinstance(obs, torch.Tensor):
                    obs_tensor = obs.to(accelerator.device, dtype=torch.float32)
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                else:
                    obs_tensor = torch.tensor(obs, device=accelerator.device, dtype=torch.float32).unsqueeze(0)
        
        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
    
    # 收集所有进程的结果
    episode_rewards_tensor = torch.tensor(all_episode_rewards, device=accelerator.device)
    episode_lengths_tensor = torch.tensor(all_episode_lengths, device=accelerator.device)
    
    gathered_rewards = accelerator.gather(episode_rewards_tensor).cpu().numpy()
    gathered_lengths = accelerator.gather(episode_lengths_tensor).cpu().numpy()
    
    if accelerator.is_main_process:
        logger.info(f"Evaluation at step {global_step}:")
        logger.info(f"  Mean episode reward: {gathered_rewards.mean():.3f} ± {gathered_rewards.std():.3f}")
        logger.info(f"  Mean episode length: {gathered_lengths.mean():.1f} ± {gathered_lengths.std():.1f}")
        
        accelerator.log({
            "eval/episode_reward_mean": gathered_rewards.mean(),
            "eval/episode_reward_std": gathered_rewards.std(),
            "eval/episode_length_mean": gathered_lengths.mean(),
        }, step=global_step)
    
    action_unet.train()
    action_decoder.train()


def test_model(config, checkpoint_path, num_test_episodes=10, gpu_id=0):
    """
    测试保存的模型并录制视频
    
    参数:
        config: 配置
        checkpoint_path: 检查点路径
        num_test_episodes: 测试episode数量
        gpu_id: 使用的GPU ID
    """
    import os
    import logging
    os.makedirs("videos", exist_ok=True)
    
    # 使用标准 Python logging 而不是 accelerate logger
    test_logger = logging.getLogger(__name__)
    test_logger.setLevel(logging.INFO)
    
    # 如果没有 handler，添加一个
    if not test_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        test_logger.addHandler(handler)
    
    test_logger.info("=" * 60)
    test_logger.info("开始测试模型")
    test_logger.info(f"检查点路径: {checkpoint_path}")
    test_logger.info(f"测试 episode 数: {num_test_episodes}")
    test_logger.info(f"物理 GPU: {gpu_id} → 虚拟设备: cuda:0")
    test_logger.info("=" * 60)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        test_logger.info(f"使用设备: {device} (当前CUDA设备: cuda:{torch.cuda.current_device()})")
    else:
        device = torch.device("cpu")
        test_logger.info(f"使用设备: {device}")
    
    # 创建测试环境（带视频录制）
    env_kwargs = {
        'sim_backend': config.env_kwargs.get('sim_backend', 'physx_cpu'),
        'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode': config.env_kwargs.get('obs_mode', 'state'),
        'render_mode': 'rgb_array',  # 确保可以录制视频
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense')
    }
    
    test_logger.info(f"创建测试环境: {config.env_id}")
    test_logger.info(f"测试环境模拟后端: {env_kwargs['sim_backend']}")
    base_env = gym.make(config.env_id, reconfiguration_freq=1, **env_kwargs)
    
    # 使用 ManiSkill 自带的 RecordEpisode wrapper 录制视频
    from mani_skill.utils.wrappers.record import RecordEpisode
    record_kwargs = {
        'save_trajectory': False,
        'save_video': True,
        'video_fps': 30
    }
    env = RecordEpisode(base_env, "videos", **record_kwargs)
    
    # 获取环境维度
    obs, _ = env.reset(seed=config.seed)
    if isinstance(obs, torch.Tensor):
        obs_dim = obs.shape[-1]
    else:
        obs_dim = obs.shape[-1] if hasattr(obs, 'shape') else len(obs)
    
    # 获取动作维度（测试环境是单个环境）
    if hasattr(env, 'single_action_space'):
        action_dim = env.single_action_space.shape[0]
    elif len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[1]
    else:
        action_dim = env.action_space.shape[0]
    
    test_logger.info(f"环境维度: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # 创建模型
    test_logger.info("创建模型...")
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(device)
    
    action_decoder = ActionDecoder(
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        hidden_dim=256,
        num_layers=3
    ).to(device)
    
    # 创建管线
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        device=str(device)
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder
    
    # 加载检查点
    test_logger.info(f"加载检查点: {checkpoint_path}")
    if os.path.isdir(checkpoint_path):
        # 如果是目录，查找model.pth
        model_path = os.path.join(checkpoint_path, "model.pth")
    else:
        model_path = checkpoint_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    action_unet.load_state_dict(checkpoint['action_unet_state_dict'])
    action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])
    test_logger.info("模型加载成功！")
    
    # 设置为评估模式
    action_unet.eval()
    action_decoder.eval()
    
    # 测试
    all_rewards = []
    all_lengths = []
    all_success = []
    
    max_steps = config.max_episode_steps if hasattr(config, 'max_episode_steps') else 500
    
    test_logger.info(f"\n开始测试 {num_test_episodes} 个 episodes...")
    for episode_idx in range(num_test_episodes):
        obs, info = env.reset(seed=config.seed + episode_idx + 10000)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        test_logger.info(f"\nEpisode {episode_idx + 1}/{num_test_episodes}")
        
        while not done and episode_length < max_steps:
            # 转换观测为tensor
            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.to(device, dtype=torch.float32)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
            else:
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # 生成动作序列（ODE确定性模式）
                # 测试时也使用generator确保可复现
                test_gen = torch.Generator(device=device)
                test_gen.manual_seed(config.seed + episode_idx + 10000)
                
                result = pipeline_with_logprob(
                    pipeline,
                    obs=obs_tensor,
                    num_inference_steps=config.sample.eval_num_steps if hasattr(config.sample, 'eval_num_steps') else 10,
                    output_type="clipped",
                    return_dict=True,
                    action_decoder=action_decoder,
                    generator=test_gen,  # ✅ 传入generator
                )
                
                actions = result["actions"][0].cpu().numpy()
            
            # 执行动作
            for j in range(config.action_horizon):
                if done or episode_length >= max_steps:
                    break
                
                action = actions[j]
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                episode_length += 1
                done = terminated
        
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_success.append(1 if done else 0)
        
        test_logger.info(f"  Reward: {episode_reward:.3f}, Length: {episode_length}, Success: {done}")
        
        # 保存视频
        env.flush_video(ignore_empty_transition=False)
    
    env.close()
    
    # 统计结果
    all_rewards = np.array(all_rewards)
    all_lengths = np.array(all_lengths)
    all_success = np.array(all_success)
    
    test_logger.info("\n" + "=" * 60)
    test_logger.info("测试结果:")
    test_logger.info(f"  平均奖励: {all_rewards.mean():.3f} ± {all_rewards.std():.3f}")
    test_logger.info(f"  平均长度: {all_lengths.mean():.1f} ± {all_lengths.std():.1f}")
    test_logger.info(f"  成功率: {all_success.mean():.2%} ({all_success.sum()}/{len(all_success)})")
    test_logger.info(f"  视频保存在: videos/")
    test_logger.info("=" * 60)
    
    # 保存结果到文件
    results = {
        'rewards': all_rewards.tolist(),
        'lengths': all_lengths.tolist(),
        'success': all_success.tolist(),
        'mean_reward': float(all_rewards.mean()),
        'std_reward': float(all_rewards.std()),
        'mean_length': float(all_lengths.mean()),
        'std_length': float(all_lengths.std()),
        'success_rate': float(all_success.mean()),
        'checkpoint': checkpoint_path,
        'num_episodes': num_test_episodes,
    }
    
    results_path = os.path.join("videos", "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    test_logger.info(f"测试结果保存到: {results_path}")


def test_sample(config, pretrain_checkpoint, gpu_id):
    """
    使用与训练相同的采样流程生成测试轨迹并保存视频
    
    参数:
        config: 配置对象
        pretrain_checkpoint: 预训练检查点路径
        gpu_id: GPU设备ID
    """
    import os
    import logging
    from mani_skill.utils.wrappers.record import RecordEpisode
    
    # 创建输出目录
    os.makedirs("sample_video", exist_ok=True)
    
    test_logger = logging.getLogger(__name__)
    test_logger.info("=" * 60)
    test_logger.info("开始生成采样测试视频")
    test_logger.info(f"预训练检查点: {pretrain_checkpoint}")
    test_logger.info(f"物理 GPU: {gpu_id} → 虚拟设备: cuda:0")
    test_logger.info("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建录制环境（2个并行环境用于2个初始状态）
    num_record_envs = 2  # 2个初始状态
    samples_per_state = 5  # 每个状态5条轨迹
    
    env_kwargs = {
        'num_envs': num_record_envs,
        'sim_backend': config.env_kwargs.get('sim_backend', 'gpu'),
        'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode': config.env_kwargs.get('obs_mode', 'state'),
        'render_mode': 'rgb_array',  # 需要渲染模式以录制视频
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense')
    }
    
    test_logger.info(f"创建录制环境: {config.env_id}")
    base_env = gym.make(config.env_id, reconfiguration_freq=1, **env_kwargs)
    
    # 包装为录制环境
    record_kwargs = {
        'save_trajectory': False,
        'save_video': True,
        'video_fps': 30,
        'max_steps_per_video': config.max_episode_steps  # GPU并行环境必需
    }
    env = RecordEpisode(base_env, "sample_video", **record_kwargs)
    
    # 获取环境维度
    obs, _ = env.reset(seed=config.seed)
    obs_dim = obs.shape[-1] if isinstance(obs, torch.Tensor) else obs.shape[-1]
    action_dim = env.single_action_space.shape[0] if hasattr(env, 'single_action_space') else env.action_space.shape[1]
    
    test_logger.info(f"环境维度: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # 创建模型
    test_logger.info("创建模型...")
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(device)
    
    action_decoder = ActionDecoder(
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        hidden_dim=256,
        num_layers=3
    ).to(device)
    
    # 创建管线
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        device=str(device)
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder
    
    # 加载预训练检查点
    if not pretrain_checkpoint or not os.path.exists(pretrain_checkpoint):
        raise ValueError(f"预训练检查点不存在: {pretrain_checkpoint}")
    
    test_logger.info(f"加载预训练检查点: {pretrain_checkpoint}")
    checkpoint = torch.load(pretrain_checkpoint, map_location=device)
    action_unet.load_state_dict(checkpoint['action_unet_state_dict'])
    action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])
    test_logger.info("✓ 模型加载成功")
    
    action_unet.eval()
    action_decoder.eval()
    
    # 创建简单的accelerator和autocast（用于复用sample_complete_trajectories）
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    autocast = accelerator.autocast
    
    # 生成轨迹 - 循环5次，每次生成2条轨迹（2个初始状态）
    test_logger.info(f"\n开始生成轨迹: 2个初始状态 × 5次采样 = 10条轨迹")
    
    all_trajectory_data = []
    
    for sample_idx in range(samples_per_state):
        test_logger.info(f"\n--- 采样轮次 {sample_idx + 1}/{samples_per_state} ---")
        
        # 使用第一个epoch的种子: config.seed + 0 * 100000 = 42
        seed_offset = config.seed + 0 * 100000
        
        # 生成2个base seeds（2个初始状态）
        base_seeds = [seed_offset + i for i in range(num_record_envs)]
        all_seeds = []
        for base_seed in base_seeds:
            all_seeds.append(base_seed)
        
        test_logger.info(f"使用种子: {all_seeds}")
        
        # 重置环境
        obs, info = env.reset(seed=all_seeds)
        
        # 记录每个环境的帧序列
        env_frames = [[] for _ in range(num_record_envs)]
        env_done = torch.zeros(num_record_envs, dtype=torch.bool, device=device)
        env_steps = torch.zeros(num_record_envs, dtype=torch.long, device=device)
        env_rewards = torch.zeros(num_record_envs, dtype=torch.float32, device=device)
        
        max_steps = config.max_episode_steps
        flow_call_count = 0
        
        # 执行轨迹
        while not env_done.all() and flow_call_count * config.action_horizon < max_steps:
            active_mask = ~env_done
            
            if active_mask.any():
                obs_tensor = _prepare_obs_tensor(obs, device)
                
                # Flow生成动作（ODE确定性模式）
                with autocast():
                    with torch.no_grad():
                        # ✅ 为每个环境创建独立的generator
                        test_generators = []
                        for env_idx in range(num_record_envs):
                            test_gen = torch.Generator(device=device)
                            # 使用 seed + sample_idx + env_idx 确保每条轨迹的随机性不同
                            test_gen.manual_seed(all_seeds[env_idx] + sample_idx * 1000 + flow_call_count * 100)
                            test_generators.append(test_gen)
                        
                        result = pipeline_with_logprob(
                            pipeline, obs=obs_tensor,
                            num_inference_steps=config.sample.num_steps,
                            output_type="clipped", return_dict=True,
                            action_decoder=action_decoder,
                            generator=test_generators,  # ✅ 传入generator列表
                        )
                
                actions_batch = result["actions"]
                
                # 执行action_horizon步
                for j in range(config.action_horizon):
                    if env_done.all():
                        break
                    
                    step_active_mask = ~env_done
                    action_step = actions_batch[:, j].contiguous()
                    obs, reward, terminated, truncated, info = env.step(action_step)
                    
                    # 转换为tensor
                    if not isinstance(reward, torch.Tensor):
                        reward = torch.tensor(reward, device=device)
                    if not isinstance(terminated, torch.Tensor):
                        terminated = torch.tensor(terminated, device=device)
                    
                    # 累积奖励
                    env_rewards[step_active_mask] += reward[step_active_mask]
                    
                    # 更新done和步数
                    env_done = terminated
                    env_steps[step_active_mask] += 1
            
            flow_call_count += 1
        
        # 保存轨迹信息
        for env_idx in range(num_record_envs):
            trajectory_info = {
                'state_id': env_idx,  # 初始状态ID (0或1)
                'sample_id': sample_idx,  # 采样ID (0-4)
                'seed': all_seeds[env_idx],
                'total_reward': env_rewards[env_idx].item(),
                'total_steps': env_steps[env_idx].item(),
                'success': env_done[env_idx].item()
            }
            all_trajectory_data.append(trajectory_info)
            
            test_logger.info(f"  状态{env_idx}_样本{sample_idx}: "
                       f"奖励={trajectory_info['total_reward']:.1f}, "
                       f"步数={trajectory_info['total_steps']}, "
                       f"成功={trajectory_info['success']}")
        
        # 刷新视频（ManiSkill的RecordEpisode会自动保存）
        env.flush_video(ignore_empty_transition=False)
    
    env.close()
    
    # 重命名视频文件以匹配我们的命名规范
    test_logger.info("\n重命名视频文件...")
    video_files = sorted([f for f in os.listdir("sample_video") if f.endswith('.mp4')])
    
    for idx, (video_file, traj_info) in enumerate(zip(video_files, all_trajectory_data)):
        old_path = os.path.join("sample_video", video_file)
        new_name = f"state{traj_info['state_id']}_sample{traj_info['sample_id']}_seed{traj_info['seed']}.mp4"
        new_path = os.path.join("sample_video", new_name)
        
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            test_logger.info(f"  {video_file} → {new_name}")
    
    # 保存轨迹统计信息
    stats_path = os.path.join("sample_video", "trajectory_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(all_trajectory_data, f, indent=2)
    test_logger.info(f"\n轨迹统计信息保存到: {stats_path}")
    
    # 打印汇总统计
    test_logger.info("\n" + "=" * 60)
    test_logger.info("采样测试完成!")
    test_logger.info(f"  总轨迹数: {len(all_trajectory_data)}")
    test_logger.info(f"  平均奖励: {np.mean([t['total_reward'] for t in all_trajectory_data]):.3f}")
    test_logger.info(f"  成功率: {sum(t['success'] for t in all_trajectory_data)}/{len(all_trajectory_data)}")
    test_logger.info(f"  视频保存在: sample_video/")
    test_logger.info("=" * 60)


def unwrap_model(model, accelerator):
    from diffusers.utils.torch_utils import is_compiled_module
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, action_unet, action_decoder, epoch, accelerator, ema, trainable_parameters, config):
    """保存检查点

    除模型权重外，额外写入架构元数据 (obs_dim / action_dim / action_horizon / 关键超参)
    以及完整 config 字典，便于测试/推理脚本严格复现训练时的模型结构与推理协议。
    """
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-epoch{epoch}")
    os.makedirs(save_root, exist_ok=True)
    
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(trainable_parameters, store_temp=True)
        
        unet_unwrapped = unwrap_model(action_unet, accelerator)
        decoder_unwrapped = unwrap_model(action_decoder, accelerator)
        
        # ── 推导架构超参（与 ConditionalUnet1D / ActionDecoder 的构造一致）──
        # action_dim / action_horizon 直接来自 ActionDecoder 的属性，可靠性最高
        action_dim_meta = int(getattr(decoder_unwrapped, 'action_dim', config.env_kwargs.get('action_dim', 4) if hasattr(config, 'env_kwargs') else 4))
        action_horizon_meta = int(getattr(decoder_unwrapped, 'action_horizon', config.action_horizon))
        # obs_dim 从 UNet 第一个 down_module 的 FiLM cond_encoder 反推：
        #   cond_in_features = diffusion_step_embed_dim (默认 256) + obs_dim
        diffusion_step_embed_dim = 256
        try:
            cond_in_features = unet_unwrapped.down_modules[0][0].cond_encoder[1].in_features
            obs_dim_meta = int(cond_in_features - diffusion_step_embed_dim)
        except Exception:
            obs_dim_meta = None
        
        # 把 ml_collections 配置转换为普通 dict 便于 torch.load 后随意访问
        try:
            config_dict = config.to_dict()
        except Exception:
            config_dict = None
        
        # 保存模型权重 + 架构元数据
        torch.save({
            'action_unet_state_dict': unet_unwrapped.state_dict(),
            'action_decoder_state_dict': decoder_unwrapped.state_dict(),
            'epoch': epoch,
            'obs_dim': obs_dim_meta,
            'action_dim': action_dim_meta,
            'action_horizon': action_horizon_meta,
            'diffusion_step_embed_dim': diffusion_step_embed_dim,
            'unet_class': 'unet.ConditionalUnet1D',
            'action_decoder_class': 'flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet.ActionDecoder',
            'loss_type': getattr(config.train, 'loss_type', 'rollout'),
            'sample_num_steps': int(config.sample.num_steps),
            'ema_enabled': bool(config.train.ema),
            'config_dict': config_dict,
        }, os.path.join(save_root, "model.pth"))
        
        if config.train.ema:
            ema.copy_temp_to(trainable_parameters)
        
        logger.info(f"Saved checkpoint to {save_root}")
        logger.info(
            f"  元数据: obs_dim={obs_dim_meta}, action_dim={action_dim_meta}, "
            f"action_horizon={action_horizon_meta}, loss_type={getattr(config.train, 'loss_type', 'rollout')}"
        )


def evaluate_pretrain_model(action_unet, action_decoder, pipeline, config, device,
                            num_episodes=10, obs_encoder=None,
                            normalizer=None, obs_mode=None):
    """
    在 ManiSkill 环境中评估预训练模型。

    支持三种观测模式（通过 obs_mode 参数显式指定，未指定时自动从 config 推断）：
      "state"      — 直接用扁平状态向量作为 obs_cond（obs_encoder=None）
      "pointcloud" — 用 PointCloudObsEncoder 编码 (point_cloud, agent_pos) 后作为 obs_cond
      "image"      — 用 ImageObsEncoder 编码 (image, agent_pos) 后作为 obs_cond

    pointcloud / image 模式下，agent_pos 在送入 obs_encoder 前会用 normalizer['agent_pos']
    归一化（与训练时完全一致）。

    参数:
        obs_encoder: PointCloudObsEncoder / ImageObsEncoder 实例（None 表示 state 模式）
        normalizer:  LinearNormalizer（pointcloud / image 模式需要）
        obs_mode:    'state' / 'pointcloud' / 'image'，None 时按 obs_encoder 是否为 None 兜底

    返回:
        success_count, total_count, avg_reward
    """
    action_unet.eval()
    if action_decoder is not None:
        action_decoder.eval()
    if obs_encoder is not None:
        obs_encoder.eval()

    # 兜底推断 obs_mode：obs_encoder 存在时优先用 config.pretrain.obs_mode
    if obs_mode is None:
        if obs_encoder is None:
            obs_mode = 'state'
        else:
            obs_mode = getattr(config.pretrain, 'obs_mode', 'pointcloud')

    if obs_mode not in ('state', 'pointcloud', 'image'):
        raise ValueError(f"未知的 obs_mode: {obs_mode!r}")

    # 环境 obs_mode：image→rgbd；其余直接复用
    env_obs_mode_map = {
        'state':      config.env_kwargs.get('obs_mode', 'state'),
        'pointcloud': 'pointcloud',
        'image':      'rgbd',
    }
    env_obs_mode = env_obs_mode_map[obs_mode]

    env_kwargs = {
        'sim_backend': config.env_kwargs.get('sim_backend', 'gpu'),
        'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode':     env_obs_mode,
        'render_mode':  'rgb_array',
        'reward_mode':  config.env_kwargs.get('reward_mode', 'dense'),
    }

    env = gym.make(config.env_id, reconfiguration_freq=1, num_envs=1, **env_kwargs)

    # ── 各模式专属超参 ────────────────────────────────────────────────
    pcd_n_obs_steps = getattr(config.pretrain, 'pcd_n_obs_steps', 2)
    pcd_n_points    = getattr(config.pretrain, 'pcd_n_points', 512)

    img_n_obs_steps = getattr(config.pretrain, 'img_n_obs_steps', 2)
    img_camera      = getattr(config.pretrain, 'img_camera', 'base_camera')
    img_use_depth   = bool(getattr(config.pretrain, 'img_use_depth', True))
    img_depth_max   = float(getattr(config.pretrain, 'img_depth_max', 5.0))

    success_count = 0
    total_rewards = []
    max_steps = config.max_episode_steps if hasattr(config, 'max_episode_steps') else 500

    from collections import deque as _deque

    def _maybe_normalize_apos(stack_dict):
        """对 stack 出来的 agent_pos 应用 normalizer（与训练严格对齐）。"""
        if normalizer is not None and 'agent_pos' in stack_dict:
            stack_dict['agent_pos'] = normalizer['agent_pos'].normalize(stack_dict['agent_pos'])
        return stack_dict

    def _encode_obs(raw_obs, obs_deque):
        """根据 obs_mode 抽取 + 编码当前 raw_obs，返回 (1, enc_dim) 的 obs_tensor。"""
        if obs_mode == 'pointcloud':
            step_obs = extract_pointcloud_obs(raw_obs, pcd_n_points)
            obs_deque.append(step_obs)
            stk = stack_pointcloud_obs(obs_deque, pcd_n_obs_steps, device)
            stk = _maybe_normalize_apos(stk)
            with torch.no_grad():
                return obs_encoder(stk['point_cloud'], stk['agent_pos'])
        elif obs_mode == 'image':
            step_obs = extract_image_obs(raw_obs, camera=img_camera,
                                          use_depth=img_use_depth,
                                          depth_max=img_depth_max)
            obs_deque.append(step_obs)
            stk = stack_image_obs(obs_deque, img_n_obs_steps, device)
            stk = _maybe_normalize_apos(stk)
            with torch.no_grad():
                return obs_encoder(stk['image'], stk['agent_pos'])
        else:
            # state 模式：转成 tensor 并用 normalizer 归一化（与训练严格对齐）
            if isinstance(raw_obs, torch.Tensor):
                t = raw_obs.to(device, dtype=torch.float32)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
            else:
                t = torch.tensor(raw_obs, device=device, dtype=torch.float32).unsqueeze(0)
            if normalizer is not None:
                t = normalizer['obs'].normalize(t)
            return t

    for episode_idx in range(num_episodes):
        raw_obs, info = env.reset(seed=20000 + episode_idx)

        # 每个 episode 重新创建滑窗 deque
        if obs_mode == 'pointcloud':
            obs_deque = _deque(maxlen=pcd_n_obs_steps)
        elif obs_mode == 'image':
            obs_deque = _deque(maxlen=img_n_obs_steps)
        else:
            obs_deque = None

        obs_tensor = _encode_obs(raw_obs, obs_deque)

        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            with torch.no_grad():
                eval_gen = torch.Generator(device=device)
                eval_gen.manual_seed(20000 + episode_idx)

                result = pipeline_with_logprob(
                    pipeline,
                    obs=obs_tensor,
                    num_inference_steps=config.sample.eval_num_steps,
                    output_type="clipped",
                    return_dict=True,
                    action_decoder=action_decoder,
                    generator=eval_gen,
                )
                actions_norm = result["actions"][0]  # (H, action_dim)，归一化空间

            # state 模式：模型输出的是归一化空间的 action，需要反归一化到真实空间
            if obs_mode == 'state' and normalizer is not None:
                actions = normalizer['action'].unnormalize(
                    actions_norm.unsqueeze(0)
                ).squeeze(0)
            else:
                actions = actions_norm

            for j in range(config.action_horizon):
                if done or episode_length >= max_steps:
                    break

                action = actions[j].contiguous() if isinstance(actions, torch.Tensor) else actions[j]
                raw_obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                episode_length += 1
                done = bool(terminated.item() if isinstance(terminated, torch.Tensor) else terminated)

            if not done and episode_length < max_steps:
                obs_tensor = _encode_obs(raw_obs, obs_deque)

        if done:
            success_count += 1
        total_rewards.append(episode_reward)

    env.close()

    action_unet.train()
    if action_decoder is not None:
        action_decoder.train()
    if obs_encoder is not None:
        obs_encoder.train()

    return success_count, num_episodes, float(np.mean(total_rewards))


def evaluate_model_inline(
    action_unet,
    action_decoder,
    pipeline,
    config,
    device,
    accelerator,
    ema=None,
    trainable_parameters=None,
    num_episodes=20,
    seed=20000,
    epoch=0,
):
    """
    在 RL 训练循环中内联评估当前策略性能，与 test_pretrain_model.py 的推理逻辑完全对齐。

    关键设计：
    - 若启用 EMA，临时将 EMA 权重拷贝到模型上进行评估，评估后恢复在线权重
    - 使用 denoise_ode_inference + _unet_timestep_input（与 pretrain 的 timestep 约定一致）
    - 使用固定 seed 生成验证集 episode_ids，保证每次评估在相同场景下比较
    - 返回成功率（百分比），供调用方记录到 wandb

    参数:
        action_unet: 当前 UNet（可能被 accelerator wrap）
        action_decoder: 当前 ActionDecoder（可能被 accelerator wrap）
        pipeline: RobotActionPipeline
        config: 配置对象
        device: torch.device
        accelerator: Accelerator 对象
        ema: EMAModuleWrapper（可选），若为 None 则直接使用在线权重
        trainable_parameters: EMA 参数列表（store_temp/copy_temp 用）
        num_episodes: 评估 episode 数
        seed: 验证集 episode_ids 的随机种子
        epoch: 当前 epoch 编号（仅用于日志）

    返回:
        dict，包含 success_rate(0-100)、num_success、mean_reward 等字段
    """
    logger.info(f"\n[Inline Eval] Epoch {epoch}: 开始评估 {num_episodes} 个 episodes...")

    unet_unwrapped = unwrap_model(action_unet, accelerator)
    decoder_unwrapped = unwrap_model(action_decoder, accelerator)

    # ── 临时拷贝 EMA 权重 ──
    ema_active = (ema is not None) and (trainable_parameters is not None)
    if ema_active:
        ema.copy_ema_to(trainable_parameters, store_temp=True)
        logger.info("[Inline Eval] 已将 EMA 权重拷贝到模型（评估用）")

    unet_unwrapped.eval()
    decoder_unwrapped.eval()

    try:
        # ── 生成固定验证集 episode_ids ──
        val_gen = torch.Generator(device='cpu')
        val_gen.manual_seed(seed)
        episode_ids = torch.randint(
            low=0, high=10000,
            size=(num_episodes,),
            generator=val_gen, dtype=torch.int64,
        ).tolist()

        # ── 创建评估环境（CPU render 开销低，与训练环境隔离） ──
        env_kwargs = {
            'sim_backend': config.env_kwargs.get('sim_backend', 'gpu'),
            'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
            'obs_mode': config.env_kwargs.get('obs_mode', 'state'),
            'render_mode': 'rgb_array',
            'reward_mode': config.env_kwargs.get('reward_mode', 'dense'),
        }
        eval_env = gym.make(config.env_id, reconfiguration_freq=1, **env_kwargs)

        action_horizon = config.action_horizon
        action_dim = getattr(pipeline, 'action_dim', 4)
        max_steps = config.max_episode_steps if hasattr(config, 'max_episode_steps') else 300
        num_inference_steps = config.sample.eval_num_steps

        all_rewards = []
        all_success = []

        for ep_idx, episode_id in enumerate(episode_ids):
            obs, _ = eval_env.reset(seed=episode_id, options={'episode_id': episode_id})

            if isinstance(obs, torch.Tensor):
                obs_tensor = obs.to(device, dtype=torch.float32)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
            else:
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)

            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done and episode_length < max_steps:
                eval_rng = torch.Generator(device=device)
                eval_rng.manual_seed(episode_id * 10000 + episode_length)

                obs_normalizer = getattr(pipeline, 'normalize_obs', None)
                actions_batch = denoise_ode_inference(
                    obs_tensor=obs_tensor,
                    action_unet=unet_unwrapped,
                    action_decoder=decoder_unwrapped,
                    num_steps=num_inference_steps,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                    device=device,
                    generator=eval_rng,
                    obs_normalizer=obs_normalizer,
                )
                actions = actions_batch[0]

                for j in range(action_horizon):
                    if done or episode_length >= max_steps:
                        break
                    action = actions[j].contiguous()
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    episode_reward += float(reward.item() if isinstance(reward, torch.Tensor) else reward)
                    episode_length += 1
                    done = bool(terminated)

                if not done and episode_length < max_steps:
                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.to(device, dtype=torch.float32)
                        if obs_tensor.dim() == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                    else:
                        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)

            all_rewards.append(episode_reward)
            all_success.append(1 if done else 0)

        eval_env.close()

        success_rate = float(np.mean(all_success)) * 100.0
        mean_reward = float(np.mean(all_rewards))
        num_success = int(sum(all_success))

        logger.info(
            f"[Inline Eval] Epoch {epoch}: "
            f"成功率 {num_success}/{num_episodes} ({success_rate:.1f}%), "
            f"平均奖励 {mean_reward:.3f}"
        )

        return {
            'success_rate': success_rate,
            'num_success': num_success,
            'num_episodes': num_episodes,
            'mean_reward': mean_reward,
            'std_reward': float(np.std(all_rewards)),
        }

    finally:
        # ── 无论评估是否异常，都恢复在线权重 ──
        if ema_active:
            ema.copy_temp_to(trainable_parameters)
            logger.info("[Inline Eval] 已恢复在线权重")
        unet_unwrapped.train()
        decoder_unwrapped.train()


@torch.no_grad()
def denoise_ode_inference(
    obs_tensor,
    action_unet,
    action_decoder,
    num_steps,
    action_horizon,
    action_dim,
    device,
    generator=None,
    obs_normalizer=None,
):
    """
    与 test_pretrain_model.py 完全对齐的 ODE 推理。

    timestep 缩放：t_unet = _unet_timestep_input(t) = t * 98.01 + 0.99
    匹配 pretrain 阶段 pipeline_with_logprob 使用的 SinusoidalPosEmb 频率范围。
    """
    batch_size = obs_tensor.shape[0]
    if generator is not None:
        x_t = torch.randn(
            batch_size, action_horizon, action_dim,
            generator=generator, device=device, dtype=torch.float32,
        )
    else:
        x_t = torch.randn(
            batch_size, action_horizon, action_dim,
            device=device, dtype=torch.float32,
        )

    obs_embeds = obs_tensor
    if obs_normalizer is not None:
        obs_embeds = obs_normalizer(obs_embeds)

    delta = 1.0 / num_steps
    for i in range(num_steps):
        t_value = i / num_steps
        t_input = torch.full(
            (batch_size,), _unet_timestep_input(t_value),
            device=device, dtype=torch.float32,
        )
        pred_raw = action_unet(sample=x_t, timestep=t_input, global_cond=obs_embeds)
        v_t = action_decoder(pred_raw) if action_decoder is not None else pred_raw
        x_t = x_t + v_t * delta

    return torch.clamp(x_t, -1.0, 1.0)


def pretrain_model(config, gpu_id):
    """
    使用成功轨迹对 Flow 策略进行预训练。

    支持两种观测模式（通过 config.pretrain.obs_mode 切换）：
      "state"      — 扁平化状态向量，与原有逻辑完全相同
      "pointcloud" — PointNet 编码点云 + agent_pos，编码结果作为 obs_cond

    两种模式共享相同的训练方式：
        generate_with_grad(pipeline, obs=obs_cond, ...)
        loss = MSE(pred_actions, target_actions)
    """
    from torch.utils.data import DataLoader
    from diffusers.training_utils import EMAModel
    from diffusers.optimization import get_scheduler

    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES 已在进入 main 前设置

    obs_mode = getattr(config.pretrain, 'obs_mode', 'state')
    pretrain_loss_type = getattr(config.pretrain, 'loss_type', 'action')
    if pretrain_loss_type not in ('action', 'velocity'):
        raise ValueError(
            f"未知的 config.pretrain.loss_type: {pretrain_loss_type!r}，"
            f"应为 'action' 或 'velocity'"
        )

    print(f"\n{'='*60}")
    print(f"开始预训练 Flow 策略")
    print(f"{'='*60}")
    print(f"使用物理GPU: {gpu_id}")
    print(f"观测模式: {obs_mode}")
    print(f"Loss 类型: {pretrain_loss_type}"
          + ("  (multi-step rollout MSE)" if pretrain_loss_type == "action"
             else "  (Flow Matching velocity MSE)"))
    print(f"数据集路径: {config.pretrain.dataset_path}")
    print(f"训练 episodes: {config.pretrain.num_episodes}")
    print(f"训练 epochs: {config.pretrain.num_epochs}")

    # ------------------------------------------------------------------
    # 1. 加载数据集
    # ------------------------------------------------------------------
    if obs_mode == 'pointcloud':
        from external.models.maniskill_dataset import ManiSkillZarrPointCloudDataset
        from external.models.pointcloud_encoder import PointCloudObsEncoder

        pcd_n_obs_steps    = getattr(config.pretrain, 'pcd_n_obs_steps', 2)
        pcd_enc_out_dim    = getattr(config.pretrain, 'pcd_encoder_output_dim', 256)
        pcd_agent_pos_dim  = getattr(config.pretrain, 'pcd_agent_pos_dim', 19)
        pcd_state_mlp_out  = getattr(config.pretrain, 'pcd_state_mlp_out', 64)
        pcd_use_layernorm  = getattr(config.pretrain, 'pcd_use_layernorm', True)
        pcd_final_norm     = getattr(config.pretrain, 'pcd_final_norm', 'layernorm')

        # pointcloud 模式优先使用 pcd_dataset_path，回退到 dataset_path
        zarr_path = getattr(config.pretrain, 'pcd_dataset_path',
                            config.pretrain.dataset_path)
        dataset = ManiSkillZarrPointCloudDataset(
            zarr_path=zarr_path,
            horizon=config.action_horizon,
            pad_before=0,
            pad_after=0,
            val_ratio=0.0,
            max_episodes=config.pretrain.num_episodes,
        )

        action_dim = dataset.replay_buffer['action'][0].shape[-1]
        # obs_cond 维度 = (pcd_enc_out_dim + pcd_state_mlp_out) * pcd_n_obs_steps
        obs_dim = (pcd_enc_out_dim + pcd_state_mlp_out) * pcd_n_obs_steps

        # 在线 FPS：从 dataset 自动探测（旧 zarr 没有 valid_count 则关闭）
        pcd_n_points  = getattr(config.pretrain, 'pcd_n_points', 512)
        pcd_online_fps = bool(getattr(dataset, 'online_fps', False))
        if pcd_online_fps and not _PCD_HAS_PYTORCH3D:
            print(f"  [警告] 在线 FPS 启用但 pytorch3d 未安装，"
                  f"将退化为 numpy 随机均匀采样（性能/质量略降）")

        # 创建归一化器（agent_pos + action → limits 归一化到 [-1, 1]）
        normalizer = dataset.get_normalizer().to(device)

        print(f"数据集信息:")
        print(f"  点云帧数 (n_obs_steps): {pcd_n_obs_steps}")
        print(f"  点云编码维度: {pcd_enc_out_dim}")
        print(f"  agent_pos 维度: {pcd_agent_pos_dim}")
        print(f"  obs_cond 维度 (global_cond_dim): {obs_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  样本数: {len(dataset)}")
        print(f"  在线 FPS: {'ON' if pcd_online_fps else 'OFF (使用 zarr 中固化的 FPS 子集)'}")
        if pcd_online_fps:
            print(f"  每帧 FPS 目标点数: {pcd_n_points}")
        print(f"  归一化器: agent_pos + action（limits 模式）")

    elif obs_mode == 'image':
        from external.models.maniskill_dataset import ManiSkillZarrImageDataset
        from external.models.image_encoder import ImageObsEncoder

        img_n_obs_steps     = getattr(config.pretrain, 'img_n_obs_steps', 2)
        img_camera          = getattr(config.pretrain, 'img_camera', 'base_camera')
        img_use_depth       = bool(getattr(config.pretrain, 'img_use_depth', True))
        img_depth_max       = float(getattr(config.pretrain, 'img_depth_max', 5.0))
        img_resnet_name     = getattr(config.pretrain, 'img_resnet_name', 'resnet18')
        img_resnet_feat_dim = int(getattr(config.pretrain, 'img_resnet_feat_dim', 512))
        img_agent_pos_dim   = int(getattr(config.pretrain, 'img_agent_pos_dim', 19))
        img_state_mlp_out   = int(getattr(config.pretrain, 'img_state_mlp_out', 64))
        img_use_groupnorm   = bool(getattr(config.pretrain, 'img_use_groupnorm', True))

        # image 模式优先使用 image_dataset_path，回退到 dataset_path
        img_zarr_path = getattr(config.pretrain, 'image_dataset_path',
                                config.pretrain.dataset_path)
        dataset = ManiSkillZarrImageDataset(
            zarr_path=img_zarr_path,
            horizon=config.action_horizon,
            pad_before=0,
            pad_after=0,
            val_ratio=0.0,
            max_episodes=config.pretrain.num_episodes,
            camera=img_camera,
            use_depth=img_use_depth,
            depth_max=img_depth_max,
        )

        img_in_channels = 4 if img_use_depth else 3
        action_dim = dataset.replay_buffer['action'][0].shape[-1]
        # obs_cond 维度 = (img_resnet_feat_dim + img_state_mlp_out) * img_n_obs_steps
        obs_dim = (img_resnet_feat_dim + img_state_mlp_out) * img_n_obs_steps

        # 归一化器（agent_pos + action → limits 归一化到 [-1, 1]，图像不进 normalizer）
        normalizer = dataset.get_normalizer().to(device)

        print(f"数据集信息:")
        print(f"  图像帧数 (n_obs_steps): {img_n_obs_steps}")
        print(f"  图像通道数: {img_in_channels}  ({'RGBD' if img_use_depth else 'RGB'})")
        print(f"  相机: {img_camera}")
        print(f"  ResNet 主干: {img_resnet_name}  feat_dim={img_resnet_feat_dim}")
        print(f"  agent_pos 维度: {img_agent_pos_dim}")
        print(f"  obs_cond 维度 (global_cond_dim): {obs_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  样本数: {len(dataset)}")
        print(f"  深度归一化 depth_max: {img_depth_max} m" if img_use_depth else "")
        print(f"  归一化器: agent_pos + action（limits 模式），图像不归一化")

    else:
        from external.models.maniskill_dataset import ManiSkillDataset

        dataset = ManiSkillDataset(
            dataset_dir=config.pretrain.dataset_path,
            horizon=config.action_horizon,
            pad_before=0,
            pad_after=0,
            val_ratio=0.0,
            max_episodes=config.pretrain.num_episodes,
            obs_key='obs'
        )

        obs_dim    = dataset.replay_buffer['obs'][0].shape[-1]
        action_dim = dataset.replay_buffer['action'][0].shape[-1]

        # state 模式：与 pointcloud/image 模式对齐，对 obs 和 action 做 limits 归一化
        normalizer = dataset.get_normalizer().to(device)

        print(f"数据集信息:")
        print(f"  观测维度: {obs_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  样本数: {len(dataset)}")
        print(f"  归一化器: obs + action（limits 模式）")

    dataloader = DataLoader(
        dataset,
        batch_size=config.pretrain.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # 2. 创建模型
    # ------------------------------------------------------------------
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
    ).to(device)

    action_decoder = ActionDecoder(
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        hidden_dim=256,
        num_layers=3,
    ).to(device)

    # 观测编码器：pointcloud / image 模式才需要（state 模式直接用扁平向量）
    if obs_mode == 'pointcloud':
        obs_encoder = PointCloudObsEncoder(
            n_obs_steps=pcd_n_obs_steps,
            pcd_out_dim=pcd_enc_out_dim,
            agent_pos_dim=pcd_agent_pos_dim,
            state_mlp_hidden=[64, pcd_state_mlp_out],
            use_layernorm=pcd_use_layernorm,
            final_norm=pcd_final_norm,
        ).to(device)
        print(f"模型参数数量:")
        print(f"  ObsEncoder 参数: {sum(p.numel() for p in obs_encoder.parameters())}")
        print(f"  UNet 参数:        {sum(p.numel() for p in action_unet.parameters())}")
        print(f"  ActionDecoder 参数: {sum(p.numel() for p in action_decoder.parameters())}")
    elif obs_mode == 'image':
        obs_encoder = ImageObsEncoder(
            n_obs_steps=img_n_obs_steps,
            in_channels=img_in_channels,
            resnet_name=img_resnet_name,
            img_feat_dim=img_resnet_feat_dim,
            agent_pos_dim=img_agent_pos_dim,
            state_mlp_hidden=[64, img_state_mlp_out],
            use_groupnorm=img_use_groupnorm,
        ).to(device)
        print(f"模型参数数量:")
        print(f"  ImageObsEncoder 参数: {sum(p.numel() for p in obs_encoder.parameters())}")
        print(f"  UNet 参数:        {sum(p.numel() for p in action_unet.parameters())}")
        print(f"  ActionDecoder 参数: {sum(p.numel() for p in action_decoder.parameters())}")
    else:
        obs_encoder = None
        print(f"模型参数数量:")
        print(f"  UNet 参数:        {sum(p.numel() for p in action_unet.parameters())}")
        print(f"  ActionDecoder 参数: {sum(p.numel() for p in action_decoder.parameters())}")

    # ------------------------------------------------------------------
    # 3. 创建管线
    # ------------------------------------------------------------------
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        device=str(device),
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder

    # ------------------------------------------------------------------
    # 4. 优化器
    # ------------------------------------------------------------------
    # velocity 模式：UNet 直接输出速度场，ActionDecoder 不参与训练，
    # 因此也不进 optimizer / EMA / grad clip，避免在零梯度下被
    # Adam 二阶动量 / EMA 累积无意义地扰动。
    decoder_trains_in_pretrain = (pretrain_loss_type != "velocity")

    trainable_params = list(action_unet.parameters())
    if decoder_trains_in_pretrain:
        trainable_params += list(action_decoder.parameters())
    if obs_encoder is not None:
        trainable_params += list(obs_encoder.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.pretrain.learning_rate,
        weight_decay=getattr(config.pretrain, 'weight_decay', 1e-6),
    )

    # ------------------------------------------------------------------
    # 5. 学习率调度器
    # ------------------------------------------------------------------
    if getattr(config.pretrain, 'use_lr_scheduler', True):
        warmup_steps = int(config.pretrain.warmup_epochs * len(dataloader))
        total_steps  = config.pretrain.num_epochs * len(dataloader)

        if config.pretrain.lr_scheduler_type == "cosine":
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            print(f"学习率调度器: Cosine Annealing")
            print(f"  Warmup steps: {warmup_steps} ({config.pretrain.warmup_epochs} epochs)")
            print(f"  Total steps: {total_steps}")
            print(f"  Min LR: {config.pretrain.min_lr}")
        else:
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
    else:
        lr_scheduler = None
        print("学习率调度器: 禁用")

    # ------------------------------------------------------------------
    # 6. EMA 模型
    # ------------------------------------------------------------------
    ema_decay        = getattr(config.pretrain, 'ema_decay', 0.995)
    ema_start_epoch  = getattr(config.pretrain, 'ema_start_epoch', 20)
    ema_update_freq  = getattr(config.pretrain, 'ema_update_freq', 10)

    ema_unet = EMAModel(
        action_unet.parameters(),
        decay=ema_decay,
        use_ema_warmup=False,
    )
    if decoder_trains_in_pretrain:
        ema_decoder = EMAModel(
            action_decoder.parameters(),
            decay=ema_decay,
            use_ema_warmup=False,
        )
    else:
        ema_decoder = None
    if obs_encoder is not None:
        ema_obs_encoder = EMAModel(
            obs_encoder.parameters(),
            decay=ema_decay,
            use_ema_warmup=False,
        )
    else:
        ema_obs_encoder = None

    print(f"\nEMA 配置:")
    print(f"  Decay: {ema_decay}")
    print(f"  Start epoch: {ema_start_epoch}")
    print(f"  Update freq: {ema_update_freq}")

    # ------------------------------------------------------------------
    # 7. 梯度包装函数（与原版相同）
    # ------------------------------------------------------------------
    def generate_with_grad(pipeline, obs, **kwargs):
        """确保 pipeline_with_logprob 在梯度启用状态下运行"""
        original = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        result = pipeline_with_logprob(pipeline, obs=obs, **kwargs)
        torch.set_grad_enabled(original)
        return result

    # ------------------------------------------------------------------
    # 8. 训练循环
    # ------------------------------------------------------------------
    print(f"\n开始训练...")

    auto_mode = config.pretrain.auto_mode if hasattr(config.pretrain, 'auto_mode') else False
    if auto_mode:
        eval_interval     = config.pretrain.auto_eval_interval
        num_test_episodes = config.pretrain.auto_num_test_episodes
        success_threshold = config.pretrain.auto_success_threshold
        max_epochs        = config.pretrain.auto_max_epochs
        print(f"[Auto模式] 每 {eval_interval} epochs 评估一次")
        print(f"[Auto模式] 每次评估 {num_test_episodes} 个episodes")
        print(f"[Auto模式] 成功阈值: {success_threshold} 次")
        print(f"[Auto模式] 最大训练轮数: {max_epochs}")
    else:
        max_epochs = config.pretrain.num_epochs
        print(f"[固定模式] 训练 {max_epochs} epochs")

    action_unet.train()
    action_decoder.train()
    if obs_encoder is not None:
        obs_encoder.train()

    global_step     = 0
    training_stopped = False
    final_epoch     = 0
    avg_loss        = 0.0

    for epoch in range(max_epochs):
        epoch_loss   = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 所有模式统一归一化 action（limits → [-1, 1]）
            action_seq = normalizer['action'].normalize(batch['action'].to(device))
            target_actions = action_seq

            # ---------- 计算 obs_cond ----------
            if obs_mode == 'pointcloud':
                # point_cloud: 旧 zarr → (B, H, n_pts=512, 3) 已 FPS
                #              新 zarr → (B, H, max_pts=4096, 3) 未 FPS，含 valid_count
                pcd      = batch['point_cloud'].to(device, non_blocking=True)   # (B, H, *, 3)
                apos_raw = batch['agent_pos'].to(device, non_blocking=True)     # (B, H, A)
                apos     = normalizer['agent_pos'].normalize(apos_raw)          # 归一化 agent_pos

                pcd_obs  = pcd[:, :pcd_n_obs_steps]                          # (B, To, *, 3)
                apos_in  = apos[:, :pcd_n_obs_steps]                         # (B, To, A)

                if pcd_online_fps and 'valid_count' in batch:
                    # 在线 GPU FPS：每个 batch / 每次访问都得到不同的 FPS 子集
                    B_, To_, M_, _ = pcd_obs.shape
                    valid = batch['valid_count'].to(device, non_blocking=True).long()  # (B, H)
                    valid_obs = valid[:, :pcd_n_obs_steps]                              # (B, To)

                    pcd_flat   = pcd_obs.reshape(B_ * To_, M_, 3)
                    valid_flat = valid_obs.reshape(B_ * To_)

                    sampled = batched_fps_sample(pcd_flat, valid_flat, pcd_n_points)
                    pcd_in  = sampled.reshape(B_, To_, pcd_n_points, 3)
                else:
                    pcd_in = pcd_obs

                obs_cond = obs_encoder(pcd_in, apos_in)  # (B, obs_dim)
            elif obs_mode == 'image':
                # image: (B, H, C, H_img, W_img)，已在数据集中归一到 [0, 1]
                img      = batch['image'].to(device, non_blocking=True)          # (B, H, C, H, W)
                apos_raw = batch['agent_pos'].to(device, non_blocking=True)      # (B, H, A)
                apos     = normalizer['agent_pos'].normalize(apos_raw)           # 归一化 agent_pos

                img_in  = img [:, :img_n_obs_steps]                              # (B, To, C, H, W)
                apos_in = apos[:, :img_n_obs_steps]                              # (B, To, A)

                obs_cond = obs_encoder(img_in, apos_in)  # (B, obs_dim)
            else:
                # state 模式：对 obs 做 limits 归一化后取第 0 帧作为 obs_cond
                obs_seq  = batch['obs'].to(device)                       # (B, H, obs_dim)
                obs_norm = normalizer['obs'].normalize(obs_seq)          # 归一化到 [-1, 1]
                obs_cond = obs_norm[:, 0, :]                             # (B, obs_dim)

            batch_size = obs_cond.shape[0]

            # ---------- 按 loss_type 派发：action（多步 rollout MSE）/ velocity（CFM）----------
            if pretrain_loss_type == "velocity":
                # Flow Matching 速度场监督（对齐 FlowPolicy：UNet 直出速度，不经过 ActionDecoder）
                #   x0 ~ N(0, I), t ~ U[0,1]
                #   xt = (1-t)·x0 + t·x1，    ut = x1 - x0
                #   loss = MSE(action_unet(xt, t_input, obs_cond), ut)
                # 说明：ConditionalUnet1D.final_conv 已经输出 (B, H, action_dim) 形状，
                # 即速度场本身。先前在此处再过一遍 ActionDecoder（含 Dropout+ReLU+全局 MLP）
                # 会破坏 UNet 的逐位置时间结构、并通过 Dropout 给输出注入约 ~0.1 量级
                # 的随机方差，导致 MSE loss 卡在 0.1 难以下降。
                x1 = target_actions.float()
                x0 = torch.randn_like(x1)
                t = torch.rand(batch_size, device=device, dtype=torch.float32)
                t_pad = t.view(batch_size, 1, 1)
                xt = (1.0 - t_pad) * x0 + t_pad * x1
                ut = x1 - x0

                # 与 RL/test 共用的 SinusoidalPosEmb 时间编码（_unet_timestep_input）
                t_input = _unet_timestep_input(t)
                vt = action_unet(
                    sample=xt,
                    timestep=t_input,
                    global_cond=obs_cond,
                )

                loss = torch.nn.functional.mse_loss(vt, ut)
            else:
                # ---------- 全流程去噪 → pred_actions（原有 action_loss）----------
                result = generate_with_grad(
                    pipeline,
                    obs=obs_cond,
                    num_inference_steps=config.sample.num_steps,
                    output_type="tensor",
                    return_dict=True,
                    latents=None,
                    action_decoder=action_decoder,
                )
                pred_actions = result["actions"]

                # MSE 损失（预测动作 vs 真实动作）
                loss = torch.nn.functional.mse_loss(pred_actions, target_actions)

            # ---------- 反向传播 ----------
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(action_unet.parameters(), max_norm=1.0)
            if decoder_trains_in_pretrain:
                torch.nn.utils.clip_grad_norm_(action_decoder.parameters(), max_norm=1.0)
            if obs_encoder is not None:
                torch.nn.utils.clip_grad_norm_(obs_encoder.parameters(), max_norm=1.0)

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # ---------- EMA 更新 ----------
            if epoch >= ema_start_epoch:
                ema_unet.step(action_unet.parameters())
                if ema_decoder is not None:
                    ema_decoder.step(action_decoder.parameters())
                if ema_obs_encoder is not None:
                    ema_obs_encoder.step(obs_encoder.parameters())

            epoch_loss  += loss.item()
            global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR':   f'{current_lr:.2e}',
                'Step': global_step,
            })

        avg_loss   = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        ema_status = "Active" if epoch >= ema_start_epoch else f"Inactive (starts at epoch {ema_start_epoch})"
        print(f"Epoch {epoch+1}/{max_epochs} 完成:")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  当前学习率: {current_lr:.2e}")
        print(f"  EMA状态: {ema_status}")

        final_epoch = epoch + 1

        # Auto 模式：定期评估（state 与 pointcloud 模式均支持）
        if auto_mode and (epoch + 1) % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"[Auto模式评估] Epoch {epoch+1}: 开始评估...")
            print(f"{'='*60}")

            ema_unet.copy_to(action_unet.parameters())
            if ema_decoder is not None:
                ema_decoder.copy_to(action_decoder.parameters())
            if ema_obs_encoder is not None:
                ema_obs_encoder.copy_to(obs_encoder.parameters())

            # 所有模式都传 normalizer（state 模式现在也需要用来归一化 obs / 反归一化 action）
            _eval_normalizer = normalizer
            # velocity 模式：decoder 未参与训练，评估时跳过 decoder（pipeline_with_logprob 在
            # action_decoder=None 时会直接把 UNet 输出当作速度场使用，与训练严格对齐）
            _eval_decoder = action_decoder if decoder_trains_in_pretrain else None
            success_count, total_count, avg_reward = evaluate_pretrain_model(
                action_unet, _eval_decoder, pipeline, config, device,
                num_test_episodes, obs_encoder=obs_encoder,
                normalizer=_eval_normalizer,
                obs_mode=obs_mode,
            )

            print(f"[Auto模式评估] 成功率: {success_count}/{total_count}")
            print(f"[Auto模式评估] 平均奖励: {avg_reward:.3f}")

            if success_count >= success_threshold:
                print(f"\n{'='*60}")
                print(f"[Auto模式] 达到成功阈值！停止训练于 epoch {epoch+1}")
                print(f"{'='*60}\n")
                training_stopped = True
                break
            else:
                print(f"[Auto模式] 未达到阈值，继续训练...\n")
                action_unet.train()
                if decoder_trains_in_pretrain:
                    action_decoder.train()

    if auto_mode and not training_stopped:
        print(f"\n[Auto模式] 达到最大训练轮数 {max_epochs}，停止训练")

    # ------------------------------------------------------------------
    # 9. 保存检查点（使用 EMA 权重）
    # ------------------------------------------------------------------
    print(f"\n训练完成，准备保存模型...")

    ema_unet.copy_to(action_unet.parameters())
    if ema_decoder is not None:
        ema_decoder.copy_to(action_decoder.parameters())
    if ema_obs_encoder is not None:
        ema_obs_encoder.copy_to(obs_encoder.parameters())

    os.makedirs(config.pretrain.checkpoint_dir, exist_ok=True)

    obs_mode_tag = f"_{obs_mode}"
    if auto_mode:
        checkpoint_name = f"pretrain_auto_epoch{final_epoch}_episodes{config.pretrain.num_episodes}{obs_mode_tag}.pth"
    else:
        checkpoint_name = f"pretrain_epoch{config.pretrain.num_epochs}_episodes{config.pretrain.num_episodes}{obs_mode_tag}.pth"

    checkpoint_path = os.path.join(config.pretrain.checkpoint_dir, checkpoint_name)

    ckpt = {
        'epoch':                      final_epoch - 1,
        'obs_mode':                   obs_mode,
        'pretrain_loss_type':         pretrain_loss_type,
        # velocity 模式：UNet 直出速度场，ActionDecoder 未训练，下游 RL/eval
        # 加载时应据此跳过 decoder（pipeline_with_logprob 已支持 action_decoder=None）
        'decoder_trained':            decoder_trains_in_pretrain,
        'model_state_dict':           action_unet.state_dict(),
        'action_unet_state_dict':     action_unet.state_dict(),   # 向后兼容
        'action_decoder_state_dict':  action_decoder.state_dict(),
        'optimizer_state_dict':       optimizer.state_dict(),
        'lr_scheduler_state_dict':    lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'ema_unet_state_dict':        ema_unet.state_dict(),
        'ema_decoder_state_dict':     ema_decoder.state_dict() if ema_decoder is not None else None,
        'loss':                       avg_loss,
        'config':                     config.to_dict(),
        'obs_dim':                    obs_dim,
        'action_dim':                 action_dim,
        'auto_mode':                  auto_mode,
        'training_stopped':           training_stopped if auto_mode else False,
    }

    # 所有模式都保存归一化器（state 模式同样需要，用于推理时归一化 obs / 反归一化 action）
    ckpt['normalizer_state_dict'] = normalizer.state_dict()

    # pointcloud / image 模式额外保存 obs_encoder
    if obs_encoder is not None:
        ckpt['obs_encoder_state_dict'] = obs_encoder.state_dict()
        ckpt['ema_obs_encoder_state_dict'] = ema_obs_encoder.state_dict()

        if obs_mode == 'pointcloud':
            ckpt['pcd_encoder_config'] = {
                'n_obs_steps':   pcd_n_obs_steps,
                'pcd_out_dim':   pcd_enc_out_dim,
                'agent_pos_dim': pcd_agent_pos_dim,
                'state_mlp_out': pcd_state_mlp_out,
                'use_layernorm': pcd_use_layernorm,
                'final_norm':    pcd_final_norm,
            }
        elif obs_mode == 'image':
            ckpt['img_encoder_config'] = {
                'n_obs_steps':    img_n_obs_steps,
                'in_channels':    img_in_channels,
                'resnet_name':    img_resnet_name,
                'img_feat_dim':   img_resnet_feat_dim,
                'agent_pos_dim':  img_agent_pos_dim,
                'state_mlp_out':  img_state_mlp_out,
                'use_groupnorm':  img_use_groupnorm,
                'camera':         img_camera,
                'use_depth':      img_use_depth,
                'depth_max':      img_depth_max,
            }

    torch.save(ckpt, checkpoint_path)

    print(f"\n预训练完成！")
    print(f"检查点已保存: {checkpoint_path}")
    if auto_mode:
        print(f"  训练模式: Auto (自动停止)")
        print(f"  实际训练轮数: {final_epoch} epochs")
        print(f"  停止原因: {'达到成功阈值' if training_stopped else '达到最大训练轮数'}")
    else:
        print(f"  训练模式: 固定 ({final_epoch} epochs)")
    print(f"  观测模式: {obs_mode}")
    print(f"  使用 episodes: {config.pretrain.num_episodes}")
    print(f"  包含 EMA 权重: ✓")
    print(f"  包含优化器状态: ✓")
    print(f"  包含学习率调度器: ✓")
    if obs_encoder is not None:
        if obs_mode == 'image':
            print(f"  包含图像编码器权重: ✓")
            print(f"  包含 img_encoder_config: ✓")
        else:
            print(f"  包含点云编码器权重: ✓")
        print(f"  包含归一化器: ✓")
    print(f"  最终损失: {avg_loss:.6f}")
    print(f"{'='*60}\n")
    
    return checkpoint_path


def main(_):
    # 基本设置
    config = FLAGS.config
    
    # GPU 已经在文件开头设置了，这里只是确认
    physical_gpu_id = FLAGS.gpu
    print(f"\n{'='*70}")
    print(f"[main 函数] 确认GPU设置:")
    print(f"  命令行指定的物理GPU: {physical_gpu_id}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # 在导入torch后立即检查
    if torch.cuda.is_available():
        print(f"  PyTorch 检测到的GPU数量: {torch.cuda.device_count()}")
        print(f"  PyTorch 当前设备: cuda:{torch.cuda.current_device()}")
        print(f"  预期映射: 物理GPU{physical_gpu_id} → PyTorch cuda:0")
    else:
        print(f"  警告: PyTorch 未检测到CUDA!")
    print(f"{'='*70}\n")
    
    # 测试模式
    if FLAGS.test:
        if not FLAGS.checkpoint:
            raise ValueError("测试模式需要指定 --checkpoint 参数")
        test_model(config, FLAGS.checkpoint, FLAGS.num_test_episodes, physical_gpu_id)
        return
    
    # 采样测试模式
    if FLAGS.sampletest:
        if not FLAGS.pretrain_checkpoint:
            raise ValueError("采样测试模式需要指定 --pretrain_checkpoint 参数")
        test_sample(config, FLAGS.pretrain_checkpoint, physical_gpu_id)
        return
    
    # 预训练模式
    if FLAGS.pretrain:
        # 如果指定了 --pretrain_auto，启用auto模式
        if FLAGS.pretrain_auto:
            config.pretrain.auto_mode = True
            print(f"\n{'='*70}")
            print(f"[预训练] 启用Auto模式（自动停止）")
            print(f"{'='*70}\n")

        # 解析 --action_loss / --velocity_loss 互斥 flag，覆盖 config.pretrain.loss_type
        if FLAGS.action_loss and FLAGS.velocity_loss:
            raise ValueError(
                "--action_loss 与 --velocity_loss 不能同时指定，请只选其一"
            )
        if FLAGS.action_loss:
            config.pretrain.loss_type = "action"
            print(f"[预训练] 命令行覆盖 loss_type = 'action'")
        elif FLAGS.velocity_loss:
            config.pretrain.loss_type = "velocity"
            print(f"[预训练] 命令行覆盖 loss_type = 'velocity'")
        else:
            print(f"[预训练] 使用 config 默认 loss_type = "
                  f"'{getattr(config.pretrain, 'loss_type', 'action')}'")

        pretrain_model(config, physical_gpu_id)
        return
    
    # 训练模式
    if not FLAGS.train:
        logger.info("既没有指定 --train 也没有指定 --test，默认执行训练")
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
    
    # 计算训练时间步数
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    
    # 配置 Accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    
    # 注意：我们在train_on_samples中手动处理梯度累积（对num_train_timesteps个timestep）
    # 所以这里不需要额外的gradient_accumulation_steps
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=1,  # 设为1，梯度累积在train_on_samples中手动处理
    )
    
    # 初始化wandb
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="flow-diffusionNFT",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")
    logger.info(f"物理 GPU: {physical_gpu_id} → 虚拟设备: cuda:0 (通过CUDA_VISIBLE_DEVICES)")
    logger.info(f"Accelerator 设备: {accelerator.device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA 可用: {torch.cuda.device_count()} 个设备, 当前设备: cuda:{torch.cuda.current_device()}")
    else:
        logger.info("CUDA 不可用")
    
    # 设置随机种子
    set_seed(config.seed, device_specific=True)
    
    # 创建环境（用于采样和评估）
    env_kwargs = {
        'num_envs': config.episode.parallel_envs,  # 64个环境（episode-level采样）
        'sim_backend': config.env_kwargs.get('sim_backend', 'physx_cpu'),
        'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode': config.env_kwargs.get('obs_mode', 'state'),
        'render_mode': config.env_kwargs.get('render_mode', 'rgb_array'),
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense')
    }
    
    # ✅ 关键修改：禁用 auto_reset（参考RLinf的GRPO配置）
    # reconfiguration_freq=None 禁用GPU环境的自动重置
    # 这确保每条轨迹对应一个完整的episode（从reset到done）
    # 满足GRPO的核心假设：组内样本从相同初始状态出发
    
    logger.info(f"创建训练环境: {config.env_id} (Episode-level采样: {env_kwargs['num_envs']} 个环境)")
    logger.info(f"GPU模拟后端: {env_kwargs['sim_backend']}")
    logger.info(f"GRPO训练模式: 禁用auto_reset，手动控制完整轨迹采样")
    logger.info(f"GRPO配置: {config.episode.num_state_groups}组 × {config.episode.samples_per_group}样本 = {config.episode.parallel_envs}个并行环境")
    env = gym.make(config.env_id, reconfiguration_freq=None, **env_kwargs)  # 禁用auto_reset
    
    # 验证环境是否使用GPU
    logger.info(f"\n{'='*60}")
    logger.info(f"验证GPU设置和环境创建:")
    logger.info(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"  物理 GPU {physical_gpu_id} → PyTorch 看到的设备: cuda:0")
    logger.info(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")
    logger.info(f"  torch.cuda.current_device() = {torch.cuda.current_device()}")
    
    test_obs, _ = env.reset(seed=config.seed)
    if isinstance(test_obs, torch.Tensor):
        logger.info(f"  ✓ 环境观测数据在设备: {test_obs.device} (类型: {test_obs.dtype})")
        # 检查物理GPU使用情况
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], 
                                   capture_output=True, text=True, timeout=2)
            logger.info(f"  当前进程 PID: {os.getpid()}")
            logger.info(f"  GPU 使用情况:\n{result.stdout if result.stdout else '无法获取'}")
        except:
            pass
    else:
        logger.warning(f"  ⚠ 环境观测数据不是torch.Tensor，GPU加速可能未启用")
    logger.info(f"{'='*60}\n")
    env.reset(seed=config.seed)  # 重新reset一次
    
    # 创建单独的评估环境（单个环境实例，也使用GPU以保持一致性）
    eval_env_kwargs = {
        'sim_backend': config.env_kwargs.get('sim_backend', 'physx_cpu'),
        'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode': config.env_kwargs.get('obs_mode', 'state'),
        'render_mode': config.env_kwargs.get('render_mode', 'rgb_array'),
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense')
    }
    logger.info(f"创建评估环境: {config.env_id} (单个环境, 后端: {eval_env_kwargs['sim_backend']})")
    # 评估环境可以启用auto_reset（每个episode独立评估，无需组内比较）
    eval_env = gym.make(config.env_id, reconfiguration_freq=1, **eval_env_kwargs)
    
    # 从环境获取维度信息
    obs, _ = env.reset(seed=config.seed)
    if isinstance(obs, torch.Tensor):
        obs_dim = obs.shape[-1]
    else:
        obs_dim = obs.shape[-1] if hasattr(obs, 'shape') else len(obs)
    
    # 对于批量环境，需要获取单个环境的动作维度
    # ManiSkill 批量环境的 action_space.shape 是 (num_envs, action_dim)
    if hasattr(env, 'single_action_space'):
        action_dim = env.single_action_space.shape[0]
    elif len(env.action_space.shape) > 1:
        # 批量环境：shape = (num_envs, action_dim)
        action_dim = env.action_space.shape[1]
    else:
        # 单个环境：shape = (action_dim,)
        action_dim = env.action_space.shape[0]
    
    logger.info(f"环境信息:")
    logger.info(f"  观测维度: {obs_dim}")
    logger.info(f"  动作维度: {action_dim}")
    logger.info(f"  动作序列长度: {config.action_horizon}")
    
    # 创建模型
    logger.info("初始化模型...")
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(accelerator.device)
    
    # 创建 ActionDecoder
    action_decoder = ActionDecoder(
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        hidden_dim=256,
        num_layers=3
    ).to(accelerator.device)
    
    # 创建管线
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        device=str(accelerator.device)
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder
    
    logger.info(f"模型初始化完成:")
    logger.info(f"  UNet参数数量: {sum(p.numel() for p in action_unet.parameters())}")
    logger.info(f"  ActionDecoder参数数量: {sum(p.numel() for p in action_decoder.parameters())}")
    
    # 获取可训练参数
    trainable_parameters = list(action_unet.parameters()) + list(action_decoder.parameters())
    
    # 创建 EMA
    ema_decay = getattr(config.train, 'ema_decay', 0.9)
    ema = EMAModuleWrapper(
        trainable_parameters,
        decay=ema_decay,
        update_step_interval=8,
        device=accelerator.device
    )
    
    # 启用 TF32
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # 初始化优化器
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam")
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
    # 准备模型和优化器
    action_unet, action_decoder, optimizer = accelerator.prepare(action_unet, action_decoder, optimizer)
    
    # === 创建学习率调度器（参照ReinFlow）===
    if getattr(config.train, 'use_lr_scheduler', False):
        from diffusers.optimization import get_scheduler
        
        # ✅ ReinFlow风格：直接使用epochs作为调度步数
        # 每个epoch结束调用一次lr_scheduler.step()
        total_training_steps = config.episode.num_epochs  # 2000 epochs = 2000次step()调用
        warmup_steps = getattr(config.train, 'warmup_epochs', 100)  # 100 epochs = 100次step()调用
        num_cycles = getattr(config.train, 'num_cycles', 0.5)
        
        lr_scheduler = get_scheduler(
            config.train.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=num_cycles,
        )
        
        # ✅ 修复：在训练开始前先step一次，使LR从0变为warmup的第一个有效值
        # 这避免了第一个epoch用LR=0训练的问题
        lr_scheduler.step()
        initial_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"学习率调度器配置 (ReinFlow风格：per-epoch调度)")
        logger.info(f"{'='*70}")
        logger.info(f"  类型: {config.train.lr_scheduler_type}")
        logger.info(f"  调度频率: 每个epoch调用一次step()（非per-batch）")
        logger.info(f"  Warmup epochs: {warmup_steps} ({warmup_steps/total_training_steps*100:.1f}%)")
        logger.info(f"  Total epochs: {total_training_steps}")
        logger.info(f"  Num cycles: {num_cycles}")
        logger.info(f"  Target LR: {config.train.learning_rate:.2e}")
        logger.info(f"  Min LR: {getattr(config.train, 'min_lr', config.train.learning_rate/10):.2e}")
        logger.info(f"  ✓ 初始化后的起始LR: {initial_lr:.2e}")
        logger.info(f"")
        logger.info(f"  预期学习率曲线:")
        logger.info(f"    Epoch 1: LR={initial_lr:.2e} (warmup第1步)")
        logger.info(f"    Epoch {warmup_steps}: LR达到{config.train.learning_rate:.2e} (warmup结束)")
        logger.info(f"    Epoch {warmup_steps+1}-{total_training_steps}: LR按cosine衰减到{getattr(config.train, 'min_lr', config.train.learning_rate/10):.2e}")
        logger.info(f"{'='*70}\n")
    else:
        lr_scheduler = None
        logger.info("学习率调度器: 禁用（使用固定学习率）\n")
    
    # === 加载预训练检查点 ===
    pretrain_checkpoint = None
    if FLAGS.pretrain_checkpoint:
        pretrain_checkpoint = FLAGS.pretrain_checkpoint
    else:
        # 自动查找最新的预训练检查点
        pretrain_dir = config.pretrain.checkpoint_dir if hasattr(config, 'pretrain') else "checkpoints/imitation"
        if os.path.exists(pretrain_dir):
            checkpoints = sorted([f for f in os.listdir(pretrain_dir) if f.endswith('.pth')])
            if checkpoints:
                pretrain_checkpoint = os.path.join(pretrain_dir, checkpoints[-1])
    
    if pretrain_checkpoint and os.path.exists(pretrain_checkpoint):
        logger.info(f"加载预训练检查点: {pretrain_checkpoint}")
        checkpoint = torch.load(pretrain_checkpoint, map_location=accelerator.device)
        
        # 只加载模型权重，不加载优化器
        unwrap_model(action_unet, accelerator).load_state_dict(checkpoint['action_unet_state_dict'])
        unwrap_model(action_decoder, accelerator).load_state_dict(checkpoint['action_decoder_state_dict'])
        
        logger.info("✓ 预训练权重加载成功")
    else:
        logger.info("未找到预训练检查点，从随机初始化开始训练")
    
    # ✅ 修复：EMA 在加载预训练之前用随机初始化权重创建过快照，需要在 pretrain 加载后
    #    重新对 trainable_parameters 做一次 EMA 快照，否则前几百个 step 的 EMA 仍然
    #    混着随机初始化的权重，导致中早期 checkpoint 不可用。
    if config.train.ema:
        ema.ema_parameters = [
            p.clone().detach().to(accelerator.device) for p in trainable_parameters
        ]
        logger.info("✓ EMA 已重新对齐到加载预训练后的权重")
    
    # ========== DiffusionNFT: 初始化三模型架构（在加载预训练权重之后） ==========
    import copy
    logger.info("初始化三模型架构（current/old/ref）...")
    
    # Old model（用于policy gradient计算）
    action_unet_old = copy.deepcopy(unwrap_model(action_unet, accelerator))
    action_decoder_old = copy.deepcopy(unwrap_model(action_decoder, accelerator))
    action_unet_old.to(accelerator.device)
    action_decoder_old.to(accelerator.device)
    action_unet_old.eval()
    action_decoder_old.eval()
    
    # Reference model（用于KL regularization）
    action_unet_ref = copy.deepcopy(unwrap_model(action_unet, accelerator))
    action_decoder_ref = copy.deepcopy(unwrap_model(action_decoder, accelerator))
    action_unet_ref.to(accelerator.device)
    action_decoder_ref.to(accelerator.device)
    
    # 冻结reference model
    for param in action_unet_ref.parameters():
        param.requires_grad = False
    for param in action_decoder_ref.parameters():
        param.requires_grad = False
    action_unet_ref.eval()
    action_decoder_ref.eval()
    
    logger.info("✓ 三模型架构初始化完成（使用预训练权重）")
    
    # autocast 设置
    autocast = accelerator.autocast
    
    logger.info("***** 开始Episode-level GRPO训练 *****")
    logger.info(f"  最大episode步数 = {config.max_episode_steps}")
    logger.info(f"  动作序列长度 = {config.action_horizon}")
    logger.info(f"  训练模式 = Episode-level GRPO")
    logger.info(f"  总epoch数 = {config.episode.num_epochs}")
    logger.info(f"  每epoch样本数 = {config.episode.num_state_groups * config.episode.samples_per_group}")
    logger.info(f"  GRPO分组 = {config.episode.num_state_groups}组 × {config.episode.samples_per_group}样本")
    logger.info(f"  Timestep权重范围 = [{config.stage.timestep_weight_start}, {config.stage.timestep_weight_end}]")
    logger.info(f"  Advantage放大系数 = {getattr(config.stage, 'advantage_scale', 1.0)}")
    logger.info(f"  训练批量大小 = {config.train.batch_size}")
    logger.info(f"  梯度累积步数 = {config.train.gradient_accumulation_steps}")
    logger.info(f"  有效批量大小 = {config.train.batch_size * config.train.gradient_accumulation_steps}")
    logger.info(f"  初始学习率 = {config.train.learning_rate:.2e}")
    if lr_scheduler is not None:
        logger.info(f"  学习率调度 = {config.train.lr_scheduler_type} (warmup={config.train.warmup_epochs} epochs)")
    else:
        logger.info(f"  学习率调度 = 禁用（固定学习率）")
    
    logger.info(f"  Clip范围 = {config.train.clip_range}")
    logger.info(f"  EMA Decay = {getattr(config.train, 'ema_decay', 0.9)}")
    logger.info(f"  保存频率 = 每{config.save_freq}个epoch")
    logger.info("")
    logger.info("***** GPU优化状态 *****")
    logger.info(f"  模型设备: {accelerator.device}")
    logger.info(f"  混合精度: {config.mixed_precision}")
    logger.info(f"  ManiSkill后端: {config.env_kwargs.get('sim_backend', 'unknown')}")
    logger.info(f"  GPU并行环境数: {config.episode.parallel_envs}")
    logger.info("")
    
    if config.resume_from:
        logger.info(f"从检查点恢复: {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    
    global_step = 0
    
    # 获取训练参数
    rollout_epoch = getattr(config.episode, 'rollout_epoch', 1)
    update_epoch = getattr(config.episode, 'update_epoch', 1)
    
    # 主训练循环：简单的epoch循环
    for epoch in range(first_epoch, config.episode.num_epochs):
        # 显示当前学习率和调度状态
        current_lr = optimizer.param_groups[0]['lr']
        warmup_epochs = getattr(config.train, 'warmup_epochs', 100) if lr_scheduler else 0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch+1}/{config.episode.num_epochs}")
        if lr_scheduler and epoch < warmup_epochs:
            logger.info(f"  当前学习率: {current_lr:.2e} [Warmup阶段 {epoch+1}/{warmup_epochs}]")
        else:
            logger.info(f"  当前学习率: {current_lr:.2e}")
        logger.info(f"  Rollout次数: {rollout_epoch} | 更新轮数: {update_epoch}")
        logger.info(f"{'='*70}")
        
        # 1. 多次rollout采样完整轨迹（对齐RLinf）
        all_samples = []
        for rollout_idx in range(rollout_epoch):
            logger.info(f"\n  --- Rollout {rollout_idx+1}/{rollout_epoch} ---")
            
            samples = sample_complete_trajectories(
                env=env,
                pipeline=pipeline,
                action_unet=action_unet,
                action_decoder=action_decoder,
                num_state_groups=config.episode.num_state_groups,
                samples_per_group=config.episode.samples_per_group,
                config=config,
                accelerator=accelerator,
                autocast=autocast,
                seed_offset=config.seed + epoch * 100000 + rollout_idx * 10000,
                epoch=epoch  # 传入epoch用于噪声退火
            )
            
            if len(samples) == 0:
                logger.warning(f"  Rollout {rollout_idx+1}: 没有有效样本，跳过")
                continue
            
            # 统计单次rollout信息
            success_count = sum(1 for s in samples if s["terminated"])
            avg_reward = np.mean([s["reward"] for s in samples])
            
            logger.info(f"  采样: {len(samples)}条轨迹, 成功率: {success_count/len(samples)*100:.1f}%, 平均奖励: {avg_reward:.3f}")
            
            all_samples.extend(samples)
        
        if len(all_samples) == 0:
            logger.warning(f"Epoch {epoch}: 所有rollout都没有有效样本，跳过")
            continue
        
        # 统计所有rollout的总体信息
        total_success_count = sum(1 for s in all_samples if s["terminated"])
        total_avg_reward = np.mean([s["reward"] for s in all_samples])
        total_avg_steps = np.mean([s["num_steps"] for s in all_samples])
        total_avg_flow_calls = np.mean([s["num_flow_calls"] for s in all_samples])
        
        logger.info(f"\n  ===== 本Epoch采样汇总 =====")
        logger.info(f"  总轨迹数: {len(all_samples)}条 ({rollout_epoch} rollouts × ~{len(all_samples)//rollout_epoch}条)")
        logger.info(f"  总成功率: {total_success_count}/{len(all_samples)} ({total_success_count/len(all_samples)*100:.1f}%)")
        logger.info(f"  平均奖励: {total_avg_reward:.3f}")
        logger.info(f"  平均步数: {total_avg_steps:.1f}")
        logger.info(f"  平均Flow调用: {total_avg_flow_calls:.1f}")
        
        # ===== 监控指标1: 采样质量统计 =====
        all_rewards = np.array([s["reward"] for s in all_samples])
        all_steps = np.array([s["num_steps"] for s in all_samples])
        reward_std = all_rewards.std()
        steps_median = np.median(all_steps)
        
        # 统计异常轨迹（步数过短或过长）
        short_traj_ratio = (all_steps < 50).sum() / len(all_steps)  # 少于50步
        long_traj_ratio = (all_steps > 250).sum() / len(all_steps)  # 超过250步
        
        logger.info(f"  奖励标准差: {reward_std:.3f}")
        logger.info(f"  步数中位数: {steps_median:.1f}")
        logger.info(f"  异常短轨迹比例: {short_traj_ratio:.2%}")
        logger.info(f"  异常长轨迹比例: {long_traj_ratio:.2%}")
        
        # 2. 计算advantages（包含Success Rate Filter）
        num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
        all_samples, filter_stats = compute_episode_level_advantages(all_samples, num_train_timesteps, config)
        
        # ===== 监控指标2: Advantage统计 =====
        # 收集所有未被过滤样本的advantage
        valid_advantages = [s["advantage"] for s in all_samples if not s.get("filtered", False)]
        
        if len(valid_advantages) > 0:
            adv_array = np.array(valid_advantages)
            adv_mean = adv_array.mean()
            adv_std = adv_array.std()
            adv_min = adv_array.min()
            adv_max = adv_array.max()
            positive_adv_ratio = (adv_array > 0).sum() / len(adv_array)
            
            logger.info(f"\n  ===== Advantage统计 =====")
            logger.info(f"  有效样本数: {len(valid_advantages)}/{len(all_samples)}")
            logger.info(f"  Advantage: mean={adv_mean:.3f}, std={adv_std:.3f}")
            logger.info(f"  Advantage: min={adv_min:.3f}, max={adv_max:.3f}")
            logger.info(f"  正Advantage比例: {positive_adv_ratio:.2%}")
        else:
            adv_mean = adv_std = adv_min = adv_max = 0.0
            positive_adv_ratio = 0.0
            logger.info(f"\n  ===== Advantage统计 =====")
            logger.info(f"  警告: 所有样本都被过滤，没有有效的advantage")
        
        # 过滤组统计
        filtered_groups_ratio = filter_stats["filtered_groups"] / filter_stats["total_groups"] if filter_stats["total_groups"] > 0 else 0.0
        logger.info(f"  过滤组比例: {filtered_groups_ratio:.2%} ({filter_stats['filtered_groups']}/{filter_stats['total_groups']})")
        logger.info(f"  过滤样本: {filter_stats['filtered_samples']}/{filter_stats['total_samples']}")
        
        # ===== 记录采样和Advantage指标到wandb =====
        if accelerator.is_main_process:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    # 采样质量指标
                    "sample/success_rate": total_success_count / len(all_samples) * 100,
                    "sample/success_count": total_success_count,
                    "sample/total_trajectories": len(all_samples),
                    "sample/avg_steps": total_avg_steps,
                    "sample/median_steps": steps_median,
                    "sample/std_steps": all_steps.std(),
                    "sample/short_traj_ratio": short_traj_ratio,
                    "sample/long_traj_ratio": long_traj_ratio,
                    "sample/avg_flow_calls": total_avg_flow_calls,
                    "sample/reward_mean": total_avg_reward,
                    "sample/reward_std": reward_std,
                    "sample/reward_min": all_rewards.min(),
                    "sample/reward_max": all_rewards.max(),
                    # Advantage指标
                    "advantage/mean": adv_mean,
                    "advantage/std": adv_std,
                    "advantage/min": adv_min,
                    "advantage/max": adv_max,
                    "advantage/positive_ratio": positive_adv_ratio,
                    "advantage/valid_samples": len(valid_advantages),
                    # 过滤统计
                    "filter/filtered_groups_ratio": filtered_groups_ratio,
                    "filter/filtered_samples": filter_stats['filtered_samples'],
                    "filter/total_samples": filter_stats['total_samples'],
                }, step=global_step)
        
        """
        # 打印过滤统计信息
        if filter_stats["filtered_groups"] > 0:
            logger.info(f"  Success Rate Filter: 过滤了 {filter_stats['filtered_groups']}/{filter_stats['total_groups']} 组 "
                       f"({filter_stats['filtered_samples']}/{filter_stats['total_samples']} 样本)")
            logger.info(f"    原因: 这些组全部成功或全部失败，无法提供有效的对比学习信号")
        
        # ===== 调试打印：第一个epoch的详细信息 =====
        if epoch == first_epoch:
            logger.info(f"\n{'='*70}")
            logger.info(f"【调试信息】Epoch {epoch+1} 采样和Advantage详情")
            logger.info(f"【奖励计算】使用相对奖励（参考RLinf）")
            logger.info(f"【优化机制】Success Rate Filter（过滤全成功/全失败组）")
            logger.info(f"{'='*70}")
            
            # 按组整理样本
            group_samples = {}
            for sample in samples:
                group_id = sample["group_id"]
                if group_id not in group_samples:
                    group_samples[group_id] = []
                group_samples[group_id].append(sample)
            
            # 1. 打印初始Seeds和Episode IDs
            logger.info(f"\n【初始状态信息】")
            logger.info(f"  num_state_groups = {config.episode.num_state_groups}")
            logger.info(f"  samples_per_group = {config.episode.samples_per_group}")
            logger.info(f"  seed_offset (epoch {epoch}) = {config.seed + epoch * 100000}\n")
            
            for group_id in sorted(group_samples.keys()):
                group = group_samples[group_id]
                episode_id = group[0]['episode_id']
                logger.info(f"  组{group_id}: episode_id={episode_id}, {len(group)}条轨迹")
            
            # 2. 打印每组的采样详情和Advantage
            logger.info(f"\n{'='*70}")
            logger.info(f"【采样和Advantage详情】")
            logger.info(f"{'='*70}")
            
            for group_id in sorted(group_samples.keys()):
                group = group_samples[group_id]
                
                # 检查是否被过滤
                is_filtered = group[0].get("filtered", False)
                success_count = sum(1 for s in group if s["terminated"])
                
                # 计算组内统计（使用相对奖励的总和）
                group_rewards = np.array([s["reward"] for s in group])
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                
                filter_status = ""
                if is_filtered:
                    if success_count == len(group):
                        filter_status = " [已过滤: 全部成功]"
                    else:
                        filter_status = " [已过滤: 全部失败]"
                
                logger.info(f"\n组{group_id} (episode_id={group[0]['episode_id']}){filter_status}:")
                logger.info(f"  组统计: 平均总奖励={group_mean:.4f}, std={group_std:.4f}, 成功率={success_count}/{len(group)}")
                
                for i, sample in enumerate(group):
                    status = "成功" if sample["terminated"] else "未完成"
                    
                    # 如果被过滤，不计算advantage
                    if is_filtered:
                        logger.info(f"    样本{i}: 总奖励(相对)={sample['reward']:6.1f}, "
                                  f"步数={sample['num_steps']:3d}, "
                                  f"advantage=N/A [已过滤] ({status})")
                    else:
                        # 计算该样本的advantage（使用总奖励）
                        total_reward = sample["reward"]
                        advantage = (total_reward - group_mean) / group_std
                        advantage *= config.stage.advantage_scale
                        
                        judgment = "好✓" if advantage > 0 else "差✗"
                        
                        logger.info(f"    样本{i}: 总奖励(相对)={sample['reward']:6.1f}, "
                                  f"步数={sample['num_steps']:3d}, "
                                  f"advantage={advantage:+.4f} [{judgment}] ({status})")
            
            # 3. 验证Auto-Reset是否正确禁用
            logger.info(f"\n{'='*70}")
            logger.info(f"【Auto-Reset验证】")
            logger.info(f"{'='*70}")
            
            # 检查是否有轨迹混合了多个episode
            mixed_count = 0
            for sample in samples:
                if sample["first_done_step"] >= 0:
                    if sample["num_steps"] > sample["first_done_step"] + 20:
                        mixed_count += 1
            
            if mixed_count > 0:
                logger.info(f"⚠️  警告：检测到{mixed_count}条可能混合了多个episode的轨迹")
                logger.info(f"   这说明auto_reset可能仍在工作（应该已禁用）")
            else:
                logger.info(f"✓  验证通过：所有轨迹都是单个完整episode")
                logger.info(f"   auto_reset已正确禁用，满足GRPO要求")
            
            # 4. 总结：advantage分配是否合理
            logger.info(f"\n{'='*70}")
            logger.info(f"【Advantage分配验证】")
            logger.info(f"{'='*70}")
            
            correct_count = 0
            valid_groups = 0
            for group_id in sorted(group_samples.keys()):
                group = group_samples[group_id]
                is_filtered = group[0].get("filtered", False)
                
                # 跳过被过滤的组
                if is_filtered:
                    logger.info(f"  组{group_id}: [已过滤，不参与训练]")
                    continue
                
                valid_groups += 1
                group_rewards = [s["reward"] for s in group]
                
                best_idx = np.argmax(group_rewards)
                worst_idx = np.argmin(group_rewards)
                
                # 计算它们的advantage
                group_mean = np.mean(group_rewards)
                group_std = np.std(group_rewards) + 1e-8
                
                best_adv = (group_rewards[best_idx] - group_mean) / group_std
                worst_adv = (group_rewards[worst_idx] - group_mean) / group_std
                
                is_correct = best_adv > 0 and worst_adv < 0
                if is_correct:
                    correct_count += 1
                    status = "✓"
                else:
                    status = "✗"
                
                logger.info(f"  组{group_id}: 最高效样本{best_idx}(adv={best_adv:+.2f}), "
                          f"最低效样本{worst_idx}(adv={worst_adv:+.2f}) [{status}]")
            
            if valid_groups > 0:
                logger.info(f"\n  总结: {correct_count}/{valid_groups}组的advantage分配正确 "
                           f"({correct_count/valid_groups*100:.1f}%)")
                logger.info(f"  过滤: {len(group_samples)-valid_groups}/{len(group_samples)}组被过滤")
            
            logger.info(f"\n{'='*70}")
            logger.info(f"调试信息打印完成，停止训练")
            logger.info(f"{'='*70}\n")
            
            # 停止训练（跳出epoch循环）
            # break
        """
        # 3. 多轮训练（对齐RLinf的update_epoch）
        logger.info(f"\n  ===== 开始训练 (更新{update_epoch}轮) =====")
        # 根据 config.train.loss_type 派发到不同训练函数
        loss_type = getattr(config.train, 'loss_type', 'rollout')
        if loss_type == 'stepnft':
            train_fn = train_on_episode_samples_stepnft
            logger.info(f"  Loss type: stepnft（单步 forward + DPO/weighted NLL）")
        elif loss_type == 'rollout':
            train_fn = train_on_episode_samples_diffusionnft
            logger.info(f"  Loss type: rollout（多步 rollout MSE，DiffusionNFT）")
        else:
            raise ValueError(
                f"Unknown config.train.loss_type: {loss_type!r}. Expected 'rollout' or 'stepnft'."
            )
        
        for update_idx in range(update_epoch):
            logger.info(f"\n  --- 更新轮次 {update_idx+1}/{update_epoch} ---")
            
            global_step, avg_epoch_loss = train_fn(
                samples=all_samples,
                action_unet=action_unet,
                action_decoder=action_decoder,
                action_unet_old=action_unet_old,
                action_decoder_old=action_decoder_old,
                action_unet_ref=action_unet_ref,
                action_decoder_ref=action_decoder_ref,
                optimizer=optimizer,
                accelerator=accelerator,
                config=config,
                ema=ema,
                trainable_parameters=trainable_parameters,
                global_step=global_step,
                epoch=epoch
            )
        
        logger.info(f"\n  ===== 训练完成 =====")
        
        # 4. 记录到wandb（使用global_step作为横轴）
        if accelerator.is_main_process:
            # 计算成功率（百分比形式）
            success_rate_pct = (total_success_count / len(all_samples)) * 100
            
            # ===== 监控指标8: 组级统计 =====
            # 计算每组的成功率
            group_success_rates = []
            num_state_groups = config.episode.num_state_groups
            samples_per_group = config.episode.samples_per_group
            
            for group_id in range(num_state_groups):
                group_samples = [s for s in all_samples if s["group_id"] == group_id]
                if len(group_samples) > 0:
                    group_success = sum(1 for s in group_samples if s["terminated"]) / len(group_samples)
                    group_success_rates.append(group_success)
            
            # 计算组间成功率的方差（检查策略的稳定性）
            group_success_variance = np.var(group_success_rates) if len(group_success_rates) > 0 else 0.0
            
            # ===== 监控指标9: EMA相关统计 =====
            # 计算EMA模型与当前模型的差异
            if config.train.ema:
                with torch.no_grad():
                    ema_current_distance = 0.0
                    # 获取EMA参数
                    ema_params = ema.ema_parameters
                    current_params = list(trainable_parameters)
                    
                    for ema_p, cur_p in zip(ema_params, current_params):
                        ema_current_distance += (ema_p - cur_p).pow(2).sum().item()
                    
                    ema_current_distance = np.sqrt(ema_current_distance)
            else:
                ema_current_distance = 0.0
            
            # ✅ 记录epoch级别的指标，使用global_step作为横轴
            import wandb
            if wandb.run is not None:
                epoch_log_dict = {
                    "epoch/loss": avg_epoch_loss,
                    "epoch/success_rate": success_rate_pct,
                    "epoch/avg_reward": total_avg_reward,
                    "epoch/avg_steps": total_avg_steps,
                    "epoch/epoch_num": epoch,
                    "epoch/group_success_variance": group_success_variance,
                }
                
                # EMA相关
                if config.train.ema:
                    epoch_log_dict["epoch/ema_current_distance"] = ema_current_distance
                
                wandb.log(epoch_log_dict, step=global_step)
        
        # 5. 定期保存
        if (epoch + 1) % config.save_freq == 0:
            logger.info(f"\n保存检查点 (Epoch {epoch+1})...")
            save_ckpt(config.save_dir, action_unet, action_decoder, epoch + 1,
                      accelerator, ema, trainable_parameters, config)
        
        # 5b. 定期内联评估（仅 main process，每 eval.interval 个 epoch 执行一次）
        eval_cfg = getattr(config, 'eval', None)
        eval_enabled = getattr(eval_cfg, 'enabled', True) if eval_cfg is not None else True
        eval_interval = getattr(eval_cfg, 'interval', 5) if eval_cfg is not None else 5
        eval_num_eps = getattr(eval_cfg, 'num_episodes', 20) if eval_cfg is not None else 20
        eval_seed = getattr(eval_cfg, 'seed', 20000) if eval_cfg is not None else 20000

        if eval_enabled and accelerator.is_main_process and (epoch + 1) % eval_interval == 0:
            eval_results = evaluate_model_inline(
                action_unet=action_unet,
                action_decoder=action_decoder,
                pipeline=pipeline,
                config=config,
                device=accelerator.device,
                accelerator=accelerator,
                ema=ema if config.train.ema else None,
                trainable_parameters=trainable_parameters if config.train.ema else None,
                num_episodes=eval_num_eps,
                seed=eval_seed,
                epoch=epoch + 1,
            )
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "test/success_rate": eval_results['success_rate'],
                    "test/num_success": eval_results['num_success'],
                    "test/num_episodes": eval_results['num_episodes'],
                    "test/mean_reward": eval_results['mean_reward'],
                    "test/std_reward": eval_results['std_reward'],
                    "test/epoch": epoch + 1,
                }, step=global_step)
                logger.info(
                    f"[Inline Eval → wandb] test/success_rate={eval_results['success_rate']:.1f}%"
                )
        
        # 6. 更新学习率调度器（epoch结束时，ReinFlow风格）
        if lr_scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # 显示学习率变化（前100个epoch每次都显示，之后每10个epoch显示一次）
            if epoch < 100 or (epoch + 1) % 10 == 0:
                logger.info(f"  [LR调度] Epoch {epoch+1}: 学习率 {old_lr:.2e} → {new_lr:.2e}")
        
        # ========== 7. 更新old model（已移至train_on_episode_samples_diffusionnft中，每20个global_step更新） ==========
        # ✅ 方案2：按 global_step 更新 old model，而不是按 epoch
        # 更新逻辑已移至 train_on_episode_samples_diffusionnft 函数内部
        # 每20个 global_step 更新一次 old model
        """
        with torch.no_grad():
            decay = return_decay(global_step, config.train.decay_type)
            
            # 更新UNet
            for src_param, tgt_param in zip(
                unwrap_model(action_unet, accelerator).parameters(),
                action_unet_old.parameters()
            ):
                tgt_param.data.copy_(
                    tgt_param.data * decay + src_param.data * (1.0 - decay)
                )
            
            # 更新Decoder
            for src_param, tgt_param in zip(
                unwrap_model(action_decoder, accelerator).parameters(),
                action_decoder_old.parameters()
            ):
                tgt_param.data.copy_(
                    tgt_param.data * decay + src_param.data * (1.0 - decay)
                )
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  [Old model EMA] decay={decay:.4f}")
        """
    
    logger.info("训练完成！")
    
    # 最终评估（使用评估环境）
    eval_in_env(pipeline, action_unet, action_decoder, eval_env, config, accelerator, global_step, num_test_episodes=10)
    
    # 保存最终模型（使用最终epoch数）
    final_epoch = config.episode.num_epochs
    save_ckpt(config.save_dir, action_unet, action_decoder, final_epoch, accelerator, ema, trainable_parameters, config)
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    # 关闭 wandb
    accelerator.end_training()


if __name__ == "__main__":
    app.run(main)
