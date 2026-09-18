#!/usr/bin/env python
"""
在 ManiSkill 中评估 train/ 保存的预训练或强化训练 .pth checkpoint。

使用方法:
python test/evaluate_checkpoint.py --checkpoint checkpoints/imitation/model.pth --gpu 0
"""

import sys
import os

# 解析GPU参数并设置CUDA_VISIBLE_DEVICES
gpu_id = 0
for i, arg in enumerate(sys.argv):
    if arg.startswith('--gpu='):
        gpu_id = int(arg.split('=')[1])
        break
    elif arg == '--gpu' and i + 1 < len(sys.argv):
        gpu_id = int(sys.argv[i + 1])
        break

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print(f"\n{'='*70}")
print(f"[GPU设置] 使用物理 GPU {gpu_id}, 设置 CUDA_VISIBLE_DEVICES={gpu_id}")
print(f"{'='*70}\n")

# 添加路径
sys.dont_write_bytecode = True
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (
    PROJECT_ROOT,
    PROJECT_ROOT / 'flow_grpo',
    PROJECT_ROOT / 'flow_grpo' / 'diffusers_patch',
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import json
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs
from tqdm import tqdm
import argparse

# 导入模型相关
# ✅ 与 train_sd3_robot_grpo.py 严格同源：
#   - UNet: ./models/unet.py:ConditionalUnet1D
#   - ActionDecoder: 来自 robot_sde_pipeline_with_logprob_unet
# 注意：pretrain 与 RL 训练已统一为 pretrain 的 timestep 缩放：
#       t_unet = (t * 0.99 + 0.01) * 99 ≈ t * 98.01 + 0.99
#       对应 SinusoidalPosEmb 频率范围 [0.99, ≈89.2]。本脚本同步缩放。
from models.unet import ConditionalUnet1D
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import (
    RobotActionPipeline, ActionDecoder
)
from diffusers import FlowMatchEulerDiscreteScheduler
from config.robot_base import get_config as get_robot_config

ROBOT_CONFIG = get_robot_config()


# ===========================================================================
# 点云观测辅助函数
# 与 convert_pointcloud_hdf5_to_zarr.py 及 train_sd3_robot_grpo.py 完全对齐
# ===========================================================================

# PickCube-v1 分割ID：保留机械臂(1-15) + 物块(18)，排除桌面(16)/地面(17)
_PCD_KEEP_SEG_LIST = list(set(range(1, 16)) | {18})

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
      2. 语义过滤：只保留机械臂(1-15) + 物块(18)，排除桌面/地面
      3. 有效点 < n_points 时补零；否则 FPS（有 pytorch3d）或随机均匀采样
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
        pts = torch.from_numpy(valid_xyz).unsqueeze(0).cuda()
        k   = torch.tensor([n_points]).cuda()
        _, idx = _pcd_torch3d_ops.sample_farthest_points(points=pts, K=k)
        return valid_xyz[idx.squeeze(0).cpu().numpy()]
    idx = _pcd_rng.choice(n_valid, n_points, replace=False)
    return valid_xyz[idx]


def extract_pointcloud_obs(raw_obs: dict, n_points: int) -> dict:
    """
    将 ManiSkill obs_mode='pointcloud' 单步 obs 转为策略输入格式：
        {'point_cloud': (n_points, 3), 'agent_pos': (19,)}
    state = [qpos(9) + goal_pos(3) + tcp_pose(7)]，与训练 zarr 完全对齐。
    """
    xyzw     = _to_numpy_pcd(raw_obs['pointcloud']['xyzw'])
    qpos     = _to_numpy_pcd(raw_obs['agent']['qpos'])
    goal_pos = _to_numpy_pcd(raw_obs['extra']['goal_pos'])
    tcp_pose = _to_numpy_pcd(raw_obs['extra']['tcp_pose'])

    seg = None
    if 'segmentation' in raw_obs['pointcloud']:
        seg = _to_numpy_pcd(raw_obs['pointcloud']['segmentation'])

    if qpos.ndim == 2:     qpos     = qpos[0]
    if goal_pos.ndim == 2: goal_pos = goal_pos[0]
    if tcp_pose.ndim == 2: tcp_pose = tcp_pose[0]

    return {
        'point_cloud': sample_pcd(xyzw, n_points, seg),              # (n_points, 3)
        'agent_pos':   np.concatenate([qpos, goal_pos, tcp_pose]),    # (19,)
    }


def stack_pointcloud_obs(obs_deque, n_obs_steps: int, device) -> dict:
    """
    将最近 n_obs_steps 帧堆叠为 PointCloudObsEncoder 输入格式。
    Returns:
        {'point_cloud': Tensor(1, T, n_pts, 3), 'agent_pos': Tensor(1, T, 19)}
    帧数不足时复制最早帧填充。
    """
    obs_list = list(obs_deque)[-n_obs_steps:]
    while len(obs_list) < n_obs_steps:
        obs_list.insert(0, obs_list[0])
    pcd_stack  = np.stack([o['point_cloud'] for o in obs_list])   # (T, n_pts, 3)
    apos_stack = np.stack([o['agent_pos']   for o in obs_list])   # (T, 19)
    return {
        'point_cloud': torch.from_numpy(pcd_stack).unsqueeze(0).to(device),
        'agent_pos':   torch.from_numpy(apos_stack).unsqueeze(0).to(device),
    }


# ===========================================================================
# 图像观测辅助函数（与 train_sd3_robot_grpo_20260527.py 完全对齐）
# ===========================================================================

_IMG_DEPTH_MM_SCALE = 1000.0   # ManiSkill 原始深度 int16 单位 mm，/1000 → 米


def _to_numpy_img(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_image_obs(raw_obs: dict,
                      camera: str = 'base_camera',
                      use_depth: bool = True,
                      depth_max: float = 5.0) -> dict:
    """
    将 ManiSkill obs_mode='rgbd' 的单步 obs 转为策略输入格式：
        {'image': (C, H, W) float32,  C=4 (RGBD) 或 C=3 (RGB only)
         'agent_pos': (19,) float32}
    """
    sensor = raw_obs['sensor_data'][camera]

    rgb = _to_numpy_img(sensor['rgb'])
    if rgb.ndim == 4:
        rgb = rgb[0]
    rgb_chw = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)   # (3, H, W)

    if use_depth:
        depth_raw = _to_numpy_img(sensor['depth'])
        if depth_raw.ndim == 4:
            depth_raw = depth_raw[0]
        depth_m = depth_raw.astype(np.float32) / _IMG_DEPTH_MM_SCALE
        depth_m = np.clip(depth_m, 0.0, depth_max) / max(depth_max, 1e-6)
        depth_chw = depth_m.transpose(2, 0, 1)                          # (1, H, W)
        image_chw = np.concatenate([rgb_chw, depth_chw], axis=0)        # (4, H, W)
    else:
        image_chw = rgb_chw                                              # (3, H, W)

    qpos     = _to_numpy_img(raw_obs['agent']['qpos'])
    goal_pos = _to_numpy_img(raw_obs['extra']['goal_pos'])
    tcp_pose = _to_numpy_img(raw_obs['extra']['tcp_pose'])

    if qpos.ndim == 2:     qpos     = qpos[0]
    if goal_pos.ndim == 2: goal_pos = goal_pos[0]
    if tcp_pose.ndim == 2: tcp_pose = tcp_pose[0]

    agent_pos = np.concatenate([qpos, goal_pos, tcp_pose], axis=0).astype(np.float32)

    return {
        'image':     image_chw.astype(np.float32),
        'agent_pos': agent_pos,
    }


def stack_image_obs(obs_deque, n_obs_steps: int, device) -> dict:
    """
    将最近 n_obs_steps 帧堆叠为 ImageObsEncoder 输入格式。
    Returns:
        {'image': Tensor(1, T, C, H, W), 'agent_pos': Tensor(1, T, 19)}
    帧数不足时复制最早帧填充。
    """
    obs_list = list(obs_deque)[-n_obs_steps:]
    while len(obs_list) < n_obs_steps:
        obs_list.insert(0, obs_list[0])
    img_stack  = np.stack([o['image']     for o in obs_list])   # (T, C, H, W)
    apos_stack = np.stack([o['agent_pos'] for o in obs_list])   # (T, 19)
    return {
        'image':     torch.from_numpy(img_stack).unsqueeze(0).to(device),
        'agent_pos': torch.from_numpy(apos_stack).unsqueeze(0).to(device),
    }


# ===========================================================================
def _unet_timestep_input(t):
    """与 train_sd3_robot_grpo.py:_unet_timestep_input 完全一致：
    把 Flow Matching 的 t ∈ [0, 1] 缩放到 pretrain stage 的 SinusoidalPosEmb 频率范围。

    pretrain (pipeline_with_logprob) 内部：
        t_normalized = (i / N) * 0.99 + 0.01
        t_scaled     = t_normalized * 99
    一般化为任意 t ∈ [0, 1]：
        t_unet = (t * 0.99 + 0.01) * 99 = t * 98.01 + 0.99
    """
    return t * 98.01 + 0.99


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
    与训练严格对齐的 ODE 推理（noise_level=0 的 denoise_with_sde_snap 简化版）。

    关键对齐点（与 train_sd3_robot_grpo.py:denoise_with_sde_snap / train_on_episode_samples_stepnft 完全一致）：
      1. 数学层面 t = i / num_steps ∈ [0, (N-1)/N]（用于 ODE 推进的 delta 计算）
      2. UNet 输入 timestep = _unet_timestep_input(t) = t*98.01+0.99，
         匹配 pretrain stage 的 SinusoidalPosEmb 频率范围 [0.99, ≈89.2]
      3. velocity 计算：v_t = action_decoder(action_unet(x_t, t_unet, obs))，必须经过 decoder
      4. 推进方式：x_next = x_t + v_t * (1/N)（ODE 欧拉一步，无噪声）
      5. obs 归一化：使用 pipeline.normalize_obs（未 set_normalizer 则 pass-through）
      6. 输出裁剪：clip 到 [-1, 1]，与 pipeline_with_logprob 的 "clipped" 模式一致
    """
    batch_size = obs_tensor.shape[0]

    if generator is not None:
        x_t = torch.randn(
            batch_size, action_horizon, action_dim,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    else:
        x_t = torch.randn(
            batch_size, action_horizon, action_dim,
            device=device,
            dtype=torch.float32,
        )

    obs_embeds = obs_tensor
    if obs_normalizer is not None:
        obs_embeds = obs_normalizer(obs_embeds)

    delta = 1.0 / num_steps
    for i in range(num_steps):
        t_value = i / num_steps
        # ✅ 关键：UNet 输入按 pretrain 的 SinusoidalPosEmb 频率范围缩放
        t_input = torch.full(
            (batch_size,), _unet_timestep_input(t_value),
            device=device, dtype=torch.float32,
        )

        pred_raw = action_unet(
            sample=x_t,
            timestep=t_input,
            global_cond=obs_embeds,
        )
        if action_decoder is not None:
            v_t = action_decoder(pred_raw)
        else:
            v_t = pred_raw

        x_t = x_t + v_t * delta

    actions = torch.clamp(x_t, -1.0, 1.0)
    return actions


class TimestepWrapper(gym.Wrapper):
    """
    在info中添加当前timestep，用于在视频中显示步数
    
    当与RecordEpisode的info_on_video=True配合使用时，
    会在视频左上角显示当前步数
    """
    def __init__(self, env):
        super().__init__(env)
        self.current_timestep = 0
    
    def reset(self, **kwargs):
        self.current_timestep = 0
        obs, info = self.env.reset(**kwargs)
        info['timestep'] = self.current_timestep
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_timestep += 1
        info['timestep'] = self.current_timestep
        return obs, reward, terminated, truncated, info


def run_episodes(
    env,
    pipeline,
    episode_ids,
    device,
    num_inference_steps,
    max_steps,
    action_horizon,
    action_decoder=None,   # 与训练严格一致：需要传入加载好权重的 ActionDecoder
    render=False,
    label="测试",
    obs_encoder=None,      # PointCloudObsEncoder / ImageObsEncoder；None 表示 state 模式
    n_obs_steps=2,         # 点云/图像模式：历史帧数
    n_points=512,          # 点云模式：每帧采样点数
    normalizer=None,       # LinearNormalizer；非 None 时归一化 agent_pos 并反归一化 actions
    obs_mode='state',      # 'state' / 'pointcloud' / 'image'
    img_camera='base_camera',
    img_use_depth=True,
    img_depth_max=5.0,
):
    """
    在给定的 episode_ids 上运行评估，返回统计结果。

    state 模式   (obs_encoder=None): 直接把 env obs 转 tensor 作为 obs_cond
    pointcloud 模式 (obs_encoder 非 None): 使用 obs_deque + extract_pointcloud_obs
                   + stack_pointcloud_obs + obs_encoder 生成 obs_cond
    """
    from collections import deque as _deque

    all_episode_rewards = []
    all_episode_lengths = []
    all_success = []
    failed_episode_ids = []

    use_pcd   = (obs_encoder is not None) and (obs_mode == 'pointcloud')
    use_image = (obs_encoder is not None) and (obs_mode == 'image')

    print(f"\n  开始{label}测试，共 {len(episode_ids)} 个episodes...")
    print(f"  episode_ids: {episode_ids}")
    if use_pcd:
        print(f"  观测模式: pointcloud (n_obs_steps={n_obs_steps}, n_points={n_points})")
        if normalizer is not None:
            print(f"  归一化: agent_pos 归一化 + actions 反归一化")
    elif use_image:
        print(f"  观测模式: image (n_obs_steps={n_obs_steps}, camera={img_camera}, "
              f"depth={img_use_depth})")
        if normalizer is not None:
            print(f"  归一化: agent_pos 归一化 + actions 反归一化")

    def _encode_pcd_obs(obs_stack):
        """归一化 agent_pos 后编码点云观测，返回 obs_tensor (1, enc_dim)。"""
        if normalizer is not None:
            apos_input = normalizer['agent_pos'].normalize(obs_stack['agent_pos'])
        else:
            apos_input = obs_stack['agent_pos']
        return obs_encoder(obs_stack['point_cloud'], apos_input)

    def _encode_image_obs(obs_stack):
        """归一化 agent_pos 后编码图像观测，返回 obs_tensor (1, enc_dim)。"""
        if normalizer is not None:
            apos_input = normalizer['agent_pos'].normalize(obs_stack['agent_pos'])
        else:
            apos_input = obs_stack['agent_pos']
        return obs_encoder(obs_stack['image'], apos_input)

    for episode_idx in tqdm(range(len(episode_ids)), desc=f"{label}进度"):
        episode_id = episode_ids[episode_idx]

        raw_obs, info = env.reset(seed=episode_id, options={'episode_id': episode_id})

        # ---------- 初始化 obs_cond ----------
        if use_pcd:
            obs_deque = _deque(maxlen=n_obs_steps)
            step_obs  = extract_pointcloud_obs(raw_obs, n_points)
            obs_deque.append(step_obs)

            # ── DEBUG：第 0 个 episode 第 0 步保存点云用于可视化 ──────────────
            if episode_idx == 0:
                pcd_debug = step_obs['point_cloud']
                apos_debug = step_obs['agent_pos']
                np.save("debug_pcd_step0.npy",  pcd_debug)
                np.save("debug_apos_step0.npy", apos_debug)
                valid_cnt = int(np.any(pcd_debug != 0, axis=1).sum())
                print(f"\n[DEBUG PCD] episode_id={episode_id}  有效点={valid_cnt}/{pcd_debug.shape[0]}")
                print(f"  x: [{pcd_debug[:,0].min():.3f}, {pcd_debug[:,0].max():.3f}]")
                print(f"  y: [{pcd_debug[:,1].min():.3f}, {pcd_debug[:,1].max():.3f}]")
                print(f"  z: [{pcd_debug[:,2].min():.3f}, {pcd_debug[:,2].max():.3f}]")
                print(f"  agent_pos (qpos前9): {apos_debug[:9].round(3)}")
                print(f"  agent_pos (goal_pos): {apos_debug[9:12].round(3)}")
                print(f"  agent_pos (tcp_pose): {apos_debug[12:].round(3)}")
                print(f"  已保存 debug_pcd_step0.npy  debug_apos_step0.npy")
            # ──────────────────────────────────────────────────────────────────

            obs_stack = stack_pointcloud_obs(obs_deque, n_obs_steps, device)
            with torch.no_grad():
                obs_tensor = _encode_pcd_obs(obs_stack)   # (1, enc_dim)
        elif use_image:
            obs_deque = _deque(maxlen=n_obs_steps)
            step_obs  = extract_image_obs(raw_obs, camera=img_camera,
                                          use_depth=img_use_depth,
                                          depth_max=img_depth_max)
            obs_deque.append(step_obs)
            obs_stack = stack_image_obs(obs_deque, n_obs_steps, device)
            with torch.no_grad():
                obs_tensor = _encode_image_obs(obs_stack)  # (1, enc_dim)
        else:
            if isinstance(raw_obs, torch.Tensor):
                obs_tensor = raw_obs.to(device, dtype=torch.float32)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
            else:
                obs_tensor = torch.tensor(raw_obs, device=device,
                                          dtype=torch.float32).unsqueeze(0)
            if normalizer is not None:
                obs_tensor = normalizer['obs'].normalize(obs_tensor)

        episode_reward = 0.0
        episode_length = 0
        done = False
        success = False

        while not done and episode_length < max_steps:
            with torch.no_grad():
                test_gen = torch.Generator(device=device)
                test_gen.manual_seed(episode_id * 10000 + episode_length)

                obs_normalizer = getattr(pipeline, 'normalize_obs', None)
                actions_batch = denoise_ode_inference(
                    obs_tensor=obs_tensor,
                    action_unet=pipeline._action_unet,
                    action_decoder=action_decoder,
                    num_steps=num_inference_steps,
                    action_horizon=action_horizon,
                    action_dim=pipeline.action_dim,
                    device=device,
                    generator=test_gen,
                    obs_normalizer=obs_normalizer,
                )
                actions = actions_batch[0]
                # 模型输出在归一化空间，反归一化还原到真实动作空间
                if normalizer is not None:
                    actions = normalizer['action'].unnormalize(actions)

            for j in range(action_horizon):
                if done or episode_length >= max_steps:
                    break

                action = actions[j].contiguous() if isinstance(actions, torch.Tensor) else actions[j]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()

                raw_obs, reward, terminated, truncated, info = env.step(action)

                reward_value = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                episode_reward += reward_value
                episode_length += 1
                terminated_flag = bool(terminated.item() if isinstance(terminated, torch.Tensor) else terminated)
                truncated_flag = bool(truncated.item() if isinstance(truncated, torch.Tensor) else truncated)
                success_value = info.get('success', terminated_flag)
                success = success or bool(success_value.item() if isinstance(success_value, torch.Tensor) else success_value)
                done = terminated_flag or truncated_flag

            # ---------- 更新 obs_cond ----------
            if not done and episode_length < max_steps:
                if use_pcd:
                    step_obs = extract_pointcloud_obs(raw_obs, n_points)
                    obs_deque.append(step_obs)
                    obs_stack = stack_pointcloud_obs(obs_deque, n_obs_steps, device)
                    with torch.no_grad():
                        obs_tensor = _encode_pcd_obs(obs_stack)
                elif use_image:
                    step_obs = extract_image_obs(raw_obs, camera=img_camera,
                                                 use_depth=img_use_depth,
                                                 depth_max=img_depth_max)
                    obs_deque.append(step_obs)
                    obs_stack = stack_image_obs(obs_deque, n_obs_steps, device)
                    with torch.no_grad():
                        obs_tensor = _encode_image_obs(obs_stack)
                else:
                    if isinstance(raw_obs, torch.Tensor):
                        obs_tensor = raw_obs.to(device, dtype=torch.float32)
                        if obs_tensor.dim() == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                    else:
                        obs_tensor = torch.tensor(raw_obs, device=device,
                                                  dtype=torch.float32).unsqueeze(0)
                    if normalizer is not None:
                        obs_tensor = normalizer['obs'].normalize(obs_tensor)

        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        all_success.append(int(success))
        if not success:
            failed_episode_ids.append(episode_id)

        if render:
            env.flush_video(ignore_empty_transition=False)

    all_episode_rewards = np.array(all_episode_rewards)
    all_episode_lengths = np.array(all_episode_lengths)
    all_success = np.array(all_success)

    return {
        'success_rate': float(all_success.mean()),
        'num_success': int(all_success.sum()),
        'num_episodes': len(episode_ids),
        'mean_reward': float(all_episode_rewards.mean()),
        'std_reward': float(all_episode_rewards.std()),
        'mean_length': float(all_episode_lengths.mean()),
        'std_length': float(all_episode_lengths.std()),
        'all_rewards': all_episode_rewards.tolist(),
        'all_lengths': all_episode_lengths.tolist(),
        'all_success': all_success.tolist(),
        'episode_ids': episode_ids,
        'failed_episode_ids': failed_episode_ids,
    }


def evaluate_checkpoint(
    checkpoint_path,
    env_id=ROBOT_CONFIG.env_id,
    num_episodes=10,
    num_inference_steps=None,
    deterministic=True,  # ✅ DiffusionNFT: 默认使用ODE确定性模式
    max_steps=300,
    action_horizon=ROBOT_CONFIG.action_horizon,
    render=False,
    gpu_id=0,
    sim_backend=None,
    train_episode_ids_file="",  # 训练集episode_ids文件路径
    test_mode="random",         # "random" / "train_set" / "compare" / "custom"
    train_epochs=None,          # 只使用前N个epoch的episode_ids，None表示使用全部
    custom_episode_ids=None,    # 直接指定要测试的episode_ids列表
):
    """
    测试预训练或强化训练 checkpoint 在 ManiSkill 环境中的性能
    
    参数:
        checkpoint_path: 预训练或强化训练检查点路径
        env_id: ManiSkill环境ID
        num_episodes: 测试episode数量（random/val集的episode数）
        num_inference_steps: 推理步数
        deterministic: 是否使用确定性生成
        max_steps: 每个episode的最大步数
        action_horizon: 动作序列长度
        render: 是否保存视频（带timestep和reward显示）
        gpu_id: 使用的GPU ID
        sim_backend: 可选的 ManiSkill 仿真后端；默认使用 checkpoint 配置
        train_episode_ids_file: JSON文件路径，包含训练时采样的episode_ids
        test_mode: 测试模式
            "random"    - 仅测试随机验证集（原始行为）
            "train_set" - 仅测试训练集episode_ids
            "compare"   - 同时测试训练集和验证集，并做对比
            "custom"    - 仅测试 custom_episode_ids 指定的episode
        train_epochs: 只使用前N个epoch的episode_ids（int），None表示使用全部epoch
        custom_episode_ids: 直接指定的episode_ids列表（test_mode="custom"时生效）
    
    返回:
        results: 包含测试结果的字典
    """
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES已设置
    
    print(f"\n{'='*60}")
    print(f"测试训练完成的 Flow 模型")
    print(f"{'='*60}")
    print(f"检查点路径: {checkpoint_path}")
    print(f"环境ID: {env_id}")
    print(f"测试episodes: {num_episodes}")
    print(f"确定性生成: {deterministic}")
    
    # 1. 加载检查点
    checkpoint_file = Path(checkpoint_path).expanduser()
    if checkpoint_file.is_dir():
        checkpoint_file = checkpoint_file / 'model.pth'
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    if num_inference_steps is None:
        num_inference_steps = checkpoint.get('sample_num_steps') or ROBOT_CONFIG.sample.eval_num_steps
    print(f"实际推理步数: {num_inference_steps}")
    
    # ✅ 优先从 checkpoint 元数据读取架构超参（训练侧 save_ckpt 已写入），
    #    缺失或为 None 时回退到默认/函数参数（兼容旧 checkpoint）
    def _meta_or(default, *keys):
        for k in keys:
            v = checkpoint.get(k, None)
            if v is not None:
                return v
        return default

    saved_config = checkpoint.get('config_dict') or checkpoint.get('config') or {}
    obs_mode = checkpoint.get('obs_mode') or saved_config.get('pretrain', {}).get(
        'obs_mode', saved_config.get('env_kwargs', {}).get('obs_mode', 'state')
    )
    if obs_mode == 'rgbd':
        obs_mode = 'image'
    action_dim     = _meta_or(4, 'action_dim')
    action_horizon = _meta_or(action_horizon, 'action_horizon')

    # ---------- 点云模式：构建 obs_encoder 与归一化器 ----------
    if obs_mode == 'pointcloud':
        from models.pointcloud_encoder import PointCloudObsEncoder
        pcd_cfg = checkpoint.get('pcd_encoder_config', {})
        n_obs_steps_pcd = pcd_cfg.get('n_obs_steps', 2)
        pcd_out_dim     = pcd_cfg.get('pcd_out_dim', 256)
        agent_pos_dim   = pcd_cfg.get('agent_pos_dim', 19)
        state_mlp_out   = pcd_cfg.get('state_mlp_out', 64)
        use_layernorm   = pcd_cfg.get('use_layernorm', True)
        final_norm      = pcd_cfg.get('final_norm', 'layernorm')
        n_points        = 512
        img_camera      = 'base_camera'
        img_use_depth   = True
        img_depth_max   = 5.0

        obs_encoder = PointCloudObsEncoder(
            n_obs_steps=n_obs_steps_pcd,
            pcd_out_dim=pcd_out_dim,
            agent_pos_dim=agent_pos_dim,
            state_mlp_hidden=[64, state_mlp_out],
            use_layernorm=use_layernorm,
            final_norm=final_norm,
        ).to(device)
        obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
        obs_encoder.eval()
        obs_dim = obs_encoder.output_dim   # (pcd_out_dim + state_mlp_out) * n_obs_steps
        print(f"  点云编码器加载成功，output_dim={obs_dim}")

        # 归一化器（新版 checkpoint 含此字段；旧版自动跳过归一化）
        if 'normalizer_state_dict' in checkpoint:
            from models.normalizer import LinearNormalizer
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(checkpoint['normalizer_state_dict'])
            normalizer.to(device).eval()
            print(f"  归一化器加载成功（agent_pos + action）")
        else:
            normalizer = None
            print(f"  ⚠ checkpoint 无 normalizer_state_dict，跳过归一化（旧版兼容）")

    # ---------- 图像模式：构建 ImageObsEncoder 与归一化器 ----------
    elif obs_mode == 'image':
        from models.image_encoder import ImageObsEncoder
        img_cfg           = checkpoint.get('img_encoder_config', {})
        img_n_obs_steps   = img_cfg.get('n_obs_steps',   2)
        img_in_channels   = img_cfg.get('in_channels',   4)
        img_resnet_name   = img_cfg.get('resnet_name',   'resnet18')
        img_feat_dim      = img_cfg.get('img_feat_dim',  512)
        img_agent_pos_dim = img_cfg.get('agent_pos_dim', 19)
        img_state_mlp_out = img_cfg.get('state_mlp_out', 64)
        img_use_groupnorm = img_cfg.get('use_groupnorm', True)
        img_camera        = img_cfg.get('camera',        'base_camera')
        img_use_depth     = img_cfg.get('use_depth',     True)
        img_depth_max     = float(img_cfg.get('depth_max', 5.0))
        n_obs_steps_pcd   = img_n_obs_steps   # 复用此变量传给 run_episodes
        n_points          = 0

        obs_encoder = ImageObsEncoder(
            n_obs_steps=img_n_obs_steps,
            in_channels=img_in_channels,
            resnet_name=img_resnet_name,
            img_feat_dim=img_feat_dim,
            agent_pos_dim=img_agent_pos_dim,
            state_mlp_hidden=[64, img_state_mlp_out],
            use_groupnorm=img_use_groupnorm,
        ).to(device)
        obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
        obs_encoder.eval()
        obs_dim = obs_encoder.output_dim
        print(f"  图像编码器加载成功，output_dim={obs_dim}")
        print(f"  相机: {img_camera}  RGBD={'是' if img_use_depth else '否'}"
              f"  depth_max={img_depth_max}m")

        if 'normalizer_state_dict' in checkpoint:
            from models.normalizer import LinearNormalizer
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(checkpoint['normalizer_state_dict'])
            normalizer.to(device).eval()
            print(f"  归一化器加载成功（agent_pos + action）")
        else:
            normalizer = None
            print(f"  ⚠ checkpoint 无 normalizer_state_dict，跳过归一化（旧版兼容）")

    else:
        obs_encoder     = None
        n_obs_steps_pcd = 1
        n_points        = 0
        img_camera      = 'base_camera'
        img_use_depth   = True
        img_depth_max   = 5.0
        obs_dim         = _meta_or(42, 'obs_dim')

        # state 模式：新版 checkpoint 包含 normalizer_state_dict（obs + action 均归一化）
        if 'normalizer_state_dict' in checkpoint:
            from models.normalizer import LinearNormalizer
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(checkpoint['normalizer_state_dict'])
            normalizer.to(device).eval()
            print(f"  归一化器加载成功（obs + action）")
        else:
            normalizer = None
            print(f"  ⚠ checkpoint 无 normalizer_state_dict，跳过归一化（旧版兼容）")

    print(f"\n模型配置（来自 checkpoint 元数据）:")
    print(f"  观测模式: {obs_mode}")
    print(f"  obs_dim (global_cond_dim): {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  动作序列长度: {action_horizon}")
    
    # 2. 创建模型（与 train_sd3_robot_grpo.py 严格同源：unet.ConditionalUnet1D + ActionDecoder）
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(device)

    # 检查 checkpoint 是否训练了 decoder：
    #   decoder_trained=True  → action_loss 路径，UNet 输出需经 decoder 再得到速度场
    #   decoder_trained=False → velocity_loss 路径，UNet 直接输出速度场，decoder 完全旁路
    # 旧版 checkpoint 无此字段，默认视为 True（向后兼容）
    decoder_trained = checkpoint.get('decoder_trained', True)

    if decoder_trained:
        action_decoder = ActionDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=256,
            num_layers=3,
        ).to(device)
        if 'action_decoder_state_dict' in checkpoint:
            action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])
            print("✓ UNet 与 ActionDecoder 权重均加载成功")
        else:
            print("⚠ checkpoint 缺少 action_decoder_state_dict，使用随机初始化的 decoder（与训练不一致）")
        action_decoder.eval()
    else:
        action_decoder = None
        print("✓ UNet 权重加载成功（velocity_loss 模式：ActionDecoder 未参与训练，已跳过）")

    # 加载 UNet 权重
    action_unet.load_state_dict(checkpoint['action_unet_state_dict'])
    action_unet.eval()

    # 3. 创建管线
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=str(device)
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder
    
    # 4. 创建环境（obs_mode 由 checkpoint 决定）
    if obs_mode == 'pointcloud':
        env_obs_mode = 'pointcloud'
    elif obs_mode == 'image':
        env_obs_mode = 'rgbd'
    else:
        env_obs_mode = 'state'
    saved_env_kwargs = saved_config.get('env_kwargs', {})
    env_kwargs = {
        'sim_backend': sim_backend or saved_env_kwargs.get('sim_backend', 'gpu'),
        'control_mode': saved_env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode': env_obs_mode,
        'render_mode': 'rgb_array',
        'reward_mode': saved_env_kwargs.get('reward_mode', 'dense'),
    }
    
    print(f"\n创建环境: {env_id}")
    base_env = gym.make(
        env_id, reconfiguration_freq=1, max_episode_steps=max_steps, **env_kwargs
    )
    
    # 如果需要录制视频
    if render:
        from mani_skill.utils.wrappers.record import RecordEpisode
        video_dir = "./test_videos"
        os.makedirs(video_dir, exist_ok=True)
        
        # 先包装TimestepWrapper以在视频中显示步数
        base_env = TimestepWrapper(base_env)
        
        record_kwargs = {
            'save_trajectory': False,
            'save_video': True,
            'info_on_video': True,  # 启用信息显示（会显示timestep和reward）
            'video_fps': 30
        }
        env = RecordEpisode(base_env, video_dir, **record_kwargs)
        print(f"视频将保存到: {video_dir}")
        print(f"视频中会显示timestep和reward信息（绿色文字在左上角）")
    else:
        env = base_env
    
    # 5. 加载训练集 episode_ids（如有）
    train_episode_ids = None
    if train_episode_ids_file:
        if not os.path.exists(train_episode_ids_file):
            raise FileNotFoundError(f"训练集episode_ids文件不存在: {train_episode_ids_file}")
        with open(train_episode_ids_file, 'r') as f:
            ids_data = json.load(f)
        
        episode_ids_by_epoch = ids_data.get('episode_ids_by_epoch', None)
        total_epochs = ids_data.get('total_epochs', None)
        
        if train_epochs is not None and episode_ids_by_epoch is not None:
            # 只取前 train_epochs 个 epoch 的 episode_ids（取并集）
            selected = set()
            for ep in range(1, train_epochs + 1):
                key = str(ep)
                if key in episode_ids_by_epoch:
                    selected.update(episode_ids_by_epoch[key])
                else:
                    print(f"  ⚠ JSON中未找到 epoch {ep} 的数据，跳过")
            train_episode_ids = sorted(selected)
            print(f"\n✓ 已加载训练集episode_ids（前{train_epochs}个epoch，共{total_epochs}个epoch）")
            print(f"  唯一ID数量: {len(train_episode_ids)}")
        else:
            # 使用全部 epoch 的唯一 ID
            train_episode_ids = ids_data.get('unique_episode_ids', ids_data.get('all_sampled_episode_ids', []))
            print(f"\n✓ 已加载训练集episode_ids（全部{total_epochs}个epoch）")
            print(f"  唯一ID数量: {len(train_episode_ids)}")
        
        print(f"  来源文件: {train_episode_ids_file}")
        print(f"  训练集episode_ids: {train_episode_ids}")
    
    # 6. 生成验证集（随机）episode_ids
    val_generator = torch.Generator(device='cpu')
    val_generator.manual_seed(20000)  # 固定seed保证可复现
    val_episode_ids = torch.randint(
        low=0,
        high=10000,
        size=(num_episodes,),
        generator=val_generator,
        dtype=torch.int64
    ).tolist()
    
    print(f"\n验证集（随机）episode_ids: {num_episodes} 个")
    print(f"  前10个: {val_episode_ids[:min(10, num_episodes)]}")
    
    # 7. 根据 test_mode 决定运行哪些测试
    results = {'checkpoint_path': checkpoint_path}

    if test_mode == "custom":
        if not custom_episode_ids:
            raise ValueError("test_mode='custom' 时必须通过 --episode_ids 或 custom_episode_ids 参数指定episode_ids")
        print(f"\n{'='*60}")
        print(f"[自定义episode测试] 使用指定的episode_ids")
        print(f"{'='*60}")
        custom_results = run_episodes(
            env=env,
            pipeline=pipeline,
            episode_ids=custom_episode_ids,
            device=device,
            num_inference_steps=num_inference_steps,
            max_steps=max_steps,
            action_horizon=action_horizon,
            action_decoder=action_decoder,
            render=render,
            label="自定义",
            obs_encoder=obs_encoder,
            n_obs_steps=n_obs_steps_pcd,
            n_points=n_points,
            normalizer=normalizer,
            obs_mode=obs_mode,
            img_camera=img_camera,
            img_use_depth=img_use_depth,
            img_depth_max=img_depth_max,
        )
        results['custom'] = custom_results
        _print_summary("自定义episode", custom_results)
        results.update({
            'num_episodes': custom_results['num_episodes'],
            'success_rate': custom_results['success_rate'],
            'num_success': custom_results['num_success'],
            'mean_reward': custom_results['mean_reward'],
            'std_reward': custom_results['std_reward'],
            'mean_length': custom_results['mean_length'],
            'std_length': custom_results['std_length'],
            'all_rewards': custom_results['all_rewards'],
            'all_lengths': custom_results['all_lengths'],
            'all_success': custom_results['all_success'],
            'failed_episode_ids': custom_results['failed_episode_ids'],
        })
        env.close()
        return results

    if test_mode in ("random", "compare"):
        print(f"\n{'='*60}")
        print(f"[验证集测试] 使用随机episode_ids（未出现在训练中）")
        print(f"{'='*60}")
        val_results = run_episodes(
            env=env,
            pipeline=pipeline,
            episode_ids=val_episode_ids,
            device=device,
            num_inference_steps=num_inference_steps,
            max_steps=max_steps,
            action_horizon=action_horizon,
            action_decoder=action_decoder,
            render=render,
            label="验证集",
            obs_encoder=obs_encoder,
            n_obs_steps=n_obs_steps_pcd,
            n_points=n_points,
            normalizer=normalizer,
            obs_mode=obs_mode,
            img_camera=img_camera,
            img_use_depth=img_use_depth,
            img_depth_max=img_depth_max,
        )
        results['val'] = val_results
        _print_summary("验证集（随机）", val_results)
    
    if test_mode in ("train_set", "compare"):
        if train_episode_ids is None:
            print("\n⚠ 未提供 --train_episode_ids_file，跳过训练集测试")
        else:
            print(f"\n{'='*60}")
            print(f"[训练集测试] 使用训练时采样的episode_ids（模型见过这些场景）")
            print(f"{'='*60}")
            train_results = run_episodes(
                env=env,
                pipeline=pipeline,
                episode_ids=train_episode_ids,
                device=device,
                num_inference_steps=num_inference_steps,
                max_steps=max_steps,
                action_horizon=action_horizon,
                action_decoder=action_decoder,
                render=render,
                label="训练集",
                obs_encoder=obs_encoder,
                n_obs_steps=n_obs_steps_pcd,
                n_points=n_points,
                normalizer=normalizer,
                obs_mode=obs_mode,
                img_camera=img_camera,
                img_use_depth=img_use_depth,
                img_depth_max=img_depth_max,
            )
            results['train'] = train_results
            _print_summary("训练集（采样过的）", train_results)
    
    if test_mode == "compare" and 'train' in results and 'val' in results:
        _print_overfitting_comparison(results['train'], results['val'])
    
    if test_mode == "random":
        # 兼容旧接口：直接返回val结果的扁平化形式
        val_r = results['val']
        results.update({
            'num_episodes': val_r['num_episodes'],
            'success_rate': val_r['success_rate'],
            'num_success': val_r['num_success'],
            'mean_reward': val_r['mean_reward'],
            'std_reward': val_r['std_reward'],
            'mean_length': val_r['mean_length'],
            'std_length': val_r['std_length'],
            'all_rewards': val_r['all_rewards'],
            'all_lengths': val_r['all_lengths'],
            'all_success': val_r['all_success'],
            'failed_episode_ids': val_r['failed_episode_ids'],
        })
    
    # 关闭环境
    env.close()
    
    return results


def _print_summary(label, res):
    """打印单次测试结果统计"""
    rewards = np.array(res['all_rewards'])
    print(f"\n{'='*60}")
    print(f"[{label}] 测试结果统计")
    print(f"{'='*60}")
    print(f"  测试episodes: {res['num_episodes']}")
    print(f"  成功次数: {res['num_success']}")
    print(f"  成功率: {res['success_rate']:.2%}")
    print(f"  平均奖励: {res['mean_reward']:.3f} ± {res['std_reward']:.3f}")
    print(f"  平均长度: {res['mean_length']:.1f} ± {res['std_length']:.1f}")
    print(f"  奖励分布: min={rewards.min():.3f}, "
          f"p25={np.percentile(rewards, 25):.3f}, "
          f"median={np.median(rewards):.3f}, "
          f"p75={np.percentile(rewards, 75):.3f}, "
          f"max={rewards.max():.3f}")
    failed_ids = res.get('failed_episode_ids', [])
    print(f"  失败episode数量: {len(failed_ids)}")
    if failed_ids:
        print(f"  失败episode_ids: {failed_ids}")


def _print_overfitting_comparison(train_res, val_res):
    """打印过拟合对比结果"""
    train_sr = train_res['success_rate']
    val_sr = val_res['success_rate']
    gap = train_sr - val_sr

    print(f"\n{'#'*70}")
    print(f"# 过拟合检测对比报告")
    print(f"{'#'*70}")
    print(f"\n  训练集成功率（模型见过的episode_ids）: {train_sr:.2%}  ({train_res['num_success']}/{train_res['num_episodes']})")
    print(f"  验证集成功率（随机episode_ids）:        {val_sr:.2%}  ({val_res['num_success']}/{val_res['num_episodes']})")
    print(f"\n  训练集 - 验证集 成功率差距: {gap:+.2%}")
    
    if gap > 0.15:
        print(f"\n  ⚠ 结论：训练集成功率显著高于验证集（差距 > 15%）")
        print(f"     这强烈提示模型已对RL采样的轨迹数据发生过拟合！")
    elif gap > 0.05:
        print(f"\n  ⚡ 结论：训练集成功率略高于验证集（差距 5%~15%）")
        print(f"     存在轻度过拟合迹象，需进一步监控。")
    elif gap < -0.05:
        print(f"\n  ✓ 结论：验证集成功率不低于训练集，未观察到过拟合。")
        print(f"     模型具有良好的泛化性能。")
    else:
        print(f"\n  ✓ 结论：训练集与验证集成功率接近（差距 ≤ 5%），未观察到明显过拟合。")
    
    print(f"\n  训练集平均奖励: {train_res['mean_reward']:.3f} ± {train_res['std_reward']:.3f}")
    print(f"  验证集平均奖励: {val_res['mean_reward']:.3f} ± {val_res['std_reward']:.3f}")
    print(f"  奖励差距: {train_res['mean_reward'] - val_res['mean_reward']:+.3f}")
    print(f"\n{'#'*70}\n")


def main():
    parser = argparse.ArgumentParser(description='评估 train/ 生成的 .pth checkpoint 在 ManiSkill 中的成功率')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='预训练 .pth 文件、强化训练 model.pth 文件或包含 model.pth 的目录')
    parser.add_argument('--env_id', type=str, default=ROBOT_CONFIG.env_id,
                       help='ManiSkill环境ID')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='验证集（随机）测试episode数量')
    parser.add_argument('--num_inference_steps', type=int, default=None,
                       help='推理步数；默认读取 checkpoint，旧 checkpoint 默认 10')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='使用确定性ODE生成（DiffusionNFT方法，默认开启）')
    parser.add_argument('--max_steps', type=int, default=300,
                       help='每个episode的最大步数')
    parser.add_argument('--action_horizon', type=int, default=ROBOT_CONFIG.action_horizon,
                       help='动作序列长度（与训练配置一致）')
    parser.add_argument('--render', action='store_true',
                       help='保存视频（会在视频中显示timestep和reward）')
    parser.add_argument('--gpu', type=int, default=0,
                       help='使用的GPU ID')
    parser.add_argument('--sim_backend', type=str, default=None,
                       choices=['gpu', 'physx_cpu'],
                       help='覆盖 checkpoint 中的仿真后端，便于和演示数据对照')
    parser.add_argument('--save_results', type=str, default='',
                       help='保存结果的JSON文件路径')
    parser.add_argument('--train_episode_ids_file', type=str, default='',
                       help='训练时采样的episode_ids JSON文件路径')
    parser.add_argument('--test_mode', type=str, default='random',
                       choices=['random', 'train_set', 'compare', 'custom'],
                       help=(
                           'random: 仅测试随机验证集（默认）; '
                           'train_set: 仅测试训练集episode_ids; '
                           'compare: 同时测试两者并输出过拟合对比报告; '
                           'custom: 仅测试 --episode_ids 指定的episode'
                       ))
    parser.add_argument('--train_epochs', type=int, default=None,
                       help='只使用前N个epoch的episode_ids作为训练集（默认None=使用全部epoch）')
    parser.add_argument('--episode_ids', type=str, nargs='+', default=[],
                       help=(
                           '直接指定要测试的episode_ids，逗号分隔整数列表，'
                           '例如 --episode_ids 0,5,42,100 或 --episode_ids 0, 5, 42, 100（逗号后有空格也可）。'
                           '会自动将 test_mode 切换为 custom。'
                       ))
    
    args = parser.parse_args()

    # 解析 --episode_ids
    # nargs='+' 使得 "1, 2, 3"（逗号后有空格）和 "1,2,3" 均可正确接收
    custom_episode_ids = None
    if args.episode_ids:
        combined = ''.join(args.episode_ids)   # 将多个 token 拼回一个字符串
        custom_episode_ids = [int(x.strip()) for x in combined.split(',') if x.strip()]
        if args.test_mode == 'random':
            # 若用户没有显式指定 test_mode，自动切换为 custom
            args.test_mode = 'custom'
        print(f"\n[自定义episode_ids] 共 {len(custom_episode_ids)} 个: {custom_episode_ids}")

    # 运行测试
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        num_inference_steps=args.num_inference_steps,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        action_horizon=args.action_horizon,
        render=args.render,
        gpu_id=args.gpu,
        sim_backend=args.sim_backend,
        train_episode_ids_file=args.train_episode_ids_file,
        test_mode=args.test_mode,
        train_epochs=args.train_epochs,
        custom_episode_ids=custom_episode_ids,
    )
    
    # 保存结果到JSON文件
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {args.save_results}")
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()
