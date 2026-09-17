#!/usr/bin/env python
"""
机器人动作生成的 Flow-GRPO 训练脚本
基于 train_sd3.py 改造，使用 ConditionalUnet1D 和 RobotActionPipeline

================================================================================
监控指标说明（以global_step为横轴）
================================================================================

1. 采样质量指标 (sample/)
   - avg_steps: 平均轨迹步数 [问题信号: 突然增加说明效率下降]
   - median_steps: 中位数步数
   - std_steps: 步数标准差 [问题信号: 方差过大说明不稳定]
   - short_traj_ratio: 异常短轨迹比例(<50步)
   - long_traj_ratio: 异常长轨迹比例(>250步)
   - avg_flow_calls: 平均Flow调用次数
   - reward_mean/std/min/max: 奖励分布统计

2. Advantage统计指标 (advantage/)
   - mean/std/min/max: Advantage分布 [问题信号: std过小说明无学习信号]
   - positive_ratio: 正advantage比例 [问题信号: 全正或全负]
   - valid_samples: 有效样本数

3. 过滤统计指标 (filter/)
   - filtered_groups_ratio: 被过滤组比例 [问题信号: >0.5说明策略两极分化]
   - filtered_samples: 被过滤样本数

4. Loss组成指标 (loss/)
   - positive_loss/negative_loss: 正负样本loss
   - pos_neg_ratio: 正负loss比例 [问题信号: 比例失衡]
   - policy_kl_ratio: 策略loss与KL loss比例 [问题信号: KL过高抑制优化]

5. 模型预测指标 (prediction/)
   - x1_error_pos/neg: 预测误差 [问题信号: 突然增大说明发散]
   - output_std: 输出标准差 [问题信号: 急剧下降说明mode collapse]
   - kl_div_per_sample: 每样本KL散度

6. 优化过程指标 (optim/)
   - grad_clipped: 梯度裁剪标志 [问题信号: 频繁裁剪说明梯度过大]
   - param_update_norm: 参数更新范数
   - param_update_ratio: 参数更新相对幅度 [问题信号: 过大或过小]
   - learning_rate: 当前学习率

7. 模型差异指标 (model/)
   - current_old_distance: 当前与old模型的L2距离
   - current_ref_distance: 当前与ref模型的L2距离 [问题信号: 过大可能导致KL失效]
   - old_ref_distance: old与ref模型的距离
   - old_decay: Old model的EMA decay系数

8. Epoch级统计 (epoch/)
   - loss: Epoch平均loss
   - success_rate: 成功率 [核心指标]
   - avg_reward/avg_steps: 平均奖励和步数
   - group_success_variance: 组间成功率方差 [问题信号: 方差过大说明不稳定]
   - ema_current_distance: EMA与当前模型距离

9. 训练基础指标 (train/)
   - policy_loss/kl_div_loss/total_loss: 各项loss
   - grad_norm_before/after_clip: 梯度范数
   - epoch: 当前epoch数

================================================================================
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

# ========== FlowPolicy 路径 ==========
# 添加 FlowPolicy 代码路径，使得可以 import flow_policy_3d 下的模块
FLOWPOLICY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'FlowPolicy', 'FlowPolicy')
sys.path.insert(0, FLOWPOLICY_ROOT)

from collections import defaultdict
import contextlib
import datetime
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
# 使用 FlowPolicy 风格的 UNet（无 var_est，timestep 由调用方缩放 *99）
from flow_unet1d import FlowConditionalUnet1D
# 保留旧 UNet 别名以便后续平滑过渡
ConditionalUnet1D = FlowConditionalUnet1D
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import RobotActionPipeline, ActionDecoder, pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import ode_step_with_logprob

# ========== FlowPolicy 模块导入 ==========
import dill
import hydra
from omegaconf import OmegaConf
from collections import deque as _deque  # 避免与 collections.deque 冲突

# 注册 OmegaConf eval resolver（与 FlowPolicy train.py 保持一致）
try:
    OmegaConf.register_new_resolver('eval', eval, replace=True)
except Exception:
    pass  # 已注册则跳过

# FlowPolicy 核心模块（依赖 FLOWPOLICY_ROOT 已加入 sys.path）
from flow_policy_3d.policy.flowpolicy import FlowPolicy
from flow_policy_3d.common.pytorch_util import dict_apply

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


# ============================================================
# FlowPolicy 预训练模型加载工具
# ============================================================

def load_flowpolicy_from_ckpt(ckpt_path: str, device):
    """
    从 FlowPolicy 训练保存的 .ckpt 文件加载完整模型（含 normalizer）。

    参数
    ----
    ckpt_path : str
        训练时保存的检查点路径，例如
        ".../checkpoints/epoch=0700.ckpt"
    device : torch.device 或 str
        目标设备

    返回
    ----
    policy : FlowPolicy
        已加载权重（优先使用 EMA 权重）并移至 device 的模型实例。
    cfg : OmegaConf DictConfig
        训练时的配置对象。
    """
    import copy

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"FlowPolicy 检查点不存在: {ckpt_path}")

    logger.info(f"加载 FlowPolicy 检查点: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location='cpu', pickle_module=dill)

    cfg = payload['cfg']
    epoch = dill.loads(payload['pickles']['epoch'])
    global_step = dill.loads(payload['pickles']['global_step'])
    logger.info(f"  Epoch={epoch},  GlobalStep={global_step}")
    logger.info(f"  Policy target: {cfg.policy._target_}")
    logger.info(f"  n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps}, "
                f"horizon={cfg.horizon}")

    # 实例化模型架构
    policy: FlowPolicy = hydra.utils.instantiate(cfg.policy)

    # 加载 base model 权重
    policy.load_state_dict(payload['state_dicts']['model'])

    # 优先使用 EMA 权重进行推理（与 FlowPolicy train.py 保持一致）
    if cfg.training.use_ema and 'ema_model' in payload['state_dicts']:
        ema_state = payload['state_dicts']['ema_model']
        policy.load_state_dict(ema_state)
        logger.info("  已加载 EMA 模型权重（用于 RL 训练初始化）")
    else:
        logger.info("  使用原始模型权重（未找到 EMA 权重）")

    # 恢复 normalizer
    if 'model' in payload['state_dicts']:
        # normalizer 存储在 model state_dict 中（FlowPolicy 的 self.normalizer）
        pass  # 已通过 load_state_dict 恢复

    policy = policy.to(device)
    return policy, cfg


def clone_flowpolicy(source_policy: FlowPolicy, device) -> FlowPolicy:
    """
    深拷贝一个 FlowPolicy 实例（含 normalizer、obs_encoder、model 全部权重）。
    用于创建 old/ref 模型副本。
    """
    import copy
    try:
        new_policy = copy.deepcopy(source_policy)
    except Exception:
        # Minkowski Engine 等模块可能无法直接 deepcopy，重新 hydra instantiate
        raise RuntimeError("FlowPolicy deepcopy 失败，请检查模型组件是否支持 deepcopy")
    new_policy = new_policy.to(device)
    return new_policy


# ============================================================


# ========== Replay Buffer for Experience Replay ==========
class AdvantageBalancedReplayBuffer:
    """
    经验回放Buffer，用于平衡正负样本
    - 存储完整的sample对象
    - FIFO策略管理旧样本
    - 自动平衡正负样本比例
    """
    
    def __init__(self, config):
        # 配置参数
        self.max_size = getattr(config.train, 'replay_buffer_size', 10000)
        self.target_positive_ratio = getattr(config.train, 'target_positive_ratio', 0.5)
        self.balance_threshold = getattr(config.train, 'balance_threshold', 0.1)
        self.enabled = getattr(config.train, 'use_replay_buffer', True)
        self.max_augment_ratio = getattr(config.train, 'buffer_sample_augment_ratio', 0.3)
        
        # 存储结构 (使用deque实现FIFO)
        from collections import deque
        self.positive_buffer = deque(maxlen=self.max_size // 2)
        self.negative_buffer = deque(maxlen=self.max_size // 2)
        
        # 统计信息
        self.current_epoch = 0
        self.total_added = 0
        self.total_sampled = 0
        self.added_this_epoch = 0
        self.sampled_this_epoch = 0
    
    def add_sample(self, sample):
        """添加样本到buffer（深拷贝并移到CPU）"""
        if not self.enabled:
            return
        
        if sample.get("filtered", False):
            return
        
        # 深拷贝sample并移到CPU
        sample_copy = self._deep_copy_sample(sample)
        
        # 根据advantage分类存储
        if sample["advantage"] > 0:
            self.positive_buffer.append(sample_copy)
        else:
            self.negative_buffer.append(sample_copy)
        
        self.total_added += 1
        self.added_this_epoch += 1
    
    def balance_samples(self, current_samples, epoch):
        """检查并平衡样本"""
        if not self.enabled:
            return current_samples
        
        # 重置本epoch统计
        self.added_this_epoch = 0
        self.sampled_this_epoch = 0
        
        # 1. 统计当前样本分布
        valid_samples = [s for s in current_samples if not s.get("filtered", False)]
        current_pos = sum(1 for s in valid_samples if s["advantage"] > 0)
        current_neg = len(valid_samples) - current_pos
        
        if current_pos + current_neg == 0:
            logger.info(f"\n  ===== 样本平衡检查 =====")
            logger.info(f"  ⚠ 没有有效样本，跳过平衡")
            return current_samples
        
        current_ratio = current_pos / (current_pos + current_neg)
        
        # 2. 检查是否需要平衡
        deviation = abs(current_ratio - self.target_positive_ratio)
        
        logger.info(f"\n  ===== 样本平衡检查 =====")
        logger.info(f"  当前正样本: {current_pos}/{current_pos+current_neg} ({current_ratio:.1%})")
        logger.info(f"  偏离目标: {deviation:.1%} (阈值: {self.balance_threshold:.1%})")
        
        if deviation < self.balance_threshold:
            logger.info(f"  ✓ 样本已平衡，无需补充")
            return current_samples
        
        # 3. 计算需要补充的样本数
        if current_ratio > self.target_positive_ratio:
            # 正样本过多，补充负样本
            target_neg = int(current_pos * (1 - self.target_positive_ratio) / self.target_positive_ratio)
            needed = target_neg - current_neg
            category = 'negative'
            source_buffer = self.negative_buffer
        else:
            # 负样本过多，补充正样本
            target_pos = int(current_neg * self.target_positive_ratio / (1 - self.target_positive_ratio))
            needed = target_pos - current_pos
            category = 'positive'
            source_buffer = self.positive_buffer
        
        # 4. 从buffer采样
        if len(source_buffer) == 0:
            logger.info(f"  ⚠ {category} buffer为空，无法补充")
            return current_samples
        
        # 限制补充数量
        max_augment = int(len(valid_samples) * self.max_augment_ratio)
        actual_sample = min(needed, len(source_buffer), max_augment)
        
        if actual_sample <= 0:
            logger.info(f"  ✓ 无需补充样本")
            return current_samples
        
        augmented = random.sample(list(source_buffer), actual_sample)
        
        # 计算补充后的预期比例
        if category == 'positive':
            expected_ratio = (current_pos + actual_sample) / (current_pos + current_neg + actual_sample)
        else:
            expected_ratio = current_pos / (current_pos + current_neg + actual_sample)
        
        logger.info(f"  → 从{category} buffer补充 {actual_sample} 个样本")
        logger.info(f"  补充后预计比例: {expected_ratio:.1%}")
        
        self.total_sampled += actual_sample
        self.sampled_this_epoch = actual_sample
        
        # 5. 转回GPU并合并
        augmented_gpu = []
        for sample in augmented:
            sample_gpu = self._move_sample_to_device(sample, device='cuda')
            augmented_gpu.append(sample_gpu)
        
        return current_samples + augmented_gpu
    
    def get_statistics(self):
        """获取buffer统计信息"""
        pos_count = len(self.positive_buffer)
        neg_count = len(self.negative_buffer)
        total_count = pos_count + neg_count
        
        # 估算内存占用 (每个sample约60KB)
        memory_mb = total_count * 0.06
        
        return {
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_capacity': self.max_size // 2,
            'negative_capacity': self.max_size // 2,
            'total_count': total_count,
            'total_capacity': self.max_size,
            'usage_ratio': total_count / self.max_size if self.max_size > 0 else 0.0,
            'added_this_epoch': self.added_this_epoch,
            'sampled_this_epoch': self.sampled_this_epoch,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'memory_mb': memory_mb,
        }
    
    def _deep_copy_sample(self, sample):
        """深拷贝sample（移到CPU）"""
        import copy
        
        sample_copy = {}
        for key, value in sample.items():
            if key == 'obs_history':
                # 将obs列表中的tensor移到CPU
                sample_copy[key] = [obs.cpu().clone() for obs in value]
            elif key == 'action_history':
                # 将action列表中的tensor移到CPU
                sample_copy[key] = [act.cpu().clone() for act in value]
            elif key == 'obs_list':
                # 兼容旧格式
                sample_copy[key] = [obs.cpu().clone() for obs in value]
            elif key == 'clean_actions_list':
                # 兼容旧格式
                sample_copy[key] = [act.cpu().clone() for act in value]
            elif key == 'timesteps':
                # timesteps也移到CPU
                if isinstance(value, torch.Tensor):
                    sample_copy[key] = value.cpu().clone()
                else:
                    sample_copy[key] = value
            else:
                # 其他字段直接拷贝
                sample_copy[key] = copy.deepcopy(value)
        
        return sample_copy
    
    def _move_sample_to_device(self, sample, device='cuda'):
        """将sample从CPU移回GPU"""
        sample_gpu = {}
        for key, value in sample.items():
            if key == 'obs_history':
                sample_gpu[key] = [obs.to(device) for obs in value]
            elif key == 'action_history':
                sample_gpu[key] = [act.to(device) for act in value]
            elif key == 'obs_list':
                sample_gpu[key] = [obs.to(device) for obs in value]
            elif key == 'clean_actions_list':
                sample_gpu[key] = [act.to(device) for act in value]
            elif key == 'timesteps':
                if isinstance(value, torch.Tensor):
                    sample_gpu[key] = value.to(device)
                else:
                    sample_gpu[key] = value
            else:
                sample_gpu[key] = value
        
        return sample_gpu


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


# ============================================================
# 点云观测处理工具函数
# （与 test_flowpolicy_maniskill.py 保持完全一致，确保训练/推理对齐）
# ============================================================

# PickCube-v1 分割ID配置（与 convert_pointcloud_hdf5_to_zarr.py 和 runner 完全一致）
# Panda 机械臂所有 link 的 seg_id 范围（panda_link0 到 panda_rightfinger_pad）
_ROBOT_SEG_IDS = set(range(1, 16))   # 1 ~ 15
_CUBE_SEG_ID   = 18                   # 红色目标物块
# 背景：table-workspace=16, ground=17，全部排除
_KEEP_SEG_IDS  = _ROBOT_SEG_IDS | {_CUBE_SEG_ID}
_KEEP_SEG_LIST = list(_KEEP_SEG_IDS)  # 预先转为 list，避免每帧重复转换

try:
    import pytorch3d.ops as _torch3d_ops
    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False

_pcd_rng = np.random.default_rng(42)


def _to_numpy_pcd(x) -> np.ndarray:
    """将 Tensor 或 ndarray 转为 float32 ndarray。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32)


def sample_pcd(xyzw: np.ndarray, n_points: int,
               seg: np.ndarray = None) -> np.ndarray:
    """
    从单帧点云 xyzw (N, 4) 中采样 n_points 个点，与训练数据转换完全一致：
      1. 基础过滤：w > 0（无效/相机外点）
      2. 语义过滤：只保留机械臂(seg 1-15)和物块(seg 18)，排除桌面/地面背景
      3. 有效点不足 n_points 时：补零至 n_points（zero-padding）
      4. 有效点充足时：FPS（pytorch3d）或随机均匀采样（回退）

    Args:
        xyzw:     (N, 4) 或 (1, N, 4) 点云坐标+有效性
        n_points: 目标采样点数
        seg:      (N, 1) 或 (1, N, 1) 分割ID，int16，None 时跳过语义过滤

    Returns:
        (n_points, 3) float32
    """
    if xyzw.ndim == 3:
        xyzw = xyzw[0]
    if seg is not None and seg.ndim == 3:
        seg = seg[0]

    # 1. 基础过滤：w > 0
    valid_mask = xyzw[:, 3] > 0

    # 2. 语义过滤：只保留机械臂(seg 1-15) + 物块(seg 18)，排除桌面/地面背景
    if seg is not None:
        seg_flat = seg.squeeze(-1).astype(np.int32)
        sem_mask = np.isin(seg_flat, _KEEP_SEG_LIST)
        valid_mask = valid_mask & sem_mask

    valid_xyz = xyzw[valid_mask, :3].astype(np.float32)
    n_valid   = len(valid_xyz)

    if n_valid == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    # 有效点不足时补零（与训练数据 zarr 转换一致）
    if n_valid < n_points:
        pad = np.zeros((n_points - n_valid, 3), dtype=np.float32)
        return np.concatenate([valid_xyz, pad], axis=0)

    if _HAS_PYTORCH3D and torch.cuda.is_available():
        pts = torch.from_numpy(valid_xyz).unsqueeze(0).cuda()  # (1, N, 3)
        k   = torch.tensor([n_points]).cuda()
        _, idx = _torch3d_ops.sample_farthest_points(points=pts, K=k)
        return valid_xyz[idx.squeeze(0).cpu().numpy()]
    else:
        idx = _pcd_rng.choice(n_valid, n_points, replace=False)
        return valid_xyz[idx]


def extract_pointcloud_obs(raw_obs: dict, n_points: int) -> dict:
    """
    将 ManiSkill pointcloud obs dict 转换为单时间步标准格式:
        {'point_cloud': (n_points, 3), 'agent_pos': (19,)}
    state = qpos(9) + goal_pos(3) + tcp_pose(7) = 19 维，与训练数据 zarr 拼接顺序一致。

    点云处理与训练数据转换（convert_pointcloud_hdf5_to_zarr.py）完全对齐：
      - 提取 segmentation 字段，过滤桌面(seg=16)/地面(seg=17)背景
      - 只保留机械臂(seg 1-15)和物块(seg 18)
    """
    xyzw     = _to_numpy_pcd(raw_obs['pointcloud']['xyzw'])
    qpos     = _to_numpy_pcd(raw_obs['agent']['qpos'])
    goal_pos = _to_numpy_pcd(raw_obs['extra']['goal_pos'])
    tcp_pose = _to_numpy_pcd(raw_obs['extra']['tcp_pose'])

    # 提取 segmentation 用于背景过滤（与训练数据对齐）
    seg = None
    if 'segmentation' in raw_obs['pointcloud']:
        seg = _to_numpy_pcd(raw_obs['pointcloud']['segmentation'])

    # GPU sim 可能有 batch 维 (1, D)
    if qpos.ndim == 2:
        qpos = qpos[0]
    if goal_pos.ndim == 2:
        goal_pos = goal_pos[0]
    if tcp_pose.ndim == 2:
        tcp_pose = tcp_pose[0]

    agent_pos = np.concatenate([qpos, goal_pos, tcp_pose], axis=0)  # (19,)

    return {
        'point_cloud': sample_pcd(xyzw, n_points, seg),  # (n_points, 3)，已过滤背景
        'agent_pos':   agent_pos,                          # (19,)
    }


def stack_pointcloud_obs(obs_deque, n_obs_steps: int, device) -> dict:
    """
    将最近 n_obs_steps 帧堆叠为策略输入格式（FlowPolicy predict_action 所需）:
        point_cloud: Tensor(1, n_obs_steps, n_points, 3)
        agent_pos:   Tensor(1, n_obs_steps, 19)

    obs_deque 中每个元素为 extract_pointcloud_obs 返回的单帧字典。
    若帧数不足则复制最早帧填充。
    """
    obs_list = list(obs_deque)[-n_obs_steps:]
    # 不足则用最早帧填充
    while len(obs_list) < n_obs_steps:
        obs_list.insert(0, obs_list[0])

    pcd_stack = np.stack([o['point_cloud'] for o in obs_list])  # (T, n_pts, 3)
    pos_stack = np.stack([o['agent_pos']   for o in obs_list])  # (T, 19)

    return {
        'point_cloud': torch.from_numpy(pcd_stack).unsqueeze(0).to(device),  # (1, T, n_pts, 3)
        'agent_pos':   torch.from_numpy(pos_stack).unsqueeze(0).to(device),  # (1, T, 19)
    }


def stack_pointcloud_obs_batch(obs_deque_list, n_obs_steps: int, device) -> dict:
    """
    批量版本：将多个环境的 obs_deque 堆叠为 batch 形式
        point_cloud: Tensor(B, n_obs_steps, n_points, 3)
        agent_pos:   Tensor(B, n_obs_steps, 19)
    """
    pcd_batch = []
    pos_batch = []
    for obs_deque in obs_deque_list:
        obs_list = list(obs_deque)[-n_obs_steps:]
        while len(obs_list) < n_obs_steps:
            obs_list.insert(0, obs_list[0])
        pcd_batch.append(np.stack([o['point_cloud'] for o in obs_list]))  # (T, n_pts, 3)
        pos_batch.append(np.stack([o['agent_pos']   for o in obs_list]))  # (T, 19)

    return {
        'point_cloud': torch.from_numpy(np.stack(pcd_batch)).to(device),  # (B, T, n_pts, 3)
        'agent_pos':   torch.from_numpy(np.stack(pos_batch)).to(device),  # (B, T, 19)
    }


# ============================================================


def sample_complete_trajectories(
    env, policy_old,
    num_state_groups, samples_per_group,
    config, accelerator, seed_offset,
    n_obs_steps=2, n_points=512,
    epoch=0,
):
    """
    FlowPolicy 点云版完整轨迹采样：从初始状态执行到任务完成或最大步数

    使用预训练 FlowPolicy（含点云编码器）作为 old policy 进行 off-policy 数据收集。

    参数
    ----
    env          : 并行 ManiSkill GPU 环境（obs_mode='pointcloud'）
    policy_old   : FlowPolicy，采样使用 old policy（eval 模式）
    num_state_groups  : GRPO 分组数（不同初始状态数）
    samples_per_group : 每组轨迹数（同一初始状态采样次数）
    config       : 训练配置
    accelerator  : Accelerator 实例
    seed_offset  : 随机种子偏移（用于每 epoch 生成不同 episode_id）
    n_obs_steps  : 观测历史帧数（与预训练保持一致，默认2）
    n_points     : 点云采样点数（默认512）
    epoch        : 当前训练 epoch（用于日志）

    返回
    ----
    List[dict]，每个 dict 包含单条轨迹的：
      - obs_list         : List[dict]，每次 FlowPolicy 调用时的 policy_obs（含 point_cloud, agent_pos）
      - clean_actions_list: List[Tensor(H, Da)]，归一化空间的干净动作序列
      - reward           : float，累积相对奖励
      - num_steps        : int，轨迹总执行步数
      - num_flow_calls   : int，FlowPolicy 调用次数
      - first_done_step  : int，首次 done 的步数（-1表示未 done）
      - terminated       : bool，是否成功完成
      - group_id         : int，GRPO 组 ID
      - episode_id       : int，初始状态的 episode_id
      - obs_history      : List[dict]，每执行步的 policy_obs（供滑动窗口训练）
      - action_history   : List[Tensor(1, Da)]，每执行步的单步归一化动作
    """
    policy_old.eval()
    device = accelerator.device
    normalizer = policy_old.normalizer  # 共享的 LinearNormalizer
    n_action_steps = getattr(config, 'n_action_steps', 8)  # 每次 predict_action 返回的执行步数

    num_envs = num_state_groups * samples_per_group

    # 生成 num_state_groups 个不同的 episode_id
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed_offset)
    episode_ids = torch.randint(
        low=0, high=10000,
        size=(num_state_groups,),
        generator=generator,
        dtype=torch.int64,
    )

    # 每个 episode_id 重复 samples_per_group 次（同组使用相同初始状态）
    all_episode_ids = episode_ids.repeat_interleave(samples_per_group)
    all_episode_ids_list = all_episode_ids.tolist()
    all_seeds = all_episode_ids_list  # seed = episode_id 确保完全固定初始状态

    # 重置并行环境
    raw_obs, info = env.reset(seed=all_seeds, options={'episode_id': all_episode_ids_list})

    # ── 初始化每个环境的 obs_deque（用 reset 后首帧填充） ──
    # raw_obs 来自并行环境：批量格式 {'pointcloud': {'xyzw': (B,N,4), 'segmentation': (B,N,1)}, ...}
    # 需要逐环境拆解
    def split_batch_obs(raw_obs_batch, env_idx):
        """
        从批量 raw_obs 中提取第 env_idx 个环境的单环境 obs。
        支持任意嵌套深度（ManiSkill obs 结构可能包含多层 dict）。
        """
        def _index_val(v):
            if isinstance(v, dict):
                # 递归处理嵌套 dict
                return {k: _index_val(sub) for k, sub in v.items()}
            elif isinstance(v, torch.Tensor):
                return v[env_idx]
            else:
                # numpy array / list / tuple 等
                try:
                    return v[env_idx]
                except (IndexError, KeyError, TypeError):
                    return v  # 无法按 env_idx 索引则原样返回（标量/非批次值）

        return {key: _index_val(val) for key, val in raw_obs_batch.items()}

    env_obs_deques = []
    for env_idx in range(num_envs):
        single_raw = split_batch_obs(raw_obs, env_idx)
        step_obs = extract_pointcloud_obs(single_raw, n_points)
        # 用首帧重复填充 n_obs_steps 帧
        obs_deque = _deque([step_obs] * n_obs_steps, maxlen=n_obs_steps + 1)
        env_obs_deques.append(obs_deque)

    # ── 轨迹跟踪变量 ──
    env_obs_list = [[] for _ in range(num_envs)]          # 每次 FlowPolicy 调用的 policy_obs (dict)
    env_clean_actions_list = [[] for _ in range(num_envs)]  # 每次 FlowPolicy 调用的归一化动作
    env_obs_history = [[] for _ in range(num_envs)]         # 每步的 policy_obs（滑动窗口训练用）
    env_action_history = [[] for _ in range(num_envs)]      # 每步的单步归一化动作
    env_step_rewards = [[] for _ in range(num_envs)]
    env_step_dones = [[] for _ in range(num_envs)]

    env_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    env_first_done_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    env_prev_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)

    max_steps = config.max_episode_steps
    flow_call_count = 0

    # ── 主采样循环 ──
    while not env_done.all() and flow_call_count * n_action_steps < max_steps:
        active_mask = ~env_done

        if active_mask.any():
            # 1. 从各环境的 obs_deque 构造批量 policy_obs（全部环境，含已 done 的）
            policy_obs_batch = stack_pointcloud_obs_batch(env_obs_deques, n_obs_steps, device)
            # policy_obs_batch: {'point_cloud': (B, T, 512, 3), 'agent_pos': (B, T, 19)}

            # 2. 调用 FlowPolicy 生成动作（使用 old policy，no_grad）
            with torch.no_grad():
                result = policy_old.predict_action(policy_obs_batch)

            # result['action']: (B, n_action_steps, Da) 真实动作空间（已反归一化）
            # result['action_pred']: (B, horizon, Da) 真实动作空间（已反归一化，FlowPolicy.predict_action 内部调用了 unnormalize）
            actions_exec = result['action']  # (B, n_action_steps, Da) 真实空间

            # 归一化空间的 clean_actions（horizon 长度，供 FlowNFT loss 使用）
            # ⚠️  predict_action 返回的 action_pred 已经是真实空间，需要重新归一化到归一化空间
            #     才能与 UNet 预训练时的 x1 分布一致（UNet 训练目标始终是归一化动作）
            action_pred_full_real = result['action_pred']  # (B, horizon, Da) 真实动作空间
            action_pred_full = policy_old.normalizer['action'].normalize(
                action_pred_full_real.to(device)
            )  # (B, horizon, Da) 归一化空间
            # 从 horizon 中截取 n_obs_steps-1 到 n_obs_steps-1+n_action_steps 对应段
            # （与 FlowPolicy.predict_action 的逻辑一致：start = To-1）
            start_idx = max(n_obs_steps - 1, 0)
            clean_actions_batch = action_pred_full[:, start_idx:start_idx + config.action_horizon]
            # (B, action_horizon, Da) 归一化空间

            # 3. 保存每次 FlowPolicy 调用的 obs 和 clean_actions
            for env_idx in range(num_envs):
                if active_mask[env_idx]:
                    # 提取单环境 policy_obs（保持 dict 格式，batch dim=1）
                    single_policy_obs = {
                        'point_cloud': policy_obs_batch['point_cloud'][env_idx:env_idx+1].cpu(),  # (1, T, 512, 3)
                        'agent_pos':   policy_obs_batch['agent_pos'][env_idx:env_idx+1].cpu(),    # (1, T, 19)
                    }
                    env_obs_list[env_idx].append(single_policy_obs)
                    # clean_actions: 截取整个 horizon 段的归一化动作
                    env_clean_actions_list[env_idx].append(
                        clean_actions_batch[env_idx].cpu().clone()  # (action_horizon, Da)
                    )

            # 4. 逐步执行 n_action_steps 个动作
            for j in range(n_action_steps):
                if env_done.all():
                    break

                step_active_mask = ~env_done
                action_step = actions_exec[:, j].contiguous()  # (B, Da) 真实动作空间

                # 4a. 保存本步的 policy_obs 和归一化 action（供滑动窗口）
                for env_idx in range(num_envs):
                    if step_active_mask[env_idx]:
                        single_step_obs = {
                            'point_cloud': policy_obs_batch['point_cloud'][env_idx:env_idx+1].cpu(),
                            'agent_pos':   policy_obs_batch['agent_pos'][env_idx:env_idx+1].cpu(),
                        }
                        # 归一化单步动作（从 clean_actions_batch 中取 j 步）
                        single_step_action_norm = clean_actions_batch[env_idx, j:j+1].cpu().clone()  # (1, Da)
                        env_obs_history[env_idx].append(single_step_obs)
                        env_action_history[env_idx].append(single_step_action_norm)

                # 4b. 与环境交互
                raw_obs, reward, terminated, truncated, info = env.step(action_step)

                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, device=device)
                if not isinstance(terminated, torch.Tensor):
                    terminated = torch.tensor(terminated, device=device)

                # 相对奖励（消除累积效应，参考 RLinf）
                reward_diff = reward - env_prev_reward
                env_prev_reward = reward.clone()

                # 4c. 记录奖励和 done（只在未完成的环境中记录）
                for env_idx in range(num_envs):
                    if step_active_mask[env_idx]:
                        env_step_rewards[env_idx].append(reward_diff[env_idx].item())
                        env_step_dones[env_idx].append(terminated[env_idx].item())

                # 4d. 更新 obs_deque（用下一帧点云观测更新滑动窗口）
                for env_idx in range(num_envs):
                    if step_active_mask[env_idx]:
                        single_raw = split_batch_obs(raw_obs, env_idx)
                        step_obs_new = extract_pointcloud_obs(single_raw, n_points)
                        env_obs_deques[env_idx].append(step_obs_new)

                # 4e. 更新 done 状态
                newly_done = terminated & (env_first_done_step == -1)
                env_first_done_step[newly_done] = env_steps[newly_done]
                env_done = env_done | terminated  # 累积 done（禁止 auto_reset 后保持状态）
                env_steps[step_active_mask] += 1

        flow_call_count += 1

    # ── 构建样本列表 ──
    all_samples = []
    for env_idx in range(num_envs):
        if len(env_obs_list[env_idx]) == 0:
            continue

        group_id = env_idx // samples_per_group

        # 倒序累积相对奖励（参考 RLinf calculate_scores）
        rewards = env_step_rewards[env_idx]
        dones = env_step_dones[env_idx]
        cumulative_reward = 0.0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                cumulative_reward = 0.0
            cumulative_reward += rewards[step]

        # ── 第一层折扣：轨迹级 γ^T Reward Shaping ──────────────────────────────
        # 目的：区分相同最终奖励但步数不同的轨迹（83步成功 vs 303步成功）
        # 公式：shaped_reward = raw_reward * gamma_traj^num_steps
        # 效果：步数越少 → γ^T 越大 → shaped_reward 越高，反之越低
        num_steps_this_env = env_steps[env_idx].item()
        use_traj_discount = getattr(config.train, 'use_traj_gamma_discount', False)
        if use_traj_discount:
            gamma_traj = getattr(config.train, 'gamma_traj', 0.999)
            shaped_reward = cumulative_reward * (gamma_traj ** num_steps_this_env)
        else:
            shaped_reward = cumulative_reward

        env_episode_id = all_episode_ids_list[env_idx]

        all_samples.append({
            "obs_list": env_obs_list[env_idx],           # List[dict{'point_cloud','agent_pos'}]
            "clean_actions_list": env_clean_actions_list[env_idx],  # List[Tensor(horizon, Da)]
            "reward": shaped_reward,
            "num_steps": env_steps[env_idx].item(),
            "num_flow_calls": len(env_obs_list[env_idx]),
            "first_done_step": env_first_done_step[env_idx].item(),
            "terminated": env_done[env_idx].item(),
            "group_id": group_id,
            "episode_id": env_episode_id,
            # 滑动窗口历史（逐步 obs 和归一化 action，供 FlowNFT 训练）
            "obs_history": env_obs_history[env_idx],       # List[dict]
            "action_history": env_action_history[env_idx], # List[Tensor(1, Da)]
        })

    return all_samples

def compute_episode_level_advantages(samples, num_train_timesteps, config):
    """
    基于完整轨迹的GRPO advantage计算

    支持两种模式（由 config.stage.advantage_type 控制）：

    "group"（原始模式）：
    - 按GRPO组分组（组内相同初始状态，每组多条轨迹）
    - 使用相对奖励的总和进行组内均值/方差归一化
    - Success Rate Filter：过滤全成功或全失败的组

    "single"（二值模式）：
    - 每组只有一条轨迹
    - 成功（reward >= config.stage.max_reward）则 advantage = +1
    - 失败则 advantage = -1
    - 无需过滤，所有样本均参与训练
    """
    num_state_groups = config.episode.num_state_groups
    samples_per_group = config.episode.samples_per_group
    advantage_type = getattr(config.stage, "advantage_type", "group")
    
    # 按组分配样本
    group_samples = [[] for _ in range(num_state_groups)]
    for sample in samples:
        group_id = sample["group_id"]
        group_samples[group_id].append(sample)
    
    # 统计过滤信息
    filtered_groups = 0
    filtered_samples = 0

    if advantage_type == "single":
        # ── Single 模式：每组一条轨迹，二值 advantage ──────────────────────
        max_reward = getattr(config.stage, "max_reward", 5.0)
        for group_id, group in enumerate(group_samples):
            if len(group) == 0:
                continue
            for sample in group:
                sample["filtered"] = False
                is_success = sample["reward"] >= max_reward
                sample["advantage"] = 3.0 if is_success else -3.0
    else:
        # ── Group 模式（原始逻辑）：组内均值/方差归一化 ────────────────────
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
                for sample in group:
                    sample["filtered"] = True
                    sample["advantage"] = 0.0
                continue
            
            # ✅ 使用相对奖励的总和（已经消除累积效应）
            group_rewards = np.array([s["reward"] for s in group])
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            
            for i, sample in enumerate(group):
                sample["filtered"] = False
                total_reward = sample["reward"]
                advantage = (total_reward - group_mean) / group_std
                advantage *= config.stage.advantage_scale
                sample["advantage"] = advantage  # scalar float
    
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
        flat, uprate, uphold = 0, 0.00001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.001, 0.999
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")
    
    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def update_old_model(
    flowpolicy_current,
    flowpolicy_old,
    config, accelerator, global_step, epoch,
    log_tag="Old model 更新",
    # 兼容旧调用签名（不再使用，保留以避免调用报错）
    action_unet=None, action_decoder=None,
    action_unet_old=None, action_decoder_old=None,
    action_unet_ref=None, action_decoder_ref=None,
):
    """
    EMA 更新 old model（FlowPolicy 完整版）：
        θ_old ← η·θ_old + (1-η)·θ_current
    同时更新 obs_encoder 和 ConditionalUnet1D 两部分权重。
    （DiffusionNFT Algorithm 1, Line 14）

    参数
    ----
    flowpolicy_current : FlowPolicy，当前可训练模型
    flowpolicy_old     : FlowPolicy，old policy（会被 EMA 更新）
    """
    with torch.no_grad():
        decay = return_decay(global_step, config.train.decay_type)

        # ── 1. 计算整体 FlowPolicy 参数 L2 距离（监控用）──
        current_old_distance = 0.0
        current_ref_distance = 0.0  # ref 不在此更新，记为 0

        cur_params  = list(unwrap_model(flowpolicy_current, accelerator).parameters())
        old_params  = list(flowpolicy_old.parameters())

        for cur_p, old_p in zip(cur_params, old_params):
            current_old_distance += (cur_p - old_p).pow(2).sum().item()
        current_old_distance = np.sqrt(current_old_distance)

        # ── 2. EMA 更新 obs_encoder ──
        for src_p, tgt_p in zip(
            unwrap_model(flowpolicy_current, accelerator).obs_encoder.parameters(),
            flowpolicy_old.obs_encoder.parameters()
        ):
            tgt_p.data.copy_(tgt_p.data * decay + src_p.data * (1.0 - decay))

        # ── 3. EMA 更新 ConditionalUnet1D（model）──
        for src_p, tgt_p in zip(
            unwrap_model(flowpolicy_current, accelerator).model.parameters(),
            flowpolicy_old.model.parameters()
        ):
            tgt_p.data.copy_(tgt_p.data * decay + src_p.data * (1.0 - decay))

    if accelerator.is_local_main_process:
        logger.info(
            f"  [{log_tag}] Step {global_step} / Epoch {epoch}: "
            f"decay={decay:.4f}, dist(cur,old)={current_old_distance:.4f}"
        )

    if accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "model/current_old_distance": current_old_distance,
                "model/current_ref_distance": current_ref_distance,
                "model/old_decay":            decay,
            }, step=global_step)



def _extract_global_cond(obs_encoder, obs_dict_batch, normalizer,
                         n_obs_steps, device, no_grad=False):
    """
    给定批量 obs_dict（含 point_cloud/agent_pos），通过 obs_encoder 提取 global_cond。

    参数
    ----
    obs_encoder    : FlowPolicyEncoder（FlowPolicy.obs_encoder）
    obs_dict_batch : dict，键为 'point_cloud' (B, T, N, 3) 和 'agent_pos' (B, T, 19)
                     所有张量应已在 device 上
    normalizer     : FlowPolicy.normalizer（LinearNormalizer）
    n_obs_steps    : 取前 n_obs_steps 帧用于编码
    device         : 目标设备
    no_grad        : 若 True 则在 no_grad 下执行（old/ref 模型）

    返回
    ----
    global_cond : Tensor (B, feat_dim * n_obs_steps)
    """
    def _do_extract():
        # 1. 归一化观测
        nobs = normalizer.normalize(obs_dict_batch)
        # 2. 截取前 n_obs_steps 帧，reshape 为 (B*T, ...)
        B = nobs['point_cloud'].shape[0]
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        # 3. 通过 obs_encoder 提取特征
        obs_feat = obs_encoder(this_nobs)       # (B*T, feat_dim)
        # 4. reshape 为 (B, T*feat_dim)，作为 global_cond
        feat_dim = obs_feat.shape[-1]
        global_cond = obs_feat.reshape(B, n_obs_steps * feat_dim)
        return global_cond

    if no_grad:
        with torch.no_grad():
            return _do_extract()
    else:
        return _do_extract()


def train_on_episode_samples_diffusionnft(
    samples,
    flowpolicy_current,    # FlowPolicy，可训练（current 模型）
    flowpolicy_old,        # FlowPolicy，不可训练（old 模型，用于 off-policy baseline）
    flowpolicy_ref,        # FlowPolicy，不可训练（ref 模型，用于 KL 正则）
    optimizer, accelerator, config, ema, trainable_parameters,
    global_step, epoch,
    n_obs_steps=2,
):
    """
    FlowNFT 风格训练函数（单步去噪，无多步 Rollout）

    核心流程：
    1. 从 samples 中展平 (policy_obs_dict, clean_action, advantage) 三元组
    2. 随机采样时间步 t ∈ (cfm_eps, 1)
    3. 构造加噪动作 xt = t * x1 + (1-t) * z
    4. 三模型提取 global_cond（current 有梯度，old/ref 无梯度）
    5. 三模型在 xt 处预测速度场 vt（current/old/ref）
    6. DiffusionNFT β 组合：正/负速度场
    7. 单步 Euler 去噪：x1_pred = xt + (1-t) * vt
    8. Advantage 加权的 MSE policy loss + KL loss

    参数
    ----
    samples             : sample_complete_trajectories 返回的样本列表
    flowpolicy_current  : FlowPolicy，当前训练模型
    flowpolicy_old      : FlowPolicy，old policy（不训练）
    flowpolicy_ref      : FlowPolicy，ref policy（不训练，KL 基准）
    optimizer           : 优化器
    accelerator         : Accelerator 实例
    config              : 训练配置
    ema                 : EMA 对象
    trainable_parameters: 可训练参数列表
    global_step         : 全局步计数
    epoch               : 当前 epoch
    n_obs_steps         : 观测帧数（与采样阶段保持一致）
    """
    # ── 模型模式设置 ──
    flowpolicy_current.train()
    flowpolicy_current.obs_encoder.train()
    flowpolicy_current.model.train()
    flowpolicy_old.eval()
    flowpolicy_ref.eval()

    action_unet       = flowpolicy_current.model   # ConditionalUnet1D
    action_unet_old   = flowpolicy_old.model
    action_unet_ref   = flowpolicy_ref.model
    obs_encoder       = flowpolicy_current.obs_encoder
    obs_encoder_old   = flowpolicy_old.obs_encoder
    obs_encoder_ref   = flowpolicy_ref.obs_encoder
    normalizer        = flowpolicy_current.normalizer  # 共享归一化器

    cfm_eps = getattr(config.train, 'cfm_eps', 0.01)
    device  = accelerator.device

    # ── 阶段1：展平样本 ──
    # 每个样本包含多次 FlowPolicy 调用（obs_list / clean_actions_list）
    # 使用滑动窗口模式将 obs_history / action_history 展开为独立训练样本
    batch_obs_dicts  = []   # List[dict{'point_cloud':(1,T,N,3), 'agent_pos':(1,T,19)}]
    batch_x1         = []   # List[Tensor(horizon, Da)]
    batch_advantages = []

    action_horizon        = config.action_horizon
    sliding_window_stride = config.train.sliding_window_stride

    # ── 第二层折扣配置读取 ──────────────────────────────────────────────────────
    # 目的：轨迹内信用分配——越靠近成功时刻的 flow call 获得越高的 advantage 权重
    # 公式：adv_i = base_adv * gamma_temporal^(num_steps - step_i)
    # 效果：长轨迹早期的探索动作 advantage 被大幅打折，临近成功的动作权重最高
    use_flow_discount  = getattr(config.train, 'use_flow_temporal_discount', False)
    gamma_temporal     = getattr(config.train, 'gamma_temporal', 0.99)
    n_action_steps_cfg = getattr(config, 'n_action_steps', action_horizon)  # 每次 flow call 执行的环境步数

    for sample in samples:
        if sample.get("filtered", False):
            continue

        traj_adv  = sample["advantage"]
        num_steps = sample.get("num_steps", 0)  # 轨迹总步数，用于第二层折扣

        obs_history    = sample.get("obs_history", [])
        action_history = sample.get("action_history", [])

        # 若无滑动窗口历史，退化为每次 FlowPolicy 调用一个样本
        if len(obs_history) == 0 or len(action_history) == 0:
            for flow_idx in range(sample["num_flow_calls"]):
                batch_obs_dicts.append(sample["obs_list"][flow_idx])
                # clean_actions_list 中每项形状 (action_horizon, Da)
                batch_x1.append(sample["clean_actions_list"][flow_idx])
                # ── 第二层折扣：基于 flow call 在轨迹中的位置 ──
                if use_flow_discount:
                    # 本次 flow call 对应的起始环境步
                    step_i = flow_idx * n_action_steps_cfg
                    # 距成功时刻还有多少步（越小 → 越靠近成功 → 折扣越高）
                    steps_remaining = max(num_steps - step_i, 0)
                    temporal_discount = gamma_temporal ** steps_remaining
                    batch_advantages.append(traj_adv * temporal_discount)
                else:
                    batch_advantages.append(traj_adv)
            continue

        # 滑动窗口：action_history 中每项为 (1, Da) 单步动作
        total_steps = len(obs_history)
        for s in range(0, total_steps - action_horizon + 1, sliding_window_stride):
            e = s + action_horizon
            # 窗口起点的 policy_obs（已含 T 帧历史）
            batch_obs_dicts.append(obs_history[s])
            # 从单步 action 拼接成 (action_horizon, Da)
            window_actions = torch.cat(action_history[s:e], dim=0)  # (action_horizon, Da)
            batch_x1.append(window_actions)
            # ── 第二层折扣：基于滑动窗口起始步在轨迹中的位置 ──
            if use_flow_discount:
                # s 是窗口在 obs_history 中的起始索引（即对应的环境步编号）
                steps_remaining = max(num_steps - s, 0)
                temporal_discount = gamma_temporal ** steps_remaining
                batch_advantages.append(traj_adv * temporal_discount)
            else:
                batch_advantages.append(traj_adv)

    if len(batch_obs_dicts) == 0:
        logger.warning(f"Epoch {epoch}: 没有有效样本，跳过训练")
        return global_step, 0.0

    # ── 阶段2：打包 obs_dict 批次 ──
    # 各样本的 point_cloud: (1, T, N, 3) → stack → (N_samples, T, N, 3)
    stacked_pcd = torch.cat([d['point_cloud'] for d in batch_obs_dicts], dim=0).to(device)
    stacked_pos = torch.cat([d['agent_pos']   for d in batch_obs_dicts], dim=0).to(device)
    obs_cond_dict = {'point_cloud': stacked_pcd, 'agent_pos': stacked_pos}

    # 归一化空间的 clean_actions
    x1_all = torch.stack(batch_x1, dim=0).to(device)  # (N_samples, H, Da)
    adv_all = torch.tensor(batch_advantages, device=device, dtype=torch.float32)

    total_batch_size = x1_all.shape[0]
    logger.info(f"Epoch {epoch}: 训练 {total_batch_size} 个 FlowNFT 样本")

    # ── 阶段3：打乱数据 ──
    perm = torch.randperm(total_batch_size, device=device)
    stacked_pcd = stacked_pcd[perm]
    stacked_pos = stacked_pos[perm]
    x1_all  = x1_all[perm]
    adv_all = adv_all[perm]

    # ── 阶段4：分批训练 ──
    num_batches = max(1, total_batch_size // config.train.batch_size)
    effective_grad_accum_steps = config.train.gradient_accumulation_steps
    current_accumulated_steps = 0
    info = defaultdict(list)
    epoch_total_losses = []

    for batch_idx in tqdm(
        range(num_batches),
        desc=f"Epoch {epoch}: FlowNFT 单步去噪训练",
        position=0,
        disable=not accelerator.is_local_main_process
    ):
        s = batch_idx * config.train.batch_size
        e = min(s + config.train.batch_size, total_batch_size)

        # 当前批次的 obs_dict 和 x1
        batch_pcd = stacked_pcd[s:e]   # (B, T, N, 3)
        batch_pos = stacked_pos[s:e]   # (B, T, 19)
        batch_obs = {'point_cloud': batch_pcd, 'agent_pos': batch_pos}
        batch_x1_cur = x1_all[s:e]     # (B, H, Da) 归一化空间
        batch_adv    = adv_all[s:e]    # (B,)

        B = batch_x1_cur.shape[0]

        with accelerator.autocast():
            # ── 1. 采样 t ∈ (cfm_eps, 1) ─────────────────────────────────────
            t        = torch.rand(B, device=device) * (1.0 - cfm_eps) + cfm_eps
            t_expand = t.view(-1, 1, 1)  # (B, 1, 1)

            # ── 2. 加噪：xt = t * x1 + (1-t) * z ─────────────────────────────
            z  = torch.randn_like(batch_x1_cur.float())
            xt = t_expand * batch_x1_cur.float() + (1.0 - t_expand) * z

            # ── 3. 提取 global_cond（obs 编码） ──────────────────────────────
            # current（有梯度）
            global_cond_current = _extract_global_cond(
                obs_encoder, batch_obs, normalizer, n_obs_steps, device, no_grad=False
            )
            # old（无梯度）
            global_cond_old = _extract_global_cond(
                obs_encoder_old, batch_obs, normalizer, n_obs_steps, device, no_grad=True
            )
            # ref（无梯度）
            global_cond_ref = _extract_global_cond(
                obs_encoder_ref, batch_obs, normalizer, n_obs_steps, device, no_grad=True
            )

            # ── 4. 三模型速度场预测（timestep 缩放 *99，与 FlowPolicy 一致） ──
            # 注意：FlowPolicy.compute_loss 中使用 `timestep * 99` 传入 UNet
            with torch.no_grad():
                vt_old = action_unet_old(
                    sample=xt, timestep=t * 99,
                    global_cond=global_cond_old.detach()
                )

            vt_current = action_unet(
                sample=xt, timestep=t * 99,
                global_cond=global_cond_current  # 有梯度
            )

            with torch.no_grad():
                vt_ref = action_unet_ref(
                    sample=xt, timestep=t * 99,
                    global_cond=global_cond_ref.detach()
                )

            # ── 5. DiffusionNFT β 正/负速度场组合 ─────────────────────────
            beta = config.train.beta
            positive_velocity = beta * vt_current + (1.0 - beta) * vt_old.detach()
            implicit_negative_velocity = (1.0 + beta) * vt_old.detach() - beta * vt_current

            # ── 6. 单步 Euler 去噪：x1_pred = xt + (1-t) * vt ─────────────
            x1_pred_pos = xt + (1.0 - t_expand) * positive_velocity
            x1_pred_neg = xt + (1.0 - t_expand) * implicit_negative_velocity

            # ── 7. Advantage 处理 → r ∈ [0, 1] ────────────────────────────
            adv_clip = torch.clamp(batch_adv,
                                   -config.train.adv_clip_max,
                                    config.train.adv_clip_max)

            adv_mode = getattr(config.train, "adv_mode", None)
            if adv_mode == "positive_only":
                adv_clip = torch.clamp(adv_clip, 0, config.train.adv_clip_max)
            elif adv_mode == "negative_only":
                adv_clip = torch.clamp(adv_clip, -config.train.adv_clip_max, 0)
            elif adv_mode == "one_only":
                adv_clip = torch.where(adv_clip > 0,
                                       torch.ones_like(adv_clip),
                                       torch.zeros_like(adv_clip))
            elif adv_mode == "binary":
                adv_clip = torch.sign(adv_clip)

            r = torch.clamp((adv_clip / config.train.adv_clip_max) / 2.0 + 0.5, 0, 1)

            # ── 8. Policy loss（MSE，advantage 加权） ─────────────────────
            pos_loss = ((x1_pred_pos - batch_x1_cur) ** 2).mean(dim=(1, 2))
            neg_loss = ((x1_pred_neg - batch_x1_cur) ** 2).mean(dim=(1, 2))
            policy_loss = (r * pos_loss / beta + (1.0 - r) * neg_loss / beta).mean()

            # ── 9. KL loss（速度场空间：current ↔ ref） ───────────────────
            kl_div_loss = ((vt_current - vt_ref) ** 2).mean(dim=(1, 2))
            kl_loss     = config.train.kl_weight * kl_div_loss.mean()

            # ── 10. Total loss ─────────────────────────────────────────────
            loss = policy_loss + kl_loss

        # ── [DEBUG] Epoch 0 前3个 batch 详细诊断 ───────────────────────────
        if epoch == 0 and batch_idx < 3 and accelerator.is_local_main_process:
            sep = "─" * 64
            logger.info(f"\n{sep}")
            logger.info(f"[DEBUG] Epoch {epoch}  Batch {batch_idx}  B={B}")
            logger.info(f"{sep}")

            def _stat(name, t):
                t_f = t.detach().float()
                logger.info(
                    f"  {name:<30s}  shape={tuple(t_f.shape)}"
                    f"  mean={t_f.mean().item():+.4f}"
                    f"  std={t_f.std().item():.4f}"
                    f"  min={t_f.min().item():+.4f}"
                    f"  max={t_f.max().item():+.4f}"
                )

            # ── 输入数据 ──────────────────────────────────────────────────
            logger.info("  [输入]")
            _stat("batch_x1 (clean_action)", batch_x1_cur)
            _stat("t (timestep)", t)
            _stat("xt (noisy_action)", xt)
            _stat("batch_adv (advantage)", batch_adv)
            _stat("r (weight 0→1)", r)

            # ── 速度场预测 ────────────────────────────────────────────────
            logger.info("  [速度场预测]")
            _stat("vt_current", vt_current.detach())
            _stat("vt_old    ", vt_old)
            _stat("vt_ref    ", vt_ref)
            _stat("positive_velocity", positive_velocity.detach())
            _stat("negative_velocity", implicit_negative_velocity.detach())

            # ── 去噪预测 ──────────────────────────────────────────────────
            logger.info("  [去噪预测]")
            _stat("x1_pred_pos", x1_pred_pos.detach())
            _stat("x1_pred_neg", x1_pred_neg.detach())
            _stat("x1_pred_pos - x1 (误差)", (x1_pred_pos.detach() - batch_x1_cur))
            _stat("x1_pred_neg - x1 (误差)", (x1_pred_neg.detach() - batch_x1_cur))

            # ── global_cond ───────────────────────────────────────────────
            logger.info("  [global_cond]")
            _stat("global_cond_current", global_cond_current.detach())
            _stat("global_cond_old    ", global_cond_old)
            _stat("global_cond_ref    ", global_cond_ref)

            # ── Loss 分解 ─────────────────────────────────────────────────
            logger.info("  [Loss 分解]")
            logger.info(f"  {'pos_loss (per sample)':<30s}  mean={pos_loss.mean().item():.4f}  max={pos_loss.max().item():.4f}")
            logger.info(f"  {'neg_loss (per sample)':<30s}  mean={neg_loss.mean().item():.4f}  max={neg_loss.max().item():.4f}")
            logger.info(f"  {'policy_loss':<30s}  {policy_loss.item():.6f}")
            logger.info(f"  {'kl_div_loss (raw MSE)':<30s}  mean={kl_div_loss.mean().item():.6f}")
            logger.info(f"  {'kl_loss (weighted)':<30s}  {kl_loss.item():.6f}")
            logger.info(f"  {'total_loss':<30s}  {loss.item():.6f}")
            logger.info(sep)

        # ── 11. 梯度累积 ──────────────────────────────────────────────────
        scaled_loss = loss / effective_grad_accum_steps
        accelerator.backward(scaled_loss)
        current_accumulated_steps += 1

        info["policy_loss"].append(policy_loss.detach())
        info["kl_div_loss"].append(kl_loss.detach())
        info["total_loss"].append(loss.detach())
        info["positive_loss"].append(pos_loss.mean().detach())
        info["negative_loss"].append(neg_loss.mean().detach())
        info["x1_error_pos"].append((x1_pred_pos - batch_x1_cur).abs().mean().detach())
        info["x1_error_neg"].append((x1_pred_neg - batch_x1_cur).abs().mean().detach())
        info["prediction_std"].append(vt_current.std().detach())
        info["kl_div_per_sample"].append(kl_div_loss.mean().detach())
        epoch_total_losses.append(loss.detach().item())

        if current_accumulated_steps % effective_grad_accum_steps == 0:
            # accelerator.clip_grad_norm_ 内部先对 fp16 梯度执行 unscale_()，
            # 再裁剪，返回裁剪前的真实梯度范数（unscaled）。
            # 不能在此之前调用 torch.nn.utils.clip_grad_norm_，
            # 否则会在 fp16 scale 后的梯度上计算，数值被放大 ~65536 倍。
            grad_norm_before = accelerator.clip_grad_norm_(
                trainable_parameters, config.train.max_grad_norm
            ).item()
            grad_norm_after  = min(grad_norm_before, config.train.max_grad_norm)
            grad_clipped     = 1.0 if grad_norm_before > config.train.max_grad_norm else 0.0

            with torch.no_grad():
                param_before = torch.cat([p.data.flatten() for p in trainable_parameters])

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                param_after = torch.cat([p.data.flatten() for p in trainable_parameters])
                param_update_norm  = (param_after - param_before).norm().item()
                param_norm         = param_after.norm().item()
                param_update_ratio = param_update_norm / (param_norm + 1e-8)

            info_aggregated = {k: torch.mean(torch.stack(v)) for k, v in info.items()}

            if global_step % 10 == 0 and accelerator.is_local_main_process:
                logger.info(
                    f"  Step {global_step}: "
                    f"loss={info_aggregated['total_loss'].item():.4f}, "
                    f"grad_norm(pre-clip)={grad_norm_before:.4f}"
                )

            if accelerator.is_main_process:
                import wandb
                if wandb.run is not None:
                    log_dict = {
                        "train/policy_loss":           info_aggregated["policy_loss"].item(),
                        "train/kl_div_loss":           info_aggregated["kl_div_loss"].item(),
                        "train/total_loss":            info_aggregated["total_loss"].item(),
                        "train/grad_norm_before_clip": grad_norm_before,
                        "train/grad_norm_after_clip":  grad_norm_after,
                        "train/epoch":                 epoch,
                        "loss/positive_loss":          info_aggregated["positive_loss"].item(),
                        "loss/negative_loss":          info_aggregated["negative_loss"].item(),
                        "loss/pos_neg_ratio":          (info_aggregated["positive_loss"] / (info_aggregated["negative_loss"] + 1e-8)).item(),
                        "loss/policy_kl_ratio":        (info_aggregated["policy_loss"] / (info_aggregated["kl_div_loss"] + 1e-8)).item(),
                        "prediction/x1_error_pos":     info_aggregated["x1_error_pos"].item(),
                        "prediction/x1_error_neg":     info_aggregated["x1_error_neg"].item(),
                        "prediction/output_std":       info_aggregated["prediction_std"].item(),
                        "prediction/kl_div_per_sample":info_aggregated["kl_div_per_sample"].item(),
                        "optim/grad_clipped":          grad_clipped,
                        "optim/param_update_norm":     param_update_norm,
                        "optim/param_norm":            param_norm,
                        "optim/param_update_ratio":    param_update_ratio,
                        "optim/learning_rate":         optimizer.param_groups[0]["lr"],
                    }
                    wandb.log(log_dict, step=global_step)

            global_step += 1
            current_accumulated_steps = 0
            info = defaultdict(list)

            if config.train.ema:
                ema.step(trainable_parameters, global_step)

            # Old model per-step EMA 更新
            old_model_update_mode = getattr(config.train, "old_model_update_mode", "per_step")
            if old_model_update_mode == "per_step" and global_step % 20 == 0:
                update_old_model(
                    flowpolicy_current=flowpolicy_current,
                    flowpolicy_old=flowpolicy_old,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step,
                    epoch=epoch,
                    log_tag="Old model 更新(per_step)"
                )

    avg_epoch_loss = np.mean(epoch_total_losses) if epoch_total_losses else 0.0
    return global_step, avg_epoch_loss



def eval_in_env(pipeline, action_unet, action_decoder, env, config, accelerator, global_step, num_test_episodes=3):
    """
    在环境中评估策略（action_decoder 已废弃，传 None 即可）
    """
    action_unet.eval()
    
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
                    output_type="tensor",   # 保持归一化空间，下方再反归一化
                    return_dict=True,
                    action_decoder=None,  # UNet 直接输出速度，无 Decoder
                    generator=eval_gen,
                )
                
                # 反归一化：pipeline 输出在归一化空间，需还原到真实动作空间
                actions_norm = result["actions"][0]
                actions = pipeline.unnormalize_action(actions_norm)
            
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
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense'),
        'sim_config': dict(
            sim_freq=config.env_kwargs.get('sim_freq', 100),
            control_freq=config.env_kwargs.get('control_freq', 20),
        )
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
    
    # 加载检查点（先加载，获取归一化参数后再建模型）
    test_logger.info(f"加载检查点: {checkpoint_path}")
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "model.pth")
    else:
        model_path = checkpoint_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # 兼容旧 checkpoint（无 obs_mean 键）
    ckpt_obs_mean    = checkpoint.get('obs_mean',    None)
    ckpt_obs_std     = checkpoint.get('obs_std',     None)
    ckpt_action_mean = checkpoint.get('action_mean', None)
    ckpt_action_std  = checkpoint.get('action_std',  None)

    # 创建模型（FlowPolicy UNet，无 ActionDecoder）
    test_logger.info("创建模型...")
    action_unet = FlowConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        kernel_size=5,
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
    pipeline.action_decoder = None  # 无 Decoder

    # 设置归一化参数（从 checkpoint 恢复）
    if ckpt_obs_mean is not None:
        pipeline.set_normalizer(
            ckpt_obs_mean.to(device), ckpt_obs_std.to(device),
            ckpt_action_mean.to(device), ckpt_action_std.to(device)
        )
        test_logger.info("✓ 归一化参数从 checkpoint 恢复")
    else:
        test_logger.warning("⚠ checkpoint 中无归一化参数，动作将不做反归一化（旧模型兼容）")

    action_unet.load_state_dict(checkpoint['action_unet_state_dict'])
    test_logger.info("模型加载成功！")

    # 设置为评估模式
    action_unet.eval()
    
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
                    output_type="tensor",   # 保持归一化空间，下方再反归一化
                    return_dict=True,
                    action_decoder=None,  # UNet 直接输出速度
                    generator=test_gen,
                )
                
                # 反归一化：还原到真实动作空间
                actions_norm = result["actions"][0]
                actions = pipeline.unnormalize_action(actions_norm).cpu().numpy()
            
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
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense'),
        'sim_config': dict(
            sim_freq=config.env_kwargs.get('sim_freq', 100),
            control_freq=config.env_kwargs.get('control_freq', 20),
        )
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
    
    # 加载预训练检查点（先加载，获取归一化参数后再建模型）
    if not pretrain_checkpoint or not os.path.exists(pretrain_checkpoint):
        raise ValueError(f"预训练检查点不存在: {pretrain_checkpoint}")

    test_logger.info(f"加载预训练检查点: {pretrain_checkpoint}")
    checkpoint = torch.load(pretrain_checkpoint, map_location=device)

    ckpt_obs_mean    = checkpoint.get('obs_mean',    None)
    ckpt_obs_std     = checkpoint.get('obs_std',     None)
    ckpt_action_mean = checkpoint.get('action_mean', None)
    ckpt_action_std  = checkpoint.get('action_std',  None)

    # 创建模型（FlowPolicy UNet，无 ActionDecoder）
    test_logger.info("创建模型...")
    action_unet = FlowConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        kernel_size=5,
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
    pipeline.action_decoder = None  # 无 Decoder

    # 设置归一化参数
    if ckpt_obs_mean is not None:
        pipeline.set_normalizer(
            ckpt_obs_mean.to(device), ckpt_obs_std.to(device),
            ckpt_action_mean.to(device), ckpt_action_std.to(device)
        )
        test_logger.info("✓ 归一化参数从 checkpoint 恢复")
    else:
        test_logger.warning("⚠ checkpoint 中无归一化参数（旧模型兼容）")

    action_unet.load_state_dict(checkpoint['action_unet_state_dict'])
    test_logger.info("✓ 模型加载成功")

    action_unet.eval()
    
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
                            output_type="tensor", return_dict=True,
                            action_decoder=None,
                            generator=test_generators,  # ✅ 传入generator列表
                        )
                
                # 反归一化：还原到真实动作空间
                actions_batch = pipeline.unnormalize_action(result["actions"])
                
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


def save_ckpt(save_dir, flowpolicy_current, epoch, accelerator, ema, trainable_parameters,
              config, flowpolicy_cfg=None,
              # 兼容旧调用签名
              action_unet=None, action_decoder=None):
    """
    保存 FlowPolicy RL 训练检查点（FlowPolicy 兼容格式）。

    检查点保存至 save_dir/checkpoint-epoch{epoch}/epoch={epoch:04d}.ckpt
    格式与 FlowPolicy 预训练检查点完全兼容，可直接用 load_flowpolicy_from_ckpt 加载。

    参数
    ----
    save_dir           : str，保存根目录（对应 config.save_dir）
    flowpolicy_current : FlowPolicy，当前训练模型
    epoch              : int，当前 epoch
    accelerator        : Accelerator 实例
    ema                : EMA 对象（可选，config.train.ema 为 True 时使用）
    trainable_parameters: 可训练参数列表
    config             : 训练配置
    flowpolicy_cfg     : OmegaConf DictConfig，原始 FlowPolicy 配置（来自预训练ckpt）
    """
    save_root = os.path.join(save_dir, f"checkpoint-epoch{epoch}")
    os.makedirs(save_root, exist_ok=True)
    ckpt_path = os.path.join(save_root, f"epoch={epoch:04d}.ckpt")

    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(trainable_parameters, store_temp=True)

        unwrapped = unwrap_model(flowpolicy_current, accelerator)

        payload = {
            'cfg': flowpolicy_cfg,
            'state_dicts': {
                'model': unwrapped.state_dict(),
            },
            'pickles': {
                'epoch':       dill.dumps(epoch),
                'global_step': dill.dumps(0),
            },
        }

        torch.save(payload, ckpt_path, pickle_module=dill)

        if config.train.ema:
            ema.copy_temp_to(trainable_parameters)

        logger.info(f"FlowPolicy RL 检查点已保存: {ckpt_path}")


def evaluate_pretrain_model(action_unet, action_decoder, pipeline, config, device, num_episodes=10):
    """
    在ManiSkill环境中评估预训练模型（无 ActionDecoder 版本）

    Parameters
    ----------
    action_unet    : FlowConditionalUnet1D
    action_decoder : 已废弃，传入 None 即可
    pipeline       : RobotActionPipeline
    config         : 配置对象
    device         : 设备
    num_episodes   : 测试 episode 数量
    """
    action_unet.eval()
    
    # 创建评估环境
    env_kwargs = {
        'sim_backend': config.env_kwargs.get('sim_backend', 'gpu'),
        'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode': config.env_kwargs.get('obs_mode', 'state'),
        'render_mode': 'rgb_array',
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense'),
        'sim_config': dict(
            sim_freq=config.env_kwargs.get('sim_freq', 100),
            control_freq=config.env_kwargs.get('control_freq', 20),
        )
    }
    
    env = gym.make(config.env_id, reconfiguration_freq=1, **env_kwargs)
    
    success_count = 0
    total_rewards = []
    
    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=20000 + episode_idx)
        
        # 转换观测为tensor
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(device, dtype=torch.float32)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        
        episode_reward = 0
        episode_length = 0
        done = False
        max_steps = config.max_episode_steps if hasattr(config, 'max_episode_steps') else 500
        
        while not done and episode_length < max_steps:
            with torch.no_grad():
                # 生成动作序列（ODE确定性模式）
                # 预训练评估时也使用generator确保可复现
                eval_gen = torch.Generator(device=device)
                eval_gen.manual_seed(20000 + episode_idx)
                
                result = pipeline_with_logprob(
                    pipeline,
                    obs=obs_tensor,
                    num_inference_steps=config.sample.eval_num_steps,
                    output_type="tensor",   # 保持归一化空间，下方再反归一化
                    return_dict=True,
                    action_decoder=None,  # UNet 直接输出速度，无 Decoder
                    generator=eval_gen,
                )
                
                # 反归一化：pipeline 输出在归一化空间，需还原到真实动作空间
                actions_norm = result["actions"][0]           # (action_horizon, action_dim) normalized
                actions = pipeline.unnormalize_action(actions_norm)  # 真实动作空间
            
            # 执行动作
            for j in range(config.action_horizon):
                if done or episode_length >= max_steps:
                    break
                
                action = actions[j].contiguous() if isinstance(actions, torch.Tensor) else actions[j]
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                episode_length += 1
                done = terminated
            
            # 更新观测
            if not done and episode_length < max_steps:
                if isinstance(obs, torch.Tensor):
                    obs_tensor = obs.to(device, dtype=torch.float32)
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                else:
                    obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        
        # 记录成功
        if done:
            success_count += 1
        
        total_rewards.append(episode_reward)
    
    env.close()
    
    action_unet.train()
    
    avg_reward = np.mean(total_rewards)
    return success_count, num_episodes, avg_reward


def pretrain_model(config, gpu_id):
    """
    使用成功轨迹对Flow策略进行预训练
    
    参数:
        config: 配置对象
        gpu_id: GPU设备ID
    """
    from external.models.maniskill_dataset import ManiSkillDataset
    from torch.utils.data import DataLoader
    from diffusers.training_utils import EMAModel
    from diffusers.optimization import get_scheduler
    
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES已设置
    print(f"\n{'='*60}")
    print(f"开始预训练Flow策略")
    print(f"{'='*60}")
    print(f"使用物理GPU: {gpu_id}")
    print(f"数据集路径: {config.pretrain.dataset_path}")
    print(f"训练episodes: {config.pretrain.num_episodes}")
    print(f"训练epochs: {config.pretrain.num_epochs}")
    
    # 1. 加载数据集
    dataset = ManiSkillDataset(
        dataset_dir=config.pretrain.dataset_path,
        horizon=config.action_horizon,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,  # 预训练不需要验证集
        max_episodes=config.pretrain.num_episodes,
        obs_key='obs'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.pretrain.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    obs_dim = dataset.replay_buffer['obs'][0].shape[-1]
    action_dim = dataset.replay_buffer['action'][0].shape[-1]
    
    print(f"数据集信息:")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  样本数: {len(dataset)}")
    
    # 2. 归一化统计（obs + action，Z-score，对齐 FlowPolicy normalizer）
    all_actions = dataset.replay_buffer['action']   # list 或 ndarray
    flat_actions = np.concatenate(all_actions, axis=0) if isinstance(all_actions, list) \
                   else all_actions.reshape(-1, all_actions[0].shape[-1])
    action_mean = torch.tensor(
        flat_actions.mean(axis=0), device=device, dtype=torch.float32)
    action_std  = torch.tensor(
        flat_actions.std(axis=0) + 1e-6, device=device, dtype=torch.float32)

    all_obs = dataset.replay_buffer['obs']          # list 或 ndarray
    flat_obs = np.concatenate(all_obs, axis=0) if isinstance(all_obs, list) \
               else all_obs.reshape(-1, all_obs[0].shape[-1])
    obs_mean = torch.tensor(
        flat_obs.mean(axis=0), device=device, dtype=torch.float32)
    obs_std  = torch.tensor(
        flat_obs.std(axis=0) + 1e-6, device=device, dtype=torch.float32)

    print(f"归一化统计 (Z-score):")
    print(f"  action_mean: {action_mean.cpu().numpy()}")
    print(f"  action_std : {action_std.cpu().numpy()}")
    print(f"  obs_mean   : {obs_mean.cpu().numpy()}")
    print(f"  obs_std    : {obs_std.cpu().numpy()}")

    # 3. 创建模型（FlowPolicy 风格 UNet，无 ActionDecoder）
    action_unet = FlowConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        kernel_size=5,
    ).to(device)

    print(f"模型参数数量:")
    print(f"  FlowConditionalUnet1D: {sum(p.numel() for p in action_unet.parameters())}")

    # 4. 创建管线（不传 action_decoder，pipeline 直接使用 UNet 输出）
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        device=str(device)
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = None   # 无 Decoder

    # 4b. 设置归一化参数（obs + action Z-score，推理时 pipeline.normalize_obs / unnormalize_action 自动使用）
    pipeline.set_normalizer(obs_mean, obs_std, action_mean, action_std)

    # 5. 优化器（仅 UNet 参数，对齐 FlowPolicy）
    optimizer = torch.optim.AdamW(
        action_unet.parameters(),
        lr=config.pretrain.learning_rate,
        betas=(
            getattr(config.pretrain, 'adam_beta1', 0.95),
            getattr(config.pretrain, 'adam_beta2', 0.999),
        ),
        eps=getattr(config.pretrain, 'adam_eps', 1e-8),
        weight_decay=getattr(config.pretrain, 'weight_decay', 1e-6)
    )
    
    # 6. 学习率调度器（参照ReinFlow）
    if getattr(config.pretrain, 'use_lr_scheduler', True):
        warmup_steps = int(config.pretrain.warmup_epochs * len(dataloader))
        total_steps = config.pretrain.num_epochs * len(dataloader)
        
        if config.pretrain.lr_scheduler_type == "cosine":
            # Cosine annealing with warmup（ReinFlow方式）
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
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
                num_training_steps=total_steps
            )
    else:
        lr_scheduler = None
        print("学习率调度器: 禁用")
    
    # 7. EMA模型（对齐 FlowPolicy：power-based warmup decay，每步更新）
    ema_use_warmup    = getattr(config.pretrain, 'ema_use_warmup',          True)
    ema_decay         = getattr(config.pretrain, 'ema_decay',               0.9999)
    ema_min_decay     = getattr(config.pretrain, 'ema_min_decay',           0.0)
    ema_update_after  = getattr(config.pretrain, 'ema_update_after_step',   0)
    ema_inv_gamma     = getattr(config.pretrain, 'ema_inv_gamma',           1.0)
    ema_power         = getattr(config.pretrain, 'ema_power',               0.75)

    ema_unet = EMAModel(
        action_unet.parameters(),
        decay=ema_decay,
        min_decay=ema_min_decay,
        update_after_step=ema_update_after,
        use_ema_warmup=ema_use_warmup,
        inv_gamma=ema_inv_gamma,
        power=ema_power,
    )

    print(f"\nEMA配置（FlowPolicy power-based warmup）:")
    print(f"  Max decay:          {ema_decay}")
    print(f"  Min decay:          {ema_min_decay}")
    print(f"  Update after step:  {ema_update_after}")
    print(f"  inv_gamma:          {ema_inv_gamma}")
    print(f"  power:              {ema_power}")
    print(f"  use_ema_warmup:     {ema_use_warmup}")
    
    # 8. 定义梯度包装函数（与flow_maniskill_sde.py相同，确保梯度传递）
    def generate_with_grad(pipeline, obs, **kwargs):
        """确保pipeline_with_logprob在梯度启用状态下运行"""
        original_torch_no_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        result = pipeline_with_logprob(pipeline, obs=obs, **kwargs)
        torch.set_grad_enabled(original_torch_no_grad)
        return result

    # 9. 训练循环
    print(f"\n开始训练...")
    
    # 检查是否启用auto模式
    auto_mode = config.pretrain.auto_mode if hasattr(config.pretrain, 'auto_mode') else False
    if auto_mode:
        eval_interval = config.pretrain.auto_eval_interval
        num_test_episodes = config.pretrain.auto_num_test_episodes
        success_threshold = config.pretrain.auto_success_threshold
        max_epochs = config.pretrain.auto_max_epochs
        print(f"[Auto模式] 每 {eval_interval} epochs 评估一次")
        print(f"[Auto模式] 每次评估 {num_test_episodes} 个episodes")
        print(f"[Auto模式] 成功阈值: {success_threshold} 次")
        print(f"[Auto模式] 最大训练轮数: {max_epochs}")
    else:
        max_epochs = config.pretrain.num_epochs
        print(f"[固定模式] 训练 {max_epochs} epochs")
    
    action_unet.train()

    global_step = 0
    training_stopped = False
    final_epoch = 0

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            obs_seq    = batch['obs'].to(device)
            action_seq = batch['action'].to(device)

            # ---- 观测归一化（Z-score，对齐 FlowPolicy）----
            obs_cond = (obs_seq[:, 0, :] - obs_mean) / obs_std   # (B, obs_dim)

            # ---- 动作归一化（Z-score，对齐 FlowPolicy 的 normalizer）----
            # target: (B, action_horizon, action_dim)，归一化到约 N(0,1)
            target = (action_seq - action_mean) / action_std

            batch_size = obs_cond.shape[0]

            # ===== Consistency Flow Matching Loss（全面对齐 FlowPolicy compute_loss）=====
            cfm_eps      = config.pretrain.cfm_eps
            cfm_delta    = config.pretrain.cfm_delta
            cfm_num_segs = config.pretrain.cfm_num_segments
            cfm_boundary = config.pretrain.cfm_boundary
            cfm_alpha    = config.pretrain.cfm_alpha

            a0 = torch.randn_like(target)  # [B, action_horizon, action_dim]

            # 1. 随机采样 t ∈ (eps, 1) 和 r = clamp(t + delta, 1)
            t = torch.rand(batch_size, device=device) * (1 - cfm_eps) + cfm_eps
            r = torch.clamp(t + cfm_delta, max=1.0)
            t_exp = t.view(-1, 1, 1).expand_as(target)
            r_exp = r.view(-1, 1, 1).expand_as(target)

            # 2. 插值得到中间状态
            xt = t_exp * target + (1. - t_exp) * a0
            xr = r_exp * target + (1. - r_exp) * a0

            # 3. 计算所在 segment 的端点
            segments    = torch.linspace(0, 1, cfm_num_segs + 1, device=device)
            seg_idx     = torch.searchsorted(segments, t, side="left").clamp(min=1)
            seg_end     = segments[seg_idx]                              # [B]
            seg_end_exp = seg_end.view(-1, 1, 1).expand_as(target)
            x_at_seg_end = seg_end_exp * target + (1. - seg_end_exp) * a0

            # 4. 预测速度场（两次 forward，timestep 乘以 99 对齐 FlowPolicy）
            #    UNet 直接输出速度，无 ActionDecoder
            vt = action_unet(sample=xt, timestep=t * 99, global_cond=obs_cond)
            vr = action_unet(sample=xr, timestep=r * 99, global_cond=obs_cond)
            vr = torch.nan_to_num(vr)  # 防止 r≈1 时数值不稳定

            # 5. 欧拉步到 segment 端点
            ft = xt + (seg_end_exp - t_exp) * vt
            if isinstance(cfm_boundary, int) and cfm_boundary == 0:
                fr = x_at_seg_end  # 退化模式
            else:
                lt_boundary = t_exp < cfm_boundary
                fr = (lt_boundary  * (xr + (seg_end_exp - r_exp) * vr)
                      + ~lt_boundary * x_at_seg_end)

            # 6. 一致性 loss（主 loss）
            losses_f = torch.square(ft - fr).reshape(batch_size, -1).mean(dim=-1)

            # 7. 速度一致性 loss（辅助 loss）
            if isinstance(cfm_boundary, int) and cfm_boundary == 0:
                losses_v = torch.zeros_like(losses_f)
            else:
                far_from_seg_end = (seg_end - t) > 1.01 * cfm_delta
                far_exp          = far_from_seg_end.view(-1, 1, 1).expand_as(target)
                lt_boundary_v    = t_exp < cfm_boundary
                losses_v = (lt_boundary_v * far_exp
                            * torch.square(vt - vr)).reshape(batch_size, -1).mean(dim=-1)

            loss = torch.mean(losses_f + cfm_alpha * losses_v)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(action_unet.parameters(), max_norm=1.0)

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # 更新EMA（每步更新，power-based warmup 自动控制 decay，对齐 FlowPolicy）
            ema_unet.step(action_unet.parameters())

            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{current_lr:.2e}',
                'EMA': f'{ema_unet.cur_decay_value:.4f}',
                'Step': global_step
            })
        
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{max_epochs} 完成:")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  当前学习率: {current_lr:.2e}")
        print(f"  EMA decay: {ema_unet.cur_decay_value:.6f}")

        final_epoch = epoch + 1

        # Auto模式：定期评估
        if auto_mode and (epoch + 1) % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"[Auto模式评估] Epoch {epoch+1}: 开始评估...")
            print(f"{'='*60}")

            # 复制EMA权重到模型进行评估
            ema_unet.copy_to(action_unet.parameters())

            # 评估模型（无 ActionDecoder）
            success_count, total_count, avg_reward = evaluate_pretrain_model(
                action_unet, None, pipeline, config, device, num_test_episodes
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
    
    # 训练结束提示
    if auto_mode and not training_stopped:
        print(f"\n[Auto模式] 达到最大训练轮数 {max_epochs}，停止训练")
    
    # 10. 保存检查点（使用EMA权重）
    print(f"\n训练完成，准备保存模型...")

    # 复制EMA权重到模型
    ema_unet.copy_to(action_unet.parameters())

    os.makedirs(config.pretrain.checkpoint_dir, exist_ok=True)

    if auto_mode:
        checkpoint_name = f"pretrain_auto_epoch{final_epoch}_episodes{config.pretrain.num_episodes}.pth"
    else:
        checkpoint_name = f"pretrain_epoch{config.pretrain.num_epochs}_episodes{config.pretrain.num_episodes}.pth"

    checkpoint_path = os.path.join(config.pretrain.checkpoint_dir, checkpoint_name)

    torch.save({
        'epoch': final_epoch - 1,
        'action_unet_state_dict': action_unet.state_dict(),   # EMA 权重
        'model_state_dict':       action_unet.state_dict(),   # 兼容旧键名
        'optimizer_state_dict':   optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        'ema_unet_state_dict':    ema_unet.state_dict(),
        'loss':     avg_loss,
        'config':   config.to_dict(),
        'obs_dim':  obs_dim,
        'action_dim': action_dim,
        # 归一化统计（推理时恢复 pipeline.set_normalizer 用）
        'action_mean': action_mean.cpu(),
        'action_std':  action_std.cpu(),
        'obs_mean':    obs_mean.cpu(),
        'obs_std':     obs_std.cpu(),
        'auto_mode': auto_mode,
        'training_stopped': training_stopped if auto_mode else False,
    }, checkpoint_path)

    print(f"\n预训练完成！")
    print(f"检查点已保存: {checkpoint_path}")
    print(f"  训练模式: {'Auto' if auto_mode else '固定'} ({final_epoch} epochs)")
    print(f"  使用episodes: {config.pretrain.num_episodes}")
    print(f"  包含EMA权重: ✓  归一化统计: ✓")
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
    
    # ========== 配置观测参数 ==========
    n_obs_steps   = getattr(config, 'n_obs_steps',   2)
    n_action_steps = getattr(config, 'n_action_steps', 8)
    n_points      = getattr(config, 'n_points',      512)

    # ========== 创建 ManiSkill 训练环境（点云模式） ==========
    _env_obs_mode = getattr(config.env_kwargs, 'obs_mode', 'pointcloud')
    env_kwargs = {
        'num_envs':    config.episode.parallel_envs,
        'sim_backend': getattr(config.env_kwargs, 'sim_backend', 'gpu'),
        'control_mode':getattr(config.env_kwargs, 'control_mode', 'pd_ee_delta_pos'),
        'obs_mode':    _env_obs_mode,
        'render_mode': getattr(config.env_kwargs, 'render_mode', 'rgb_array'),
        'reward_mode': getattr(config.env_kwargs, 'reward_mode', 'dense'),
        'sim_config': dict(
            sim_freq=getattr(config.env_kwargs, 'sim_freq', 100),
            control_freq=getattr(config.env_kwargs, 'control_freq', 20),
        )
    }

    logger.info(f"创建训练环境: {config.env_id}  (obs_mode={_env_obs_mode}, "
                f"{config.episode.parallel_envs} 个并行环境)")
    logger.info(f"GRPO 配置: {config.episode.num_state_groups} 组 × "
                f"{config.episode.samples_per_group} 样本 = "
                f"{config.episode.parallel_envs} 个并行环境")
    # reconfiguration_freq=None 禁用 auto_reset，确保每条轨迹对应完整 episode
    env = gym.make(config.env_id, reconfiguration_freq=None, **env_kwargs)

    # 动作维度
    if hasattr(env, 'single_action_space'):
        action_dim = env.single_action_space.shape[0]
    elif len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[1]
    else:
        action_dim = env.action_space.shape[0]
    logger.info(f"动作维度: {action_dim}")

    # 评估环境（单个环境，也使用点云模式）
    eval_env_kwargs = {k: v for k, v in env_kwargs.items() if k != 'num_envs'}
    eval_env = gym.make(config.env_id, reconfiguration_freq=1, **eval_env_kwargs)

    # ========== 加载预训练 FlowPolicy 并构建三模型架构 ==========
    import copy as _copy

    # 1. 从命令行 --pretrain_checkpoint 或 config.flowpolicy_ckpt 确定 ckpt 路径
    flowpolicy_ckpt_path = None
    if hasattr(FLAGS, 'pretrain_checkpoint') and FLAGS.pretrain_checkpoint:
        flowpolicy_ckpt_path = FLAGS.pretrain_checkpoint
    else:
        flowpolicy_ckpt_path = getattr(config, 'flowpolicy_ckpt', None)

    if not flowpolicy_ckpt_path or not os.path.exists(flowpolicy_ckpt_path):
        raise FileNotFoundError(
            f"FlowPolicy 预训练模型不存在: {flowpolicy_ckpt_path}\n"
            "请通过 config.flowpolicy_ckpt 或 --pretrain_checkpoint 指定有效路径"
        )

    logger.info(f"\n{'='*70}")
    logger.info("加载 FlowPolicy 预训练模型并构建三模型架构（current/old/ref）...")
    logger.info(f"  ckpt: {flowpolicy_ckpt_path}")

    # 2. 加载 current 模型
    flowpolicy_current, flowpolicy_cfg = load_flowpolicy_from_ckpt(
        flowpolicy_ckpt_path, device=accelerator.device
    )

    # 3. 深拷贝 old 模型（off-policy 采样用，定期 EMA 更新）
    flowpolicy_old = clone_flowpolicy(flowpolicy_current, accelerator.device)
    flowpolicy_old.eval()

    # 4. 深拷贝 ref 模型（KL 正则化基准，完全冻结）
    flowpolicy_ref = clone_flowpolicy(flowpolicy_current, accelerator.device)
    for p in flowpolicy_ref.parameters():
        p.requires_grad = False
    flowpolicy_ref.eval()

    # 可训练参数：current 模型的 obs_encoder + ConditionalUnet1D
    trainable_parameters = (
        list(flowpolicy_current.obs_encoder.parameters()) +
        list(flowpolicy_current.model.parameters())
    )
    unet_param_count  = sum(p.numel() for p in flowpolicy_current.model.parameters())
    enc_param_count   = sum(p.numel() for p in flowpolicy_current.obs_encoder.parameters())
    logger.info(f"  UNet 参数量:   {unet_param_count:,}")
    logger.info(f"  Encoder 参数量:{enc_param_count:,}")
    logger.info(f"  可训练参数量:  {unet_param_count + enc_param_count:,}")
    logger.info(f"✓ 三模型架构初始化完成（current/old/ref 均已加载预训练权重）")
    logger.info(f"{'='*70}\n")

    # ── 创建 EMA（用于训练中参数平滑） ──
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

    # ── 初始化优化器 ──
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

    # 用 accelerator 准备 current FlowPolicy 和 optimizer
    flowpolicy_current, optimizer = accelerator.prepare(flowpolicy_current, optimizer)

    # ── 学习率调度器 ──
    if getattr(config.train, 'use_lr_scheduler', False):
        from diffusers.optimization import get_scheduler
        total_training_steps = config.episode.num_epochs
        warmup_steps = getattr(config.train, 'warmup_epochs', 100)
        num_cycles   = getattr(config.train, 'num_cycles', 0.5)
        lr_scheduler = get_scheduler(
            config.train.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=num_cycles,
        )
        lr_scheduler.step()  # 使初始 LR 不为 0
        initial_lr = optimizer.param_groups[0]['lr']
        logger.info(f"学习率调度器: {config.train.lr_scheduler_type}, "
                    f"warmup={warmup_steps} epochs, 起始LR={initial_lr:.2e}")
    else:
        lr_scheduler = None
        logger.info("学习率调度器: 禁用（固定学习率）")

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
    logger.info(f"  Advantage计算方式 = {getattr(config.stage, 'advantage_type', 'group')}")
    if getattr(config.stage, "advantage_type", "group") == "single":
        logger.info(f"  最大奖励阈值(成功判定) = {getattr(config.stage, 'max_reward', 5.0)}")
    else:
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
    
    # 初始化Replay Buffer
    replay_buffer = AdvantageBalancedReplayBuffer(config)
    logger.info("\n***** Replay Buffer配置 *****")
    logger.info(f"  状态: {'启用' if replay_buffer.enabled else '禁用'}")
    if replay_buffer.enabled:
        logger.info(f"  Buffer大小: {replay_buffer.max_size} samples")
        logger.info(f"  目标正样本比例: {replay_buffer.target_positive_ratio:.0%}")
        logger.info(f"  平衡阈值: ±{replay_buffer.balance_threshold:.0%} (范围: {(replay_buffer.target_positive_ratio-replay_buffer.balance_threshold)*100:.0f}%-{(replay_buffer.target_positive_ratio+replay_buffer.balance_threshold)*100:.0f}%)")
        logger.info(f"  最大补充比例: {replay_buffer.max_augment_ratio:.0%}")
        logger.info(f"  预估内存占用: ~{replay_buffer.max_size * 0.06:.0f}MB (满载时)")
    logger.info("")
    
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
                policy_old=flowpolicy_old,  # off-policy 采样使用 old policy
                num_state_groups=config.episode.num_state_groups,
                samples_per_group=config.episode.samples_per_group,
                config=config,
                accelerator=accelerator,
                seed_offset=config.seed + epoch * 100000 + rollout_idx * 10000,
                n_obs_steps=n_obs_steps,
                n_points=n_points,
                epoch=epoch,
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
        
        # ===== 使用Replay Buffer平衡样本（已注释：暂时禁用样本补充） =====
        # 更新epoch计数
        replay_buffer.current_epoch = epoch
        
        # # 平衡样本
        # original_sample_count = len(all_samples)
        # all_samples = replay_buffer.balance_samples(all_samples, epoch)
        
        # # 如果有样本被补充，重新统计advantage
        # if len(all_samples) > original_sample_count:
        #     valid_advantages = [s["advantage"] for s in all_samples if not s.get("filtered", False)]
        #     if len(valid_advantages) > 0:
        #         adv_array = np.array(valid_advantages)
        #         positive_adv_ratio = (adv_array > 0).sum() / len(adv_array)
        #         logger.info(f"  平衡后正样本比例: {positive_adv_ratio:.2%}")
        
        # ===== 记录采样和Advantage指标到wandb =====
        if accelerator.is_main_process:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    # 采样质量指标
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
                
                # # Replay Buffer统计（已注释：暂时禁用样本补充）
                # if replay_buffer.enabled:
                #     buffer_stats = replay_buffer.get_statistics()
                #     wandb.log({
                #         "buffer/positive_count": buffer_stats['positive_count'],
                #         "buffer/negative_count": buffer_stats['negative_count'],
                #         "buffer/total_count": buffer_stats['total_count'],
                #         "buffer/usage_ratio": buffer_stats['usage_ratio'],
                #         "buffer/added_this_epoch": buffer_stats['added_this_epoch'],
                #         "buffer/sampled_this_epoch": buffer_stats['sampled_this_epoch'],
                #         "buffer/memory_mb": buffer_stats['memory_mb'],
                #     }, step=global_step)
        
        """
        # 打印过滤统计信息
        _advantage_type = getattr(config.stage, "advantage_type", "group")
        if _advantage_type == "group" and filter_stats["filtered_groups"] > 0:
            logger.info(f"  Success Rate Filter: 过滤了 {filter_stats['filtered_groups']}/{filter_stats['total_groups']} 组 "
                       f"({filter_stats['filtered_samples']}/{filter_stats['total_samples']} 样本)")
            logger.info(f"    原因: 这些组全部成功或全部失败，无法提供有效的对比学习信号")
        
        # ===== 调试打印：第一个epoch的详细信息 =====
        if epoch == first_epoch:
            logger.info(f"\n{'='*70}")
            logger.info(f"【调试信息】Epoch {epoch+1} 采样和Advantage详情")
            logger.info(f"【奖励计算】使用相对奖励（参考RLinf）")
            if _advantage_type == "single":
                logger.info(f"【优化机制】Single模式：成功(reward>={getattr(config.stage, 'max_reward', 5.0)})->+1，失败->-1")
            else:
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
            
            _dbg_advantage_type = getattr(config.stage, "advantage_type", "group")
            _dbg_max_reward = getattr(config.stage, "max_reward", 5.0)
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
                    
                    if _dbg_advantage_type == "single":
                        # Single 模式：直接读取已计算的二值 advantage
                        advantage = sample["advantage"]
                        is_success = sample["reward"] >= _dbg_max_reward
                        judgment = "成功✓" if is_success else "失败✗"
                        logger.info(f"    样本{i}: 总奖励(相对)={sample['reward']:6.1f}, "
                                  f"步数={sample['num_steps']:3d}, "
                                  f"advantage={advantage:+.1f} [{judgment}] ({status})")
                    elif is_filtered:
                        logger.info(f"    样本{i}: 总奖励(相对)={sample['reward']:6.1f}, "
                                  f"步数={sample['num_steps']:3d}, "
                                  f"advantage=N/A [已过滤] ({status})")
                    else:
                        # Group 模式：重新计算用于显示
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
            
            _val_advantage_type = getattr(config.stage, "advantage_type", "group")
            _val_max_reward = getattr(config.stage, "max_reward", 5.0)
            if _val_advantage_type == "single":
                # Single 模式：统计成功/失败比例
                success_cnt = sum(1 for s in samples if s["reward"] >= _val_max_reward)
                fail_cnt = len(samples) - success_cnt
                logger.info(f"  Single模式（max_reward={_val_max_reward}）：")
                logger.info(f"  成功轨迹（advantage=+1）: {success_cnt}/{len(samples)}")
                logger.info(f"  失败轨迹（advantage=-1）: {fail_cnt}/{len(samples)}")
            else:
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
        for update_idx in range(update_epoch):
            logger.info(f"\n  --- 更新轮次 {update_idx+1}/{update_epoch} ---")
            
            # FlowNFT 训练（FlowPolicy 点云版）
            global_step, avg_epoch_loss = train_on_episode_samples_diffusionnft(
                samples=all_samples,
                flowpolicy_current=flowpolicy_current,
                flowpolicy_old=flowpolicy_old,
                flowpolicy_ref=flowpolicy_ref,
                optimizer=optimizer,
                accelerator=accelerator,
                config=config,
                ema=ema,
                trainable_parameters=trainable_parameters,
                global_step=global_step,
                epoch=epoch,
                n_obs_steps=n_obs_steps,
            )
        
        logger.info(f"\n  ===== 训练完成 =====")
        
        # ===== 更新Replay Buffer（已注释：暂时禁用样本补充） =====
        # if replay_buffer.enabled:
        #     # 将当前epoch的样本添加到buffer
        #     for sample in all_samples:
        #         replay_buffer.add_sample(sample)
        #
        #     # 获取buffer统计信息
        #     buffer_stats = replay_buffer.get_statistics()
        #     logger.info(f"\n  ===== Replay Buffer统计 =====")
        #     logger.info(f"  Positive Buffer: {buffer_stats['positive_count']}/{buffer_stats['positive_capacity']}")
        #     logger.info(f"  Negative Buffer: {buffer_stats['negative_count']}/{buffer_stats['negative_capacity']}")
        #     logger.info(f"  总样本数: {buffer_stats['total_count']}/{buffer_stats['total_capacity']}")
        #     logger.info(f"  Buffer使用率: {buffer_stats['usage_ratio']:.1%}")
        #     logger.info(f"  本epoch添加: {buffer_stats['added_this_epoch']}")
        #     logger.info(f"  本epoch补充: {buffer_stats['sampled_this_epoch']}")
        #     logger.info(f"  内存占用估计: {buffer_stats['memory_mb']:.1f}MB")
        
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
            save_ckpt(
                save_dir=config.save_dir,
                flowpolicy_current=flowpolicy_current,
                epoch=epoch + 1,
                accelerator=accelerator,
                ema=ema,
                trainable_parameters=trainable_parameters,
                config=config,
                flowpolicy_cfg=flowpolicy_cfg,
            )
        
        # 6. 更新学习率调度器（epoch结束时，ReinFlow风格）
        if lr_scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # 显示学习率变化（前100个epoch每次都显示，之后每10个epoch显示一次）
            if epoch < 100 or (epoch + 1) % 10 == 0:
                logger.info(f"  [LR调度] Epoch {epoch+1}: 学习率 {old_lr:.2e} → {new_lr:.2e}")
        
        # ===== 7. Replay Buffer健康检查（每10个epoch） =====
        # ===== Buffer健康检查（已注释：暂时禁用样本补充） =====
        # if replay_buffer.enabled and (epoch + 1) % 10 == 0:
        #     buffer_stats = replay_buffer.get_statistics()
        #
        #     logger.info(f"\n  ===== Buffer健康检查 (Epoch {epoch+1}) =====")
        #
        #     # 检查buffer使用率
        #     if buffer_stats['usage_ratio'] < 0.5:
        #         logger.warning(f"  ⚠ Buffer使用率低 ({buffer_stats['usage_ratio']:.1%})")
        #         logger.warning(f"     建议: 考虑减小buffer_size以节省内存")
        #     elif buffer_stats['usage_ratio'] > 0.95:
        #         logger.info(f"  ✓ Buffer使用率高 ({buffer_stats['usage_ratio']:.1%})，接近满载")
        #     else:
        #         logger.info(f"  ✓ Buffer使用率正常 ({buffer_stats['usage_ratio']:.1%})")
        #
        #     # 检查buffer内正负比例
        #     if buffer_stats['positive_count'] > 0 and buffer_stats['negative_count'] > 0:
        #         buffer_ratio = buffer_stats['positive_count'] / (buffer_stats['positive_count'] + buffer_stats['negative_count'])
        #
        #         if abs(buffer_ratio - 0.5) > 0.3:
        #             logger.warning(f"  ⚠ Buffer内正负比例失衡 ({buffer_ratio:.1%})")
        #             logger.warning(f"     Positive: {buffer_stats['positive_count']}, Negative: {buffer_stats['negative_count']}")
        #             logger.warning(f"     建议: 这可能影响样本平衡效果，检查训练策略是否偏向")
        #         else:
        #             logger.info(f"  ✓ Buffer内正负比例平衡 ({buffer_ratio:.1%})")
        #     elif buffer_stats['positive_count'] == 0:
        #         logger.warning(f"  ⚠ Positive buffer为空，无法补充正样本")
        #     elif buffer_stats['negative_count'] == 0:
        #         logger.warning(f"  ⚠ Negative buffer为空，无法补充负样本")
        #
        #     # 检查补充效果
        #     if buffer_stats['sampled_this_epoch'] > 0:
        #         logger.info(f"  ✓ Buffer正常工作，本epoch补充了 {buffer_stats['sampled_this_epoch']} 个样本")
        #
        #     logger.info(f"  累计统计: 添加 {buffer_stats['total_added']} | 补充 {buffer_stats['total_sampled']}")
        #     logger.info(f"  内存占用: {buffer_stats['memory_mb']:.1f}MB / ~{replay_buffer.max_size * 0.06:.0f}MB")
        
        # ========== 8. 更新 old model（per_epoch 模式：每个 epoch 结束后更新一次） ==========
        # per_step 模式的更新逻辑在 train_on_episode_samples_diffusionnft 内部完成
        # per_epoch 模式对齐论文 Algorithm 1 Line 14，每个 iteration 结束后软更新一次
        old_model_update_mode = getattr(config.train, 'old_model_update_mode', 'per_step')
        if old_model_update_mode == 'per_epoch':
            update_old_model(
                flowpolicy_current=flowpolicy_current,
                flowpolicy_old=flowpolicy_old,
                config=config,
                accelerator=accelerator,
                global_step=global_step,
                epoch=epoch,
                log_tag="Old model 更新(per_epoch)"
            )
    
    logger.info("训练完成！")
    
    # 最终评估（使用评估环境）
    # eval_in_env 已基于旧 pipeline 接口，暂时跳过，后续可重构
    logger.info("训练循环结束，跳过最终 eval_in_env（需点云版本适配）")

    # 保存最终模型（使用最终epoch数）
    final_epoch = config.episode.num_epochs
    save_ckpt(
        save_dir=config.save_dir,
        flowpolicy_current=flowpolicy_current,
        epoch=final_epoch,
        accelerator=accelerator,
        ema=ema,
        trainable_parameters=trainable_parameters,
        config=config,
        flowpolicy_cfg=flowpolicy_cfg,
    )
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    # 关闭 wandb
    accelerator.end_training()


if __name__ == "__main__":
    app.run(main)


