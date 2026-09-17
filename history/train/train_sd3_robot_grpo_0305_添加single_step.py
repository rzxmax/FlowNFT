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
flags.DEFINE_bool("single_step", False, "Use single-step x1 prediction (x1=xt+(1-t)*v) instead of multi-step rollout for training loss. Faster and memory-efficient, mathematically equivalent to flow matching loss.")

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
    
    # ✅ DiffusionNFT: 将 pipeline 内部的 unet 绑定到传入的采样模型
    # pipeline._action_unet 默认指向 vθ（训练模型），此处临时切换为 vold，
    # 确保 pipeline_with_logprob 使用 vold 进行数据收集（off-policy rollout）
    pipeline._action_unet = action_unet
    
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
                    
                    result = pipeline_with_logprob(
                        pipeline, obs=obs_tensor,
                        num_inference_steps=config.sample.num_steps,
                        output_type="clipped", return_dict=True,
                        action_decoder=action_decoder,
                        generator=generators,
                    )
            
            actions_batch = result["actions"]
            
            # ✅ DiffusionNFT: 保存clean actions (ODE采样的确定性输出)
            for i in range(num_envs):
                if active_mask[i]:
                    clean_actions = actions_batch[i].clone()  # [action_horizon, action_dim]
                    env_clean_actions_list[i].append(clean_actions)
                    env_obs_list[i].append(obs_tensor[i:i+1].clone())
            
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
        })
    
    return all_samples


def compute_episode_level_advantages(samples, num_train_timesteps, config):
    """
    基于完整轨迹的GRPO advantage计算
    
    实现：
    - 按GRPO组分组（组内相同初始状态）
    - ✅ 使用相对奖励的总和（参考RLinf）
    - 相对奖励已经消除了累积效应，可以直接使用总奖励比较
    - ✅ Success Rate Filter（参考RLinf）：过滤全成功或全失败的组
    - 为每条轨迹的每个Flow调用的每个timestep分配相同的advantage
    - 应用timestep权重（可选）
    
    关键修改：
    - 使用相对奖励后，总奖励就是所有增量的和
    - 不受轨迹长度影响，可以直接比较
    - Success Rate Filter 加速收敛并提高性能
    """
    num_state_groups = config.stage.num_state_groups
    samples_per_group = config.stage.samples_per_group
    
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


def update_old_model(
    action_unet, action_decoder,
    action_unet_old, action_decoder_old,
    action_unet_ref, action_decoder_ref,
    config, accelerator, global_step, epoch,
    log_tag="Old model 更新"
):
    """
    EMA 更新 old model：θold ← η·θold + (1-η)·θ  （DiffusionNFT Algorithm 1, Line 14）
    
    可被 per_step 和 per_epoch 两种模式复用。
    """
    with torch.no_grad():
        decay = return_decay(global_step, config.train.decay_type)

        # ===== 计算模型间 L2 距离（用于监控） =====
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

        # ===== EMA 更新 UNet =====
        for src_param, tgt_param in zip(
            unwrap_model(action_unet, accelerator).parameters(),
            action_unet_old.parameters()
        ):
            tgt_param.data.copy_(
                tgt_param.data * decay + src_param.data * (1.0 - decay)
            )

        # ===== EMA 更新 Decoder =====
        for src_param, tgt_param in zip(
            unwrap_model(action_decoder, accelerator).parameters(),
            action_decoder_old.parameters()
        ):
            tgt_param.data.copy_(
                tgt_param.data * decay + src_param.data * (1.0 - decay)
            )

    if accelerator.is_local_main_process:
        logger.info(f"  [{log_tag}] Step {global_step} / Epoch {epoch}: decay={decay:.4f}, "
                    f"dist(cur,old)={current_old_distance:.4f}")

    # ===== 记录到 wandb =====
    if accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "model/current_old_distance": current_old_distance,
                "model/current_ref_distance": current_ref_distance,
                "model/old_ref_distance": old_ref_distance,
                "model/old_decay": decay,
            }, step=global_step)


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
        # 计算当前timestep（归一化到[0,1]）
        t_normalized = torch.full(
            (current_sample.shape[0],),
            step_idx / num_train_timesteps,
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


def train_on_episode_samples_diffusionnft(
    samples, 
    action_unet, action_decoder,
    action_unet_old, action_decoder_old,
    action_unet_ref, action_decoder_ref,
    optimizer, accelerator, config, ema, trainable_parameters,
    global_step, epoch,
    use_single_step=False,
):
    """
    DiffusionNFT风格训练函数（完全替换原GRPO训练）
    
    核心特性：
    1. 遍历所有timesteps（完全对齐DiffusionNFT Line 854）
    2. 三模型velocity prediction（current/old/ref）
    3. Clip+归一化+clamp advantage（DiffusionNFT Line 910-929）
    4. 标准 MSE loss（移除 Adaptive Weighting 以适应机器人控制）
    5. Policy loss + KL loss（移除 adv_clip_max 缩放）
    
    use_single_step: 是否使用单步预测（True）还是多步Rollout（False）
        - 单步预测：x1_pred = xt + (1-t) * v，速度快，等价于标准FM loss
        - 多步Rollout：从xt开始多步ODE积分到x1，精度更高但计算量大
        启动参数：--single_step（单步） 或 默认（多步）
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
    sliding_window_stride = config.train.sliding_window_stride  # 滑动窗口步长（在config.train中配置）
    
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
                # Old model（no grad，DiffusionNFT Line 874-882）
                with torch.no_grad():
                    old_pred_raw = action_unet_old(
                        sample=xt, 
                        timestep=t,  # 使用归一化的timestep [0, (num_train_timesteps-1)/num_train_timesteps]
                        global_cond=batch_obs
                    )
                    old_prediction = action_decoder_old(old_pred_raw)
                
                # Current model（with grad，DiffusionNFT Line 886-892）
                current_pred_raw = action_unet(
                    sample=xt,
                    timestep=t,  # 使用归一化的timestep
                    global_cond=batch_obs
                )
                current_prediction = action_decoder(current_pred_raw)
                
                # Reference model（no grad，DiffusionNFT Line 894-907）
                with torch.no_grad():
                    ref_pred_raw = action_unet_ref(
                        sample=xt,
                        timestep=t,  # 使用归一化的timestep
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
            
            # # ========== DEBUG: 第一个epoch打印完整r值 ==========
            # if epoch == 0 and j_idx == 1 and accelerator.is_local_main_process:
            #     logger.info(f"\n{'='*60}")
            #     logger.info(f"[r值验证] Epoch=0, Batch={batch_idx}/{num_batches-1}, t={t_value:.2f}")
            #     logger.info(f"  batch_size = {batch_advantages.shape[0]}")
            #     logger.info(f"  --- 原始 advantage (组内z-score后) ---")
            #     logger.info(f"  batch_advantages = {batch_advantages.cpu().tolist()}")
            #     logger.info(f"  统计: mean={batch_advantages.mean().item():.4f}, std={batch_advantages.std().item():.4f}, "
            #                 f"min={batch_advantages.min().item():.4f}, max={batch_advantages.max().item():.4f}")
            #     logger.info(f"  --- clip后 (adv_clip_max={config.train.adv_clip_max}) ---")
            #     logger.info(f"  advantages_clip = {advantages_clip.cpu().tolist()}")
            #     logger.info(f"  --- 线性归一化到[0,1]: (clip/adv_clip_max)/2 + 0.5 ---")
            #     logger.info(f"  normalized_advantages = {normalized_advantages.cpu().tolist()}")
            #     logger.info(f"  --- 最终 r = clamp(normalized, 0, 1) ---")
            #     logger.info(f"  r = {r.cpu().tolist()}")
            #     logger.info(f"  r统计: mean={r.mean().item():.4f}, std={r.std().item():.4f}, "
            #                 f"min={r.min().item():.4f}, max={r.max().item():.4f}")
            #     logger.info(f"  r接近0的数量(负advantage): {(r < 0.5).sum().item()}")
            #     logger.info(f"  r接近1的数量(正advantage): {(r > 0.5).sum().item()}")
            #     logger.info(f"  r=0.5的数量(零advantage):  {(r == 0.5).sum().item()}")
            #     logger.info(f"{'='*60}\n")
                
            # ========== 6. x1估计：单步预测 or 多步Rollout ==========
            if use_single_step:
                # ========== 单步预测（--single_step 模式） ==========
                # 你的约定：t=0为纯噪声，t=1为干净数据
                # 速度场 v = data - noise，从噪声流向数据
                # 单步Euler积分：x1_pred = xt + (1-t) * v
                # 数学等价于标准Flow Matching loss（(1-t)²加权的速度场MSE）
                x1_prediction_pos = xt + (1 - t_expanded) * positive_prediction
                x1_prediction_neg = xt + (1 - t_expanded) * implicit_negative_prediction
                
                # ========== DEBUG: 单步预测结果 ==========
                if epoch == 0 and batch_idx == 0 and j_idx == 1:
                    logger.info(f"\n【x1预测（单步预测模式）】")
                    logger.info(f"单步公式：x1_pred = xt + (1-t) * v，t={t_value:.1f}")
                    logger.info(f"x1_prediction_pos 统计: min={x1_prediction_pos.min().item():.6f}, max={x1_prediction_pos.max().item():.6f}, mean={x1_prediction_pos.mean().item():.6f}")
                    logger.info(f"x1_prediction_neg 统计: min={x1_prediction_neg.min().item():.6f}, max={x1_prediction_neg.max().item():.6f}, mean={x1_prediction_neg.mean().item():.6f}")
                    logger.info(f"x1预测误差 (pos): mean={(x1_prediction_pos - batch_x1).abs().mean().item():.6f}, max={(x1_prediction_pos - batch_x1).abs().max().item():.6f}")
                    logger.info(f"x1预测误差 (neg): mean={(x1_prediction_neg - batch_x1).abs().mean().item():.6f}, max={(x1_prediction_neg - batch_x1).abs().max().item():.6f}")
            else:
                # ========== 多步Rollout（默认模式） ==========
                # 使用rollout函数，从xt开始沿着positive和negative两条路径去噪到x1
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
                
                # ========== DEBUG: 多步Rollout结果 ==========
                if epoch == 0 and batch_idx == 0 and j_idx == 1:
                    logger.info(f"\n【x1预测（多步Rollout模式）】")
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
                
                # ========== Old model 更新（per_step 模式：每20个global_step更新一次） ==========
                old_model_update_mode = getattr(config.train, 'old_model_update_mode', 'per_step')
                if old_model_update_mode == 'per_step' and global_step % 20 == 0:
                    update_old_model(
                        action_unet=action_unet,
                        action_decoder=action_decoder,
                        action_unet_old=action_unet_old,
                        action_decoder_old=action_decoder_old,
                        action_unet_ref=action_unet_ref,
                        action_decoder_ref=action_decoder_ref,
                        config=config,
                        accelerator=accelerator,
                        global_step=global_step,
                        epoch=epoch,
                        log_tag="Old model 更新(per_step)"
                    )
    
    # 计算整个epoch的平均loss
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
    """保存检查点"""
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-epoch{epoch}")
    os.makedirs(save_root, exist_ok=True)
    
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(trainable_parameters, store_temp=True)
        
        # 保存模型权重
        torch.save({
            'action_unet_state_dict': unwrap_model(action_unet, accelerator).state_dict(),
            'action_decoder_state_dict': unwrap_model(action_decoder, accelerator).state_dict(),
            'epoch': epoch,
        }, os.path.join(save_root, "model.pth"))
        
        if config.train.ema:
            ema.copy_temp_to(trainable_parameters)
        
        logger.info(f"Saved checkpoint to {save_root}")


def evaluate_pretrain_model(action_unet, action_decoder, pipeline, config, device, num_episodes=10):
    """
    在ManiSkill环境中评估预训练模型
    
    参数:
        action_unet: UNet模型
        action_decoder: 动作解码器
        pipeline: RobotActionPipeline
        config: 配置对象
        device: 设备
        num_episodes: 测试episode数量
    
    返回:
        success_count: 成功次数
        total_count: 总测试次数
        avg_reward: 平均奖励
    """
    action_unet.eval()
    action_decoder.eval()
    
    # 创建评估环境
    env_kwargs = {
        'sim_backend': config.env_kwargs.get('sim_backend', 'gpu'),
        'control_mode': config.env_kwargs.get('control_mode', 'pd_ee_delta_pos'),
        'obs_mode': config.env_kwargs.get('obs_mode', 'state'),
        'render_mode': 'rgb_array',
        'reward_mode': config.env_kwargs.get('reward_mode', 'dense')
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
                    output_type="clipped",
                    return_dict=True,
                    action_decoder=action_decoder,
                    generator=eval_gen,  # ✅ 传入generator
                )
                
                actions = result["actions"][0]
            
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
    action_decoder.train()
    
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
    
    # 2. 创建模型
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
    
    print(f"模型参数数量:")
    print(f"  UNet参数: {sum(p.numel() for p in action_unet.parameters())}")
    print(f"  ActionDecoder参数: {sum(p.numel() for p in action_decoder.parameters())}")
    
    # 3. 创建管线
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
    
    # 4. 优化器
    optimizer = torch.optim.AdamW(
        list(action_unet.parameters()) + list(action_decoder.parameters()),
        lr=config.pretrain.learning_rate,
        weight_decay=getattr(config.pretrain, 'weight_decay', 1e-6)
    )
    
    # 5. 学习率调度器（参照ReinFlow）
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
    
    # 6. EMA模型（参照ReinFlow改进）
    ema_decay = getattr(config.pretrain, 'ema_decay', 0.995)
    ema_start_epoch = getattr(config.pretrain, 'ema_start_epoch', 20)
    ema_update_freq = getattr(config.pretrain, 'ema_update_freq', 10)
    
    ema_unet = EMAModel(
        action_unet.parameters(),
        decay=ema_decay,
        use_ema_warmup=False,  # 禁用内置warmup，手动控制
    )
    
    ema_decoder = EMAModel(
        action_decoder.parameters(),
        decay=ema_decay,
        use_ema_warmup=False,
    )
    
    print(f"\nEMA配置:")
    print(f"  Decay: {ema_decay}")
    print(f"  Start epoch: {ema_start_epoch}")
    print(f"  Update freq: {ema_update_freq}")
    
    # 7. 定义梯度包装函数（与flow_maniskill_sde.py相同，确保梯度传递）
    def generate_with_grad(pipeline, obs, **kwargs):
        """确保pipeline_with_logprob在梯度启用状态下运行"""
        original_torch_no_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        # 调用管线函数（不使用@torch.no_grad()装饰器）
        result = pipeline_with_logprob(pipeline, obs=obs, **kwargs)
        
        # 恢复原始梯度设置
        torch.set_grad_enabled(original_torch_no_grad)
        return result
    
    # 8. 训练循环
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
    action_decoder.train()
    
    global_step = 0
    training_stopped = False
    final_epoch = 0
    
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            obs_seq = batch['obs'].to(device)
            action_seq = batch['action'].to(device)
            
            obs_cond = obs_seq[:, 0, :]
            target_actions = action_seq
            batch_size = obs_cond.shape[0]
            
            # 生成动作（允许梯度）- 使用与flow_maniskill_sde.py相同的方式
            # initial_noise = torch.randn(
            #     batch_size, config.action_horizon, action_dim, 
            #     device=device, requires_grad=True
            # )
            
            # 使用generate_with_grad包装器（与flow_maniskill_sde.py相同）
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
            
            # 计算MSE损失
            loss = torch.nn.functional.mse_loss(pred_actions, target_actions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（与flow_maniskill_sde.py相同，分别裁剪）
            torch.nn.utils.clip_grad_norm_(action_unet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(action_decoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # 更新EMA（参照ReinFlow：延迟启动）
            # 注意：每个batch都会更新step，但只在指定epoch后才真正生效
            if epoch >= ema_start_epoch:
                ema_unet.step(action_unet.parameters())
                ema_decoder.step(action_decoder.parameters())
            
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{current_lr:.2e}',
                'Step': global_step
            })
        
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印epoch摘要
        ema_status = "Active" if epoch >= ema_start_epoch else f"Inactive (starts at epoch {ema_start_epoch})"
        print(f"Epoch {epoch+1}/{max_epochs} 完成:")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  当前学习率: {current_lr:.2e}")
        print(f"  EMA状态: {ema_status}")
        
        final_epoch = epoch + 1
        
        # Auto模式：定期评估
        if auto_mode and (epoch + 1) % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"[Auto模式评估] Epoch {epoch+1}: 开始评估...")
            print(f"{'='*60}")
            
            # 复制EMA权重到模型进行评估
            ema_unet.copy_to(action_unet.parameters())
            ema_decoder.copy_to(action_decoder.parameters())
            
            # 评估模型
            success_count, total_count, avg_reward = evaluate_pretrain_model(
                action_unet, action_decoder, pipeline, config, device, num_test_episodes
            )
            
            print(f"[Auto模式评估] 成功率: {success_count}/{total_count}")
            print(f"[Auto模式评估] 平均奖励: {avg_reward:.3f}")
            
            # 检查是否达到成功阈值
            if success_count >= success_threshold:
                print(f"\n{'='*60}")
                print(f"[Auto模式] 达到成功阈值！")
                print(f"[Auto模式] 成功次数: {success_count}/{total_count}")
                print(f"[Auto模式] 停止训练于 epoch {epoch+1}")
                print(f"{'='*60}\n")
                training_stopped = True
                break
            else:
                print(f"[Auto模式] 未达到阈值，继续训练...\n")
                # 恢复训练模式
                action_unet.train()
                action_decoder.train()
    
    # 训练结束提示
    if auto_mode and not training_stopped:
        print(f"\n[Auto模式] 达到最大训练轮数 {max_epochs}，停止训练")
    
    # 9. 保存检查点（使用EMA权重，与flow_maniskill_sde.py相同）
    print(f"\n训练完成，准备保存模型...")
    
    # 复制EMA权重到模型（与flow_maniskill_sde.py相同）
    ema_unet.copy_to(action_unet.parameters())
    ema_decoder.copy_to(action_decoder.parameters())
    
    os.makedirs(config.pretrain.checkpoint_dir, exist_ok=True)
    
    # 根据模式决定checkpoint名称
    if auto_mode:
        checkpoint_name = f"pretrain_auto_epoch{final_epoch}_episodes{config.pretrain.num_episodes}.pth"
    else:
        checkpoint_name = f"pretrain_epoch{config.pretrain.num_epochs}_episodes{config.pretrain.num_episodes}.pth"
    
    checkpoint_path = os.path.join(config.pretrain.checkpoint_dir, checkpoint_name)
    
    # 保存模型状态（包含EMA权重、优化器、调度器等，与flow_maniskill_sde.py相同）
    torch.save({
        'epoch': final_epoch - 1,
        'model_state_dict': action_unet.state_dict(),  # 已包含EMA权重
        'action_decoder_state_dict': action_decoder.state_dict(),  # 已包含EMA权重
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'ema_unet_state_dict': ema_unet.state_dict(),
        'ema_decoder_state_dict': ema_decoder.state_dict(),
        'loss': avg_loss,
        'config': config.to_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'auto_mode': auto_mode,
        'training_stopped': training_stopped if auto_mode else False,
        # 为了向后兼容，也保存旧的键名
        'action_unet_state_dict': action_unet.state_dict(),
    }, checkpoint_path)
    
    print(f"\n预训练完成！")
    print(f"检查点已保存: {checkpoint_path}")
    if auto_mode:
        print(f"  训练模式: Auto (自动停止)")
        print(f"  实际训练轮数: {final_epoch} epochs")
        if training_stopped:
            print(f"  停止原因: 达到成功阈值")
        else:
            print(f"  停止原因: 达到最大训练轮数")
    else:
        print(f"  训练模式: 固定 ({final_epoch} epochs)")
    print(f"  使用episodes: {config.pretrain.num_episodes}")
    print(f"  包含EMA权重: ✓")
    print(f"  包含优化器状态: ✓")
    print(f"  包含学习率调度器: ✓")
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
        pretrain_dir = config.pretrain.checkpoint_dir if hasattr(config, 'pretrain') else "pretrain-checkpoints"
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
    if FLAGS.single_step:
        logger.info(f"  x1估计模式 = 【单步预测】x1_pred = xt + (1-t)*v  (--single_step)")
        logger.info(f"              快速、低显存，等价于标准Flow Matching loss")
    else:
        logger.info(f"  x1估计模式 = 【多步Rollout】从xt逐步ODE积分到x1  (默认，使用 --single_step 切换)")
        logger.info(f"              精度较高但计算量大")
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
                pipeline=pipeline,
                action_unet=action_unet_old,       # ✅ DiffusionNFT: 使用 vold 进行数据收集（off-policy）
                action_decoder=action_decoder_old,  # ✅ DiffusionNFT: 使用 vold 进行数据收集（off-policy）
                num_state_groups=config.episode.num_state_groups,
                samples_per_group=config.episode.samples_per_group,
                config=config,
                accelerator=accelerator,
                autocast=autocast,
                seed_offset=config.seed + epoch * 100000 + rollout_idx * 10000,
                epoch=epoch  # 传入epoch用于噪声退火
            )
            
            # ✅ DiffusionNFT: 采样结束，将 pipeline 内部的 unet 还原为训练模型 vθ
            # 确保后续训练阶段 pipeline 使用 vθ（若 pipeline 在训练中被复用）
            pipeline._action_unet = unwrap_model(action_unet, accelerator)
            
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
            logger.info(f"  num_state_groups = {config.stage.num_state_groups}")
            logger.info(f"  samples_per_group = {config.stage.samples_per_group}")
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
        for update_idx in range(update_epoch):
            logger.info(f"\n  --- 更新轮次 {update_idx+1}/{update_epoch} ---")
            
            # ✅ DiffusionNFT训练
            global_step, avg_epoch_loss = train_on_episode_samples_diffusionnft(
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
                epoch=epoch,
                use_single_step=FLAGS.single_step,
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
            save_ckpt(config.save_dir, action_unet, action_decoder, epoch + 1,
                      accelerator, ema, trainable_parameters, config)
        
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
                action_unet=action_unet,
                action_decoder=action_decoder,
                action_unet_old=action_unet_old,
                action_decoder_old=action_decoder_old,
                action_unet_ref=action_unet_ref,
                action_decoder_ref=action_decoder_ref,
                config=config,
                accelerator=accelerator,
                global_step=global_step,
                epoch=epoch,
                log_tag="Old model 更新(per_epoch)"
            )
    
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

