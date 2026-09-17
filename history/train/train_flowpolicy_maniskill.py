#!/usr/bin/env python
"""
FlowPolicy 预训练脚本（ManiSkill PickCube 任务）

使用 FlowPolicy 的 Consistency Flow Matching (CFM) 方法进行预训练。
与 train_sd3_robot_grpo.py 的区别：
  - 归一化：使用 LinearNormalizer(mode='limits') 替代 Z-score
  - 推理：使用 FlowPolicy 原始的 ODE 推理流程

使用方法：
    python train_flowpolicy_maniskill.py \
        --config=config/robot_base.py \
        --gpu=1 \
        --num_episodes=1000 \
        --num_epochs=50

测试已保存的 checkpoint：
    python train_flowpolicy_maniskill.py \
        --config=config/robot_base.py \
        --gpu=1 \
        --test \
        --checkpoint=pretrain-checkpoints/flowpolicy_epoch50_episodes1000.pth
"""

import sys
import os

# !!!!! 重要：在导入任何 CUDA 库之前设置 GPU !!!!!
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
print(f"[早期设置] 使用物理 GPU {gpu_id}, 设置 CUDA_VISIBLE_DEVICES={gpu_id}")
print(f"[早期设置] 后续 PyTorch 将把物理 GPU {gpu_id} 映射为 cuda:0")
print(f"{'='*70}\n")

sys.dont_write_bytecode = True
sys.path.insert(0, './flow_grpo')
sys.path.insert(0, './flow_grpo/diffusers_patch')
sys.path.insert(0, './external/models')
sys.path.insert(0, './external')
# FlowPolicy 路径（用于 LinearNormalizer 等工具）
sys.path.insert(0, './FlowPolicy/FlowPolicy')

import argparse
import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler

# 数据集（复用现有代码）
from external.models.maniskill_dataset import ManiSkillDataset

# UNet（FlowPolicy 风格，无 ActionDecoder）
from flow_unet1d import FlowConditionalUnet1D

# Pipeline（用于推理评估）
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import (
    RobotActionPipeline, pipeline_with_logprob
)

# LinearNormalizer（FlowPolicy 的归一化方式）
from diffusion_policy.model.common.normalizer import LinearNormalizer

import gymnasium as gym
import mani_skill.envs


# ============================================================
# 推理评估函数（复用 test_pretrain_model.py 的逻辑）
# ============================================================

def run_episodes(env, pipeline, normalizer, episode_ids, device,
                 num_inference_steps, max_steps, action_horizon, label="测试"):
    """在给定 episode_ids 上运行评估，返回统计结果。"""
    all_rewards, all_lengths, all_success = [], [], []

    print(f"\n  开始{label}测试，共 {len(episode_ids)} 个episodes...")

    for episode_idx in tqdm(range(len(episode_ids)), desc=f"{label}进度"):
        episode_id = episode_ids[episode_idx]
        obs, info = env.reset(seed=episode_id, options={'episode_id': episode_id})

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
            with torch.no_grad():
                gen = torch.Generator(device=device)
                gen.manual_seed(episode_id * 10000 + episode_length)

                result = pipeline_with_logprob(
                    pipeline,
                    obs=obs_tensor,
                    num_inference_steps=num_inference_steps,
                    output_type="tensor",
                    return_dict=True,
                    determistic=True,
                    action_decoder=None,
                    generator=gen,
                )
                # 反归一化到真实动作空间
                actions_norm = result["actions"][0]          # [H, Da]，归一化空间
                actions = pipeline.unnormalize_action(actions_norm)  # 真实空间

            for j in range(action_horizon):
                if done or episode_length >= max_steps:
                    break
                action = actions[j]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action)
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
        all_lengths.append(episode_length)
        all_success.append(1 if done else 0)

    all_rewards = np.array(all_rewards)
    all_lengths = np.array(all_lengths)
    all_success = np.array(all_success)

    print(f"\n{'='*60}")
    print(f"[{label}] 测试结果统计")
    print(f"{'='*60}")
    print(f"  测试episodes: {len(episode_ids)}")
    print(f"  成功次数: {int(all_success.sum())}")
    print(f"  成功率: {all_success.mean():.2%}")
    print(f"  平均奖励: {all_rewards.mean():.3f} ± {all_rewards.std():.3f}")
    print(f"  平均长度: {all_lengths.mean():.1f} ± {all_lengths.std():.1f}")

    return {
        'success_rate': float(all_success.mean()),
        'num_success': int(all_success.sum()),
        'num_episodes': len(episode_ids),
        'mean_reward': float(all_rewards.mean()),
        'std_reward': float(all_rewards.std()),
        'mean_length': float(all_lengths.mean()),
    }


def evaluate_model(action_unet, normalizer, config, device, num_episodes=10):
    """在 ManiSkill 环境中评估当前模型。"""
    obs_dim = normalizer['obs'].output_max.shape[0]
    action_dim = normalizer['action'].output_max.shape[0]

    # 构建 pipeline
    pipeline = _build_pipeline(action_unet, normalizer, obs_dim, action_dim,
                                config.action_horizon, device)

    env_kwargs = {
        'sim_backend': 'gpu',
        'control_mode': 'pd_ee_delta_pos',
        'obs_mode': 'state',
        'render_mode': 'rgb_array',
        'reward_mode': 'dense',
        'sim_config': dict(
            sim_freq=config.env_kwargs.get('sim_freq', 100),
            control_freq=config.env_kwargs.get('control_freq', 20),
        )
    }
    env = gym.make(config.env_id, reconfiguration_freq=1, **env_kwargs)

    gen = torch.Generator(device='cpu')
    gen.manual_seed(99999)
    episode_ids = torch.randint(0, 10000, (num_episodes,), generator=gen).tolist()

    results = run_episodes(
        env=env,
        pipeline=pipeline,
        normalizer=normalizer,
        episode_ids=episode_ids,
        device=device,
        num_inference_steps=config.sample.eval_num_steps,
        max_steps=config.max_episode_steps,
        action_horizon=config.action_horizon,
        label="评估",
    )
    env.close()
    return results['num_success'], results['num_episodes'], results['mean_reward']


def _build_pipeline(action_unet, normalizer, obs_dim, action_dim, action_horizon, device):
    """构建推理 pipeline 并设置归一化参数。"""
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=str(device),
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = None

    # 从 LinearNormalizer 提取 Z-score 等效参数供 pipeline 使用
    # LinearNormalizer(mode='limits') 将值线性映射到 [-1, 1]
    # 推理时需要反归一化：pipeline.unnormalize_action 使用 action_mean/std
    # 这里直接把 normalizer 挂到 pipeline 上，重写 unnormalize_action
    pipeline._linear_normalizer = normalizer

    # 覆盖 unnormalize_action 使其使用 LinearNormalizer
    _norm = normalizer  # 捕获引用

    def _unnormalize_action(action_normed):
        """将归一化空间 [-1,1] 的动作还原到真实空间。"""
        return _norm['action'].unnormalize(action_normed)

    def _normalize_obs_fn(obs):
        """将原始 obs 归一化到 [-1,1] 空间。"""
        return _norm['obs'].normalize(obs)

    pipeline.unnormalize_action = _unnormalize_action
    pipeline.normalize_obs      = _normalize_obs_fn

    return pipeline


# ============================================================
# 主训练函数
# ============================================================

def train(config, gpu_id, num_episodes_override=None, num_epochs_override=None,
          auto_mode=False, wandb_project=None, wandb_run_name=None, use_wandb=True):
    """
    使用 FlowPolicy CFM 方法对 Flow 策略进行预训练。

    与 train_sd3_robot_grpo.py 的 pretrain_model 的关键区别：
      - 归一化：LinearNormalizer(mode='limits')，与 FlowPolicy 完全一致
      - checkpoint 格式：兼容 train_sd3_robot_grpo.py 的加载接口
      - wandb 日志：对齐 FlowPolicy/train.py 的指标（train_loss、bc_loss、lr、
                    train_action_mse_error、eval_success_rate 等）
    """
    device = torch.device("cuda:0")

    print(f"\n{'='*60}")
    print(f"开始 FlowPolicy CFM 预训练（ManiSkill PickCube）")
    print(f"{'='*60}")
    print(f"使用物理GPU: {gpu_id}")

    # ── 超参数 ──────────────────────────────────────────────
    dataset_path   = config.pretrain.dataset_path
    num_episodes   = num_episodes_override or config.pretrain.num_episodes
    num_epochs     = num_epochs_override   or config.pretrain.num_epochs
    batch_size     = config.pretrain.batch_size
    lr             = config.pretrain.learning_rate
    checkpoint_dir = config.pretrain.checkpoint_dir
    action_horizon = config.action_horizon

    cfm_eps      = getattr(config.pretrain, 'cfm_eps',          1e-2)
    cfm_delta    = getattr(config.pretrain, 'cfm_delta',         1e-2)
    cfm_num_segs = getattr(config.pretrain, 'cfm_num_segments',  2)
    cfm_boundary = getattr(config.pretrain, 'cfm_boundary',      1)
    cfm_alpha    = getattr(config.pretrain, 'cfm_alpha',         1e-5)

    auto_eval_interval       = getattr(config.pretrain, 'auto_eval_interval',       10)
    auto_num_test_episodes   = getattr(config.pretrain, 'auto_num_test_episodes',   20)
    auto_success_threshold   = getattr(config.pretrain, 'auto_success_threshold',   15)
    auto_max_epochs          = getattr(config.pretrain, 'auto_max_epochs',          1000)
    max_epochs = auto_max_epochs if auto_mode else num_epochs

    print(f"数据集路径: {dataset_path}")
    print(f"训练episodes: {num_episodes}")
    print(f"训练epochs: {max_epochs}")
    print(f"CFM 超参数: eps={cfm_eps}, delta={cfm_delta}, "
          f"num_segments={cfm_num_segs}, boundary={cfm_boundary}, alpha={cfm_alpha}")

    # ── 0. WandB 初始化（对齐 FlowPolicy/train.py）────────────
    project_name = wandb_project or "flowpolicy-maniskill-pretrain"
    run_name     = wandb_run_name or (
        f"{'auto' if auto_mode else 'fixed'}_ep{num_episodes}_"
        f"lr{lr:.0e}_delta{cfm_delta}"
    )
    if use_wandb:
        wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                # 数据 / 任务
                'env_id':         config.env_id,
                'num_episodes':   num_episodes,
                'action_horizon': action_horizon,
                'obs_dim':        None,   # 数据集加载后更新
                'action_dim':     None,
                # CFM 超参数
                'cfm_eps':        cfm_eps,
                'cfm_delta':      cfm_delta,
                'cfm_num_segs':   cfm_num_segs,
                'cfm_boundary':   cfm_boundary,
                'cfm_alpha':      cfm_alpha,
                # 优化器
                'learning_rate':  lr,
                'batch_size':     batch_size,
                'max_epochs':     max_epochs,
                'weight_decay':   getattr(config.pretrain, 'weight_decay', 1e-6),
                'max_grad_norm':  config.pretrain.max_grad_norm,
                # EMA
                'ema_decay':      getattr(config.pretrain, 'ema_decay', 0.9999),
                'ema_use_warmup': getattr(config.pretrain, 'ema_use_warmup', False),
                # 模式
                'auto_mode':      auto_mode,
                'normalizer':     'LinearNormalizer(mode=limits)',
                'gpu_id':         gpu_id,
            },
            resume='allow',
        )
        print(f"\n✓ WandB 初始化: project={project_name}, run={run_name}")
    else:
        wandb_run = None
        print("\nWandB: 已禁用")

    # ── 1. 数据集 ────────────────────────────────────────────
    dataset = ManiSkillDataset(
        dataset_dir=dataset_path,
        horizon=action_horizon,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        max_episodes=num_episodes,
        obs_key='obs',
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    obs_dim    = dataset.replay_buffer['obs'][0].shape[-1]
    action_dim = dataset.replay_buffer['action'][0].shape[-1]

    print(f"\n数据集信息:")
    print(f"  观测维度:   {obs_dim}")
    print(f"  动作维度:   {action_dim}")
    print(f"  样本数:     {len(dataset)}")

    # 数据集加载完成后更新 wandb config（obs_dim / action_dim 现在已知）
    if wandb_run is not None:
        wandb.config.update({'obs_dim': obs_dim, 'action_dim': action_dim},
                            allow_val_change=True)

    # ── 2. LinearNormalizer（对齐 FlowPolicy）─────────────────
    normalizer = dataset.get_normalizer(mode='limits')   # 映射到 [-1, 1]
    normalizer_device_state = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in normalizer.state_dict().items()
    }

    # 提取 Z-score 等效参数，用于兼容 pipeline.set_normalizer 接口
    # LinearNormalizer normalize: y = x * scale + offset
    # unnormalize:                x = (y - offset) / scale
    # 对应旧接口：mean=offset, std=1/scale（但这里直接用 offset/scale 存储）
    def _get_norm_stats(key):
        """从 LinearNormalizer 提取 offset/scale，供 checkpoint 兼容存储使用。"""
        single = normalizer[key]
        sd = single.params_dict
        offset = sd['offset'].detach().to(device)   # shape: (dim,)
        scale  = sd['scale'].detach().to(device)    # shape: (dim,)
        return offset, scale

    action_offset, action_scale = _get_norm_stats('action')
    obs_offset,    obs_scale    = _get_norm_stats('obs')

    print(f"\n归一化统计 (LinearNormalizer, mode='limits'):")
    print(f"  action offset: {action_offset.cpu().numpy()}")
    print(f"  action scale:  {action_scale.cpu().numpy()}")
    print(f"  obs offset:    {obs_offset.cpu().numpy()}")
    print(f"  obs scale:     {obs_scale.cpu().numpy()}")

    # ── 3. 模型 ──────────────────────────────────────────────
    action_unet = FlowConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        kernel_size=5,
    ).to(device)

    n_params = sum(p.numel() for p in action_unet.parameters())
    print(f"\n模型参数数量:")
    print(f"  FlowConditionalUnet1D: {n_params:,}")

    # ── 4. Pipeline（用于 Auto 模式评估）─────────────────────
    pipeline = _build_pipeline(action_unet, normalizer, obs_dim, action_dim,
                                action_horizon, device)

    # ── 5. 优化器 ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        action_unet.parameters(),
        lr=lr,
        betas=(
            getattr(config.pretrain, 'adam_beta1', 0.95),
            getattr(config.pretrain, 'adam_beta2', 0.999),
        ),
        eps=getattr(config.pretrain,   'adam_eps',      1e-8),
        weight_decay=getattr(config.pretrain, 'weight_decay', 1e-6),
    )

    # ── 6. 学习率调度器 ───────────────────────────────────────
    if getattr(config.pretrain, 'use_lr_scheduler', False):
        warmup_steps = int(getattr(config.pretrain, 'warmup_epochs', 5) * len(dataloader))
        total_steps  = max_epochs * len(dataloader)
        lr_scheduler = get_scheduler(
            getattr(config.pretrain, 'lr_scheduler_type', 'cosine'),
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"学习率调度器: {config.pretrain.lr_scheduler_type}, "
              f"warmup={warmup_steps} steps, total={total_steps} steps")
    else:
        lr_scheduler = None
        print("学习率调度器: 禁用")

    # ── 7. EMA（power-based warmup，对齐 FlowPolicy）─────────
    ema_decay        = getattr(config.pretrain, 'ema_decay',              0.9999)
    ema_min_decay    = getattr(config.pretrain, 'ema_min_decay',          0.0)
    ema_update_after = getattr(config.pretrain, 'ema_update_after_step',  0)
    ema_use_warmup   = getattr(config.pretrain, 'ema_use_warmup',         False)
    ema_inv_gamma    = getattr(config.pretrain, 'ema_inv_gamma',          1.0)
    ema_power        = getattr(config.pretrain, 'ema_power',              0.75)

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

    # ── 8. 训练循环 ───────────────────────────────────────────
    print(f"\n开始训练...")
    if auto_mode:
        print(f"[Auto模式] 每 {auto_eval_interval} epochs 评估一次")
        print(f"[Auto模式] 每次评估 {auto_num_test_episodes} 个episodes")
        print(f"[Auto模式] 成功阈值: {auto_success_threshold} 次")
        print(f"[Auto模式] 最大训练轮数: {max_epochs}")
    else:
        print(f"[固定模式] 训练 {max_epochs} epochs")

    action_unet.train()
    global_step      = 0
    final_epoch      = 0
    training_stopped = False
    avg_loss         = 0.0
    # 保存一个 batch 用于 epoch 结束时计算 train_action_mse_error（对齐 FlowPolicy）
    train_sampling_batch = None

    for epoch in range(max_epochs):
        epoch_loss    = 0.0
        step_losses   = []   # 每个 step 的 raw loss，用于计算 epoch 均值
        progress_bar  = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            obs_seq    = batch['obs'].to(device)       # [B, T, obs_dim]
            action_seq = batch['action'].to(device)    # [B, H, action_dim]

            # 保存第一个 batch 用于 epoch 结束时的 MSE 评估
            if train_sampling_batch is None:
                train_sampling_batch = {k: v.clone() for k, v in batch.items()}

            # ---- 归一化（LinearNormalizer，对齐 FlowPolicy）----
            # obs: 只取第一帧作为全局条件
            obs_raw  = obs_seq[:, 0, :]                # [B, obs_dim]
            obs_cond = normalizer['obs'].normalize(obs_raw.cpu()).to(device)   # [B, obs_dim]
            target   = normalizer['action'].normalize(action_seq.cpu()).to(device)  # [B, H, Da]

            batch_size_actual = obs_cond.shape[0]

            # ===== CFM Loss（完全对齐 FlowPolicy compute_loss）=====
            a0 = torch.randn_like(target)

            # 1. 随机采样 t ∈ (eps, 1)，r = clamp(t + delta, 1)
            t = torch.rand(batch_size_actual, device=device) * (1 - cfm_eps) + cfm_eps
            r = torch.clamp(t + cfm_delta, max=1.0)
            t_exp = t.view(-1, 1, 1).expand_as(target)
            r_exp = r.view(-1, 1, 1).expand_as(target)

            # 2. 插值中间状态
            xt = t_exp * target + (1. - t_exp) * a0
            xr = r_exp * target + (1. - r_exp) * a0

            # 3. Segment 端点
            segments    = torch.linspace(0, 1, cfm_num_segs + 1, device=device)
            seg_idx     = torch.searchsorted(segments, t, side="left").clamp(min=1)
            seg_end     = segments[seg_idx]                            # [B]
            seg_end_exp = seg_end.view(-1, 1, 1).expand_as(target)
            x_at_seg_end = seg_end_exp * target + (1. - seg_end_exp) * a0

            # 4. 速度场预测（t * 99 对齐 FlowPolicy）
            vt = action_unet(sample=xt, timestep=t * 99, global_cond=obs_cond)
            vr = action_unet(sample=xr, timestep=r * 99, global_cond=obs_cond)
            vr = torch.nan_to_num(vr)

            # 5. 欧拉步到 segment 端点
            ft = xt + (seg_end_exp - t_exp) * vt
            if isinstance(cfm_boundary, int) and cfm_boundary == 0:
                fr = x_at_seg_end
            else:
                lt_boundary = t_exp < cfm_boundary
                fr = (lt_boundary  * (xr + (seg_end_exp - r_exp) * vr)
                      + ~lt_boundary * x_at_seg_end)

            # 6. 一致性 loss（主）
            losses_f = torch.square(ft - fr).reshape(batch_size_actual, -1).mean(dim=-1)

            # 7. 速度一致性 loss（辅，权重 alpha）
            if isinstance(cfm_boundary, int) and cfm_boundary == 0:
                losses_v = torch.zeros_like(losses_f)
            else:
                far_from_seg_end = (seg_end - t) > 1.01 * cfm_delta
                far_exp          = far_from_seg_end.view(-1, 1, 1).expand_as(target)
                lt_boundary_v    = t_exp < cfm_boundary
                losses_v = (lt_boundary_v * far_exp
                            * torch.square(vt - vr)).reshape(batch_size_actual, -1).mean(dim=-1)

            loss = torch.mean(losses_f + cfm_alpha * losses_v)
            loss_f_mean = losses_f.mean().item()   # 一致性 loss（主项）
            loss_v_mean = losses_v.mean().item()   # 速度 loss（辅项）

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_unet.parameters(),
                                           max_norm=config.pretrain.max_grad_norm)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # EMA 更新
            ema_unet.step(action_unet.parameters())

            raw_loss_val = loss.item()
            epoch_loss  += raw_loss_val
            step_losses.append(raw_loss_val)
            global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{raw_loss_val:.6f}',
                'LR':   f'{current_lr:.2e}',
                'EMA':  f'{ema_unet.cur_decay_value:.4f}',
            })

            # ── Step 级 WandB 日志（对齐 FlowPolicy：每个非末尾 batch 记录一次）──
            is_last_batch = (batch_idx == len(dataloader) - 1)
            if wandb_run is not None and not is_last_batch:
                step_log = {
                    'train_loss': raw_loss_val,     # 与 FlowPolicy 一致的主键
                    'bc_loss':    raw_loss_val,     # FlowPolicy loss_dict 里的 bc_loss
                    'loss_f':     loss_f_mean,      # 一致性 loss（主项）
                    'loss_v':     loss_v_mean,      # 速度一致性 loss（辅项）
                    'lr':         current_lr,
                    'ema_decay':  ema_unet.cur_decay_value,
                    'global_step': global_step,
                    'epoch':      epoch,
                }
                wandb_run.log(step_log, step=global_step)

        # ── Epoch 结束：计算 epoch 均值 loss ──────────────────────
        avg_loss  = epoch_loss / len(dataloader)
        epoch_log = {
            'train_loss': avg_loss,         # epoch 均值（覆盖 step 级，对齐 FlowPolicy）
            'bc_loss':    avg_loss,
            'lr':         optimizer.param_groups[0]['lr'],
            'ema_decay':  ema_unet.cur_decay_value,
            'global_step': global_step,
            'epoch':      epoch,
        }

        print(f"Epoch {epoch+1}/{max_epochs} 完成:")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  EMA decay: {ema_unet.cur_decay_value:.6f}")

        final_epoch = epoch + 1

        # ── train_action_mse_error（对齐 FlowPolicy sample_every 逻辑）────
        # 每 epoch 在训练 batch 上跑一次单步推理，计算预测动作与真实动作的 MSE
        action_unet.eval()
        with torch.no_grad():
            sb_obs    = train_sampling_batch['obs'].to(device)
            sb_action = train_sampling_batch['action'].to(device)
            sb_obs_cond = normalizer['obs'].normalize(
                sb_obs[:, 0, :].cpu()).to(device)                   # [B, obs_dim]
            sb_target   = normalizer['action'].normalize(
                sb_action.cpu()).to(device)                         # [B, H, Da]

            # 单步 ODE：从纯噪声出发，走一步（t=eps→1）
            z     = torch.randn_like(sb_target)
            t_inf = torch.full((z.shape[0],), cfm_eps, device=device)
            v_inf = action_unet(sample=z, timestep=t_inf * 99, global_cond=sb_obs_cond)
            pred_norm = z + (1.0 - cfm_eps) * v_inf                # 归一化空间预测
            mse = torch.nn.functional.mse_loss(pred_norm, sb_target)
            epoch_log['train_action_mse_error'] = mse.item()
            print(f"  train_action_mse_error: {mse.item():.6f}")
        action_unet.train()

        # Auto 模式：定期评估
        if auto_mode and final_epoch % auto_eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"[Auto模式评估] Epoch {final_epoch}: 开始评估...")

            # 复制 EMA 权重到模型
            ema_unet.copy_to(action_unet.parameters())
            action_unet.eval()

            success_count, total_count, avg_reward = evaluate_model(
                action_unet, normalizer, config, device, auto_num_test_episodes)

            eval_success_rate = success_count / max(total_count, 1)
            print(f"[Auto模式评估] 成功率: {success_count}/{total_count} "
                  f"({eval_success_rate:.2%})")
            print(f"[Auto模式评估] 平均奖励: {avg_reward:.3f}")

            # 记录评估指标
            epoch_log['eval_success_count']  = success_count
            epoch_log['eval_total_episodes'] = total_count
            epoch_log['eval_success_rate']   = eval_success_rate
            epoch_log['eval_mean_reward']    = avg_reward

            if success_count >= auto_success_threshold:
                print(f"\n[Auto模式] 达到成功阈值！停止训练于 epoch {final_epoch}")
                training_stopped = True
                if wandb_run is not None:
                    wandb_run.log(epoch_log, step=global_step)
                break
            else:
                print(f"[Auto模式] 未达到阈值，继续训练...\n")
                action_unet.train()
        else:
            pass

        # ── Epoch 级 WandB 日志（末尾 batch + epoch 汇总，对齐 FlowPolicy）──
        if wandb_run is not None:
            wandb_run.log(epoch_log, step=global_step)
            global_step += 1   # epoch 结束后 global_step +1（对齐 FlowPolicy run() 末尾）

    # ── 9. 保存 checkpoint（使用 EMA 权重）────────────────────
    print(f"\n训练完成，准备保存模型...")
    ema_unet.copy_to(action_unet.parameters())
    os.makedirs(checkpoint_dir, exist_ok=True)

    if auto_mode:
        ckpt_name = f"flowpolicy_auto_epoch{final_epoch}_episodes{num_episodes}.pth"
    else:
        ckpt_name = f"flowpolicy_epoch{num_epochs}_episodes{num_episodes}.pth"

    checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

    # 提取兼容旧接口的 mean/std（从 LinearNormalizer 的 offset/scale 推导）
    # unnormalize: x_real = x_normed * scale + offset
    # 对应 Z-score 接口：mean=offset, std=scale
    torch.save({
        # 模型权重
        'epoch':                   final_epoch - 1,
        'action_unet_state_dict':  action_unet.state_dict(),   # EMA 权重
        'model_state_dict':        action_unet.state_dict(),   # 兼容旧键名
        'optimizer_state_dict':    optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        'ema_unet_state_dict':     ema_unet.state_dict(),
        # 损失 / 配置
        'loss':        avg_loss,
        'config':      config.to_dict(),
        'obs_dim':     obs_dim,
        'action_dim':  action_dim,
        # LinearNormalizer 完整状态（推荐加载方式）
        'normalizer_state_dict': normalizer.state_dict(),
        # 兼容旧接口（set_normalizer(mean, std, mean, std)）
        # LinearNormalizer unnormalize: x = normed * scale + offset
        'action_mean': action_offset.cpu(),   # offset → mean
        'action_std':  action_scale.cpu(),    # scale  → std
        'obs_mean':    obs_offset.cpu(),
        'obs_std':     obs_scale.cpu(),
        # 元信息
        'normalizer_mode': 'limits',          # 标记归一化方式，区别于 Z-score
        'auto_mode':       auto_mode,
        'training_stopped': training_stopped if auto_mode else False,
    }, checkpoint_path)

    print(f"\n预训练完成！")
    print(f"检查点已保存: {checkpoint_path}")
    print(f"  训练模式: {'Auto' if auto_mode else '固定'} ({final_epoch} epochs)")
    print(f"  使用episodes: {num_episodes}")
    print(f"  归一化方式: LinearNormalizer(mode='limits')")
    print(f"  包含EMA权重: ✓  包含 normalizer_state_dict: ✓")
    print(f"  最终损失: {avg_loss:.6f}")
    print(f"{'='*60}\n")

    # 最终汇总日志并关闭 wandb
    if wandb_run is not None:
        wandb_run.summary['final_train_loss']   = avg_loss
        wandb_run.summary['final_epoch']        = final_epoch
        wandb_run.summary['checkpoint_path']    = checkpoint_path
        wandb_run.summary['training_stopped']   = training_stopped
        wandb_run.finish()
        print("✓ WandB run 已关闭")

    return checkpoint_path


# ============================================================
# 测试已保存的 checkpoint
# ============================================================

def test_checkpoint(checkpoint_path, config, gpu_id, num_episodes=10, render=False):
    """加载 checkpoint 并在 ManiSkill 环境中测试。"""
    device = torch.device("cuda:0")

    print(f"\n{'='*60}")
    print(f"测试 FlowPolicy 预训练模型")
    print(f"{'='*60}")
    print(f"检查点路径: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    obs_dim    = ckpt.get('obs_dim',    42)
    action_dim = ckpt.get('action_dim', 4)
    action_horizon = config.action_horizon

    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  归一化方式: {ckpt.get('normalizer_mode', 'unknown')}")

    # 创建模型并加载权重
    action_unet = FlowConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
        kernel_size=5,
    ).to(device)
    action_unet.load_state_dict(ckpt['action_unet_state_dict'])
    action_unet.eval()
    print("✓ 模型权重加载成功")

    # 加载 normalizer
    normalizer = LinearNormalizer()
    if 'normalizer_state_dict' in ckpt:
        normalizer.load_state_dict(ckpt['normalizer_state_dict'])
        print("✓ LinearNormalizer 加载成功")
    else:
        raise KeyError("checkpoint 中未找到 normalizer_state_dict，请使用本脚本训练的 checkpoint")

    # 构建 pipeline
    pipeline = _build_pipeline(action_unet, normalizer, obs_dim, action_dim,
                                action_horizon, device)

    # 创建环境
    env_kwargs = {
        'sim_backend': 'gpu',
        'control_mode': 'pd_ee_delta_pos',
        'obs_mode': 'state',
        'render_mode': 'rgb_array',
        'reward_mode': 'dense',
        'sim_config': dict(
            sim_freq=config.env_kwargs.get('sim_freq', 100),
            control_freq=config.env_kwargs.get('control_freq', 20),
        )
    }
    env = gym.make(config.env_id, reconfiguration_freq=1, **env_kwargs)

    if render:
        from mani_skill.utils.wrappers.record import RecordEpisode
        video_dir = "./test_videos"
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(env, video_dir, save_trajectory=False,
                            save_video=True, info_on_video=True, video_fps=30)
        print(f"视频将保存到: {video_dir}")

    # 生成测试 episode_ids
    gen = torch.Generator(device='cpu')
    gen.manual_seed(20000)
    episode_ids = torch.randint(0, 10000, (num_episodes,), generator=gen).tolist()
    print(f"\n测试 episode_ids: {episode_ids}")

    results = run_episodes(
        env=env,
        pipeline=pipeline,
        normalizer=normalizer,
        episode_ids=episode_ids,
        device=device,
        num_inference_steps=config.sample.eval_num_steps,
        max_steps=config.max_episode_steps,
        action_horizon=action_horizon,
        label="测试",
    )
    env.close()
    return results


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='FlowPolicy CFM 预训练（ManiSkill PickCube）')
    parser.add_argument('--config', type=str, default='config/robot_base.py',
                        help='配置文件路径')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的物理GPU ID')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='训练使用的 episode 数量（覆盖 config）')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='训练 epoch 数量（覆盖 config）')
    parser.add_argument('--auto', action='store_true',
                        help='启用 Auto 模式（自动评估并在达到成功阈值时停止）')
    parser.add_argument('--test', action='store_true',
                        help='测试模式：加载 checkpoint 并在环境中评估')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='测试模式下的 checkpoint 路径')
    parser.add_argument('--num_test_episodes', type=int, default=10,
                        help='测试 episode 数量')
    parser.add_argument('--render', action='store_true',
                        help='测试时保存视频')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='WandB project 名称（默认: flowpolicy-maniskill-pretrain）')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='WandB run 名称（默认: 自动生成）')
    parser.add_argument('--no_wandb', action='store_true',
                        help='禁用 WandB 日志')
    args = parser.parse_args()

    # 加载配置（复用 ml_collections 格式）
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()

    if args.test:
        if not args.checkpoint:
            parser.error("--test 模式需要提供 --checkpoint 路径")
        test_checkpoint(
            checkpoint_path=args.checkpoint,
            config=config,
            gpu_id=args.gpu,
            num_episodes=args.num_test_episodes,
            render=args.render,
        )
    else:
        train(
            config=config,
            gpu_id=args.gpu,
            num_episodes_override=args.num_episodes,
            num_epochs_override=args.num_epochs,
            auto_mode=args.auto,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_name,
            use_wandb=not args.no_wandb,
        )


if __name__ == '__main__':
    main()
