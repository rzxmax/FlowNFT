"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    Accelerator,
    ActionDecoder,
    ConditionalUnet1D,
    FlowMatchEulerDiscreteScheduler,
    RobotActionPipeline,
    gym,
    json,
    logger,
    logging,
    np,
    os,
    pipeline_with_logprob,
    torch,
)

from train.common.observations import _prepare_obs_tensor
from train.common.sampling import denoise_ode_inference
from train.common.checkpoint import unwrap_model

def eval_in_env(pipeline, action_unet, action_decoder, env, config, accelerator,
                global_step, num_test_episodes=3, observation_adapter=None):
    """
    在环境中评估策略（与预训练 evaluate_pretrain_model 推理协议对齐）

    支持：
    - action_decoder=None（velocity_loss 模式，UNet 直出速度场）
    - pipeline.linear_normalizer 存在时自动做 obs/action 归一化反归一化

    参数:
        pipeline: RobotActionPipeline
        action_unet: ConditionalUnet1D模型
        action_decoder: ActionDecoder 或 None
        env: ManiSkill环境
        config: 配置
        accelerator: Accelerator
        global_step: 全局步数
        num_test_episodes: 测试episode数量
    """
    action_unet.eval()
    if action_decoder is not None:
        action_decoder.eval()

    # ── 对齐 sample 阶段：用 pipeline 上挂的 LinearNormalizer 做 obs 归一化 / action 反归一化 ──
    linear_normalizer = getattr(pipeline, 'linear_normalizer', None)
    if linear_normalizer is not None:
        action_unnormalize_fn = lambda a: linear_normalizer['action'].unnormalize(a)
    else:
        action_unnormalize_fn = lambda a: a
    if observation_adapter is None:
        observation_adapter = getattr(pipeline, 'observation_adapter', None)
    if observation_adapter is None:
        from train.reinforcement.observation import StateObservationAdapter
        observation_adapter = StateObservationAdapter(
            accelerator.device, normalizer=linear_normalizer
        )

    all_episode_rewards = []
    all_episode_lengths = []

    max_steps = config.max_episode_steps if hasattr(config, 'max_episode_steps') else 500

    for test_idx in range(num_test_episodes):
        obs, info = env.reset(seed=10000 + test_idx + accelerator.process_index * 100)
        observation_adapter.reset()
        obs_tensor = observation_adapter.encode(obs)

        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            with torch.no_grad():
                eval_gen = torch.Generator(device=accelerator.device)
                eval_gen.manual_seed(test_idx + 10000)

                result = pipeline_with_logprob(
                    pipeline,
                    obs=obs_tensor,
                    num_inference_steps=config.sample.eval_num_steps if hasattr(config.sample, 'eval_num_steps') else 10,
                    output_type="clipped",
                    return_dict=True,
                    action_decoder=action_decoder,
                    generator=eval_gen,
                )

                # actions 在归一化空间
                actions = result["actions"][0]

            # 执行完整的 action_horizon 步动作（送入 env 前反归一化）
            for j in range(config.action_horizon):
                if done or episode_length >= max_steps:
                    break

                action = actions[j].contiguous() if isinstance(actions, torch.Tensor) else actions[j]
                action_env = action_unnormalize_fn(action)
                obs, reward, terminated, truncated, info = env.step(action_env)

                episode_reward += reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                episode_length += 1
                done = terminated
                if not done and episode_length < max_steps:
                    observation_adapter.observe(obs)

            if not done and episode_length < max_steps:
                obs_tensor = observation_adapter.condition()

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
    if action_decoder is not None:
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
    在 RL 训练循环中内联评估当前策略性能，与 test/evaluate_checkpoint.py 的推理逻辑完全对齐。

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
    decoder_unwrapped = unwrap_model(action_decoder, accelerator) if action_decoder is not None else None

    # ── 临时拷贝 EMA 权重 ──
    ema_active = (ema is not None) and (trainable_parameters is not None)
    if ema_active:
        ema.copy_ema_to(trainable_parameters, store_temp=True)
        logger.info("[Inline Eval] 已将 EMA 权重拷贝到模型（评估用）")

    unet_unwrapped.eval()
    if decoder_unwrapped is not None:
        decoder_unwrapped.eval()

    # ── 对齐 sample 阶段：用 pipeline 上挂的 LinearNormalizer 做 obs / action 处理 ──
    linear_normalizer = getattr(pipeline, 'linear_normalizer', None)
    if linear_normalizer is not None:
        action_unnormalize_fn = lambda a: linear_normalizer['action'].unnormalize(a)
    else:
        action_unnormalize_fn = lambda a: a
    observation_adapter = getattr(pipeline, 'observation_adapter', None)
    if observation_adapter is None:
        from train.reinforcement.observation import StateObservationAdapter
        observation_adapter = StateObservationAdapter(device, normalizer=linear_normalizer)

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
            observation_adapter.reset()
            obs_tensor = observation_adapter.encode(obs)

            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done and episode_length < max_steps:
                eval_rng = torch.Generator(device=device)
                eval_rng.manual_seed(episode_id * 10000 + episode_length)

                # obs 已在外层归一化 → 这里传 obs_normalizer=None 避免重复归一化
                actions_batch = denoise_ode_inference(
                    obs_tensor=obs_tensor,
                    action_unet=unet_unwrapped,
                    action_decoder=decoder_unwrapped,
                    num_steps=num_inference_steps,
                    action_horizon=action_horizon,
                    action_dim=action_dim,
                    device=device,
                    generator=eval_rng,
                    obs_normalizer=None,
                )
                actions = actions_batch[0]  # 归一化空间

                for j in range(action_horizon):
                    if done or episode_length >= max_steps:
                        break
                    action = actions[j].contiguous()
                    # 反归一化后送入环境
                    action_env = action_unnormalize_fn(action)
                    if isinstance(action_env, torch.Tensor):
                        action_env = action_env.cpu().numpy()
                    obs, reward, terminated, truncated, info = eval_env.step(action_env)
                    episode_reward += float(reward.item() if isinstance(reward, torch.Tensor) else reward)
                    episode_length += 1
                    done = bool(terminated)
                    if not done and episode_length < max_steps:
                        observation_adapter.observe(obs)

                if not done and episode_length < max_steps:
                    obs_tensor = observation_adapter.condition()

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
        if decoder_unwrapped is not None:
            decoder_unwrapped.train()
