"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    pipeline_with_logprob,
    torch,
)

from train.common.sampling import denoise_with_sde_snap

def sample_complete_trajectories(
    env, pipeline, action_unet, action_decoder,
    num_state_groups, samples_per_group,
    config, accelerator, autocast, seed_offset,
    observation_adapter=None,
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
    if action_decoder is not None:
        action_decoder.eval()

    # ── 对齐预训练：adapter 负责观测预处理，normalizer 负责动作反归一化 ──
    # 当 pipeline.linear_normalizer 为 None（无 normalizer 的旧 checkpoint）时，
    # 两个 fn 均为 identity，保持向后兼容
    linear_normalizer = getattr(pipeline, 'linear_normalizer', None)
    if linear_normalizer is not None:
        action_unnormalize_fn = lambda a: linear_normalizer['action'].unnormalize(a)
    else:
        action_unnormalize_fn = lambda a: a

    if observation_adapter is None:
        from train.reinforcement.observation import StateObservationAdapter
        observation_adapter = StateObservationAdapter(
            accelerator.device, normalizer=linear_normalizer
        )

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
    observation_adapter.reset()
    obs_tensor = observation_adapter.encode(obs)

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
                        # obs 已在上方用 LinearNormalizer 归一化过 → 这里传 None 避免重复归一化
                        sde_action_dim = getattr(pipeline, 'action_dim', None)
                        if sde_action_dim is None and action_decoder is not None:
                            sde_action_dim = getattr(action_decoder, 'action_dim', None)
                        if sde_action_dim is None:
                            single_space = getattr(env, 'single_action_space', env.action_space)
                            sde_action_dim = single_space.shape[-1]
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
                            obs_normalizer=None,
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

            # 最后一批动作只执行剩余步数，避免 300 步上限被 16 步动作块推进到 304。
            remaining_steps = max_steps - flow_call_count * config.action_horizon
            for j in range(min(config.action_horizon, remaining_steps)):
                if env_done.all():
                    break

                # 每步动态计算active_mask
                step_active_mask = ~env_done

                # action_step：归一化空间（来自 pipeline 输出）→ 保留给 history 供训练用
                action_step = actions_batch[:, j].contiguous()

                # ✅ 滑动窗口：在执行前保存当前obs（归一化空间）和归一化空间 action
                obs_tensor_before = obs_tensor
                for env_idx in range(num_envs):
                    if step_active_mask[env_idx]:
                        env_obs_history[env_idx].append(obs_tensor_before[env_idx:env_idx+1].clone())
                        env_action_history[env_idx].append(action_step[env_idx:env_idx+1].clone())

                # 送入环境前反归一化到真实动作空间
                action_step_env = action_unnormalize_fn(action_step)
                obs, reward, terminated, truncated, info = env.step(action_step_env)
                # StepNFT 只在每个 action chunk 开头使用条件；先低成本更新
                # 原始历史，chunk 结束后再运行一次视觉编码器。
                if use_stepnft:
                    observation_adapter.observe(obs)
                else:
                    obs_tensor = observation_adapter.encode(obs)

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

            if use_stepnft:
                obs_tensor = observation_adapter.condition()

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
