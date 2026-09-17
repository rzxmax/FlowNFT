"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    gym,
    np,
    pipeline_with_logprob,
    torch,
)

from train.common.observations import extract_image_obs
from train.common.observations import extract_pointcloud_obs
from train.common.observations import stack_image_obs
from train.common.observations import stack_pointcloud_obs

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
