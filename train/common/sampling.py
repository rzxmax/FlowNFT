"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    math,
    torch,
)

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

            # UNet (+ 可选 Decoder) forward
            # velocity_loss 预训练：UNet 直出速度场 → action_decoder=None
            # action_loss   预训练：UNet 输出经 Decoder 映射为速度
            pred_raw = action_unet(
                sample=x_t,
                timestep=t_input,
                global_cond=obs_embeds,
            )
            v_t = action_decoder(pred_raw) if action_decoder is not None else pred_raw

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
    与 test/evaluate_checkpoint.py 完全对齐的 ODE 推理。

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
