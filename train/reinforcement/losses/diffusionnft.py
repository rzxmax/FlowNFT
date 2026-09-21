"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    defaultdict,
    logger,
    np,
    ode_step_with_logprob,
    torch,
    tqdm,
    wandb,
)

from train.common.sampling import _unet_timestep_input
from train.reinforcement.old_policy import return_decay
from train.common.checkpoint import unwrap_model

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
        # velocity_loss 模式：action_decoder 为 None，UNet 直接输出速度
        pred_raw = action_unet(
            sample=current_sample,
            timestep=t_normalized,
            global_cond=batch_obs
        )
        velocity = action_decoder(pred_raw) if action_decoder is not None else pred_raw

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
    global_step, epoch, obs_encoder=None, obs_encoder_old=None,
    obs_encoder_ref=None,
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
    if action_decoder is not None:
        action_decoder.train()
    action_unet_old.eval() # 这里是 eval 因为函数外后面会更新
    if action_decoder_old is not None:
        action_decoder_old.eval()
    action_unet_ref.eval()
    if action_decoder_ref is not None:
        action_decoder_ref.eval()

    # velocity_loss 模式：decoder 旁路，三组 decoder（current/old/ref）均为 None
    decoder_active = action_decoder is not None

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
                    old_prediction = action_decoder_old(old_pred_raw) if decoder_active else old_pred_raw

                # Current model（with grad，DiffusionNFT Line 886-892）
                current_pred_raw = action_unet(
                    sample=xt,
                    timestep=t_unet,
                    global_cond=batch_obs
                )
                current_prediction = action_decoder(current_pred_raw) if decoder_active else current_pred_raw

                # Reference model（no grad，DiffusionNFT Line 894-907）
                with torch.no_grad():
                    ref_pred_raw = action_unet_ref(
                        sample=xt,
                        timestep=t_unet,
                        global_cond=batch_obs
                    )
                    ref_prediction = action_decoder_ref(ref_pred_raw) if decoder_active else ref_pred_raw

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

                        # 加上decoder的差异（仅当 decoder_active）
                        if decoder_active:
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

                        # 更新Decoder（仅当 decoder_active）
                        if decoder_active:
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
