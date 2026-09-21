"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    defaultdict,
    logger,
    np,
    torch,
    tqdm,
    wandb,
)

from train.common.sampling import _unet_timestep_input
from train.reinforcement.old_policy import return_decay
from train.common.checkpoint import unwrap_model

def train_on_episode_samples_stepnft(
    samples,
    action_unet, action_decoder,
    action_unet_old, action_decoder_old,
    action_unet_ref, action_decoder_ref,
    optimizer, accelerator, config, ema, trainable_parameters,
    global_step, epoch, obs_encoder=None, obs_encoder_old=None,
    obs_encoder_ref=None,
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
    if action_decoder is not None:
        action_decoder.train()
    action_unet_old.eval()
    if action_decoder_old is not None:
        action_decoder_old.eval()
    action_unet_ref.eval()
    if action_decoder_ref is not None:
        action_decoder_ref.eval()
    train_visual_encoder = obs_encoder is not None
    if train_visual_encoder:
        obs_encoder.train()
        if obs_encoder_old is not None:
            obs_encoder_old.eval()
        if obs_encoder_ref is None:
            raise ValueError('训练图像策略时必须提供 obs_encoder_ref')
        obs_encoder_ref.eval()

    decoder_active = action_decoder is not None

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
    batch_images = []
    batch_agent_pos = []
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
        raw_obs_list = sample.get("raw_obs_list", [])
        num_flow_calls = len(nft_traces)
        if train_visual_encoder and len(raw_obs_list) != num_flow_calls:
            raise ValueError(
                "图像 StepNFT 样本缺少与 nft_traces 对齐的 raw_obs_list"
            )

        for flow_idx in range(num_flow_calls):
            trace = nft_traces[flow_idx]
            if train_visual_encoder:
                batch_images.append(raw_obs_list[flow_idx]['image'])
                batch_agent_pos.append(raw_obs_list[flow_idx]['agent_pos'])
            else:
                batch_obs.append(obs_list[flow_idx])
            batch_xt.append(trace["nft_xt"])
            batch_v_old.append(trace["nft_v"])
            batch_xnext.append(trace["nft_xnext"])
            batch_step_index.append(int(trace["nft_step_index"]))
            batch_std.append(float(trace["nft_std"]))
            batch_advantages.append(trajectory_advantage)

    if len(batch_xt) == 0:
        logger.warning(f"Epoch {epoch}: 没有有效 StepNFT 样本，跳过训练")
        return global_step, 0.0

    # ========== 阶段2：组装 tensors ==========
    obs_cond = None
    if not train_visual_encoder:
        obs_cond = torch.cat(batch_obs, dim=0)  # [N, obs_dim]
    xt_all = torch.stack(batch_xt, dim=0)   # [N, action_horizon, action_dim]
    v_old_all = torch.stack(batch_v_old, dim=0)
    xnext_all = torch.stack(batch_xnext, dim=0)
    step_idx_all = torch.tensor(batch_step_index, device=accelerator.device, dtype=torch.long)
    std_all = torch.tensor(batch_std, device=accelerator.device, dtype=torch.float32)
    adv_all = torch.tensor(batch_advantages, device=accelerator.device, dtype=torch.float32)

    total_batch_size = len(batch_xt)
    logger.info(f"Epoch {epoch} [StepNFT]: 训练 {total_batch_size} 个 nft 样本（每个仅单步 forward）")

    # ========== 阶段3：打乱数据 ==========
    perm = torch.randperm(total_batch_size, device=accelerator.device)
    if train_visual_encoder:
        perm_list = perm.detach().cpu().tolist()
        batch_images = [batch_images[index] for index in perm_list]
        batch_agent_pos = [batch_agent_pos[index] for index in perm_list]
    else:
        obs_cond = obs_cond[perm]
    xt_all = xt_all[perm]
    v_old_all = v_old_all[perm]
    xnext_all = xnext_all[perm]
    step_idx_all = step_idx_all[perm]
    std_all = std_all[perm]
    adv_all = adv_all[perm]

    # ========== 阶段4：分批训练 ==========
    num_batches = max(
        1, (total_batch_size + config.train.batch_size - 1) // config.train.batch_size
    )
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

        if train_visual_encoder:
            b_image = torch.stack(batch_images[start_idx:end_idx]).to(
                accelerator.device, dtype=torch.float32, non_blocking=True
            )
            b_agent_pos = torch.stack(batch_agent_pos[start_idx:end_idx]).to(
                accelerator.device, dtype=torch.float32, non_blocking=True
            )
        else:
            b_obs = obs_cond[start_idx:end_idx]
        b_xt = xt_all[start_idx:end_idx]
        b_v_old = v_old_all[start_idx:end_idx]
        b_xnext = xnext_all[start_idx:end_idx]
        b_step_idx = step_idx_all[start_idx:end_idx]
        b_std = std_all[start_idx:end_idx]
        b_adv = adv_all[start_idx:end_idx]

        bsz = end_idx - start_idx

        with accelerator.autocast():
            if train_visual_encoder:
                # Re-encode the rollout RGB-D window inside the autograd graph,
                # so NFT jointly updates the visual representation and UNet.
                b_obs = obs_encoder(b_image, b_agent_pos)
                with torch.no_grad():
                    b_obs_ref = obs_encoder_ref(b_image, b_agent_pos)
            else:
                b_obs_ref = b_obs

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
            v_theta = action_decoder(current_pred_raw) if decoder_active else current_pred_raw

            # Reference model（KL 项用，no grad）
            with torch.no_grad():
                ref_pred_raw = action_unet_ref(
                    sample=b_xt,
                    timestep=t_unet,
                    global_cond=b_obs_ref,
                )
                v_ref = action_decoder_ref(ref_pred_raw) if decoder_active else ref_pred_raw

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
        accumulation_group_start = (
            batch_idx // effective_grad_accum_steps
        ) * effective_grad_accum_steps
        accumulation_group_size = min(
            effective_grad_accum_steps,
            num_batches - accumulation_group_start,
        )
        scaled_loss = loss / accumulation_group_size
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
        should_step = (
            current_accumulated_steps % effective_grad_accum_steps == 0
            or batch_idx == num_batches - 1
        )
        if should_step:
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
                    if train_visual_encoder and len(optimizer.param_groups) > 1:
                        log_dict["optim/obs_encoder_learning_rate"] = (
                            optimizer.param_groups[1]['lr']
                        )
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

                    if decoder_active:
                        for cur_p, old_p, ref_p in zip(
                            unwrap_model(action_decoder, accelerator).parameters(),
                            action_decoder_old.parameters(),
                            action_decoder_ref.parameters()
                        ):
                            current_old_distance += (cur_p - old_p).pow(2).sum().item()
                            current_ref_distance += (cur_p - ref_p).pow(2).sum().item()
                            old_ref_distance += (old_p - ref_p).pow(2).sum().item()

                    if train_visual_encoder:
                        for cur_p, old_p, ref_p in zip(
                            unwrap_model(obs_encoder, accelerator).parameters(),
                            obs_encoder_old.parameters(),
                            obs_encoder_ref.parameters(),
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

                    if decoder_active:
                        for src_param, tgt_param in zip(
                            unwrap_model(action_decoder, accelerator).parameters(),
                            action_decoder_old.parameters()
                        ):
                            tgt_param.data.copy_(
                                tgt_param.data * decay + src_param.data * (1.0 - decay)
                            )

                    if train_visual_encoder:
                        for src_param, tgt_param in zip(
                            unwrap_model(obs_encoder, accelerator).parameters(),
                            obs_encoder_old.parameters(),
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
