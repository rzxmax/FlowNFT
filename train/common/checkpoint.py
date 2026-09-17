"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    logger,
    os,
    torch,
)

def unwrap_model(model, accelerator):
    from diffusers.utils.torch_utils import is_compiled_module
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, action_unet, action_decoder, epoch, accelerator, ema, trainable_parameters, config,
              linear_normalizer=None, decoder_trained=None, pretrain_loss_type=None):
    """保存检查点

    除模型权重外，额外写入架构元数据 (obs_dim / action_dim / action_horizon / 关键超参)
    以及完整 config 字典，便于测试/推理脚本严格复现训练时的模型结构与推理协议。

    新增字段（与预训练 checkpoint 对齐，让 test/evaluate_checkpoint.py 可以加载 RL 产物）：
      - decoder_trained:        decoder 是否参与训练（velocity_loss 模式下为 False）
      - normalizer_state_dict:  pipeline.linear_normalizer 的 state_dict
      - pretrain_loss_type:     'velocity' 或 'action'，标记预训练阶段的 loss 路径
    """
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-epoch{epoch}")
    os.makedirs(save_root, exist_ok=True)

    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(trainable_parameters, store_temp=True)

        unet_unwrapped = unwrap_model(action_unet, accelerator)
        decoder_unwrapped = unwrap_model(action_decoder, accelerator) if action_decoder is not None else None

        # 推导 decoder_trained（优先用显式参数，否则从 decoder 存在与否兜底）
        if decoder_trained is None:
            decoder_trained = decoder_unwrapped is not None

        # ── 推导架构超参（与 ConditionalUnet1D / ActionDecoder 的构造一致）──
        if decoder_unwrapped is not None:
            action_dim_meta = int(getattr(decoder_unwrapped, 'action_dim',
                                          config.env_kwargs.get('action_dim', 4) if hasattr(config, 'env_kwargs') else 4))
            action_horizon_meta = int(getattr(decoder_unwrapped, 'action_horizon', config.action_horizon))
        else:
            # velocity 模式：从 UNet 反推 action_dim（UNet input_dim 即 action_dim）
            try:
                action_dim_meta = int(unet_unwrapped.input_dim)
            except Exception:
                action_dim_meta = int(config.env_kwargs.get('action_dim', 4)) if hasattr(config, 'env_kwargs') else 4
            action_horizon_meta = int(config.action_horizon)

        # obs_dim 从 UNet 第一个 down_module 的 FiLM cond_encoder 反推
        diffusion_step_embed_dim = 256
        try:
            cond_in_features = unet_unwrapped.down_modules[0][0].cond_encoder[1].in_features
            obs_dim_meta = int(cond_in_features - diffusion_step_embed_dim)
        except Exception:
            obs_dim_meta = None

        # 把 ml_collections 配置转换为普通 dict 便于 torch.load 后随意访问
        try:
            config_dict = config.to_dict()
        except Exception:
            config_dict = None

        # ── 组装 checkpoint dict ──
        ckpt = {
            'action_unet_state_dict': unet_unwrapped.state_dict(),
            'epoch': epoch,
            'obs_dim': obs_dim_meta,
            'action_dim': action_dim_meta,
            'action_horizon': action_horizon_meta,
            'diffusion_step_embed_dim': diffusion_step_embed_dim,
            'unet_class': 'unet.ConditionalUnet1D',
            'action_decoder_class': 'flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet.ActionDecoder',
            'loss_type': getattr(config.train, 'loss_type', 'rollout'),  # RL loss type
            'sample_num_steps': int(config.sample.num_steps),
            'ema_enabled': bool(config.train.ema),
            'config_dict': config_dict,
            # ── 与预训练 checkpoint 对齐的关键字段 ──
            'decoder_trained': bool(decoder_trained),
            'pretrain_loss_type': pretrain_loss_type,  # 可能为 None（旧 checkpoint）
        }

        if decoder_unwrapped is not None:
            ckpt['action_decoder_state_dict'] = decoder_unwrapped.state_dict()

        # 写入 LinearNormalizer state_dict（供 test/evaluate_checkpoint.py 复用）
        if linear_normalizer is not None:
            ckpt['normalizer_state_dict'] = linear_normalizer.state_dict()

        torch.save(ckpt, os.path.join(save_root, "model.pth"))

        if config.train.ema:
            ema.copy_temp_to(trainable_parameters)

        logger.info(f"Saved checkpoint to {save_root}")
        logger.info(
            f"  元数据: obs_dim={obs_dim_meta}, action_dim={action_dim_meta}, "
            f"action_horizon={action_horizon_meta}, "
            f"decoder_trained={decoder_trained}, "
            f"pretrain_loss_type={pretrain_loss_type!r}, "
            f"normalizer={'yes' if linear_normalizer is not None else 'no'}"
        )
