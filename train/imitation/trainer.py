"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    ActionDecoder,
    ConditionalUnet1D,
    FlowMatchEulerDiscreteScheduler,
    RobotActionPipeline,
    os,
    pipeline_with_logprob,
    torch,
    tqdm,
)

from train.common.sampling import _unet_timestep_input
from train.common.observations import _PCD_HAS_PYTORCH3D, batched_fps_sample
from train.imitation.checkpoint_files import create_run_dir, checkpoint_path, prune_checkpoints
from train.imitation.data_sources import load_dataset_paths
from train.imitation.evaluator import evaluate_pretrain_model

def pretrain_model(config, gpu_id, wandb_run=None):
    """
    使用成功轨迹对 Flow 策略进行预训练。

    支持三种观测模式（通过 config.pretrain.obs_mode 切换）：
      "state"      — 扁平化状态向量，与原有逻辑完全相同
      "pointcloud" — PointNet 编码点云 + agent_pos，编码结果作为 obs_cond
      "image"      — ResNet 编码 RGB(D) 图像 + agent_pos，编码结果作为 obs_cond

    三种模式共享相同的训练方式：
        generate_with_grad(pipeline, obs=obs_cond, ...)
        loss = MSE(pred_actions, target_actions)
    """
    from torch.utils.data import DataLoader
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        InitProcessGroupKwargs,
        broadcast,
        broadcast_object_list,
        set_seed,
    )
    from diffusers.training_utils import EMAModel
    from diffusers.optimization import get_scheduler

    accumulation_steps = int(getattr(
        config.pretrain, 'gradient_accumulation_steps', 1
    ))
    if accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps 必须至少为 1')
    mixed_precision = str(getattr(config.pretrain, 'mixed_precision', 'no'))
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(getattr(
            config.pretrain, 'find_unused_parameters', False
        ))
    )
    process_group_kwargs = InitProcessGroupKwargs(
        backend=str(getattr(config.pretrain, 'distributed_backend', 'nccl'))
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=accumulation_steps,
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=int(config.seed),
        ),
        kwargs_handlers=[ddp_kwargs, process_group_kwargs],
    )
    device = accelerator.device
    set_seed(int(config.seed), device_specific=True)
    print = accelerator.print

    owns_wandb_run = False
    project = os.environ.get('WANDB_PROJECT')
    if wandb_run is None and project and accelerator.is_main_process:
        import wandb

        wandb_run = wandb.init(
            project=project,
            entity=os.environ.get('WANDB_ENTITY') or None,
            config=config.to_dict(),
            job_type='imitation',
            tags=[config.pretrain.obs_mode, config.pretrain.loss_type],
        )
        owns_wandb_run = True

    obs_mode = getattr(config.pretrain, 'obs_mode', 'state')
    pretrain_loss_type = getattr(config.pretrain, 'loss_type', 'action')
    if pretrain_loss_type not in ('action', 'velocity'):
        raise ValueError(
            f"未知的 config.pretrain.loss_type: {pretrain_loss_type!r}，"
            f"应为 'action' 或 'velocity'"
        )

    auto_mode = bool(getattr(config.pretrain, 'auto_mode', False))
    if auto_mode:
        eval_interval = int(config.pretrain.auto_eval_interval)
        num_test_episodes = int(config.pretrain.auto_num_test_episodes)
        success_threshold = int(config.pretrain.auto_success_threshold)
        max_epochs = int(config.pretrain.auto_max_epochs)
    else:
        max_epochs = int(config.pretrain.num_epochs)

    launched_from_config = (
        os.environ.get('FLOWNFT_ACCELERATE_WORKER') == '1'
        or accelerator.num_processes > 1
    )
    configured_gpu_ids = (
        list(getattr(config.pretrain, 'gpu_ids', [gpu_id]))
        if launched_from_config else [int(gpu_id)]
    )
    physical_gpu = (
        configured_gpu_ids[accelerator.local_process_index]
        if accelerator.local_process_index < len(configured_gpu_ids)
        else gpu_id
    )

    print(f"\n{'='*60}")
    print(f"开始预训练 Flow 策略")
    print(f"{'='*60}")
    print(f"物理 GPU IDs: {configured_gpu_ids}")
    print(f"分布式进程数: {accelerator.num_processes}")
    print(f"主进程设备: {device}（物理 GPU {physical_gpu}）")
    print(f"观测模式: {obs_mode}")
    print(f"Loss 类型: {pretrain_loss_type}"
          + ("  (multi-step rollout MSE)" if pretrain_loss_type == "action"
             else "  (Flow Matching velocity MSE)"))
    dataset_paths = load_dataset_paths(config.pretrain.dataset_path)
    print(f"数据集清单: {config.pretrain.dataset_path}")
    print(f"H5 文件数: {len(dataset_paths)}")
    for path in dataset_paths:
        print(f"  - {path}")
    print(f"每个 H5 最多读取 episodes: {config.pretrain.num_episodes}")
    print(f"训练 epochs: {config.pretrain.num_epochs}")

    # ------------------------------------------------------------------
    # 1. 加载数据集
    # ------------------------------------------------------------------
    if obs_mode == 'pointcloud':
        from dataset import ManiSkillPointCloudDataset
        from models.pointcloud_encoder import PointCloudObsEncoder

        pcd_n_obs_steps    = getattr(config.pretrain, 'pcd_n_obs_steps', 2)
        pcd_enc_out_dim    = getattr(config.pretrain, 'pcd_encoder_output_dim', 256)
        pcd_agent_pos_dim  = getattr(config.pretrain, 'pcd_agent_pos_dim', 19)
        pcd_state_mlp_out  = getattr(config.pretrain, 'pcd_state_mlp_out', 64)
        pcd_use_layernorm  = getattr(config.pretrain, 'pcd_use_layernorm', True)
        pcd_final_norm     = getattr(config.pretrain, 'pcd_final_norm', 'layernorm')

        dataset = ManiSkillPointCloudDataset(
            h5_path=dataset_paths,
            horizon=config.action_horizon,
            pad_before=0,
            val_ratio=0.0,
            max_episodes=config.pretrain.num_episodes,
            max_episodes_per_file=True,
            max_pts=getattr(config.pretrain, 'pcd_max_pts', 4096),
            obs_steps=pcd_n_obs_steps,
        )

        action_dim = dataset.action_dim
        # obs_cond 维度 = (pcd_enc_out_dim + pcd_state_mlp_out) * pcd_n_obs_steps
        obs_dim = (pcd_enc_out_dim + pcd_state_mlp_out) * pcd_n_obs_steps

        # replay 点云读入后保留有效点数，batch 内在线 FPS。
        pcd_n_points  = getattr(config.pretrain, 'pcd_n_points', 512)
        pcd_online_fps = bool(getattr(dataset, 'online_fps', False))
        if pcd_online_fps and not _PCD_HAS_PYTORCH3D:
            print(f"  [警告] 在线 FPS 启用但 pytorch3d 未安装，"
                  f"将退化为 numpy 随机均匀采样（性能/质量略降）")

        # 创建归一化器（agent_pos + action → limits 归一化到 [-1, 1]）
        normalizer = dataset.get_normalizer().to(device)

        print(f"数据集信息:")
        print(f"  点云帧数 (n_obs_steps): {pcd_n_obs_steps}")
        print(f"  点云编码维度: {pcd_enc_out_dim}")
        print(f"  agent_pos 维度: {pcd_agent_pos_dim}")
        print(f"  obs_cond 维度 (global_cond_dim): {obs_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  样本数: {len(dataset)}")
        print(f"  在线 FPS: {'ON' if pcd_online_fps else 'OFF'}")
        if pcd_online_fps:
            print(f"  每帧 FPS 目标点数: {pcd_n_points}")
        print(f"  归一化器: agent_pos + action（limits 模式）")

    elif obs_mode == 'image':
        from dataset import ManiSkillImageDataset
        from models.image_encoder import ImageObsEncoder

        img_n_obs_steps     = getattr(config.pretrain, 'img_n_obs_steps', 2)
        img_camera          = getattr(config.pretrain, 'img_camera', 'base_camera')
        img_use_depth       = bool(getattr(config.pretrain, 'img_use_depth', True))
        img_depth_max       = float(getattr(config.pretrain, 'img_depth_max', 5.0))
        img_resnet_name     = getattr(config.pretrain, 'img_resnet_name', 'resnet18')
        img_resnet_feat_dim = int(getattr(config.pretrain, 'img_resnet_feat_dim', 512))
        img_agent_pos_dim   = int(getattr(config.pretrain, 'img_agent_pos_dim', 19))
        img_state_mlp_out   = int(getattr(config.pretrain, 'img_state_mlp_out', 64))
        img_use_groupnorm   = bool(getattr(config.pretrain, 'img_use_groupnorm', True))

        dataset = ManiSkillImageDataset(
            h5_path=dataset_paths,
            horizon=config.action_horizon,
            pad_before=0,
            val_ratio=0.0,
            max_episodes=config.pretrain.num_episodes,
            max_episodes_per_file=True,
            camera=img_camera,
            use_depth=img_use_depth,
            depth_max=img_depth_max,
            image_size=getattr(config.pretrain, 'img_size', None),
            obs_steps=img_n_obs_steps,
        )

        img_in_channels = 4 if img_use_depth else 3
        action_dim = dataset.action_dim
        # obs_cond 维度 = (img_resnet_feat_dim + img_state_mlp_out) * img_n_obs_steps
        obs_dim = (img_resnet_feat_dim + img_state_mlp_out) * img_n_obs_steps

        # 归一化器（agent_pos + action → limits 归一化到 [-1, 1]，图像不进 normalizer）
        normalizer = dataset.get_normalizer().to(device)

        print(f"数据集信息:")
        print(f"  图像帧数 (n_obs_steps): {img_n_obs_steps}")
        print(f"  图像通道数: {img_in_channels}  ({'RGBD' if img_use_depth else 'RGB'})")
        print(f"  相机: {img_camera}")
        print(f"  ResNet 主干: {img_resnet_name}  feat_dim={img_resnet_feat_dim}")
        print(f"  agent_pos 维度: {img_agent_pos_dim}")
        print(f"  obs_cond 维度 (global_cond_dim): {obs_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  样本数: {len(dataset)}")
        print(f"  深度归一化 depth_max: {img_depth_max} m" if img_use_depth else "")
        print(f"  归一化器: agent_pos + action（limits 模式），图像不归一化")

    else:
        from dataset import ManiSkillStateDataset

        dataset = ManiSkillStateDataset(
            h5_path=dataset_paths,
            horizon=config.action_horizon,
            pad_before=0,
            val_ratio=0.0,
            max_episodes=config.pretrain.num_episodes,
            max_episodes_per_file=True,
        )

        obs_dim    = dataset.obs_dim
        action_dim = dataset.action_dim

        # state 模式：与 pointcloud/image 模式对齐，对 obs 和 action 做 limits 归一化
        normalizer = dataset.get_normalizer().to(device)

        print(f"数据集信息:")
        print(f"  观测维度: {obs_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  样本数: {len(dataset)}")
        print(f"  归一化器: obs + action（limits 模式）")

    training_episode_count = len(dataset.episodes)
    print(f"实际加载 episodes: {training_episode_count}")

    batch_size_per_gpu = int(getattr(
        config.pretrain, 'batch_size_per_gpu', config.pretrain.batch_size
    ))
    num_workers = int(getattr(config.pretrain, 'num_workers', 4))
    if batch_size_per_gpu < 1 or num_workers < 0:
        raise ValueError('batch_size_per_gpu 必须为正数，num_workers 不能为负数')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(getattr(config.pretrain, 'pin_memory', True)),
        persistent_workers=(
            bool(getattr(config.pretrain, 'persistent_workers', True))
            and num_workers > 0
        ),
    )

    # ------------------------------------------------------------------
    # 2. 创建模型
    # ------------------------------------------------------------------
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim,
    ).to(device)

    action_decoder = ActionDecoder(
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        hidden_dim=256,
        num_layers=3,
    ).to(device)

    # 观测编码器：pointcloud / image 模式才需要（state 模式直接用扁平向量）
    if obs_mode == 'pointcloud':
        obs_encoder = PointCloudObsEncoder(
            n_obs_steps=pcd_n_obs_steps,
            pcd_out_dim=pcd_enc_out_dim,
            agent_pos_dim=pcd_agent_pos_dim,
            state_mlp_hidden=[64, pcd_state_mlp_out],
            use_layernorm=pcd_use_layernorm,
            final_norm=pcd_final_norm,
        ).to(device)
        print(f"模型参数数量:")
        print(f"  ObsEncoder 参数: {sum(p.numel() for p in obs_encoder.parameters())}")
        print(f"  UNet 参数:        {sum(p.numel() for p in action_unet.parameters())}")
        print(f"  ActionDecoder 参数: {sum(p.numel() for p in action_decoder.parameters())}")
    elif obs_mode == 'image':
        obs_encoder = ImageObsEncoder(
            n_obs_steps=img_n_obs_steps,
            in_channels=img_in_channels,
            resnet_name=img_resnet_name,
            img_feat_dim=img_resnet_feat_dim,
            agent_pos_dim=img_agent_pos_dim,
            state_mlp_hidden=[64, img_state_mlp_out],
            use_groupnorm=img_use_groupnorm,
        ).to(device)
        print(f"模型参数数量:")
        print(f"  ImageObsEncoder 参数: {sum(p.numel() for p in obs_encoder.parameters())}")
        print(f"  UNet 参数:        {sum(p.numel() for p in action_unet.parameters())}")
        print(f"  ActionDecoder 参数: {sum(p.numel() for p in action_decoder.parameters())}")
    else:
        obs_encoder = None
        print(f"模型参数数量:")
        print(f"  UNet 参数:        {sum(p.numel() for p in action_unet.parameters())}")
        print(f"  ActionDecoder 参数: {sum(p.numel() for p in action_decoder.parameters())}")

    # ------------------------------------------------------------------
    # 3. 创建管线
    # ------------------------------------------------------------------
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=config.action_horizon,
        device=str(device),
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder

    # ------------------------------------------------------------------
    # 4. 优化器
    # ------------------------------------------------------------------
    # velocity 模式：UNet 直接输出速度场，ActionDecoder 不参与训练，
    # 因此也不进 optimizer / EMA / grad clip，避免在零梯度下被
    # Adam 二阶动量 / EMA 累积无意义地扰动。
    decoder_trains_in_pretrain = (pretrain_loss_type != "velocity")

    trainable_params = list(action_unet.parameters())
    if decoder_trains_in_pretrain:
        trainable_params += list(action_decoder.parameters())
    if obs_encoder is not None:
        trainable_params += list(obs_encoder.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.pretrain.learning_rate,
        betas=(
            float(getattr(config.pretrain, 'adam_beta1', 0.9)),
            float(getattr(config.pretrain, 'adam_beta2', 0.999)),
        ),
        eps=float(getattr(config.pretrain, 'adam_eps', 1e-8)),
        weight_decay=getattr(config.pretrain, 'weight_decay', 1e-6),
    )

    # ------------------------------------------------------------------
    # 5. DDP/DataLoader 准备与学习率调度器
    # ------------------------------------------------------------------
    prepared = [action_unet]
    if decoder_trains_in_pretrain:
        prepared.append(action_decoder)
    if obs_encoder is not None:
        prepared.append(obs_encoder)
    prepared.extend([optimizer, dataloader])
    prepared = list(accelerator.prepare(*prepared))

    prepared_index = 0
    action_unet = prepared[prepared_index]
    prepared_index += 1
    if decoder_trains_in_pretrain:
        action_decoder = prepared[prepared_index]
        prepared_index += 1
    if obs_encoder is not None:
        obs_encoder = prepared[prepared_index]
        prepared_index += 1
    optimizer = prepared[prepared_index]
    dataloader = prepared[prepared_index + 1]

    # The pipeline must call the DDP-wrapped modules during training.
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder

    updates_per_epoch = (len(dataloader) + accumulation_steps - 1) // accumulation_steps
    if getattr(config.pretrain, 'use_lr_scheduler', True):
        warmup_steps = int(config.pretrain.warmup_epochs * updates_per_epoch)
        total_steps = max_epochs * updates_per_epoch

        if config.pretrain.lr_scheduler_type == "cosine":
            lr_scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
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
                num_training_steps=total_steps,
            )
    else:
        lr_scheduler = None
        print("学习率调度器: 禁用")

    # ------------------------------------------------------------------
    # 6. EMA 模型
    # ------------------------------------------------------------------
    ema_decay        = getattr(config.pretrain, 'ema_decay', 0.995)
    ema_start_epoch  = getattr(config.pretrain, 'ema_start_epoch', 20)
    ema_update_freq  = getattr(config.pretrain, 'ema_update_freq', 10)

    raw_action_unet = accelerator.unwrap_model(action_unet)
    raw_action_decoder = accelerator.unwrap_model(action_decoder)
    raw_obs_encoder = (
        accelerator.unwrap_model(obs_encoder) if obs_encoder is not None else None
    )

    ema_unet = EMAModel(
        raw_action_unet.parameters(),
        decay=ema_decay,
        use_ema_warmup=False,
    )
    if decoder_trains_in_pretrain:
        ema_decoder = EMAModel(
            raw_action_decoder.parameters(),
            decay=ema_decay,
            use_ema_warmup=False,
        )
    else:
        ema_decoder = None
    if obs_encoder is not None:
        ema_obs_encoder = EMAModel(
            raw_obs_encoder.parameters(),
            decay=ema_decay,
            use_ema_warmup=False,
        )
    else:
        ema_obs_encoder = None

    print(f"\nEMA 配置:")
    print(f"  Decay: {ema_decay}")
    print(f"  Start epoch: {ema_start_epoch}")
    print(f"  Update freq: {ema_update_freq}")

    # ------------------------------------------------------------------
    # 7. 梯度包装函数（与原版相同）
    # ------------------------------------------------------------------
    def generate_with_grad(pipeline, obs, **kwargs):
        """确保 pipeline_with_logprob 在梯度启用状态下运行"""
        original = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        result = pipeline_with_logprob(pipeline, obs=obs, **kwargs)
        torch.set_grad_enabled(original)
        return result

    # ------------------------------------------------------------------
    # 8. 训练循环
    # ------------------------------------------------------------------
    print(f"\n开始训练...")

    if auto_mode:
        print(f"[Auto模式] 每 {eval_interval} epochs 评估一次")
        print(f"[Auto模式] 每次评估 {num_test_episodes} 个episodes")
        print(f"[Auto模式] 成功阈值: {success_threshold} 次")
        print(f"[Auto模式] 最大训练轮数: {max_epochs}")
    else:
        print(f"[固定模式] 训练 {max_epochs} epochs")
    effective_global_batch = (
        batch_size_per_gpu * accelerator.num_processes * accumulation_steps
    )
    print(f"每卡 batch size: {batch_size_per_gpu}")
    print(f"梯度累积步数: {accumulation_steps}")
    print(f"有效全局 batch size: {effective_global_batch}")

    checkpoint_interval = int(getattr(config.pretrain, 'checkpoint_interval', 50))
    checkpoint_keep = int(getattr(config.pretrain, 'checkpoint_keep', 3))
    if checkpoint_interval < 1 or checkpoint_keep < 1:
        raise ValueError('checkpoint_interval 和 checkpoint_keep 必须至少为 1')
    run_dir_value = None
    if accelerator.is_main_process:
        run_dir_value = str(create_run_dir(
            config.pretrain.checkpoint_dir,
            epochs=max_epochs,
            episodes=training_episode_count,
            obs_mode=obs_mode,
            auto_mode=auto_mode,
        ))
    run_dir_value = broadcast_object_list([run_dir_value], from_process=0)[0]
    from pathlib import Path
    run_dir = Path(run_dir_value)
    print(f"检查点目录: {run_dir}（每 {checkpoint_interval} epochs 保存，保留最新 {checkpoint_keep} 个）")

    def ema_state_dict(model, ema):
        """Use EMA parameters without replacing the weights used by the next epoch."""
        model = accelerator.unwrap_model(model)
        state = model.state_dict()
        if ema is not None:
            for (name, _), shadow in zip(model.named_parameters(), ema.shadow_params):
                state[name] = shadow
        return state

    def save_checkpoint(epoch_number, loss, stopped=False):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            saved_path = broadcast_object_list([None], from_process=0)[0]
            accelerator.wait_for_everyone()
            return saved_path

        path = checkpoint_path(
            run_dir, epoch=epoch_number, episodes=training_episode_count,
            obs_mode=obs_mode, auto_mode=auto_mode,
        )
        unet_state = ema_state_dict(action_unet, ema_unet)
        ckpt = {
            'epoch':                      epoch_number - 1,
            'obs_mode':                   obs_mode,
            'pretrain_loss_type':         pretrain_loss_type,
            'decoder_trained':            decoder_trains_in_pretrain,
            'model_state_dict':           unet_state,
            'action_unet_state_dict':     unet_state,
            'action_decoder_state_dict':  ema_state_dict(action_decoder, ema_decoder),
            'optimizer_state_dict':       optimizer.state_dict(),
            'lr_scheduler_state_dict':    lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'ema_unet_state_dict':        ema_unet.state_dict(),
            'ema_decoder_state_dict':     ema_decoder.state_dict() if ema_decoder is not None else None,
            'loss':                       loss,
            'config':                     config.to_dict(),
            'dataset_paths':              dataset_paths,
            'training_episode_count':     training_episode_count,
            'obs_dim':                    obs_dim,
            'action_dim':                 action_dim,
            'auto_mode':                  auto_mode,
            'training_stopped':           stopped if auto_mode else False,
            'normalizer_state_dict':      normalizer.state_dict(),
            'world_size':                 accelerator.num_processes,
            'gpu_ids':                    configured_gpu_ids,
            'batch_size_per_gpu':         batch_size_per_gpu,
            'effective_global_batch':     effective_global_batch,
            'gradient_accumulation_steps': accumulation_steps,
            'mixed_precision':            mixed_precision,
        }
        if obs_encoder is not None:
            ckpt['obs_encoder_state_dict'] = ema_state_dict(obs_encoder, ema_obs_encoder)
            ckpt['ema_obs_encoder_state_dict'] = ema_obs_encoder.state_dict()
            if obs_mode == 'pointcloud':
                ckpt['pcd_encoder_config'] = {
                    'n_obs_steps': pcd_n_obs_steps,
                    'pcd_out_dim': pcd_enc_out_dim,
                    'agent_pos_dim': pcd_agent_pos_dim,
                    'state_mlp_out': pcd_state_mlp_out,
                    'use_layernorm': pcd_use_layernorm,
                    'final_norm': pcd_final_norm,
                }
            elif obs_mode == 'image':
                ckpt['img_encoder_config'] = {
                    'n_obs_steps': img_n_obs_steps,
                    'in_channels': img_in_channels,
                    'resnet_name': img_resnet_name,
                    'img_feat_dim': img_resnet_feat_dim,
                    'agent_pos_dim': img_agent_pos_dim,
                    'state_mlp_out': img_state_mlp_out,
                    'use_groupnorm': img_use_groupnorm,
                    'camera': img_camera,
                    'use_depth': img_use_depth,
                    'depth_max': img_depth_max,
                }

        # Only prune after the new file is fully written.
        temporary_path = path.with_suffix('.pth.tmp')
        try:
            torch.save(ckpt, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        prune_checkpoints(
            run_dir, episodes=training_episode_count,
            obs_mode=obs_mode, auto_mode=auto_mode, keep=checkpoint_keep,
        )
        if wandb_run is not None:
            wandb_run.summary['checkpoint_path'] = str(path)
            wandb_run.summary['final_loss'] = loss
        print(f"检查点已保存: {path}")
        saved_path = str(path)
        broadcast_object_list([saved_path], from_process=0)
        accelerator.wait_for_everyone()
        return saved_path

    action_unet.train()
    action_decoder.train()
    if obs_encoder is not None:
        obs_encoder.train()

    global_step     = 0
    training_stopped = False
    final_epoch     = 0
    avg_loss        = 0.0
    sync_models = [action_unet]
    if decoder_trains_in_pretrain:
        sync_models.append(action_decoder)
    if obs_encoder is not None:
        sync_models.append(obs_encoder)

    for epoch in range(max_epochs):
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{max_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(progress_bar):
            accumulation_context = accelerator.accumulate(*sync_models)
            accumulation_context.__enter__()
            # 所有模式统一归一化 action（limits → [-1, 1]）
            action_seq = normalizer['action'].normalize(batch['action'].to(device))
            target_actions = action_seq

            # ---------- 计算 obs_cond ----------
            if obs_mode == 'pointcloud':
                # replay H5 → 前 To 帧过滤后的点云，含 valid_count。
                pcd      = batch['point_cloud'].to(device, non_blocking=True)   # (B, To, max_pts, 3)
                apos_raw = batch['agent_pos'].to(device, non_blocking=True)     # (B, To, A)
                apos     = normalizer['agent_pos'].normalize(apos_raw)          # 归一化 agent_pos

                pcd_obs  = pcd[:, :pcd_n_obs_steps]                          # (B, To, *, 3)
                apos_in  = apos[:, :pcd_n_obs_steps]                         # (B, To, A)

                if pcd_online_fps and 'valid_count' in batch:
                    # 在线 GPU FPS：每个 batch / 每次访问都得到不同的 FPS 子集
                    B_, To_, M_, _ = pcd_obs.shape
                    valid = batch['valid_count'].to(device, non_blocking=True).long()  # (B, To)
                    valid_obs = valid[:, :pcd_n_obs_steps]                              # (B, To)

                    pcd_flat   = pcd_obs.reshape(B_ * To_, M_, 3)
                    valid_flat = valid_obs.reshape(B_ * To_)

                    sampled = batched_fps_sample(pcd_flat, valid_flat, pcd_n_points)
                    pcd_in  = sampled.reshape(B_, To_, pcd_n_points, 3)
                else:
                    pcd_in = pcd_obs

                obs_cond = obs_encoder(pcd_in, apos_in)  # (B, obs_dim)
            elif obs_mode == 'image':
                # image: (B, To, C, H_img, W_img)，读取时归一到 [0, 1]
                img      = batch['image'].to(device, non_blocking=True)          # (B, To, C, H, W)
                apos_raw = batch['agent_pos'].to(device, non_blocking=True)      # (B, To, A)
                apos     = normalizer['agent_pos'].normalize(apos_raw)           # 归一化 agent_pos

                img_in  = img [:, :img_n_obs_steps]                              # (B, To, C, H, W)
                apos_in = apos[:, :img_n_obs_steps]                              # (B, To, A)

                obs_cond = obs_encoder(img_in, apos_in)  # (B, obs_dim)
            else:
                # state 模式：对 obs 做 limits 归一化后取第 0 帧作为 obs_cond
                obs_seq  = batch['obs'].to(device)                       # (B, H, obs_dim)
                obs_norm = normalizer['obs'].normalize(obs_seq)          # 归一化到 [-1, 1]
                obs_cond = obs_norm[:, 0, :]                             # (B, obs_dim)

            batch_size = obs_cond.shape[0]

            # ---------- 按 loss_type 派发：action（多步 rollout MSE）/ velocity（CFM）----------
            if pretrain_loss_type == "velocity":
                # Flow Matching 速度场监督（对齐 FlowPolicy：UNet 直出速度，不经过 ActionDecoder）
                #   x0 ~ N(0, I), t ~ U[0,1]
                #   xt = (1-t)·x0 + t·x1，    ut = x1 - x0
                #   loss = MSE(action_unet(xt, t_input, obs_cond), ut)
                # 说明：ConditionalUnet1D.final_conv 已经输出 (B, H, action_dim) 形状，
                # 即速度场本身。先前在此处再过一遍 ActionDecoder（含 Dropout+ReLU+全局 MLP）
                # 会破坏 UNet 的逐位置时间结构、并通过 Dropout 给输出注入约 ~0.1 量级
                # 的随机方差，导致 MSE loss 卡在 0.1 难以下降。
                x1 = target_actions.float()
                x0 = torch.randn_like(x1)
                t = torch.rand(batch_size, device=device, dtype=torch.float32)
                t_pad = t.view(batch_size, 1, 1)
                xt = (1.0 - t_pad) * x0 + t_pad * x1
                ut = x1 - x0

                # 与 RL/test 共用的 SinusoidalPosEmb 时间编码（_unet_timestep_input）
                t_input = _unet_timestep_input(t)
                vt = action_unet(
                    sample=xt,
                    timestep=t_input,
                    global_cond=obs_cond,
                )

                loss = torch.nn.functional.mse_loss(vt, ut)
            else:
                # ---------- 全流程去噪 → pred_actions（原有 action_loss）----------
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

                # MSE 损失（预测动作 vs 真实动作）
                loss = torch.nn.functional.mse_loss(pred_actions, target_actions)

            # ---------- 反向传播 ----------
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    trainable_params,
                    max_norm=float(getattr(config.pretrain, 'max_grad_norm', 1.0)),
                )

            optimizer.step()
            if (lr_scheduler is not None and accelerator.sync_gradients
                    and not accelerator.optimizer_step_was_skipped):
                lr_scheduler.step()
            optimizer.zero_grad()

            # ---------- EMA 更新 ----------
            update_happened = (
                accelerator.sync_gradients
                and not accelerator.optimizer_step_was_skipped
            )
            next_global_step = global_step + int(update_happened)
            if (update_happened and epoch >= ema_start_epoch
                    and next_global_step % max(1, ema_update_freq) == 0):
                ema_unet.step(raw_action_unet.parameters())
                if ema_decoder is not None:
                    ema_decoder.step(raw_action_decoder.parameters())
                if ema_obs_encoder is not None:
                    ema_obs_encoder.step(raw_obs_encoder.parameters())

            accumulation_context.__exit__(None, None, None)

            epoch_loss_sum += loss.detach().float().item() * batch_size
            epoch_sample_count += batch_size
            if update_happened:
                global_step = next_global_step

            current_lr = optimizer.param_groups[0]['lr']
            if wandb_run is not None and update_happened:
                wandb_run.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch + 1,
                }, step=global_step, commit=False)
            if accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'LR':   f'{current_lr:.2e}',
                    'Step': global_step,
                })

        loss_stats = torch.tensor(
            [epoch_loss_sum, epoch_sample_count], device=device, dtype=torch.float64
        )
        loss_stats = accelerator.reduce(loss_stats, reduction='sum')
        avg_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
        current_lr = optimizer.param_groups[0]['lr']

        ema_status = "Active" if epoch >= ema_start_epoch else f"Inactive (starts at epoch {ema_start_epoch})"
        print(f"Epoch {epoch+1}/{max_epochs} 完成:")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  当前学习率: {current_lr:.2e}")
        print(f"  EMA状态: {ema_status}")
        if wandb_run is not None:
            wandb_run.log({
                'train/epoch_loss': avg_loss,
                'train/learning_rate': current_lr,
            }, step=global_step, commit=not (
                auto_mode and (epoch + 1) % eval_interval == 0
            ))

        final_epoch = epoch + 1

        if final_epoch % checkpoint_interval == 0:
            checkpoint_path_saved = save_checkpoint(final_epoch, avg_loss)

        # Auto 模式：定期评估（state 与 pointcloud 模式均支持）
        if auto_mode and (epoch + 1) % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"[Auto模式评估] Epoch {epoch+1}: 开始评估...")
            print(f"{'='*60}")

            # Every rank swaps to EMA and later restores the exact training
            # weights. Only rank 0 runs ManiSkill, using unwrapped modules so
            # DDP does not start collectives while the other ranks wait.
            ema_unet.store(raw_action_unet.parameters())
            ema_unet.copy_to(raw_action_unet.parameters())
            if ema_decoder is not None:
                ema_decoder.store(raw_action_decoder.parameters())
                ema_decoder.copy_to(raw_action_decoder.parameters())
            if ema_obs_encoder is not None:
                ema_obs_encoder.store(raw_obs_encoder.parameters())
                ema_obs_encoder.copy_to(raw_obs_encoder.parameters())

            accelerator.wait_for_everyone()
            should_stop = False
            if accelerator.is_main_process:
                pipeline._action_unet = raw_action_unet
                pipeline.action_decoder = raw_action_decoder
                _eval_decoder = (
                    raw_action_decoder if decoder_trains_in_pretrain else None
                )
                success_count, total_count, avg_reward = evaluate_pretrain_model(
                    raw_action_unet, _eval_decoder, pipeline, config, device,
                    num_test_episodes, obs_encoder=raw_obs_encoder,
                    normalizer=normalizer,
                    obs_mode=obs_mode,
                )

                print(f"[Auto模式评估] 成功率: {success_count}/{total_count}")
                print(f"[Auto模式评估] 平均奖励: {avg_reward:.3f}")
                if wandb_run is not None:
                    wandb_run.log({
                        'eval/success_rate': success_count / total_count if total_count else 0.0,
                        'eval/average_reward': avg_reward,
                    }, step=global_step)
                should_stop = success_count >= success_threshold

            stop_tensor = torch.tensor(
                int(should_stop), device=device, dtype=torch.int32
            )
            stop_tensor = broadcast(stop_tensor, from_process=0)

            ema_unet.restore(raw_action_unet.parameters())
            if ema_decoder is not None:
                ema_decoder.restore(raw_action_decoder.parameters())
            if ema_obs_encoder is not None:
                ema_obs_encoder.restore(raw_obs_encoder.parameters())
            pipeline._action_unet = action_unet
            pipeline.action_decoder = action_decoder
            action_unet.train()
            if decoder_trains_in_pretrain:
                action_decoder.train()
            if obs_encoder is not None:
                obs_encoder.train()
            accelerator.wait_for_everyone()

            if bool(stop_tensor.item()):
                print(f"\n{'='*60}")
                print(f"[Auto模式] 达到成功阈值！停止训练于 epoch {epoch+1}")
                print(f"{'='*60}\n")
                training_stopped = True
                break
            else:
                print(f"[Auto模式] 未达到阈值，继续训练...\n")

    if auto_mode and not training_stopped:
        print(f"\n[Auto模式] 达到最大训练轮数 {max_epochs}，停止训练")

    # ------------------------------------------------------------------
    # 9. 保存最后一个非整周期 epoch（使用 EMA 权重）
    # ------------------------------------------------------------------
    print(f"\n训练完成，准备保存模型...")

    if final_epoch % checkpoint_interval != 0 or training_stopped:
        checkpoint_path_saved = save_checkpoint(final_epoch, avg_loss, stopped=training_stopped)

    print(f"\n预训练完成！")
    print(f"最终检查点: {checkpoint_path_saved}")
    if auto_mode:
        print(f"  训练模式: Auto (自动停止)")
        print(f"  实际训练轮数: {final_epoch} epochs")
        print(f"  停止原因: {'达到成功阈值' if training_stopped else '达到最大训练轮数'}")
    else:
        print(f"  训练模式: 固定 ({final_epoch} epochs)")
    print(f"  观测模式: {obs_mode}")
    print(f"  使用 episodes: {training_episode_count}")
    print(f"  包含 EMA 权重: ✓")
    print(f"  包含优化器状态: ✓")
    print(f"  包含学习率调度器: ✓")
    if obs_encoder is not None:
        if obs_mode == 'image':
            print(f"  包含图像编码器权重: ✓")
            print(f"  包含 img_encoder_config: ✓")
        else:
            print(f"  包含点云编码器权重: ✓")
        print(f"  包含归一化器: ✓")
    print(f"  最终损失: {avg_loss:.6f}")
    print(f"{'='*60}\n")

    accelerator.wait_for_everyone()
    if owns_wandb_run:
        wandb_run.finish()
    accelerator.end_training()
    return checkpoint_path_saved
