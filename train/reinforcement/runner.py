"""Reinforcement training and checkpoint initialization."""

from pathlib import Path

from train.common.runtime import (
    Accelerator,
    ActionDecoder,
    ConditionalUnet1D,
    EMAModuleWrapper,
    FlowMatchEulerDiscreteScheduler,
    ProjectConfiguration,
    RobotActionPipeline,
    datetime,
    gym,
    logger,
    np,
    os,
    set_seed,
    torch,
    wandb,
)

from train.reinforcement.advantage import compute_episode_level_advantages
from train.reinforcement.evaluation import eval_in_env
from train.reinforcement.evaluation import evaluate_model_inline
from train.reinforcement.sampler import sample_complete_trajectories
from train.common.checkpoint import reinforcement_run_dir, save_ckpt
from train.reinforcement.losses.diffusionnft import train_on_episode_samples_diffusionnft
from train.reinforcement.losses.stepnft import train_on_episode_samples_stepnft
from train.common.checkpoint import unwrap_model
from train.reinforcement.checkpoint_input import resolve_pretrain_checkpoint

def run_reinforcement(config, physical_gpu_id, pretrain_checkpoint=None):
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

    requested_checkpoint = pretrain_checkpoint
    pretrain_checkpoint = resolve_pretrain_checkpoint(config, requested_checkpoint)

    # Read modality and model metadata before creating the environment/models.
    # Image policies cannot infer their condition dimension from the raw RGB-D
    # observation dictionary, so the imitation checkpoint is the source of truth.
    pretrain_meta = {}
    if pretrain_checkpoint:
        pretrain_meta = torch.load(pretrain_checkpoint, map_location='cpu', weights_only=False)
    configured_obs_mode = getattr(config, 'policy_obs_mode', 'state')
    policy_obs_mode = pretrain_meta.get('obs_mode', configured_obs_mode)
    if policy_obs_mode != configured_obs_mode:
        raise ValueError(
            f"配置要求 policy_obs_mode={configured_obs_mode!r}，但 checkpoint 是 "
            f"obs_mode={policy_obs_mode!r}；请使用匹配的强化学习配置。"
        )
    if policy_obs_mode not in ('state', 'image'):
        raise ValueError(
            f"NFT 强化学习暂不支持观测模式 {policy_obs_mode!r}；支持 'state' 和 'image'。"
        )
    if policy_obs_mode == 'image' and config.env_kwargs.obs_mode != 'rgbd':
        raise ValueError("图像策略要求 config.env_kwargs.obs_mode='rgbd'")

    # 每次 RL 训练使用独立目录：<预训练权重名>_<NFT算法名>_<月日>。
    # 例如：pretrain_epoch100_episodes500_state_StepNFT_0920
    if pretrain_checkpoint is not None:
        config.save_dir = reinforcement_run_dir(
            config.save_dir,
            pretrain_checkpoint,
            config.train.loss_type,
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
    if pretrain_checkpoint is not None:
        logger.info(f"强化学习 checkpoint 保存目录: {config.save_dir}")
    if requested_checkpoint and pretrain_checkpoint != Path(requested_checkpoint).expanduser().resolve():
        logger.info(f"预训练 checkpoint 已从旧路径解析到: {pretrain_checkpoint}")
    elif pretrain_checkpoint:
        logger.info(f"预训练 checkpoint: {pretrain_checkpoint}")

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
    elif isinstance(test_obs, dict):
        logger.info(f"  ✓ 环境返回结构化观测（policy_obs_mode={policy_obs_mode}）")
    else:
        logger.warning(f"  ⚠ 未识别的环境观测类型: {type(test_obs).__name__}")
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

    # 从环境获取动作维度；图像策略的 obs_dim 由视觉编码器决定。
    obs, _ = env.reset(seed=config.seed)

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

    # 预训练侧契约：
    #   decoder_trained=True  → action_loss 路径，decoder 必须参与 RL
    #   decoder_trained=False → velocity_loss 路径，decoder 完全旁路
    # 旧 checkpoint 无此字段 → 默认 True，保持向后兼容
    decoder_active = bool(pretrain_meta.get('decoder_trained', True))
    pretrain_loss_type = pretrain_meta.get('pretrain_loss_type', None)
    has_normalizer = 'normalizer_state_dict' in pretrain_meta

    linear_normalizer = None
    if has_normalizer:
        from models.normalizer import LinearNormalizer
        linear_normalizer = LinearNormalizer()
        linear_normalizer.load_state_dict(pretrain_meta['normalizer_state_dict'])
        linear_normalizer.to(accelerator.device).eval()

    obs_encoder = None
    img_encoder_config = None
    if policy_obs_mode == 'image':
        if 'obs_encoder_state_dict' not in pretrain_meta:
            raise ValueError('图像预训练 checkpoint 缺少 obs_encoder_state_dict')
        from models.image_encoder import ImageObsEncoder

        fallback = getattr(config, 'observation', {})
        img_encoder_config = dict(pretrain_meta.get('img_encoder_config', {}))
        defaults = {
            'n_obs_steps': 1,
            'in_channels': 4,
            'resnet_name': 'resnet18',
            'img_feat_dim': 512,
            'agent_pos_dim': 19,
            'state_mlp_out': 64,
            'use_groupnorm': True,
            'camera': 'base_camera',
            'use_depth': True,
            'depth_max': 5.0,
        }
        for key, default in defaults.items():
            img_encoder_config.setdefault(key, getattr(fallback, key, default))
        obs_encoder = ImageObsEncoder(
            n_obs_steps=int(img_encoder_config['n_obs_steps']),
            in_channels=int(img_encoder_config['in_channels']),
            resnet_name=img_encoder_config['resnet_name'],
            img_feat_dim=int(img_encoder_config['img_feat_dim']),
            agent_pos_dim=int(img_encoder_config['agent_pos_dim']),
            state_mlp_hidden=[64, int(img_encoder_config['state_mlp_out'])],
            use_groupnorm=bool(img_encoder_config['use_groupnorm']),
        ).to(accelerator.device)
        obs_encoder.load_state_dict(pretrain_meta['obs_encoder_state_dict'])
        obs_encoder.requires_grad_(False).eval()
        obs_dim = obs_encoder.output_dim
        checkpoint_obs_dim = pretrain_meta.get('obs_dim')
        if checkpoint_obs_dim is not None and int(checkpoint_obs_dim) != obs_dim:
            raise ValueError(
                f"图像编码器输出维度 {obs_dim} 与 checkpoint obs_dim "
                f"{checkpoint_obs_dim} 不一致"
            )
    else:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)
        obs_dim = int(obs.shape[-1])

    logger.info(f"环境与策略信息:")
    logger.info(f"  策略观测模式: {policy_obs_mode}")
    logger.info(f"  环境观测模式: {config.env_kwargs.obs_mode}")
    logger.info(f"  策略条件维度: {obs_dim}")
    logger.info(f"  动作维度: {action_dim}")
    logger.info(f"  动作序列长度: {config.action_horizon}")

    logger.info(f"预训练 checkpoint 元数据扫描:")
    logger.info(f"  路径: {pretrain_checkpoint if pretrain_checkpoint else '<未指定>'}")
    logger.info(f"  decoder_trained: {decoder_active}")
    logger.info(f"  pretrain_loss_type: {pretrain_loss_type!r}")
    logger.info(f"  含 normalizer_state_dict: {has_normalizer}")
    logger.info(f"  含 obs_encoder: {obs_encoder is not None}")

    # 创建模型
    logger.info("初始化模型...")
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(accelerator.device)

    # 创建 ActionDecoder（仅当 decoder_active 时；velocity 预训练直接旁路）
    if decoder_active:
        action_decoder = ActionDecoder(
            action_dim=action_dim,
            action_horizon=config.action_horizon,
            hidden_dim=256,
            num_layers=3
        ).to(accelerator.device)
    else:
        action_decoder = None
        logger.info("  ActionDecoder: 不创建（velocity_loss 模式 UNet 直出速度）")

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
    pipeline.action_decoder = action_decoder  # None 时 pipeline 也旁路 decoder
    # ── 挂载 LinearNormalizer：所有下游函数通过 pipeline.linear_normalizer 取 ──
    pipeline.linear_normalizer = linear_normalizer
    pipeline.decoder_active = decoder_active

    from train.reinforcement.observation import (
        ImageObservationAdapter,
        StateObservationAdapter,
    )
    if policy_obs_mode == 'image':
        checkpoint_config = pretrain_meta.get('config', {})
        checkpoint_pretrain = checkpoint_config.get('pretrain', {})
        image_size = checkpoint_pretrain.get(
            'img_size', getattr(config.observation, 'image_size', None)
        )
        observation_adapter = ImageObservationAdapter(
            encoder=obs_encoder,
            device=accelerator.device,
            encoder_config=img_encoder_config,
            normalizer=linear_normalizer,
            image_size=image_size,
        )
    else:
        observation_adapter = StateObservationAdapter(
            accelerator.device, normalizer=linear_normalizer
        )
    pipeline.observation_adapter = observation_adapter

    logger.info(f"模型初始化完成:")
    logger.info(f"  UNet参数数量: {sum(p.numel() for p in action_unet.parameters())}")
    if action_decoder is not None:
        logger.info(f"  ActionDecoder参数数量: {sum(p.numel() for p in action_decoder.parameters())}")
    else:
        logger.info(f"  ActionDecoder参数数量: 0（已旁路）")

    # 获取可训练参数（velocity 模式不训练 decoder）
    trainable_parameters = list(action_unet.parameters())
    if action_decoder is not None:
        trainable_parameters += list(action_decoder.parameters())

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

    # 准备模型和优化器（decoder 不存在时跳过）
    if action_decoder is not None:
        action_unet, action_decoder, optimizer = accelerator.prepare(action_unet, action_decoder, optimizer)
        pipeline.action_decoder = action_decoder  # 重新挂载（accelerator.prepare 可能包装过）
    else:
        action_unet, optimizer = accelerator.prepare(action_unet, optimizer)
    pipeline._action_unet = action_unet

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

    # === 加载预训练检查点（元数据已在前面扫描；这里加载权重 + normalizer）===
    if pretrain_checkpoint:
        logger.info(f"加载预训练检查点: {pretrain_checkpoint}")
        # 从 CPU 载荷复制权重，避免重复读取文件及额外占用 GPU 显存。
        checkpoint = pretrain_meta

        # 1) 加载 UNet
        unwrap_model(action_unet, accelerator).load_state_dict(checkpoint['action_unet_state_dict'])
        logger.info("  ✓ UNet 权重加载成功")

        # 2) 加载 ActionDecoder（仅当 decoder_active 时）
        if decoder_active:
            if 'action_decoder_state_dict' in checkpoint:
                unwrap_model(action_decoder, accelerator).load_state_dict(
                    checkpoint['action_decoder_state_dict']
                )
                logger.info("  ✓ ActionDecoder 权重加载成功")
            else:
                logger.warning(
                    "  ⚠ checkpoint 缺少 action_decoder_state_dict，但 decoder_trained=True，"
                    "ActionDecoder 将保持随机初始化"
                )
        else:
            logger.info("  ✓ ActionDecoder 已旁路（velocity_loss 模式，不加载权重）")

        # 3) LinearNormalizer 已在构建观测 adapter 前加载。
        if linear_normalizer is not None:
            logger.info("  ✓ LinearNormalizer 加载成功并挂载到 pipeline.linear_normalizer")
        else:
            logger.info("  ⚠ checkpoint 无 normalizer_state_dict（旧版兼容；RL 阶段将跳过归一化）")

        logger.info("✓ 预训练权重加载完成")
        del checkpoint, pretrain_meta
    else:
        logger.info("未指定预训练 checkpoint；将从 resume_from 恢复训练状态")

    # ✅ 修复：EMA 在加载预训练之前用随机初始化权重创建过快照，需要在 pretrain 加载后
    #    重新对 trainable_parameters 做一次 EMA 快照，否则前几百个 step 的 EMA 仍然
    #    混着随机初始化的权重，导致中早期 checkpoint 不可用。
    if config.train.ema:
        ema.ema_parameters = [
            p.clone().detach().to(accelerator.device) for p in trainable_parameters
        ]
        logger.info("✓ EMA 已重新对齐到加载预训练后的权重")

    # ========== DiffusionNFT: 初始化三模型架构（在加载预训练权重之后） ==========
    import copy
    logger.info("初始化三模型架构（current/old/ref）...")

    # Old model（用于policy gradient计算）
    action_unet_old = copy.deepcopy(unwrap_model(action_unet, accelerator))
    action_unet_old.to(accelerator.device).eval()
    if decoder_active:
        action_decoder_old = copy.deepcopy(unwrap_model(action_decoder, accelerator))
        action_decoder_old.to(accelerator.device).eval()
    else:
        action_decoder_old = None

    # Reference model（用于KL regularization）
    action_unet_ref = copy.deepcopy(unwrap_model(action_unet, accelerator))
    action_unet_ref.to(accelerator.device)
    for param in action_unet_ref.parameters():
        param.requires_grad = False
    action_unet_ref.eval()

    if decoder_active:
        action_decoder_ref = copy.deepcopy(unwrap_model(action_decoder, accelerator))
        action_decoder_ref.to(accelerator.device)
        for param in action_decoder_ref.parameters():
            param.requires_grad = False
        action_decoder_ref.eval()
    else:
        action_decoder_ref = None

    logger.info(
        f"✓ 三模型架构初始化完成（decoder_active={decoder_active}，"
        f"{'三模型均含 decoder' if decoder_active else 'decoder 全链旁路'}）"
    )

    # autocast 设置
    autocast = accelerator.autocast

    logger.info("***** 开始Episode-level GRPO训练 *****")
    logger.info(f"  最大episode步数 = {config.max_episode_steps}")
    logger.info(f"  动作序列长度 = {config.action_horizon}")
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
                action_unet=action_unet,
                action_decoder=action_decoder,
                num_state_groups=config.episode.num_state_groups,
                samples_per_group=config.episode.samples_per_group,
                config=config,
                accelerator=accelerator,
                autocast=autocast,
                seed_offset=config.seed + epoch * 100000 + rollout_idx * 10000,
                observation_adapter=observation_adapter,
                epoch=epoch  # 传入epoch用于噪声退火
            )

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

        # ===== 记录采样和Advantage指标到wandb =====
        if accelerator.is_main_process:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    # 采样质量指标
                    "sample/success_rate": total_success_count / len(all_samples) * 100,
                    "sample/success_count": total_success_count,
                    "sample/total_trajectories": len(all_samples),
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
            logger.info(f"  num_state_groups = {config.episode.num_state_groups}")
            logger.info(f"  samples_per_group = {config.episode.samples_per_group}")
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
        # 根据 config.train.loss_type 派发到不同训练函数
        loss_type = getattr(config.train, 'loss_type', 'rollout')
        if loss_type == 'stepnft':
            train_fn = train_on_episode_samples_stepnft
            logger.info(f"  Loss type: stepnft（单步 forward + DPO/weighted NLL）")
        elif loss_type == 'rollout':
            train_fn = train_on_episode_samples_diffusionnft
            logger.info(f"  Loss type: rollout（多步 rollout MSE，DiffusionNFT）")
        else:
            raise ValueError(
                f"Unknown config.train.loss_type: {loss_type!r}. Expected 'rollout' or 'stepnft'."
            )

        for update_idx in range(update_epoch):
            logger.info(f"\n  --- 更新轮次 {update_idx+1}/{update_epoch} ---")

            global_step, avg_epoch_loss = train_fn(
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
                epoch=epoch
            )

        logger.info(f"\n  ===== 训练完成 =====")

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
                      accelerator, ema, trainable_parameters, config,
                      linear_normalizer=getattr(pipeline, 'linear_normalizer', None),
                      decoder_trained=decoder_active,
                      pretrain_loss_type=pretrain_loss_type,
                      obs_encoder=obs_encoder,
                      obs_mode=policy_obs_mode,
                      obs_encoder_config=img_encoder_config)

        # 5b. 定期内联评估（仅 main process，每 eval.interval 个 epoch 执行一次）
        eval_cfg = getattr(config, 'eval', None)
        eval_enabled = getattr(eval_cfg, 'enabled', True) if eval_cfg is not None else True
        eval_interval = getattr(eval_cfg, 'interval', 5) if eval_cfg is not None else 5
        eval_num_eps = getattr(eval_cfg, 'num_episodes', 20) if eval_cfg is not None else 20
        eval_seed = getattr(eval_cfg, 'seed', 20000) if eval_cfg is not None else 20000

        if eval_enabled and accelerator.is_main_process and (epoch + 1) % eval_interval == 0:
            eval_results = evaluate_model_inline(
                action_unet=action_unet,
                action_decoder=action_decoder,
                pipeline=pipeline,
                config=config,
                device=accelerator.device,
                accelerator=accelerator,
                ema=ema if config.train.ema else None,
                trainable_parameters=trainable_parameters if config.train.ema else None,
                num_episodes=eval_num_eps,
                seed=eval_seed,
                epoch=epoch + 1,
            )
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "test/success_rate": eval_results['success_rate'],
                    "test/num_success": eval_results['num_success'],
                    "test/num_episodes": eval_results['num_episodes'],
                    "test/mean_reward": eval_results['mean_reward'],
                    "test/std_reward": eval_results['std_reward'],
                    "test/epoch": epoch + 1,
                }, step=global_step)
                logger.info(
                    f"[Inline Eval → wandb] test/success_rate={eval_results['success_rate']:.1f}%"
                )

        # 6. 更新学习率调度器（epoch结束时，ReinFlow风格）
        if lr_scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']

            # 显示学习率变化（前100个epoch每次都显示，之后每10个epoch显示一次）
            if epoch < 100 or (epoch + 1) % 10 == 0:
                logger.info(f"  [LR调度] Epoch {epoch+1}: 学习率 {old_lr:.2e} → {new_lr:.2e}")

        # ========== 7. 更新old model（已移至train_on_episode_samples_diffusionnft中，每20个global_step更新） ==========
        # ✅ 方案2：按 global_step 更新 old model，而不是按 epoch
        # 更新逻辑已移至 train_on_episode_samples_diffusionnft 函数内部
        # 每20个 global_step 更新一次 old model
        """
        with torch.no_grad():
            decay = return_decay(global_step, config.train.decay_type)

            # 更新UNet
            for src_param, tgt_param in zip(
                unwrap_model(action_unet, accelerator).parameters(),
                action_unet_old.parameters()
            ):
                tgt_param.data.copy_(
                    tgt_param.data * decay + src_param.data * (1.0 - decay)
                )

            # 更新Decoder
            for src_param, tgt_param in zip(
                unwrap_model(action_decoder, accelerator).parameters(),
                action_decoder_old.parameters()
            ):
                tgt_param.data.copy_(
                    tgt_param.data * decay + src_param.data * (1.0 - decay)
                )

        if (epoch + 1) % 10 == 0:
            logger.info(f"  [Old model EMA] decay={decay:.4f}")
        """

    logger.info("训练完成！")

    # 最终评估（使用评估环境）
    eval_in_env(
        pipeline, action_unet, action_decoder, eval_env, config, accelerator,
        global_step, num_test_episodes=10,
        observation_adapter=observation_adapter,
    )

    # 保存最终模型（使用最终epoch数）
    final_epoch = config.episode.num_epochs
    save_ckpt(config.save_dir, action_unet, action_decoder, final_epoch,
              accelerator, ema, trainable_parameters, config,
              linear_normalizer=getattr(pipeline, 'linear_normalizer', None),
              decoder_trained=decoder_active,
              pretrain_loss_type=pretrain_loss_type,
              obs_encoder=obs_encoder,
              obs_mode=policy_obs_mode,
              obs_encoder_config=img_encoder_config)

    # 关闭环境
    env.close()
    eval_env.close()

    # 关闭 wandb
    accelerator.end_training()
