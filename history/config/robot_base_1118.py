import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.save_freq = 5  # 每5个阶段保存一次
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.save_dir = "checkpoints"

    ###### 环境配置 ######
    config.env_id = "PickCube-v1"
    config.max_episode_steps = 500
    config.action_horizon = 16
    config.num_envs = 16               # GPU并行环境数（探索和采样都用16个）
    config.use_gpu_sim = True          # 启用GPU模拟
    
    config.env_kwargs = ml_collections.ConfigDict()
    config.env_kwargs.sim_backend = "gpu"  # GPU后端
    config.env_kwargs.control_mode = "pd_ee_delta_pos"
    config.env_kwargs.obs_mode = "state"
    config.env_kwargs.render_mode = "rgb_array"
    config.env_kwargs.reward_mode = "dense"

    ###### 阶段训练配置 ######
    config.stage = stage = ml_collections.ConfigDict()
    stage.num_rollouts_per_state = 64  # 每个初始状态生成的训练轨迹数（16*4=64，正好2批）
    stage.num_exploration_trajs = 16  # 探索轨迹数（正好填满32个并行环境的一半）
    stage.inner_epochs_per_stage = 20  # 每个阶段训练轮数
    stage.advantage_scale = 1.0  # advantages缩放系数

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 10  # Flow去噪步数（从20减少到10，质量损失很小但速度提升2倍）
    sample.eval_num_steps = 10
    sample.kl_reward = 0.0

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 32
    train.use_8bit_adam = False
    train.learning_rate = 1e-4  # 提高学习率，加快策略更新
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0
    train.num_inner_epochs = 1  # 在阶段级别使用stage.inner_epochs_per_stage
    train.adv_clip_max = 5  # 增大advantage clip范围，允许更大的梯度
    train.clip_range = 0.2  # 增大clip_range，允许更大的策略更新
    train.timestep_fraction = 1.0
    train.ema = True
    train.ema_decay = 0.95  # 降低EMA decay，让新策略更快生效

    ###### 预训练配置 ######
    config.pretrain = pretrain = ml_collections.ConfigDict()
    pretrain.dataset_path = "episodes"
    pretrain.num_episodes = 500  # 使用episode0-49
    pretrain.num_epochs = 30     # 固定模式的训练epoch数
    pretrain.batch_size = 32
    pretrain.learning_rate = 1e-4
    pretrain.checkpoint_dir = "pretrain-checkpoints"
    pretrain.gradient_accumulation_steps = 1
    pretrain.max_grad_norm = 1.0
    
    # Auto模式配置
    pretrain.auto_mode = False              # 是否启用auto模式
    pretrain.auto_eval_interval = 6        # auto模式：每N个epoch评估一次
    pretrain.auto_num_test_episodes = 10    # auto模式：每次评估运行的episode数
    pretrain.auto_success_threshold = 1     # auto模式：至少N次成功才停止
    pretrain.auto_max_epochs = 200          # auto模式：最大训练epoch数

    return config

