import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.save_freq = 50
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.save_dir = "checkpoints"

    ###### 环境配置 ######
    config.env_id = "PickCube-v1"
    config.max_episode_steps = 300
    config.action_horizon = 16  # 一次flow生成的动作步数
    
    config.env_kwargs = ml_collections.ConfigDict()
    config.env_kwargs.sim_backend = "gpu"  # GPU后端
    config.env_kwargs.control_mode = "pd_ee_delta_pos"
    config.env_kwargs.obs_mode = "state"
    config.env_kwargs.render_mode = "rgb_array"
    config.env_kwargs.reward_mode = "dense"

    ###### Episode-level GRPO训练配置 ######
    config.episode = episode = ml_collections.ConfigDict()
    episode.num_state_groups = 8        # 每个epoch的初始seed数量
    episode.samples_per_group = 8       # 每个seed生成的轨迹数
    episode.parallel_envs = 64          # 并行环境数 (8×8=64)
    episode.num_epochs = 2000             # 总训练epoch数
    
    ###### GRPO分组配置 ######
    config.stage = stage = ml_collections.ConfigDict()
    stage.advantage_scale = 1.0  # advantages缩放系数
    
    # GRPO分组采样配置
    stage.num_state_groups = 8        # GRPO分组：8组
    stage.samples_per_group = 8       # 每组8个样本
    stage.timestep_weight_start = 1.0   # timestep权重起始值
    stage.timestep_weight_end = 1.0     # timestep权重结束值

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 10
    sample.eval_num_steps = 10
    sample.kl_reward = 0 # 不使用

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 64
    train.use_8bit_adam = False
    train.learning_rate = 1e-5
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-5
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 10.0
    train.adv_clip_max = 5.0
    train.clip_range = 0.2
    train.timestep_fraction = 1.0
    train.ema = True
    train.ema_decay = 0.9999
    
    # 学习率warm-up配置
    train.use_warmup = True           # 是否启用warm-up
    train.warmup_epochs = 5           # warm-up的epoch数
    train.warmup_start_factor = 0.1   # warm-up起始学习率系数  

    ###### 预训练配置 ######
    config.pretrain = pretrain = ml_collections.ConfigDict()
    pretrain.dataset_path = "episodes"
    pretrain.num_episodes = 500
    pretrain.num_epochs = 10
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

