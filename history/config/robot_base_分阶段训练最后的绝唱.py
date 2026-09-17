import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.save_freq = 1  # 每10个epoch保存一次
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.save_dir = "checkpoints"

    ###### 环境配置 ######
    config.env_id = "PickCube-v1"
    config.max_episode_steps = 200
    config.action_horizon = 16  # 一次flow生成的动作步数（从16增加到32，减少flow调用次数）
    config.num_envs = 16               # GPU并行环境数（探索和采样都用16个）
    config.use_gpu_sim = True          # 启用GPU模拟
    
    config.env_kwargs = ml_collections.ConfigDict()
    config.env_kwargs.sim_backend = "gpu"  # GPU后端
    config.env_kwargs.control_mode = "pd_ee_delta_pos"
    config.env_kwargs.obs_mode = "state"
    config.env_kwargs.render_mode = "rgb_array"
    config.env_kwargs.reward_mode = "dense"

    ###### 多阶段混合训练配置 ######
    config.multi_stage = multi_stage = ml_collections.ConfigDict()
    multi_stage.enabled = True  # 启用多阶段混合训练
    multi_stage.stages_per_epoch = 128  # 每个epoch采样的阶段数
    multi_stage.num_epochs = 50  # 总训练epoch数
    multi_stage.num_mini_epochs = 4  # 每个epoch分4次采样-训练
    multi_stage.parallel_envs = 1024  # 总并行环境数
    multi_stage.stages_per_mini_epoch = 32  # 每个mini-epoch的阶段数
    
    ###### 阶段训练配置 ######
    config.stage = stage = ml_collections.ConfigDict()
    stage.num_rollouts_per_state = 64  # 已弃用
    stage.num_exploration_trajs = 16  # 已弃用（多阶段混合训练中不再使用）
    stage.inner_epochs_per_stage = 10  # 已弃用（多阶段混合训练中不再使用）
    stage.advantage_scale = 1.0  # advantages缩放系数
    
    # GRPO分组采样配置（多阶段混合训练）
    stage.num_state_groups = 8        # 每个阶段的状态组数量
    stage.samples_per_group = 4        # 每个状态组的样本数
    stage.timestep_weight_start = 1.0   # timestep权重起始值（1.0表示相同权重）
    stage.timestep_weight_end = 1.0     # timestep权重结束值（1.0表示相同权重）
    
    # 奖励填充策略（固定填充方案）
    # 一旦轨迹在16步中成功，后续步骤自动填充成功时的奖励值
    # 效果：第3步成功(r=10)填充13次 → 总reward=140; 第15步成功填充1次 → 总reward=23
    # 优势：自动奖励早成功的轨迹，解决累积步数不同导致的advantage计算问题
    
    # 高质量起点筛选配置（阶段1+）
    stage.use_quality_filter = False          # 禁用高质量起点筛选（改用随机种子）
    stage.num_candidate_seeds = 4096         # 候选种子数（阶段1+）
    stage.num_selected_seeds = 256           # 筛选后的种子数
    stage.quality_filter_batch_size = 256   # 筛选时的批次大小（避免显存不足）

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    sample.num_steps = 10  # Flow去噪步数（从20减少到10，质量损失很小但速度提升2倍）
    sample.eval_num_steps = 10
    sample.kl_reward = 0.01  # 从0.0改为0.01，添加KL约束防止策略偏离过远

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 32
    train.use_8bit_adam = False
    train.learning_rate = 2e-6  # 保守微调：从5e-6降低到2e-6，更小的更新步长
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 1.0  # 修复：放宽梯度裁剪，原来的0.5太严格
    train.num_inner_epochs = 1  # 在阶段级别使用stage.inner_epochs_per_stage
    train.adv_clip_max = 5.0  # 修复：放宽advantage裁剪，原来的2.0太小
    train.clip_range = 0.1  # 修复：使用标准PPO clip范围，原来的0.05太保守导致训练不稳定
    train.timestep_fraction = 1.0
    train.ema = True
    train.ema_decay = 0.9999  

    ###### 预训练配置 ######
    config.pretrain = pretrain = ml_collections.ConfigDict()
    pretrain.dataset_path = "episodes"
    pretrain.num_episodes = 500  # 使用episode0-49
    pretrain.num_epochs = 12     # 固定模式的训练epoch数
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

