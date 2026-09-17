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
    config.action_horizon = 16  # 一次flow生成的动作步数（从16增加到32，减少flow调用次数）
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
    stage.num_rollouts_per_state = 64  # 已弃用，使用num_state_groups和samples_per_group
    stage.num_exploration_trajs = 16  # 探索轨迹数（正好填满32个并行环境的一半）
    stage.inner_epochs_per_stage = 16  # 增加训练轮次（从10增加到16，更充分训练）
    stage.advantage_scale = 1.0  # advantages缩放系数
    
    # GRPO分组采样配置
    stage.num_state_groups = 256        # GRPO状态组数量
    stage.samples_per_group = 32        # 每个状态组的样本数（总样本=256×32=8192）
    stage.timestep_weight_start = 1.0   # timestep权重起始值（1.0表示相同权重）
    stage.timestep_weight_end = 1.0     # timestep权重结束值（1.0表示相同权重）
    
    # 高质量起点筛选配置（阶段1+）
    stage.use_quality_filter = True          # 是否启用高质量起点筛选
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
    train.learning_rate = 1e-4  # 保守微调：从5e-6降低到2e-6，更小的更新步长
    train.adam_beta1 = 0.9
    train.adam_beta2 = 0.999
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-8
    train.gradient_accumulation_steps = 1
    train.max_grad_norm = 0.5  # 保守微调：从1.0降低到0.5，更严格的梯度裁剪
    train.num_inner_epochs = 1  # 在阶段级别使用stage.inner_epochs_per_stage
    train.adv_clip_max = 2.0  # 保守微调：从5降低到2，限制极端advantage
    train.clip_range = 0.05  # 保守微调：从0.1降低到0.05，更保守的策略更新（±5%）
    train.timestep_fraction = 1.0
    train.ema = True
    train.ema_decay = 0.9999  

    ###### 预训练配置 ######
    config.pretrain = pretrain = ml_collections.ConfigDict()
    pretrain.dataset_path = "episodes"
    pretrain.num_episodes = 500  # 使用episode0-49
    pretrain.num_epochs = 7     # 固定模式的训练epoch数
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

