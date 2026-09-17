import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    config.run_name = ""
    config.seed = 42
    config.logdir = "logs"
    config.save_freq = 5
    config.num_checkpoint_limit = 5
    config.mixed_precision = "fp16"
    config.allow_tf32 = True
    config.resume_from = ""
    config.save_dir = "checkpoints/reinforcement/FlowNFT"

    ###### FlowPolicy 预训练模型配置 ######
    # 预训练 FlowPolicy 检查点路径（PickCube 点云任务，700 epoch）
    # config.flowpolicy_ckpt = (
    #     "checkpoints/flowpolicy/"
    #     "pickcube_pointcloud-pickcube_pointcloud_flowpolicy-0312_seed0/"
    #     "epoch=0700.ckpt"
    # )

    # # 点云观测配置（与预训练保持一致）
    # config.n_obs_steps = 2       # 观测历史帧数（FlowPolicy: n_obs_steps=2）
    # config.n_action_steps = 8    # 每次 predict_action 返回的执行步数（FlowPolicy: n_action_steps=8）
    # config.n_points = 512        # 点云采样点数（FlowPolicy: 512 点）

    ###### 环境配置 ######
    config.env_id = "PickCube-v1"
    config.max_episode_steps = 300
    config.action_horizon = 16   # Flow 生成的完整动作序列长度（FlowPolicy: horizon=16）

    config.env_kwargs = ml_collections.ConfigDict()
    config.env_kwargs.sim_backend = "gpu"   # GPU后端
    config.env_kwargs.control_mode = "pd_ee_delta_pos"
    config.env_kwargs.obs_mode = "state"   # episodes 以 state obs 收集（42维扁平状态）
    config.env_kwargs.render_mode = "rgb_array"
    config.env_kwargs.reward_mode = "dense"
    config.env_kwargs.sim_freq = 500
    config.env_kwargs.control_freq = 5

    ###### Episode-level GRPO训练配置 ######
    config.episode = episode = ml_collections.ConfigDict()
    episode.num_state_groups = 256       # 每个rollout的初始状态数量
    episode.samples_per_group = 1      # 每个状态生成的轨迹数
    # parallel_envs 由 num_state_groups × samples_per_group 自动计算，无需手动维护
    episode.parallel_envs = episode.num_state_groups * episode.samples_per_group
    episode.rollout_epoch = 1           # 每个训练epoch采样多少次（对齐RLinf）
    episode.update_epoch = 1            # 每个训练epoch更新多少轮（对齐RLinf）
    episode.num_epochs = 100           # 总训练epoch数

    ###### GRPO分组配置（advantage 超参数）######
    # num_state_groups / samples_per_group 统一以 config.episode 为准，此处不再重复定义
    config.stage = stage = ml_collections.ConfigDict()
    stage.advantage_scale = 1.0        # advantage 缩放系数
    stage.timestep_weight_start = 1.0  # timestep 权重起始值
    stage.timestep_weight_end = 1.0    # timestep 权重结束值
    # advantage 计算方式：
    #   "group"  - 组内均值/方差归一化（每组多条轨迹，组内相对比较）
    #   "single" - 二值 advantage（每组单条轨迹，成功=+1，失败=-1）
    stage.advantage_type = "single"
    # 任务最大奖励（用于 single 模式判定成功）
    # PickCube: 5，后续根据具体任务修改
    stage.max_reward = 5.0

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
    train.adam_beta2 = 0.95
    train.adam_weight_decay = 1e-4
    train.adam_epsilon = 1e-5
    train.gradient_accumulation_steps = 4
    train.max_grad_norm = 1.0
    train.adv_clip_max = 3.0
    train.clip_range = 0.2
    train.timestep_fraction = 1.0
    train.ema = True
    train.ema_decay = 0.9999

    # Rollout (DiffusionNFT) loss 专用超参（loss_type="rollout" 时生效）
    # 这套字段对应 train_on_episode_samples_diffusionnft：
    #   positive_prediction = β·current + (1-β)·old
    #   implicit_negative   = (1+β)·old - β·current
    # 与 StepNFT 路径的 stepnft_beta / stepnft_kl_beta 同义但用于不同 loss 路径
    train.beta = 0.5                       # current/old 预测混合系数（β）
    train.kl_weight = 1e-4                 # KL 项权重（current vs ref 的 MSE）

    # Old model EMA decay 类型（return_decay 函数）：
    #   0: 始终为0（不使用 old model 同步）
    #   1: 线性增长到 0.5
    #   2: 前75步为0，之后增长到0.999
    train.decay_type = 1
    
    # 学习率调度器配置（参照ReinFlow）
    train.use_lr_scheduler = False           # 启用学习率调度
    train.lr_scheduler_type = "cosine"      # cosine annealing
    train.warmup_epochs = 20               # warm-up 100个epochs（约5%）
    train.min_lr = 5e-6                     # 最小学习率（提高以避免训练停滞）
    train.num_cycles = 0.5                  # cosine cycles（0.5=半个周期，更平滑）  
    
    # Replay Buffer配置
    train.use_replay_buffer = False          # 启用经验回放buffer
    train.replay_buffer_size = 10000        # Buffer最大容量（存储完整sample数量）
    train.target_positive_ratio = 0.5       # 目标正样本比例（50%）
    train.balance_threshold = 0.05           # 平衡阈值（偏离10%即触发平衡，范围40%-60%）
    train.buffer_sample_augment_ratio = 0.2 # 补充样本占当前样本的最大比例（30%）
    train.sliding_window_stride = 1         # 滑动窗口步长（1=密集采样，>1=稀疏采样）

    # StepNFT 训练配置（loss_type="stepnft" 时生效）
    # 参考 pi-StepNFT/rlinf/algorithms/losses.py:compute_nft_actor_loss
    train.loss_type = "stepnft"            # "rollout"（多步 rollout MSE，现有）| "stepnft"（单步 NFT loss）
    train.stepnft_noise_level = 0.3        # SDE 噪声系数（>0 才能让 StepNFT loss 不退化为常数）
    train.stepnft_beta = 1.0               # v_pos = v_old + β·clip(Δv)
    train.stepnft_max_drift = 0.5          # delta_v 的 norm clip 阈值（防止单步漂移过大）
    train.stepnft_loss_form = "dpo"   # "weighted"（r·E_pos+(1-r)·E_neg）| "dpo"（softplus(logit)）
    train.stepnft_dpo_beta = 1.0           # DPO 形式 logit 缩放系数
    train.stepnft_kl_beta = 1e-4           # KL 项权重（v_theta 与 v_ref 的 MSE）
    train.stepnft_std_epsilon = 1e-4       # var 兜底，避免除零

    ###### 预训练配置 ######
    config.pretrain = pretrain = ml_collections.ConfigDict()
    # ManiSkill replay H5；当前 state 数据保留为已拆分的 episode 目录。
    # 各 imitation 配置可直接将 dataset_path 指向相应 obs_mode 的 replay H5。
    pretrain.dataset_path = "data/datasets/state/episodes"
    pretrain.num_episodes = 50
    pretrain.num_epochs = 200
    pretrain.batch_size = 64             
    pretrain.learning_rate = 1e-4
    pretrain.checkpoint_dir = "checkpoints/imitation"
    pretrain.gradient_accumulation_steps = 1
    pretrain.max_grad_norm = 1.0
    # pretrain.weight_decay = 1e-6

    # 预训练 loss 类型：
    #   "action"   - 原有方式：generate_with_grad 多步去噪 → MSE(pred_actions, target_actions)
    #   "velocity" - Flow Matching 速度场监督：MSE(v_theta(xt, t, obs), x1 - x0)
    # 命令行 --action_loss / --velocity_loss 可覆盖此默认值
    pretrain.loss_type = "action"

    # 优化器参数（对齐 FlowPolicy: betas=[0.95, 0.999], eps=1e-8）
    pretrain.adam_beta1 = 0.95
    pretrain.adam_beta2 = 0.999
    pretrain.adam_eps = 1e-8

    # 学习率调度器配置（暂时禁用）
    pretrain.use_lr_scheduler = False
    pretrain.lr_scheduler_type = "cosine"  # cosine annealing
    pretrain.warmup_epochs = 50            # warm-up的epoch数
    pretrain.min_lr = 1e-5                 # 最小学习率

    # EMA配置（对齐 FlowPolicy：power-based warmup decay）
    pretrain.ema_use_warmup = False         # 启用动态 decay warmup
    pretrain.ema_decay = 0.9999            # 最大 decay 上限
    pretrain.ema_min_decay = 0.0           # warmup 起始 decay（从0逐步升至0.9999）
    pretrain.ema_update_after_step = 0     # 从第0步开始更新
    pretrain.ema_inv_gamma = 1.0           # FlowPolicy: inv_gamma=1.0
    pretrain.ema_power = 0.75             # FlowPolicy: power=0.75（约10k步达到0.999）
    
    # # Consistency Flow Matching 超参数（来自 FlowPolicy）
    # pretrain.cfm_eps = 1e-2            # t 的最小值，避免 t=0
    # pretrain.cfm_num_segments = 2      # 轨迹分段数（>=1；=1 退化为单段标准 CFM）
    # pretrain.cfm_boundary = 1          # 一致性约束阈值（整数 0 表示退化模式）
    # pretrain.cfm_delta = 0.01          # t 到 r 的间隔
    # pretrain.cfm_alpha = 1e-5          # 速度一致性 loss 的权重

    # 观测模式：决定预训练使用哪种输入
    #   "state"      - 扁平化状态向量（当前默认，与现有逻辑完全兼容）
    #   "pointcloud" - PointNet 编码点云 + agent_pos 向量拼接
    #   "image"      - ResNet18 编码 RGB(D) 图像 + agent_pos 向量拼接
    pretrain.obs_mode = "state"

    # 点云编码器配置（仅 obs_mode="pointcloud" 时生效）
    pretrain.pcd_n_obs_steps = 2            # 历史帧数（取最新 T 帧作为条件）
    pretrain.pcd_n_points = 512             # 每帧 FPS 后的点数（训练/推理均用此值）
    pretrain.pcd_encoder_output_dim = 64    # PointNetEncoderXYZ 输出维度（与 FlowPolicy 对齐，obs_cond = (64+64)*2 = 256）
    pretrain.pcd_agent_pos_dim = 19         # agent_pos 向量维度
    pretrain.pcd_state_mlp_out = 64         # state MLP 输出维度
    pretrain.pcd_use_layernorm = True       # PointNet 中间层是否用 LayerNorm
    pretrain.pcd_final_norm = "layernorm"   # PointNet 最终 projection 后的 norm

    # 点云按需从 replay H5 过滤并补齐至 max_pts，训练 batch 中在线 FPS。
    pretrain.pcd_max_pts = 4096

    # 图像编码器配置（仅 obs_mode="image" 时生效）
    pretrain.img_n_obs_steps     = 2            # 历史帧数（取最新 T 帧作为条件）
    pretrain.img_camera          = "base_camera"  # replay H5 的 obs/sensor_data 相机名
    pretrain.img_use_depth       = True         # True=RGBD(4ch), False=RGB(3ch)
    pretrain.img_depth_max       = 5.0          # depth 归一化最大值（米），>该值 clip 到 1.0
    pretrain.img_resnet_name     = "resnet18"   # 主干网络（resnet18 / resnet34 / resnet50）
    pretrain.img_resnet_feat_dim = 512          # 主干 fc 替换后的输出维度（resnet18=512）
    pretrain.img_agent_pos_dim   = 19           # agent_pos 向量维度（与点云模式保持一致）
    pretrain.img_state_mlp_out   = 64           # state MLP 输出维度
    pretrain.img_use_groupnorm   = True         # 将 ResNet 的 BatchNorm 全部替换为 GroupNorm
    pretrain.img_size            = (128, 128)   # 训练/推理图像尺寸（H, W）

    # Auto模式配置
    pretrain.auto_mode = False              # 是否启用auto模式
    pretrain.auto_eval_interval = 10       # auto模式：每N个epoch评估一次
    pretrain.auto_num_test_episodes = 20    # auto模式：每次评估运行的episode数
    pretrain.auto_success_threshold = 15    # auto模式：至少N次成功才停止（15/20=75%）
    pretrain.auto_max_epochs = 1000         # auto模式：最大训练epoch数

    ###### RL 训练中内联测试配置 ######
    config.eval = eval_cfg = ml_collections.ConfigDict()
    eval_cfg.enabled = True              # 是否开启定期评估
    eval_cfg.interval = 5               # 每隔几个 epoch 评估一次
    eval_cfg.num_episodes = 20          # 每次评估的 episode 数
    eval_cfg.seed = 20000               # 固定验证集 seed，保证可复现

    return config
