"""Settings shared by the reinforcement loss variants."""

import ml_collections

from config.robot_base import get_config as get_robot_config


def get_config():
    config = get_robot_config()
    config.max_episode_steps = 300
    config.env_kwargs.obs_mode = 'state'
    config.run_name = ''
    config.logdir = 'logs'
    config.save_freq = 10
    config.num_checkpoint_limit = 5
    config.mixed_precision = 'fp16'
    config.allow_tf32 = True
    config.resume_from = ''
    config.save_dir = 'checkpoints/reinforcement/FlowNFT'
    config.pretrain_checkpoint_dir = 'checkpoints/imitation'
    episode = config.episode = ml_collections.ConfigDict()
    episode.num_state_groups = 256
    episode.samples_per_group = 1
    episode.parallel_envs = episode.num_state_groups * episode.samples_per_group
    episode.rollout_epoch = 1
    episode.update_epoch = 1
    episode.num_epochs = 100
    stage = config.stage = ml_collections.ConfigDict()
    stage.advantage_scale = 1.0
    stage.timestep_weight_start = 1.0
    stage.timestep_weight_end = 1.0
    stage.advantage_type = 'single'
    stage.max_reward = 5.0
    config.sample.kl_reward = 0
    train = config.train = ml_collections.ConfigDict()
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
    train.decay_type = 1
    train.use_lr_scheduler = False
    train.lr_scheduler_type = 'cosine'
    train.warmup_epochs = 20
    train.min_lr = 5e-6
    train.num_cycles = 0.5
    train.use_replay_buffer = False
    train.replay_buffer_size = 10000
    train.target_positive_ratio = 0.5
    train.balance_threshold = 0.05
    train.buffer_sample_augment_ratio = 0.2
    train.sliding_window_stride = 1
    eval_cfg = config.eval = ml_collections.ConfigDict()
    eval_cfg.enabled = True
    eval_cfg.interval = 5
    eval_cfg.num_episodes = 20
    eval_cfg.seed = 20000
    return config
