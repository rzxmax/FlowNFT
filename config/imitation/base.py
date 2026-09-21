"""Settings shared by the imitation observation modes."""

import ml_collections

from config.robot_base import get_config as get_robot_config


def get_config():
    config = get_robot_config()
    pretrain = config.pretrain = ml_collections.ConfigDict()
    pretrain.num_episodes = 1000
    pretrain.num_epochs = 200
    # Per-device batch size. ``batch_size`` is kept as a compatibility alias
    # for older local presets; the trainer prefers this explicit field.
    pretrain.batch_size = 64
    pretrain.batch_size_per_gpu = 64
    # Physical GPU IDs consumed by ``python -m train.imitation.launch``.
    pretrain.gpu_ids = [0]
    pretrain.mixed_precision = 'no'
    pretrain.find_unused_parameters = False
    pretrain.num_workers = 4
    pretrain.pin_memory = True
    pretrain.persistent_workers = True
    pretrain.distributed_backend = 'nccl'
    pretrain.learning_rate = 1e-4
    pretrain.checkpoint_dir = 'checkpoints/imitation'
    pretrain.checkpoint_interval = 50
    pretrain.checkpoint_keep = 3
    pretrain.gradient_accumulation_steps = 1
    pretrain.max_grad_norm = 1.0
    pretrain.loss_type = 'action'
    pretrain.adam_beta1 = 0.95
    pretrain.adam_beta2 = 0.999
    pretrain.adam_eps = 1e-8
    pretrain.use_lr_scheduler = False
    pretrain.lr_scheduler_type = 'cosine'
    pretrain.warmup_epochs = 50
    pretrain.min_lr = 1e-5
    pretrain.ema_use_warmup = False
    pretrain.ema_decay = 0.9999
    pretrain.ema_min_decay = 0.0
    pretrain.ema_update_after_step = 0
    pretrain.ema_inv_gamma = 1.0
    pretrain.ema_power = 0.75
    pretrain.auto_mode = False
    pretrain.auto_eval_interval = 10
    pretrain.auto_num_test_episodes = 20
    pretrain.auto_success_threshold = 15
    pretrain.auto_max_epochs = 1000
    return config
