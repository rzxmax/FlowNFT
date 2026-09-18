"""Low-dimensional state imitation training."""

from config.imitation.base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.pretrain.obs_mode = 'state'
    config.env_kwargs.obs_mode = 'state'
    config.pretrain.dataset_path = 'config/imitation/datasets_state.json'
    config.pretrain.checkpoint_dir = 'checkpoints/imitation/state/'
    config.pretrain.loss_type = 'velocity'
    return config
