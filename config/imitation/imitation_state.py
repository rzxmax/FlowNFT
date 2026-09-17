"""Low-dimensional state imitation training."""

from config.imitation.base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.pretrain.obs_mode = 'state'
    config.env_kwargs.obs_mode = 'state'
    config.pretrain.dataset_path = 'data/datasets/state/episodes'
    return config
