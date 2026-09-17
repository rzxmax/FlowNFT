"""Low-dimensional state imitation training."""

from config.robot_base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.pretrain.obs_mode = 'state'
    return config
