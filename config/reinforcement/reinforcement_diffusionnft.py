"""DiffusionNFT rollout reinforcement fine-tuning."""

from config.reinforcement.base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.train.loss_type = 'rollout'
    config.train.beta = 0.5
    config.train.kl_weight = 1e-4
    return config
