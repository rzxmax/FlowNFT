"""StepNFT reinforcement fine-tuning."""

from config.robot_base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.train.loss_type = 'stepnft'
    return config
