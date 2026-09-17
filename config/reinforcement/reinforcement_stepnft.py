"""StepNFT reinforcement fine-tuning."""

from config.reinforcement.base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.train.loss_type = 'stepnft'
    train = config.train
    train.stepnft_noise_level = 0.3
    train.stepnft_beta = 1.0
    train.stepnft_max_drift = 0.5
    train.stepnft_loss_form = 'dpo'
    train.stepnft_dpo_beta = 1.0
    train.stepnft_kl_beta = 1e-4
    train.stepnft_std_epsilon = 1e-4
    return config
