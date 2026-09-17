"""Historical DiffusionNFT preset retained for existing experiments."""

from config.reinforcement.reinforcement_diffusionnft import get_config as get_diffusionnft_config


def get_config():
    config = get_diffusionnft_config()
    config.train.kl_weight = 0.01
    config.train.old_model_update_mode = 'per_epoch'
    config.stage.num_state_groups = config.episode.num_state_groups
    config.stage.samples_per_group = config.episode.samples_per_group
    config.train.use_traj_gamma_discount = True
    config.train.gamma_traj = 0.99
    config.train.use_flow_temporal_discount = False
    config.train.gamma_temporal = 0.99
    return config
