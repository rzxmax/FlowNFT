"""Settings shared by imitation and reinforcement training."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.env_id = 'PickCube-v1'
    config.action_horizon = 16
    config.env_kwargs = ml_collections.ConfigDict()
    config.env_kwargs.sim_backend = 'gpu'
    config.env_kwargs.control_mode = 'pd_ee_delta_pos'
    config.env_kwargs.render_mode = 'rgb_array'
    config.env_kwargs.reward_mode = 'dense'
    config.env_kwargs.sim_freq = 500
    config.env_kwargs.control_freq = 5
    config.sample = ml_collections.ConfigDict()
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 10
    return config
