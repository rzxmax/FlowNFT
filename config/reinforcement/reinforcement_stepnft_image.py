"""StepNFT reinforcement fine-tuning for an RGB-D image policy."""

import ml_collections

from config.reinforcement.reinforcement_stepnft import get_config as get_stepnft_config


def get_config():
    config = get_stepnft_config()

    # ``policy_obs_mode`` describes the input consumed by the policy.  ManiSkill
    # calls the corresponding raw environment observation mode ``rgbd``.
    config.policy_obs_mode = 'image'
    config.env_kwargs.obs_mode = 'rgbd'

    # These values are fallbacks for old checkpoints.  New image checkpoints
    # carry ``img_encoder_config`` and take precedence at runtime.
    observation = config.observation = ml_collections.ConfigDict()
    observation.camera = 'base_camera'
    observation.use_depth = True
    observation.depth_max = 5.0
    observation.n_obs_steps = 1
    observation.in_channels = 4
    observation.resnet_name = 'resnet18'
    observation.img_feat_dim = 512
    observation.agent_pos_dim = 19
    observation.state_mlp_out = 64
    observation.use_groupnorm = True
    observation.image_size = (128, 128)

    return config
