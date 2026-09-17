"""RGB-D imitation training."""

from config.imitation.base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.pretrain.obs_mode = 'image'
    config.env_kwargs.obs_mode = 'rgbd'
    config.pretrain.dataset_path = (
        '~/.maniskill/demos/PickCube-v1/motionplanning/'
        'trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5'
    )
    pretrain = config.pretrain
    pretrain.img_n_obs_steps = 2
    pretrain.img_camera = 'base_camera'
    pretrain.img_use_depth = True
    pretrain.img_depth_max = 5.0
    pretrain.img_resnet_name = 'resnet18'
    pretrain.img_resnet_feat_dim = 512
    pretrain.img_agent_pos_dim = 19
    pretrain.img_state_mlp_out = 64
    pretrain.img_use_groupnorm = True
    pretrain.img_size = (128, 128)
    return config
