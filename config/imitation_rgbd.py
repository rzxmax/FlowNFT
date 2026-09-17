"""RGB-D imitation training."""

from config.robot_base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.pretrain.obs_mode = 'image'
    config.pretrain.dataset_path = (
        '~/.maniskill/demos/PickCube-v1/motionplanning/'
        'trajectory.rgbd.pd_ee_delta_pos.physx_cpu.h5'
    )
    config.pretrain.img_use_depth = True
    return config
