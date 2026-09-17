"""Point-cloud imitation training."""

from config.robot_base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.pretrain.obs_mode = 'pointcloud'
    config.pretrain.dataset_path = (
        '~/.maniskill/demos/PickCube-v1/motionplanning/'
        'trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5'
    )
    return config
