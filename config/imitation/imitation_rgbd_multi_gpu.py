"""RGB-D imitation pretraining on physical GPUs 4, 5, and 6."""

from config.imitation.imitation_rgbd import get_config as get_rgbd_config


def get_config():
    config = get_rgbd_config()
    config.pretrain.gpu_ids = [4, 5, 6]
    return config
