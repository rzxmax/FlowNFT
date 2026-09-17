"""Point-cloud imitation training."""

from config.imitation.base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.pretrain.obs_mode = 'pointcloud'
    config.env_kwargs.obs_mode = 'pointcloud'
    config.pretrain.dataset_path = (
        '~/.maniskill/demos/PickCube-v1/motionplanning/'
        'trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5'
    )
    pretrain = config.pretrain
    pretrain.pcd_n_obs_steps = 2
    pretrain.pcd_n_points = 512
    pretrain.pcd_encoder_output_dim = 64
    pretrain.pcd_agent_pos_dim = 19
    pretrain.pcd_state_mlp_out = 64
    pretrain.pcd_use_layernorm = True
    pretrain.pcd_final_norm = 'layernorm'
    pretrain.pcd_max_pts = 4096
    pretrain.checkpoint_dir = 'checkpoints/imitation/pointcloud'
    return config
