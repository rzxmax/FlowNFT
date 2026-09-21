"""CPU smoke tests for the refactored training boundaries."""

from datetime import datetime
import os

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from config.robot_base import get_config as robot_config
from config.imitation.imitation_pointcloud import get_config as pointcloud_config
from config.imitation.imitation_rgbd import get_config as rgbd_config
from config.imitation.imitation_state import get_config as state_config
from config.reinforcement.reinforcement_diffusionnft import get_config as diffusionnft_config
from config.reinforcement.reinforcement_stepnft import get_config as stepnft_config
from train.common.checkpoint import reinforcement_run_dir, save_ckpt
from train.common.observations import sample_pcd
from train.common.sampling import _unet_timestep_input, denoise_with_sde_snap
from train.imitation import main as imitation_main
from train.imitation.evaluator import _actions_to_env_space
from train.reinforcement.advantage import compute_episode_level_advantages
from train.reinforcement.old_policy import return_decay
from train.reinforcement.losses import diffusionnft, stepnft


def test_stage_configs_are_independent():
    shared = robot_config()
    state = state_config()
    pointcloud = pointcloud_config()
    rgbd = rgbd_config()
    diffusionnft = diffusionnft_config()
    stepnft = stepnft_config()

    assert (state.pretrain.obs_mode, pointcloud.pretrain.obs_mode,
            rgbd.pretrain.obs_mode) == ('state', 'pointcloud', 'image')
    assert diffusionnft.train.loss_type == 'rollout'
    assert stepnft.train.loss_type == 'stepnft'
    assert 'pretrain' not in shared and 'train' not in shared
    assert 'obs_mode' not in shared.env_kwargs and 'max_episode_steps' not in shared
    assert 'train' not in state and 'pretrain' not in diffusionnft
    assert (state.env_kwargs.obs_mode, pointcloud.env_kwargs.obs_mode,
            rgbd.env_kwargs.obs_mode) == ('state', 'pointcloud', 'rgbd')
    assert 'pcd_n_points' not in state.pretrain
    assert 'img_size' not in pointcloud.pretrain
    assert 'stepnft_beta' not in diffusionnft.train
    assert 'beta' not in stepnft.train
    assert state.pretrain.gpu_ids == [0]
    assert state.pretrain.batch_size_per_gpu == 64
    assert state.pretrain.mixed_precision == 'no'
    pointcloud.pretrain.obs_mode = 'state'
    assert rgbd.pretrain.obs_mode == 'image'


def test_distributed_worker_does_not_overwrite_gpu_visibility(monkeypatch):
    from train.common.bootstrap import configure_gpu

    monkeypatch.setenv('FLOWNFT_ACCELERATE_WORKER', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,5')
    configure_gpu(['--gpu=7'])
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,5'


def test_imitation_dispatch_and_loss_flags(monkeypatch):
    config = state_config()
    calls = []
    monkeypatch.setattr(
        imitation_main,
        'pretrain_model',
        lambda cfg, gpu: calls.append((cfg.pretrain.loss_type, gpu)),
    )

    imitation_main.run_imitation(config, 2, auto=True, velocity_loss=True)
    assert calls == [('velocity', 2)]
    assert config.pretrain.auto_mode is True
    with pytest.raises(ValueError, match='mutually exclusive'):
        imitation_main.run_imitation(
            config, 2, action_loss=True, velocity_loss=True
        )


def test_imitation_dispatch_leaves_wandb_to_trainer(monkeypatch):
    config = pointcloud_config()
    train = MagicMock(return_value='checkpoint.pth')
    monkeypatch.setenv('WANDB_PROJECT', 'FlowNFT')
    monkeypatch.setenv('WANDB_ENTITY', '545242836-zju')
    monkeypatch.setattr(imitation_main, 'pretrain_model', train)

    assert imitation_main.run_imitation(config, 0) == 'checkpoint.pth'
    train.assert_called_once_with(config, 0)


def test_imitation_multi_gpu_launch_command_and_validation():
    from train.imitation.launch import build_launch_command, validate_gpu_ids

    config = state_config()
    config.pretrain.gpu_ids = [1, 3]
    config.pretrain.mixed_precision = 'bf16'
    command = build_launch_command(
        config,
        ['--config=config/imitation/imitation_state.py'],
        python='/python',
    )

    assert command[:3] == ['/python', '-m', 'accelerate.commands.launch']
    assert command[command.index('--num_processes') + 1] == '2'
    assert command[command.index('--gpu_ids') + 1] == '1,3'
    assert command[command.index('--mixed_precision') + 1] == 'bf16'
    assert command[-3:] == [
        '-m',
        'train.imitation.main',
        '--config=config/imitation/imitation_state.py',
    ]

    for invalid in ([], [0, 0], [-1], [True], ['0']):
        with pytest.raises(ValueError):
            validate_gpu_ids(invalid)


def test_stage_entrypoints_dispatch(monkeypatch):
    from train.reinforcement import main as reinforcement_main

    imitation = state_config()
    reinforcement = diffusionnft_config()
    calls = []
    imitation_flags = SimpleNamespace(
        config=imitation, gpu=3, pretrain_auto=False, action_loss=False,
        velocity_loss=True, validate_only=False,
    )
    reinforcement_flags = SimpleNamespace(
        config=reinforcement, gpu=3, pretrain_checkpoint='checkpoint.pth',
        validate_only=False,
    )
    monkeypatch.setattr(imitation_main, 'FLAGS', imitation_flags)
    monkeypatch.setattr(reinforcement_main, 'FLAGS', reinforcement_flags)
    monkeypatch.setattr(
        imitation_main, 'run_imitation',
        lambda cfg, gpu, **kwargs: calls.append(('imitation', cfg, gpu, kwargs)),
    )
    monkeypatch.setattr(
        reinforcement_main, 'run_reinforcement',
        lambda cfg, gpu, **kwargs: calls.append(('reinforcement', cfg, gpu, kwargs)),
    )

    assert imitation_main.main(None) is None
    assert calls[-1] == (
        'imitation', imitation, 3,
        {'auto': False, 'action_loss': False, 'velocity_loss': True},
    )

    assert reinforcement_main.main(None) is None
    assert calls[-1] == (
        'reinforcement', reinforcement, 3,
        {'pretrain_checkpoint': 'checkpoint.pth'},
    )


def test_pointcloud_filter_and_padding():
    xyzw = np.array([
        [1.0, 2.0, 3.0, 1.0],
        [4.0, 5.0, 6.0, 1.0],
        [7.0, 8.0, 9.0, 0.0],
    ], dtype=np.float32)
    segmentation = np.array([[18], [16], [18]], dtype=np.int32)
    result = sample_pcd(xyzw, n_points=2, seg=segmentation)
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result[1], [0.0, 0.0, 0.0])


def test_advantage_modes():
    config = diffusionnft_config()
    config.stage.advantage_type = 'single'
    config.stage.max_reward = 5.0
    samples = [{'reward': 5.0}, {'reward': 1.0}]
    result, stats = compute_episode_level_advantages(samples, 10, config)
    assert [item['advantage'] for item in result] == [1.0, -1.0]
    assert stats['filtered_samples'] == 0

    config.stage.advantage_type = 'group'
    config.episode.num_state_groups = 1
    config.episode.samples_per_group = 2
    samples = [
        {'group_id': 0, 'reward': 5.0, 'terminated': True},
        {'group_id': 0, 'reward': 1.0, 'terminated': False},
    ]
    result, stats = compute_episode_level_advantages(samples, 10, config)
    assert result[0]['advantage'] == pytest.approx(1.0)
    assert result[1]['advantage'] == pytest.approx(-1.0)
    assert stats['filtered_groups'] == 0


def test_flow_schedule_and_old_policy_decay():
    assert _unet_timestep_input(0.0) == pytest.approx(0.99)
    assert _unet_timestep_input(1.0) == pytest.approx(99.0)
    assert return_decay(0, 0) == 0.0
    assert return_decay(500, 1) == pytest.approx(0.5)
    assert return_decay(74, 2) == 0.0
    with pytest.raises(ValueError, match='Unknown decay_type'):
        return_decay(0, 3)


def test_imitation_actions_are_converted_to_environment_space():
    class OffsetActionNormalizer:
        @staticmethod
        def unnormalize(actions):
            return actions * 2.0 + 3.0

    actions_normalized = torch.tensor([
        [-1.0, 0.0],
        [0.5, 1.0],
    ])
    normalizer = {'action': OffsetActionNormalizer()}

    actions = _actions_to_env_space(actions_normalized, normalizer)

    torch.testing.assert_close(
        actions,
        torch.tensor([[1.0, 3.0], [4.0, 5.0]]),
    )
    assert _actions_to_env_space(actions_normalized, None) is actions_normalized


def test_stepnft_sampler_shapes_on_cpu():
    class ConstantVelocity:
        def __call__(self, sample, timestep, global_cond):
            return torch.ones_like(sample) * 0.1

    device = torch.device('cpu')
    generators = [torch.Generator(device='cpu').manual_seed(i) for i in (1, 2)]
    actions, traces = denoise_with_sde_snap(
        obs_tensor=torch.zeros(2, 3),
        action_unet=ConstantVelocity(),
        action_decoder=None,
        num_steps=3,
        noise_level=0.2,
        action_horizon=4,
        action_dim=2,
        generators=generators,
        device=device,
    )
    assert actions.shape == (2, 4, 2)
    assert traces['nft_xt'].shape == (2, 4, 2)
    assert traces['nft_v'].shape == (2, 4, 2)
    assert traces['nft_xnext'].shape == (2, 4, 2)
    assert traces['nft_step_index'].shape == (2,)
    assert torch.isfinite(actions).all()


def test_reinforcement_checkpoint_keeps_imitation_metadata(tmp_path, monkeypatch):
    import logging
    from train.common import checkpoint

    # Production creates Accelerator before saving; avoid that GPU-oriented
    # runtime setup in this CPU-only checkpoint format test.
    monkeypatch.setattr(checkpoint, 'logger', logging.getLogger(__name__))

    class DummyAccelerator:
        is_main_process = True

        @staticmethod
        def unwrap_model(model):
            return model

    class TinyPolicy(torch.nn.Linear):
        input_dim = 4

    config = diffusionnft_config()
    config.train.ema = False
    policy = TinyPolicy(4, 4)
    save_ckpt(
        str(tmp_path), policy, None, 1, DummyAccelerator(), None,
        list(policy.parameters()), config,
        decoder_trained=False, pretrain_loss_type='velocity',
    )
    saved = torch.load(
        tmp_path / 'checkpoint-epoch1' / 'model.pth',
        map_location='cpu', weights_only=False,
    )
    assert saved['action_dim'] == 4
    assert saved['decoder_trained'] is False
    assert saved['pretrain_loss_type'] == 'velocity'
    assert saved['loss_type'] == 'rollout'
    assert 'action_decoder_state_dict' not in saved


def test_reinforcement_run_dir_uses_pretrain_loss_and_date(tmp_path):
    run_dir = reinforcement_run_dir(
        tmp_path,
        '/models/pretrain_epoch100_episodes500_state.pth',
        'stepnft',
        now=datetime(2026, 9, 20),
    )

    assert run_dir == str(
        tmp_path / 'pretrain_epoch100_episodes500_state_StepNFT_0920'
    )


@pytest.mark.parametrize(
    ('loss_module', 'loss_function', 'config_factory'),
    [
        (diffusionnft, diffusionnft.train_on_episode_samples_diffusionnft, diffusionnft_config),
        (stepnft, stepnft.train_on_episode_samples_stepnft, stepnft_config),
    ],
)
def test_rl_losses_handle_empty_rollout_on_cpu(monkeypatch, loss_module, loss_function, config_factory):
    import logging

    monkeypatch.setattr(loss_module, 'logger', logging.getLogger(__name__))
    config = config_factory()
    policy = torch.nn.Linear(4, 4)
    old_policy = torch.nn.Linear(4, 4)
    ref_policy = torch.nn.Linear(4, 4)
    accelerator = SimpleNamespace(device=torch.device('cpu'))
    step, loss = loss_function(
        samples=[],
        action_unet=policy, action_decoder=None,
        action_unet_old=old_policy, action_decoder_old=None,
        action_unet_ref=ref_policy, action_decoder_ref=None,
        optimizer=None, accelerator=accelerator, config=config,
        ema=None, trainable_parameters=list(policy.parameters()),
        global_step=5, epoch=0,
    )
    assert step == 5
    assert loss == 0.0
