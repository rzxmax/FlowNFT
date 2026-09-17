"""CPU smoke tests for the refactored training boundaries."""

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from config.imitation_pointcloud import get_config as pointcloud_config
from config.imitation_rgbd import get_config as rgbd_config
from config.imitation_state import get_config as state_config
from config.reinforcement_diffusionnft import get_config as diffusionnft_config
from config.reinforcement_stepnft import get_config as stepnft_config
from train.common.checkpoint import save_ckpt
from train.common.observations import sample_pcd
from train.common.sampling import _unet_timestep_input, denoise_with_sde_snap
from train.imitation import main as imitation_main
from train.reinforcement.advantage import compute_episode_level_advantages
from train.reinforcement.old_policy import return_decay
from train.reinforcement.losses import diffusionnft, stepnft


def test_stage_configs_are_independent():
    state = state_config()
    pointcloud = pointcloud_config()
    rgbd = rgbd_config()
    diffusionnft = diffusionnft_config()
    stepnft = stepnft_config()

    assert (state.pretrain.obs_mode, pointcloud.pretrain.obs_mode,
            rgbd.pretrain.obs_mode) == ('state', 'pointcloud', 'image')
    assert diffusionnft.train.loss_type == 'rollout'
    assert stepnft.train.loss_type == 'stepnft'
    pointcloud.pretrain.obs_mode = 'state'
    assert rgbd.pretrain.obs_mode == 'image'


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


def test_legacy_entrypoint_keeps_stage_dispatch(monkeypatch):
    import train_sd3_robot_grpo as legacy
    from train.reinforcement import runner as reinforcement_runner

    config = state_config()
    calls = []
    flags = SimpleNamespace(
        config=config, gpu=3, test=False, sampletest=False,
        pretrain=True, pretrain_auto=False, action_loss=False,
        velocity_loss=True, pretrain_checkpoint='checkpoint.pth',
        validate_only=False,
    )
    monkeypatch.setattr(legacy, 'FLAGS', flags)
    monkeypatch.setattr(
        imitation_main, 'run_imitation',
        lambda cfg, gpu, **kwargs: calls.append(('imitation', cfg, gpu, kwargs)),
    )
    monkeypatch.setattr(
        reinforcement_runner, 'run_reinforcement',
        lambda cfg, gpu, **kwargs: calls.append(('reinforcement', cfg, gpu, kwargs)),
    )

    assert legacy.main(None) is None
    assert calls[-1] == (
        'imitation', config, 3,
        {'auto': False, 'action_loss': False, 'velocity_loss': True},
    )

    flags.pretrain = False
    assert legacy.main(None) is None
    assert calls[-1] == (
        'reinforcement', config, 3,
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
    config = state_config()
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
        tmp_path / 'checkpoints' / 'checkpoint-epoch1' / 'model.pth',
        map_location='cpu', weights_only=False,
    )
    assert saved['action_dim'] == 4
    assert saved['decoder_trained'] is False
    assert saved['pretrain_loss_type'] == 'velocity'
    assert saved['loss_type'] == 'rollout'
    assert 'action_decoder_state_dict' not in saved


@pytest.mark.parametrize(
    ('loss_module', 'loss_function'),
    [
        (diffusionnft, diffusionnft.train_on_episode_samples_diffusionnft),
        (stepnft, stepnft.train_on_episode_samples_stepnft),
    ],
)
def test_rl_losses_handle_empty_rollout_on_cpu(monkeypatch, loss_module, loss_function):
    import logging

    monkeypatch.setattr(loss_module, 'logger', logging.getLogger(__name__))
    config = state_config()
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
