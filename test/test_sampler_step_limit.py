"""A partial final action chunk must respect the episode step limit."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from train.reinforcement import sampler


def test_sampler_stops_at_max_episode_steps(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.steps = 0

        def reset(self, *, seed, options):
            self.steps = 0
            return torch.zeros((1, 42)), {}

        def step(self, action):
            self.steps += 1
            return (
                torch.zeros((1, 42)), torch.zeros(1),
                torch.tensor([False]), torch.tensor([False]), {},
            )

    env = FakeEnv()
    pipeline = SimpleNamespace(scheduler=SimpleNamespace(timesteps=torch.tensor([0])))
    config = SimpleNamespace(
        train=SimpleNamespace(loss_type='rollout'),
        sample=SimpleNamespace(num_steps=1),
        action_horizon=16,
        max_episode_steps=30,
    )
    monkeypatch.setattr(
        sampler, 'pipeline_with_logprob',
        lambda *args, **kwargs: {'actions': torch.zeros((1, 16, 4))},
    )

    samples = sampler.sample_complete_trajectories(
        env, pipeline, SimpleNamespace(eval=lambda: None), None,
        num_state_groups=1, samples_per_group=1, config=config,
        accelerator=SimpleNamespace(device=torch.device('cpu')),
        autocast=nullcontext, seed_offset=42,
    )

    assert env.steps == 30
    assert samples[0]['num_steps'] == 30
    assert samples[0]['num_flow_calls'] == 2
    assert samples[0]['success'] is False
    assert samples[0]['terminated'] is False
    assert samples[0]['truncated'] is True


def test_sampler_keeps_success_termination_and_truncation_separate(monkeypatch):
    class FakeEnv:
        def __init__(self, *, success, terminated, truncated):
            self.outcome = (success, terminated, truncated)

        def reset(self, *, seed, options):
            return torch.zeros((1, 42)), {}

        def step(self, action):
            success, terminated, truncated = self.outcome
            return (
                torch.zeros((1, 42)),
                torch.tensor([5.0 if success else 0.0]),
                torch.tensor([terminated]),
                torch.tensor([truncated]),
                {'success': torch.tensor([success])},
            )

    pipeline = SimpleNamespace(scheduler=SimpleNamespace(timesteps=torch.tensor([0])))
    config = SimpleNamespace(
        train=SimpleNamespace(loss_type='rollout'),
        sample=SimpleNamespace(num_steps=1),
        action_horizon=16,
        max_episode_steps=30,
    )
    monkeypatch.setattr(
        sampler, 'pipeline_with_logprob',
        lambda *args, **kwargs: {'actions': torch.zeros((1, 16, 4))},
    )

    for expected in (
        {'success': True, 'terminated': True, 'truncated': False},
        {'success': False, 'terminated': True, 'truncated': False},
        {'success': False, 'terminated': False, 'truncated': True},
    ):
        samples = sampler.sample_complete_trajectories(
            FakeEnv(**expected), pipeline, SimpleNamespace(eval=lambda: None), None,
            num_state_groups=1, samples_per_group=1, config=config,
            accelerator=SimpleNamespace(device=torch.device('cpu')),
            autocast=nullcontext, seed_offset=42,
        )

        assert samples[0]['num_steps'] == 1
        assert {key: samples[0][key] for key in expected} == expected
