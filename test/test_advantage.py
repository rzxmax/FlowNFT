"""Advantage labels must follow the environment success signal."""

from types import SimpleNamespace

from train.reinforcement.advantage import compute_episode_level_advantages


def test_single_advantage_uses_success_not_reward_threshold():
    samples = [
        {'success': True, 'reward': 4.9999999},
        {'success': False, 'reward': 5.0},
    ]
    config = SimpleNamespace(
        stage=SimpleNamespace(
            advantage_type='single', advantage_scale=1.0, max_reward=5.0
        ),
        episode=SimpleNamespace(num_state_groups=2),
    )

    labeled, _ = compute_episode_level_advantages(samples, 10, config)

    assert [sample['advantage'] for sample in labeled] == [1.0, -1.0]
