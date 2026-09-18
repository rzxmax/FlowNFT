"""Checkpoint directories and retention stay scoped to one training run."""

from train.imitation.checkpoint_files import (
    checkpoint_path,
    create_run_dir,
    prune_checkpoints,
)


def test_run_directory_and_latest_three_checkpoints(tmp_path):
    run_dir = create_run_dir(
        tmp_path, epochs=500, episodes=1000, obs_mode='state'
    )
    assert run_dir.name.startswith('pretrain_epoch500_episodes1000_state_')
    assert create_run_dir(
        tmp_path, epochs=500, episodes=1000, obs_mode='state'
    ).name == run_dir.name + '_2'

    for epoch in (50, 100, 150, 200):
        checkpoint_path(
            run_dir, epoch=epoch, episodes=1000, obs_mode='state'
        ).touch()
    unrelated = run_dir / 'pretrain_epoch10_episodes1000_image.pth'
    unrelated.touch()
    prune_checkpoints(run_dir, episodes=1000, obs_mode='state', keep=3)

    assert not checkpoint_path(
        run_dir, epoch=50, episodes=1000, obs_mode='state'
    ).exists()
    assert sorted(path.name for path in run_dir.iterdir()) == sorted([
        unrelated.name,
        *(f'pretrain_epoch{epoch}_episodes1000_state.pth'
          for epoch in (100, 150, 200)),
    ])
