"""Prevent reinforcement training from silently starting with random weights."""

from types import SimpleNamespace

import pytest

from train.reinforcement import checkpoint_input


def _config(directory, *, resume_from=''):
    return SimpleNamespace(
        pretrain=SimpleNamespace(checkpoint_dir=str(directory), obs_mode='state'),
        resume_from=resume_from,
    )


def test_legacy_pretrain_path_resolves_to_moved_file(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_input, 'PROJECT_ROOT', tmp_path)
    directory = tmp_path / 'checkpoints' / 'imitation'
    directory.mkdir(parents=True)
    checkpoint = directory / 'pretrain_epoch200_episodes500.pth'
    checkpoint.touch()

    result = checkpoint_input.resolve_pretrain_checkpoint(
        _config(directory), 'pretrain-checkpoints/pretrain_epoch200_episodes500.pth'
    )
    assert result == checkpoint


def test_missing_explicit_checkpoint_fails_even_when_another_exists(tmp_path):
    (tmp_path / 'other_state.pth').touch()
    with pytest.raises(FileNotFoundError, match='预训练 checkpoint 不存在'):
        checkpoint_input.resolve_pretrain_checkpoint(
            _config(tmp_path), str(tmp_path / 'missing_state.pth')
        )


def test_automatic_selection_rejects_ambiguity_and_random_start(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match='没有适用于'):
        checkpoint_input.resolve_pretrain_checkpoint(config)
    first = tmp_path / 'first_state.pth'
    first.touch()
    assert checkpoint_input.resolve_pretrain_checkpoint(config) == first
    (tmp_path / 'second_state.pth').touch()
    with pytest.raises(ValueError, match='多个适用于'):
        checkpoint_input.resolve_pretrain_checkpoint(config)
    assert checkpoint_input.resolve_pretrain_checkpoint(
        _config(tmp_path, resume_from='saved-state')
    ) is None
