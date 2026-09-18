"""Imitation manifests combine replay files without hiding later tasks."""

import json

import h5py
import numpy as np
import pytest

from dataset import ManiSkillStateDataset
from train.imitation.data_sources import load_dataset_paths


def _make_replay(path, value):
    with h5py.File(path, 'w') as h5:
        for episode_id in range(2):
            group = h5.create_group(f'traj_{episode_id}')
            group.create_dataset('obs', data=np.full((4, 2), value, dtype=np.float32))
            group.create_dataset('actions', data=np.full((3, 4), value, dtype=np.float32))
            group.create_dataset('success', data=[False, False, True])


def test_manifest_loads_multiple_files_and_caps_each_file(tmp_path):
    first, second = tmp_path / 'first.h5', tmp_path / 'second.h5'
    _make_replay(first, 1)
    _make_replay(second, 2)
    manifest = tmp_path / 'datasets.json'
    manifest.write_text(json.dumps({'datasets': ['first.h5', 'second.h5']}))

    paths = load_dataset_paths(manifest)
    assert paths == [str(first), str(second)]
    dataset = ManiSkillStateDataset(
        paths, horizon=4, pad_after=3, max_episodes=1,
        max_episodes_per_file=True,
    )
    assert len(dataset.episodes) == 2
    assert len(dataset) == 6
    assert [episode.path for episode in dataset.episodes] == paths
    np.testing.assert_array_equal(dataset[0]['action'].numpy()[0], [1] * 4)
    np.testing.assert_array_equal(dataset[3]['action'].numpy()[0], [2] * 4)


@pytest.mark.parametrize('contents', [[], {'datasets': []}, {'datasets': [42]}])
def test_manifest_rejects_invalid_entries(tmp_path, contents):
    manifest = tmp_path / 'datasets.json'
    manifest.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        load_dataset_paths(manifest)
