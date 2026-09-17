"""H5 replay readers preserve action alignment and observation transforms."""

import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader

from dataset import ManiSkillStateDataset, ManiSkillImageDataset, ManiSkillPointCloudDataset


@pytest.fixture
def replay_h5(tmp_path):
    path = tmp_path / "replay.h5"
    with h5py.File(path, "w") as h5:
        for episode_id in (0, 1):
            group = h5.create_group(f"traj_{episode_id}")
            group.create_dataset("actions", data=np.arange(4, dtype=np.float32)[:, None].repeat(4, 1))
            group.create_dataset("terminated", data=[False, False, episode_id == 0, False])
            group.create_dataset("success", data=[False, False, episode_id == 0, False])
            obs = group.create_group("obs")
            obs.create_dataset("state", data=np.arange(5, dtype=np.float32)[:, None])
            agent = obs.create_group("agent")
            agent.create_dataset("qpos", data=np.ones((5, 9), dtype=np.float32))
            extra = obs.create_group("extra")
            extra.create_dataset("goal_pos", data=np.full((5, 3), 2, dtype=np.float32))
            extra.create_dataset("tcp_pose", data=np.full((5, 7), 3, dtype=np.float32))
            camera = obs.create_group("sensor_data/base_camera")
            rgb = np.zeros((5, 2, 2, 3), dtype=np.uint8)
            rgb[:, :, :, 0] = 255
            camera.create_dataset("rgb", data=rgb)
            camera.create_dataset("depth", data=np.full((5, 2, 2, 1), 2000, dtype=np.int16))
            cloud = obs.create_group("pointcloud")
            cloud.create_dataset("xyzw", data=np.tile(np.array(
                [[1, 2, 3, 1], [4, 5, 6, 1], [7, 8, 9, 0]], dtype=np.float32), (5, 1, 1)))
            cloud.create_dataset("segmentation", data=np.tile(
                np.array([[18], [16], [18]], dtype=np.int32), (5, 1, 1)))
    return path


def test_state_alignment_success_and_padding(replay_h5):
    # The state reader requires the flat obs layout used by state replay.
    with h5py.File(replay_h5, "r+") as h5:
        for key in h5:
            obs = h5[key]["obs"]
            obs.create_dataset("flat", data=obs["state"][:])
            del obs["state"]
            h5[key].create_dataset("flat_obs", data=obs["flat"][:])
            del h5[key]["obs"]
            h5[key].move("flat_obs", "obs")
    ds = ManiSkillStateDataset(replay_h5, horizon=2, pad_before=1)
    assert len(ds.episodes) == 1
    assert len(ds) == 3
    np.testing.assert_array_equal(ds[0]["obs"].numpy().ravel(), [0, 0])
    np.testing.assert_array_equal(ds[2]["action"].numpy()[:, 0], [1, 2])
    assert next(iter(DataLoader(ds, batch_size=2, num_workers=2)))["obs"].shape == (2, 2, 1)


def test_rgbd_and_pointcloud_from_replay(replay_h5):
    image = ManiSkillImageDataset(replay_h5, horizon=3, obs_steps=2, depth_max=5)
    sample = image[0]
    assert sample["image"].shape == (2, 4, 2, 2)
    assert sample["agent_pos"].shape == (2, 19)
    assert sample["action"].shape == (3, 4)
    np.testing.assert_allclose(sample["image"][:, 0], 1)
    np.testing.assert_allclose(sample["image"][:, 3], 0.4)
    np.testing.assert_array_equal(sample["agent_pos"][0, :9], np.ones(9))

    cloud = ManiSkillPointCloudDataset(replay_h5, horizon=3, obs_steps=2, max_pts=4)
    sample = cloud[0]
    assert cloud.online_fps
    assert sample["point_cloud"].shape == (2, 4, 3)
    np.testing.assert_array_equal(sample["valid_count"], [1, 1])
    np.testing.assert_array_equal(sample["point_cloud"][0, 0], [1, 2, 3])
    np.testing.assert_array_equal(sample["point_cloud"][0, 1:], 0)
