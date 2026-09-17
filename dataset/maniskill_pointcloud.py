"""Filtered XYZ point clouds from ManiSkill pointcloud replay H5."""

import numpy as np

from .maniskill_base import ManiSkillReplayDataset
from .maniskill_image import ManiSkillImageDataset
from .maniskill_transforms import PICKCUBE_KEEP_SEG_IDS, filtered_xyz


class ManiSkillPointCloudDataset(ManiSkillReplayDataset):
    obs_key = "obs/pointcloud"
    state_key = "agent_pos"

    def __init__(self, h5_path, max_pts=4096, keep_seg_ids=None,
                 filter_background=True, obs_steps=None, **kwargs):
        if max_pts < 1:
            raise ValueError("max_pts must be positive")
        self.max_pts = int(max_pts)
        self.keep_seg_ids = PICKCUBE_KEEP_SEG_IDS if keep_seg_ids is None else tuple(keep_seg_ids)
        self.filter_background = filter_background
        self.online_fps = True
        super().__init__(h5_path, **kwargs)
        self.obs_steps = self.horizon if obs_steps is None else int(obs_steps)
        if not 1 <= self.obs_steps <= self.horizon:
            raise ValueError("obs_steps must be between 1 and horizon")

    def _observation_dataset(self, group):
        if "obs/pointcloud/xyzw" not in group:
            raise ValueError("Replay H5 lacks obs/pointcloud/xyzw")
        if self.filter_background and "obs/pointcloud/segmentation" not in group:
            raise ValueError("Background filtering requires obs/pointcloud/segmentation")
        ManiSkillImageDataset._check_agent_pos(group)
        return group["obs/pointcloud/xyzw"]

    def _state(self, group, indices):
        return ManiSkillImageDataset._state(self, group, indices)

    def _sample(self, group, indices, index):
        indices = indices[:self.obs_steps]
        xyzw = self._read(group["obs/pointcloud/xyzw"], indices)
        segmentation = (self._read(group["obs/pointcloud/segmentation"], indices)
                        if self.filter_background else None)
        points = np.zeros((len(indices), self.max_pts, 3), dtype=np.float32)
        counts = np.zeros(len(indices), dtype=np.int32)
        for t, frame in enumerate(xyzw):
            valid = filtered_xyz(frame,
                                 segmentation[t] if segmentation is not None else None,
                                 self.keep_seg_ids)
            if len(valid) > self.max_pts:
                # The PickCube filtered cloud normally fits in max_pts. If it
                # does not, choose a reproducible subset before online FPS.
                rng = np.random.default_rng(self._seed + index + t)
                valid = valid[rng.choice(len(valid), self.max_pts, replace=False)]
            counts[t] = len(valid)
            points[t, :len(valid)] = valid
        return {"point_cloud": points, "valid_count": counts,
                "agent_pos": self._state(group, indices)}
