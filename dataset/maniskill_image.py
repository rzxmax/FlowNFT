"""RGB and depth observations from ManiSkill rgbd replay H5."""

import numpy as np

from .maniskill_base import ManiSkillReplayDataset
from .maniskill_transforms import agent_position, rgbd_to_chw


class ManiSkillImageDataset(ManiSkillReplayDataset):
    obs_key = "obs/sensor_data"
    state_key = "agent_pos"

    def __init__(self, h5_path, camera="base_camera", use_depth=True,
                 depth_max=5.0, image_size=None, obs_steps=None, **kwargs):
        if depth_max <= 0:
            raise ValueError("depth_max must be positive")
        self.camera = camera
        self.use_depth = use_depth
        self.depth_max = float(depth_max)
        self.image_size = tuple(image_size) if image_size is not None else None
        super().__init__(h5_path, **kwargs)
        self.obs_steps = self.horizon if obs_steps is None else int(obs_steps)
        if not 1 <= self.obs_steps <= self.horizon:
            raise ValueError("obs_steps must be between 1 and horizon")

    def _observation_dataset(self, group):
        camera = f"obs/sensor_data/{self.camera}"
        if f"{camera}/rgb" not in group:
            raise ValueError(f"Replay H5 lacks {camera}/rgb")
        if self.use_depth and f"{camera}/depth" not in group:
            raise ValueError(f"Replay H5 lacks {camera}/depth")
        self._check_agent_pos(group)
        return group[f"{camera}/rgb"]

    @staticmethod
    def _check_agent_pos(group):
        for key, dim in (("obs/agent/qpos", 9), ("obs/extra/goal_pos", 3),
                         ("obs/extra/tcp_pose", 7)):
            if key not in group or group[key].shape[-1] != dim:
                raise ValueError(f"Replay H5 requires {key} with last dimension {dim}")

    def _state(self, group, indices):
        return agent_position(*[
            self._read(group[key], indices).astype(np.float32)
            for key in ("obs/agent/qpos", "obs/extra/goal_pos", "obs/extra/tcp_pose")
        ])

    def _sample(self, group, indices, index):
        indices = indices[:self.obs_steps]
        camera = f"obs/sensor_data/{self.camera}"
        rgb = self._read(group[f"{camera}/rgb"], indices)
        depth = self._read(group[f"{camera}/depth"], indices) if self.use_depth else None
        if self.image_size is not None and rgb.shape[1:3] != self.image_size:
            import cv2
            height, width = self.image_size
            rgb = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                            for frame in rgb])
            if depth is not None:
                depth = np.stack([
                    cv2.resize(frame[..., 0], (width, height), interpolation=cv2.INTER_NEAREST)
                    for frame in depth
                ])[..., None]
        return {"image": rgbd_to_chw(rgb, depth, self.depth_max),
                "agent_pos": self._state(group, indices)}
