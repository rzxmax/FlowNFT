"""Low-dimensional state observations from ManiSkill replay H5."""

import numpy as np

from .maniskill_base import ManiSkillReplayDataset


class ManiSkillStateDataset(ManiSkillReplayDataset):
    obs_key = "obs"
    state_key = "obs"

    def _observation_dataset(self, group):
        obs = group["obs"]
        if not hasattr(obs, "shape") or obs.ndim != 2:
            raise ValueError("State mode requires a replay H5 with flat obs (T+1, D)")
        if self.obs_dim is None:
            self.obs_dim = obs.shape[1]
        elif self.obs_dim != obs.shape[1]:
            raise ValueError("Mixed state dimensions in replay files")
        return obs

    def _state(self, group, indices):
        return self._read(group["obs"], indices).astype(np.float32)

    def _sample(self, group, indices, index):
        return {"obs": self._state(group, indices)}
