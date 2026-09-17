"""Episode indexing and lazy HDF5 access shared by ManiSkill observation modes."""

from bisect import bisect_right
from dataclasses import dataclass
import copy
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Episode:
    path: str
    key: Optional[str]
    length: int


def _trajectory_keys(h5):
    keys = [key for key in h5 if key.startswith("traj_") and key[5:].isdigit()]
    if keys:
        return sorted(keys, key=lambda key: int(key[5:]))
    if "actions" in h5:
        return [None]  # Already split into one episode per file.
    raise ValueError("Expected traj_N groups or a root-level actions dataset")


class ManiSkillReplayDataset(Dataset):
    """Map replay H5 trajectories to fixed-length action windows.

    An observation at index t is paired with actions[t]. H5 handles are opened
    inside each DataLoader worker, never serialized into worker processes.
    """

    obs_key = None

    def __init__(self, h5_path, horizon=1, pad_before=0, pad_after=0,
                 seed=42, val_ratio=0.0, max_episodes=None, success_only=True):
        if horizon < 1 or pad_before < 0 or pad_after < 0:
            raise ValueError("horizon must be positive and padding nonnegative")
        paths = self._resolve_paths(h5_path)
        self.episodes = []
        self.action_dim = None
        self.obs_dim = None
        for path in paths:
            with h5py.File(path, "r") as h5:
                for key in _trajectory_keys(h5):
                    group = h5[key] if key is not None else h5
                    if self.obs_key not in group:
                        raise ValueError(
                            f"{path}:{key or '/'} lacks {self.obs_key!r}; "
                            "use a ManiSkill replay H5 with the requested obs_mode"
                        )
                    actions = group["actions"]
                    if actions.ndim != 2:
                        raise ValueError(f"{path}:{key}: actions must be rank 2")
                    if self.action_dim is None:
                        self.action_dim = actions.shape[1]
                    elif self.action_dim != actions.shape[1]:
                        raise ValueError("Mixed action dimensions in replay files")
                    if "success" in group:
                        completed = np.flatnonzero(group["success"][:])
                    elif "terminated" in group:
                        completed = np.flatnonzero(group["terminated"][:])
                    else:
                        completed = np.empty(0, dtype=np.int64)
                    if success_only and not len(completed):
                        continue
                    length = int(completed[0]) + 1 if len(completed) else len(actions)
                    if length == 0:
                        continue
                    obs_len = len(self._observation_dataset(group))
                    if obs_len < length:
                        raise ValueError(f"{path}:{key}: observations shorter than actions")
                    self.episodes.append(Episode(str(path), key, length))
                    if max_episodes is not None and len(self.episodes) >= max_episodes:
                        break
            if max_episodes is not None and len(self.episodes) >= max_episodes:
                break
        if not self.episodes:
            raise ValueError(f"No usable episodes in {h5_path}")

        self.horizon = int(horizon)
        self.pad_before = min(int(pad_before), self.horizon - 1)
        self.pad_after = min(int(pad_after), self.horizon - 1)
        self._files = {}
        self._stats = None
        self._seed = seed
        self._val_ratio = val_ratio
        self.train_mask = np.ones(len(self.episodes), dtype=bool)
        if val_ratio > 0:
            n_val = min(max(1, round(len(self.episodes) * val_ratio)), len(self.episodes) - 1)
            val_ids = np.random.default_rng(seed).choice(len(self.episodes), n_val, replace=False)
            self.train_mask[val_ids] = False
        self._set_mask(self.train_mask)

    @staticmethod
    def _resolve_paths(source):
        sources = source if isinstance(source, (list, tuple)) else [source]
        paths = []
        for item in sources:
            path = Path(item).expanduser()
            if path.is_dir():
                paths.extend(sorted((*path.glob("*.h5"), *path.glob("*.hdf5")),
                                    key=lambda p: (0, int(p.stem.removeprefix("episode")))
                                    if p.stem.removeprefix("episode").isdigit()
                                    else (1, p.name)))
            elif path.is_file():
                paths.append(path)
            else:
                raise FileNotFoundError(path)
        if not paths:
            raise ValueError(f"No H5 files found in {source}")
        return paths

    def _observation_dataset(self, group):
        raise NotImplementedError

    def _set_mask(self, mask):
        self._selected = [i for i, keep in enumerate(mask) if keep]
        self._counts = []
        total = 0
        for i in self._selected:
            length = self.episodes[i].length
            total += max(0, length - self.horizon + 1 + self.pad_before + self.pad_after)
            self._counts.append(total)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result._files = {}
        result._stats = None
        result.train_mask = ~self.train_mask
        result._set_mask(result.train_mask)
        return result

    def __len__(self):
        return self._counts[-1] if self._counts else 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def __del__(self):
        self._close_files()

    def _close_files(self):
        for _, handle in getattr(self, "_files", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        self._files = {}

    def _group(self, episode):
        pid = os.getpid()
        entry = self._files.get(episode.path)
        if entry is None or entry[0] != pid:
            handle = h5py.File(episode.path, "r")
            self._files[episode.path] = (pid, handle)
        else:
            handle = entry[1]
        return handle[episode.key] if episode.key is not None else handle

    def _window(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        selected_idx = bisect_right(self._counts, index)
        previous = self._counts[selected_idx - 1] if selected_idx else 0
        episode = self.episodes[self._selected[selected_idx]]
        start = index - previous - self.pad_before
        return episode, np.clip(np.arange(start, start + self.horizon), 0, episode.length - 1)

    @staticmethod
    def _read(dataset, indices):
        # HDF5 fancy indexing requires increasing indices; one contiguous read
        # also handles repeated boundary frames used for padding.
        first, last = int(indices[0]), int(indices[-1]) + 1
        return np.asarray(dataset[first:last])[indices - first]

    def _state(self, group, indices):
        raise NotImplementedError

    def _sample(self, group, indices, index):
        raise NotImplementedError

    def __getitem__(self, index):
        episode, indices = self._window(index)
        group = self._group(episode)
        sample = self._sample(group, indices, index)
        sample["action"] = self._read(group["actions"], indices).astype(np.float32)
        return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in sample.items()}

    def _normalizer_data(self):
        if self._stats is None:
            states, actions = [], []
            for i in self._selected:
                episode = self.episodes[i]
                group = self._group(episode)
                indices = np.arange(episode.length)
                states.append(self._state(group, indices))
                actions.append(np.asarray(group["actions"][:episode.length], dtype=np.float32))
            self._stats = {
                self.state_key: np.concatenate(states),
                "action": np.concatenate(actions),
            }
            self._close_files()
        return self._stats

    def get_normalizer(self, mode="limits", **kwargs):
        from models.normalizer import LinearNormalizer
        kwargs.setdefault("range_eps", 5e-2)
        normalizer = LinearNormalizer()
        normalizer.fit(self._normalizer_data(), last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self._normalizer_data()["action"])
