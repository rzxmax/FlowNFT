"""Observation adapters used by reinforcement rollout and evaluation."""

from collections import deque

import torch
import torch.nn.functional as F


def _as_batch(value, device, *, image=False):
    value = torch.as_tensor(value, device=device)
    expected_unbatched_dims = 3 if image else 1
    if value.dim() == expected_unbatched_dims:
        value = value.unsqueeze(0)
    return value


class StateObservationAdapter:
    mode = 'state'

    def __init__(self, device, normalizer=None):
        self.device = device
        self.normalizer = normalizer
        self._observation = None

    def reset(self):
        self._observation = None

    def observe(self, observation):
        self._observation = observation

    def condition(self):
        value = torch.as_tensor(
            self._observation, device=self.device, dtype=torch.float32
        )
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if self.normalizer is not None:
            value = self.normalizer['obs'].normalize(value)
        return value

    def encode(self, observation):
        self.observe(observation)
        return self.condition()

    def snapshot(self):
        """State samples already store the policy input directly."""
        return None


class ImageObservationAdapter:
    """Convert ManiSkill RGB-D dictionaries into ImageObsEncoder conditions."""

    mode = 'image'

    def __init__(self, encoder, device, encoder_config, normalizer=None, image_size=None):
        self.encoder = encoder
        self.device = device
        self.normalizer = normalizer
        self.camera = encoder_config.get('camera', 'base_camera')
        self.use_depth = bool(encoder_config.get('use_depth', True))
        self.depth_max = float(encoder_config.get('depth_max', 5.0))
        self.n_obs_steps = int(encoder_config.get('n_obs_steps', 1))
        self.image_size = tuple(image_size) if image_size is not None else None
        self._image_history = deque(maxlen=self.n_obs_steps)
        self._agent_history = deque(maxlen=self.n_obs_steps)

        if self.depth_max <= 0:
            raise ValueError('image depth_max must be positive')

    def reset(self):
        self._image_history.clear()
        self._agent_history.clear()

    def _extract(self, observation):
        try:
            sensor = observation['sensor_data'][self.camera]
            rgb = _as_batch(sensor['rgb'], self.device, image=True)
            agent = observation['agent']
            extra = observation['extra']
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"RGB-D observation is missing data for camera {self.camera!r}"
            ) from exc

        rgb = rgb.to(dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        if self.image_size is not None and tuple(rgb.shape[-2:]) != self.image_size:
            rgb = F.interpolate(
                rgb, size=self.image_size, mode='bilinear', align_corners=False
            )
        channels = [rgb]
        if self.use_depth:
            depth = _as_batch(sensor['depth'], self.device, image=True)
            depth = depth.to(dtype=torch.float32).permute(0, 3, 1, 2)
            depth = (depth / 1000.0).clamp(0.0, self.depth_max) / self.depth_max
            if self.image_size is not None and tuple(depth.shape[-2:]) != self.image_size:
                depth = F.interpolate(depth, size=self.image_size, mode='nearest')
            channels.append(depth)
        image = torch.cat(channels, dim=1)

        qpos = _as_batch(agent['qpos'], self.device).to(dtype=torch.float32)
        goal_pos = _as_batch(extra['goal_pos'], self.device).to(dtype=torch.float32)
        tcp_pose = _as_batch(extra['tcp_pose'], self.device).to(dtype=torch.float32)
        agent_pos = torch.cat((qpos, goal_pos, tcp_pose), dim=-1)
        if self.normalizer is not None:
            agent_pos = self.normalizer['agent_pos'].normalize(agent_pos)
        return image, agent_pos

    def observe(self, observation):
        image, agent_pos = self._extract(observation)
        if self._image_history and self._image_history[-1].shape[0] != image.shape[0]:
            self.reset()
        self._image_history.append(image)
        self._agent_history.append(agent_pos)

    def condition(self):
        images = list(self._image_history)
        agents = list(self._agent_history)
        if not images:
            raise RuntimeError('No image observation is available; call observe() first')
        while len(images) < self.n_obs_steps:
            images.insert(0, images[0])
            agents.insert(0, agents[0])

        image_stack = torch.stack(images, dim=1)
        agent_stack = torch.stack(agents, dim=1)
        with torch.no_grad():
            return self.encoder(image_stack, agent_stack)

    def snapshot(self):
        """Return an encoder-ready RGB-D history on CPU for NFT updates.

        The rollout graph is intentionally discarded.  Images are retained as
        float16 to keep the replay payload bounded; they are converted back to
        float32 before the trainable encoder forward pass.
        """
        images = list(self._image_history)
        agents = list(self._agent_history)
        if not images:
            raise RuntimeError('No image observation is available; call observe() first')
        while len(images) < self.n_obs_steps:
            images.insert(0, images[0])
            agents.insert(0, agents[0])
        return {
            'image': torch.stack(images, dim=1).detach().to(
                device='cpu', dtype=torch.float16
            ),
            'agent_pos': torch.stack(agents, dim=1).detach().to(
                device='cpu', dtype=torch.float32
            ),
        }

    def encode(self, observation):
        self.observe(observation)
        return self.condition()
