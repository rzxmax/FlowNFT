"""Observation transforms shared by replay datasets and live evaluation."""

import numpy as np


PICKCUBE_KEEP_SEG_IDS = tuple(range(1, 16)) + (18,)


def agent_position(qpos, goal_pos, tcp_pose):
    """Return [qpos(9), goal_pos(3), tcp_pose(7)] as float32."""
    return np.concatenate((qpos, goal_pos, tcp_pose), axis=-1).astype(np.float32)


def rgbd_to_chw(rgb, depth=None, depth_max=5.0):
    """Convert HWC or THWC ManiSkill images to CHW or TCHW float32."""
    if depth_max <= 0:
        raise ValueError("depth_max must be positive")
    channels = [np.moveaxis(np.asarray(rgb, dtype=np.float32) / 255.0, -1, -3)]
    if depth is not None:
        depth_m = np.asarray(depth, dtype=np.float32) / 1000.0
        channels.append(np.moveaxis(np.clip(depth_m, 0, depth_max) / depth_max, -1, -3))
    return np.concatenate(channels, axis=-3).astype(np.float32)


def filtered_xyz(xyzw, segmentation=None, keep_seg_ids=PICKCUBE_KEEP_SEG_IDS):
    """Remove invalid points and optionally retain configured actor IDs."""
    mask = xyzw[:, 3] > 0
    if segmentation is not None:
        mask &= np.isin(np.asarray(segmentation).reshape(-1), keep_seg_ids)
    return np.asarray(xyzw[mask, :3], dtype=np.float32)
