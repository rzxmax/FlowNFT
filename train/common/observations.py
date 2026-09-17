"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    np,
    torch,
)
from dataset.maniskill_transforms import (
    PICKCUBE_KEEP_SEG_IDS, agent_position, filtered_xyz, rgbd_to_chw,
)


try:
    import pytorch3d.ops as _pcd_torch3d_ops
    _PCD_HAS_PYTORCH3D = True
except ImportError:
    _PCD_HAS_PYTORCH3D = False

_pcd_rng = np.random.default_rng(42)

def _obs_to_tensor(obs, device):
    """辅助函数：转换观测为tensor"""
    if isinstance(obs, torch.Tensor):
        obs_tensor = obs.to(device, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
    else:
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
    return obs_tensor


def _prepare_obs_tensor(obs, device):
    """统一的obs转tensor逻辑（用于筛选和采样）"""
    if isinstance(obs, torch.Tensor):
        if obs.device != device:
            obs_tensor = obs.to(device, dtype=torch.float32)
        else:
            obs_tensor = obs.to(dtype=torch.float32)
    else:
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)

    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)

    return obs_tensor


def _to_numpy_pcd(x) -> np.ndarray:
    """Tensor 或 ndarray → float32 ndarray"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32)


def sample_pcd(xyzw: np.ndarray, n_points: int, seg=None) -> np.ndarray:
    """
    从单帧点云 xyzw (N,4) 中采样 n_points 个点。
    与 convert_pointcloud_hdf5_to_zarr.py 处理逻辑完全一致：
      1. w > 0 基础过滤
      2. 语义过滤：只保留机械臂(1-15) + 物块(18)，排除桌面/地面背景
      3. 有效点 < n_points 时补零；否则 FPS（有 pytorch3d）或随机均匀采样

    Args:
        xyzw:     (N,4) 或 (1,N,4) float32
        n_points: 目标采样点数
        seg:      (N,1) 或 (1,N,1) int，None 时跳过语义过滤

    Returns:
        (n_points, 3) float32
    """
    if xyzw.ndim == 3:
        xyzw = xyzw[0]
    if seg is not None and seg.ndim == 3:
        seg = seg[0]

    valid_xyz = filtered_xyz(xyzw, seg, PICKCUBE_KEEP_SEG_IDS)
    n_valid   = len(valid_xyz)

    if n_valid == 0:
        return np.zeros((n_points, 3), dtype=np.float32)

    if n_valid < n_points:
        pad = np.zeros((n_points - n_valid, 3), dtype=np.float32)
        return np.concatenate([valid_xyz, pad], axis=0)

    if _PCD_HAS_PYTORCH3D and torch.cuda.is_available():
        pts = torch.from_numpy(valid_xyz).unsqueeze(0).cuda()   # (1, N, 3)
        k   = torch.tensor([n_points]).cuda()
        _, idx = _pcd_torch3d_ops.sample_farthest_points(points=pts, K=k)
        return valid_xyz[idx.squeeze(0).cpu().numpy()]
    else:
        idx = _pcd_rng.choice(n_valid, n_points, replace=False)
        return valid_xyz[idx]


def batched_fps_sample(
    padded_pts: torch.Tensor,
    valid_count: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """对一个 batch 的"过滤后但未采样"点云做 GPU 批量 FPS。

    每帧 padded_pts[b] 的前 valid_count[b] 个点是有效点，后面是补零 padding。
    使用 pytorch3d.sample_farthest_points 的 lengths 参数让 FPS 只在有效点中选取。

    采样规则（与 sample_pcd / convert 脚本对齐）：
      - valid == 0          → 全零 (n_points, 3)
      - 0 < valid < n_points → 前 valid 个保留，余下补零（不做 FPS）
      - valid >= n_points    → GPU FPS 采样 n_points 个（pytorch3d.sample_farthest_points
                              默认 random_start_point=True，每次起点随机，得到不同子集）

    Args:
        padded_pts:  (B, max_pts, 3)  float32, 在 GPU 上
        valid_count: (B,)             long, 每帧有效点数
        n_points:    目标采样点数（如 512）

    Returns:
        (B, n_points, 3) float32
    """
    assert padded_pts.dim() == 3 and padded_pts.shape[-1] == 3, (
        f"padded_pts shape={padded_pts.shape}, expected (B, max_pts, 3)"
    )
    assert valid_count.dim() == 1 and valid_count.shape[0] == padded_pts.shape[0], (
        f"valid_count shape={valid_count.shape}, expected ({padded_pts.shape[0]},)"
    )
    B, max_pts, _ = padded_pts.shape
    device = padded_pts.device
    out = torch.zeros(B, n_points, 3, device=device, dtype=padded_pts.dtype)

    if not (_PCD_HAS_PYTORCH3D and padded_pts.is_cuda):
        # 退化路径：纯 numpy 随机均匀采样
        pts_np = padded_pts.detach().cpu().numpy()
        vc_np  = valid_count.detach().cpu().numpy()
        rng = _pcd_rng
        out_np = np.zeros((B, n_points, 3), dtype=np.float32)
        for b in range(B):
            vc = int(vc_np[b])
            if vc == 0:
                continue
            if vc <= n_points:
                out_np[b, :vc] = pts_np[b, :vc]
                continue
            idx = rng.choice(vc, n_points, replace=False)
            out_np[b] = pts_np[b, idx]
        return torch.from_numpy(out_np).to(device)

    # ── GPU 批量 FPS（pytorch3d）─────────────────────────────────────
    # sample_farthest_points 的 lengths 参数会让每行只在前 lengths[i] 个点中采样
    # 但当 lengths[i] < K 时会出错，需要先按 valid 分两组处理
    valid_count = valid_count.to(device=device, dtype=torch.long)

    # 第一组：valid >= n_points → GPU FPS
    fps_mask = valid_count >= n_points
    if fps_mask.any():
        sub_pts  = padded_pts[fps_mask]                       # (B', max_pts, 3)
        sub_vc   = valid_count[fps_mask].clamp(max=max_pts)   # (B',)
        # pytorch3d 要求 lengths 是 int64
        sampled, _idx = _pcd_torch3d_ops.sample_farthest_points(
            points=sub_pts,
            lengths=sub_vc,
            K=n_points,
            random_start_point=True,
        )
        out[fps_mask] = sampled

    # 第二组：0 < valid < n_points → 前 vc 个保留 + 补零
    small_mask = (valid_count > 0) & (valid_count < n_points)
    if small_mask.any():
        small_idx_list = small_mask.nonzero(as_tuple=False).flatten().tolist()
        for b in small_idx_list:
            vc = int(valid_count[b].item())
            out[b, :vc] = padded_pts[b, :vc]

    return out


def extract_pointcloud_obs(raw_obs: dict, n_points: int) -> dict:
    """
    将 ManiSkill obs_mode='pointcloud' 的单步 obs 转为策略输入格式：
        {'point_cloud': (n_points, 3), 'agent_pos': (19,)}

    state = [qpos(9) + goal_pos(3) + tcp_pose(7)] = 19维，
    与训练 zarr 数据集中 data/state 的拼接顺序完全一致。

    Args:
        raw_obs:  ManiSkill 环境返回的 obs dict（obs_mode='pointcloud'）
        n_points: 目标点数（通常 512）
    """
    xyzw     = _to_numpy_pcd(raw_obs['pointcloud']['xyzw'])
    qpos     = _to_numpy_pcd(raw_obs['agent']['qpos'])
    goal_pos = _to_numpy_pcd(raw_obs['extra']['goal_pos'])
    tcp_pose = _to_numpy_pcd(raw_obs['extra']['tcp_pose'])

    seg = None
    if 'segmentation' in raw_obs['pointcloud']:
        seg = _to_numpy_pcd(raw_obs['pointcloud']['segmentation'])

    # GPU sim 并行 env 可能有 batch 维 (1, D)
    if qpos.ndim == 2:     qpos     = qpos[0]
    if goal_pos.ndim == 2: goal_pos = goal_pos[0]
    if tcp_pose.ndim == 2: tcp_pose = tcp_pose[0]

    agent_pos = agent_position(qpos, goal_pos, tcp_pose)

    return {
        'point_cloud': sample_pcd(xyzw, n_points, seg),   # (n_points, 3)
        'agent_pos':   agent_pos,                           # (19,)
    }


def stack_pointcloud_obs(obs_deque, n_obs_steps: int, device) -> dict:
    """
    将最近 n_obs_steps 帧堆叠为 PointCloudObsEncoder 所需的输入格式。

    Returns:
        {
            'point_cloud': Tensor (1, n_obs_steps, n_points, 3),
            'agent_pos':   Tensor (1, n_obs_steps, 19),
        }
    帧数不足时复制最早帧填充。
    """
    obs_list = list(obs_deque)[-n_obs_steps:]
    while len(obs_list) < n_obs_steps:
        obs_list.insert(0, obs_list[0])

    pcd_stack  = np.stack([o['point_cloud'] for o in obs_list])   # (T, n_pts, 3)
    apos_stack = np.stack([o['agent_pos']   for o in obs_list])   # (T, 19)

    return {
        'point_cloud': torch.from_numpy(pcd_stack).unsqueeze(0).to(device),   # (1, T, n_pts, 3)
        'agent_pos':   torch.from_numpy(apos_stack).unsqueeze(0).to(device),  # (1, T, 19)
    }


def _to_numpy_img(x):
    """统一把 torch.Tensor / np.ndarray 转 numpy。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_image_obs(raw_obs: dict,
                      camera: str = 'base_camera',
                      use_depth: bool = True,
                      depth_max: float = 5.0) -> dict:
    """
    将 ManiSkill obs_mode='rgbd' 的单步 obs 转为策略输入格式：
        {
            'image':     (C, H, W) float32,  C=4 (RGBD) 或 C=3 (RGB only)
            'agent_pos': (19,)     float32
        }

    state = [qpos(9) + goal_pos(3) + tcp_pose(7)] = 19 维，与训练 zarr 中
    data/state 的拼接顺序完全一致。
    """
    sensor = raw_obs['sensor_data'][camera]

    rgb = _to_numpy_img(sensor['rgb'])         # (1, H, W, 3) 或 (H, W, 3)
    if rgb.ndim == 4:
        rgb = rgb[0]                            # → (H, W, 3) uint8
    if use_depth:
        depth_raw = _to_numpy_img(sensor['depth'])  # (1, H, W, 1) 或 (H, W, 1)
        if depth_raw.ndim == 4:
            depth_raw = depth_raw[0]                # → (H, W, 1)
        image_chw = rgbd_to_chw(rgb, depth_raw, depth_max)
    else:
        image_chw = rgbd_to_chw(rgb)

    qpos     = _to_numpy_img(raw_obs['agent']['qpos'])
    goal_pos = _to_numpy_img(raw_obs['extra']['goal_pos'])
    tcp_pose = _to_numpy_img(raw_obs['extra']['tcp_pose'])

    if qpos.ndim == 2:     qpos     = qpos[0]
    if goal_pos.ndim == 2: goal_pos = goal_pos[0]
    if tcp_pose.ndim == 2: tcp_pose = tcp_pose[0]

    agent_pos = agent_position(qpos, goal_pos, tcp_pose)

    return {
        'image':     image_chw.astype(np.float32),
        'agent_pos': agent_pos,
    }


def stack_image_obs(obs_deque, n_obs_steps: int, device) -> dict:
    """
    将最近 n_obs_steps 帧堆叠为 ImageObsEncoder 所需的输入格式。

    Returns:
        {
            'image':     Tensor (1, n_obs_steps, C, H, W),
            'agent_pos': Tensor (1, n_obs_steps, 19),
        }
    帧数不足时复制最早帧填充。
    """
    obs_list = list(obs_deque)[-n_obs_steps:]
    while len(obs_list) < n_obs_steps:
        obs_list.insert(0, obs_list[0])

    img_stack  = np.stack([o['image']     for o in obs_list])   # (T, C, H, W)
    apos_stack = np.stack([o['agent_pos'] for o in obs_list])   # (T, 19)

    return {
        'image':     torch.from_numpy(img_stack).unsqueeze(0).to(device),    # (1, T, C, H, W)
        'agent_pos': torch.from_numpy(apos_stack).unsqueeze(0).to(device),   # (1, T, 19)
    }
