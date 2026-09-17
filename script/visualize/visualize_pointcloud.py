"""
可视化 ManiSkill PickCube-v1 点云，支持三种数据来源：
  - zarr  : 从已转换的 Zarr 文件读取
  - hdf5  : 从 pointcloud 模式的原始 HDF5 文件读取（仅 w>0 过滤）
  - env   : 从 ManiSkill 环境实时采样，同时展示【过滤前 vs 过滤后】对比图

用法示例：
  # 实时对比（推荐，无需预先生成HDF5）
  python visualize_pointcloud.py --source env --n_points 512 --out vis_compare.png

  # 从已转换的 Zarr 读取
  python visualize_pointcloud.py --source zarr \
      --path /home/lsx/rzx/flowNFT/data/pickcube_pcd_filtered.zarr

  # 从 pointcloud 模式 HDF5 读取（无语义过滤）
  python visualize_pointcloud.py --source hdf5 \
      --path /path/to/trajectory.pointcloud.h5 --n_points 512

可选参数：
  --out      输出图片路径（默认 pointcloud_vis.png）
  --ep_idx   查看第几个 episode（默认 0，即第一个）
  --seed     环境随机种子（env 模式，默认 0）
  --views    渲染视角数量，1~4（默认 3：正面/侧面/俯视）
  --n_points 采样点数（默认 512）
  --method   采样方法：fps 或 uniform（默认 fps）
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# PickCube-v1 的语义过滤配置（与 convert_pointcloud_hdf5_to_zarr.py 完全一致）
ROBOT_SEG_IDS = set(range(1, 16))   # Panda 机械臂 link（panda_link0 ~ panda_rightfinger_pad）
CUBE_SEG_ID   = 18                   # 红色目标物块
GOAL_SEG_ID   = 19                   # 目标位置球（hidden，实际不出现）
KEEP_SEG_IDS  = ROBOT_SEG_IDS | {CUBE_SEG_ID, GOAL_SEG_ID}
BG_SEG_IDS    = {16, 17}            # table-workspace=16, ground=17

SEG_LABEL = {
    **{i: ('robot', '#f5a623') for i in range(1, 16)},
    16: ('table',  '#4a90d9'),
    17: ('ground', '#7b68ee'),
    18: ('cube',   '#e74c3c'),
    19: ('goal',   '#2ecc71'),
}

# ──────────────────────────────────────────────────────────────────────────────
# 采样工具
# ──────────────────────────────────────────────────────────────────────────────

try:
    import pytorch3d.ops as torch3d_ops
    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False

import torch


def _sample(pts: np.ndarray, n: int, method: str) -> np.ndarray:
    """对点云 pts (N,3) 做降采样到 n 个点，与 mjpc_wrapper.point_cloud_sampling 完全一致。"""
    if n == 'all' or pts.shape[0] == 0:
        return pts
    if pts.shape[0] <= n:
        pad = np.zeros((n - pts.shape[0], 3), dtype=np.float32)
        return np.concatenate([pts, pad], axis=0)
    if method == 'uniform' or (method == 'fps' and not _HAS_PYTORCH3D):
        if method == 'fps' and not _HAS_PYTORCH3D:
            print("[警告] pytorch3d 未安装，自动切换为 uniform 采样")
        idx = np.random.choice(pts.shape[0], n, replace=False)
        return pts[idx]
    # fps — 与 mjpc_wrapper.point_cloud_sampling 完全一致
    pc_tensor = torch.from_numpy(pts).unsqueeze(0).cuda()   # (1, N, 3)
    k = torch.tensor([n]).cuda()
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=pc_tensor[..., :3], K=k)
    return pts[sampled_indices.squeeze(0).cpu().numpy()]


# ──────────────────────────────────────────────────────────────────────────────
# 数据读取
# ──────────────────────────────────────────────────────────────────────────────

def load_from_zarr(zarr_path: str, ep_idx: int = 0) -> np.ndarray:
    """从转换后的 Zarr 读取指定 episode 初始帧点云，返回 (N, 3) float32。"""
    import zarr
    root = zarr.open(zarr_path, mode='r')
    episode_ends = root['meta/episode_ends'][:]
    start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
    pcd = root['data/point_cloud'][start]
    return pcd[:, :3].astype(np.float32)


def load_from_hdf5(h5_path: str, ep_idx: int = 0,
                   n_points: int = 512, method: str = 'fps') -> np.ndarray:
    """从 pointcloud 模式 HDF5 读取初始帧，仅做 w>0 过滤（无语义过滤）。"""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith('traj_')],
            key=lambda x: int(x.split('_')[1])
        )
        if ep_idx >= len(traj_keys):
            raise IndexError(f"ep_idx={ep_idx} 超出范围，共 {len(traj_keys)} 条")
        traj = f[traj_keys[ep_idx]]
        xyzw = traj['obs/pointcloud/xyzw'][0]   # (N_raw, 4)

    valid_xyz = xyzw[xyzw[:, 3] > 0, :3].astype(np.float32)
    print(f"  原始有效点: {valid_xyz.shape[0]}")
    return _sample(valid_xyz, n_points, method)


def load_from_env(seed: int = 0, n_points: int = 512, method: str = 'fps'):
    """
    从 ManiSkill 环境实时采样，返回：
      pcd_raw      : (N, 3) 仅 w>0 过滤后的原始点云（模拟旧版本）
      pcd_filtered : (M, 3) 语义过滤后采样到 n_points 的点云（模拟新版本）
      seg_colors   : (N, 3) 原始点云按 seg_id 着色，用于对比图
      stats        : dict，各类别点数统计
    """
    import gymnasium as gym
    import torch
    import mani_skill.envs  # 注册 ManiSkill 环境到 gymnasium

    env = gym.make('PickCube-v1', obs_mode='pointcloud', num_envs=1, render_mode='sensors')
    obs, _ = env.reset(seed=seed)

    xyzw = obs['pointcloud']['xyzw'][0].cpu().numpy()      # (N_raw, 4)
    seg  = obs['pointcloud']['segmentation'][0].cpu().numpy().squeeze(-1)  # (N_raw,)
    env.close()

    # ── 旧版本：仅 w>0 ──────────────────────────────────────────────────────
    raw_mask = xyzw[:, 3] > 0
    raw_xyz  = xyzw[raw_mask, :3].astype(np.float32)
    raw_seg  = seg[raw_mask]

    # 按 seg_id 染色，方便区分各物体
    raw_colors = np.ones((raw_xyz.shape[0], 3), dtype=float) * 0.5
    for sid, (label, hexc) in SEG_LABEL.items():
        m = raw_seg == sid
        if m.any():
            r, g, b = int(hexc[1:3], 16)/255, int(hexc[3:5], 16)/255, int(hexc[5:7], 16)/255
            raw_colors[m] = [r, g, b]

    # ── 新版本：语义过滤 + n_points 采样 ────────────────────────────────────
    sem_mask     = np.isin(raw_seg, list(KEEP_SEG_IDS))
    filtered_xyz = raw_xyz[sem_mask]
    filtered_xyz = _sample(filtered_xyz, n_points, method)

    # 统计
    stats = {}
    for sid, (label, _) in SEG_LABEL.items():
        cnt = int((raw_seg == sid).sum())
        if cnt > 0:
            stats[label if sid not in (16, 17) else f'{label}(bg)'] = cnt

    print(f"  原始有效点: {raw_xyz.shape[0]}")
    print(f"  语义过滤后: {sem_mask.sum()} 点 → 采样到 {filtered_xyz.shape[0]} 点")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    return raw_xyz, filtered_xyz, raw_colors, stats


# ──────────────────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────────────────

VIEWS = [
    ("正面 (XZ)",  (15,  0)),
    ("侧面 (YZ)",  (15, 90)),
    ("俯视 (XY)",  (88,  0)),
    ("斜45°",      (30, 45)),
]


def colorize_by_height(pcd: np.ndarray) -> np.ndarray:
    z = pcd[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    return plt.cm.plasma(z_norm)[:, :3]


def _draw_pcd(axes_row, pcd: np.ndarray, colors: np.ndarray,
              title_prefix: str, n_views: int):
    """在一行 axes 中绘制多视角点云。"""
    point_size = max(1.0, 6000 / max(pcd.shape[0], 1))
    for i, (view_title, (elev, azim)) in enumerate(VIEWS[:n_views]):
        ax = axes_row[i]
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
                   c=colors, s=point_size, linewidths=0, alpha=0.9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title_prefix}\n{view_title}", color='#aaddff',
                     fontsize=9, pad=3)
        ax.tick_params(colors='#667', labelsize=7)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#223')
        ax.yaxis.pane.set_edgecolor('#223')
        ax.zaxis.pane.set_edgecolor('#223')
        ax.grid(True, color='#334', linewidth=0.3)
        r = max(np.abs(pcd).max() * 0.55, 0.2)
        cx, cy, cz = pcd.mean(axis=0)
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.set_zlim(cz - r, cz + r)


def visualize_single(pcd: np.ndarray, out_path: str, n_views: int = 3,
                     ep_idx: int = 0, source: str = ''):
    """单行可视化（zarr / hdf5 模式）。"""
    n_views = min(max(1, n_views), len(VIEWS))
    colors  = colorize_by_height(pcd)

    fig, axes = plt.subplots(1, n_views, subplot_kw={'projection': '3d'},
                             figsize=(5 * n_views, 5), facecolor='#1a1a2e')
    if n_views == 1:
        axes = [axes]
    fig.suptitle(
        f"Point Cloud — Episode {ep_idx}  |  {pcd.shape[0]} 点  |  来源: {source}",
        color='white', fontsize=12, y=1.01
    )
    _draw_pcd(axes, pcd, colors, '', n_views)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ 已保存到: {out_path}")


def visualize_compare(raw_xyz: np.ndarray, filtered_xyz: np.ndarray,
                      raw_colors: np.ndarray, stats: dict,
                      out_path: str, n_views: int = 3, seed: int = 0,
                      n_points: int = 512):
    """双行对比可视化（env 模式）：上行=过滤前（按语义着色），下行=过滤后（按高度着色）。"""
    n_views = min(max(1, n_views), len(VIEWS))

    # 图例文字（紧凑单行）
    stat_str = ' | '.join(f"{k}:{v}" for k, v in stats.items())
    legend_lines = []
    for sid, (label, hexc) in SEG_LABEL.items():
        is_bg = sid in BG_SEG_IDS
        tag = '[✗BG]' if is_bg else '[✓OK]'
        legend_lines.append(f"{tag}seg={sid:<2} {label:<12}{hexc}")
    # 两列排列
    half = (len(legend_lines) + 1) // 2
    legend_str = '    '.join(
        f"{legend_lines[i]:<32}{legend_lines[i + half] if i + half < len(legend_lines) else ''}"
        for i in range(half)
    )

    fig = plt.figure(figsize=(5 * n_views, 13), facecolor='#1a1a2e')

    # 用 GridSpec：顶部留一小行放标题/图例，下面两大行放点云
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, n_views, figure=fig,
                  height_ratios=[0.18, 1, 1],
                  hspace=0.35, wspace=0.15)

    # ── 标题行 ─────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = (
        f"ManiSkill PickCube-v1 点云对比  |  seed={seed}  n_points={n_points}\n"
        f"点数: {stat_str}\n\n"
        f"颜色图例（上行按 seg_id 着色）:\n"
        + '\n'.join(
            f"  {legend_lines[i]:<36}  {legend_lines[i+half] if i+half < len(legend_lines) else ''}"
            for i in range(half)
        )
    )
    ax_title.text(0.01, 0.99, title_text, transform=ax_title.transAxes,
                  color='white', fontsize=8.5, va='top', ha='left',
                  family='monospace',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d0d2a', alpha=0.8))

    # ── 上行：过滤前（按 seg_id 着色）───────────────────────────────────────
    axes_top = [fig.add_subplot(gs[1, i], projection='3d',
                                facecolor='#0d0d1a') for i in range(n_views)]
    _draw_pcd(axes_top, raw_xyz, raw_colors,
              f'【过滤前】{raw_xyz.shape[0]} 点 (seg_id 着色)', n_views)

    # ── 下行：过滤后（按高度着色）──────────────────────────────────────────
    filt_non_zero = filtered_xyz[np.any(filtered_xyz != 0, axis=1)]
    if filt_non_zero.shape[0] == 0:
        filt_non_zero = np.zeros((1, 3), dtype=np.float32)
    filt_colors = colorize_by_height(filt_non_zero)
    axes_bot = [fig.add_subplot(gs[2, i], projection='3d',
                                facecolor='#0d0d1a') for i in range(n_views)]
    _draw_pcd(axes_bot, filt_non_zero, filt_colors,
              f'【过滤后】robot+cube，采样 {n_points} 点 (高度着色)',
              n_views)

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ 已保存到: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='可视化 ManiSkill 点云（支持 env 实时对比、zarr、hdf5）'
    )
    parser.add_argument('--source', type=str, default='env',
                        choices=['env', 'zarr', 'hdf5'],
                        help='数据来源：env（实时对比，默认）/ zarr / hdf5')
    parser.add_argument('--path', type=str, default=None,
                        help='zarr 或 hdf5 模式下的文件路径（env 模式不需要）')
    parser.add_argument('--ep_idx', type=int, default=0,
                        help='查看第几个 episode（zarr/hdf5 模式，默认 0）')
    parser.add_argument('--seed', type=int, default=0,
                        help='环境随机种子（env 模式，默认 0）')
    parser.add_argument('--n_points', type=int, default=512,
                        help='采样点数（默认 512）')
    parser.add_argument('--method', type=str, default='fps',
                        choices=['fps', 'uniform'],
                        help='采样方法（默认 fps）')
    parser.add_argument('--views', type=int, default=3,
                        help='渲染视角数量 1~4（默认 3）')
    parser.add_argument('--out', type=str, default='pointcloud_vis.png',
                        help='输出图片路径（默认 pointcloud_vis.png）')
    args = parser.parse_args()

    if args.source == 'env':
        print(f"[实时采样] ManiSkill PickCube-v1  seed={args.seed}  n_points={args.n_points}")
        raw_xyz, filtered_xyz, raw_colors, stats = load_from_env(
            seed=args.seed, n_points=args.n_points, method=args.method)
        visualize_compare(raw_xyz, filtered_xyz, raw_colors, stats,
                          out_path=args.out, n_views=args.views,
                          seed=args.seed, n_points=args.n_points)

    elif args.source == 'zarr':
        if args.path is None:
            parser.error("--source zarr 需要提供 --path")
        print(f"[读取 Zarr] {args.path}  ep_idx={args.ep_idx}")
        pcd = load_from_zarr(args.path, ep_idx=args.ep_idx)
        pcd = pcd[np.any(pcd != 0, axis=1)]
        print(f"  有效点: {pcd.shape[0]}")
        visualize_single(pcd, out_path=args.out, n_views=args.views,
                         ep_idx=args.ep_idx, source='zarr')

    else:  # hdf5
        if args.path is None:
            parser.error("--source hdf5 需要提供 --path")
        print(f"[读取 HDF5] {args.path}  ep_idx={args.ep_idx}")
        pcd = load_from_hdf5(args.path, ep_idx=args.ep_idx,
                             n_points=args.n_points, method=args.method)
        pcd = pcd[np.any(pcd != 0, axis=1)]
        print(f"  有效点: {pcd.shape[0]}")
        visualize_single(pcd, out_path=args.out, n_views=args.views,
                         ep_idx=args.ep_idx, source='hdf5')


if __name__ == '__main__':
    main()
