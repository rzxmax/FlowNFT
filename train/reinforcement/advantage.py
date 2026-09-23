"""Extracted training components; function bodies preserve the legacy implementation."""

from train.common.runtime import (
    np,
)

def compute_episode_level_advantages(samples, num_train_timesteps, config):
    """
    基于完整轨迹的GRPO advantage计算

    支持两种 advantage 计算模式（由 config.stage.advantage_type 切换）：

    1. "group"（默认）：组内相对优势
       - 按GRPO组分组（组内相同初始状态）
       - 使用相对奖励的总和（参考RLinf）
       - Success Rate Filter（参考RLinf）：过滤全成功或全失败的组
       - 组内均值/方差归一化

    2. "single"：单条轨迹绝对优势
       - 不依赖组内对比，每条轨迹独立判定
       - 环境 success=True → +1
       - 失败 → -1
       - 不做 Success Rate Filter（绝对值 advantage 本身有学习信号）
    """
    advantage_type = getattr(config.stage, 'advantage_type', 'group')

    # ========== Single 模式：每条轨迹独立 ±1 ==========
    if advantage_type == 'single':
        advantage_scale = getattr(config.stage, 'advantage_scale', 1.0)

        success_count = 0
        for sample in samples:
            is_success = bool(sample["success"])
            sample["filtered"] = False
            sample["advantage"] = (1.0 if is_success else -1.0) * advantage_scale
            success_count += int(is_success)

        filter_stats = {
            "filtered_groups": 0,
            "filtered_samples": 0,
            "total_groups": config.episode.num_state_groups,
            "total_samples": len(samples),
        }
        return samples, filter_stats

    # ========== Group 模式（原有逻辑）==========
    if advantage_type != 'group':
        raise ValueError(
            f"Unknown advantage_type: {advantage_type!r}. "
            f"Expected 'group' or 'single'."
        )

    num_state_groups = config.episode.num_state_groups
    samples_per_group = config.episode.samples_per_group

    # 按组分配样本
    group_samples = [[] for _ in range(num_state_groups)]
    for sample in samples:
        group_id = sample["group_id"]
        group_samples[group_id].append(sample)

    # 统计过滤信息
    filtered_groups = 0
    filtered_samples = 0

    # 为每个样本计算advantage
    for group_id, group in enumerate(group_samples):
        if len(group) == 0:
            continue

        # ✅ Success Rate Filter（参考RLinf）
        # 检查组内是否全部成功或全部失败
        success_count = sum(1 for s in group if s["success"])
        all_success = (success_count == len(group))
        all_fail = (success_count == 0)

        # 如果全成功或全失败，过滤掉这个组
        if all_success or all_fail:
            filtered_groups += 1
            filtered_samples += len(group)
            # 标记这些样本为过滤状态
            for sample in group:
                sample["filtered"] = True
                sample["advantage"] = 0.0  # ✅ DiffusionNFT: 标量
            continue

        # ✅ 使用相对奖励的总和（已经消除累积效应）
        # 相对奖励 = 当前累积奖励 - 上一步累积奖励
        # 总和 = 最终累积奖励 - 初始累积奖励 = 任务完成度
        group_rewards = np.array([s["reward"] for s in group])
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-8

        # ✅ DiffusionNFT: 标量advantage（整条轨迹一个值）
        for i, sample in enumerate(group):
            sample["filtered"] = False
            total_reward = sample["reward"]
            advantage = (total_reward - group_mean) / group_std
            advantage *= config.stage.advantage_scale

            # ✅ 整条轨迹一个标量advantage（不再是列表）
            sample["advantage"] = advantage  # scalar float

    # ✅ 移除全局标准化（对齐ReinFlow）
    # ReinFlow只在组内进行标准化，不做全局标准化
    # 全局标准化会过度压缩advantages，削弱组间差异的学习信号

    # 返回样本和过滤统计信息
    filter_stats = {
        "filtered_groups": filtered_groups,
        "filtered_samples": filtered_samples,
        "total_groups": num_state_groups,
        "total_samples": len(samples),
    }

    return samples, filter_stats
