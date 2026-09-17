"""DiffusionNFT风格训练配置"""
import ml_collections
from config.robot_base import get_config as get_base_config


def get_config():
    config = get_base_config()
    
    # ========== DiffusionNFT专用参数 ==========
    
    # 三模型架构参数
    config.train.beta = 0.5  # Positive/negative prediction混合系数
    config.train.kl_weight = 0.01  # KL divergence loss权重
    config.train.decay_type = 1  # Old model EMA更新策略（0/1/2）
    
    # Old model 更新频率：
    #   "per_epoch" - 每个 epoch 结束后更新一次（对齐论文 Algorithm 1 Line 14，当前采用）
    #   "per_step"  - 每 20 个 global_step 更新一次（更频繁，适合 batch 数量多的场景）
    config.train.old_model_update_mode = "per_epoch"
    
    # Advantage处理参数
    config.train.adv_clip_max = 3.0  # Advantage clip范围

    # ── config.stage 补充（训练脚本从 config.stage 读取分组信息）──────────────
    # train_sd3_robot_grpo_rollout.py 第 645-646 行读取的是 config.stage，
    # 而 robot_base.py 将它们定义在 config.episode 下，此处做镜像同步
    config.stage.num_state_groups = config.episode.num_state_groups
    config.stage.samples_per_group = config.episode.samples_per_group
    # config.train.adv_mode = "positive_only"  # 可选："positive_only"/"negative_only"/"one_only"/"binary"

    # ── 两层时间折扣（区分相同最终奖励但步数不同的轨迹）──────────────────────────
    #
    # 第一层：轨迹级 γ^T 折扣（Trajectory-level Discount） 采样时
    #   作用：shaped_reward = raw_reward * gamma_traj^num_steps
    #   效果：同样成功（reward=5），83步→更高折扣后奖励，303步→更低折扣后奖励
    #   建议值：gamma_traj=0.999（温和）/ 0.99（激进区分）
    config.train.use_traj_gamma_discount = True   # ✅ 训练时开启第一层
    config.train.gamma_traj = 0.99               # 轨迹折扣系数（越小→对步数越敏感）
    #
    # 第二层：per-flow-call 时间折扣（Temporal Credit Assignment） 训练时
    #   作用：轨迹内越靠近成功的 flow call 获得越高的 advantage 权重
    #   公式：adv_i = base_adv * gamma_temporal^(num_steps - step_i)
    #   效果：早期的探索动作 advantage 被大幅打折，临近成功的动作获得高权重
    #   建议值：gamma_temporal=0.99（与第一层配合使用）
    config.train.use_flow_temporal_discount = False  
    config.train.gamma_temporal = 0.99               # per-flow-call 折扣系数
    
    # 批量大小（根据总flows数量调整以控制更新频率）
    # 更新次数 = 总flows / batch_size
    config.train.batch_size = 64  # 建议根据训练数据量调整
    
    return config

