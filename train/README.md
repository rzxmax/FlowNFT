# FlowNFT 训练代码结构

训练代码按训练阶段分离：

当前 ManiSkill 数据集、训练和 checkpoint 评估共用根目录的
`models/normalizer.py`。它保留了原 `diffusion_policy` 归一化器的
checkpoint `state_dict` 格式。`models/` 中的旧 Kitchen、CompoSuite 和
Transformer 示例仍使用根目录 `diffusion_policy/` 的其他模块。

```text
train/
├── common/          # 观测处理、时间步与 SDE/ODE 采样、checkpoint、运行环境
├── imitation/       # 专家示范上的普通 Flow/Diffusion 模仿训练
│   ├── launch.py     # 从 config 读取 GPU 列表并启动 Accelerate worker
│   ├── main.py       # 模仿训练 CLI
│   ├── trainer.py    # action / velocity 预训练
│   └── evaluator.py  # 预训练策略评估
└── reinforcement/   # ManiSkill episode-level 强化微调
    ├── main.py       # 强化训练 CLI
    ├── runner.py     # 环境、模型、优化器与 epoch 主循环
    ├── sampler.py    # 完整 episode 采样
    ├── advantage.py  # group / single advantage
    ├── evaluation.py
    ├── old_policy.py
    └── losses/
        ├── diffusionnft.py  # rollout loss
        └── stepnft.py       # 单步 NFT/DPO loss
```

`config/robot_base.py` 只包含两类训练共用的环境和采样参数；
模仿训练使用 `config/imitation/`，强化训练使用 `config/reinforcement/`。
启动训练时必须通过 `--config` 选择对应阶段的预设。每次调用
`get_config()` 都返回独立对象。

从仓库根目录运行：

```bash
# 低维状态模仿预训练
python -m train.imitation.main --config=config/imitation/imitation_state.py --gpu=0

# 配置驱动的单机多卡预训练（GPU 列表见对应 imitation config）
python -m train.imitation.launch --config=config/imitation/imitation_state.py

# 点云模仿预训练
python -m train.imitation.main --config=config/imitation/imitation_pointcloud.py --gpu=0

# 点云训练并上传指标到 545242836-zju/FlowNFT（首次使用需先 wandb login）
./script/train_pointcloud.sh --gpu=0

# RGB-D 模仿预训练
python -m train.imitation.main --config=config/imitation/imitation_rgbd.py --gpu=0

# Flow Matching 速度场监督（也可对上面的任一模仿配置使用）
python -m train.imitation.main --config=config/imitation/imitation_state.py --velocity_loss --gpu=0

# DiffusionNFT rollout 强化微调
python -m train.reinforcement.main \
  --config=config/reinforcement/reinforcement_diffusionnft.py \
  --pretrain_checkpoint=checkpoints/imitation/<checkpoint>.pth --gpu=0

# StepNFT 强化微调
python -m train.reinforcement.main \
  --config=config/reinforcement/reinforcement_stepnft.py \
  --pretrain_checkpoint=checkpoints/imitation/<checkpoint>.pth --gpu=0
```

### 单机多卡模仿预训练

模仿预训练使用 Accelerate + PyTorch DDP。先在所用的
`config/imitation/*.py` 中配置物理 GPU 编号和每卡 batch，例如：

```python
pretrain.gpu_ids = [0, 2, 4, 6]
pretrain.batch_size_per_gpu = 32
pretrain.gradient_accumulation_steps = 2
pretrain.mixed_precision = 'bf16'  # no / fp16 / bf16
```

然后统一通过 launcher 启动：

```bash
python -m train.imitation.launch \
  --config=config/imitation/imitation_pointcloud.py
```

有效全局 batch 为 `每卡 batch × GPU 数 × 梯度累积步数`。launcher 会在
导入 CUDA 前读取 `gpu_ids`；不要对多卡配置直接运行
`train.imitation.main`。旧的 `train.imitation.main --gpu=N` 仍可用于单卡
兼容模式。多卡实现、同步方式、风险和验收方案详见根目录
`MULTI_GPU_PRETRAINING.md`。

点云脚本会把训练指标发送到 W&B 的 `545242836-zju/FlowNFT`。首次运行前，
使用该账号的 API key 执行 `wandb login`；密钥无需写入脚本或仓库。
训练期间会记录 batch/epoch 损失、学习率，以及 Auto 模式下的评估结果。
模仿训练每 50 个 epoch 保存一次 checkpoint，每次运行在配置的
`pretrain.checkpoint_dir` 下创建独立目录，例如
`checkpoints/imitation/state/pretrain_epoch500_episodes1000_state_0918/`。
目录内的文件按实际 epoch 命名；每次保存成功后只保留该次运行最新的 3 个
`.pth` 文件。若训练在非 50 的整数倍 epoch 结束，也会额外保存最终权重。
同一天同配置重复运行时，目录名会追加 `_2`、`_3` 等序号。
评估或强化训练时，使用目录内 `.pth` 文件的完整路径。

数据与模型权重按用途保存在以下位置（从仓库根目录运行命令）：

```text
checkpoints/
├── imitation/       # 预训练 .pth
├── reinforcement/   # FlowNFT 强化训练与保留的历史权重
├── flowpolicy/      # 根项目 FlowPolicy 实验的 .ckpt
├── experiments/     # 其他实验的 .ckpt
└── examples/        # 示例模型权重
data/datasets/
├── state/episodes/  # 当前可用的已拆分状态轨迹
├── pointcloud/      # 旧点云 Zarr，模仿训练不再读取
├── image/           # 旧 RGB-D Zarr，模仿训练不再读取
└── examples/        # Kitchen 示例数据及原始归档
```

模仿训练的数据读取器位于 `dataset/`，直接从 ManiSkill replay 生成的 H5
按需读取轨迹。三个模仿预设的 `pretrain.dataset_path` 分别指向
`config/imitation/datasets_state.json`、`datasets_pointcloud.json` 和
`datasets_rgbd.json`。在对应清单的 `datasets` 数组中加入更多 H5，即可
按顺序加载多个数据源。例如点云清单：

```json
{
  "datasets": [
    "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5",
    "~/.maniskill/demos/OtherTask-v1/motionplanning/trajectory.pointcloud.pd_ee_delta_pos.physx_cpu.h5"
  ]
}
```

清单也可以直接是路径字符串数组。`~` 会展开为用户目录，相对路径以清单所在
目录为基准。`pretrain.num_episodes` 限制**每个 H5** 读取的成功 episode 数，
归一化统计由所有已加载轨迹共同计算。当前联合加载要求各 H5 的观测维度和
动作维度与所选预设一致；完整的多任务评估和任务条件输入尚未加入。
预训练的末尾窗口按 `action_horizon - 1` 补齐：观测重复最后一个有效观测，
补出的 `pd_ee_delta_pos` 动作前三维为零，第 4 维沿用最后的夹爪指令。
最初下载的 `trajectory.h5` 没有训练所需的观测，且动作控制模式不同，
不能直接作为这些配置的数据源。

例如状态向量 StepNFT 的预训练参数现在使用
`--pretrain_checkpoint=checkpoints/imitation/pretrain_epoch200_episodes500.pth`。
旧的 `pretrain-checkpoints/` 路径已不存在；强化训练入口会把
`pretrain-checkpoints/<文件名>.pth` 映射到 `checkpoints/imitation/` 下的同名文件。
其他不存在的显式路径会在创建环境前报错。独立 Git 仓库 `FlowPolicy/` 和 `flow_grpo/` 内部的
数据与模型资产不在本次迁移范围。

旧脚本 `train_sd3_robot_grpo.py` 仍保留为独立入口；新功能请使用上面的模块入口。
训练完成后的独立环境评估入口见 `test/evaluate_checkpoint.py`；FlowPolicy `.ckpt`
评估入口见 `test/evaluate_flowpolicy.py`。使用方法见 `test/README.md`。

训练核心函数从旧脚本机械提取，原有模型计算、loss 和 checkpoint 字段不变。
唯一的算法路径修正是 StepNFT 采样的动作维度兜底：当 pipeline 和 decoder 都未提供
`action_dim` 时，现在从环境动作空间读取，而不是引用未定义变量。

## 验证

```bash
./venv_fm/bin/python -m pytest -q test/test_training_structure.py
./venv_fm/bin/python -m compileall -q train config train_sd3_robot_grpo.py
python -m train.imitation.main --config=config/imitation/imitation_state.py --validate_only
python -m train.imitation.launch --config=config/imitation/imitation_state.py --validate_only
python -m train.reinforcement.main --config=config/reinforcement/reinforcement_diffusionnft.py --validate_only
```

CPU 测试覆盖配置预设、旧入口分发、点云过滤、advantage、时间步/EMA 系数以及
StepNFT SDE 采样形状。完整预训练和 ManiSkill GPU rollout 仍需 NVIDIA 驱动、GPU、
本地数据和完整的 `flow_grpo` 本地改动。当前环境检测不到 NVIDIA 驱动，不能在此
证明 GPU 端到端训练已经成功。
`--validate_only` 只验证导入和配置，不会创建环境、加载数据或运行训练。

注意：根仓库中的 `flow_grpo/` 仍是存在本地未提交改动的嵌套 Git 仓库。仅提交根
仓库并不能把其机器人 pipeline 代码一同带走；对外上传前必须另行整理该仓库。
