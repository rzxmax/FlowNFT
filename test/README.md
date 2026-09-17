# Checkpoint 评估

从仓库根目录运行。`evaluate_checkpoint.py` 加载 `train/` 生成的 `.pth` 权重，在 ManiSkill 中运行 episode，输出成功率与平均奖励。`--checkpoint` 可传预训练文件、强化训练的 `model.pth`，或包含 `model.pth` 的目录。

```bash
./venv_fm/bin/python test/evaluate_checkpoint.py \
  --checkpoint checkpoints/imitation/pretrain_epoch200_episodes500.pth \
  --num_episodes 50 --gpu 0

./venv_fm/bin/python test/evaluate_checkpoint.py \
  --checkpoint checkpoints/reinforcement/FlowNFT/checkpoints/checkpoint-epoch10 \
  --num_episodes 50 --gpu 0
```

默认使用 checkpoint 中的推理步数（旧权重默认为 10）；可用 `--num_inference_steps` 覆盖。可用 `--save_results results.json` 保存详细结果，`--render` 保存视频。成功率优先根据 ManiSkill 的 `info['success']` 计算。

FlowPolicy 的 `.ckpt` 格式使用另一入口：

```bash
./venv_fm/bin/python test/evaluate_flowpolicy.py \
  --checkpoint checkpoints/flowpolicy/pickcube_pointcloud-pickcube_pointcloud_flowpolicy-0312_seed0/epoch=0100.ckpt \
  --obs_mode pointcloud --num_episodes 50 --gpu 0
```

`test/test_training_structure.py` 与 `test/test_maniskill_replay_dataset.py` 是 pytest 回归测试，不会加载训练完成的模型做环境评估。
