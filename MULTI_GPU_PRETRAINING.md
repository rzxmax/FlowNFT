# FlowNFT 预训练多卡部署可行性与实施方案

> 实施状态：文档中的 Accelerate + DDP 第一阶段方案已经部署到
> `train/imitation`。当前入口为 `python -m train.imitation.launch --config=...`；
> DeepSpeed/FSDP 和多机训练仍不在本次范围内。

## 1. 结论

当前项目的模仿预训练（`train/imitation`）**可以改造成单机多卡训练**。建议采用项目已经固定依赖的 **Hugging Face Accelerate 1.4.0 + PyTorch DistributedDataParallel（DDP）**，第一阶段不要直接上 DeepSpeed，也不要使用 `torch.nn.DataParallel`。

推荐目标形态：

- 在模仿训练配置中设置物理 GPU 编号，例如 `pretrain.gpu_ids = [1, 3, 5, 7]`；
- 统一通过一个轻量 launcher 读取该配置，再启动相同数量的 Accelerate worker；
- 每张 GPU 对应一个进程，数据集由 Accelerate/DDP 自动分片，梯度在反向传播时同步；
- 只有主进程创建运行目录、写 checkpoint、记录 W&B、显示完整进度条和执行 Auto 模式评估；
- 保留单卡模式以及现有 checkpoint 的字段和下游加载兼容性。

这项改造可行，但不是只替换启动命令即可完成。当前预训练的 GPU 可见性、DataLoader、EMA、checkpoint、W&B 和 Auto 评估都按单进程编写，需要一起处理。

## 2. 当前代码审查结果

### 2.1 当前为什么只能使用一张卡

1. `train/imitation/main.py` 在导入训练器前调用 `configure_gpu()`。
2. `train/common/bootstrap.py::configure_gpu()` 只识别单个 `--gpu`，没有传入时默认执行：

   ```python
   os.environ['CUDA_VISIBLE_DEVICES'] = "0"
   ```

   这会覆盖外部 launcher 提供的多卡可见性。即使执行 `accelerate launch --gpu_ids 1,3 ...`，worker 也会再次只看见物理 GPU 0。
3. `train/imitation/trainer.py` 固定使用 `torch.device("cuda:0")`，没有使用进程的 `LOCAL_RANK` 或 `Accelerator.device`。
4. 模型、优化器和 DataLoader 未经过 DDP 或 `Accelerator.prepare()`，没有梯度同步与数据分片。
5. 所有进程若直接运行当前代码，会同时创建同名目录、保存文件、写 W&B 和执行评估，产生竞争与重复记录。

因此，当前的 `--gpu=0` 是“把一张物理卡映射为进程内的 `cuda:0`”，不是分布式训练参数。

### 2.2 已有的有利条件

- `requirements.txt` 已固定 `torch==2.6.0`、`accelerate==1.4.0` 和 `deepspeed==0.16.4`，第一阶段无需引入新的训练框架依赖。
- `train/reinforcement/runner.py` 已经使用 `Accelerator`、`accelerator.prepare()`、主进程判断和分布式指标收集，可复用其工程模式。
- 三种预训练模型（state、pointcloud、image）都是常规 PyTorch `nn.Module`，其前向计算可被 DDP 包装。
- ManiSkill replay 数据集是 map-style `Dataset`，H5 文件句柄在每个 DataLoader worker 内惰性打开，并通过 PID 隔离；适合多进程读取。
- 预训练主要是标准 loss、反向传播、AdamW 和 EMA，没有无法分布式化的自定义 autograd 操作。

### 2.3 顺便发现的配置一致性问题

这些不是多卡不可行的原因，但实施时应一并对齐，否则配置看似生效、实际不生效：

- `pretrain.gradient_accumulation_steps` 已在配置中定义，当前训练循环没有使用；
- `pretrain.max_grad_norm` 已定义，当前实际固定使用 `1.0`；
- `pretrain.adam_beta1`、`adam_beta2`、`adam_eps` 已定义，当前 AdamW 没有读取它们；
- `ema_update_freq` 被读取并打印，但当前每个 batch 都更新 EMA；
- 当前没有在预训练入口调用 `set_seed(config.seed, device_specific=True)`。

多卡改造应让这些字段真正参与训练，防止批大小或进程数变化时产生隐式行为差异。

## 3. 框架选择

| 方案 | 是否推荐 | 原因 |
| --- | --- | --- |
| Accelerate + DDP | **推荐，第一阶段采用** | 项目已安装且强化训练已有范例；改动适中；适合单机 2～8 卡的数据并行；checkpoint 易保持兼容 |
| 原生 `torchrun` + DDP | 可行 | 控制更直接，但需要自行处理 rank、设备、DistributedSampler、同步、日志和混合精度，重复造轮子较多 |
| `torch.nn.DataParallel` | 不推荐 | 单进程瓶颈明显、负载不均、扩展性和错误隔离较差，PyTorch 多卡训练通常优先 DDP |
| Accelerate + DeepSpeed ZeRO | 第二阶段按需采用 | 适合单卡显存放不下模型/优化器状态；当前模型规模未证明需要 ZeRO，会先增加 checkpoint 与恢复复杂度 |
| FSDP | 暂不推荐 | 更适合超大模型；当前收益可能小于工程复杂度，并会改变保存/恢复方式 |

DDP 的主要收益是吞吐量，而不是把单张卡显存简单相加。每张卡仍保留完整模型副本；每卡显存主要由该卡的 local batch 决定。如果单卡连模型都放不下，再评估 DeepSpeed ZeRO/FSDP。

## 4. 推荐配置设计

在 `config/imitation/base.py` 的 `pretrain` 下增加以下字段，三个 observation preset 默认继承：

```python
# 使用物理 GPU 1、3、5、7。顺序决定 rank 到物理卡的映射。
pretrain.gpu_ids = [1, 3, 5, 7]

# batch_size 明确定义为每张卡的 batch，避免扩卡时含义模糊。
pretrain.batch_size_per_gpu = 64
pretrain.gradient_accumulation_steps = 1

# no / fp16 / bf16；先以 no 做数值等价验收，再启用混合精度。
pretrain.mixed_precision = 'no'

# DDP 下某个可训练参数若在某条分支完全不参与 loss，才需设为 True。
# 当前按 loss_type 构建 optimizer，预期可以保持 False，性能更好。
pretrain.find_unused_parameters = False

# DataLoader 参数。总 worker 数约为进程数乘以该值。
pretrain.num_workers = 4
pretrain.pin_memory = True
pretrain.persistent_workers = True

# 单机分布式后端。NVIDIA GPU 使用 nccl。
pretrain.distributed_backend = 'nccl'
```

建议废弃模仿预训练的单值 `--gpu` 语义，或者只把它保留为向后兼容的单卡 CLI 覆盖项。配置应是 GPU 集合的唯一默认来源，避免同时存在 `--gpu`、`CUDA_VISIBLE_DEVICES` 和 `gpu_ids` 三套相互覆盖的规则。

批大小约定必须写清楚：

```text
effective_global_batch
  = batch_size_per_gpu × GPU 数量 × gradient_accumulation_steps
```

例如 4 卡、每卡 32、累积 2 步，全局有效 batch 为 256。若希望从单卡迁移后保持算法条件不变，应保持 effective global batch 不变；若随 GPU 数线性放大全局 batch，再单独验证是否需要按比例调整学习率。

## 5. 为什么需要配置驱动的 launcher

Accelerate 在训练脚本执行前就必须知道 `--gpu_ids` 和 `--num_processes`，但普通的 `Accelerator()` 只能在 worker 内读取训练配置，届时已经无法改变进程数。因此需要将“读取配置并启动进程”和“实际训练”分成两层。

建议新增 `train/imitation/launch.py`，职责严格限制为：

1. 在**不导入 torch、trainer、ManiSkill** 的前提下读取 `--config`；
2. 校验 `pretrain.gpu_ids`：非空、为非负整数、无重复；
3. 把列表转换为 Accelerate 的 `--gpu_ids 1,3,5,7`；
4. 设置 `--num_processes 4`、`--num_machines 1` 和配置中的 mixed precision；
5. 使用参数列表启动 `python -m accelerate.commands.launch ... -m train.imitation.main`；
6. 传递一个内部 worker 标志，避免训练入口再次递归启动 launcher。

不要用 shell 字符串拼接启动命令；使用 `subprocess.run([...], check=True)` 或 `os.execv(...)` 的参数列表形式。

预期用户命令只有一条：

```bash
./venv_fm/bin/python -m train.imitation.launch \
  --config=config/imitation/imitation_state.py
```

例如配置为 `[1, 3]` 后，进程内映射应为：

| distributed rank | `Accelerator.device` | 物理 GPU |
| ---: | --- | ---: |
| 0 | `cuda:0` | 1 |
| 1 | `cuda:1` | 3 |

日志中应同时打印 rank、local rank、进程内 CUDA 编号与配置的物理 GPU 编号，避免排障时混淆。

## 6. 已实施的代码位置

以下启动、DDP、日志、checkpoint 和 Auto 评估改造已经实现；条目同时保留为
后续维护时的行为约束。

### 6.1 `train/common/bootstrap.py`

- 不允许 worker 无条件把 `CUDA_VISIBLE_DEVICES` 改为 `0`；
- 如果检测到 `LOCAL_RANK`/`WORLD_SIZE`，必须保留 launcher 设置的设备环境；
- 单卡旧入口才可以继续处理 `--gpu`；
- 更稳妥的最终形态是由 `launch.py` 统一管理 GPU 可见性，worker 不再写该环境变量。

### 6.2 `train/imitation/main.py`

- 避免在分布式环境建立前导入会初始化 CUDA 的模块；
- 不要让每个 rank 分别执行 `wandb.init()`；W&B 初始化移到拥有 `Accelerator` 的训练器，并限定主进程；
- 保留 `--validate_only`，并增加对 `gpu_ids`、mixed precision、batch 约定的静态校验；
- 清晰区分 launcher 模式和内部 worker 模式。

### 6.3 `train/imitation/trainer.py`

创建 Accelerator：

```python
accelerator = Accelerator(
    mixed_precision=config.pretrain.mixed_precision,
    gradient_accumulation_steps=config.pretrain.gradient_accumulation_steps,
)
device = accelerator.device
set_seed(config.seed, device_specific=True)
```

然后完成以下调整：

1. DataLoader 保持 `shuffle=True`，交给 `accelerator.prepare()` 创建分布式分片；不要再额外叠加一个 `DistributedSampler`。
2. 按实际存在的模块组装 prepare 参数：
   - state + velocity：UNet、optimizer、DataLoader、可选 scheduler；
   - state + action：再加入 ActionDecoder；
   - pointcloud/image：再加入 ObsEncoder。
3. prepare 后把包装后的 UNet/Decoder 重新挂到 `pipeline`，否则 pipeline 可能继续引用未包装模型。
4. 用 `with accelerator.accumulate(...):` 包住前向和反向；用 `accelerator.backward(loss)` 替代 `loss.backward()`。
5. 只在 `accelerator.sync_gradients` 时裁剪梯度，并使用 `accelerator.clip_grad_norm_()`；最大范数来自 `pretrain.max_grad_norm`。
6. scheduler 的 total steps 按**每个 rank 分片后的 DataLoader 长度和 optimizer update 次数**计算，不要沿用单卡 batch 数直接乘 epoch。
7. epoch loss 不能只取 rank 0 的本地均值。每个 rank 累计 `loss_sum` 和 `sample_count`，经 `accelerator.reduce(..., reduction='sum')` 后计算全局加权均值。最后一个不完整 batch 使简单的“各 rank 均值再平均”可能有偏差。
8. tqdm、普通日志、W&B batch/epoch 指标只在主进程输出；必要的错误信息仍允许各 rank 输出。

模型 prepare 可采用动态列表，也可以按 obs/loss 分支明确展开。后者代码略长但更容易审查，不建议依赖模型恰好为 `None` 的隐式处理。

### 6.4 EMA

最稳妥的第一版是每个 rank 都维护相同的 EMA：DDP 保证优化后的模型参数一致，因此各 rank 在相同 optimizer step 后更新 EMA，结果也应一致。更新只发生在 `accelerator.sync_gradients` 为真时，并落实 `ema_update_freq`。

保存和评估时使用：

```python
raw_unet = accelerator.unwrap_model(action_unet)
```

不能直接假设包装后模型的 `state_dict` 与当前 checkpoint key 完全一致，否则可能写入 `module.` 前缀。

当前 Auto 评估会用 `ema.copy_to()` 覆盖训练模型，若评估后继续训练，没有显式恢复原参数。多卡下这还会导致主 rank 与其他 rank 参数不一致。实施时必须选择一种正确做法：

- 推荐：所有 rank 暂存当前参数、同步拷入 EMA；仅主 rank 评估；之后所有 rank 恢复参数；
- 或在主 rank 构建独立的评估模型副本，不触碰 DDP 训练模型。

后者更隔离，但会增加主卡显存占用。

### 6.5 checkpoint 与运行目录

- `create_run_dir()` 只允许主进程执行，然后将路径广播给其他 rank；
- `torch.save()`、`os.replace()` 和 prune 只允许主进程执行；
- 保存前使用 `accelerator.unwrap_model()`；
- checkpoint 对外字段保持不变，尤其是：
  `model_state_dict`、`action_unet_state_dict`、`action_decoder_state_dict`、
  `obs_encoder_state_dict`、normalizer、encoder config 和训练元数据；
- 增加非破坏性的分布式元数据，例如 `world_size`、`gpu_ids`、
  `batch_size_per_gpu`、`effective_global_batch`、`mixed_precision`；
- DDP 下 AdamW 状态在各 rank 一致，主进程保存完整 optimizer state 即可；若未来切到 ZeRO，必须改用 Accelerate/DeepSpeed 对应的 state 保存与恢复流程，不能照搬这里。

保存前后加 `accelerator.wait_for_everyone()`，避免其他 rank 在文件尚未完成时进入依赖该状态的阶段。

### 6.6 Auto 模式评估与停止信号

ManiSkill 环境评估先保持只在主进程执行，其他 rank 在 barrier 等待。主进程得到 `training_stopped` 后需要广播布尔值，保证所有 rank 在同一个 epoch 退出。

正确时序：

```text
所有 rank 完成本 epoch
        ↓ barrier
主 rank 使用 EMA 权重评估；其他 rank 等待
        ↓
主 rank 生成 stop / continue 决策
        ↓ broadcast
所有 rank 恢复训练权重并一起退出或继续
```

不要让非主 rank 提前开始下一 epoch，否则会在下一次梯度 collective 处死锁。

### 6.7 W&B

- 只在 `accelerator.is_main_process` 初始化和写入一个 run；
- 推荐使用 `accelerator.init_trackers()`，也可以保留直接 W&B API，但必须有主进程守卫；
- 配置中记录物理 `gpu_ids`、world size 和 effective global batch；
- 不要在 launcher 父进程创建 W&B run。

### 6.8 DataLoader 与 HDF5

当前 H5 惰性打开设计适合 DDP，但 8 GPU × 每进程 4 workers 会产生最多约 32 个 DataLoader worker，并对同一批 H5 发起并发读取。部署时应监控：

- CPU 利用率与内存；
- 磁盘吞吐和 I/O wait；
- HDF5 文件所在介质是否为本地 SSD/NVMe；
- GPU 是否因为数据读取而长期空闲。

如果 I/O 成为瓶颈，先降低每进程 `num_workers` 到 1～2，或将数据放到本地高速盘；不要默认认为 worker 越多越快。normalizer 统计目前由每个 rank 分别扫描完整数据集，会造成重复 I/O，后续可优化为主 rank 计算后广播 state dict，第一版至少要保证各 rank 得到完全相同的 normalizer。

## 7. 学习率、随机性和数值等价

### 7.1 学习率策略

首轮迁移建议保持原学习率，先比较单卡与多卡的 loss 曲线。只有在 effective global batch 随 GPU 数增大且训练稳定后，再试验线性缩放学习率。不要把“GPU 数翻倍”与“学习率必须翻倍”绑定为一次改动。

### 7.2 每个 rank 的随机数

velocity loss 中的 `torch.randn_like(x1)` 和随机 `t`、点云 FPS 起点、DataLoader shuffle 都包含随机性。应调用：

```python
set_seed(config.seed, device_specific=True)
```

这样各 rank 可复现但不会生成完全相同的噪声序列。DataLoader 的 epoch 变化应由 Accelerate 分片器正确推进。

### 7.3 混合精度

实施顺序建议：

1. `mixed_precision='no'` 验证 DDP 正确性；
2. 支持 BF16 的 GPU 优先测试 `bf16`；
3. 其他 GPU 测试 `fp16`，关注 NaN/Inf、梯度范数和 loss 是否异常；
4. 混合精度的 checkpoint 仍保存可正常加载的模型权重，不应要求评估端必须使用同一精度。

## 8. 预期启动方式

配置示例：

```python
# config/imitation/imitation_pointcloud.py
pretrain.gpu_ids = [0, 2, 4, 6]
pretrain.batch_size_per_gpu = 16
pretrain.gradient_accumulation_steps = 2
pretrain.mixed_precision = 'bf16'
```

启动：

```bash
./venv_fm/bin/python -m train.imitation.launch \
  --config=config/imitation/imitation_pointcloud.py
```

可选的诊断输出应类似：

```text
physical gpu_ids: [0, 2, 4, 6]
world_size: 4
rank 0 -> cuda:0 -> physical GPU 0
rank 1 -> cuda:1 -> physical GPU 2
rank 2 -> cuda:2 -> physical GPU 4
rank 3 -> cuda:3 -> physical GPU 6
per-device batch: 16
gradient accumulation: 2
effective global batch: 128
```

用户不应再需要手工先执行 `export CUDA_VISIBLE_DEVICES=...`。如确实允许外部环境变量覆盖配置，必须明确优先级并在冲突时直接报错，不能静默选择另一组 GPU。

## 9. 验证计划与验收标准

### 9.1 CPU/静态测试

新增测试至少覆盖：

- `gpu_ids=[]`、重复编号、负数、非整数时配置校验失败；
- `[3]` 生成 1 个进程，多卡列表生成对应 `num_processes` 和逗号分隔 GPU 参数；
- launcher 参数透传且不会递归启动；
- 分布式 worker 不覆盖已有 `CUDA_VISIBLE_DEVICES`；
- `--validate_only` 不启动训练进程、不加载数据；
- checkpoint 的旧字段保持不变且不带 `module.` 前缀；
- global batch 和 scheduler update 数计算正确。

### 9.2 单机 GPU 冒烟测试

使用很小的数据和 epoch 数，依次执行：

1. 配置 `[0]`：确认与旧单卡入口的前几个 loss 接近；
2. 配置 `[0, 1]`：确认两个进程分别占用一张卡；
3. 三种 obs mode 分别跑通 velocity/action 中实际使用的组合；
4. 检查每 epoch 消费的全局样本数，没有完整重复训练整个数据集；
5. checkpoint 只生成一份，W&B 只生成一个 run；
6. checkpoint 可被 `test/evaluate_checkpoint.py` 和强化训练入口加载；
7. 开启 Auto 模式，确认评估期间无 collective 死锁，并且 stop 决策在所有 rank 同步生效。

### 9.3 正确性判据

- 所有 rank 的模型参数在 optimizer step 后一致；
- 一个 epoch 的全局样本覆盖与单卡语义一致；
- 全局 epoch loss 是按样本加权的聚合值；
- 固定 effective global batch、精度与种子时，单卡/多卡 loss 趋势应接近；浮点归约顺序不同会造成微小差异，不要求逐 bit 一致；
- 保存的权重 key 不出现 DDP 的 `module.` 前缀；
- 非主 rank 不创建目录、不保存文件、不写 W&B、不运行环境评估。

### 9.4 性能判据

记录单卡与 2/4/8 卡的：

- samples/s；
- 每 epoch 时间；
- 各卡显存峰值和 GPU utilization；
- CPU、内存、磁盘吞吐和 I/O wait；
- DDP 通信占比。

小模型或 state 模式可能因通信与 HDF5 I/O 导致扩展效率有限；image/pointcloud 或 action rollout 计算更重，通常更可能从多卡获得明显吞吐收益。是否“值得”最终应以 samples/s 和达到相同验证质量所需的总时间判断，而不只看 GPU utilization。

## 10. 实施顺序

建议拆成四个可独立验收的阶段：

1. **启动与兼容层**：增加配置字段和 launcher，修复 GPU 可见性覆盖，单卡新入口跑通；
2. **DDP 训练核心**：Accelerator、prepare、数据分片、反向传播、梯度累积、全局 loss 与 scheduler；
3. **副作用收口**：EMA、checkpoint、W&B、进度条、运行目录全部限定主进程并正确 unwrap；
4. **Auto 评估与性能优化**：barrier/广播停止信号、normalizer 广播、混合精度及 HDF5 吞吐调优。

每个阶段都先跑单卡兼容测试，再跑两卡冒烟测试。不要等所有代码一次完成后才排查分布式死锁。

## 11. 风险与边界

- **最大风险是进程同步错误**：某个 rank 因评估、异常 batch 或提前 break 少进入一次 backward/collective，其他 rank 会一直等待。
- **Auto 评估必须恢复训练权重**：当前 EMA 覆盖行为在多卡下会造成 rank 参数分叉，必须修复后才能启用 Auto 多卡。
- **全局 batch 改变会改变训练动力学**：吞吐提升不等于相同 epoch 数能得到相同效果。
- **HDF5 可能成为瓶颈**：多卡增加读进程，不保证线性加速。
- **GPU 编号必须在启动前确定**：不能在 worker 已创建后靠修改 config 改变 world size。
- **本方案只覆盖单机多卡预训练**：多机训练还需要 rendezvous 地址、machine rank、共享 checkpoint 路径和跨节点数据可见性，不应与第一版同时上线。
- **强化训练不在本次改造范围**：它虽已使用 Accelerate，但有自己的环境采样与同步逻辑；本方案只针对 `train/imitation` 预训练。

## 12. 最终建议

项目具备部署多卡预训练的代码和依赖基础，建议按“配置驱动 launcher + Accelerate DDP”实施。第一版目标应是：单机任意 GPU 子集可选、三种观测模式可运行、单卡兼容、checkpoint 格式不变、Auto 模式无死锁。完成 DDP 的正确性和吞吐验证后，只有在显存确实成为瓶颈时，再增加 DeepSpeed ZeRO-2，而不是一开始同时引入两套复杂度。
