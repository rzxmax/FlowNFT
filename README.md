# FlowNFT

针对 Flow 的强化学习算法的研究。

## 配置 `venv_fm` 训练/测试环境

下面的版本与 2026-09-20 本地可用环境对齐：Ubuntu 22.04
x86_64、Python 3.9.25、PyTorch 2.6.0 + CUDA 12.4、torchvision 0.21.0、
torchaudio 2.6.0 和 xformers 0.0.29.post3。`requirements.txt` 锁定了项目直接
依赖；仓库内的 `flow_grpo` 子模块锁定了本项目所需的机器人
pipeline 修改，不要用上游仓库的最新分支替换它。

### 1. 系统前置条件

- Linux x86_64（已验证 Ubuntu 22.04）。
- 可用的 `python3.9`，推荐精确使用 Python 3.9.25。
- Git、Git submodule、C/C++ 编译器和 `ffmpeg`。
- GPU 训练/环境评估需 NVIDIA GPU 及支持 CUDA 12.4 runtime 的驱动。
  PyTorch wheel 自带 CUDA runtime，无需为纯 Python 依赖额外安装 CUDA
  Toolkit；只有编译 CUDA 扩展时才需 `nvcc`。

如果是新 clone，先在仓库根目录取回固定版本的子模块：

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status
```

`git submodule status` 中 `flow_grpo` 应当位于父仓库记录的 commit，
输出行首不应是 `-` 或 `+`。

### 2. 创建并安装环境

始终从仓库根目录执行：

```bash
# 先确认不是 3.10/3.11 等其他 Python
python3.9 --version

python3.9 -m venv venv_fm
source venv_fm/bin/activate

# 对齐当前环境的打包工具，避免旧包构建行为变化
python -m pip install \
  pip==23.0.1 setuptools==58.1.0 wheel==0.45.1

# 先安装仓库锁定的本地 flow_grpo；不要 pip install GitHub 的 HEAD
python -m pip install -e ./flow_grpo

# 最后安装根项目的固定版本依赖，根项目的版本约束为准
python -m pip install -r requirements.txt
```

不要使用 `sudo pip`。以后进入项目时执行
`source venv_fm/bin/activate`，或在命令中显式使用
`./venv_fm/bin/python`。训练入口会自动将项目根目录和本地
`flow_grpo` 放入导入路径，不需要手动设置 `PYTHONPATH`。

### 3. 验证安装

先检查关键版本、本地源码位置和 CUDA：

```bash
python - <<'PY'
from pathlib import Path
import sys
import torch
import torchvision
import torchaudio
import xformers
import diffusers
import transformers
import accelerate
import mani_skill
import wandb
from train.common.bootstrap import configure_import_paths

configure_import_paths()
from flow_grpo import ema

project = Path.cwd().resolve()
assert sys.version_info[:3] == (3, 9, 25), sys.version
assert torch.__version__.startswith("2.6.0+cu124"), torch.__version__
assert torchvision.__version__.startswith("0.21.0+cu124"), torchvision.__version__
assert torchaudio.__version__.startswith("2.6.0+cu124"), torchaudio.__version__
assert xformers.__version__ == "0.0.29.post3", xformers.__version__
assert diffusers.__version__ == "0.33.1", diffusers.__version__
assert transformers.__version__ == "4.40.0", transformers.__version__
assert accelerate.__version__ == "1.4.0", accelerate.__version__
assert mani_skill.__version__ == "3.0.0b21", mani_skill.__version__
assert wandb.__version__ == "0.22.3", wandb.__version__
assert Path(ema.__file__).resolve().is_relative_to(project / "flow_grpo"), ema.__file__

print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "built for CUDA", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("flow_grpo:", ema.__file__)
print("project:", project)
PY

python -m pip check
python -m pytest -q test
python -m compileall -q train config dataset models test
python -m train.imitation.main \
  --config=config/imitation/imitation_state.py --validate_only
python -m train.reinforcement.main \
  --config=config/reinforcement/reinforcement_diffusionnft.py --validate_only
```

`--validate_only` 只检查导入和配置，不加载数据、不创建 ManiSkill
环境，也不执行 GPU 训练。要验证 GPU，`nvidia-smi` 必须正常，
且上面的 `CUDA available` 必须为 `True`。如果它为 `False`，先修复
NVIDIA 驱动/容器 GPU 映射，重装 Python 包通常无法解决。

### 4. 训练与评估

数据集路径由 `config/imitation/datasets_*.json` 管理；实际训练前需把
ManiSkill replay H5 放到这些清单所指的位置。示例命令：

```bash
# 状态模仿预训练
python -m train.imitation.main \
  --config=config/imitation/imitation_state.py --gpu=0

# DiffusionNFT 强化微调
python -m train.reinforcement.main \
  --config=config/reinforcement/reinforcement_diffusionnft.py \
  --pretrain_checkpoint=checkpoints/imitation/<checkpoint>.pth --gpu=0

# checkpoint 环境评估
python test/evaluate_checkpoint.py \
  --checkpoint checkpoints/imitation/<checkpoint>.pth \
  --num_episodes=50 --gpu=0
```

更多配置、数据清单和 checkpoint 说明见 `train/README.md`，评估参数见
`test/README.md`。W&B 只在需要上传训练指标时执行 `wandb login`，
API key 不要写入脚本或仓库。
