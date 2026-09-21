#!/usr/bin/env bash
# set -euo pipefail

# Single-node RGB-D pretraining on physical GPUs 4, 5, and 6.
#
# Usage (from any directory):
#   ./script/train/imitation/train_image_multi_GPU.sh
#
# Extra training flags are forwarded to train.imitation.main, for example:
#   ./script/train/imitation/train_image_multi_GPU.sh --velocity_loss
#
# GPU selection and the number of DDP workers come from
# config/imitation/imitation_rgbd_multi_gpu.py. The effective global batch is:
#   batch_size_per_gpu (64) * GPU count (3) * accumulation steps (1) = 192

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

python_bin="$repo_root/venv_fm/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found: $python_bin" >&2
  exit 1
fi

export WANDB_PROJECT=FlowNFT
export WANDB_ENTITY=545242836-zju
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-flownft}"
mkdir -p "$MPLCONFIGDIR"

exec "$python_bin" -m train.imitation.launch \
  --config=config/imitation/imitation_rgbd_multi_gpu.py \
  "$@"
