#!/usr/bin/env bash
# set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

checkpoint="${PRETRAIN_CHECKPOINT:-$repo_root/checkpoints/imitation/pretrain_epoch1000_episodes1000_image_0920/pretrain_epoch1000_episodes1000_image.pth}"
gpu_id="${GPU_ID:-1}"
if [[ ! -f "$checkpoint" ]]; then
  printf 'Image pretraining checkpoint not found: %s\n' "$checkpoint" >&2
  exit 1
fi

./venv_fm/bin/python -m train.reinforcement.main \
  --config=config/reinforcement/reinforcement_stepnft_image.py \
  --pretrain_checkpoint="$checkpoint" \
  --gpu=1 \
  "$@"
