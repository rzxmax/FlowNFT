#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

./venv_fm/bin/python test/evaluate_checkpoint.py \
  --checkpoint="$repo_root/checkpoints/imitation/pretrain_epoch200_episodes1000_image_0920/pretrain_epoch200_episodes1000_image.pth" \
  --num_episodes=5 \
  --gpu=1 \
  --render \
  "$@"
