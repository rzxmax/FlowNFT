#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec ./venv_fm/bin/python test/evaluate_checkpoint.py \
  --checkpoint=/home/lsx/rzx/flowNFT/checkpoints/imitation/state/pretrain_epoch200_episodes500_state_0918/pretrain_epoch50_episodes500_state.pth \
  --num_episodes=50 \
  --gpu=1 \
  "$@"
