#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

checkpoint="/home/lsx/rzx/flowNFT/checkpoints/imitation/state/pretrain_epoch200_episodes500_state_0918/pretrain_epoch100_episodes500_state.pth"
gpu_id="${GPU_ID:-1}"
if [[ ! -f "$checkpoint" ]]; then
  printf 'Pretraining checkpoint not found: %s\n' "$repo_root/$checkpoint" >&2
  exit 1
fi

exec ./venv_fm/bin/python -m train.reinforcement.main \
  --config=config/reinforcement/reinforcement_stepnft.py \
  --pretrain_checkpoint="$checkpoint" \
  --gpu="$gpu_id" \
  "$@"
