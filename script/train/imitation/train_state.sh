#!/usr/bin/env bash
# set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export WANDB_PROJECT=FlowNFT
export WANDB_ENTITY=545242836-zju

exec python -m train.imitation.main \
  --config=config/imitation/imitation_state.py \
  --gpu=1 \
  "$@"
