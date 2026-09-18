"""Paths and retention for one imitation training run."""

from datetime import datetime
from pathlib import Path
import re


def create_run_dir(checkpoint_dir, *, epochs, episodes, obs_mode, auto_mode=False):
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = 'pretrain_auto' if auto_mode else 'pretrain'
    name = f'{prefix}_epoch{epochs}_episodes{episodes}_{obs_mode}_{datetime.now():%m%d}'
    for index in range(1, 10000):
        run_dir = root / (name if index == 1 else f'{name}_{index}')
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f'无法为训练创建唯一的 checkpoint 目录: {root / name}')


def checkpoint_path(run_dir, *, epoch, episodes, obs_mode, auto_mode=False):
    prefix = 'pretrain_auto' if auto_mode else 'pretrain'
    return Path(run_dir) / f'{prefix}_epoch{epoch}_episodes{episodes}_{obs_mode}.pth'


def prune_checkpoints(run_dir, *, episodes, obs_mode, auto_mode=False, keep=3):
    if keep < 1:
        raise ValueError('checkpoint 保留数量必须至少为 1')
    prefix = 'pretrain_auto' if auto_mode else 'pretrain'
    pattern = re.compile(rf'^{prefix}_epoch(\d+)_episodes{episodes}_{re.escape(obs_mode)}\.pth$')
    checkpoints = []
    for path in Path(run_dir).iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            checkpoints.append((int(match.group(1)), path))
    for _, path in sorted(checkpoints, key=lambda item: item[0])[:-keep]:
        path.unlink()
