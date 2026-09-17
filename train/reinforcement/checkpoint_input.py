"""Resolve the imitation checkpoint before starting reinforcement training."""

from pathlib import Path

from train.common.bootstrap import PROJECT_ROOT


def resolve_pretrain_checkpoint(config, requested=None):
    checkpoint_dir = Path(config.pretrain_checkpoint_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    if requested:
        path = Path(requested).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.append(PROJECT_ROOT / path)
        # The imitation checkpoint directory moved from pretrain-checkpoints/.
        legacy_parent = PROJECT_ROOT / 'pretrain-checkpoints'
        if (path.parent == Path('pretrain-checkpoints')
                or path.parent == legacy_parent):
            candidates.append(checkpoint_dir / path.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"预训练 checkpoint 不存在: {requested}。"
            f"迁移后的目录: {checkpoint_dir}"
        )

    if config.resume_from:
        return None

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"预训练 checkpoint 目录不存在: {checkpoint_dir}")

    obs_mode = config.env_kwargs.obs_mode
    checkpoints = [
        path for path in checkpoint_dir.glob('*.pth')
        if 'failed' not in path.stem
        and (path.stem.endswith(f'_{obs_mode}')
             or (obs_mode == 'state' and not path.stem.endswith(('_pointcloud', '_image'))))
    ]
    if len(checkpoints) == 1:
        return checkpoints[0].resolve()
    if not checkpoints:
        raise FileNotFoundError(
            f"{checkpoint_dir} 中没有适用于 obs_mode={obs_mode!r} 的预训练 checkpoint；"
            "请通过 --pretrain_checkpoint 指定文件。"
        )
    raise ValueError(
        f"{checkpoint_dir} 中有多个适用于 obs_mode={obs_mode!r} 的预训练 checkpoint；"
        "请通过 --pretrain_checkpoint 指定一个文件。"
    )
