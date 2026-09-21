"""Config-driven single-node launcher for imitation pretraining.

This module deliberately avoids importing torch, ManiSkill, or the trainer so
GPU visibility and the worker count are fixed before CUDA is initialized.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _flag_value(argv, name):
    prefix = f'--{name}='
    for index, argument in enumerate(argv):
        if argument.startswith(prefix):
            return argument[len(prefix):]
        if argument == f'--{name}' and index + 1 < len(argv):
            return argv[index + 1]
    return None


def load_config(config_argument):
    """Load one of this repository's Python config files without torch."""
    if not config_argument:
        raise ValueError('--config is required')
    config_path = Path(config_argument).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file() or config_path.suffix != '.py':
        raise FileNotFoundError(config_path)

    spec = importlib.util.spec_from_file_location(
        f'_flownft_launch_config_{config_path.stem}', config_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load config: {config_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'get_config'):
        raise ValueError(f'{config_path} does not define get_config()')
    return module.get_config()


def validate_gpu_ids(gpu_ids):
    """Return validated physical GPU IDs as a plain list."""
    values = list(gpu_ids)
    if not values:
        raise ValueError('config.pretrain.gpu_ids must not be empty')
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError('config.pretrain.gpu_ids must contain integers')
    if any(value < 0 for value in values):
        raise ValueError('config.pretrain.gpu_ids must contain non-negative IDs')
    if len(set(values)) != len(values):
        raise ValueError('config.pretrain.gpu_ids must not contain duplicates')
    return values


def build_launch_command(config, training_args, python=sys.executable):
    """Build an Accelerate command from the training config."""
    gpu_ids = validate_gpu_ids(config.pretrain.gpu_ids)
    precision = str(getattr(config.pretrain, 'mixed_precision', 'no'))
    if precision not in ('no', 'fp16', 'bf16'):
        raise ValueError("pretrain.mixed_precision must be 'no', 'fp16', or 'bf16'")
    return [
        python,
        '-m',
        'accelerate.commands.launch',
        '--multi_gpu',
        '--num_processes',
        str(len(gpu_ids)),
        '--num_machines',
        '1',
        '--gpu_ids',
        ','.join(map(str, gpu_ids)),
        '--mixed_precision',
        precision,
        '-m',
        'train.imitation.main',
        *training_args,
    ]


def main(argv=None):
    training_args = list(sys.argv[1:] if argv is None else argv)
    config = load_config(_flag_value(training_args, 'config'))
    gpu_ids = validate_gpu_ids(config.pretrain.gpu_ids)
    precision = str(getattr(config.pretrain, 'mixed_precision', 'no'))
    if precision not in ('no', 'fp16', 'bf16'):
        raise ValueError("pretrain.mixed_precision must be 'no', 'fp16', or 'bf16'")

    env = os.environ.copy()
    env['FLOWNFT_ACCELERATE_WORKER'] = '1'

    # Validation does not need a process group or CUDA device.
    if '--validate_only' in training_args:
        command = [sys.executable, '-m', 'train.imitation.main', *training_args]
    elif len(gpu_ids) == 1:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        command = [sys.executable, '-m', 'train.imitation.main', *training_args]
    else:
        # The config is the source of truth for physical IDs; do not let an
        # inherited shell mask silently reinterpret them.
        env.pop('CUDA_VISIBLE_DEVICES', None)
        command = build_launch_command(config, training_args)

    print(f'Launching imitation pretraining on physical GPUs {gpu_ids}', flush=True)
    print(f'Mixed precision: {precision}', flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True).returncode


if __name__ == '__main__':
    raise SystemExit(main())
