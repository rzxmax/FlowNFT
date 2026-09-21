"""Prepare the process before importing PyTorch or ManiSkill."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_gpu(argv=None):
    """Preserve the legacy ``--gpu`` selection before CUDA is imported."""
    # A launcher must decide visibility before importing the worker.  Never
    # collapse its device set back to GPU 0 inside every distributed process.
    if (os.environ.get('FLOWNFT_ACCELERATE_WORKER') == '1'
            or 'LOCAL_RANK' in os.environ
            or int(os.environ.get('WORLD_SIZE', '1')) > 1):
        return os.environ.get('LOCAL_RANK', '0')

    args = list(sys.argv[1:] if argv is None else argv)
    gpu_id = 0
    for index, arg in enumerate(args):
        if arg.startswith('--gpu='):
            gpu_id = int(arg.split('=', 1)[1])
            break
        if arg == '--gpu' and index + 1 < len(args):
            gpu_id = int(args[index + 1])
            break
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return gpu_id


def configure_import_paths():
    """Resolve project packages independently of the shell cwd."""
    sys.dont_write_bytecode = True
    for path in (
        PROJECT_ROOT,
        PROJECT_ROOT / 'flow_grpo',
        PROJECT_ROOT / 'flow_grpo' / 'diffusers_patch',
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
