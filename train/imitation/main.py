"""Entry point for demonstration-based Flow/Diffusion pretraining."""

import os

from train.common.bootstrap import configure_gpu

configure_gpu()

from absl import app

from train.common.cli import FLAGS
from train.imitation.trainer import pretrain_model


def run_imitation(config, gpu_id, *, auto=False, action_loss=False, velocity_loss=False):
    """Run imitation training while retaining the old loss flag semantics."""
    if action_loss and velocity_loss:
        raise ValueError('--action_loss and --velocity_loss are mutually exclusive')
    if auto:
        config.pretrain.auto_mode = True
    if action_loss:
        config.pretrain.loss_type = 'action'
    elif velocity_loss:
        config.pretrain.loss_type = 'velocity'
    # W&B is initialized inside ``pretrain_model`` after Accelerator knows
    # which process is the global main process.
    return pretrain_model(config, gpu_id)


def main(_):
    if FLAGS.config is None or not hasattr(FLAGS.config, 'pretrain'):
        raise ValueError('Use --config=config/imitation/imitation_state.py (or another imitation preset)')
    if FLAGS.validate_only:
        from train.imitation.launch import validate_gpu_ids

        if FLAGS.action_loss and FLAGS.velocity_loss:
            raise ValueError('--action_loss and --velocity_loss are mutually exclusive')
        mode = FLAGS.config.pretrain.obs_mode
        loss = ('velocity' if FLAGS.velocity_loss else
                'action' if FLAGS.action_loss else FLAGS.config.pretrain.loss_type)
        if mode not in ('state', 'pointcloud', 'image'):
            raise ValueError(f'Unsupported imitation observation mode: {mode!r}')
        if loss not in ('action', 'velocity'):
            raise ValueError(f'Unsupported imitation loss: {loss!r}')
        gpu_ids = validate_gpu_ids(FLAGS.config.pretrain.gpu_ids)
        precision = str(getattr(FLAGS.config.pretrain, 'mixed_precision', 'no'))
        if precision not in ('no', 'fp16', 'bf16'):
            raise ValueError(f'Unsupported mixed precision: {precision!r}')
        print(
            f'Imitation configuration OK: obs_mode={mode}, loss_type={loss}, '
            f'gpu_ids={gpu_ids}, mixed_precision={precision}'
        )
        return
    configured_gpu_ids = list(getattr(FLAGS.config.pretrain, 'gpu_ids', [FLAGS.gpu]))
    if (len(configured_gpu_ids) > 1
            and os.environ.get('FLOWNFT_ACCELERATE_WORKER') != '1'
            and int(os.environ.get('WORLD_SIZE', '1')) == 1):
        raise RuntimeError(
            'Multiple pretrain.gpu_ids require the config-driven launcher: '
            'python -m train.imitation.launch --config=...'
        )
    run_imitation(
        FLAGS.config,
        FLAGS.gpu,
        auto=FLAGS.pretrain_auto,
        action_loss=FLAGS.action_loss,
        velocity_loss=FLAGS.velocity_loss,
    )


if __name__ == '__main__':
    app.run(main)
