"""Entry point for DiffusionNFT and StepNFT reinforcement fine-tuning."""

from train.common.bootstrap import configure_gpu

configure_gpu()

from absl import app

from train.common.cli import FLAGS
from train.reinforcement.runner import run_reinforcement


def main(_):
    if FLAGS.config is None or not hasattr(FLAGS.config, 'train'):
        raise ValueError('Use --config=config/reinforcement/reinforcement_stepnft.py (or another reinforcement preset)')
    if FLAGS.validate_only:
        loss = FLAGS.config.train.loss_type
        if loss not in ('rollout', 'stepnft'):
            raise ValueError(f'Unsupported reinforcement loss: {loss!r}')
        print(f'Reinforcement configuration OK: loss_type={loss}')
        return
    return run_reinforcement(
        FLAGS.config, FLAGS.gpu, pretrain_checkpoint=FLAGS.pretrain_checkpoint
    )


if __name__ == '__main__':
    app.run(main)
