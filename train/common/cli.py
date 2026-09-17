"""The legacy absl flags shared by the two new training entry points."""

from absl import flags
from ml_collections import config_flags


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', 'config/robot_base.py', 'Training configuration.'
)
flags.DEFINE_bool('train', True, 'Run reinforcement training')
flags.DEFINE_bool('test', False, 'Run testing')
flags.DEFINE_bool('pretrain', False, 'Run imitation pretraining')
flags.DEFINE_bool('pretrain_auto', False, 'Auto-stop pretraining on success')
flags.DEFINE_string('checkpoint', '', 'Path to checkpoint for testing')
flags.DEFINE_string(
    'pretrain_checkpoint', '', 'Pretrained checkpoint for reinforcement training'
)
flags.DEFINE_integer('num_test_episodes', 5, 'Number of test episodes')
flags.DEFINE_integer('gpu', 0, 'Physical GPU ID')
flags.DEFINE_bool('sampletest', False, 'Record sample trajectories')
flags.DEFINE_bool('action_loss', False, 'Use action reconstruction pretraining loss')
flags.DEFINE_bool('velocity_loss', False, 'Use flow velocity pretraining loss')
flags.DEFINE_bool('validate_only', False, 'Validate imports and configuration without training')
