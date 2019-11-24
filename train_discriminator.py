
# it's so hard to disable logging ;_;
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore',category=FutureWarning)
    import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# tf.logging.set_verbosity(tf.logging.ERROR)
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

from absl import app
from absl import flags

from agents.rainbow.third_party.dopamine import logger

from agents.rainbow import run_experiment

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', '',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')


def launch_experiment():
  """Launches the experiment.

  Specifically:
  - Load the gin configs and bindings.
  - Initialize the Logger object.
  - Initialize the environment.
  - Initialize the observation stacker.
  - Initialize the agent.
  - Reload from the latest checkpoint, if available, and initialize the
    Checkpointer object.
  - Run the experiment.
  """
  if FLAGS.base_dir == None:
    raise ValueError('--base_dir is None: please provide a path for '
                     'logs and checkpoints.')

  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

  environment = run_experiment.create_environment()
  obs_stacker = run_experiment.create_obs_stacker(environment)
  discriminator = run_experiment.create_discriminator(
      environment, obs_stacker, lambda: run_experiment.create_agent(environment, obs_stacker))

  # for i, agent in enumerate(discriminator.get_agents()):
  #   checkpoint_dir = '{}/agent_{}/checkpoints'.format(FLAGS.base_dir, i)
  #   start_iteration, experiment_checkpointer = (
  #       run_experiment.initialize_checkpointing(agent,
  #                                               experiment_logger,
  #                                               checkpoint_dir,
  #                                               FLAGS.checkpoint_file_prefix))
  checkpoint_dir = '{}/discr/checkpoints'.format(FLAGS.base_dir)
  start_iteration, experiment_checkpointer = (
      run_experiment.initialize_checkpointing(discriminator,
                                              experiment_logger,
                                              checkpoint_dir,
                                              FLAGS.checkpoint_file_prefix))

  run_experiment.run_discriminator_experiment(
      discriminator,
      environment,
      start_iteration,
      obs_stacker,
      experiment_logger,
      experiment_checkpointer,
      checkpoint_dir,
      logging_file_prefix=FLAGS.logging_file_prefix)


def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  launch_experiment()

if __name__ == '__main__':
  app.run(main)
