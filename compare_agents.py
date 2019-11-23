# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import random
import os
import psutil

# it's so hard to disable logging ;_;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import sys
import getopt
import rl_env
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from agents.loss_averse_agent import LossAverseAgent
from agents.rainbow.rainbow_agent import RainbowAgent
from agents.rainbow.dqn_agent import DQNAgent
from agents.rainbow.run_experiment import create_agent, create_obs_stacker, format_legal_moves
from agents.rainbow.third_party.dopamine import checkpointer
from agents.heuristic_agent import HeuristicAgent


class LearnedAgent(rl_env.Agent):
  def __init__(self, internal_agent, obs_stacker, environment):
    self.internal_agent = internal_agent
    self.obs_stacker = obs_stacker
    self.environment = environment
    
  def act(self, observation):
    current_player = observation['current_player']
    legal_moves = observation['legal_moves_as_int']
    legal_moves = format_legal_moves(legal_moves, self.environment.num_moves())

    observation_vector = np.array(observation['vectorized'])
    #self.obs_stacker.add_observation(observation_vector, current_player)
    #observation_vector = self.obs_stacker.get_observation_stack(current_player)

    action = self.internal_agent._select_action(observation_vector, legal_moves)
    return action.item()
  
  def reset(self, config):
    self.obs_stacker.reset_stack()
    # need to reset anything else?

def create_tf_agent(environment, type, checkpoint_dir):
  # wrap in a new graph, so we don't clash variable names
  g = tf.Graph()
  with g.as_default():
    obs_stacker = create_obs_stacker(environment, 1)
    obs_stacker.reset_stack()
    agent = create_agent(environment, obs_stacker, type)
    experiment_checkpointer = checkpointer.Checkpointer(checkpoint_dir, 'ckpt')
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(checkpoint_dir)
    dqn_dictionary = experiment_checkpointer.load_checkpoint(latest_checkpoint_version)
    agent.unbundle(checkpoint_dir, latest_checkpoint_version, dqn_dictionary)
    agent.eval_mode = True
    return LearnedAgent(agent, obs_stacker, environment)

def memory():
  pid = os.getpid()
  py = psutil.Process(pid)
  memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
  print('memory use: ' + str(memoryUse) + ' GB', flush=True)
    
class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players']}
    self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
    self.agent_class = [
        SimpleAgent,
        RandomAgent,
        LossAverseAgent,
        lambda config: create_tf_agent(self.environment, 'Rainbow', 'agents/rainbow/tmp/hanabi_rainbow/checkpoints'),
        lambda config: create_tf_agent(self.environment, 'DQN', 'agents/rainbow/tmp/hanabi_dqn/checkpoints'),
        lambda config: create_tf_agent(self.environment, 'Rainbow', 'agents/rainbow/tmp/pretrained/'),
        HeuristicAgent,
    ]

  def run(self):
    """Run episodes."""
    num_agent_classes = len(self.agent_class)
    num_agents = self.agent_config['players']
    rewards = [[None for _ in range(num_agent_classes)] for __ in range(num_agent_classes)]
    
    agent_instances = list([[ctor(self.agent_config) for _ in range(num_agents)] for ctor in self.agent_class])
      
    for i in range(num_agent_classes):
      if num_agents == 2:
        # no need to run both (i, j) and (j, i)
        j_end = i+1
      else:
        j_end = num_agent_classes
      for j in range(j_end):
        print('running agent {} with agent {}'.format(i, j), flush=True)
        
        pair_rewards = []
        for episode in range(flags['num_episodes']):
          memory()
          observations = self.environment.reset()
          #agents = [self.agent_class[i](self.agent_config)] + [self.agent_class[j](self.agent_config) for _ in range(num_agents-1)]
          agents = list([agent_instances[i][0]] + agent_instances[j][1:])
          random.shuffle(agents)
          for agent in agents:
            agent.reset(self.agent_config)
          done = False
          episode_reward = 0
          while not done:
            for agent_id, agent in enumerate(agents):
              observation = observations['player_observations'][agent_id]
              action = agent.act(observation)
              if observation['current_player'] == agent_id:
                assert action is not None
                current_player_action = action
              else:
                assert action is None
              # Make an environment step.
              observations, reward, done, unused_info = self.environment.step(
                  current_player_action)
              episode_reward += reward
              if done:
                break
          pair_rewards.append(episode_reward)
        rewards[i][j] = pair_rewards
        if num_agents == 2:
          rewards[j][i] = pair_rewards
          
    return rewards

if __name__ == "__main__":
  flags = {'players': 2, 'num_episodes': 3}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  if arguments:
    sys.exit('usage: compare_agents.py [options]\n'
             '--num_episodes  number of game episodes to run.\n')
  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  runner = Runner(flags)
  reward_matrix = runner.run()
  reward_matrix = np.asarray(reward_matrix)
  print('mean:')
  print(np.mean(reward_matrix, axis=2))
  print('stddev:')
  print(np.std(reward_matrix, axis=2))
  print('median:')
  print(np.median(reward_matrix, axis=2))
  print('max:')
  print(np.max(reward_matrix, axis=2))
  print('min:')
  print(np.min(reward_matrix, axis=2))

