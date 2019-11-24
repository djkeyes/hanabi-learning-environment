import gin.tf
import tensorflow as tf
import numpy as np
import random
import os

slim = tf.contrib.slim


def discr_template(state, input_networks, agent_selector, layer_size=512, num_layers=1):
  r"""Builds a DQN Network mapping states to Q-values.

  Args:
    state: A `tf.placeholder` for the RL state.
    input_networks: A network for the sub-agent action.
    agent_selector: A `tf.placeholder` choosing the sub-agent.
    layer_size: int, number of hidden units per layer.
    num_layers: int, Number of hidden layers.

  Returns:
    net: A `tf.Graphdef` for discriminator
  """
  weights_initializer = slim.variance_scaling_initializer(
    factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  # switch_case expects a dict which returns functions which return tensors
  # wow, the prices of genericization T_T
  branch_fxns = {i: lambda: a._q_argmax for i, a in enumerate(input_networks)}
  net = tf.compat.v1.switch_case(tf.cast(agent_selector, tf.int32), branch_fxns, name='agent_selector')
  net = tf.reshape(net, (1, 1, 1))
  net = tf.concat([tf.cast(state, tf.float32), tf.cast(net, tf.float32)], axis=1)
  net = tf.squeeze(net, axis=2)
  for _ in range(num_layers):
    net = slim.fully_connected(net, layer_size,
                               activation_fn=tf.nn.relu)
  net = slim.fully_connected(net, len(input_networks), activation_fn=None,
                             weights_initializer=weights_initializer)

  return net


@gin.configurable
class RainbowDiscriminator:
  """An predictor which discriminates between known Rainbow agents."""

  @gin.configurable
  def __init__(self,
               agent_template,
               num_actions=None,
               observation_size=None,
               stack_size=1,
               num_agents=2,
               graph_template=discr_template):
    self.agents = [agent_template() for _ in range(num_agents)]

    self.num_actions = num_actions

    self._first_agent = None
    self._second_agent = None

    states_shape = (1, observation_size, stack_size)
    self.state_ph = tf.compat.v1.placeholder(tf.uint8, states_shape, name='state_ph')
    self.agent_selector_ph = tf.compat.v1.placeholder(tf.uint8, (), name='selector_ph')

    # TODO: need to make separate online and target nets to stabilize learning?
    template_net = tf.compat.v1.make_template('Discriminator', graph_template)
    self._agent_prediction = template_net(
      state=self.state_ph, input_networks=self.agents, agent_selector=self.agent_selector_ph)
    self._agent_prediction = tf.nn.softmax(self._agent_prediction)

    self._discr_train_op = self.build_discr_train_op()

    self._sess = tf.compat.v1.Session(
      '', config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
    self._init_op = tf.compat.v1.global_variables_initializer()
    self._sess.run(self._init_op)

    self._saver = tf.compat.v1.train.Saver(max_to_keep=3)

  def build_discr_train_op(self):
    # TODO: consider batching actions over several frames
    # TODO: choice of constants?
    learning_rate = 0.000025
    optimizer_epsilon = 0.00003125
    optimizer = tf.compat.v1.train.AdamOptimizer(
      learning_rate=learning_rate,
      epsilon=optimizer_epsilon)

    labels = tf.one_hot(self.agent_selector_ph, len(self.agents))
    pred = self._agent_prediction
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(pred), reduction_indices=1))
    return optimizer.minimize(loss)

  def get_agents(self):
    return self.agents

  def get_network(self):
    return self._agent_prediction

  def sample_agent_pair(self):
    # TODO: the choice of sampling strategy here seems important. If we choose both agent types at random, most games
    # will be mixed play, which might be undesirable. for now, always select mirror matches.
    # TODO: in the mixed-place case, both agents' actions contribute to the final result. So we need to sure we backprop
    # to agent A even on agent B's turn
    self._first_agent = random.sample(range(len(self.agents)), k=1)[0]
    self._second_agent = self._first_agent

  def begin_episode(self, current_player, legal_actions, observation):
    if current_player == 0:
      return self.agents[self._first_agent].begin_episode(current_player, legal_actions, observation)
    else:
      return self.agents[self._second_agent].begin_episode(current_player, legal_actions, observation)

  def step(self, reward, current_player, legal_actions, observation):
    if current_player == 0:
      agent_idx = self._first_agent
    else:
      agent_idx = self._second_agent
    agent = self.agents[agent_idx]

    # # same as rainbow
    # agent._train_step()
    #
    # agent.action = agent._select_action(observation, legal_actions)
    # agent._record_transition(current_player, reward, observation, legal_actions,
    #                          agent.action)

    # same as rainbut, but with discr step
    agent._train_step()
    agent.action = agent._select_action(observation, legal_actions)
    agent._record_transition(current_player, reward, observation, legal_actions,
                             agent.action)
    selector = np.array(agent_idx, dtype=np.uint8)
    feed_dict = {agent.legal_actions_ph: legal_actions,
                 self.agent_selector_ph: selector,
                 self.state_ph: agent.state}
    # need to fill placeholders for all agents, even if they are unused
    for a in self.agents:
      feed_dict[a.state_ph] = agent.state
      feed_dict[a.legal_actions_ph] = legal_actions
    self._sess.run(self._discr_train_op, feed_dict)
    # result, prediction = self._sess.run([self._discr_train_op, self._agent_prediction], feed_dict)
    # prediction_argmax = np.argmax(prediction)
    # error_sq = (agent_idx - prediction_argmax) ** 2  # for 2-agent case
    # print('prediction: {}, actual: {}, error: {}'.format(prediction_argmax, agent_idx, error_sq))

    # # don't know what I was trying to do here
    # if agent.eval_mode:
    #   epsilon = agent.epsilon_eval
    # else:
    #   epsilon = agent.epsilon_fn(agent.epsilon_decay_period, agent.training_steps,
    #                              agent.min_replay_history, agent.epsilon_train)
    #
    # if random.random() <= epsilon:
    #   # Choose a random action with probability epsilon.
    #   legal_action_indices = np.where(legal_actions == 0.0)
    #   action = np.random.choice(legal_action_indices[0])
    # else:
    #   # Convert observation into a batch-based format.
    #   agent.state[0, :, 0] = observation
    #
    #   selector = np.array(agent_idx, dtype=np.uint8)
    #
    #   feed_dict = {agent.legal_actions_ph: legal_actions,
    #                self.agent_selector_ph: selector,
    #                self.state_ph: agent.state}
    #   # need to fill placeholders for all agents, even if they are unused
    #   for a in self.agents:
    #     feed_dict[a.state_ph] = agent.state
    #     feed_dict[a.legal_actions_ph] = legal_actions
    #
    #   # Choose the action maximizing the q function for the current state.
    #   action = agent._sess.run(agent._q_argmax, feed_dict)
    #   assert legal_actions[action] == 0.0, 'Expected legal action.'
    #
    #   self._sess.run(self._discr_train_op, feed_dict)
    #
    # agent.action = action
    # # note: this is the reward accrued since this player last made a move
    # agent._record_transition(current_player, reward, observation, legal_actions,
    #                          agent.action)

    return agent.action

  def end_episode(self, current_player, final_rewards):
    # Do we only need to do this once? Don't want to double-count final_rewards.
    if current_player == 0:
      return self.agents[self._first_agent].end_episode(final_rewards)
    else:
      return self.agents[self._second_agent].end_episode(final_rewards)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    self._saver.save(
      self._sess,
      os.path.join(checkpoint_dir, 'tf_ckpt'),
      global_step=iteration_number)
    bundle_dictionary = {
      'first_agent': self._first_agent,
      'second_agent': self._second_agent
    }
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    for key in self.__dict__:
      if key in bundle_dictionary:
        self.__dict__[key] = bundle_dictionary[key]
    self._saver.restore(self._sess, tf.compat.v1.train.latest_checkpoint(checkpoint_dir))
    return True
