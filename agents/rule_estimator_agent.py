from operator import itemgetter

from rl_env import Agent

from .ruleset import *

# RULES = [discard_oldest_first, osawa_discard, tell_unknown, tell_randomly, play_safe_card, play_if_certain, tell_playable_card_outer, tell_dispensable_factory(), tell_anyone_useful_card, tell_anyone_useless_card, tell_most_information, tell_playable_card, legal_random, discard_randomly, play_probably_safe_factory(0.0), play_probably_safe_factory(0.2), play_probably_safe_factory(0.25), play_probably_safe_factory(0.4), play_probably_safe_factory(0.6), play_probably_safe_factory(0.8), play_probably_safe_factory(0.0, True), play_probably_safe_factory(0.2, True), play_probably_safe_factory(0.4, True), play_probably_safe_factory(0.6, True), play_probably_safe_factory(0.8, True), discard_probably_useless_factory(0.0), discard_probably_useless_factory(0.2), discard_probably_useless_factory(0.4), discard_probably_useless_factory(0.6), discard_probably_useless_factory(0.8)]
DETERMINISTIC_RULES = [
    Ruleset.discard_oldest_first,
    Ruleset.osawa_discard, 
    Ruleset.tell_unknown, 
    Ruleset.play_safe_card,
    Ruleset.play_if_certain, 
    Ruleset.tell_playable_card_outer, 
    Ruleset.tell_dispensable_factory(), 
    Ruleset.tell_anyone_useless_card,
    Ruleset.tell_most_information,
    Ruleset.play_probably_safe_factory(0.0),
    Ruleset.play_probably_safe_factory(0.2), 
    Ruleset.play_probably_safe_factory(0.25), 
    Ruleset.play_probably_safe_factory(0.4), 
    Ruleset.play_probably_safe_factory(0.6),
    Ruleset.play_probably_safe_factory(0.8), 
    Ruleset.play_probably_safe_factory(0.0, True), 
    Ruleset.play_probably_safe_factory(0.2, True), 
    Ruleset.play_probably_safe_factory(0.4, True),
    Ruleset.play_probably_safe_factory(0.6, True),
    Ruleset.play_probably_safe_factory(0.8, True),
    Ruleset.discard_probably_useless_factory(0.0),
    Ruleset.discard_probably_useless_factory(0.2),
    Ruleset.discard_probably_useless_factory(0.4),
    Ruleset.discard_probably_useless_factory(0.6),
    Ruleset.discard_probably_useless_factory(0.8)
]

# class RuleEstimatorAgent(Agent):

  # def __init__(self, config, backup_strategy, *args, **kwargs):
    # """Initialize the agent."""
    # if backup_strategy is not None:
      # self.backup_strategy = backup_strategy
      # self.backup_strategy.reset(config)
    # self.rules = None
    # self.rule_observations = {}
    # self.rule_implications = np.zeros(shape=(len(DETERMINISTIC_RULES), len(DETERMINISTIC_RULES)))

  # def act(self, observation):
    # if self.rules is None:
      # return self.backup_strategy.act(observation)
    # else:
      # for rule in self.rules:
        # action = rule(observation)
        # if action is not None:
          # return action
      # return Ruleset.legal_random(observation)

  # def reset(self, config):
    # if self.backup_strategy is not None:
      # self.backup_strategy.reset(config)
    # self.rules = None
    
  # def start_observations(self):
    # self.rule_observations.clear()
    # self.rule_implications.fill(0)

  # def start_observe_episode(self):
    # # default: noop
    # pass

  # def observe(self, agent_id, observation, action):
    # matched_rules = []
    # mismatched_rules = []
    
    # for i, rule in enumerate(DETERMINISTIC_RULES):
      # a = rule(observation)
      # if a == action:
        # matched_rules.append(i)
      # elif a is not None:
        # mismatched_rules.append(i)
    # if len(matched_rules) > 0:
      # weight = 1./len(matched_rules)
      # for i in matched_rules:
        # self.add_rule_observation(i, weight)
        # for j in mismatched_rules:
          # self.add_rule_implication(i, j, weight)

  # def end_observe_episode(self):
    # self.rules = []
    
  # def end_observations(self):
    # #print('observed the following rules: ', self.rule_observations)
    # #print('observed the implications: ', self.rule_implications)
    # # should we drop some rules? some weights are too low to be meaningful
    # # our observations imply a (possibly inconsistent) partial ordering
    # # i.e. a should be rated higher than b, with weight w_i
    # # Here's a pretty dumb ordering method. we could probably do better
    # rhs = []
    # lhs = []
    # for i in self.rule_observations:
      # for j in self.rule_observations:
        # weight = self.rule_implications[i, j] - self.rule_implications[j, i]
        # # want theta dot (i hat - j hat) == weight
        # one_hot_diff = np.zeros(shape=(len(DETERMINISTIC_RULES), ))
        # one_hot_diff[i] -= 1
        # one_hot_diff[j] += 1
        # rhs.append(one_hot_diff)
        # lhs.append(weight)
    # # regularize
    # lambd = 0.01
    # for i in self.rule_observations:
      # one_hot = np.zeros(shape=(len(DETERMINISTIC_RULES), ))
      # one_hot[i] = lambd
      # rhs.append(one_hot)
      # lhs.append(0.0)
        
    
    # rhs = np.array(rhs)
    # lhs = np.array(lhs)
    # #print('rhs, ', rhs)
    # #print(rhs.shape)
    # #print('lhs, ', lhs)
    # #print(lhs.shape)
    # sln, _, _, _ = np.linalg.lstsq(rhs, lhs)
    # #print('sln: ', sln)
    # #print(sln.shape)
    # sorted_rules = [(sln[i], DETERMINISTIC_RULES[i]) for i in self.rule_observations]
    # sorted_rules.sort(key=itemgetter(0))
    # self.rules = [rule for (_, rule) in sorted_rules]
    # #print('sorted rules: ', self.rules)
    

  # def add_rule_observation(self, rule, weight):
    # if rule in self.rule_observations:
      # self.rule_observations[rule] += weight
    # else:
      # self.rule_observations[rule] = weight

  # def add_rule_implication(self, implied, unobserved, weight):
    # self.rule_implications[implied, unobserved] += weight
    
    

class RuleEstimatorAgent(Agent):

  def __init__(self, config, backup_strategy, *args, **kwargs):
    """Initialize the agent."""
    if backup_strategy is not None:
      self.backup_strategy = backup_strategy
      self.backup_strategy.reset(config)
    self.rules = None

  def act(self, observation):
    if self.rules is None:
      return self.backup_strategy.act(observation)
    else:
      for rule in self.rules:
        action = rule(observation)
        if action is not None:
          return action
      return Ruleset.legal_random(observation)

  def reset(self, config):
    if self.backup_strategy is not None:
      self.backup_strategy.reset(config)
    
  def start_observations(self):
    self.explanations = []
    self.rule_frequency = np.zeros(shape=(len(DETERMINISTIC_RULES),))

  def start_observe_episode(self):
    # default: noop
    pass

  def observe(self, agent_id, observation, action):
    matched_rules = []
    mismatched_rules = []
    
    for i, rule in enumerate(DETERMINISTIC_RULES):
      a = rule(observation)
      if a == action:
        matched_rules.append(i)
      elif a is not None:
        mismatched_rules.append(i)
    if len(matched_rules) > 0:
      self.explanations.append(matched_rules)
      for i in matched_rules:
        self.rule_frequency[i] += 1

  def end_observe_episode(self):
    pass
    
  def end_observations(self):
    # pick the best rules greedily
    best_rules = []
    for iter in range(10):
      i = np.argmax(self.rule_frequency)
      if self.rule_frequency[i] == 0:
        break
      best_rules.append(i)
      next_explanations = []
      for arr in self.explanations:
        if i in arr:
          for j in arr:
            self.rule_frequency[j] -= 1
        else:
          next_explanations.append(arr)
      self.explanations = next_explanations
    # now we have some rules
    # TODO: establish an ordering which explains our observations
    
    self.rules = [DETERMINISTIC_RULES[i] for i in best_rules]
    print('chosen rules are {}'.format(self.rules))
    
  def reset_observations(self):
    self.rules = None

