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
"""Simple Agent."""

from rl_env import Agent
from operator import itemgetter
import copy


class HeuristicAgent(Agent):
  """Agent that applies a more complicated heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.reset(config)

  @staticmethod
  def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']]

  @staticmethod
  def dead_card(card, fireworks, discard):
    """A card is dead if it is too old."""
    if card['rank'] < fireworks[card['color']]:
      return True
    # a card is also dead if we have discarded all copies of its prerequisites
    # TODO: we can cache this if speed is a concern
    counts = {}
    for discarded in discard:
      if discarded['color'] == card['color'] and discarded['rank'] < card['rank']:
        if discarded['rank'] in counts:
          counts[discarded['rank']] += 1
        else:
          counts[discarded['rank']] = 1

    # can we fetch these hardcoded constants from somewhere?
    min_dead = 1000
    for rank, count in counts.items():
      complete = False
      if rank == 0 and count == 3:
        complete = True
      elif 2 <= rank <= 4 and count == 2:
        complete = True
      elif rank == 5 and count == 1:
        complete = True
      if complete:
        min_dead = min(min_dead, rank)
    return card['rank'] > min_dead

  @staticmethod
  def get_total_copies(rank):
    if rank == 1:
      return 3
    elif 2 <= rank <= 4:
      return 2
    else:
      return 1

  @staticmethod
  def has_remaining_rank(color, fireworks, discard, other_hands):
    min_rank = fireworks[color]
    count = 0
    for discarded in discard:
      if discarded['color'] == color and discarded['rank'] == min_rank:
        count += 1
    for hand in other_hands:
      for card in hand:
        if card['color'] == color and card['rank'] == min_rank:
          count += 1
    return count < HeuristicAgent.get_total_copies(min_rank)

  @staticmethod
  def min_rank(fireworks):
    # TODO: we could also do a legality check here. maybe the game is already complete.
    return min(rank for (_, rank) in fireworks.items())

  @staticmethod
  def rank_actions(fireworks, card_knowledge, observed_hands, discard_pile, num_hint_tokens, num_life_tokens):

    # How can we do this?
    # Let's make priorities.
    # For playing and discarding, our priorities will be decided reflexively, i.e. without considering future states
    # and actions of other agents.
    # For hinting, we will consider the actions of other agents.
    # Cards with full knowledge of playability are highest priority
    # Cards with hinted knowledge of playability have next priority
    # Cards with full knowledge of deadness have next priority
    # Cards with hinted knowledge of deadness risky to discard. only do this given no other choice.
    KNOWN_PLAY_PRIORITY = 5
    if num_life_tokens > 1:
      HINTED_PLAY_PRIORITY = 4
    else:
      HINTED_PLAY_PRIORITY = 0.5
    KNOWN_DEAD_DISCARD_PRIORITY = 3
    HINTED_DEAD_DISCARD_PRIORITY = 2
    UNKNOWN_DEAD_DISCARD_PRIORITY = 1

    ranked_actions = []
    for card_index, hint in enumerate(card_knowledge):
      if hint['color'] is not None and hint['rank'] is not None:
        if HeuristicAgent.playable_card(hint, fireworks):
          action = {'action_type': 'PLAY', 'card_index': card_index}
          ranked_actions.append((KNOWN_PLAY_PRIORITY, action))
        elif HeuristicAgent.dead_card(hint, fireworks, discard_pile):
          action = {'action_type': 'DISCARD', 'card_index': card_index}
          ranked_actions.append((KNOWN_DEAD_DISCARD_PRIORITY, action))
        else:  # might want to save this card.
          # TODO: still consider discarding, e.g. if there's no other options.
          # In that case, we should prefer to discard cards with remaining copies
          pass
      elif hint['color'] is not None or hint['rank'] is not None:
        # consider playing the card
        # TODO: we should only do this in situations where we can reason about our ally's intent,
        # e.g if the last hint was for this card, or if this card was uniquely identified by a hint
        action = {'action_type': 'DISCARD', 'card_index': card_index}
        ranked_actions.append((KNOWN_DEAD_DISCARD_PRIORITY, action))

        if hint['color'] is not None:
          # play, unless firework is full
          has_remaining = HeuristicAgent.has_remaining_rank(
            hint['color'], fireworks, discard_pile, observed_hands)
          if has_remaining:
            action = {'action_type': 'PLAY', 'card_index': card_index}
            ranked_actions.append((HINTED_PLAY_PRIORITY, action))
          else:
            action = {'action_type': 'DISCARD', 'card_index': card_index}
            ranked_actions.append((HINTED_DEAD_DISCARD_PRIORITY, action))
        else:  # hint['rank'] is not None:
          min_rank = HeuristicAgent.min_rank(fireworks)
          if hint['rank'] == min_rank:
            action = {'action_type': 'PLAY', 'card_index': card_index}
            ranked_actions.append((HINTED_PLAY_PRIORITY, action))
          elif hint['rank'] > min_rank:
            # hold on
            pass
          else:
            action = {'action_type': 'DISCARD', 'card_index': card_index}
            ranked_actions.append((HINTED_DEAD_DISCARD_PRIORITY, action))
      else:
        # we know nothing about the card
        action = {'action_type': 'DISCARD', 'card_index': card_index}
        # offset by card_index, so we break ties by oldest first
        ranked_actions.append((UNKNOWN_DEAD_DISCARD_PRIORITY + card_index / 10., action))

    ranked_actions = HeuristicAgent.filter_legal(ranked_actions, num_hint_tokens)
    ranked_actions.sort(key=itemgetter(0), reverse=True)
    return ranked_actions

  @staticmethod
  def is_legal(action, num_hint_tokens):
    if action['action_type'] == 'PLAY':
      # can always play a card
      return True
    if action['action_type'] == 'REVEAL_RANK' or action['action_type'] == 'REVEAL_COLOR':
      # can hint a card if we have tokens
      return num_hint_tokens > 0
    # action['action_type'] == 'DISCARD'
    # can hint if we haven't hinted too much
    return num_hint_tokens < 8

  @staticmethod
  def filter_legal(ranked_actions, num_hint_tokens):
    return [(weight, action) for weight, action in ranked_actions if HeuristicAgent.is_legal(action, num_hint_tokens)]

  def act(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] != 0:
      return None

    # We have three choices:
    # 1. Hint a card card, either to play, protect, or discard
    # 2. play a (ideally hinted) card
    # 3. discard an (ideally dead) card

    fireworks = observation['fireworks']
    my_ranked_actions = self.rank_actions(
      fireworks,
      observation['card_knowledge'][0],
      observation['observed_hands'],
      observation['discard_pile'],
      observation['information_tokens'],
      observation['life_tokens'])

    # for each hint, check if any hint will give us a point next turn
    # if none are available, try some of our ranked actions.

    HINT_PLAY_PRIORITY = 3.5
    HINT_DISCARD_PRIORITY = 1.5
    HINT_WRONG_DISCARD_PRIORITY = -1
    HINT_WRONG_PLAY_PRIORITY = -5

    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the opponents.
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        player_hints = observation['card_knowledge'][player_offset]

        # What is the cost of giving no hint?
        will_play_wrong_card_without_hint = False
        other_ranked_actions = self.rank_actions(
          fireworks,
          player_hints,
          [],
          observation['discard_pile'],
          min(observation['information_tokens'] + 1, self.max_information_tokens),
          observation['life_tokens'])
        if len(other_ranked_actions) > 0:
          chosen = other_ranked_actions[0][1]
          chosen_card = player_hand[chosen['card_index']]
          if chosen['action_type'] == 'PLAY':
            if not HeuristicAgent.playable_card(chosen_card, fireworks):
              will_play_wrong_card_without_hint = True

        hints_possible = set()
        for card in player_hand:
          hints_possible.add((card['color'], 'color', 'REVEAL_COLOR'))
          hints_possible.add((card['rank'], 'rank', 'REVEAL_RANK'))

        for hint_value, hint_type, hint_action in hints_possible:
          updated_player_hints = copy.deepcopy(player_hints)
          # 1. apply hint to the knowledge
          for card, hint in zip(player_hand, updated_player_hints):
            if card[hint_type] == hint_value:
              hint[hint_type] = hint_value

          # 2. get best action
          other_ranked_actions = self.rank_actions(
            fireworks,
            updated_player_hints,
            [],
            observation['discard_pile'],
            observation['information_tokens'] - 1,
            observation['life_tokens'])

          # 3. check best action

          if will_play_wrong_card_without_hint:
            AVOIDANCE_MODIFIER = 100
            if len(other_ranked_actions) == 0:
              # awkward
              continue

            chosen = other_ranked_actions[0][1]
            chosen_card = player_hand[chosen['card_index']]
            # these have different values
            # if the action is play and the card is playable, this has high value
            # if the action is play and the card is wrong, this has low value
            # if the action is discard and the card is dead, this has medium high value
            # if the action is discard and the card is alive, this has medium low value
            if chosen['action_type'] == 'PLAY':
              if HeuristicAgent.playable_card(chosen_card, fireworks):
                action = {'action_type': hint_action, hint_type: hint_value, 'target_offset': 1}
                my_ranked_actions.append((HINT_PLAY_PRIORITY + AVOIDANCE_MODIFIER, action))
              else:
                # the move is already bad, might as well give some information
                action = {'action_type': hint_action, hint_type: hint_value, 'target_offset': 1}
                my_ranked_actions.append((HINT_WRONG_PLAY_PRIORITY + AVOIDANCE_MODIFIER, action))
            elif chosen['action_type'] == 'DISCARD':
              if HeuristicAgent.dead_card(chosen_card, fireworks, observation['discard_pile']):
                action = {'action_type': hint_action, hint_type: hint_value, 'target_offset': 1}
                my_ranked_actions.append((HINT_WRONG_DISCARD_PRIORITY + AVOIDANCE_MODIFIER, action))
              else:
                # better force a discard of a good than a play of a bad card
                action = {'action_type': hint_action, hint_type: hint_value, 'target_offset': 1}
                my_ranked_actions.append((HINT_DISCARD_PRIORITY + AVOIDANCE_MODIFIER, action))
          else:
            if len(other_ranked_actions) == 0:
              # awkward
              continue
            chosen = other_ranked_actions[0][1]
            chosen_card = player_hand[chosen['card_index']]
            # if the action is play and the card is playable, this has high value
            # if the action is discard and the card is dead, this has medium high value
            if chosen['action_type'] == 'PLAY':
              if HeuristicAgent.playable_card(chosen_card, fireworks):
                action = {'action_type': hint_action, hint_type: hint_value, 'target_offset': 1}
                my_ranked_actions.append((HINT_PLAY_PRIORITY, action))
            elif chosen['action_type'] == 'DISCARD':
              if HeuristicAgent.dead_card(chosen_card, fireworks, observation['discard_pile']):
                action = {'action_type': hint_action, hint_type: hint_value, 'target_offset': 1}
                my_ranked_actions.append((HINT_DISCARD_PRIORITY, action))

    my_ranked_actions = HeuristicAgent.filter_legal(my_ranked_actions, observation['information_tokens'])
    my_ranked_actions.sort(key=itemgetter(0), reverse=True)

    # If no card is hintable then discard or hint arbitrarily.
    if len(my_ranked_actions) == 0:
      if observation['information_tokens'] < self.max_information_tokens:
        return {'action_type': 'DISCARD', 'card_index': 0}
      else:
        return {
          'action_type': 'REVEAL_COLOR',
          'color': observation['observed_hands'][1][0]['color'],
          'target_offset': 1
        }

    return my_ranked_actions[0][1]

  def reset(self, config):
    # Refresh config, in case it changed
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)
