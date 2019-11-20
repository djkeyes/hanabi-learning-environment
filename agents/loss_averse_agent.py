
"""Loss Averse Agent."""

from rl_env import Agent


class LossAverseAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.reset(config)

  @staticmethod
  def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']]

  def act(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] != 0:
      return None

    # Check if there are any pending hints and play the card corresponding to
    # the hint.
    for card_index, hint in enumerate(observation['card_knowledge'][0]):
      if hint['color'] is not None and hint['rank'] is not None:
        return {'action_type': 'PLAY', 'card_index': card_index}
      # But don't be too dumb if we're about to loose
      if observation['life_tokens'] > 1:
        if hint['color'] is not None or hint['rank'] is not None:
          return {'action_type': 'PLAY', 'card_index': card_index}

    # Check if it's possible to hint a card to your colleagues.
    fireworks = observation['fireworks']
    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the opponents.
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        player_hints = observation['card_knowledge'][player_offset]
        # Check if the card in the hand of the opponent is playable.
        for card, hint in zip(player_hand, player_hints):
          if LossAverseAgent.playable_card(card, fireworks):
            if hint['color'] is None:
              return {
                  'action_type': 'REVEAL_COLOR',
                  'color': card['color'],
                  'target_offset': player_offset
              }

    # If no card is hintable then discard or hint arbitrarily.
    if observation['information_tokens'] < self.max_information_tokens:
      return {'action_type': 'DISCARD', 'card_index': 0}
    else:
      return {
          'action_type': 'REVEAL_COLOR',
          'color': observation['observed_hands'][1][0]['color'],
          'target_offset': 1
      }

  def reset(self, config):
    # Refresh config, in case it changed
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)
