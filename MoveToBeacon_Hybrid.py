
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps

from absl import flags, app

import numpy as np
import random


def get_marine_location(ai_relative_view):
  '''get the indices where the world is equal to 1'''
  return (ai_relative_view == features.PlayerRelative.SELF).nonzero()


def get_rand_location(ai_location):
  '''gets a random location at least n away from current x,y point.'''
  return [np.random.randint(0, 64), np.random.randint(0, 64)]


class Agent1(base_agent.BaseAgent):
  """An agent for doing a simple movement form one point to another."""

  def step(self, obs):
    '''step function gets called automatically by pysc2 environment'''

    # call the parent class to have pysc2 setup rewards/etc for us
    super(Agent1, self).step(obs)
    # if we can move our army (we have something selected)
    if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      # get what the ai can see about the world
      ai_view = obs.observation.feature_minimap.player_relative
      # get the location of our marine in this world
      marine_x, marine_y = get_marine_location(ai_view)
      # it our marine is not on the screen do nothing.
      # this happens if we scroll away and look at a different
      # part of the world
      if not marine_x.any():
        return actions.FUNCTIONS.no_op()
      target = get_rand_location([marine_x, marine_y])
      return actions.FUNCTIONS.Move_screen('now', target)
    # if we can't move, we havent selected our army, so selecto ur army
    else:
      return actions.FUNCTIONS.select_army('select')