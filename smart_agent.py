import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions, units, features

from pysc2.env import sc2_env

from absl import app

# Functions ID
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_ARMY_SUPPLY = features.Player.food_army

# Constants
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_HOSTILE = features.PlayerRelative.ENEMY

# Units
_TERRAN_COMMANDCENTER = units.Terran.CommandCenter
_TERRAN_SCV = units.Terran.SCV
_TERRAN_SUPPLY_DEPOT = units.Terran.SupplyDepot
_TERRAN_BARRACKS = units.Terran.Barracks
_NEUTRAL_MINERAL_FIELD = units.Neutral.MineralField

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

# Q weights saved in a file
DATA_FILE = 'sparse_agent_data'

ACTION_DO_NOTHING = 'donothing'
# ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
# ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
# ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
  ACTION_DO_NOTHING,
  # ACTION_SELECT_SCV,
  ACTION_BUILD_SUPPLY_DEPOT,
  ACTION_BUILD_BARRACKS,
  # ACTION_SELECT_BARRACKS,
  ACTION_BUILD_MARINE,
  # ACTION_SELECT_ARMY,
  # ACTION_ATTACK,
]

# Add 4x4 attack locations, make it easier to track than 64x64, and also avoid hard code location into magic numbers
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
      if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
        smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

# Reward signals
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


# RL brain
class QLearningTable:
  def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
    self.actions = actions  # a list
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon = e_greedy
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

  def choose_action(self, observation):
    # check if state exists else create it
    self.check_state_exist(observation)

    if np.random.uniform() < self.epsilon:
      # choose best action
      state_action = self.q_table.ix[observation, :]

      # some actions have the same value, avoid always choosing the first maximum value
      state_action = state_action.reindex(np.random.permutation(state_action.index))

      action = state_action.idxmax()
    else:
      # choose random action
      action = np.random.choice(self.actions)

    return action

  def learn(self, s, a, r, s_):
    self.check_state_exist(s_)
    self.check_state_exist(s)

    q_predict = self.q_table.ix[s, a]
    # q_target = r + self.gamma * self.q_table.ix[s_, :].max()
    if s_ != 'terminal':
      q_target = r + self.gamma * self.q_table.ix[s_, :].max()
    else:
      q_target = r  # next state is terminal

    # update
    self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

  def check_state_exist(self, state):
    if state not in self.q_table.index:
      # append new state to q table
      self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SparseAgent(base_agent.BaseAgent):
  def __init__(self):
    super().__init__()

    self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

    # self.previous_killed_unit_score = 0
    # self.previous_killed_building_score = 0

    self.previous_action = None
    self.previous_state = None

    self.cc_y = None
    self.cc_x = None

    self.move_number = 0

    if os.path.isfile(DATA_FILE + '.gz'):
      print('------------------Use previous Q table-----------------')
      self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

  @staticmethod
  def check_position_validity(*position, size=84):
    # Clip invalid position due to maybe camera calculation (refer to Timo)
    return 0 <= position[0] < size and 0 <= position[1] < size

  def transform_distance(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
      return x - x_distance, y - y_distance

    return x + x_distance, y + y_distance

  def transform_location(self, x, y):
    if not self.base_top_left:
      return 64 - x, 64 - y
    return x, y

  def splitAction(self, action_id):
    smart_action = smart_actions[action_id]

    x = 0
    y = 0
    if '_' in smart_action:
      smart_action, x, y = smart_action.split('_')

    return smart_action, x, y

  def step(self, obs):
    super().step(obs)

    if obs.last():
      reward = obs.reward
      self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')

      print('------------------Save current Q table-----------------')
      self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

      self.previous_action = None
      self.previous_state = None

      self.move_number = 0

      return actions.FUNCTIONS.no_op()

    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative == _PLAYER_SELF).nonzero()
      self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
      ccs = [u for u in obs.observation.feature_units if u.unit_type == _TERRAN_COMMANDCENTER]
      if len(ccs) > 0:
        cc = random.choice(ccs)
        self.cc_y, self.cc_x = cc.y, cc.x # (obs.observation.feature_screen.unit_type == _TERRAN_COMMANDCENTER).nonzero()

    cc_count = len([u for u in obs.observation.feature_units if u.unit_type == _TERRAN_COMMANDCENTER])
    depot_count = len([u for u in obs.observation.feature_units if u.unit_type == _TERRAN_SUPPLY_DEPOT])
    rack_count = len([u for u in obs.observation.feature_units if u.unit_type == _TERRAN_BARRACKS])
    army_supply = obs.observation.player.food_army


    # unit_type = obs.observation.feature_screen.unit_type

    # depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
    # supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

    # barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    # barracks_count = 1 if barracks_y.any() else 0

    # supply_limit = obs.observation.player.food_cap


    # killed_unit_score = obs.observation.score_cumulative.killed_value_units
    # killed_building_score = obs.observation.score_cumulative.killed_value_structures



    if self.move_number == 0:
      self.move_number += 1

      current_state = np.zeros(8)
      current_state[0] = cc_count
      current_state[1] = depot_count
      current_state[2] = rack_count
      current_state[3] = army_supply
      enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == _PLAYER_HOSTILE).nonzero()
      hot_squares = np.zeros(4)
      for i in range(0, len(enemy_y)):
        y = int(math.ceil((enemy_y[i] + 1) / 32))
        x = int(math.ceil((enemy_x[i] + 1) / 32))

        hot_squares[((y - 1) * 2) + (x - 1)] = 1

      if not self.base_top_left:
        hot_squares = hot_squares[::-1]

      for i in range(0, 4):
        current_state[i + 4] = hot_squares[i]

      if self.previous_action is not None:
        reward = 0

        # if killed_unit_score > self.previous_killed_unit_score:
        #   reward += KILL_UNIT_REWARD
        #
        # if killed_building_score > self.previous_killed_building_score:
        #   reward += KILL_BUILDING_REWARD

        self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

      rl_action = self.qlearn.choose_action(str(current_state))
      # smart_action = smart_actions[rl_action]

      # self.previous_killed_unit_score = killed_unit_score
      # self.previous_killed_building_score = killed_building_score
      self.previous_state = current_state
      self.previous_action = rl_action

      smart_action, x, y = self.splitAction(self.previous_action)

      if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        scvs = [u for u in obs.observation.feature_units if u.unit_type == _TERRAN_SCV]

        if len(scvs) > 0:
          scv = random.choice(scvs)
          if self.check_position_validity(scv.x, scv.y):
            return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))

      elif smart_action == ACTION_BUILD_MARINE:
        racks = [u for u in obs.observation.feature_units if u.unit_type == _TERRAN_BARRACKS]
        if len(racks) > 0:
          rack = random.choice(racks)
          if self.check_position_validity(rack.x, rack.y):
            return actions.FUNCTIONS.select_point('select_all_type', (rack.x, rack.y))

      elif smart_action == ACTION_ATTACK:
        if _SELECT_ARMY in obs.observation.available_actions:
          return actions.FUNCTIONS.select_army('select')

    elif self.move_number == 1:
      self.move_number += 1

      smart_action, x, y = self.splitAction(self.previous_action)

      if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        if depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation.available_actions:
          if cc_count > 0:
            if depot_count == 0:
              target = self.transform_distance(self.cc_x, -35, self.cc_y, 0)
            elif depot_count == 1:
              target = self.transform_distance(self.cc_x, -25, self.cc_y, -25)

            return actions.FUNCTIONS.Build_SupplyDepot_screen('now', target)

      elif smart_action == ACTION_BUILD_BARRACKS:
        if rack_count < 2 and _BUILD_BARRACKS in obs.observation.available_actions:
          if cc_count > 0:
            if rack_count == 0:
              target = self.transform_distance(self.cc_x, 15, self.cc_y, -9)
            elif rack_count == 1:
              target = self.transform_distance(self.cc_x, 15, self.cc_y, 12)

            return actions.FUNCTIONS.Build_Barracks_screen('now', target)

      elif smart_action == ACTION_BUILD_MARINE:
        if _TRAIN_MARINE in obs.observation.available_actions:
          return actions.FUNCTIONS.Train_Marine_quick('now')

      elif smart_action == ACTION_ATTACK:
        scv_is_selected = False

        if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == _TERRAN_SCV:
          scv_is_selected = True

        if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == _TERRAN_SCV:
          scv_is_selected = True

        if not scv_is_selected and _ATTACK_MINIMAP in obs.observation.available_actions:
          # notice that
          # random.randint(a,b) --> [a,b]
          # np.random.randint(a,b) --> [a,b)
          x_offset = random.randint(-1, 1)
          y_offset = random.randint(-1, 1)

          target = self.transform_location(int(x) + (x_offset * 8), int(y) + (y_offset * 8))
          if self.check_position_validity(*target, 64):
            return actions.FUNCTIONS.Attack_minimap('now', target)

    if self.move_number == 2:
      self.move_number = 0

      smart_action, x, y = self.splitAction(self.previous_action)

      if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        if _HARVEST_GATHER in obs.observation.available_actions:
          mineral_fields = [u for u in obs.observation.feature_units if u.unit_type == _NEUTRAL_MINERAL_FIELD]

          if len(mineral_fields) > 0:
            mineral_field = random.choice(mineral_fields)

            return actions.FUNCTIONS.Harvest_Gather_screen('queued', (mineral_field.x, mineral_field.y))



    # if smart_action == ACTION_DO_NOTHING:
    #   return actions.FUNCTIONS.no_op()
    #
    # elif smart_action == ACTION_SELECT_SCV:
    #   scvs = [u for u in obs.observation.feature_units if u.unit_type ==_TERRAN_SCV]
    #   # unit_type = obs.observation.feature_screen.unit_type
    #   # unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
    #
    #   # if unit_y.any():
    #   if len(scvs) > 0:
    #     scv = random.choice(scvs)
    #     # i = random.randint(0, len(unit_y) - 1)
    #     # target = [unit_x[i], unit_y[i]]
    #     if self.check_position_validity(scv.x, scv.y):
    #       return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))
    #
    # elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
    #   if _BUILD_SUPPLY_DEPOT in obs.observation.available_actions:
    #     ccs = [u for u in obs.observation.feature_units if u.unit_type ==_TERRAN_COMMANDCENTER]
    #     # unit_type = obs.observation.feature_screen.unit_type
    #     # unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
    #
    #     # if unit_y.any():
    #     if len(ccs) > 0:
    #       cc = random.choice(ccs)
    #       target = self.transform_distance(cc.x, 0, cc.y, 20)
    #       if self.check_position_validity(*target):
    #         return actions.FUNCTIONS.Build_SupplyDepot_screen('now', target)
    #
    # elif smart_action == ACTION_BUILD_BARRACKS:
    #   if _BUILD_BARRACKS in obs.observation.available_actions:
    #     racks = [u for u in obs.observation.feature_units if u.unit_type ==_TERRAN_BARRACKS]
    #     # unit_type = obs.observation.feature_screen.unit_type
    #     # unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
    #
    #     # if unit_y.any():
    #     if len(racks) > 0:
    #       rack = random.choice(racks)
    #       target = self.transform_distance(rack.x, 20, rack.y, 0)
    #       if self.check_position_validity(*target):
    #         return actions.FUNCTIONS.Build_Barracks_screen('now', target)
    #
    # elif smart_action == ACTION_SELECT_BARRACKS:
    #   racks = [u for u in obs.observation.feature_units if u.unit_type == _TERRAN_BARRACKS]
    #   # unit_type = obs.observation.feature_screen.unit_type
    #   # unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    #
    #   # if unit_y.any():
    #   if len(racks) > 0:
    #     rack = random.choice(racks)
    #     # target = [int(unit_x.mean()), int(unit_y.mean())]
    #     if self.check_position_validity(rack.x, rack.y):
    #       return actions.FUNCTIONS.select_point('select', (rack.x, rack.y))
    #
    # elif smart_action == ACTION_BUILD_MARINE:
    #   if _TRAIN_MARINE in obs.observation.available_actions:
    #     return actions.FUNCTIONS.Train_Marine_quick('now')
    #
    # elif smart_action == ACTION_SELECT_ARMY:
    #   if _SELECT_ARMY in obs.observation.available_actions:
    #     return actions.FUNCTIONS.select_army('select')
    #
    # elif smart_action == ACTION_ATTACK:
    #   if obs.observation.single_select[0].unit_type != _TERRAN_SCV and\
    #           _ATTACK_MINIMAP in obs.observation.available_actions:
    #     return actions.FUNCTIONS.Attack_minimap('now', self.transform_location(int(x), int(y)))


    return actions.FUNCTIONS.no_op()


def main(unused_argv):
  agent = SparseAgent()
  try:
    while True:
      with sc2_env.SC2Env(
              map_name="Simple64",
              players=[sc2_env.Agent(sc2_env.Race.terran),
                       sc2_env.Bot(sc2_env.Race.terran,
                                   sc2_env.Difficulty.very_easy)],
              agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
              step_mul=8,
              game_steps_per_episode=0,
              visualize=False) as env:

        agent.setup(env.observation_spec(), env.action_spec())

        timesteps = env.reset()
        agent.reset()

        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)

  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)