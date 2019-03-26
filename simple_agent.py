from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from absl import app

import time
import random

class SimpleAgent(base_agent.BaseAgent):
	def __init__(self):
		super().__init__()
		self.base_top_left = False
		self.barracks_rallied = False

	def transform_location(self, x, x_dist, y, y_dist):
		if not self.base_top_left:
			return [x - x_dist, y - y_dist]

		return [x + x_dist, y + y_dist]

	def unit_type_is_selected(self, obs, _unit_type):
		if len(obs.observation.single_select) > 0 and\
			obs.observation.single_select[0].unit_type == _unit_type:
			return True

		if len(obs.observation.multi_select) > 0 and\
			obs.observation.multi_select[0].unit_type == _unit_type:
			return True

		return False

	def can_do(self, obs, action):
		return action in obs.observation.available_actions

	def get_unit_by_type(self, obs, _unit_type):
		return [unit for unit in obs.observation.feature_units if unit.unit_type == _unit_type]

	def check_position_validity(self, *position):
		return 0<= position[0]< 84 and 0<= position[1] < 84

	def step(self, obs):
		# super(SimpleAgent, self).step(obs)
		super().step(obs)

		if obs.first():
			player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()

			xmean = player_x.mean()
			ymean = player_y.mean()

			self.base_top_left = ymean <= 31

		SCVs = self.get_unit_by_type(obs, units.Terran.SCV)

		# build SupplyDepot
		supply_depots = self.get_unit_by_type(obs, units.Terran.SupplyDepot) + self.get_unit_by_type(obs, units.Terran.SupplyDepotLowered)
		if len(supply_depots) == 0:
			if self.unit_type_is_selected(obs, units.Terran.SCV):
				if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
					# x = random.randint(0, 83)
					# y = random.randint(0, 83)
					CCs = self.get_unit_by_type(obs, units.Terran.CommandCenter)
					if len(CCs) > 0:
						CC = random.choice(CCs)
						x, y = self.transform_location(CC.x, 0, CC.y, 20)
						if self.check_position_validity(x,y):
							return actions.FUNCTIONS.Build_SupplyDepot_screen('now', (x, y)) # (x, y)

			else:
				# select SCVs
				if len(SCVs) > 0:
					SCV = random.choice(SCVs)
					if self.check_position_validity(SCV.x, SCV.y):
						return actions.FUNCTIONS.select_point('select_all_type', (SCV.x, SCV.y))

		# build Barracks
		barracks = self.get_unit_by_type(obs, units.Terran.Barracks)
		if len(barracks) == 0:
			if self.unit_type_is_selected(obs, units.Terran.SCV):
				if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
					CCs = self.get_unit_by_type(obs, units.Terran.CommandCenter)
					if len(CCs) > 0:
						CC = random.choice(CCs)
						x, y = self.transform_location(CC.x, 20, CC.y, 0)
						if self.check_position_validity(x,y):
							return actions.FUNCTIONS.Build_Barracks_screen('now', (x, y)) # (x, y)

			else:
				# select SCVs
				if len(SCVs) > 0:
					SCV = random.choice(SCVs)
					if self.check_position_validity(SCV.x, SCV.y):
						return actions.FUNCTIONS.select_point('select_all_type', (SCV.x, SCV.y))


		# rally Barracks and train Marines
		if not self.barracks_rallied:
			if self.unit_type_is_selected(obs, units.Terran.Barracks):
				if self.can_do(obs, actions.FUNCTIONS.Rally_Units_minimap.id):
					self.barracks_rallied = True
					if self.base_top_left:
						return actions.FUNCTIONS.Rally_Units_minimap('now', (12, 16))
					return actions.FUNCTIONS.Rally_Units_minimap('now', (49, 49))

			else:
				# select Barracks
				if len(barracks) > 0:
					barrack = random.choice(barracks)
					if self.check_position_validity(barrack.x, barrack.y):
						return actions.FUNCTIONS.select_point('select_all_type', (barrack.x, barrack.y))

		# train Marines
		if obs.observation.player.food_used < obs.observation.player.food_cap:
			if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
				return actions.FUNCTIONS.Train_Marine_quick('now')


		# # build SCVs
		# if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
		# 	if self.can_do(obs, actions.FUNCTIONS.Train_SCV_quick.id):
		# 		if obs.observation.player.food_used < obs.observation.player.food_cap:
		# 			return actions.FUNCTIONS.Train_SCV_quick('now')

		# # select CommandCenters
		# CCs = self.get_unit_by_type(obs, units.Terran.CommandCenter)
		# if len(CCs) > 0:
		# 	CC = random.choice(CCs)
		# 	if self.check_position_validity(CC.x, CC.y):
		# 		return actions.FUNCTIONS.select_point('select_all_type', (CC.x, CC.y))

		# if maxed out then attack enemy location
		if obs.observation.player.food_used == obs.observation.player.food_cap:
			if self.unit_type_is_selected(obs, units.Terran.Marine):
				if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
					if self.base_top_left:
						return actions.FUNCTIONS.Attack_minimap('now', (49, 49))
					else:
						return actions.FUNCTIONS.Attack_minimap('now', (12, 16))
			else:
				if self.can_do(obs, actions.FUNCTIONS.select_army.id):
					return actions.FUNCTIONS.select_army('select')


		return actions.FUNCTIONS.no_op()


def main(argv):
	del argv

	agent = SimpleAgent()

	try:
		while True:
			with sc2_env.SC2Env(
				map_name='AbyssalReef',
				players=[sc2_env.Agent(sc2_env.Race.terran),
						sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
				agent_interface_format=features.AgentInterfaceFormat(
					feature_dimensions=features.Dimensions(screen=84, minimap=64),
					use_feature_units=True),
				step_mul=16,
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


if __name__ == '__main__':
	app.run(main)