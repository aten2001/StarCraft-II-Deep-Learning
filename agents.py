import random
import math

import numpy as np
import pandas as pd
import pdb
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from models import RelationalNN as relnn
from models import ResNet, MLP


'''
Actions and Variables
'''

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]



'''
Agents
'''
class SmartAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(SmartAgent, self).step(obs)

        pdb.set_trace()
        # print(obs.observation['feature_minimap'].shape)
        # print(obs.observation['feature_screen'].shape)
        # print(obs.observation['last_actions'])
        # print(obs.observation['player'])
        (minimap, screen, player, last_action) = self.input_preprocessing(obs.observation)
        return actions.FUNCTIONS.no_op()

    def input_preprocessing(self, observation):
        minimap = np.log(observation['feature_minimap'] + 1)
        screen = np.log(observation['feature_screen'] + 1)
        player = np.log(observation['player'] + 1)
        last_action = observation['last_actions'] / 100
        return (minimap, screen, player, last_action)

    # def train(self):
