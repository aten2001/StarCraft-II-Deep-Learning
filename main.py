import numpy as np
import pandas as pd
import pdb

from agents import SmartAgent

from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features


def main():
    agent = SmartAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name='DefeatRoaches',
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64), #should both be 64
                rgb_dimensions=features.Dimensions(screen=84, minimap=64),
                action_space=actions.ActionSpace.FEATURES),
                step_mul=16,
                game_steps_per_episode=0,
                visualize=True) as env:
                '''
                map_name: The map to play on
                players: A list of players for this map
                agent_interface_format: The size of the feature and rgb dimensions
                step_mul: The amount of game steps before an the agent take an action
                game_steps_per_episode: Length of the game. Set to 0 for unlimited
                visualize: Whether to render the game as viewed by a normal player
                '''
                agent.setup(obs_spec=env.observation_spec(), action_spec=env.action_spec())
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
    main()
