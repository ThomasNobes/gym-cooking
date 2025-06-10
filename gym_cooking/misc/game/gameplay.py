# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
import pickle
from datetime import datetime
from navigation_planner.utils import get_single_actions


class GamePlay(Game):
    def __init__(self, env):
        Game.__init__(self, env.world, env.sim_agents, play=True)
        self.env = env
        self.filename = env.filename
        self.save_dir = 'misc/game/screenshots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                print('just saved image {} to {}'.format(image_name, self.save_dir))
                return
            
            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return

            # Control current agent
            x, y = self.current_agent.location
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.current_agent.action = action
                # add wait action for all other agents
                action_dict = {}
                for agent in self.sim_agents:
                    action_dict[agent.name] = (0, 0)
                action_dict[self.current_agent.name] = action
                obs, reward, done, info = self.env.step(action_dict)


    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running and not self.env.done():
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()


    def get_action_from_locations(self, loc1, loc2):
        if loc1[0] == loc2[0]:
            if loc1[1] < loc2[1]:
                return (0, 1)
            else:
                return (0, -1)
        elif loc1[1] == loc2[1]:
            if loc1[0] < loc2[0]:
                return (1, 0)
            else:
                return (-1, 0)
        else:
            return (0, 0)


    def on_replay_event(self, event, solution, current_timestep):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:            
            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return current_timestep
            
            # do nothing if we have not pressed a valid key
            if event.key not in KeyToTuple.keys():
                return current_timestep
        
            action = KeyToTuple[event.key]
            if action == (0, 1) or action == (1, 0):
                time_dir = 1
            elif action == (0, -1) or action == (-1, 0):
                time_dir = -1
            max_time = max([len(agent_solution) for agent_solution in solution])
            new_timestep = max(0, min(max_time-1, current_timestep + time_dir))
            if new_timestep == current_timestep:
                return current_timestep
            print("timestep:", current_timestep, new_timestep, time_dir)
            current_timestep = new_timestep
            
            # move each agent according to the solution
            for agent_id in range(len(self.sim_agents)):
                
                # ensures that we cannot go beyond available timesteps in the solution
                if current_timestep < 0:
                    print(f"Cannot move backwards in time: simulation starts at timestep 0")
                    continue
                elif current_timestep >= len(solution[agent_id]):
                    print(f"Cannot move forwards in time: simulation ends at timestep {len(solution[agent_id]) - 1}")
                    continue
                
                # grabs the next action for the agent at the current timestep
                current_location = self.sim_agents[agent_id].location
                next_action = solution[agent_id][current_timestep]
                new_location = tuple(np.asarray(current_location) + np.asarray(next_action))
                if time_dir > 0:
                    print(f"agent {agent_id} step FORWARD: moves {ActionToString[next_action]} from {current_location} to {new_location} at timestep {current_timestep}")
                elif time_dir < 0:
                    print(f"agent {agent_id} step BACKWARD: moves {ActionToString[next_action]} from {new_location} to {solution[agent_id][current_timestep]} at timestep {current_timestep-time_dir}")
                
                # # sanity checking valid moves
                # valid_moves = get_single_actions(self.world, self.sim_agents[agent_id]) # TODO: figure out how to access world observations
                # assert next_action not in valid_moves, f"Invalid move: {next_action} from {current_location} to {new_location}"
                
                # perform next action in simulator
                self.sim_agents[agent_id].action = next_action
                interact(self.sim_agents[agent_id], self.world)
                print("new_location:", self.sim_agents[agent_id].location)
        return current_timestep


    def on_replay(self, replay_fname):
        """takes solution: table of actions for each agent at each timestep"""
        solution = self.read_solution(replay_fname)
        if self.on_init() == False:
            self._running = False
        
        print("number of agents:", len(self.sim_agents))
        assert len(solution) == len(self.sim_agents), "Number of agents in solution does not match number of agents in simulation"
        assert all([len(agent_solution) == len(solution[0]) for agent_solution in solution]), "All agents must have the same number of timesteps (include wait actions)"

        current_timestep = 0
        while self._running:
            for event in pygame.event.get():
                current_timestep = self.on_replay_event(event, solution, current_timestep)
            self.on_render()
        self.on_cleanup()


    def read_solution(self, fname):
        data = pickle.load(open(fname, "rb"))
        return [data['actions'][agent] for agent in data['actions'].keys()]

