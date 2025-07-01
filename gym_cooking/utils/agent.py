# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
import navigation_planner.utils as nav_utils

# Other core modules
from utils.core import Counter, Cutboard
from utils.utils import agent_settings

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple
from abc import ABCMeta, abstractmethod

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']

# You should not need to import outside libraries for this assignment.
# Reach out to us if you need to use a non-standard library.


##################################################################################
# README! --- Agent Classes
# Here, we define:
# 1) BaseAgent: Base class for all agents. 
#       - IMPORTANT: Read through this class to understand the basic structure
#           and to understand the basic variables and functions available to you.
#       - You do not need to modify this class, but feel free to do so if you want.
#
# 2) YourAgent: Your agent that performs task inference and plans.
#       - This class is a template for your agent implementation.
#       - You must modify this class to implement your own agent.
#       - You can add your own class variables and functions.
#       - Feel free to rename this class and to add new agent classes to suit
#           your own implementation and strategic choices.
##################################################################################
class BaseAgent:
    """Base Agent object: READ THROUGH THIS CLASS TO UNDERSTAND THE STRUCTURE."""
    #-----Feel free to add your own class variables-----
    
    def __init__(self, arglist, name, id_color, recipes):
        #-----Feel free to add to the init procedure-----
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes
        self.holding = None

        self.model_type = agent_settings(arglist, name)
        self.priors = 'random'

        # Navigation planner.
        self.planner = None

    # Default representation prints agent name in colour
    def __str__(self):
        return color(self.name[-1], self.color)

    # Defines copy method to duplicate your agents
    def __copy__(self, ):
        a = self.__class__(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    # Helper for printing agent holding state & object
    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    # IMPORTANT: This function is called by the environment to get the next action.
    # You must implement this function in your agent class.
    @abstractmethod
    def select_action(self, obs):
        """Return next action for this agent given observations."""
        pass

    # Helper for updating player position for navigation
    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))


##################################################################################
# README! --- TASK: Implement your agent(s) here. 
#
# YourAgent: Your agent that performs task inference and plans.
#   - This class is a template for your agent implementation.
#   - You must modify this class to implement your own agent.
#   - You can add your own class variables and functions.
#   - Feel free to rename this class and to add new agent classes to suit
#       your own implementation and strategic choices.
##################################################################################
class YourAgent(BaseAgent):
    """Your Agent object that performs task inference and plans."""
    #-----Feel free to add your own class variables-----
    
    def __init__(self, arglist, name, id_color, recipes):
        super().__init__(arglist, name, id_color, recipes)

        #-----Feel free to add your own init procedure-----
        # self.your_var = your_var
        # self.new_value = your_function()

        # Navigation planner.
        #-----Feel free to replace with your own planner-----
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)
    
    # MAIN TASK: replace with your own action selection logic
    def select_action(self, obs):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action
        
        actions = nav_utils.get_single_actions(env=copy.copy(obs), agent=self) # returns a list of VALID actions    
        return actions[np.random.choice(len(actions))] # currently chooses a random action
    
    #-----Feel free to add your own helper functions-----


##################################################################################
# Feel free to create new agent classes below that inheret from BaseAgent.
##################################################################################


##################################################################################
# README! --- SimAgent: Simulation agent used in the environment object.
# IMPORTANT: Do NOT modify this class.
# This class is used in the environment object to represent agents.
# Read through the class to understand which variables interact with 
# the environment directly. Agents can aquire objects, hold objects,
# release objects, and move around the environment.
# NOTE: merging objects is destructive: this action cannot be undone. Be careful!
# Merging examples: placing an ingredient on a plate, cutting an ingredient, etc.
##################################################################################
class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.has_delivered = False

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color,
                location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self):
        return AgentRepr(name=self.name, location=self.location, holding=self.get_holding())

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def print_status(self):
        print("{} currently at {}, action {}, holding {}".format(
                color(self.name, self.color),
                self.location,
                self.action,
                self.get_holding()))

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj) # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location
