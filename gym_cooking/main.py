# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import YourAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple

import gym


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--replay", type=str, default=None, help="Visualise a replay of a saved game. Example usage: python main.py --replay misc/metrics/pickles/partial-divider_salad_agents2_seed1_model1-random_model2-random.pkl")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (random, user, advanced)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (random, user, advacned)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (random, user, advacned)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (random, user, advacned)")

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def initialize_agents(arglist):
    agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(agents) < arglist.num_agents:
                    loc = line.split(' ')
                    agent = YourAgent(
                            arglist=arglist,
                            name='agent-'+str(len(agents)+1),
                            id_color=COLORS[len(agents)],
                            recipes=recipes)
                    agents.append(agent)

    return agents


def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    agents = initialize_agents(arglist=arglist)    

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)

    while not env.done():
        action_dict = {}

        for agent in agents:
            action = agent.select_action(obs=obs)
            action_dict[agent.name] = action

        # NOTE: read /envs/overcooked_environment.py
        obs, reward, done, info = env.step(action_dict=action_dict)

        # Agents
        # for agent in agents:
        #     if type(agent) == YourAgent:
        #         agent.refresh_subtasks(world=env.world)

        # Saving info
        bag.add_status(cur_time=info['t'], agents=agents)

    # Saving final information before saving pkl file
    bag.set_collisions(collisions=env.collisions)
    bag.set_termination(termination_info=env.termination_info,
            successful=env.successful)


if __name__ == '__main__':
    arglist = parse_arguments()
    # test environment for manually playing the game
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env)
        # game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_execute()
    # test environment for replaying a saved game
    elif arglist.replay is not None:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_replay(arglist.replay)
    # test environment for running experiments
    else:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)
