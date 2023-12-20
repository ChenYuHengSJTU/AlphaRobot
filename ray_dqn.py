import gym
import ray
from ray.rllib.algorithms import ppo
from grid_map_env.classes.grid_map_env_ray import GridMapEnvCompile

from grid_map_env.utils import sample_start_and_goal
# from ray.rllib.algorithms.algorithm_config import config

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
config = AlgorithmConfig(
    # env=GridMapEnvCompile
)


start_pos, goal_pos = sample_start_and_goal("/home/cyh627/SJTU/Junior_1/AI/project/grid_maps/Wiconisco/occ_grid_small.txt")

config.environment(disable_env_checking=True)

ray.init()
algo = ppo.PPO(env=GridMapEnvCompile, config={
    "env_config": {
    "map_file_path": "/home/cyh627/SJTU/Junior_1/AI/project/grid_maps/Wiconisco/occ_grid_small.txt",
    "start_pos": start_pos,
    "goal_pos": goal_pos,
    "n":100,
    "headless":False
    }
},

                   
)

while True:
    print(algo.train())
