from stable_baselines3 import DQN
import os
import threading
import datetime
import json
import gym
import warnings
import time
import random
import numpy as np
# import gym
# import os

from grid_map_env.classes.robot_state import RobotState
from grid_map_env.utils import sample_start_and_goal

MAP_NAME="Wiconisco"

current_directory = os.path.dirname(__file__)

map_file_path = os.path.join(current_directory, "grid_maps",MAP_NAME,"occ_grid_small.txt")
start_pos, goal_pos = sample_start_and_goal(map_file_path)

env = gym.make("grid_map_env/GridMapEnv-v0", n=100,
                   map_file_path=map_file_path, start_pos=start_pos, goal_pos=goal_pos, headless=True)

                   
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save(MAP_NAME)
