import os
import gym
from gym import spaces
import importlib.util

from grid_map_env.utils import EMPTY, OBSTACLE, START, GOAL, ROBOT, ROBOT_SIZE

"""
Origin grid_map_env.py has been compiled into grid_map_env.cpython-38.pyc.
class GridMapEnvCompile is a new class that re-wrap the module into gym environment
"""
path = os.path.dirname(__file__)
bytecode_file_path = path = os.path.join(path, "__pycache__", "grid_map_env.cpython-38.pyc")
module_spec = importlib.util.spec_from_file_location('GridMapEnv', bytecode_file_path)
GridMapEnv_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(GridMapEnv_module)

class GridMapEnvCompile(gym.Env):
    """
    A gym environment for AlphaRobot grid map envrionment. Its interfaces are the same with origin GridMapEnv class.
    Member variables:
        observation space (gym.spaces.Dict): observation space specifies the format of observations. 
            It is a dictionary of Gym-supported spaces. The dictionary keys include:
            - “map” (gym.spaces.Box): an n × n matrix of integers representing the occupancy grid of the map. 
            - robot pos” (gym.spaces.Box): a list of two integers representing the robot position (row, col) on the map.
            - “robot speed” (gym.spaces.Discrete): one integer indicating the speed of the robot.
            - “robot direction” (gym.spaces.Discrete): one integer indicating the facing direction of the robot. 
    """
    def __init__(self, n, map_file_path, start_pos, goal_pos, headless=False):
        """
        Initializes the AlphaRobot grid map environment. 
        Args: 
            n (int): an integer indicating the side length of the map.
            map file path (str): the path to the text file storing the map.
            start pos (tuple): the robot's start position (row, col) on the map.
            goal pos (tuple): the robot's goal position (row, col) on the map.
            headless (bool): whether to run the environment in headless mode (without ren-dering). 
                The default value is False (with rendering).
        """

        super(GridMapEnvCompile, self).__init__()

        self.env = GridMapEnv_module.GridMapEnv(
                                                n=n,     
                                                map_file_path=map_file_path, 
                                                start_pos=start_pos, 
                                                goal_pos=goal_pos, 
                                                headless=headless)

        self.observation_space = spaces.Dict( 
            {
                "map": spaces.Box(low=0, high=GOAL, shape=(n, n), dtype=int),
                "robot_pos": spaces.Box(low=0, high=n-1, shape=(2,), dtype=int),
                "robot_speed": spaces.Discrete(4),  # 0,1,2,3
                "robot_direction": spaces.Discrete(4)  # 0: left, 1: up, 2: right, 3: down
            }
        )
        self.action_space = spaces.Discrete(5)  # five kind of actions

    def step(self, action):
        """
        Execute a given action, transit the robot state, and feed the observation and reward back to the robot.
        Args: 
            action (Action): an instance of Action class representing the action to be executed.
        
        Returns: 
            observation (gym.spaces.Dict): the observation received after executing the action. 
                It follows the format specified in self.observation space.
            step number (int): the current time step.
            terminated (bool): whether the task has terminated (due to success or failure).
            is goal (bool): whether the robot has stopped at the goal.
            info (dict): a dummy dictionary here, used to match the Gym interface.
        """
        return self.env.step(action)

    def reset(self):
        """
        Reset the robot state to the initial configuration and return an observation after resetting.
        """
        return self.env.reset()

    def render(self):
        """
        Render the environment, including the map, the start and goal positions, and the
robot.

        - White represents empty space.
        - Black represents obstacles.
        - Blue represents the starting point.
        - Red represents the ending point.
        - Green represents the robot's position.
        """

        self.env.render()

    def close(self):
        """
        Stops and quit the rendering window.
        """

        self.env.close()
