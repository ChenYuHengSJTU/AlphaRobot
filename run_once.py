import os
from policy import Policy
from grid_map_env.classes.robot_state import RobotState
from grid_map_env.utils import sample_start_and_goal
import threading
import datetime
import json
import gym
import warnings

MAP_NAME="winconisco"

def run_once(map_file_path, policy, start_pos=None, goal_pos=None, store=False, store_path="~/", step_limit=1000, time_limit=1.0, headless=False):
    """
    Run a navigation task once.

    Parameters:
        map_file_path (str): The file path to the original text file storing the map.
        policy (Policy): An instance of your policy class.
        start_pos (tuple) and goal_pos (tuple): The start position (row, col) and goal position (row, col).
            If one of them is None, both will be replaced by randomly sampled positions.
        store (bool): Whether to store data. If True, data will be saved as a JSON file, including:
            - map_file_path (str): The file path to the original text file storing the map.
            - start_pos (tuple): The start position (row, col).
            - goal_pos (tuple): The goal position (row, col).
            - robot_rows (list): List of row indices of the robot's position for each state.
            - robot_cols (list): List of column indices of the robot's position for each state.
            - robot_speeds (list): List of robot's speed for each state.
            - robot_directions (list): List of robot's direction for each state.
            - action_accs (list): List of accelerations for each action. Note that the number of actions is one less than the number of states.
            - action_rots (list): List of rotations for each action.
            - steps (int): Total number of time steps.
            - is_goal (bool): Whether the robot has reached the goal.
        store_path (str): The file path to store the data (if store is True).
        step_limit (int): The maximum number of steps allowed. None means no step limit.
        time_limit (float): The maximum thinking time in seconds for each step.
        headless (bool): Whether to run in headless mode without visualization.

    Returns:
        is_goal (bool): Whether the robot has reached the goal.
        steps (int): Total number of time steps.
    """

    if start_pos == None or goal_pos == None:
        start_pos, goal_pos = sample_start_and_goal(map_file_path)

    warnings.filterwarnings("ignore", category=UserWarning, module="gym")

    env = gym.make("grid_map_env/GridMapEnv-v0", n=100,
                   map_file_path=map_file_path, start_pos=start_pos, goal_pos=goal_pos, headless=headless)

    initial_observation, _ = env.reset()

    map = initial_observation["map"]

    robot_state = RobotState(row=initial_observation["robot_pos"][0], col=initial_observation["robot_pos"]
                             [1], speed=initial_observation["robot_speed"], direction=initial_observation["robot_direction"])
    # print(robot.row,robot.col,robot.speed,robot.direction)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')



    json_dict = {}
    json_dict["map_file_path"] = map_file_path
    json_dict["start_pos"] = (int(start_pos[0]), int(start_pos[1]))
    json_dict["goal_pos"] = (int(goal_pos[0]), int(goal_pos[1]))

    robot_rows = []
    robot_cols = []
    robot_speeds = []
    robot_directions = []
    action_accs = []
    action_rots = []

    if store:
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        json_file_path = os.path.join(store_path, f'data_{current_time}.json')
        
        robot_rows.append(int(robot_state.row))
        robot_cols.append(int(robot_state.col))
        robot_speeds.append(int(robot_state.speed))
        robot_directions.append(int(robot_state.direction))

    for _ in range(step_limit):
        result_list = []
        thread = threading.Thread(target=lambda: result_list.append(
            policy.get_action(map, robot_state)))
        thread.start()
        thread.join(timeout=time_limit)

        if thread.is_alive():
            # thread.terminate()
            print("excution time larger than 1s")
            terminated = True

        action = result_list[0]

        observation, curr_steps, terminated, is_goal, _ = env.step(action)
        robot_state.row = observation["robot_pos"][0]
        robot_state.col = observation["robot_pos"][1]
        robot_state.speed = observation["robot_speed"]
        robot_state.direction = observation["robot_direction"]

        if store:
            robot_rows.append(int(robot_state.row))
            robot_cols.append(int(robot_state.col))
            robot_speeds.append(int(robot_state.speed))
            robot_directions.append(int(robot_state.direction))
            action_accs.append(int(action.acc))
            action_rots.append(int(action.rot))

        if terminated:
            episode_length = curr_steps
            print("finish!")
            print("total step number: ", episode_length)
            env.close()
            break
        if not headless:
            env.render()
        
    if store:
        json_dict["robot_rows"] = robot_rows
        json_dict["robot_cols"] = robot_cols
        json_dict["robot_speeds"] = robot_speeds
        json_dict["robot_directions"] = robot_directions
        json_dict["action_accs"] = action_accs
        json_dict["action_rots"] = action_rots
        json_dict["steps"] = episode_length
        json_dict["is_goal"] = is_goal

        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file)

    return is_goal, episode_length


if __name__ == "__main__":


    # and example for how to run a navigation task for one time and store the data

    current_directory = os.path.dirname(__file__)
    map_file_path = os.path.join(current_directory, "grid_maps",MAP_NAME,"occ_grid_small.txt")

    store_path = os.path.join(current_directory, "replay_data")

    policy = Policy()

    run_once(map_file_path=map_file_path,
             policy=policy,
             store=True,
             store_path=store_path)
