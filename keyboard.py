from scipy import ndimage
import gym
import os
import datetime
import json
import pygame
import warnings
import argparse

from grid_map_env.classes.robot_state import RobotState
from grid_map_env.classes.action import Action
from grid_map_env.utils import sample_start_and_goal


def play_with_keyboard(map_file_path, start_pos=None, goal_pos=None, store=False, store_path="~/",step_limit=1000):
    """
    Control the robot to complete a navigation task by keyboard.
    """
    if start_pos == None or goal_pos == None:
        start_pos, goal_pos = sample_start_and_goal(map_file_path)

    if start_pos == None or goal_pos == None:
        start_pos, goal_pos = sample_start_and_goal(map_file_path)

    warnings.filterwarnings("ignore", category=UserWarning, module="gym")

    env = gym.make("grid_map_env/GridMapEnv-v0", n=100,
                   map_file_path=map_file_path, start_pos=start_pos, goal_pos=goal_pos)

    initial_observation, _ = env.reset()

    map = initial_observation["map"]

    robot_state = RobotState(row=initial_observation["robot_pos"][0], col=initial_observation["robot_pos"]
                             [1], speed=initial_observation["robot_speed"], direction=initial_observation["robot_direction"])
    # print(robot.row,robot.col,robot.speed,robot.direction)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    json_file_path = os.path.join(store_path, f'data_{current_time}.json')

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
        robot_rows.append(int(robot_state.row))
        robot_cols.append(int(robot_state.col))
        robot_speeds.append(int(robot_state.speed))
        robot_directions.append(int(robot_state.direction))

    for _ in range(step_limit):
        waiting_for_input = True
        while (waiting_for_input):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        action = Action(1, 0)
                        waiting_for_input = False
                    elif event.key == pygame.K_s:
                        action = Action(-1, 0)
                        waiting_for_input = False
                    elif event.key == pygame.K_a:
                        action = Action(0, -1)
                        waiting_for_input = False
                    elif event.key == pygame.K_d:
                        action = Action(0, 1)
                        waiting_for_input = False
                    elif event.key == pygame.K_SPACE:
                        action = Action(0, 0)
                        waiting_for_input = False
            # else:
            #     continue
        # action = result_list[0]
        observation, episode_length, terminated, is_goal, _ = env.step(action)
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
            print("finish!")
            print("total step number: ", episode_length)
            env.close()
            break
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

    #and example for how to run a navigation task for one time and store the data

    current_directory = os.path.dirname(__file__)
    map_file_path = os.path.join(current_directory, "grid_maps","Wiconisco","occ_grid_small.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=bool, default=False, help='whether to store data')
    parser.add_argument("--store_path", type=str, default = 'replay_data', help='folder to store your data')
    args = parser.parse_args()

    # store_path = os.path.join(current_directory, "replay_data")
    store = args.store
    store_path = args.store_path

    play_with_keyboard(map_file_path=map_file_path,
             store=store,
             store_path=store_path)