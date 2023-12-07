import numpy as np
import random
from scipy import ndimage
import gym
import os
import datetime
import json
import time
import pygame

from grid_map_env.classes.robot_state import RobotState
from grid_map_env.classes.action import Action

# Global constants for the environment, including map encoding and robot size
EMPTY = 0         # Represents an empty space on the map
OBSTACLE = 1      # Represents an obstacle on the map
START = 2         # Represents the starting position on the map
GOAL = 3          # Represents the goal position on the map
ROBOT = 4         # Represents the robot's current position on the map
ROBOT_SIZE = 2    # Represents the size of the robot (e.g., 2x2 grid)

# ****************************************************************************************************
# Function below are intended for internal use only. Please do not use them directly.
# ****************************************************************************************************


def find_connected_components(map_matrix):
    for i in range(len(map_matrix)):
        for j in range(len(map_matrix[0])):
            if map_matrix[i, j] == 1:
                if i > 0 and j > 0:
                    map_matrix[i-1:i+1, j-1:j+1] = 1
                elif i > 0:
                    map_matrix[i-1:i+1, j] = 1
                elif j > 0:
                    map_matrix[i, j-1:j+1] = 1

    labeled_array, num_features = ndimage.label(map_matrix == 0)
    connected_components = []

    for label in range(1, num_features + 1):
        component = (labeled_array == label).astype(int)
        connected_components.append(component)

    return connected_components


def get_goal_pos(house_map):
    goal_pos_row, goal_pos_col = np.where(np.array(house_map) == GOAL)
    return goal_pos_row[0], goal_pos_col[0]
# **************************************************
# Function below are tools provided for you to use.
# **************************************************


def sample_start_and_goal(map_file_path):
    """
    sample a pair of feasible start position and goal position.
    - Both the start position and goal position are collision-free.
    - There exists at least one feasible path from the start position to the goal position.
    - The Manhattan distance between the start and the goal should be no smaller than 50.

    Args:
        map file path (string): a string representing the path to path file (text).

    Returns:
        start pos (tuple): a tuple of two integers representing the sampled start (row, col)
        goal pos (tuple): a tuple of two integers representing the sampled goal (row, col).

    """

    with open(map_file_path, 'r') as file:
        lines = file.readlines()
        map_data = np.array(
            [list(map(float, line.strip().split())) for line in lines])
        map_data = map_data.astype(int)

    connected_components = find_connected_components(map_data)

    while (True):
        random_component = random.randint(0, len(connected_components)-1)
        if np.count_nonzero(connected_components[random_component]) < 2000:
            continue
        indices = np.where(connected_components[random_component] == 1)

        (start_row, start_col) = random.choice(
            list(zip(indices[0], indices[1])))
        (goal_row, goal_col) = random.choice(list(zip(indices[0], indices[1])))

        if np.any(map_data[start_row:start_row+ROBOT_SIZE, start_col:start_col+ROBOT_SIZE] == 1):
            continue
        if np.any(map_data[goal_row:goal_row+ROBOT_SIZE, goal_col:goal_col+ROBOT_SIZE] == 1):
            continue
        if (abs(start_row-goal_row) + abs(start_col-goal_col)) < 50:
            continue

        break

    return (start_row, start_col), (goal_row, goal_col)


def is_collision(house_map, robot_state):
    """
    Check if a collision will occur in current state.

    Args:
        house map (list of list): a 2D list of integers representing the house map.
        robot state (RobotState): an instance of the RobotState class representing the current state of the robot.

    Returns:
        bool: True if any collision will occur, False otherwise.
    """

    col = robot_state.col
    row = robot_state.row

    if col < 0 or col+ROBOT_SIZE > len(house_map[0]) or row < 0 or row+ROBOT_SIZE > len(house_map[0]):
        return True
    sub_map = np.array(house_map)[row:row+ROBOT_SIZE, col:col+ROBOT_SIZE]
    collision = np.any(sub_map == 1)

    return collision


def is_goal(house_map, robot_state):
    """
    Check if the robot has reached the goal position on the house map.
    Only if the robot stops at the goal position (speed is zero) that it reaches the goal.

    Args:
        house map (list of list): a 2D list of integers representing the house map.
        robot state (RobotState): an instance of the RobotState class representing the current state of the robot.

    Returns:
        bool: True if the robot is at the goal position with speed 0, False otherwise.
    """

    goal_pos_row, goal_pos_col = np.where(np.array(house_map) == GOAL)
    if robot_state.speed == 0 and goal_pos_row[0] == robot_state.row and goal_pos_col[0] == robot_state.col:
        return True
    return False
