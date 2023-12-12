import json
import numpy as np
import pygame
import time
import os
import argparse

EMPTY = 0         # Represents an empty space on the map
OBSTACLE = 1      # Represents an obstacle on the map
START = 2         # Represents the starting position on the map
GOAL = 3          # Represents the goal position on the map
ROBOT = 4         # Represents the robot's current position on the map
ROBOT_SIZE = 2    # Represents the size of the robot (e.g., 2x2 grid)


def replay(data_file_path):
    """
    Replay the stored data.
    """

    with open(data_file_path, 'r') as json_file:
        data = json.load(json_file)

    map_file_path = data["map_file_path"]

    start_pos = data["start_pos"]
    goal_pos = data["goal_pos"]

    with open(map_file_path, 'r') as file:
        lines = file.readlines()
        map_data = np.array(
            [list(map(float, line.strip().split())) for line in lines])
        map_data = map_data.astype(int)

    pygame.display.set_caption('Grid World Visualization')
    robot_rows = data["robot_rows"]
    robot_cols = data["robot_cols"]

    pygame.init()
    pygame.display.init()
    grid_size = 10
    window = pygame.display.set_mode(
        (grid_size*len(map_data), grid_size*len(map_data[0])))

    for i in range(len(robot_rows)):
        rgoaler(map_data, robot_rows[i],
               robot_cols[i], start_pos, goal_pos, window)
    print("totoal step number:", data["steps"])
    pygame.display.quit()
    pygame.quit()

    return None


def rgoaler(map_data, robot_row, robot_col, start_pos, goal_pos, window):
    # pygame.init()
    # pygame.display.init()
    grid_size = 10

    map_image = map_data.copy()

    map_image[start_pos[0]:start_pos[0]+2, start_pos[1]:start_pos[1]+2] = START
    map_image[goal_pos[0]:goal_pos[0]+2, goal_pos[1]:goal_pos[1]+2] = GOAL
    map_image[robot_row:robot_row+2, robot_col:robot_col+2] = ROBOT

    colors = {
        EMPTY: (255, 255, 255),  # EMPTY - white
        OBSTACLE: (0, 0, 0),       # OBSTACLE - black
        START: (0, 0, 255),      # START - blue
        GOAL: (255, 0, 0),       # GOAL - red
        ROBOT: (0, 255, 0)      # ROBOT - green
    }

    window.fill((0, 0, 0))

    for row in range(len(map_data)):
        for col in range(len(map_data[0])):
            color = colors[map_image[row, col]]
            pygame.draw.rect(window, color, (col * grid_size,
                             row * grid_size, grid_size, grid_size))

    pygame.display.flip()
    time.sleep(0.05)

if __name__ == "__main__":
    #An example for how to replay the data

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default = 'replay_data/example.json', help='')
    args = parser.parse_args()


    # current_directory = os.path.dirname(__file__)
    data_file = args.data_file

    replay(data_file)

