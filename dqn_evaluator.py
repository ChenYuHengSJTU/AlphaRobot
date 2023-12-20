import os
import numpy as np

from policy import Policy
from grid_map_env.classes.robot_state import RobotState
from grid_map_env.utils import sample_start_and_goal

from run_once import run_once


TASK_NUM = 20  # The number of tasks for each map
RUN_TIME = 10  # The number of times to run each task
STEP_LIMIT = 1000  # The maximum number of steps allowed for each run
TIME_LIMIT = 1.0  # The maximum thinking time in seconds for each step


def get_score(results):

    """
    Get final score using SPL measure.
    """

    results = np.array(results)
    values = (results[:, 0] * results[:, 2]) / np.maximum(results[:, 1], results[:, 2])
    average_value = np.mean(values)

    return average_value


def get_optimal_steps(start_pos, goal_pos):
    """
    Get optimal step number. It's the number of time steps moving from start position to goal position under the assumption that:
    - There's no obstacle on the shortest path.
    - The robot's strategy aims to maximize its speed while allowing only one turn during navigation. 
    - If a direction's path length doesn't permit accelerating to the maximum speed, the robot maintains a speed of 1 in that direction.
    """

    row_length = abs(start_pos[1]-goal_pos[1])
    col_length = abs(start_pos[0]-goal_pos[1])

    acc_length = 1+2 # length for accelerating to max speed

    acc_steps = 2 # step number for accelerating to max speed

    rotation_steps = 2  # decelerate to zero speed and rotate

    if row_length > acc_length*2: # permit acceleration and deceleration
        row_steps = (row_length-acc_length*2) // 3 + (row_length-acc_length*2) % 3 +  acc_steps*2 
    else:
        row_steps = row_length

    if col_length > acc_length*2:
        col_steps = (col_length-acc_length*2) // 3 + (col_length-acc_length*2) % 3 + acc_steps*2
    else:
        col_steps = col_length

    return row_steps+col_steps+rotation_steps


# 为一局游戏获取分数 -> reward
def get_reward(start_pos, end_pos, steps, is_goal:float) -> float:
    opt_steps = get_optimal_steps(start_pos, end_pos)
    # print(opt_steps, steps)
    # print(steps/opt_steps)
    return is_goal * float(opt_steps) / float(max(opt_steps, steps))
    pass


# 用来评价一个特定的策略
def evaluate():
    
    pass

if __name__ == "__main__":

    """
    Evaluator example with random sampled start position and goal position for each task.
    """

    current_directory = os.path.dirname(__file__)

    map_list = ['Collierville', 'Corozal',
                'Ihlen', 'Markleeville', 'Wiconisco']

    total_reward = 0



    results = []

    for map_name in map_list:
        map_file_path = os.path.join(
            current_directory, "grid_maps", map_name, "occ_grid_small.txt")

        for i in range(TASK_NUM):
            start_pos, goal_pos = sample_start_and_goal(map_file_path)

            optimal_steps = get_optimal_steps(start_pos, goal_pos)

            print("start position:", start_pos)
            print("goal position:", goal_pos)
            print("optimal step number:", optimal_steps)

            for j in range(RUN_TIME):

                policy = Policy()

                is_goal, steps = run_once(
                    map_file_path=map_file_path,
                    policy=policy,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    step_limit=STEP_LIMIT,
                    time_limit=TIME_LIMIT)

                print("finish", i+1, "task on map",
                      map_name, "for", j+1, 'round')

                results.append([is_goal, steps, optimal_steps])

    print("final score: ", get_score(results))
