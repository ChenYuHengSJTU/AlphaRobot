import os
from policy import Policy

from run_once import run_once

TASK_NUM = 2000
MAP_NAME = 'Wiconisco'
if __name__ == "__main__":

    # and example for how to run TASK_NUM tasks in one map in headless mode and store the data

    current_directory = os.path.dirname(__file__)
    map_file_path = os.path.join(
        current_directory, "grid_maps",MAP_NAME,"occ_grid_small.txt")

    store_path = os.path.join(current_directory, "replay_data/samples/")

    # policy = Policy(0.6)
    # for i in range(TASK_NUM):
    #     print("run task", i+1)
    #     run_once(map_file_path=map_file_path,
    #              policy=policy,
    #              store=True,
    #              store_path=store_path,
    #              headless=True)


    policy = Policy()
    # policy = Policy(0)
    for i in range(TASK_NUM):
        print("run task", i)
        run_once(map_file_path=map_file_path,
                 policy=policy,
                 store=True,
                 store_path=store_path,
                 headless=True,
                 idx=i
                 )