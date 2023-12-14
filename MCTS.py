import math
import os
from policy import Policy
from grid_map_env.classes.robot_state import RobotState
from grid_map_env.utils import sample_start_and_goal
import threading
import datetime
import json
import gym
import warnings
from grid_map_env.classes.action import Action
import copy
import random
import time

MAP_NAME="Wiconisco" # "Markleeville"

DEBUG = False

"""
(20, 4)
(20, 16)
"""

class MCTS():
    def __init__(self, map_file_path, iterations, start_pos, init_distance, init_speed, condition, args):
        """
        We set the noise = 0.1 (Or use the env directly)
        We want to find a dict,
        with key=(distance, speed, condition, args), value=-1 0 1(means the accelerate)
        init_distance from 12 to 0
        init_speed from 3 to 0
        condition includes:
        1. -------\/
        2. ------
                |
            1   |   2
                \/
        3. ------
                |
            1   |   2
                ------->
        4. ------
                |
            1   |   2
           <-----
        args: tuple(bounce, bounder1, bounder2)
        bouce include(1,2,3,>=4)
        bounder1 include(1,2,3,>=4)
        bounder2 include(1,2,3,>=4)
        """
        self.err_rate = 0.1

        # self.total_node = 0
        self.rvse = 2.2 # turn around
        self.min_cost = (0, 2.78, 4.06, 5.41, 5.7, 8, 8.5, 10, 10.6, 20, 30, 30, 40) # (0,2,3,4,4,5,5,6,6,6)
        self.factor = 1000
        self.MCTree = {} 

        # key(distance, speed, action), all nodes are chance nodes
        # value[num, utility]
        # utility means the -(min_cost-actual_res) 

        # state means the (distance, speed)
        # node means the (distance, speed, acc)

        self.iterations = iterations
        self.start_pos = start_pos
        self.goal_pos = (start_pos[0], start_pos[1]-init_distance)
        self.init_distance = init_distance
        self.init_speed = init_speed

        self.condition = condition
        self.args = args

        if DEBUG:
            print("\t[start_pos]: ", self.start_pos)
            print("\t[goal_pos]: ", self.goal_pos)
            print("\t[init_speed]: ", self.init_speed)
            print("\t[condition]:", condition)
            print("\t[Args]:", args)
        
        # self.env = gym.make("grid_map_env/GridMapEnv-v0", n=100,
        #            map_file_path=map_file_path, start_pos=start_pos, goal_pos=self.goal_pos, headless=True)
        
        self.path = []
        self.terminal_flag = False
        self.bounce_flag = False

        # self.before_start()

    # def before_start(self):
    #     initial_observation, _ = self.env.reset()
        
    #     speed=initial_observation["robot_speed"]
    #     direction=initial_observation["robot_direction"]
    #     # robot_state = RobotState(row=initial_observation["robot_pos"][0], col=initial_observation["robot_pos"]
    #     #                         [1], speed=initial_observation["robot_speed"], direction=initial_observation["robot_direction"])
    #     while direction != 0:
    #         observation, curr_steps, terminated, is_goal, _ = self.env.step(Action(0, 1))
    #         direction=observation["robot_direction"]
        
    #     print("\tInitial state ensured!")
    #     print("\tDirection=0 Speed=0")


    def cal_l1_distance(self, pos1, pos2):
        """
        pos1: (x1, y1)
        pos2: (x2, y2)
        return the l1 distance
        """
        return abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1])

    def Environment(self, Node):
        """
        Node: a chance node
        speed: the current speed

        return a state
        """
        distance, speed, acc = Node

        speed_o = copy.deepcopy(speed)

        x = 0
        if acc == 0 :
            x = random.choices([-1, 0, 1], weights=[0.05, 0.9, 0.05])[0]
        elif acc == 1:
            x = random.choices([0, 1], weights=[0.1, 0.9])[0]
        elif acc == -1:
            x = random.choices([0, -1], weights=[0.1, 0.9])[0]

        speed += x
        if speed < 0:
            speed = 0
        elif speed > 3:
            speed = 3

        distance -= speed

        if distance <= -args[0]:
            self.bounce_flag = True
            speed = 0
            distance = -args[0] + speed_o + 1
        return (distance, speed)



    def UCB_heuristics(self, Node):
        """
        Given a real node
        return a chance node
        """
        # 可以选择在中间点停，但是不能选择在起点停。
        if Node == (self.init_distance, self.init_speed) and self.init_speed == 0:
            return (Node[0], Node[1], 1)
        
        num1, util1, = self.MCTree[(Node[0], Node[1], 1)]
        num2, util2, = self.MCTree[(Node[0], Node[1], -1)]
        num3, util3, = self.MCTree[(Node[0], Node[1], 0)]

        parent_num = num1 + num2 + num3
        UCB_ac = util1 / (num1 + 1) + self.factor * math.sqrt(math.log(parent_num + 1) / (num1 + 1))
        UCB_de = util2 / (num2 + 1) + self.factor * math.sqrt(math.log(parent_num + 1) / (num2 + 1))
        UCB_ke = util3 / (num3 + 1) + self.factor * math.sqrt(math.log(parent_num + 1) / (num3 + 1))

        # if Node[1] == 0:
        #     UCB_ke = -1

        if UCB_de >= UCB_ac and UCB_de >= UCB_ke:
            return (Node[0], Node[1], -1)
        if UCB_ke >= UCB_ac and UCB_ke >= UCB_de:
            return (Node[0], Node[1], 0)
        if UCB_ac >= UCB_de and UCB_ac >= UCB_ke:
            return (Node[0], Node[1], 1)
        
        assert 2 == 0


    def select_and_expand(self):
        """
        return a list[] -> node
        """
        self.terminal_flag = False
        self.bounce_flag = False
        self.path = []
        state = (self.init_distance, self.init_speed)
        node = None
        while True:
            if state[0] < 0 :
                return state

            if state[0] == 0 and state[1] == 0:
                return state     
                       
            if (state[0], state[1], 1) not in self.MCTree.keys():
                self.MCTree[(state[0], state[1], 1)] = [0, 0]
                self.path.append((state[0], state[1], 1))
                return self.Environment((state[0], state[1], 1))
            
            if (state[0], state[1], -1) not in self.MCTree.keys(): 
                self.MCTree[(state[0], state[1], -1)] = [0, 0]
                self.path.append((state[0], state[1], -1))
                return self.Environment((state[0], state[1], -1))
            
            if (state[0], state[1], 0) not in self.MCTree.keys(): 
                self.MCTree[(state[0], state[1], 0)] = [0, 0]
                self.path.append((state[0], state[1], 0))
                return self.Environment((state[0], state[1], 0))
            
            node = self.UCB_heuristics(state)

            self.path.append(node)

            state = self.Environment(node)

            if self.bounce_flag:
                return state

            if node[2] == -1 and node[1] == 0:
                self.terminal_flag = True
                return state


    def rollout(self, start_state):
        """
        start_state: (distance, speed)

        return:
            Steps when decelerate to 0.
            distance when stopped: positive means not arriving, negative means exceeding.
        """
        if self.terminal_flag:
            return 0, start_state[0]
        count = 0
        cur_speed = start_state[1]
        distance = start_state[0]
        while cur_speed > 0:
            distance, cur_speed = self.Environment((distance, cur_speed, -1))
            # observation, curr_steps, terminated, is_goal, _ = self.env.step(Action(-1, 0))
            count += 1
            # cur_speed = observation["robot_speed"]
            # row = observation["robot_pos"][0]
            # col = observation["robot_pos"][1]
            # distance = (abs(self.start_pos[0] - self.goal_pos[0]) + abs(self.start_pos[1] - self.goal_pos[1]))\
            #              - (abs(row - self.start_pos[0]) + abs(col - self.start_pos[1])) 
 
            # negative means 冲过头 positive means 没到
        return count, distance
        
    # 5 6 7
    def con_0(self, distance):
        if distance <= 0:
            bias = 0
            if distance in [-1, -2, -3]:
                bias = self.rvse
            elif distance >= -4:
                bias = self.rvse
            return self.min_cost[abs(distance)] + bias # averaged bias
        if distance > 0:
            return self.min_cost[distance] + 2


    def con_1(self, distance):
        if distance == 0:
            return 0
        if  distance <0 :
            if args[2] + distance <= 0:
                return self.rvse + self.min_cost[abs(distance)] + 0.2
            else:
                return self.rvse // 2 + self.min_cost[abs(distance)]
        if distance > 0:
            return self.min_cost[abs(distance)]

    def con_2(self, distance):
        if distance == 0:
            return 0
        if distance < 0:
            if args[2] + distance <= 0:
                return self.rvse + self.min_cost[abs(distance+args[2]-1)]
            else:
                return -0.2
        if distance > 0:
            if distance - args[1] >= 0:
                return  self.min_cost[abs(distance)] + 0.5
            else:
                return 0.3
            
    def con_3(self, distance):
        if distance == 0:
            return 0
        if distance < 0:
            if args[2] + distance <= 0:
                return self.rvse + self.min_cost[abs(distance)]
            else:
                return 1
        if distance > 0:
            if distance - args[1] >= 0:
                return self.min_cost[abs(distance)]
            else:
                return -1

    def con_actual(self, distance):
        pass

    def cal_util(self, distance):
        # return self.con_actual(distance)

        if self.condition == 0:
            return self.con_0(distance)
        elif self.condition == 1:
            return self.con_1(distance)
        elif self.condition == 2:
            return self.con_2(distance)
        elif self.condition == 3:
            return self.con_3(distance)
        
    def backpropagate(self, steps):
        if self.path == None:
            assert 1 == 0
        self.path.reverse()
        for idx, i in enumerate(self.path):
            num, util = self.MCTree[i]
            if idx == 0 and i[1] == 0 and i[2] == -1:
                steps -= 1
            if idx != 0 and i == self.path[idx-1]:
                util -= 1
            else:
                num += 1
                util -= (steps + idx + 1) # has chosen. or i+1
            self.MCTree[i] = [num, util]


    def begin_search(self):
        for i in range(self.iterations):
            # ini, _ = self.env.reset()
            # assert ini["robot_speed"] == 0
            # if i % 1 == 0:
            #     print("\tIter:", i)
            state = self.select_and_expand()
            steps, distance = self.rollout(state)
            steps += self.cal_util(distance)
            self.backpropagate(steps)

            self.factor *= 0.992

class timer:
    def __init__(self,time_limit) -> None:
        self.start_time=time.perf_counter()
        self.time_limit=time_limit

    def start(self)->float:
        self.start_time=time.perf_counter()
        
    def check(self)->float:
        return time.perf_counter()-self.start_time
    
    def must_stop(self)->bool:
        return (time.perf_counter()-self.start_time)>self.time_limit
    
    def remaining(self)->float:
        return self.time_limit-(time.perf_counter()-self.start_time)
            

if __name__ == "__main__":
    current_directory = os.path.dirname(__file__)
    map_file_path = os.path.join(current_directory, "grid_maps",MAP_NAME,"occ_grid_small.txt")

    warnings.filterwarnings("ignore", category=UserWarning, module="gym")

    # mytimer = timer(0.5)
    # mytimer.start()
    # print(mytimer.check())

    # init_distance = 3 
    # init_speed = 2
    # start_pos = (20,20)

    res_dict = {}

    start_pos = (20, 20)

    for condition in range(4):
        for i in range(1,6):
            for j in range(1,5):
                for k in range(1,5):
                    for ini_distance in range(9): 
                        for ini_speed in range(4):
                            if ini_distance == 0 and ini_speed == 0:
                                continue
                            args = None
                            if i == 5:
                                args=(9,j,k)
                            else:
                                args=(i,j,k)
                            mcts = MCTS(map_file_path=map_file_path,
                                        iterations=1000,
                                        start_pos=start_pos, 
                                        init_distance=ini_distance,
                                        init_speed=ini_speed, 
                                        condition=condition, 
                                        args=args)    
                            
                            mcts.begin_search()

                            if args[0] == 9:
                                args = (5,args[1],args[2])

                            acc_de = mcts.MCTree[(ini_distance, ini_speed, -1)][0]
                            acc_ke = mcts.MCTree[(ini_distance, ini_speed, 0)][0]
                            acc_ac = mcts.MCTree[(ini_distance, ini_speed, 1)][0]

                            if acc_ac >= acc_de and acc_ac >= acc_ke:
                                res_dict[(ini_distance, ini_speed, condition, args)] = 1
                            if acc_ke >= acc_de and acc_ke >= acc_ac:
                                res_dict[(ini_distance, ini_speed, condition, args)] = 0
                            if acc_de >= acc_ke and acc_de >= acc_ac:
                                res_dict[(ini_distance, ini_speed, condition, args)] = -1
                            
                    
                    # for key, items in res_dict.items():
                    #     print(key, items)
                    
        print("This condition and args Done")            # assert 0
    print(len(res_dict))
    another_copy = {}
    with open("MCTS-v1.csv", "w") as output_file, open("MCTS-v1.json", "w") as out_json:
        output_file.write("distance,speed,conditon,args_0,args_1,args_2,Action\n")
        for key, items in res_dict.items():
            # print(key, items)
            output_file.write(str(key[0])+ ' ' \
                              +str(key[1])+ ' ' \
                              +str(key[2])+ ' ' \
                              +str(key[3][0])+ ' ' \
                              +str(key[3][1])+ ' ' \
                              +str(key[3][2])+ ' ' \
                              +str(items) + "\n")
            # another_copy.pop(key)
            another_copy[str(key[0])+ '_' \
                              +str(key[1])+ '_' \
                              +str(key[2])+ '_' \
                              +str(key[3][0])+ '_' \
                              +str(key[3][1])+ '_' \
                              +str(key[3][2])] = items
        
        json.dump(another_copy, out_json)

    # condition = 0 # from 0 to 3
    # args = (8, 0, 0) # from 1 to 4

    # mcts = MCTS(map_file_path=map_file_path,
    #             iterations=1000,
    #             start_pos=start_pos, 
    #             init_distance=init_distance,
    #             init_speed=init_speed, 
    #             condition=condition, 
    #             args=args)    
    
    # mcts.begin_search()
    

    # # mcts.env.close()

    # search_result = mcts.MCTree

    # print(len(search_result))

    # with open("mcts_res_con"+str(condition)+"_di"+str(init_distance)\
    #           +"_sp"+str(init_speed) +".csv", 'w') as output_file:
    #     output_file.write("distance,speed,acc,num,util,avg_util\n")

    #     for distance in range(mcts.init_distance+1):
    #         for speed in range(4):
    #             for acc in (-1,0,1):
    #                 if (distance, speed, acc) in search_result:
    #                     value = search_result[(distance, speed, acc)]
    #                     output_file.write(str(distance) + ', ' +str(speed)+ ', '+str(acc)+ ', '+\
    #                                     str(value[0])+ ', '+str(value[1])[:8] + ', '+str(value[1]/value[0])[:8]+'\n')

