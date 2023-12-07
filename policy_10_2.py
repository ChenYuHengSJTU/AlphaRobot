from grid_map_env.classes.action import Action
from grid_map_env.utils import *
import math,heapq
import numpy
import time


import random

DEBUG=False

A_POP_LIMIT=None
A_TIME_LIMIT=1

class timer:
    def __init__(self) -> None:
        self.start_time=None
    def start(self):
        self.start_time=time.perf_counter()
    def check(self):
        return time.perf_counter()-self.start_time


class Policy:
    def __init__(self) -> None:
        self.Cost_list=[0,2,3,4,4,5,5,6,6,6]
        self.target=None
        self.A_rout=None
        self.dir_offset=((0,-1),(-1,0),(0,1),(1,0))
        self.node_index,self.node=None,None
        self.policy=dict()
        self.A_full_path=False
        self.lib=[]
        self.A_start_condition=None
        self.A_timer=timer()

    def acc_dacc(self,v,l):
        acc=0
        if l>=6:
            acc+=1
        if l>=3:
            acc+=1
        if l>=1:
            acc+=1
        acc-=v
        if acc<-1:
            acc=-1
        if acc>1:
            acc=1
        return acc
        
    
    def line_cost(self,l):
        if l<10:
            return self.Cost_list[l]
        return math.ceil((l-9)/3)+6
    
    def Heuristic(self,position):
        a,b=abs(position[0]-self.target[0]),abs(position[1]-self.target[1])
        return self.line_cost(a)+self.line_cost(b)
    
    def Space_check(self,house_map,position):
        if -1<position[0]<99 and -1<position[0]<99:
            hitbox=RobotState(position[0],position[1])
            return not is_collision(house_map,hitbox)
        else:
            return False
        
    def Policy_generation(self):
        self.policy=dict()
        
        if self.A_rout==None:
            print("A_rout is NULL exiting")
            raise "A_rout is NULL"

        for i,node in enumerate(self.A_rout):
            if i==len(self.A_rout)-1:
                break
            next_node=self.A_rout[i+1]
            desired_direction=node[2]
            offsets=row_offset,col_offset=self.dir_offset[desired_direction]
            

            for dir in range(4):
                if dir==desired_direction:
                    continue
                condition=node[0],node[1],0,dir
                self.policy[condition]=desired_direction+10

            totallen=row_offset*(next_node[0]-node[0])+col_offset*(next_node[1]-node[1])

            for j in range(totallen+1):
                rest=totallen-j
                row=node[0]+j*row_offset
                col=node[1]+j*col_offset
                for v in range(4):
                    if j==totallen and v==0:
                        continue
                    condition=row,col,v,desired_direction
                    self.policy[condition]=self.acc_dacc(v,rest)
        
    #problem unsolved here
    #not fully function
    # def check_on_node(self,robot_state):
    #     if self.A_rout[self.node_index]==(robot_state.row,robot_state.col):
    #         if self.node_index==len(self.A_rout)-1:
    #             return 200
            
    #         next_node=self.A_rout[self.node_index+1]
    #         node=self.A_rout[self.node_index]



    #         offsets=row_offset,col_offset=numpy.sign(next_node[0]-node[0]),numpy.sign(next_node[1]-node[1])
    #         desired_direction=self.dir_offset.index(offsets)
            
    #         turn=desired_direction-robot_state.direction

    #         if turn==0:
    #             self.node_index+=1

    #         if abs(turn)<=2:
    #             return abs(turn)
    #         else:
    #             if turn>0:
    #                 return -1
    #             else:
    #                 return 1
    #     return 0

    def Calibrate_target(self,house_map):
        if self.target!=None:
            return
        for i in range(100):
            for j in range(100):
                if is_goal(house_map,RobotState(i,j)):
                    self.target=(i,j)
                    return

        
        
    def A_Star(self,house_map,robot_state):
        self.A_timer.start()
        self.Calibrate_target(house_map)
        condition=robot_state.row,robot_state.col,robot_state.speed,robot_state.direction
        rlt=self.policy.get(condition,-404)

        if rlt==-404:
            if DEBUG:
                print("[debug]: \t\tunexpected condition!")
            self.A_Star_path_init(house_map,robot_state)
            self.A_Star_path_calculate(house_map)
            self.Policy_generation()
            return Action(-1,0)
        
        if not self.A_full_path:
            if DEBUG:
                print("[debug]: continue to cal!")
            self.A_Star_path_calculate(house_map)
            if self.A_full_path:
                self.Policy_generation()
            return Action(-1,0)

        act=Action(0,0)
        if rlt>=10:
            #turn
            rlt-=10+robot_state.direction
            if rlt>=2:
                rlt=-1
            if rlt<=-2:
                rlt=1
            act=Action(0,rlt)
        else:
            #speed
            act=Action(rlt,0)

        return act

    def A_Star_path_init(self,house_map,robot_state):
        self.Calibrate_target(house_map)
        if DEBUG:
            print(f"[debug]: \t\trunning A* from ({robot_state.row},{robot_state.col}:{robot_state.direction}) to {self.target}")
        self.lib=[]
        self.anc=dict()
        
        pos=(robot_state.row,robot_state.col)
        condition=(robot_state.row,robot_state.col,robot_state.direction)
        self.A_start_condition=condition

        # h()+f(),h(),row,col,direction,cost,from
        h=self.Heuristic(pos)
        heapq.heappush(self.lib,(0+h,h,robot_state.row,robot_state.col,robot_state.direction,0,condition))
        heapq.heappush(self.lib,(1+h,h,robot_state.row,robot_state.col,(robot_state.direction+1)%4,1,condition))
        heapq.heappush(self.lib,(1+h,h,robot_state.row,robot_state.col,(robot_state.direction-1)%4,1,condition))
        heapq.heappush(self.lib,(2+h,h,robot_state.row,robot_state.col,(robot_state.direction+2)%4,2,condition))

        

    def A_Star_path_calculate(self,house_map):
        rout=[]
        process_count=0
        while True:
            crr=heapq.heappop(self.lib)
            crr_condition=(crr[2],crr[3],crr[4])
            process_count+=1

            acrt=self.anc.get(crr_condition,-1)

            if acrt!=-1:
                continue

            if A_POP_LIMIT and process_count>A_POP_LIMIT:
                self.A_full_path=False
                rout.append(self.target)
                rout.append(crr[6])
                if DEBUG:
                    print("[debug]: \tpop limit reached!")
                break

            if A_TIME_LIMIT and self.A_timer.check()>A_TIME_LIMIT:
                self.A_full_path=False
                rout.append(self.target)
                rout.append(crr[6])
                if DEBUG:
                    print("[debug]: \ttime limit reached!")
                break

            
            self.anc[crr_condition]=crr[6]
            
            if (crr[2],crr[3])==self.target:
                self.A_full_path=True
                rout.append(self.target)
                rout.append(crr[6])
                if DEBUG:
                    print("[debug]: \t\ttarget found!")
                break

            row_offset,col_offset=self.dir_offset[crr[4]]
            for i in range(1,100):
                new_row,new_col=new_pos=crr[2]+row_offset*i,crr[3]+col_offset*i
                if not self.Space_check(house_map,(new_row,new_col)):
                    break
                
                new_condition=(new_row,new_col,crr[4])
                if self.anc.get(new_condition,-1)!=-1:
                    break

                new_cost=crr[5]+self.line_cost(i)+1
                new_h=self.Heuristic(new_pos)
                heapq.heappush(self.lib,(new_cost+new_h,new_h,new_row,new_col,(crr[4]+1)%4,new_cost,crr_condition))
                heapq.heappush(self.lib,(new_cost+new_h,new_h,new_row,new_col,(crr[4]-1)%4,new_cost,crr_condition))
        
        #backtracing
        new_node=rout[-1]
        while new_node!=self.A_start_condition:
            new_node=self.anc[new_node]
            rout.append(new_node)

        rout.reverse()
        self.A_rout=rout
        if DEBUG:
            print(f"[debug]: \tprocessed {process_count} item")

        

    def AD_Search(self,house_map,robot_state):
        pass

    def policy_iter(self,house_map,robot_state):
        if self.policy==None:
            self.Policy_generation(house_map)
        

    def RL_Learning(self,house_map,robot_state):
        pass


    def get_action(self, house_map, robot_state):
        ''' 
        Calculate a legal action.
        Here we demonstrate a very simple policy that 
        does not perform any from of search.
        Args: 
            house map (list of list): a 2D list of integers representing the house map. Please refer to Table 6 for its encoding.
            robot state (RobotState): an instance of the RobotState class representing the current state of the robot.
        Returns:
             action (Action): an instance of Action class representing the action for execution.
        '''
        return self.A_Star(house_map,robot_state)

        if robot_state.speed < 2:
            acc = 1  # accelerate
        else:
            acc = -1 # decelerate

        action = Action(acc=acc, rot=0)  # construct an instance of the Action class
        
        next_state = self.transition(robot_state=robot_state, action=action) # predict the transition

        # collision checking and response
        if is_collision(house_map=house_map, robot_state=next_state):
            #change the action due to collision in the predicted enxt state
            if robot_state.speed > 0: # decelerate to stop
                action = Action(acc=-1, rot=0)
            else: # choose random action
                random_action = random.choice([(0, 1), (0, -1)])
                action = Action(acc=random_action[0], rot=random_action[1])

        return action  # return the action for execution
    
    def transition(self,robot_state, action):
        '''
        a simple example for transition function
        Args:
            robot state (RobotState): an instance of the RobotState class representing the current state of the robot.
            action (Action): an instance of Action class representing the action for execution.
        Returns:
            next state (RobotState): an instance of the RobotState class representing the predicted state of the robot.
        '''

        next_state = robot_state.copy() #deep copy the robot state

        # update the robot's speed
        next_state.speed += action.acc 
        next_state.speed = max(min(next_state.speed, 3), 0)

        #update the robot's position
        if next_state.speed != 0:
            if next_state.direction == 0:
                next_state.col -= next_state.speed
            elif next_state.direction == 1:
                next_state.row -= next_state.speed
            elif next_state.direction == 2:
                next_state.col += next_state.speed
            elif next_state.direction == 3:
                next_state.row += next_state.speed

        #update the robot's direction
        else:
            next_state.direction = (next_state.direction+action.rot) % 4   
            
        return next_state 

