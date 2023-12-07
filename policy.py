from grid_map_env.classes.action import Action
from grid_map_env.utils import *
import math,heapq
import numpy
import time



import random

####### pre-defined const

#### a const to decide whether to print detailed debug information
DEBUG=True



#### seting the time restriction in computation, None for unlimited
## A*'s is under 2 limitation(A_TIME_LIMIT is preferred, although it limit actual time, not cpu time)
A_POP_LIMIT=None    #A* pop action limit
A_TIME_LIMIT=1      #A* time action limit



#### const used
COST_LINE=(0,2,3,4,4,5,5,6,6,6)  
DIR_OFFSET=((0,-1),(-1,0),(0,1),(1,0))      #定义不同方向的行为，每次运动会导致 (row,col)+=DIR_OFFSET[direction]
                                            #**请统一使用这一标准，并使用这个常量变换offset和direction**

####################################################################################################
# help function
def direction_to_offset(direction:int)->(int,int):
    """
    用于将0~3的方向转换为row,col的变化方向
    Argv:
        direction(int): 方向
    Return:
        (int,int): (row_offset,col_offset)定义row和col的变化方向
            在direction方向前进会导致(row,col)+=(row_offset,col_offset)
    """
    return DIR_OFFSET[direction]

def offset_to_direction(self,offsets):
    """
    用于将row,col的变化方向转换为0~3的方向
    Argv:
        offsets(int,int): (row_offset,col_offset)定义row和col的变化方向
    Return:
        (int): 方向
    """
    offsets_normalized=self.pos_or_neg(offsets)
    return DIR_OFFSET.index(offsets_normalized)
    
def acc_dacc(v:int,l:int)->int:
    """
    在直线上计算加减速反应, 基于剩余距离l, 与速度v
    **注意, 非常激进的映射, 在减速时一旦减速失败便会冲过头**
    Argv:
        v(int): 速度
        l(int): 剩余距离
    Return:
        (int): 1:加速, 0:保持, -1:减速
    """
    #一个阶梯函数的实现
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
    

def line_cost(l:int)->int:
    """
    用于计算运动长度为l的直线所花的步数(从速度为0开始到速度为0结束)
    Argv;
        l(int): 距离
    Return:
        (int):
    """
    if l<10:
        return COST_LINE[l]
    return math.ceil((l-9)/3)+6

def Space_check(house_map,position)->bool:
    """
    一个is_collision()的包装函数，同时检查数值范围是否合法
    Argv:
        house_map(int[int[int]]):
        position(int,int):
    Return:
        (bool): True: 合法可通行位置
                False: 不合法/不可通行
    """
    if -1<position[0]<100 and -1<position[0]<100:
        hitbox=RobotState(position[0],position[1])
        return not is_collision(house_map,hitbox)
    else:
        return False

def pos_or_neg(value)->int:
    if value>0:
        return 1
    if value<0:
        return -1
    return value

def distance_check_RS(house_map,robot_state:RobotState)->int:
    """
    检查方向上还有几格空间
    """
    row_offset,col_offset=DIR_OFFSET[robot_state.direction]
    for i in range(1,100):
        new_pos=(robot_state.row+row_offset*i,robot_state.col+col_offset*i)
        if not Space_check(house_map,new_pos):
            return i-1
        
def distance_check_Cd(house_map,condition)->int:
    """
    检查方向上还有几格空间
    """
    row_offset,col_offset=DIR_OFFSET[condition[2]]
    for i in range(1,100):
        new_pos=(condition[0]+row_offset*i,condition[1]+col_offset*i)
        if not Space_check(house_map,new_pos):
            return i-1
        
def distance_check_Pos(house_map,position,direction:int)->int:
    """
    检查方向上还有几格空间
    """
    row_offset,col_offset=DIR_OFFSET[direction]
    for i in range(1,100):
        new_pos=(position[0]+row_offset*i,position[1]+col_offset*i)
        if not Space_check(house_map,new_pos):
            return i-1

def compare_state_RS(rs_1:RobotState,rs_2:RobotState)->bool:
    """
    比较rs_1与rs_2的值是否相等
    """
    return rs_1.row==rs_2.row and rs_1.col==rs_2.col and rs_1.direction==rs_2.direction and rs_1.speed==rs_2.speed

    
        

####### pre-defined class

#### the timer class for limiting time for calculation
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
    
    


###############
#   algorithm
class A_star:
    def __init__(self,target=None) -> None:
        self.target=target        #用于存储goal，避免多次计算

        self.policy=dict()      #记录各个状态对应的policy
                                #key为(row,col,direction,speed)
                                #value编码如下
        #self.policy    -1: 减速
        #               1: 加速
        #               0: 保持
        #               10+x: x是希望的方向(应该只能在speed==0时返回)
        #               -404: 默认，代表未定义该状态
        # **请使用rlt=self.policy.get(condition,-404)，将默认返回设置为-404**
        self.cost=dict()

        self.rout=None        #A*，用于存储self.A_Star_path_calculate()计算得到的路径的转弯节点
                                #编码为 (row,cow,direction)
                                #最后一个编码为 (goal_row,goal_col)
        self.full_path=False  #A*，是否找到了最优到goal的路径（是否被limit打断导致A*未完全跑完）
        self.lib=[]           #A*，用于储存fringe的最小堆，编码方式见函数内说明
        self.start_condition=None     #A*，记录本轮A*初始化的出发
        self.timer=timer(A_TIME_LIMIT)    #A*，用于检测限定时间是否用尽
        self.best_cost=-1             #A*，记录找到最优路径在无干扰情况下最少的步数
    
    def Heuristic(self,position,direction:int=None)->int:
        """
        Heuristic used for A*
        a admissive,consistent Heuristic, 请确保self.target已被计算得到
        Argv:
            position((int,int)): 位置
            direction: 方向
        Return:
            (int): 在无阻碍情况下的步数, 忽略中间转向(为了更快)
        """
        c=0
        if direction!=None:
            offsets=DIR_OFFSET[direction]
            offsets_to_goal=pos_or_neg(position[0]-self.target[0]),pos_or_neg(position[1]-self.target[1])
            if offsets[0]!=offsets_to_goal[0] and offsets[1]!=offsets_to_goal[1]:
                #如果当前朝向远离goal，增加一个punishment
                c=1

        a,b=abs(position[0]-self.target[0]),abs(position[1]-self.target[1])
        return line_cost(a)+line_cost(b)+c
    
    def A_Star_path_init(self,robot_state:RobotState)->None:
        """initiate A*, 并且设置self.start_condition=robot_state"""
        if self.target==None:
            raise "\t[A*]: \ttarget uninitialized!"
        if DEBUG:
            print(f"\t[A*]: \t\trunning A* from ({robot_state.row},{robot_state.col}:{robot_state.direction}) to {self.target}")
        self.lib=[]       #最小堆
        self.anc=dict()     #用于回溯的ancestor字典，编码如下

        #self.anc:
        #   key:(row,col,direction)
        #   value:(row,col,direction)
        
        pos=(robot_state.row,robot_state.col)
        condition=(robot_state.row,robot_state.col,robot_state.direction)
        self.start_condition=condition

        #self.lib的编码如下
        #   0    ,   1 ,     2 ,     3 ,        4    ,   5  ,      6
        # h()+f(),  h(),    row,    col,    direction,  cost,   from(ancestor)

        #将所有4方向压入
        d_0=robot_state.direction
        d_1=(d_0+1)%4
        d_2=(d_0-1)%4
        d_3=(d_0+2)%4

        h_0=self.Heuristic(pos,d_0)
        h_1=self.Heuristic(pos,d_1)
        h_2=self.Heuristic(pos,d_2)
        h_3=self.Heuristic(pos,d_3)
        heapq.heappush(self.lib,(0+h_0,h_0,robot_state.row,robot_state.col,d_0,0,condition))
        heapq.heappush(self.lib,(1+h_1,h_1,robot_state.row,robot_state.col,d_1,1,condition))
        heapq.heappush(self.lib,(1+h_2,h_2,robot_state.row,robot_state.col,d_2,1,condition))
        heapq.heappush(self.lib,(2+h_3,h_3,robot_state.row,robot_state.col,d_3,2,condition))
     

    def A_Star_path_calculate(self,house_map)->None:
        """
        a caluculation function for A*, can halt with time limit
        """
        #start timer
        self.timer.start()

        rout=[]
        process_count=0
        while True:
            crr=heapq.heappop(self.lib)
            crr_condition=(crr[2],crr[3],crr[4])
            process_count+=1

            acrt=self.anc.get(crr_condition,-1)

            if acrt!=-1:
                continue

            self.cost[crr_condition]=crr[5]

            #确认是否超过pop上限
            if A_POP_LIMIT and process_count>A_POP_LIMIT:
                self.full_path=False
                rout.append(self.target)
                rout.append(crr[6])
                if DEBUG:
                    print("\t[A*]: \tpop limit reached!")
                break

            #确认是否超过time上限
            if A_TIME_LIMIT and self.timer.check()>A_TIME_LIMIT:
                self.full_path=False
                rout.append(self.target)
                rout.append(crr[6])
                if DEBUG:
                    print("\t[A*]: \ttime limit reached!")
                break

            
            self.anc[crr_condition]=crr[6]
            
            #如果找到目标
            if (crr[2],crr[3])==self.target:
                self.full_path=True
                rout.append(self.target)
                rout.append(crr[6])
                self.best_cost=crr[5]
                if DEBUG:
                    print(f"\t[A*]: \t\ttarget found! best step:{self.best_cost}")
                break

            row_offset,col_offset=DIR_OFFSET[crr[4]]
            for i in range(1,100):
                new_row,new_col=new_pos=crr[2]+row_offset*i,crr[3]+col_offset*i
                if not Space_check(house_map,(new_row,new_col)):
                    break
                
                new_condition=(new_row,new_col,crr[4])
                if self.anc.get(new_condition,-1)!=-1:
                    break

                new_cost=crr[5]+line_cost(i)+1
                new_h_0=self.Heuristic(new_pos,(crr[4]+1)%4)
                new_h_1=self.Heuristic(new_pos,(crr[4]-1)%4)
                heapq.heappush(self.lib,(new_cost+new_h_0,new_h_0,new_row,new_col,(crr[4]+1)%4,new_cost,crr_condition))
                heapq.heappush(self.lib,(new_cost+new_h_1,new_h_1,new_row,new_col,(crr[4]-1)%4,new_cost,crr_condition))
        
        #backtracing
        new_node=rout[-1]
        while new_node!=self.start_condition:
            new_node=self.anc[new_node]
            rout.append(new_node)
        rout.reverse()
        self.rout=rout
        if DEBUG:
            print(f"\t[A*]: \tprocessed {process_count} item")
            print(f"\t[A*]: \tprocessed {self.timer.check()} time")

    def Policy_generation(self)->None:
        """
        使用self.rout生成self.policy
        """
        #重置self.policy
        self.policy=dict()
        
        if self.rout==None:
            print("A_rout is NULL exiting")
            raise "A_rout is NULL"

        for i,node in enumerate(self.rout):
            if i==len(self.rout)-1:
                break
            #解包self.rout
            next_node=self.rout[i+1]
            desired_direction=node[2]
            row_offset,col_offset=DIR_OFFSET[desired_direction]
            

            for dir in range(4):
                if dir==desired_direction:
                    continue
                condition=node[0],node[1],0,dir
                #对于出发点，对全部方向，速度为0，设置为需要的方向
                self.policy[condition]=desired_direction+10

            totallen=row_offset*(next_node[0]-node[0])+col_offset*(next_node[1]-node[1])

            for j in range(totallen+1):
                rest=totallen-j
                row=node[0]+j*row_offset
                col=node[1]+j*col_offset
                for v in range(4):
                    if j==totallen and v==0:
                        #跳过终点速度为0
                        continue
                    condition=row,col,v,desired_direction
                    #对路径上的所有点，朝向正确的方向，计算加减速操作
                    self.policy[condition]=acc_dacc(v,rest)
      
    


##############################################################################################################################

class Policy:
    def __init__(self) -> None:
        self.target=None
        self.A_Star=None
        #err rate check使用
        self.simul_expected=None    #期待的下一个状态
        self.simul_step_count=0     #获取Action总次数
        self.simul_err_count=0      #出现噪声的次数

    def __del__(self):
        print("[del]: policy istance deleted")
        if DEBUG and self.simul_step_count!=0:
            err_rate=self.simul_err_count/self.simul_step_count
            print(f"[err rate]: one round err rate:{err_rate}")
            with open("err rate.txt","a") as f:
                f.write(f"{err_rate}\n")

    def bounce(self,house_map,robot_state:RobotState)->Action:
        """
        测试程序, 会不断撞墙
        请在最终版本删除
        """
        offsets=row_offset,col_offset=DIR_OFFSET[robot_state.direction]
        speed_limit=3


        new_pos=(robot_state.row+row_offset*robot_state.speed,robot_state.col+col_offset*robot_state.speed)
        if not self.Space_check(house_map,new_pos):
            print("[collision]: brace for impact")
            return Action(0,0)


        if robot_state.speed==0:
            c=random.randrange(-1,2)
            if c==0:
                return Action(1,0)
            else:
                return Action(0,c)
        else:
            if robot_state.speed<speed_limit:
                return Action(1,0)
            else:
                return Action(0,0)
       

    def simulate_RS(self,house_map,robot_state:RobotState,action:Action)->RobotState:
        """
        用于模拟运行(无噪声), 由self.check_err_rate()调用
        """
        row_offset,col_offset=DIR_OFFSET[robot_state.direction]
        speed=robot_state.speed+action.acc
        if speed>3:
            speed=3
        if speed<0:
            speed=0
        new_pos=new_row,new_col=(robot_state.row+row_offset*speed,robot_state.col+col_offset*speed)

        new_dir=(robot_state.direction+action.rot)%4

        if Space_check(house_map,new_pos):
            return RobotState(new_row,new_col,new_dir,speed)
        else:
            dist=distance_check_RS(house_map,robot_state)
            steps=dist-speed+1
            new_row,new_col=(robot_state.row+row_offset*steps,robot_state.col+col_offset*steps)
            print("\t[simulate]: bounce")
            return RobotState(new_row,new_col,new_dir,0)
        

    def check_err_rate(self,house_map,robot_state:RobotState,ag)->Action:
        """
        封装函数, 用于确定err rate
        """
        if self.simul_expected and not compare_state_RS(self.simul_expected,robot_state):
            if DEBUG:
                print("\t[err rate]: err occur")
            self.simul_err_count+=1
            if DEBUG and self.simul_expected!=None:
                print(f"\t[err rate]: expect({self.simul_expected.row},{self.simul_expected.col}:{self.simul_expected.direction}:{self.simul_expected.speed}); got ({robot_state.row},{robot_state.col}:{robot_state.direction}:{robot_state.speed})")
        rlt=ag(house_map,robot_state)
        self.simul_step_count+=1
        self.simul_expected=self.simulate_RS(house_map,robot_state,rlt)
        return rlt
  
    def Calibrate_target(self,house_map)->None:
        """
        find the goal and save in self.target
        """
        if self.target!=None:
            return
        for i in range(100):
            for j in range(100):
                if is_goal(house_map,RobotState(i,j)):
                    self.target=(i,j)
                    return
    

    def get_Action_from_policy(self,robot_state:RobotState,condition:(int,int,int,int),policy_source:dict)->Action:
        """
        从policy_source中获得Action, 如果未定义, 返回None

        Argv:
            robot_state(RobotState):

            condition(int,int,int,int): (row,col,direction,speed)
        Return:
            Action
        """
        rlt=policy_source.get(condition,-404)
        if rlt==-404:
            return None
        if rlt>=10:
            #turn
            rlt-=10+robot_state.direction
            if rlt>=2:
                rlt=-1
            if rlt<=-2:
                rlt=1
            return Action(0,rlt)
        else:
            #speed
            return Action(rlt,0)

    
    def A_Star_entry(self,house_map,robot_state:RobotState)->Action:
        """
        general entry of A*
        """
        if self.A_Star==None:
            self.Calibrate_target(house_map)
            self.A_Star=A_star(self.target)
            self.A_Star.A_Star_path_init(robot_state)
            self.A_Star.A_Star_path_calculate(house_map)
            self.A_Star.Policy_generation()

        #查询当前状态的policy
        condition=robot_state.row,robot_state.col,robot_state.speed,robot_state.direction
        act=self.get_Action_from_policy(robot_state,condition,self.A_Star.policy)

        #如果没有找到
        if act==None:
            if DEBUG:
                print("\t[debug]: \t\tunexpected condition!")
            #初始化A*并计算，生成policy，返回减速指令
            self.A_Star=A_star(self.target)
            self.A_Star.A_Star_path_init(robot_state)
            self.A_Star.A_Star_path_calculate(house_map)
            self.A_Star.Policy_generation()
            act=self.get_Action_from_policy(robot_state,condition,self.A_Star.policy)
            
        #正常情况，直接返回act（不是None）
        return act
        


        

    def Expect_max(self,house_map,robot_state):
        """deprecated"""
        pass

    def MCT_search(self,house_map,robot_state):
        pass

    def policy_iter(self,house_map,robot_state):
        pass
        

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
        #plz wrap different policy like below
        return self.check_err_rate(house_map,robot_state,self.A_Star_entry)


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

