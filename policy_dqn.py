from grid_map_env.classes.action import Action
from grid_map_env.utils import *

import random

import torch
from model import DQN

def get_state(state, Map):
    # check
    
    # row, col, direction, speed = state
    row = state.row
    col = state.col
    direction = state.direction
    speed = state.speed
    
    tmp=Map.copy()
    
    row = row.item()
    col = col.item()

    # tmp[start_pos[0], start_pos[1]] = -1000
    # tmp[start_pos[0] + 1, start_pos[1]] = -1000 
    # tmp[start_pos[0], start_pos[1] + 1] = -1000
    # tmp[start_pos[0] + 1, start_pos[1] + 1] = -1000
    # tmp[goal_pos[0], goal_pos[1]] = 1000
    # tmp[goal_pos[0] + 1, goal_pos[1]] = 1000
    # tmp[goal_pos[0], goal_pos[1] + 1] = 1000
    # tmp[goal_pos[0] + 1, goal_pos[1] + 1] = 1000
    tmp[row][col] = 500
    tmp[row + 1][col] = 500
    tmp[row, col + 1] = 500
    tmp[row + 1, col + 1] = 500

    return tmp
    pass

class PolicyDQN:
    def __init__(self, input_dim, action_space) -> None:
        self.model = DQN(input_dim=input_dim, action_space=action_space).to(dtype=torch.float64)
        self.model.load_state_dict(torch.load('model/double_dqn_value_tuned_tmp.pth'))
        self.first_action = True
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
        
        if house_map[robot_state.row][robot_state.col] != 0:
            print("the first action")
            # first_action = True
        
        state_map = get_state(robot_state, house_map)
    
        
        action_value = self.model(torch.tensor(state_map).unsqueeze(0).unsqueeze(0).double()).squeeze(0)
        
        if self.first_action:
            action_value[0] = -np.inf
            action_value[1] = -np.inf
            action_value[2] = -np.inf
            self.first_action = False
    
        print(action_value)
        
        ind = torch.argmax(action_value)
        
        chosen = False
        failed = 0

        print("robot state: ",robot_state.speed, robot_state.direction)

        while True:
            ind = torch.argmax(action_value)
            
            if failed == 6:
                print("failed to choose a valid op...")
                exit(1)

            if ind == 0:
                if robot_state.speed <= 0:
                    action_value[ind] = -np.inf
                    failed += 1
                else:
                    acc = -1
                    rot = 0
                    break
            elif ind == 1:
                acc = 0
                rot = 0
                break
            elif ind == 2:
                if robot_state.speed >= 3:
                    action_value[ind] = -np.inf
                    failed += 1
                else:
                    acc = 1
                    rot = 0
                    break
            elif ind == 3:
                if robot_state.speed != 0:
                    action_value[ind] = -np.inf
                    failed += 1
                else:
                    acc = 0
                    rot = -1
                    break
            elif ind == 4:
                acc = 0
                rot = 0
                break
            elif ind == 5:
                if robot_state.speed != 0:
                    action_value[ind] = -np.inf
                    failed += 1
                else:
                    acc = 0
                    rot = 1
                    break
            else:
                exit(1)
        
        print("action: ", acc, rot)
        action = Action(acc=acc, rot=rot)
        # next_state = self.transition(robot_state=robot_state, action=action)
        
        
        # if robot_state.speed < 2:
        #     acc = 1  # accelerate
        # else:
        #     acc = -1 # decelerate

        # action = Action(acc=acc, rot=0)  # construct an instance of the Action class
        
        next_state = self.transition(robot_state=robot_state, action=action) # predict the transition

        # collision checking and response
        # if is_collision(house_map=house_map, robot_state=next_state):
        #     #change the action due to collision in the predicted enxt state
        #     if robot_state.speed > 0: # decelerate to stop
        #         action = Action(acc=-1, rot=0)
        #     else: # choose random action
        #         random_action = random.choice([(0, 1), (0, -1)])
        #         action = Action(acc=random_action[0], rot=random_action[1])

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

