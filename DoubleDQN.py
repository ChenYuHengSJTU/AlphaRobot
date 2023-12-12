import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import json
from grid_map_env.classes.robot_state import RobotState
from grid_map_env.classes.action import Action
from torch.utils.data import Dataset, DataLoader
from grid_map_env.utils import sample_start_and_goal
import os


class DQN(torch.nn.Module):
    def __init__(self, input_dim, action_space):
        super(DQN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        feature_size = self.features(
            torch.zeros(1, input_dim , input_dim)).cuda().view(1, -1).size(1) + action_space
        
        print(feature_size)
        
        self.value = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_space)
        )

        # self.actor = torch.nn.Sequential(
        #     torch.nn.Linear(feature_size, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, action_space),
        #     torch.nn.Softmax(dim=-1)
        # )
        
        # torch.nn.init.xavier_normal_(self.features.weight.data)
        # torch.nn.init.xavier_normal_(self.value.get_parameters())

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)


    def forward(self, x, a):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = x.view(1, -1)
        # print(x.shape)
        
        x = torch.concat((x, a), dim=-1)
                
        value = self.value(x)
        # actions = self.actor(x)
        return value

class ReplayBuffer(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length = len(buffer)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.buffer[index]
    

def get_action():
    pass

# start_pos, goal_pos, robot_state
def get_state(state, Map):
    # check
    # print(len(state))
    # print(state)
    
    start_pos, goal_pos, (row, col, direction, speed) = state
    map = Map
    # print(map)
    
    # print(type(map))
    # print(type(start_pos))
    # print(type(Map)）
    # print(type(state))
    
    map[start_pos[0], start_pos[1]] = map[start_pos[0] + 1, start_pos[1]] = map[start_pos[0], start_pos[1] + 1] = map[start_pos[0] + 1, start_pos[1] + 1] = -1000
    map[goal_pos[0], goal_pos[1]] = map[goal_pos[0] + 1, goal_pos[1]] = map[goal_pos[0], goal_pos[1] + 1] = map[goal_pos[0] + 1, goal_pos[1] + 1] = 1000
    map[row, col] = map[row + 1, col] = map[row, col + 1] = map[row + 1, col + 1] = 500
    # print(len(map.shape))
    # print(map)
    return map
    pass


def build_replay_buffer(buffer_size=100):
    json_file_path = 'replay/samples/'
    replay_buffer = list()
    for i in range(buffer_size):
        steps = 0
        with open(json_file_path + f"data_{i}.json") as f:
            data = json.load(f)
            map_file_path = data['map_file_path']
            start_pos = data['start_pos']
            goal_pos = data['start_pos']
            robot_rows = data['robot_rows']
            robot_cols = data['robot_cols']
            robot_directions = data['robot_directions']
            robot_speeds = data['robot_speeds']
            action_rots = data['action_rots']
            action_accs = data['action_accs']
            is_goal = data['is_goal']
            steps = data['steps']
            # Map = data['map']
        if not is_goal:
            continue
    
        state_nxt = None
        for j in range(1, steps + 1):
            action = (action_accs[-j], action_rots[-j])
            # print(action)
            state = start_pos, goal_pos, (robot_rows[-j], robot_cols[-j], robot_directions[-j], robot_speeds[-j])
            cost = 1
            if state_nxt:
                replay_buffer.append((state, state_nxt, action, cost))
            state_nxt = state
            
            
    return replay_buffer, map_file_path
    pass





if __name__ == "__main__":
    # torch.set_default_device(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # torch.set_default_device(torch.device("cpu"))
    
    input_dim = 100
    value_net = DQN(input_dim=input_dim, action_space=6)
    target_net = DQN(input_dim=input_dim, action_space=6)


    replay_buffer, map_file_path= build_replay_buffer()
    
    # exit(0)
    
    current_directory = os.path.dirname(__file__)
    MAP_NAME="Wiconisco" 
    map_file_path = os.path.join(current_directory, "grid_maps",MAP_NAME,"occ_grid_small.txt")

    start_pos, goal_pos = sample_start_and_goal(map_file_path)
    
    env = gym.make("grid_map_env/GridMapEnv-v0", n=100,
                   map_file_path=map_file_path, start_pos=start_pos, goal_pos=goal_pos, headless=True)

    # for name, param in self.rnn.named_parameters():
	# if name.startswith("weight"):
	# 	nn.init.xavier_normal_(param)
	# else:
	# 	nn.init.zeros_(param)


    # 恢复地图
    Map = env.reset()[0]['map']
    Map[start_pos[0]][start_pos[1]] = 0
    Map[start_pos[0] + 1][start_pos[1]] = 0
    Map[start_pos[0]][start_pos[1] + 1] = 0
    Map[start_pos[0] + 1][start_pos[1] + 1] = 0
    Map[goal_pos[0]][goal_pos[1]] = 0
    Map[goal_pos[0] + 1][goal_pos[1]] = 0
    Map[goal_pos[0]][goal_pos[1] + 1] = 0
    Map[goal_pos[0] + 1][goal_pos[1] + 1] = 0
    
    # Map[0][0] -= 2
    # Map[0][1] -= 2
    # Map[1][0] -= 2
    # Map[1][1] -= 2
    # print(Map)
    # print(len(Map))
    print(len(replay_buffer))

    epochs = 100000
    bs = 1
    
    learning_rate = 1e-4
    
    dataLoader = DataLoader(ReplayBuffer(replay_buffer), batch_size=bs, shuffle=True, 
                            # generator=torch.Generator(device='cuda')
                            )
    optimizer = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    
    
    for e in range(epochs):
        for state, state_nxt, action, cost in dataLoader:
            # state_ = RobotState(row=state[0][0], col=state[0][1], direction=state[0][2], speed=state[0][3])
            # print(action)
            # print(state)
            # print(len(state))
            
            optimizer.zero_grad()
            state_map = get_state(state, Map)
            state_nxt_map = get_state(state_nxt, Map)
            state_map = torch.from_numpy(state_map).float().view(-1, input_dim, input_dim)
            state_nxt_map = torch.from_numpy(state_nxt_map).float().view(-1, input_dim, input_dim)
            
            action_vec = torch.zeros(6)
            action_vec[action[0] + 1] = 1
            action_vec[action[1] + 4] = 1
            
            # 先不使用action，后续可以进行拼接或者使用卷积层
            # output of network
            # (bs * action_space)
            # td_target = cost + target_net(state_nxt_map)

            # value network根据当前状态-动作对的输出值
            action_value = value_net(state_map, action_vec.unsqueeze(0))
            # 取最优动作下标
            actions = action_value.argmax(dim=-1)
            # td_target为TD target
            td_target = action_value.data.detach().cpu().numpy().copy()
            # 根据TD算法更新TD target
            # 只更新最优动作对应的值
            
            # print(action_value.shape)
            # print(actions.shape)
            # print(td_target.shape)
            
            # print(actions)
            # print(action_value)
            # print(actions.item())
            # print(target_net(state_nxt_map)[0][actions[0]])
            # tmp = actions.item()
            
            # predict = target_net(state_nxt_map)
            
            # print(predict)
            
            # print(cost)
            
            # td_target[0][actions[0]] = predict[0][actions[0]]
            # with torch.no_grad():
            td_target[:, actions] = 1 + target_net(state_nxt_map, action_vec.unsqueeze(0)).detach().numpy()[0][actions]
            # td_target[:, actions] = 1 + target_net(state_nxt_map)[0][actions]
            # td_target.requires_grad = True
            td_target_v = torch.tensor(td_target)

            print(action_vec, actions[0], td_target, action_value, sep='\t')
            # print(action_value)
            
            
            loss = mse_loss(action_value, td_target_v)
            # print(loss)
            loss.backward()       
        
        if e % 10 == 0 and e != 0:
            print(f"epoch: {e}, loss: {loss.item()}")
            # torch.save(value_net.state_dict(), f"model/dqn_epoch_{e}.pth") 
        
        if e % 5 == 0:
            # 还可以采用其他的更新条件和更新方式
            target_net.load_state_dict(value_net.state_dict())