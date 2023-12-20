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
import json
import warnings
import argparse

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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
        self.feature_size = self.features(
            torch.zeros(1, input_dim , input_dim)).cuda().view(1, -1).size(1)
        
        # print(self.feature_size)
        
        self.value = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, 512),
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
        x = x.view(-1, self.feature_size)
        # print(x.shape)
        # print(a.shape)
        
        # x = torch.concat((x, a), dim=-1)
                
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
    
    start_pos, goal_pos, (row, col, direction, speed) = state
    tmp=Map.copy()
    
    row = row.item()
    col = col.item()

    tmp[start_pos[0], start_pos[1]] = -1000
    tmp[start_pos[0] + 1, start_pos[1]] = -1000 
    tmp[start_pos[0], start_pos[1] + 1] = -1000
    tmp[start_pos[0] + 1, start_pos[1] + 1] = -1000
    tmp[goal_pos[0], goal_pos[1]] = 1000
    tmp[goal_pos[0] + 1, goal_pos[1]] = 1000
    tmp[goal_pos[0], goal_pos[1] + 1] = 1000
    tmp[goal_pos[0] + 1, goal_pos[1] + 1] = 1000
    tmp[row][col] = 500
    tmp[row + 1][col] = 500
    tmp[row, col + 1] = 500
    tmp[row + 1, col + 1] = 500

    return tmp
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
        state_nxt_map = None
        for j in range(1, steps + 1):
            action = (action_accs[-j], action_rots[-j])
            # print(action)
            state = start_pos, goal_pos, (robot_rows[-j], robot_cols[-j], robot_directions[-j], robot_speeds[-j])
            # state_map = get_state(state, Map)
            # print(state, state_nxt)
            cost = 1
            if state_nxt:
                # print(j)
                # assert state_nxt.all() != state.all()
                # print(state_nxt, state, sep='\t')
                replay_buffer.append((state, state_nxt, action, cost))
            state_nxt = state
            
            
    return replay_buffer, map_file_path
    pass





if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device=device)
    print(device)
    # torch.set_default_device(torch.device("cpu"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=int, default = 1)
    
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    
    input_dim = 100
    value_net = DQN(input_dim=input_dim, action_space=6).to(device=device)
    target_net = DQN(input_dim=input_dim, action_space=6).to(device=device)


    replay_buffer, map_file_path= build_replay_buffer()
    
    
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
    

    np.set_printoptions(threshold=np.inf)
    
    with open("map.json", "w+") as f:
        data = {
            'Map': Map
        }
        json.dump(data, f, cls=NumpyArrayEncoder)

    
    print(len(replay_buffer))

    epochs = 100000

    bs = 128
    
    learning_rate = 5e-5
    
    dataLoader = DataLoader(ReplayBuffer(replay_buffer), batch_size=bs, shuffle=False, 
                            generator=torch.Generator(device='cuda')
                            )
    

    optimizer = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss(reduction='sum')
    

    model_path = "model/dqn_tmp_new.pth"
    if args.load == 1:
        if os.path.exists(model_path):
            value_net.load_state_dict(torch.load(model_path))
            target_net.load_state_dict(torch.load(model_path))
            print("load model from model/dqn_tmp_new.pth")
            value_net.to(device=device)
            target_net.to(device=device)
    
    # print(model)
    
    
    
    for e in range(epochs):
        
        # TODO
        # 1. bs
        # 2. cuda
        # 3. double DQN
        # 4. numpy, tensor,   eq, all, any, ==
        # 5. loss backward
        # 6. leaf node
        # 7. 计算图 tensor ops
        # 8. deep / shallow copy 
        # 9. python 数据模型，引用等，何时copy，是否和data size有关
        # 10. 添加可视化
    
        loss_all = 0
        loss_prev = 0

        for state, state_nxt, action, cost in dataLoader:
            
            # print()
            # print(state)
            # print(state_nxt)
            # print(action)
            # print(cost)
            # print()
            
            # bs != 1, state,state_nxt,action,cost的每个tensor是堆叠起来的长度不为1的tensor，并不是数组
            # exit(0)
            
            optimizer.zero_grad()
            
            # bs!=1, 需要拆开处理
            # state_map = get_state(state, Map)
            # state_nxt_map = get_state(state_nxt, Map)
            
            
            state_map = None
            state_nxt_map = None

            # print(len(state))
            # print(state[0][0].shape)
            
            # exit(0)
            
            length = state[0][0].shape[0]
            
            # 到dataset的末尾时，可能不够一个batch
            for i in range(length):
                tmp = get_state(((state[0][0][i], state[0][1][i]), (state[1][0][i], state[1][1][i]), (state[2][0][i], state[2][1][i], state[2][2][i], state[2][3][i])), Map)
                tmp_nxt = get_state(((state_nxt[0][0][i], state_nxt[0][1][i]), (state_nxt[1][0][i], state_nxt[1][1][i]), (state_nxt[2][0][i], state_nxt[2][1][i], state_nxt[2][2][i], state_nxt[2][3][i])), Map)
                # tmp_nxt = get_state(state_nxt[0][i], state[1][i], (state[2][i], state[3][i], state[4][i], state[5][i]), Map)
                if state_map is None:
                    state_map = tmp
                    state_nxt_map = tmp_nxt
                    # state_map = torch.from_numpy(tmp).to(device=device)
                    # state_nxt_map = torch.from_numpy(tmp_nxt).to(device=device)
                else:
                    state_map = np.concatenate((state_map, tmp), axis=0)
                    state_nxt_map = np.concatenate((state_nxt_map, tmp_nxt), axis=0)
                    # state_map = torch.concat((state_map, tmp), dim=0)
                    # state_nxt_map = torch.concat((state_nxt_map, tmp_nxt), dim=0)
            
            
            state_map = torch.from_numpy(state_map).float().view(length, -1, input_dim, input_dim).to(device=device)
            state_nxt_map = torch.from_numpy(state_nxt_map).float().view(length, -1, input_dim, input_dim).to(device=device)
            
            
            action_vec = torch.zeros(length, 6).to(device=device)
            action_vec[:, action[0] + 1] = 1
            action_vec[:, action[1] + 4] = 1
            
            state_map.requires_grad_()
            state_nxt_map.requires_grad_()
            action_vec.requires_grad_()
            
            # 先不使用action，后续可以进行拼接或者使用卷积层
            # output of network
            # (bs * action_space)
            # td_target = cost + target_net(state_nxt_map)

            # value network根据当前状态-动作对的输出值
            action_value = value_net(state_map, action_vec)
            # 取最优动作下标
            # print(action_value.requires_grad)
            actions = action_value.argmax(dim=-1)
            # td_target为TD target
            
            
            # clone the tensor
            td_target = action_value.clone()

            # print(action_value.shape)
            # print(actions.shape)
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
            # assert state_nxt_map.all() != state_map.all()
            # print(action_value.requires_grad, td_target.requires_grad, sep='\t')
            # print(value_net(state_nxt_map, action_vec.unsqueeze(0)))
            td_target[:, actions] = 1 + value_net(state_nxt_map, action_vec)[:, actions]
            # td_target[:, actions] = 1 + target_net(state_nxt_map)[0][actions]
            # td_target.requires_grad = True
            # td_target_v = torch.tensor(td_target)

            loss = mse_loss(action_value, td_target)
            
            loss_all += loss.item()
            
            loss.retain_grad()
            loss.backward()       

            # action_value.retain_grad()
            # td_target.retain_grad()
            optimizer.step()
        
        # if e % 10 == 0 and e != 0:
        # print(f"epoch: {e}, loss: {loss_all}")
        
        print(f"epoch: {e}, loss: {loss_all}, loss avg: {loss_all / len(replay_buffer)}")
        
        if loss_all < loss_prev:
            torch.save(value_net.state_dict(), model_path) 
        
        loss_prev = loss_all

        # if e % 5 == 0:
            # 还可以采用其他的更新条件和更新方式
            # target_net.load_state_dict(value_net.state_dict())