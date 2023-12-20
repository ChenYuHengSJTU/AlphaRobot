# 使用A*产生的数据集预训练DQN网络
# 使用epsilon-greedy策略进行再训练
# 使用REINFORCE策略
# 需要play一局完整的游戏

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
from tqdm import tqdm, trange

from dqn_evaluator import get_reward

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
    START_MARK = 1
    END_MARK = 2
    ROBOT_MARK = 5
    
    start_pos, goal_pos, (row, col, direction, speed) = state
    tmp=Map.copy()
    
    row = row.item()
    col = col.item()

    # -1000
    tmp[start_pos[0], start_pos[1]] = START_MARK
    tmp[start_pos[0] + 1, start_pos[1]] = START_MARK
    tmp[start_pos[0], start_pos[1] + 1] = START_MARK
    tmp[start_pos[0] + 1, start_pos[1] + 1] = START_MARK
    # 1000
    tmp[goal_pos[0], goal_pos[1]] = END_MARK
    tmp[goal_pos[0] + 1, goal_pos[1]] = END_MARK
    tmp[goal_pos[0], goal_pos[1] + 1] = END_MARK
    tmp[goal_pos[0] + 1, goal_pos[1] + 1] = END_MARK
    # 500
    tmp[row][col] = ROBOT_MARK
    tmp[row + 1][col] = ROBOT_MARK
    tmp[row, col + 1] = ROBOT_MARK
    tmp[row + 1, col + 1] = ROBOT_MARK

    return tmp
    pass


# need to get value
# buffersize -> json file number
def build_replay_buffer(buffer_size=100):
    json_file_path = 'replay_data/samples/'
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
    
        reward = get_reward(start_pos, goal_pos, steps, float(is_goal))
        # # print(is_goal)
        # print(reward)
        state_nxt = None
        state_nxt_map = None
        for j in range(1, steps + 1):
            action = (action_accs[-j], action_rots[-j])
            # print(action)
            state = start_pos, goal_pos, (robot_rows[-j], robot_cols[-j], robot_directions[-j], robot_speeds[-j])
            # state_map = get_state(state, Map)
            # print(state, state_nxt)
            # cost = 1
            if state_nxt:
                # print(j)
                # assert state_nxt.all() != state.all()
                # print(state_nxt, state, sep='\t')
                replay_buffer.append((state, state_nxt, action, reward))
            state_nxt = state
            
            
    return replay_buffer, map_file_path
    pass





if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device=device)
    print(device)
    # torch.set_default_device(torch.device("cpu"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=int, default = 1)
    
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--size", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--ep", type=float, default=0.2)
    
    args = parser.parse_args()



    # 因为需要加入noise和epsilon-greedy策略，所以pretrain==0
    assert args.pretrain == 0




    warnings.filterwarnings("ignore", category=UserWarning)
    
    input_dim = 100
    value_net = DQN(input_dim=input_dim, action_space=6).to(device=device)
    target_net = DQN(input_dim=input_dim, action_space=6).to(device=device)


    replay_buffer, map_file_path= build_replay_buffer(args.size)
    
    
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

    epochs = args.epochs

    bs = 128
    
    learning_rate = 5e-5

    # 使用epsilon-greedy策略进行再训练
    epsilon = 0.2
    
    dataLoader = DataLoader(ReplayBuffer(replay_buffer), batch_size=args.bs, shuffle=True, 
                            generator=torch.Generator(device='cuda')
                            )
    

    optimizer = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    

    model_path = "model/dqn_value_tuned_tmp.pth"
    if args.load == 1:
        if os.path.exists(model_path):
            value_net.load_state_dict(torch.load(model_path))
            target_net.load_state_dict(torch.load(model_path))
            print("load model from model/dqn_value_tuned_tmp.pth")
            value_net.to(device=device)
            target_net.to(device=device)
        else:
            print("no model found!")
            exit(1)
    
    # print(model)
    
    
    loss_min = np.inf
    
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

        for state, state_nxt, action, value in tqdm(dataLoader):
            
            
            # print(value)
            
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
            
            
            state_map = torch.from_numpy(state_map).double().view(length, -1, input_dim, input_dim).to(device=device)
            state_nxt_map = torch.from_numpy(state_nxt_map).double().view(length, -1, input_dim, input_dim).to(device=device)
            
            
            # action编码 -> 
            # 0: acc-1
            # 1: acc
            # 2: acc+1
            # 3: rot-1
            # 4: rot
            # 5: rot+1
            
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
            # print(state_map.dtype)
            action_value = value_net(state_map, action_vec)
            # 取最优动作下标
            # print(action_value.requires_grad)
            actions = action_value.argmax(dim=-1)
            # td_target为TD target
            
            # TODO：添加epislon-greedy策略
            if args.pretrain == 0:
                pass            

            
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
            
            
            # TODO：这里的value可以尝试对没有成功的value进行惩罚
            
            # print(value.dtype)
            # print(td_target.dtype)
            # print(value_net(state_nxt_map, action_vec).dtype)
            
            # value = value.to(dtype=torch.float32)
            td_target[:, actions] = value + value_net(state_nxt_map, action_vec)[:, actions]
            # td_target[:, actions] = 1 + target_net(state_nxt_map)[0][actions]
            # td_target.requires_grad = True
            # td_target_v = torch.tensor(td_target)


            # print(action_value, td_target)

            loss = mse_loss(action_value, td_target)
            
            # print(loss.item())
            
            loss_all += loss.item()
            
            loss.retain_grad()
            loss.backward()       

            # action_value.retain_grad()
            # td_target.retain_grad()
            optimizer.step()
        
        # if e % 10 == 0 and e != 0:
        # print(f"epoch: {e}, loss: {loss_all}")
        if loss_all < loss_min:
            torch.save(value_net.state_dict(), model_path) 
            print("save model!")
        
        loss_min = np.min([loss_min, loss_all])
        print(f"epoch: {e}, loss: {loss_all}, loss avg: {loss_all / len(replay_buffer)}, current min loss: {loss_min}")
        

        
        loss_prev = loss_all

        # if e % 5 == 0:
            # 还可以采用其他的更新条件和更新方式
            # target_net.load_state_dict(value_net.state_dict())