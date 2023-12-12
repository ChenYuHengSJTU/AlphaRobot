import numpy as np
import json
np.set_printoptions(threshold=np.inf)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_state(col, row, Map):
    # check
    # print(len(state))
    # print(state)
    
    # start_pos, goal_pos, (row, col, direction, speed) = state
    map = Map

    # start_pos = start_pos
    # goal_pos = goal_pos.numpy()
    # row = row.item()
    # col = col.item()

    print(col, row, sep=" ")
    # print(map)
    
    # print(type(map))
    # print(type(start_pos))
    # print(type(Map)）
    # print(type(state))
    
    # map[start_pos[0], start_pos[1]] = -1000
    # map[start_pos[0] + 1, start_pos[1]] = -1000 
    # map[start_pos[0], start_pos[1] + 1] = -1000
    # map[start_pos[0] + 1, start_pos[1] + 1] = -1000
    # map[goal_pos[0], goal_pos[1]] = 1000
    # map[goal_pos[0] + 1, goal_pos[1]] = 1000
    # map[goal_pos[0], goal_pos[1] + 1] = 1000
    # map[goal_pos[0] + 1, goal_pos[1] + 1] = 1000
    map[row][col] = 500
    map[row + 1][col] = 500
    map[row][col + 1] = 500
    map[row + 1][col + 1] = 500
    # print(len(map.shape))
    # print(map)
    return map

Map = json.load(open("map.json", "r+"))['Map']
tmp = Map
map1 = get_state(60, 21, Map)
map2 = get_state(61, 21, Map)
print(get_state(60, 21, Map) == get_state(61, 21, Map))

with open("map1.txt", "w+") as f:
    f.write(str(get_state(60, 21, Map)))

with open("map2.txt", "w+") as f:
    f.write(str(get_state(61, 21, Map)))

print(tmp == Map)
print(map1 == Map)

print(map1[60])
print(map2[60])
