import numpy as np
import cv2
import random

EPISODES = 2000
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.5
MIN_EPSILON = 0.01
DECAY_RATE = 0.995
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] 

def load_pgm_map(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    grid_map = (image < 255 * 0.65).astype(np.int8) 
    return grid_map

class GridEnvironment:
    def __init__(self, grid_map, start, goal):
        self.grid = grid_map
        self.rows, self.cols = grid_map.shape
        self.start = start
        self.goal = goal

    def is_valid(self, state):
        r, c = state
        return (0 <= r < self.rows) and (0 <= c < self.cols) and (self.grid[r, c] == 0)

    def get_reward(self, state):
        if state == self.goal:
            return 100
        if self.grid[state] == 1:
            return -100
        return -1

def q_learning(env):
    q_table = np.zeros((env.rows, env.cols, 4))
    epsilon = EPSILON
    
    for _ in range(EPISODES):
        state = list(env.start)
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(range(4))
            else:
                action = np.argmax(q_table[state[0], state[1], :])
            
            new_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])
            
            if not env.is_valid(new_state):
                new_state = tuple(state)
            
            reward = env.get_reward(new_state)
            done = (new_state == env.goal) or (env.grid[new_state] == 1)
            
            current_q = q_table[state[0], state[1], action]
            max_next_q = np.max(q_table[new_state[0], new_state[1], :]) if not done else 0
            new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
            
            q_table[state[0], state[1], action] = new_q
            state = list(new_state)
        
        epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)
    
    return q_table

def extract_path(q_table, env):
    path = [env.start]
    current = list(env.start)
    steps = 0
    max_steps = 200  
    
    while current != list(env.goal) and steps < max_steps:
        action = np.argmax(q_table[current[0], current[1], :])
        new_r = current[0] + ACTIONS[action][0]
        new_c = current[1] + ACTIONS[action][1]
        
        if env.is_valid((new_r, new_c)):
            current = [new_r, new_c]
            path.append(tuple(current))
        steps += 1
    
    return path if current == list(env.goal) else []

def save_path(path, filename):
    with open(filename, "w") as file:
        for point in path:
            file.write(f"{point[0]}, {point[1]}\n")

if __name__ == "__main__":
    grid_map = load_pgm_map("map/map.pgm")
    start, goal = (0, 0), (4, 5)
    env = GridEnvironment(grid_map, start, goal)
    
    q_table = q_learning(env)
    path = extract_path(q_table, env)
    
    if path:
        save_path(path, "mission_path.txt")
        print("Path found and saved to mission_path.txt")
    else:
        print("Could not find a valid path")
