import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

GRID_SIZE = 10
START = (0, 0)
END = (9, 9)

FIXED_GRID = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
])

EPISODES = 500
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.5
MIN_EPSILON = 0.01
DECAY_RATE = 0.995
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)] 

class FixedGridEnvironment:
    def __init__(self):
        self.grid = FIXED_GRID.copy()
        self.start = START
        self.end = END

    def is_valid(self, state):
        r, c = state
        return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and self.grid[r,c] == 0

    def get_reward(self, state):
        if state == self.end: return 100
        if self.grid[state] == 1: return -100
        return -1

def visualize(env, path):
    plt.figure(figsize=(10, 10))
    
    for x in range(GRID_SIZE+1):
        plt.axvline(x, color='black', linewidth=1)
        plt.axhline(x, color='black', linewidth=1)
    
    obstacles = np.argwhere(env.grid == 1)
    for r, c in obstacles:
        plt.fill_between([c, c+1], r, r+1, color='black', alpha=1)
    
    if path:
        x = [p[1] + 0.5 for p in path]
        y = [p[0] + 0.5 for p in path]
        plt.plot(x, y, 'r-', linewidth=2, marker='o', markersize=8)
    
    plt.text(START[1]+0.3, START[0]+0.3, 'START', fontsize=12, color='blue')
    plt.text(END[1]+0.3, END[0]+0.3, 'GOAL', fontsize=12, color='green')
    
    plt.xlim(0, GRID_SIZE)
    plt.ylim(GRID_SIZE, 0)
    plt.xticks(
        ticks=[i + 0.5 for i in range(GRID_SIZE)],
        labels=[str(i) for i in range(GRID_SIZE)]
    )
    plt.yticks(
        ticks=[i + 0.5 for i in range(GRID_SIZE)],
        labels=[str(i) for i in range(GRID_SIZE)]
    )
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def extract_path(q_table, env):
    path = [START]
    current = list(START)
    visited = set()
    
    while current != list(END) and len(path) < 50:
        action = np.argmax(q_table[current[0], current[1], :])
        new_state = (current[0] + ACTIONS[action][0], 
                    current[1] + ACTIONS[action][1])
        
        if env.is_valid(new_state) and new_state not in visited:
            current = list(new_state)
            path.append(tuple(current))
            visited.add(new_state)
        else:
            break
    return path

def plot_training(ax, path, episode, rewards, steps, epsilons):
    ax[0].clear()
    ax[0].plot(rewards, 'b-')
    ax[0].set_title(f'Episode {episode} Reward')
    
    ax[1].clear()
    ax[1].plot(steps, 'g-')
    ax[1].set_title('Steps per Episode')
    
    ax[2].clear()
    ax[2].plot(epsilons, 'r-')
    ax[2].set_title('Epsilon Decay')
    

def q_learning():
    env = FixedGridEnvironment()
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    
    episode_rewards = []
    episode_steps = []
    epsilon_history = []
    
    plt.figure(figsize=(15, 10))
    ax = (plt.subplot2grid((2, 6), (0, 0), colspan=2),  # Reward
          plt.subplot2grid((2, 6), (0, 2), colspan=2), # Steps
          plt.subplot2grid((2, 6), (0, 4), colspan=2)) # Epsilon
    plt.ion()
    
    epsilon = EPSILON
    for episode in range(EPISODES):
        state = list(START)
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            if random.random() < epsilon:
                action = random.choice(range(4))
            else:
                action = np.argmax(q_table[state[0], state[1], :])
            
            new_state = (state[0] + ACTIONS[action][0], 
                        state[1] + ACTIONS[action][1])
            
            if not env.is_valid(new_state):
                new_state = tuple(state)
                
            reward = env.get_reward(new_state)
            done = (new_state == END) or (env.grid[new_state] == 1)
            
            # update Q-learning
            current_q = q_table[state[0], state[1], action]
            max_next_q = np.max(q_table[new_state[0], new_state[1], :]) if not done else 0
            q_table[state[0], state[1], action] = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)
            
            state = list(new_state)
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        epsilon_history.append(epsilon)
        epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)
        
        if episode % 100 == 0 or episode == EPISODES-1:
            current_path = extract_path(q_table, env)
            plot_training(ax, current_path, episode, 
                          episode_rewards, episode_steps, epsilon_history)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
    
    plt.ioff()
    plt.show()
    
    final_path = extract_path(q_table, env)
    if final_path[-1] == END:
        print(f"Training is complete! Final path length:{len(final_path)-1}")
    else:
        print("Could not find a valid path")
    visualize(env, final_path)

if __name__ == "__main__":
    q_learning()